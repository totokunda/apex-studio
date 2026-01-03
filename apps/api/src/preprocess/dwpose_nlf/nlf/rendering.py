import numpy as np
import pyrender
import trimesh
import cameralib


class MeshViewer:
    def __init__(self, imshape, camera, color=None):
        self.scene = pyrender.Scene(bg_color=[1, 1, 1, 0], ambient_light=(0.5, 0.5, 0.5))
        self.color = (160 / 255, 170 / 255, 205 / 255, 1.0) if color is None else color
        self.pyrender_camera = WrappedCamera(camera)
        self.camera_node = self.scene.add(self.pyrender_camera, name='pc-camera')
        self.scene.set_pose(self.camera_node, pose=self.get_extrinsics(camera))
        self.viewer = pyrender.OffscreenRenderer(imshape[1], imshape[0])
        self.add_raymond_light(intensity=1)
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.3, roughnessFactor=0.5, alphaMode='OPAQUE', baseColorFactor=self.color,
            doubleSided=True)

    def remove_meshes(self):
        for node in self.scene.get_nodes(name='mesh'):
            self.scene.remove_node(node)

    def add_mesh(self, mesh, color=None, material=None):
        if material is None and color is not None:
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.3, roughnessFactor=0.8, alphaMode='BLEND',
                baseColorFactor=color, doubleSided=True)

        if material is None and color is None:
            material = self.material

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        self.scene.add(mesh, 'mesh')

    def set_mesh(self, mesh, color=None, material=None):
        self.remove_meshes()
        self.add_mesh(mesh, color=color, material=material)

    def set_camera(self, camera):
        self.pyrender_camera.camera = camera
        self.scene.set_pose(self.camera_node, pose=self.get_extrinsics(camera))

    @staticmethod
    def get_extrinsics(camera):
        t = -camera.R @ np.expand_dims(camera.R @ camera.t * [-1, 1, 1], -1)
        return np.block(
            [[camera.R, t],
             [0, 0, 0, 1]]).astype(np.float32)

    def add_raymond_light(self, intensity):
        theta = np.pi * np.array([1 / 6, 1 / 6, 1 / 6], np.float32)
        phi = np.pi * np.array([0, 2 / 3, 4 / 3], np.float32)
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.stack([xp, yp, zp], axis=-1)
        z = z / np.linalg.norm(z, axis=-1)
        x = np.stack([-z[:, 1], z[:, 0], np.zeros_like(z[:, 0])], axis=-1)

        for x_, z_ in zip(x, z):
            if np.linalg.norm(x_) == 0:
                x_ = np.array([1.0, 0.0, 0.0])
            x_ = x_ / np.linalg.norm(x_)
            y_ = np.cross(z_, x_)
            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x_, y_, z_]
            self.scene.add_node(pyrender.node.Node(
                light=pyrender.light.DirectionalLight(color=np.ones(3), intensity=intensity),
                matrix=matrix))

    def render(self, render_wireframe=False, RGBA=False):
        flags = pyrender.constants.RenderFlags.SHADOWS_DIRECTIONAL
        if RGBA:
            flags |= pyrender.constants.RenderFlags.RGBA
        if render_wireframe:
            flags |= pyrender.constants.RenderFlags.ALL_WIREFRAME
        color_img, depth_img = self.viewer.render(self.scene, flags=flags)
        return color_img, depth_img

    def render_seg(self):
        return np.uint8(
            self.viewer.render(self.scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY) > 0)


class WrappedCamera(pyrender.camera.Camera):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera

    def get_projection_matrix(self, width, height):
        P = np.zeros((4, 4), np.float32)
        P[0, 0] = 2 * self.camera.intrinsic_matrix[0, 0] / width
        P[1, 1] = 2 * self.camera.intrinsic_matrix[1, 1] / height

        P[0, 2] = 1 - 2 * (self.camera.intrinsic_matrix[0, 2] + 0.5) / width
        P[1, 2] = 2 * (self.camera.intrinsic_matrix[1, 2] + 0.5) / height - 1
        P[3, 2] = -1
        n = 0.05
        f = 100
        P[2, 2] = (f + n) / (n - f)
        P[2, 3] = (2 * f * n) / (n - f)
        return P


def render(vertices, faces, camera, wireframe=False, colors=None, imshape=(800, 800)):
    viewer = MeshViewer(imshape=imshape, camera=camera)
    n_verts = vertices.shape[1]
    images = []
    depth_images = []
    if colors is None:
        colors = np.repeat(np.array([.6, .6, .6])[np.newaxis], n_verts, axis=0).reshape(-1)

    for vertices_single in vertices:
        mesh = trimesh.Trimesh(
            vertices_single * np.array([1, -1, -1], dtype=np.float32), faces,
            vertex_colors=colors)
        viewer.set_mesh(mesh)
        color_image, depth_image = viewer.render(render_wireframe=wireframe)
        images.append(color_image)
        depth_images.append(depth_image)

    return np.stack(images, axis=0), np.stack(depth_images, axis=0)


class Renderer2:
    def __init__(self):
        self.viewer = MeshViewer(
            imshape=(10, 10), camera=cameralib.Camera.from_fov(40, (10, 10)))

    def set_image_shape(self, imshape):
        self.viewer.viewer.viewport_height = imshape[0]
        self.viewer.viewer.viewport_width = imshape[1]

    def render_meshes(
            self, trimeshes, materials, camera, RGBA=False, wireframe=False, imshape=None,
            return_depth=False):
        self.set_image_shape(imshape)
        self.viewer.remove_meshes()
        for tmesh, material in zip(trimeshes, materials):
            verts_transformed = tmesh.vertices * np.array([1, -1, -1], dtype=np.float32)
            mesh = trimesh.Trimesh(verts_transformed, tmesh.faces)
            self.viewer.add_mesh(mesh, material=material)

        self.viewer.set_camera(camera)
        color_image, depth_image = self.viewer.render(render_wireframe=wireframe, RGBA=RGBA)
        if return_depth:
            return color_image, depth_image
        else:
            return color_image


class Renderer:
    def __init__(self, imshape=None, faces=None, wireframe=False, return_depth=False, color=None,
                 vertex_colors=None):
        if imshape is None:
            imshape = (10, 10)

        self.viewer = MeshViewer(
            imshape=imshape, camera=cameralib.Camera.from_fov(40, imshape), color=color)
        self.faces = faces
        self.wireframe = wireframe
        self.return_depth = return_depth
        self.vertex_colors = vertex_colors

    def set_image_shape(self, imshape):
        self.viewer.viewer.viewport_height = imshape[0]
        self.viewer.viewer.viewport_width = imshape[1]

    def render_meshes(
            self, trimeshes, camera, RGBA=False, wireframe=False, imshape=None, return_depth=False):
        self.viewer.remove_meshes()
        if imshape is not None:
            self.set_image_shape(imshape)
        for mesh in trimeshes:
            self.viewer.add_mesh(mesh)
        self.viewer.set_camera(camera)

        color_image, depth_image = self.viewer.render(render_wireframe=wireframe, RGBA=RGBA)
        if return_depth:
            return color_image, depth_image
        else:
            return color_image

    def render(self, vertices, camera, RGBA=False, seg=False, imshape=None, faces=None):
        verts_transformed = vertices * np.array([1, -1, -1], dtype=np.float32)
        if faces is None:
            faces = self.faces
        mesh = trimesh.Trimesh(verts_transformed, faces, vertex_colors=self.vertex_colors)
        if imshape is not None:
            self.set_image_shape(imshape)

        self.viewer.set_mesh(mesh)
        self.viewer.set_camera(camera)

        if seg:
            return self.viewer.render_seg()

        color_image, depth_image = self.viewer.render(render_wireframe=self.wireframe, RGBA=RGBA)
        if self.return_depth:
            return color_image, depth_image
        else:
            return color_image
