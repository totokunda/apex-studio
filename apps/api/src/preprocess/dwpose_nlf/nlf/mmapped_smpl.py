import functools
import os

import numpy as np
import smplfitter.common
import smplfitter.np
from smplfitter.np.rotation import mat2rotvec, rotvec2mat
from smplfitter.np.util import matmul_transp_a

from nlf.paths import DATA_ROOT

@functools.lru_cache()
def get_cached_body_model(model_name, gender):
    return SMPLBodyModelMmap(model_name, gender)

class SMPLBodyModelMmap:
    def __init__(self, model_name='smpl', gender='neutral', num_betas=None):
        """
        Represents a statistical body model of the SMPL family.

        The SMPL (Skinned Multi-Person Linear) model provides a way to represent articulated 3D
        human
        meshes through a compact shape vector (beta) and pose (body part rotation) parameters.

        Parameters:
            model_name: Name of the model type.
            gender: Gender of the model, which can be 'neutral', 'female' or 'male'.
            model_root: Path to the directory containing model files. By default,
                {DATA_ROOT}/body_models/{model_name} is used, with the DATA_ROOT environment
                variable.
            num_betas: Number of shape parameters (betas) to use. By default, all available betas are
                used.
        """

        self.gender = gender
        self.model_name = model_name
        gender = dict(f='female', n='neutral', m='male')[gender[0].lower()]
        dirpath = f'{DATA_ROOT}/body_models/mmap/mmap_{model_name}_{gender}'

        self.v_template = np.load(f'{dirpath}/v_template.npy', mmap_mode='r')
        self.shapedirs = np.load(f'{dirpath}/shapedirs.npy', mmap_mode='r')
        self.posedirs = np.load(f'{dirpath}/posedirs.npy', mmap_mode='r')
        self.J_regressor = np.load(f'{dirpath}/J_regressor.npy', mmap_mode='r')
        self.J_template = np.load(f'{dirpath}/J_template.npy', mmap_mode='r')
        self.J_shapedirs = np.load(f'{dirpath}/J_shapedirs.npy', mmap_mode='r')
        self.kid_shapedir = np.load(f'{dirpath}/kid_shapedir.npy', mmap_mode='r')
        self.kid_J_shapedir = np.load(f'{dirpath}/kid_J_shapedir.npy', mmap_mode='r')
        self.weights = np.load(f'{dirpath}/weights.npy', mmap_mode='r')
        self.kintree_parents = np.load(f'{dirpath}/kintree_parents.npy', mmap_mode='r')
        self.faces = np.load(f'{dirpath}/faces.npy', mmap_mode='r')
        self.num_joints = len(self.J_template)
        self.num_vertices = len(self.v_template)

    def __call__(
            self, pose_rotvecs=None, shape_betas=None, trans=None, kid_factor=None,
            rel_rotmats=None, glob_rotmats=None, *, return_vertices=True):
        """Calculate the SMPL body model vertices, joint positions and orientations given the input
        pose and shape parameters.

        Args:
            pose_rotvecs (np.ndarray): An array of shape (batch_size, num_joints * 3),
                representing the rotation vectors for each joint in the pose.
            shape_betas (np.ndarray): An array of shape (batch_size, num_shape_coeffs),
                representing the shape coefficients (betas) for the body shape.
            trans (np.ndarray, optional): An array of shape (batch_size, 3), representing the
                translation of the root joint. Defaults to None, in which case a zero translation is
                applied.
            return_vertices (bool, optional): A flag indicating whether to return the body model
                vertices. If False, only joint positions and orientations are returned.
                Defaults to True.

        Returns:
            A dictionary containing the following keys and values:
                - 'vertices': An array of shape (batch_size, num_vertices, 3), representing the
                    3D body model vertices in the posed state. This key is only present if
                    `return_vertices` is True.
                - 'joints': An array of shape (batch_size, num_joints, 3), representing the 3D
                    positions of the body joints.
                - 'orientations': An array of shape (batch_size, num_joints, 3, 3), representing
                    the 3D orientation matrices for each joint.
        """

        batch_size = check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats)

        if rel_rotmats is not None:
            rel_rotmats = np.asarray(rel_rotmats, np.float32)
        elif pose_rotvecs is not None:
            pose_rotvecs = np.asarray(pose_rotvecs, np.float32)
            rel_rotmats = rotvec2mat(np.reshape(pose_rotvecs, (batch_size, self.num_joints, 3)))
        elif glob_rotmats is None:
            rel_rotmats = np.tile(
                np.eye(3, dtype=np.float32), [batch_size, self.num_joints, 1, 1])

        if glob_rotmats is None:
            glob_rotmats = [rel_rotmats[:, 0]]
            for i_joint in range(1, self.num_joints):
                i_parent = self.kintree_parents[i_joint]
                glob_rotmats.append(glob_rotmats[i_parent] @ rel_rotmats[:, i_joint])
            glob_rotmats = np.stack(glob_rotmats, axis=1)

        parent_indices = self.kintree_parents[1:]
        parent_glob_rotmats = np.concatenate([
            np.tile(np.eye(3), [glob_rotmats.shape[0], 1, 1, 1]),
            glob_rotmats[:, parent_indices]], axis=1)

        if rel_rotmats is None:
            rel_rotmats = matmul_transp_a(parent_glob_rotmats, glob_rotmats)

        if shape_betas is None:
            shape_betas = np.zeros((batch_size, 0), np.float32)
        else:
            shape_betas = np.asarray(shape_betas, np.float32)
        num_betas = np.minimum(shape_betas.shape[1], self.shapedirs.shape[2])

        if kid_factor is None:
            kid_factor = np.zeros((1,), np.float32)
        else:
            kid_factor = np.float32(kid_factor)

        j = (self.J_template +
             np.einsum(
                 'jcs,bs->bjc', self.J_shapedirs[:, :, :num_betas],
                 shape_betas[:, :num_betas]) +
             np.einsum('jc,b->bjc', self.kid_J_shapedir, kid_factor))

        glob_rotmats = [rel_rotmats[:, 0]]
        glob_positions = [j[:, 0]]

        for i_joint in range(1, self.num_joints):
            i_parent = self.kintree_parents[i_joint]
            glob_rotmats.append(glob_rotmats[i_parent] @ rel_rotmats[:, i_joint])
            glob_positions.append(
                glob_positions[i_parent] +
                np.einsum('bCc,bc->bC', glob_rotmats[i_parent], j[:, i_joint] - j[:, i_parent]))

        glob_rotmats = np.stack(glob_rotmats, axis=1)
        glob_positions = np.stack(glob_positions, axis=1)

        if trans is None:
            trans = np.zeros((1, 3), np.float32)
        else:
            trans = trans.astype(np.float32)

        if not return_vertices:
            return dict(
                joints=(glob_positions + trans[:, np.newaxis]),
                orientations=glob_rotmats)

        pose_feature = np.reshape(rel_rotmats[:, 1:], [-1, (self.num_joints - 1) * 3 * 3])
        v_posed = (
                self.v_template +
                np.einsum(
                    'vcp,bp->bvc', self.shapedirs[:, :, :num_betas], shape_betas[:, :num_betas]) +
                np.einsum('vcp,bp->bvc', self.posedirs, pose_feature) +
                np.einsum('vc,b->bvc', self.kid_shapedir, kid_factor))

        translations = glob_positions - np.einsum('bjCc,bjc->bjC', glob_rotmats, j)
        vertices = (
                np.einsum('bjCc,vj,bvc->bvC', glob_rotmats, self.weights, v_posed) +
                self.weights @ translations)

        return dict(
            vertices=vertices + trans[:, np.newaxis],
            joints=glob_positions + trans[:, np.newaxis],
            orientations=glob_rotmats)

    def single(self, *args, return_vertices=True, **kwargs):
        args = [np.expand_dims(x, axis=0) for x in args]
        kwargs = {k: np.expand_dims(v, axis=0) for k, v in kwargs.items()}
        if len(args) == 0 and len(kwargs) == 0:
            kwargs['shape_betas'] = np.zeros((1, 0), np.float32)
        result = self(*args, return_vertices=return_vertices, **kwargs)
        return {k: np.squeeze(v, axis=0) for k, v in result.items()}

    def rototranslate(
            self, R, t, pose_rotvecs, shape_betas, trans, kid_factor=0, post_translate=True):
        """Rotate and translate the SMPL body model carefully, taking into account that the
        global orientation is applied with the pelvis as anchor, not the origin of the canonical
        coordinate system!
        The translation vector needs to be changed accordingly, too, not just the pose.
        """
        current_rotmat = rotvec2mat(pose_rotvecs[:3])
        new_rotmat = R @ current_rotmat
        new_pose_rotvec = np.concatenate(
            [mat2rotvec(new_rotmat), pose_rotvecs[3:]], axis=0)

        pelvis = (
                self.J_template[0] +
                self.J_shapedirs[0, :, :shape_betas.shape[0]] @ shape_betas +
                self.kid_J_shapedir[0] * kid_factor
                 )

        if post_translate:
            new_trans = pelvis @ (R.T - np.eye(3)) + trans @ R.T + t
        else:
            new_trans = pelvis @ (R.T - np.eye(3)) + (trans - t) @ R.T
        return new_pose_rotvec, new_trans



def check_batch_size(pose_rotvecs, shape_betas, trans, rel_rotmats):
    batch_sizes = [
        np.asarray(x).shape[0] for x in [pose_rotvecs, shape_betas, trans, rel_rotmats]
        if x is not None]

    if len(batch_sizes) == 0:
        raise RuntimeError(
            'At least one argument must be given among pose_rotvecs, shape_betas, trans, '
            'rel_rotmats.')

    if not all(b == batch_sizes[0] for b in batch_sizes[1:]):
        raise RuntimeError('The batch sizes must be equal.')

    return batch_sizes[0]


def prepare(model_name, gender, num_betas):
    model = smplfitter.np.BodyModel(model_name, gender)
    out_dir = f'{DATA_ROOT}/body_models/mmap/mmap_{model_name}_{gender}'
    os.makedirs(out_dir, exist_ok=True)
    for key in ['v_template', 'shapedirs', 'posedirs', 'J_regressor', 'J_template', 'J_shapedirs',
                'kid_shapedir', 'kid_J_shapedir', 'weights', 'kintree_parents', 'faces']:
        val = getattr(model, key)
        if key in ['shapedirs', 'J_shapedirs']:
            val = val[:, :, :num_betas]
        np.save(f'{out_dir}/{key}.npy', val)


def prepare_all(num_betas=128):
    for model_name in ['smpl', 'smplx', 'smplxlh', 'smplh16']:
        for gender in ['neutral', 'male', 'female']:
            prepare(model_name, gender, num_betas)

    for model_name in ['smplh']:
        for gender in ['male', 'female']:
            prepare(model_name, gender, num_betas)


if __name__ == '__main__':
    prepare_all(num_betas=128)
