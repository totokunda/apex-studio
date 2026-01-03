import multiprocessing

import cameralib
import cv2
import florch.callbacks
import imageio.v2 as imageio
import numpy as np
import simplepyutils as spu
import torch
from simplepyutils import FLAGS

from nlf.paths import DATA_ROOT, PROJDIR
from nlf.rendering import Renderer


class RenderPredictionCallback(florch.callbacks.Callback):
    def __init__(self, start_step=0, interval=100, image_dir=None):
        super().__init__()

        if image_dir is None:
            image_dir = f'{DATA_ROOT}/wild_crops/'

        self.image_dir = image_dir
        self.start_step = start_step
        self.interval = interval
        self.image_stack_torch = None
        self.intrinsics = None
        self.canonical_points = None
        self.faces = None
        self.q = None
        self.renderer_process = None

    def on_train_begin(self, initial_step):
        image_paths = spu.sorted_recursive_glob(f'{self.image_dir}/*.*')
        self.image_stack_np = np.stack(
            [
                cv2.resize(imageio.imread(p)[..., :3], (FLAGS.proc_side, FLAGS.proc_side))
                for p in image_paths
            ],
            axis=0,
        )
        self.camera = cameralib.Camera.from_fov(30, [FLAGS.proc_side, FLAGS.proc_side])

        with self.device:
            self.image_stack_torch = torch.tensor(
                self.image_stack_np, dtype=torch.float32
            ).permute(0, 3, 1, 2).contiguous() / np.float32(255.0)

            self.intrinsics = (
                torch.tensor(self.camera.intrinsic_matrix, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(len(image_paths), 1, 1)
            )
            self.canonical_points = torch.tensor(
                np.load(f'{PROJDIR}/canonical_vertices_smplx.npy'), dtype=torch.float32
            )

        self.camera.scale_output(512 / FLAGS.proc_side)
        self.faces = np.load(f'{PROJDIR}/smplx_faces.npy')
        self.q = multiprocessing.JoinableQueue(10)
        self.renderer_process = multiprocessing.Process(
            target=smpl_render_loop,
            args=(self.q, self.image_stack_np, self.camera, self.faces, FLAGS.logdir),
            daemon=True,
        )
        self.renderer_process.start()

    def on_train_batch_end(self, step, logs):
        if step % self.interval == 0 and step >= self.start_step:
            with torch.inference_mode(), torch.amp.autocast(
                'cuda', dtype=torch.float16
            ), self.device:
                self.trainer.eval()
                pred_vertices, uncerts = self.trainer.model.predict_multi_same_canonicals(
                    self.image_stack_torch, self.intrinsics, self.canonical_points
                )
                pred_vertices = pred_vertices.cpu().numpy() / 1000
            self.q.put((step, pred_vertices))

    def on_train_end(self, step):
        self.q.put(None)
        self.q.join()
        # self.renderer_process.join()

    def __del__(self):
        if self.renderer_process is not None and self.renderer_process.is_alive():
            self.q.put(None)
            self.q.join()


def smpl_render_loop(q, image_stack, camera, faces, logdir):
    spu.terminate_on_parent_death()  # maybe not needed with daemon=True
    renderer = Renderer(imshape=(512, 512), faces=faces)
    image_stack = np.array(
        [cv2.resize(im, (512, 512), interpolation=cv2.INTER_CUBIC) for im in image_stack]
    )

    while True:
        elem = q.get()
        if elem is not None:
            batch, pred_vertices = elem
            triplets = [
                make_triplet(im, verts, renderer, camera)
                for im, verts in zip(image_stack, pred_vertices)
            ]
            grid = np.concatenate(
                [
                    np.concatenate(triplets[:3], axis=1),
                    np.concatenate(triplets[3:6], axis=1),
                    np.concatenate(triplets[6:9], axis=1),
                ],
                axis=0,
            )
            path = f'{logdir}/pred_{batch:07d}.jpg'
            imageio.imwrite(path, grid, quality=93)

        q.task_done()
        if elem is None:
            break


def alpha_blend(im1, im2, w1):
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    w1 = w1.astype(np.float32) / 255
    w1 = np.expand_dims(w1, axis=-1)
    res = im1 * w1 + im2 * (1 - w1)
    return np.clip(res, 0, 255).astype(np.uint8)


def make_triplet(image_original, pred_vertices, renderer, camera_front):
    mean = np.mean(pred_vertices, axis=0)
    image_front = renderer.render(pred_vertices, camera_front, RGBA=True)
    image_front = alpha_blend(image_front[..., :3], image_original, image_front[..., 3])
    camera_side = camera_front.orbit_around(mean, np.pi / 2, inplace=False)
    # move a bit further back from the mean, by 1.5 times the distance from the mean to the camera
    camera_side.t = camera_side.t + (camera_side.t - mean)
    image_side = renderer.render(pred_vertices, camera_side)
    return np.concatenate([image_original, image_front, image_side], axis=1)
