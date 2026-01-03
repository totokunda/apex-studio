import itertools
import multiprocessing
import os.path as osp
import subprocess

import PIL.Image
import cv2
import ffmpeg
import imageio
import more_itertools
import numpy as np
import torch
from posepile.util.improc import decode_jpeg


def image_files(
        image_paths, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=False, frame_preproc_fn=None, frame_preproc_size_fn=None,
        varying_resolutions=False, downscale_factor=1):
    if varying_resolutions:
        width, height = None, None
    else:
        height, width = image_extents(image_paths[0])
        width //= downscale_factor
        height //= downscale_factor
        if frame_preproc_size_fn is not None:
            width, height = frame_preproc_size_fn(width, height)

    return image_dataset_from_queue(
        images_from_paths_gen, args=(image_paths, downscale_factor), imshape=[height, width],
        extra_data=extra_data,
        internal_queue_size=internal_queue_size, batch_size=batch_size, prefetch_gpu=prefetch_gpu,
        tee_cpu=tee_cpu, frame_preproc_fn=frame_preproc_fn)


def video_file(
        video_path, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=False, video_slice=slice(None), downscale_factor=1,
        frame_preproc_fn=None, frame_preproc_size_fn=None):
    width, height = video_extents(video_path)
    width //= downscale_factor
    height //= downscale_factor

    if frame_preproc_size_fn is not None:
        width, height = frame_preproc_size_fn(width, height)

    return image_dataset_from_queue(
        sliced_reader, args=(video_path, video_slice, downscale_factor), imshape=[height, width],
        extra_data=extra_data, internal_queue_size=internal_queue_size, batch_size=batch_size,
        prefetch_gpu=prefetch_gpu, tee_cpu=tee_cpu, frame_preproc_fn=frame_preproc_fn)


def video_files(
        video_paths, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=False, video_slice=slice(None), downscale_factor=1, frame_preproc_fn=None,
        frame_preproc_size_fn=None):
    width, height = video_extents(video_paths[0])
    width //= downscale_factor
    height //= downscale_factor

    if frame_preproc_size_fn is not None:
        width, height = frame_preproc_size_fn(width, height)
    return image_dataset_from_queue(
        concat_frame_gen, args=(video_paths, video_slice, downscale_factor),
        imshape=[height, width],
        extra_data=extra_data, internal_queue_size=internal_queue_size, batch_size=batch_size,
        prefetch_gpu=prefetch_gpu, tee_cpu=tee_cpu, frame_preproc_fn=frame_preproc_fn)


def webcam(
        capture_id=0, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=True):
    height, width = video_capture_extents(capture_id)

    return image_dataset_from_queue(
        frames_from_webcam, args=(capture_id,), imshape=[height, width],
        extra_data=extra_data, internal_queue_size=internal_queue_size, batch_size=batch_size,
        prefetch_gpu=prefetch_gpu, tee_cpu=tee_cpu)


def youtube(
        url, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=True):
    import pafy
    import sys
    import yt_dlp
    sys.modules['youtube_dl'] = yt_dlp
    best_stream = pafy.new(url).getbest()
    width, height = best_stream.dimensions
    return image_dataset_from_queue(
        frames_of_youtube_stream, args=(best_stream.url,), imshape=[height, width],
        extra_data=extra_data, internal_queue_size=internal_queue_size, batch_size=batch_size,
        prefetch_gpu=prefetch_gpu, tee_cpu=tee_cpu)


def interleaved_video_files(
        video_paths, extra_data=None, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=False, video_slice=slice(None), frame_preproc_fn=None, frame_preproc_size_fn=None,
        downscale_factor=1):
    width, height = video_extents(video_paths[0])
    width /= downscale_factor
    height /= downscale_factor

    if frame_preproc_size_fn is not None:
        width, height = frame_preproc_size_fn(width, height)
    return image_dataset_from_queue(
        interleaved_frame_gen, args=(video_paths, video_slice, downscale_factor),
        imshape=[height, width],
        extra_data=extra_data, internal_queue_size=internal_queue_size, batch_size=batch_size,
        prefetch_gpu=prefetch_gpu, tee_cpu=tee_cpu, frame_preproc_fn=frame_preproc_fn)


def sliced_reader(path, video_slice, downscale_factor):
    output_params = ['-map', '0:v:0']
    if downscale_factor != 1:
        output_params.extend(['-vf', f'scale=iw/{downscale_factor}:ih/{downscale_factor}'])

    with imageio.get_reader(path, 'ffmpeg', output_params=output_params) as video:
        yield from itertools.islice(video, video_slice.start, video_slice.stop, video_slice.step)


def frames_from_webcam(capture_id):
    yield from video_capture_frames(capture_id)


def frames_of_youtube_stream(internal_url):
    yield from video_capture_frames(internal_url, cv2.CAP_FFMPEG)


def interleaved_frame_gen(video_paths, video_slice, downscale_factor):
    video_readers = [sliced_reader(p, video_slice, downscale_factor) for p in video_paths]
    yield from roundrobin(video_readers, [1] * len(video_readers))


def concat_frame_gen(video_paths, video_slice, downscale_factor):
    for p in video_paths:
        for frame in sliced_reader(p, video_slice, downscale_factor):
            yield frame, p


def images_from_paths_gen(paths, downscale_factor):
    for path in paths:
        if downscale_factor != 1 and osp.splitext(path)[1].lower() in ('.jpg', '.jpeg'):
            yield decode_jpeg(path, scale=1 / downscale_factor)
        else:
            frame = cv2.imread(path)
            if frame is None:
                raise FileNotFoundError(path)
            yield frame[..., ::-1]


def video_capture_frames(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        while (frame_bgr := cap.read()[1]) is not None:
            yield frame_bgr[..., ::-1]
    finally:
        cap.release()


def image_dataset_from_queue(
        generator_fn, imshape, extra_data, internal_queue_size=None, batch_size=64, prefetch_gpu=1,
        tee_cpu=False, frame_preproc_fn=None, args=None, kwargs=None, output_signature=None):
    if internal_queue_size is None:
        internal_queue_size = batch_size * 2 if batch_size is not None else 64

    q = multiprocessing.Queue(internal_queue_size)
    t = multiprocessing.Process(
        target=queue_filler_process, args=(generator_fn, q, args, kwargs))
    t.start()
    if frame_preproc_fn is None:
        frame_preproc_fn = lambda x: x

    def queue_reader():
        while (frame := q.get()) is not None:
            yield frame_preproc_fn(frame)

    frames = queue_reader()
    if tee_cpu:
        frames, frames2 = itertools.tee(frames, 2)
    else:
        frames2 = itertools.repeat(None)

    ds = tf.data.Dataset.from_generator(lambda: frames, output_signature=output_signature)

    if extra_data is not None:
        ds = tf.data.Dataset.zip((ds, extra_data))

    if batch_size is not None:
        ds = ds.batch(batch_size)
    if prefetch_gpu:
        ds = ds.apply(tf.data.experimental.prefetch_to_device('GPU:0', prefetch_gpu))

    if batch_size is not None:
        frames2 = more_itertools.chunked(frames2, batch_size)

    return ds, frames2


def queue_filler_process(generator_fn, q, args, kwargs):
    args = () if args is None else args
    kwargs = {} if kwargs is None else kwargs
    for item in generator_fn(*args, **kwargs):
        q.put(item)
    q.put(None)


def frames_from_video(video_path):
    # This direct method of piping an ffmpeg process was found to be fastest
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    args = ffmpeg.input(video_path).output('pipe:', format='rawvideo', pix_fmt='rgb24').compile()

    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        while read_bytes := proc.stdout.read(width * height * 3):
            yield np.frombuffer(read_bytes, np.uint8).reshape([height, width, 3])


def roundrobin(iterables, sizes):
    iterators = [iter(iterable) for iterable in iterables]
    for iterator, size in zip(itertools.cycle(iterators), itertools.cycle(sizes)):
        for _ in range(size):
            try:
                yield next(iterator)
            except StopIteration:
                return


def video_extents(filepath):
    """Returns the video (width, height) as a numpy array, without loading the pixel data."""

    with imageio.get_reader(filepath, 'ffmpeg') as reader:
        return tuple(reader.get_meta_data()['source_size'])


def image_extents(path):
    with PIL.Image.open(path) as im:
        width, height = im.size
    return height, width


def video_capture_extents(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return height, width
