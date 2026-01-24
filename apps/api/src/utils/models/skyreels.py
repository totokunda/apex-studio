import torch

DISABLE_COMPILE = False


def mul_add(x, y, z):
    return x.float() + y.float() * z


def mul_add_add(x, y, z):
    return x.float() * (1 + y) + z


def mul_add_unflatten(x, y, z, num_frames, frame_seqlen):
    return x.float() * (
        y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) + z
    ).flatten(1, 2)


def mul_add_add_unflatten(x, y, z, num_frames, frame_seqlen):
    return x.float().unflatten(dim=1, sizes=(num_frames, frame_seqlen)) + (1 + y) * z


mul_add_compile = torch.compile(mul_add, dynamic=True, disable=DISABLE_COMPILE)
mul_add_add_compile = torch.compile(mul_add_add, dynamic=True, disable=DISABLE_COMPILE)
mul_add_unflatten_compile = torch.compile(
    mul_add_unflatten, dynamic=True, disable=DISABLE_COMPILE
)
mul_add_add_unflatten_compile = torch.compile(
    mul_add_add_unflatten, dynamic=True, disable=DISABLE_COMPILE
)
