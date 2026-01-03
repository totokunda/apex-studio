import os.path as osp

from nlf.paths import DATA_ROOT

TRAIN = 0
VALID = 1
TEST = 2


def ensure_absolute_path(path, root=DATA_ROOT):
    if not root:
        return path

    if osp.isabs(path):
        return path
    else:
        return osp.join(root, path)
