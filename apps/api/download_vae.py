p = "Tongyi-MAI/Z-Image-Turbo/vae"
from src.mixins.download_mixin import DownloadMixin
dl = DownloadMixin()
dl.download(p, "./weights")