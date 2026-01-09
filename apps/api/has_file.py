from src.mixins.download_mixin import DownloadMixin
from src.utils.defaults import get_components_path
from safetensors.torch import load_file
import hashlib
components_path = get_components_path()
mx = DownloadMixin()
path = "totoku/ovi/transformer/ovi-fp8-960x960.safetensors"
real_path = mx.is_downloaded(path, components_path)

contents = load_file(real_path)
# create a sha hash of the contents
sha_hash = hashlib.sha256(str(contents).encode()).hexdigest()
print(sha_hash)