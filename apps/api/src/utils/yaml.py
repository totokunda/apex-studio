import yaml
from pathlib import Path
import pprint

from src.utils.scheduler_manifest import expand_scheduler_manifests


# 1) Create a Loader subclass that we can attach extra state to
class LoaderWithInclude(yaml.FullLoader):
    # we'll attach .shared_manifests = { alias: Path(...) }
    pass


def _dummy_include(loader, node):
    # we don't care about the value, just skip it
    return None


def _real_include(loader: LoaderWithInclude, node):
    """
    Called when parsing !include alias:component_name.
    """
    # the scalar is e.g. "shared:wan/vae"
    prefix, comp_name = loader.construct_scalar(node).split(":", 1)

    if prefix == "shared":
        # e.g. comp_name is "wan/vae", so we extract "wan" as the alias
        alias = comp_name.split("/", 1)[0]
    else:
        alias = prefix

    manifest = loader.shared_manifests.get(alias)
    if manifest is None:
        # Try to auto-resolve typical shared file layouts
        base_dir = getattr(loader, "base_dir", None)
        manifest_root = getattr(loader, "manifest_root", None)
        candidates = []
        for root in filter(None, [base_dir, manifest_root]):
            root = Path(root)
            candidates.extend(
                [
                    root / f"shared_{alias}.yml",
                    root / f"shared_{alias}.yaml",
                    root / alias / "shared.yml",
                    root / alias / "shared.yaml",
                    root / alias / "shared.v1.yml",
                    root / alias / "shared.v1.yaml",
                ]
            )
        found = next((p for p in candidates if p.exists()), None)
        if found is not None:
            loader.shared_manifests[alias] = found
            manifest = found
        else:
            raise yaml.constructor.ConstructorError(
                None, None, f"Unknown shared alias {alias!r}", node.start_mark
            )

    # Load shared manifest. Support v1 shared files via shared_loader.
    text = manifest.read_text()
    try:
        from src.manifest.shared_loader import load_shared_manifest

        shared_doc = load_shared_manifest(manifest)
    except Exception:
        # Fallback: load raw with includes
        shared_doc = yaml.load(text, Loader=LoaderWithInclude)

    # find the component by name in any top-level list
    for value in shared_doc.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("name") == comp_name:
                    return item

    raise yaml.constructor.ConstructorError(
        None,
        None,
        f"Component named {comp_name!r} not found in {manifest}",
        node.start_mark,
    )


yaml.FullLoader.add_constructor("!include", _dummy_include)
LoaderWithInclude.add_constructor("!include", _real_include)


def load_yaml(file_path: str | Path):
    file_path = Path(file_path)
    text = file_path.read_text()
    # --- PASS 1: extract `shared:` or `spec.shared` with a loader that skips !include tags ---
    prelim = yaml.load(text, Loader=yaml.FullLoader)
    shared_entries = []
    if isinstance(prelim, dict):
        shared_entries.extend(prelim.get("shared", []) or [])
        spec = prelim.get("spec", {}) or {}
        if isinstance(spec, dict):
            shared_entries.extend(spec.get("shared", []) or [])
    # build alias → manifest Path
    shared_manifests = {}
    for entry in shared_entries:
        p = (file_path.parent / entry).resolve()
        stem = p.stem  # e.g. 'shared_wan' or 'shared.v1' or 'shared'
        alias = None
        if stem.startswith("shared_"):
            alias = stem.split("_", 1)[1]
        elif stem == "shared" or stem.startswith("shared."):
            # Infer alias from parent directory (e.g., manifest/wan/shared.v1.yml → 'wan')
            parent_name = p.parent.name
            if parent_name and parent_name != "manifest":
                alias = parent_name
        if not alias:
            # Fallbacks: try parent dir name, else stem
            parent_name = p.parent.name
            alias = parent_name if parent_name else stem
        shared_manifests[alias] = p
    # attach it to our custom loader

    LoaderWithInclude.shared_manifests = shared_manifests
    # Provide resolution hints to the include loader
    LoaderWithInclude.base_dir = file_path.parent
    # Find nearest 'manifest' directory as root for shared lookups
    manifest_root = None
    for parent in file_path.parents:
        if parent.name == "manifest":
            manifest_root = parent
            break
    LoaderWithInclude.manifest_root = manifest_root or file_path.parent
    # --- PASS 2: real load with !include expansion ---
    loaded = yaml.load(text, Loader=LoaderWithInclude)
    # Best-effort: allow scheduler manifests to live separately
    try:
        loaded = expand_scheduler_manifests(
            loaded,
            base_path=file_path,
            manifest_root=LoaderWithInclude.manifest_root,
        )
    except Exception:
        pass
    return loaded
