import re

def list_adapters(obj):
    out = {}

    # Diffusers mixin-style (named adapters)
    for attr in ["get_adapters", "list_adapters", "adapters"]:
        if hasattr(obj, attr):
            try:
                v = getattr(obj, attr)
                out[attr] = v() if callable(v) else v
            except Exception as e:
                out[attr] = f"<error: {e}>"

    # PEFT-style (adapter names are keys)
    if hasattr(obj, "peft_config"):
        try:
            out["peft_config_keys"] = list(getattr(obj, "peft_config").keys())
        except Exception as e:
            out["peft_config_keys"] = f"<error: {e}>"

    # Active adapters (varies by PEFT version)
    for attr in ["active_adapters", "active_adapter", "get_active_adapters"]:
        if hasattr(obj, attr):
            try:
                v = getattr(obj, attr)
                out[attr] = v() if callable(v) else v
            except Exception as e:
                out[attr] = f"<error: {e}>"

    return out


def disable_adapters_by_filter(obj, *, contains=None, regex=None):
    # Get all names
    info = list_adapters(obj)
    if isinstance(info.get("peft_config_keys"), list):
        all_names = info["peft_config_keys"]
    else:
        # best-effort fallback
        all_names = []
        for k in ("get_adapters", "list_adapters"):
            if isinstance(info.get(k), list):
                all_names = info[k]
                break

    target = set(filter_names(all_names, contains=contains, regex=regex))
    if not target:
        return [], "no matches"

    # If we can set adapter weights, do it
    if hasattr(obj, "set_adapters"):
        keep = [n for n in all_names]  # keep ordering
        weights = [0.0 if n in target else 1.0 for n in keep]
        obj.set_adapters(keep, adapter_weights=weights)
        return list(target), "disabled via set_adapters(weights=0)"

    # PEFT fallback: if there’s a global disable
    if hasattr(obj, "disable_adapter"):
        # disables all adapters in PEFT; not per-name
        obj.disable_adapter()
        return list(target), "called disable_adapter() (global, not per-name)"

    return list(target), "no supported per-name disable API found"