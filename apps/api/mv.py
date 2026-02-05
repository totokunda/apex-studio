from huggingface_hub import HfFileSystem

fs = HfFileSystem()

paths_to_change = {
    "MOVA-720": "MOVA-720p"
}

def collect_all_paths(fs, current_path, all_paths):
    """Recursively collect all file and directory paths."""
    try:
        items = fs.ls(current_path, detail=True)
        
        for item in items:
            item_path = item["name"]
            all_paths.append((item_path, item["type"]))
            
            if item["type"] == "directory":
                collect_all_paths(fs, item_path, all_paths)
    except Exception as e:
        print(f"Error listing {current_path}: {e}")


def move_recursive(fs, base_path, old_path, new_path):
    """Recursively move all files and directories, processing deepest items first."""
    # Collect all paths first
    all_paths = []
    collect_all_paths(fs, base_path, all_paths)
    
    # Sort by path depth (deepest first) to avoid moving parent before children
    all_paths.sort(key=lambda x: x[0].count("/"), reverse=True)
    
    # Move each item
    for old_item_path, item_type in all_paths:
        new_item_path = old_item_path.replace(old_path, new_path)
        
        if old_item_path != new_item_path:
            try:
                print(f"Moving {item_type}: {old_item_path} -> {new_item_path}")
                fs.mv(old_item_path, new_item_path)
            except Exception as e:
                print(f"Error moving {old_item_path}: {e}")
    
    # Finally, move the root directory itself if it still exists
    new_base_path = base_path.replace(old_path, new_path)
    if base_path != new_base_path and fs.exists(base_path):
        try:
            print(f"Moving root directory: {base_path} -> {new_base_path}")
            fs.mv(base_path, new_base_path)
        except Exception as e:
            print(f"Error moving root directory {base_path}: {e}")


for old_path, new_path in paths_to_change.items():
    base_path = f"totoku/apex-models/{old_path}"
    
    print(f"\n{'='*60}")
    print(f"Processing: {base_path} -> {base_path.replace(old_path, new_path)}")
    print(f"{'='*60}")
    
    if fs.exists(base_path):
        move_recursive(fs, base_path, old_path, new_path)
    else:
        print(f"Path does not exist: {base_path}")