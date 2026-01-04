bundle_path=/Users/tosinkuye/apex-workspace/apex-studio/apps/api/dist/python-api/apex-engine
python_path=$bundle_path/apex-studio/bin/python

$python_path $bundle_path/scripts/setup.py \
    --apex_home_dir /Users/tosinkuye \
    --mask_model_type sam2_base_plus \
     --install_rife \
     --enable_image_render_steps \
    --enable_video_render_steps \
    --log_progress_callbacks