<h1 align="center">Apex Studio</h1>

<hr />

<p align="center">
  Content creation made easy with a video editor built around open-source models.
</p>

<p align="center">
  <img src="assets/images/apex-studio.png" alt="Apex Studio" />
</p>

<p align="center">
  We built Apex around a simple belief: using free and open-source diffusion models should feel effortless. Making the model run shouldn’t be the challenge, the challenge should be the ambition, depth, and complexity of the content you choose to create.
</p>

## Documentation

<hr />

### [docuementation links](#)
### [the basics](#)

## Getting Started

<hr />

### Optional: Rust-accelerated downloader (for max download throughput)

`src/mixins/download_mixin.py` can use an optional Rust/PyO3 module (`apex_download_rs`) for faster URL downloads
(less Python overhead, better chance of saturating network/disk).

Build/install (requires Rust toolchain + cargo):

```bash
python -m pip install -U pip maturin
cd rust/apex_download_rs
maturin develop --release
```

Runtime toggles:
- `APEX_USE_RUST_DOWNLOAD=0` to disable (default is enabled if the module is importable)
- `APEX_DOWNLOAD_PROGRESS_INTERVAL` (seconds, default `0.2`) to throttle progress callbacks
- `APEX_DOWNLOAD_PROGRESS_MIN_BYTES` (bytes, default `1048576`) to throttle progress callbacks

Rust parallel range-download tuning (hf_transfer-style, signature unchanged; all optional):
- `APEX_RUST_DOWNLOAD_MAX_FILES` (default `8`): max concurrent range requests / file handles
- `APEX_RUST_DOWNLOAD_PARALLEL_FAILURES` (default `0`) + `APEX_RUST_DOWNLOAD_MAX_RETRIES` (default `0`): enable per-chunk retry with backoff
- `APEX_RUST_DOWNLOAD_RETRY_BASE_MS` (default `300`) and `APEX_RUST_DOWNLOAD_RETRY_MAX_MS` (default `10000`)

## 🚀Quick Start

- Packaged application
- Available for MacOS and Windows machines

### Terminal startup ([details](#terminal-startup))

- Seperated studio and engine
- Can be used with remote machines

## Features

<hr />

- Video editor built with JavaScript and Electron for easy video creation, and timeline based editing.
- Easy preprocessing by dragging your desired preprocessor model onto a media clip.
- Point-based masking with positive and negative markers for precise, intuitive control over exactly what gets masked.
- Use any LoRA from popular model hubs like Hugging Face or Civitai—bring your own checkpoints, styles, and character LoRAs and plug them directly into your workflow.
- Projects are saved as simple JSON files, making them easy to version, hand-edit and share
- No node graphs: projects stay straightforward and portable, improving cross-compatibility across machines, setups, and collaborators.
- Built-in queueing so you can line up multiple renders/generations and let Apex process them in order.
- Denoised latent previews at predetermined intervals, so you can watch generations evolve as they render.
- Built-in video controls including speed changes, frame interpolation, and keyframe selection.
- Editing controls for trimming, slicing, cropping, and much more.
- Hundreds of effects available to use within your creation.
- Audio controls including detaching audio from video, waveform manipulation, noise reduction, and more.

### Models

- **Image Models**
  - [chroma hd text to image](manifest/verified/image/chroma-hd-text-to-image-1.0.0.v1.yml)
  - [flux dev kontext](manifest/verified/image/flux-dev-kontext-1.0.0.v1.yml)
  - [flux dev text to image](manifest/verified/image/flux-dev-text-to-image-1.0.0.v1.yml)
  - [flux krea text to image](manifest/verified/image/flux-krea-text-to-image-1.0.0.v1.yml)
  - [nunchaku flux dev kontext](manifest/verified/image/nunchaku-flux-dev-kontext-1.0.0.v1.yml)
  - [nunchaku flux dev text to image](manifest/verified/image/nunchaku-flux-dev-text-to-image-1.0.0.v1.yml)
  - [nunchaku flux krea text to image](manifest/verified/image/nunchaku-flux-krea-text-to-image-1.0.0.v1.yml)
  - [nunchaku qwenimage edit 2509 lightning 8steps](manifest/verified/image/nunchaku-qwenimage-edit-2509-lightning-8steps-1.0.0.v1.yml)
  - [nunchaku qwenimage edit lightning 8steps](manifest/verified/image/nunchaku-qwenimage-edit-lightning-8steps-1.0.0.v1.yml)
  - [nunchaku qwenimage lightning 8steps](manifest/verified/image/nunchaku-qwenimage-lightning-8steps-1.0.0.v1.yml)
  - [qwenimage](manifest/verified/image/qwenimage-1.0.0.v1.yml)
  - [qwenimage edit](manifest/verified/image/qwenimage-edit-1.0.0.v1.yml)
  - [qwenimage edit 2509](manifest/verified/image/qwenimage-edit-2509-1.0.0.v1.yml)
  - [wan 2.2 a14b text to image 4 steps](manifest/verified/image/wan-2.2-a14b-text-to-image-4-steps-1.0.0.v1.yml)
  - [zimage turbo](manifest/verified/image/zimage-turbo-1.0.0.v1.yml)
  - [zimage turbo control](manifest/verified/image/zimage-turbo-control-1.0.0.v1.yml)

- **Video Models**
  - [hunyuanvideo 1.5 i2v](manifest/verified/video/hunyuanvideo-1.5-i2v-1.0.0.v1.yml)
  - [hunyuanvideo 1.5 t2v](manifest/verified/video/hunyuanvideo-1.5-t2v-1.0.0.v1.yml)
  - [wan 2.1 14b image to video 480p](manifest/verified/video/wan-2.1-14b-image-to-video-480p-1.0.0.v1.yml)
  - [wan 2.1 14b infinitetalk text to video](manifest/verified/video/wan-2.1-14b-infinitetalk-text-to-video-1.0.0.v1.yml)
  - [wan 2.1 14b multitalk text to video](manifest/verified/video/wan-2.1-14b-multitalk-text-to-video-1.0.0.v1.yml)
  - [wan 2.1 14b vace control](manifest/verified/video/wan-2.1-14b-vace-control-1.0.0.v1.yml)
  - [wan 2.1 14b vace expand swap](manifest/verified/video/wan-2.1-14b-vace-expand-swap-1.0.0.v1.yml)
  - [wan 2.1 14b vace painting](manifest/verified/video/wan-2.1-14b-vace-painting-1.0.0.v1.yml)
  - [wan 2.1 14b vace reference to video](manifest/verified/video/wan-2.1-14b-vace-reference-to-video-1.0.0.v1.yml)
  - [wan 2.2 14b animate](manifest/verified/video/wan-2.2-14b-animate-1.0.0.v1.yml)
  - [wan 2.2 5b text to image to video turbo](manifest/verified/video/wan-2.2-5b-text-to-image-to-video-turbo.1.0.0.v1.yml)
  - [wan 2.2 a14b text to video](manifest/verified/video/wan-2.2-a14b-text-to-video-1.0.0.v1.yml)
  - [wan 2.2 fun a14b control](manifest/verified/video/wan-2.2-fun-a14b-control-1.0.0.v1.yml)
  - [wan2.2 a14b first frame last frame](manifest/verified/video/wan2.2-a14b-first-frame-last-frame-1.0.0.v1.yml)
  - [wan2.2 a14b image to video](manifest/verified/video/wan2.2-a14b-image-to-video-1.0.0.v1.yml)

## Terminal Startup

<hr />



