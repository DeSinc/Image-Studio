# Image Studio

Fast, native desktop app for image editing using open-weights diffusion models (Qwen Image 2.1 Multi-Ref) powered by ComfyUI running in the background.

## Features

- **Pure Native Rust UI**: Built with `egui` / `eframe` — zero WebView, zero Chromium, zero web frameworks. Instant startup and responsive native performance.
- **Background Engine Supervisor**: Automatically detects, bootstraps, and supervises ComfyUI in the background without terminal clutter.
- **Model Manifest & On-Demand Downloader**: Verifies required model weights on startup (`models.json`) and downloads missing weights from Hugging Face with live progress.
- **Safe Process Lifecycle**: Uses Windows Job Objects (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) to guarantee that all ComfyUI and Python subprocesses terminate immediately when Image Studio closes or crashes.
- **Branching History Tree**: Non-destructive editing with branching undo/redo history. Jump back to any previous step, diverge into a new branch without losing existing work, or switch between revision threads.
- **Split Comparison & Centered Canvas**: Pan and zoom inspection canvas with interactive A/B comparison slider.
- **Presets & Speed Optimization**: Quick presets (Fast, Balanced, Quality) with configurable resolutions, steps, samplers, schedulers, and live execution warnings.

## Running

Double-click `ImageStudio.exe` or launch from terminal:

```powershell
.\ImageStudio.exe
```

## Building from Source

Requires Rust 1.85+:

```powershell
cargo build --release
```

## License

This project is licensed under the [MIT License](LICENSE).
Model weights referenced in `models.json` are open-weights under their respective licenses (e.g. Apache 2.0). No model weights or proprietary assets are bundled in this repository.
