# Image Studio - Agent Guide & Repository Knowledge

## Overview
Image Studio is a high-performance native desktop AI creation suite for Windows, built entirely in Rust with `eframe` (egui + glow). Zero Electron, zero Chromium.

It features:
1. **Image Studio**: Non-destructive sequential branching editor powered by Qwen Image 2.1 (standard INT8 or 8GB quant).
2. **Video Studio**: High-fidelity text-to-video and image-to-video generation powered by LTX-Video 2.3 (22B Dev FP8 with integrated spatial upscaler, audio VAE, and optional Gemma 3 12B cinematic prompt enhancer).
3. **Native Video Player**: Integrated custom video player (`src/video.rs`) using OpenH264 decoding and Rodio audio playback rendered directly into egui textures with scrubbing, playback speed, looping, and frame stepping.
4. **Embedded Engine Supervisor**: Self-contained ComfyUI portable environment runner (`src/engine.rs`) managed inside a Windows Job Object to guarantee clean process lifecycle and termination.

---

## Core Guidelines & Architectural Rules

### 1. UI Constraints
- **Strictly NO Emojis**: Do not use emojis in UI labels, tooltips, buttons, logs, or error popups. Maintain a sleek, modern, technical aesthetic.
- **Color Palette**: Dark theme with tailored accents (`PRIMARY`, `CARD_BG`, `BORDER`, `ACCENT`, `GREEN`, `DIM`).

### 2. ComfyUI & Engine Integration
- **Never pass `--reserve-vram` buffer flags**: ComfyUI's dynamic memory manager (`comfy-aimdo`) natively offloads and pages model weights between system RAM and GPU VRAM. An artificial VRAM buffer deprives the engine and causes severe PCIe swapping stalls.
- **Prompt Enhancer**: Uses `TextGenerateLTX2Prompt` with `max_length: 2048`. The LLM automatically terminates on its `<end_of_turn>` token (~250-300 tokens, taking ~100-120s on a 12GB GPU).
- **Model Initializing Swap**: When transitioning from Gemma 3 text generation to LTXAV video diffusion, ComfyUI pauses for ~25-30s while weights are staged into VRAM. Do not interpret this pause as a stall.
- **Cancellation**: When cancelling generations, send `/interrupt` and clear `/queue` to free memory immediately.

---

## Deployment & Update Server

The application contains an automatic self-updater (`src/updater.rs`) that checks `https://desinc.my.to/image-studio/version.json` at launch.

### Update Server Specification
- **Host**: `ds@desinc.my.to`
- **SSH Auth**: Key-based (connects via `ssh -o BatchMode=yes ds@desinc.my.to`).
- **Remote App Directory**: `/opt/projects/image-studio/app`
- **Web Root Symlink**: `/var/www/desite/image-studio -> /opt/projects/image-studio/app`
- **Files Hosted**:
  - `ImageStudio.exe` (Release binary)
  - `version.json` and `manifest.json`:
    ```json
    {
      "version": "x.y.z",
      "download_url": "https://desinc.my.to/image-studio/ImageStudio.exe"
    }
    ```

### How to Deploy
Run the automated deploy script:
```powershell
./deploy.ps1
```
This script will:
1. Detect version from `Cargo.toml`.
2. Compile optimized release binary (`cargo build --release`).
3. Generate `version.json` and `manifest.json` (UTF-8 without BOM).
4. Upload `ImageStudio.exe`, `version.json`, and `manifest.json` to `ds@desinc.my.to:/opt/projects/image-studio/app` via SCP.
5. Verify live endpoints at `https://desinc.my.to/image-studio/version.json` and `manifest.json`.

---

## Versioning & GitHub Releases

1. **Bump Version**: Increment `version = "x.y.z"` in `Cargo.toml`.
2. **Git Commit & Push**:
   ```bash
   git add .
   git commit -m "feat/fix: ..."
   git push origin main
   ```
3. **Git Tag**:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
4. **Deploy**:
   ```powershell
   ./deploy.ps1
   ```
