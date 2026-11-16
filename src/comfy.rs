//! ComfyUI HTTP client: upload, queue (verified Qwen Image 2.1 edit graph), poll, fetch.
use crate::engine::base_url;
use reqwest::blocking::{multipart, Client};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct GenResult {
    pub png: Vec<u8>,
    pub output_filename: String,
    pub output_subfolder: String,
    pub temp_files: Vec<String>,
}

pub struct VideoGenResult {
    pub mp4: Vec<u8>,
    pub output_filename: String,
    pub output_subfolder: String,
    pub temp_files: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VideoArch {
    MiniMaxH3,
    Ltx25,
    Ltx23,
}

pub struct VideoParams {
    pub arch: VideoArch,
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub duration: u32,
    pub fps: u32,
    pub seed: u64,
    pub steps: u32,
    pub cfg: f32,
    pub sampler: String,
    pub scheduler: String,
    pub denoise: f32,
    pub prompt_enhance: bool,
    pub last_frame: Option<Vec<u8>>,
    pub ref_images: Vec<Vec<u8>>,
    pub use_r2v: bool,
    pub lora_name: Option<String>,
    pub lora_strength: f32,
    pub unet_name: String,
    pub ref2va_unet_name: String,
    pub clip_name: String,
    pub vae_name: String,
    pub audio_vae_name: String,
    pub ckpt_name: String,
    pub text_encoder: String,
    pub latent_upscale_model: String,
    pub distilled_lora: String,
    pub gemma_lora: String,
}

impl Default for VideoParams {
    fn default() -> Self {
        Self {
            arch: VideoArch::MiniMaxH3,
            prompt: String::new(),
            negative: String::new(),
            width: 1280,
            height: 704,
            duration: 5,
            fps: 24,
            seed: 42,
            steps: 20,
            cfg: 1.0,
            sampler: "er_sde".to_string(),
            scheduler: "sgm_uniform".to_string(),
            denoise: 1.0,
            prompt_enhance: true,
            last_frame: None,
            ref_images: Vec::new(),
            use_r2v: false,
            lora_name: None,
            lora_strength: 1.0,
            unet_name: "MiniMax-H3-FL2VA-Pruned-Q4_K_M.gguf".to_string(),
            ref2va_unet_name: "MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf".to_string(),
            clip_name: "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors".to_string(),
            vae_name: "minimax_h3_video_vae_int8_convrot.safetensors".to_string(),
            audio_vae_name: "minimax_h3_audio_vae_fp32.safetensors".to_string(),
            ckpt_name: "ltx-2.3-22b-dev-fp8.safetensors".to_string(),
            text_encoder: "gemma_3_12B_it_fp4_mixed.safetensors".to_string(),
            latent_upscale_model: "ltx-2.3-spatial-upscaler-x2-1.1.safetensors".to_string(),
            distilled_lora: "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors".to_string(),
            gemma_lora: "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors".to_string(),
        }
    }
}

pub struct Params {
    pub prompt: String,
    pub negative: String,
    pub steps: u32,
    pub cfg: f32,
    pub seed: u64,
    pub resolution: u32,
    /// None = canvas follows image 1; Some((w,h)) = custom empty latent
    pub custom: Option<(u32, u32)>,
    pub unet: String,
    pub clip: String,
    pub vae: String,
    pub sampler: String,
    pub scheduler: String,
}

fn upload(c: &Client, png: Vec<u8>, name: &str) -> Result<String, String> {
    let part = multipart::Part::bytes(png).file_name(name.to_string()).mime_str("image/png").map_err(|e| e.to_string())?;
    let form = multipart::Form::new().part("image", part).text("type", "temp").text("overwrite", "true");
    let r = c.post(format!("{}/upload/image", base_url())).multipart(form).send().map_err(|e| format!("upload: {e}"))?;
    let status = r.status();
    if !status.is_success() {
        let err_body = r.text().unwrap_or_default();
        return Err(format!("upload error ({status}): {err_body}"));
    }
    let v: Value = r.json().map_err(|e| format!("upload reply: {e}"))?;
    let n = v["name"].as_str().ok_or("upload: no name in reply")?;
    Ok(format!("{n} [temp]"))
}

fn workflow(p: &Params, names: &[String]) -> Value {
    let mut te = json!({
        "clip": ["2", 0], "vae": ["3", 0],
        "prompt": p.prompt, "negative_prompt": p.negative,
        "resolution": p.resolution
    });
    let mut g = json!({
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": p.unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": p.clip, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": p.vae}},
        "4": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
    });
    for (i, n) in names.iter().enumerate() {
        let id = format!("{}", 100 + i);
        g[&id] = json!({"class_type": "LoadImage", "inputs": {"image": n}});
        te[format!("images.image_{}", i + 1)] = json!([id, 0]);
    }
    let (w, h) = p.custom.unwrap_or((1024, 1024));
    g["5"] = json!({"class_type": "TextEncodeQwenImage21", "inputs": te});
    g["6"] = json!({"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}});
    g["7"] = json!({"class_type": "ComfySwitchNode", "inputs": {"switch": p.custom.is_some() || names.is_empty() || names.is_empty(), "on_false": ["5", 2], "on_true": ["6", 0]}});
    g["8"] = json!({"class_type": "KSampler", "inputs": {
        "model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["7", 0],
        "seed": p.seed, "steps": p.steps, "cfg": p.cfg,
        "sampler_name": p.sampler, "scheduler": p.scheduler, "denoise": 1.0}});
    g["9"] = json!({"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}});
    g["11"] = json!({"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "ImageStudio/edit"}});
    g
}

fn workflow_minimax_h3(
    p: &VideoParams,
    first_frame_name: Option<&str>,
    last_frame_name: Option<&str>,
    ref_image_names: &[String],
) -> Value {
    let w = ((p.width.max(256) + 15) / 32) * 32;
    let h = ((p.height.max(256) + 15) / 32) * 32;
    // MiniMax H3 frame grid is 17k + 5 at 24fps (e.g. 5s -> 124 frames, 3s -> 73 frames)
    let raw_frames = (p.duration * 24).max(5);
    let rem = (5 - (raw_frames % 17) as i32).rem_euclid(17) as u32;
    let frames = raw_frames + rem;

    let use_ref_graph = p.use_r2v && !ref_image_names.is_empty();
    let active_unet = if use_ref_graph && !p.ref2va_unet_name.is_empty() {
        p.ref2va_unet_name.as_str()
    } else {
        p.unet_name.as_str()
    };

    let unet_node = if active_unet.to_ascii_lowercase().ends_with(".gguf") {
        json!({ "class_type": "UnetLoaderGGUF", "inputs": { "unet_name": active_unet } })
    } else {
        json!({ "class_type": "UNETLoader", "inputs": { "unet_name": active_unet, "weight_dtype": "default" } })
    };

    let clip_node = if p.clip_name.to_ascii_lowercase().ends_with(".gguf") {
        json!({ "class_type": "CLIPLoaderGGUF", "inputs": { "clip_name": p.clip_name, "type": "minimax" } })
    } else {
        json!({ "class_type": "CLIPLoader", "inputs": { "clip_name": p.clip_name, "type": "minimax", "device": "default" } })
    };

    let mut g = json!({
        "100": unet_node,
        "101": clip_node,
        "102": { "class_type": "VAELoader", "inputs": { "vae_name": p.vae_name } },
        "103": { "class_type": "VAELoader", "inputs": { "vae_name": p.audio_vae_name } }
    });

    let custom_lora = p.lora_name.as_deref().filter(|s| !s.is_empty() && *s != "None");
    let model_out = if let Some(lora) = custom_lora {
        g["105"] = json!({
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["100", 0],
                "lora_name": lora,
                "strength_model": p.lora_strength
            }
        });
        json!(["105", 0])
    } else {
        json!(["100", 0])
    };

    if use_ref_graph {
        let mut r2v_inputs = json!({
            "clip": ["101", 0],
            "vae": ["102", 0],
            "audio_vae": ["103", 0],
            "prompt": p.prompt,
            "width": w,
            "height": h,
            "length": frames as i64,
            "ref_image_size": "match"
        });
        for (i, rname) in ref_image_names.iter().take(9).enumerate() {
            let nid = format!("{}", 200 + i);
            g[&nid] = json!({ "class_type": "LoadImage", "inputs": { "image": rname } });
            r2v_inputs[format!("ref_images.ref_image_{i}")] = json!([nid, 0]);
        }
        g["110"] = json!({ "class_type": "MiniMaxH3ReferenceToVideo", "inputs": r2v_inputs });
    } else {
        let mut i2v_inputs = json!({
            "clip": ["101", 0],
            "vae": ["102", 0],
            "prompt": p.prompt,
            "width": w,
            "height": h,
            "length": frames as i64
        });
        let mp = ((w as f64 * h as f64) / 1_000_000.0).max(0.25);
        if let Some(ff) = first_frame_name {
            g["200"] = json!({ "class_type": "LoadImage", "inputs": { "image": ff } });
            g["201"] = json!({
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {
                    "image": ["200", 0],
                    "upscale_method": "lanczos",
                    "megapixels": mp,
                    "resolution_steps": 32
                }
            });
            i2v_inputs["first_frame"] = json!(["201", 0]);
        }
        if let Some(lf) = last_frame_name {
            g["202"] = json!({ "class_type": "LoadImage", "inputs": { "image": lf } });
            g["203"] = json!({
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {
                    "image": ["202", 0],
                    "upscale_method": "lanczos",
                    "megapixels": mp,
                    "resolution_steps": 32
                }
            });
            i2v_inputs["last_frame"] = json!(["203", 0]);
        }
        g["110"] = json!({ "class_type": "MiniMaxH3ImageToVideo", "inputs": i2v_inputs });
    }

    let sampler_name = if p.sampler.is_empty() { "er_sde" } else { p.sampler.as_str() };
    let scheduler_name = if p.scheduler.is_empty() { "sgm_uniform" } else { p.scheduler.as_str() };

    g["111"] = json!({ "class_type": "RandomNoise", "inputs": { "noise_seed": p.seed } });
    g["112"] = json!({ "class_type": "BasicGuider", "inputs": { "model": model_out, "conditioning": ["110", 0] } });
    g["113"] = json!({ "class_type": "KSamplerSelect", "inputs": { "sampler_name": sampler_name } });
    g["114"] = json!({
        "class_type": "BasicScheduler",
        "inputs": {
            "model": model_out,
            "scheduler": scheduler_name,
            "steps": p.steps.max(1),
            "denoise": p.denoise.clamp(0.05, 1.0)
        }
    });
    g["115"] = json!({
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["111", 0],
            "guider": ["112", 0],
            "sampler": ["113", 0],
            "sigmas": ["114", 0],
            "latent_image": ["110", 1]
        }
    });
    g["116"] = json!({ "class_type": "VAEDecode", "inputs": { "samples": ["115", 0], "vae": ["102", 0] } });
    g["117"] = json!({ "class_type": "VAEDecodeAudio", "inputs": { "samples": ["115", 0], "vae": ["103", 0] } });
    g["118"] = json!({
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["116", 0],
            "audio": ["117", 0],
            "fps": 24,
            "bit_depth": 8
        }
    });
    g["75"] = json!({
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["118", 0],
            "filename_prefix": "ImageStudio/video",
            "format": "auto",
            "codec": "auto"
        }
    });

    g
}

fn workflow_ltx25(p: &VideoParams, image_name: Option<&str>) -> Value {
    let w = ((p.width.max(128) + 31) / 64) * 64;
    let h = ((p.height.max(128) + 31) / 64) * 64;
    let frames = ((p.duration * p.fps) / 8).max(1) * 8 + 1;

    let unet_node = if p.unet_name.to_ascii_lowercase().ends_with(".gguf") {
        json!({ "class_type": "UnetLoaderGGUF", "inputs": { "unet_name": p.unet_name } })
    } else {
        json!({ "class_type": "UNETLoader", "inputs": { "unet_name": p.unet_name, "weight_dtype": "default" } })
    };

    let clip_node = if p.clip_name.to_ascii_lowercase().ends_with(".gguf") {
        json!({ "class_type": "CLIPLoaderGGUF", "inputs": { "clip_name": p.clip_name, "type": "ltxv" } })
    } else {
        json!({ "class_type": "CLIPLoader", "inputs": { "clip_name": p.clip_name, "type": "ltxv", "device": "default" } })
    };

    let sampler_name = if p.sampler.is_empty() { "er_sde" } else { p.sampler.as_str() };

    let mut g = json!({
        "236": unet_node,
        "237": clip_node,
        "248": { "class_type": "VAELoader", "inputs": { "vae_name": p.vae_name } },
        "212": { "class_type": "VAELoader", "inputs": { "vae_name": p.audio_vae_name } },
        "233": { "class_type": "LatentUpscaleModelLoader", "inputs": { "model_name": p.latent_upscale_model } },

        "277": { "class_type": "RandomNoise", "inputs": { "noise_seed": p.seed } },
        "276": { "class_type": "RandomNoise", "inputs": { "noise_seed": 42 } },
        "291": { "class_type": "KSamplerSelect", "inputs": { "sampler_name": sampler_name } },
        "280": { "class_type": "KSamplerSelect", "inputs": { "sampler_name": sampler_name } },
        "306": { "class_type": "ManualSigmas", "inputs": { "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0" } },
        "281": { "class_type": "ManualSigmas", "inputs": { "sigmas": "0.85, 0.7250, 0.4219, 0.0" } },

        "240": { "class_type": "CLIPTextEncode", "inputs": { "clip": ["237", 0], "text": p.prompt } },
        "247": { "class_type": "CLIPTextEncode", "inputs": { "clip": ["237", 0], "text": p.negative } },
        "239": { "class_type": "LTXVConditioning", "inputs": { "positive": ["240", 0], "negative": ["247", 0], "frame_rate": p.fps as f32 } },

        "295": { "class_type": "EmptyLTXVLatentVideo", "inputs": { "width": w / 2, "height": h / 2, "length": frames as i64, "batch_size": 1 } },
        "305": { "class_type": "LTXVEmptyLatentAudio", "inputs": { "audio_vae": ["212", 0], "frames_number": frames as i64, "frame_rate": p.fps as i64, "batch_size": 1 } },

        "318": { "class_type": "LTXVConcatAVLatent", "inputs": { "video_latent": ["296", 0], "audio_latent": ["305", 0] } },
        "307": { "class_type": "LTXVSeparateAVLatent", "inputs": { "av_latent": ["215", 0] } },

        "287": { "class_type": "LTXVLatentUpsampler", "inputs": { "samples": ["307", 0], "upscale_model": ["233", 0], "vae": ["248", 0] } },
        "278": { "class_type": "LTXVConcatAVLatent", "inputs": { "video_latent": ["288", 0], "audio_latent": ["307", 1] } },
        "309": { "class_type": "LTXVSeparateAVLatent", "inputs": { "av_latent": ["246", 0] } },

        "315": { "class_type": "VAEDecodeTiled", "inputs": { "samples": ["309", 0], "vae": ["248", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 16 } },
        "297": { "class_type": "LTXVAudioVAEDecode", "inputs": { "samples": ["309", 1], "audio_vae": ["212", 0] } },
        "310": { "class_type": "CreateVideo", "inputs": { "images": ["315", 0], "audio": ["297", 0], "fps": p.fps as f32 } },
        "75": { "class_type": "SaveVideo", "inputs": { "video": ["310", 0], "filename_prefix": "ImageStudio/video", "format": "auto", "codec": "auto" } }
    });

    let custom_lora = p.lora_name.as_deref().filter(|s| !s.is_empty() && *s != "None");
    let model_out = if let Some(lora) = custom_lora {
        g["285"] = json!({
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["236", 0],
                "lora_name": lora,
                "strength_model": p.lora_strength
            }
        });
        json!(["285", 0])
    } else {
        json!(["236", 0])
    };

    g["241"] = json!({
        "class_type": "CFGGuider",
        "inputs": {
            "model": model_out,
            "positive": ["239", 0],
            "negative": ["239", 1],
            "cfg": p.cfg.max(0.5)
        }
    });
    g["215"] = json!({
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["277", 0],
            "guider": ["241", 0],
            "sampler": ["291", 0],
            "sigmas": ["306", 0],
            "latent_image": ["318", 0]
        }
    });
    g["246"] = json!({
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["276", 0],
            "guider": ["241", 0],
            "sampler": ["280", 0],
            "sigmas": ["281", 0],
            "latent_image": ["278", 0]
        }
    });

    if let Some(img_name) = image_name {
        g["269"] = json!({ "class_type": "LoadImage", "inputs": { "image": img_name } });
        g["290"] = json!({ "class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["269", 0], "resize_type": "scale dimensions", "resize_type.width": w, "resize_type.height": h, "resize_type.crop": "center", "scale_method": "lanczos"
        }});
        g["289"] = json!({ "class_type": "LTXVPreprocess", "inputs": { "image": ["290", 0], "img_compression": 18 } });
        g["296"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["248", 0], "image": ["289", 0], "latent": ["295", 0], "bypass": false, "strength": 0.7 } });
        g["288"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["248", 0], "image": ["289", 0], "latent": ["287", 0], "bypass": false, "strength": 1.0 } });
    } else {
        g["269"] = json!({ "class_type": "EmptyImage", "inputs": { "width": 512, "height": 512, "batch_size": 1, "color": 0 } });
        g["290"] = json!({ "class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["269", 0], "resize_type": "scale dimensions", "resize_type.width": w, "resize_type.height": h, "resize_type.crop": "center", "scale_method": "lanczos"
        }});
        g["289"] = json!({ "class_type": "LTXVPreprocess", "inputs": { "image": ["290", 0], "img_compression": 18 } });
        g["296"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["248", 0], "image": ["289", 0], "latent": ["295", 0], "bypass": true, "strength": 0.7 } });
        g["288"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["248", 0], "image": ["289", 0], "latent": ["287", 0], "bypass": true, "strength": 1.0 } });
    }

    g
}

fn workflow_video(p: &VideoParams, image_name: Option<&str>) -> Value {
    let w = ((p.width.max(128) + 31) / 64) * 64;
    let h = ((p.height.max(128) + 31) / 64) * 64;
    let sampler_name = if p.sampler.is_empty() { "euler" } else { p.sampler.as_str() };
    let mut g = json!({
        "316": { "class_type": "CheckpointLoaderSimple", "inputs": { "ckpt_name": p.ckpt_name } },
        "317": { "class_type": "LTXAVTextEncoderLoader", "inputs": { "text_encoder": p.text_encoder, "ckpt_name": p.ckpt_name, "device": "default" } },
        "279": { "class_type": "LTXVAudioVAELoader", "inputs": { "ckpt_name": p.ckpt_name } },
        "285": { "class_type": "LoraLoaderModelOnly", "inputs": { "model": ["316", 0], "lora_name": p.distilled_lora, "strength_model": 0.5 } },

        "277": { "class_type": "RandomNoise", "inputs": { "noise_seed": p.seed } },
        "276": { "class_type": "RandomNoise", "inputs": { "noise_seed": 42 } },
        "291": { "class_type": "KSamplerSelect", "inputs": { "sampler_name": sampler_name } },
        "280": { "class_type": "KSamplerSelect", "inputs": { "sampler_name": sampler_name } },
        "306": { "class_type": "ManualSigmas", "inputs": { "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0" } },
        "281": { "class_type": "ManualSigmas", "inputs": { "sigmas": "0.85, 0.7250, 0.4219, 0.0" } },

        "313": { "class_type": "CLIPTextEncode", "inputs": { "clip": ["317", 0], "text": p.negative } },
        "304": { "class_type": "LTXVConditioning", "inputs": { "positive": ["303", 0], "negative": ["313", 0], "frame_rate": p.fps as f32 } },
        "314": { "class_type": "CFGGuider", "inputs": { "model": ["285", 0], "positive": ["304", 0], "negative": ["304", 1], "cfg": 1.0 } },

        "295": { "class_type": "EmptyLTXVLatentVideo", "inputs": { "width": w / 2, "height": h / 2, "length": (p.duration * p.fps + 1) as i64, "batch_size": 1 } },
        "305": { "class_type": "LTXVEmptyLatentAudio", "inputs": { "audio_vae": ["279", 0], "frames_number": (p.duration * p.fps + 1) as i64, "frame_rate": p.fps as i64, "batch_size": 1 } },

        "318": { "class_type": "LTXVConcatAVLatent", "inputs": { "video_latent": ["296", 0], "audio_latent": ["305", 0] } },
        "283": { "class_type": "SamplerCustomAdvanced", "inputs": { "noise": ["277", 0], "guider": ["314", 0], "sampler": ["291", 0], "sigmas": ["306", 0], "latent_image": ["318", 0] } },
        "307": { "class_type": "LTXVSeparateAVLatent", "inputs": { "av_latent": ["283", 0] } },

        "311": { "class_type": "LatentUpscaleModelLoader", "inputs": { "model_name": p.latent_upscale_model } },
        "287": { "class_type": "LTXVLatentUpsampler", "inputs": { "samples": ["307", 0], "upscale_model": ["311", 0], "vae": ["316", 2] } },
        "284": { "class_type": "LTXVCropGuides", "inputs": { "positive": ["304", 0], "negative": ["304", 1], "latent": ["307", 0] } },
        "282": { "class_type": "CFGGuider", "inputs": { "model": ["285", 0], "positive": ["284", 0], "negative": ["284", 1], "cfg": 1.0 } },
        "278": { "class_type": "LTXVConcatAVLatent", "inputs": { "video_latent": ["288", 0], "audio_latent": ["307", 1] } },
        "308": { "class_type": "SamplerCustomAdvanced", "inputs": { "noise": ["276", 0], "guider": ["282", 0], "sampler": ["280", 0], "sigmas": ["281", 0], "latent_image": ["278", 0] } },
        "309": { "class_type": "LTXVSeparateAVLatent", "inputs": { "av_latent": ["308", 0] } },

        "315": { "class_type": "VAEDecodeTiled", "inputs": { "samples": ["309", 0], "vae": ["316", 2], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8 } },
        "297": { "class_type": "LTXVAudioVAEDecode", "inputs": { "samples": ["309", 1], "audio_vae": ["279", 0] } },
        "310": { "class_type": "CreateVideo", "inputs": { "images": ["315", 0], "audio": ["297", 0], "fps": p.fps as f32 } },
        "75": { "class_type": "SaveVideo", "inputs": { "video": ["310", 0], "filename_prefix": "ImageStudio/video", "format": "auto", "codec": "auto" } }
    });

    let custom_lora = p.lora_name.as_deref().filter(|s| !s.is_empty() && *s != "None");
    let model_out = if let Some(lora) = custom_lora {
        g["286"] = json!({
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["285", 0],
                "lora_name": lora,
                "strength_model": p.lora_strength
            }
        });
        g["314"]["inputs"]["model"] = json!(["286", 0]);
        g["282"]["inputs"]["model"] = json!(["286", 0]);
        json!(["286", 0])
    } else {
        json!(["285", 0])
    };

    if let Some(img_name) = image_name {
        g["269"] = json!({ "class_type": "LoadImage", "inputs": { "image": img_name } });
        g["290"] = json!({ "class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["269", 0], "resize_type": "scale dimensions", "resize_type.width": w, "resize_type.height": h, "resize_type.crop": "center", "scale_method": "lanczos"
        }});
        g["330"] = json!({ "class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["290", 0], "resize_type": "scale longer dimension", "resize_type.longer_size": 512, "scale_method": "lanczos"
        }});
        g["289"] = json!({ "class_type": "LTXVPreprocess", "inputs": { "image": ["330", 0], "img_compression": 18 } });

        g["296"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["316", 2], "image": ["289", 0], "latent": ["295", 0], "bypass": false, "strength": 0.7 } });
        g["288"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["316", 2], "image": ["289", 0], "latent": ["287", 0], "bypass": false, "strength": 1.0 } });

        if p.prompt_enhance {
            g["325"] = json!({ "class_type": "TextGenerateLTX2Prompt", "inputs": {
                "clip": ["324", 1], "image": ["330", 0], "prompt": p.prompt, "max_length": 384, "sampling_mode": "on",
                "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64, "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
                "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": (p.seed % 1000000) as i64, "sampling_mode.presence_penalty": 0, "thinking": false, "use_default_template": false
            }});
        }
    } else {
        g["269"] = json!({ "class_type": "EmptyImage", "inputs": { "width": 512, "height": 512, "batch_size": 1, "color": 0 } });
        g["290"] = json!({ "class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["269", 0], "resize_type": "scale dimensions", "resize_type.width": w, "resize_type.height": h, "resize_type.crop": "center", "scale_method": "lanczos"
        }});
        g["330"] = json!({ "class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["290", 0], "resize_type": "scale longer dimension", "resize_type.longer_size": 512, "scale_method": "lanczos"
        }});
        g["289"] = json!({ "class_type": "LTXVPreprocess", "inputs": { "image": ["330", 0], "img_compression": 18 } });

        g["296"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["316", 2], "image": ["289", 0], "latent": ["295", 0], "bypass": true, "strength": 0.7 } });
        g["288"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["316", 2], "image": ["289", 0], "latent": ["287", 0], "bypass": true, "strength": 1.0 } });

        if p.prompt_enhance {
            g["325"] = json!({ "class_type": "TextGenerateLTX2Prompt", "inputs": {
                "clip": ["324", 1], "prompt": p.prompt, "max_length": 384, "sampling_mode": "on",
                "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64, "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
                "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": (p.seed % 1000000) as i64, "sampling_mode.presence_penalty": 0, "thinking": false, "use_default_template": false
            }});
        }
    }

    if p.prompt_enhance {
        g["324"] = json!({ "class_type": "LoraLoader", "inputs": { "model": model_out, "clip": ["317", 0], "lora_name": p.gemma_lora, "strength_model": 1.0, "strength_clip": 1.0 } });
        g["303"] = json!({ "class_type": "CLIPTextEncode", "inputs": { "clip": ["317", 0], "text": ["325", 0] } });
    } else {
        g["303"] = json!({ "class_type": "CLIPTextEncode", "inputs": { "clip": ["317", 0], "text": p.prompt } });
    }

    g
}


fn explain(v: &Value) -> String {
    let mut out = vec![];
    if let Some(m) = v["error"]["message"].as_str() {
        out.push(m.to_string());
    }
    if let Some(d) = v["error"]["details"].as_str().filter(|s| !s.is_empty()) {
        out.push(d.to_string());
    }
    if let Some(ne) = v["node_errors"].as_object() {
        for (id, n) in ne {
            if let Some(errors) = n["errors"].as_array() {
                for e in errors {
                    out.push(format!(
                        "node {id} ({}): {} {}",
                        n["class_type"].as_str().unwrap_or("?"),
                        e["message"].as_str().unwrap_or(""),
                        e["details"].as_str().unwrap_or("")
                    ));
                }
            }
        }
    }
    if out.is_empty() {
        v.to_string()
    } else {
        out.join("\n")
    }
}

#[derive(Clone, Copy, Debug)]
enum WorkflowProgressSpec {
    SingleStage {
        sampler_node: &'static str,
    },
    TwoStage {
        stage1_node: &'static str,
        stage2_node: &'static str,
        stage1_steps: u32,
        stage2_steps: u32,
    },
}

impl WorkflowProgressSpec {
    fn progress_for_node_step(self, node: Option<&str>, cur: u32, max: u32) -> Option<f32> {
        let step_frac = cur as f32 / max.max(1) as f32;
        match self {
            Self::SingleStage { sampler_node } => {
                if node == Some(sampler_node) || node.is_none() {
                    Some((0.05 + 0.90 * step_frac).clamp(0.05, 0.95))
                } else if node == Some("315") {
                    Some((0.94 + 0.05 * step_frac).clamp(0.94, 0.99))
                } else {
                    None
                }
            }
            Self::TwoStage { stage1_node, stage2_node, stage1_steps, stage2_steps } => {
                let total_steps = (stage1_steps + stage2_steps).max(1) as f32;
                if node == Some(stage1_node) {
                    let completed = step_frac * stage1_steps as f32;
                    Some((0.05 + 0.89 * (completed / total_steps)).clamp(0.05, 0.94))
                } else if node == Some(stage2_node) {
                    let completed = stage1_steps as f32 + step_frac * stage2_steps as f32;
                    Some((0.05 + 0.89 * (completed / total_steps)).clamp(0.05, 0.94))
                } else if node == Some("315") {
                    Some((0.94 + 0.05 * step_frac).clamp(0.94, 0.99))
                } else {
                    None
                }
            }
        }
    }

    fn progress_for_executing_node(self, node: &str) -> Option<f32> {
        match node {
            "9" | "116" | "117" | "309" | "315" | "297" => Some(0.95),
            "11" | "118" | "310" | "75" => Some(0.98),
            "307" | "287" | "278" | "284" | "282" => match self {
                Self::TwoStage { stage1_steps, stage2_steps, .. } => {
                    let total_steps = (stage1_steps + stage2_steps).max(1) as f32;
                    Some((0.05 + 0.89 * (stage1_steps as f32 / total_steps)).clamp(0.05, 0.94))
                }
                Self::SingleStage { .. } => None,
            },
            _ => Some(0.03),
        }
    }
}

#[derive(Default)]
struct WsTracker {
    last_activity: Mutex<Option<Instant>>,
    executing_node: Mutex<Option<String>>,
    node_has_progress: AtomicBool,
    execution_error: Mutex<Option<String>>,
    execution_interrupted: AtomicBool,
}

fn spawn_ws_progress(
    stop: Arc<AtomicBool>,
    tracker: Arc<WsTracker>,
    spec: WorkflowProgressSpec,
    on_progress: Arc<dyn Fn(u32, u32) + Send + Sync>,
    on_total_progress: Arc<dyn Fn(f32) + Send + Sync>,
    on_preview: Arc<dyn Fn(Vec<u8>) + Send + Sync>,
) {
    std::thread::spawn(move || {
        let mut best_total: f32 = 0.0;
        let mut emit_total = |frac: f32| {
            let clamped = frac.clamp(0.0, 1.0);
            if clamped > best_total {
                best_total = clamped;
                on_total_progress(best_total);
            }
        };
        while !stop.load(Ordering::SeqCst) {
            let stream = match std::net::TcpStream::connect(format!("127.0.0.1:{}", crate::engine::PORT)) {
                Ok(s) => s,
                Err(_) => {
                    std::thread::sleep(Duration::from_millis(250));
                    continue;
                }
            };
            let _ = stream.set_read_timeout(Some(Duration::from_millis(350)));
            use std::io::{BufRead, Read, Write};
            let mut reader = std::io::BufReader::new(stream);
            let req = format!(
                "GET /ws?clientId=image-studio HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n",
                crate::engine::PORT
            );
            if reader.get_mut().write_all(req.as_bytes()).is_err() {
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }
            let mut line = String::new();
            let mut upgraded = false;
            while let Ok(n) = reader.read_line(&mut line) {
                if n == 0 || line == "\r\n" || line == "\n" {
                    upgraded = true;
                    break;
                }
                line.clear();
            }
            if !upgraded {
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }
            while !stop.load(Ordering::SeqCst) {
                let mut header = [0u8; 2];
                match reader.read_exact(&mut header) {
                    Ok(()) => {}
                    Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut || e.kind() == std::io::ErrorKind::WouldBlock => {
                        continue;
                    }
                    Err(_) => break,
                }
                let opcode = header[0] & 0x0F;
                if opcode == 0x8 {
                    break;
                }
                let mut len = (header[1] & 0x7F) as usize;
                if len == 126 {
                    let mut ext = [0u8; 2];
                    if reader.read_exact(&mut ext).is_err() { break; }
                    len = u16::from_be_bytes(ext) as usize;
                } else if len == 127 {
                    let mut ext = [0u8; 8];
                    if reader.read_exact(&mut ext).is_err() { break; }
                    len = u64::from_be_bytes(ext) as usize;
                }
                let mut payload = vec![0u8; len];
                let mut read_bytes = 0;
                let mut read_err = false;
                while read_bytes < len {
                    if stop.load(Ordering::SeqCst) { break; }
                    match reader.read(&mut payload[read_bytes..]) {
                        Ok(0) => { read_err = true; break; }
                        Ok(n) => read_bytes += n,
                        Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut || e.kind() == std::io::ErrorKind::WouldBlock => continue,
                        Err(_) => { read_err = true; break; }
                    }
                }
                if read_err || read_bytes < len {
                    break;
                }
                if opcode == 0x9 {
                    continue;
                }
                if opcode == 0x1 {
                    if let Ok(text) = std::str::from_utf8(&payload) {
                        if let Ok(val) = serde_json::from_str::<serde_json::Value>(text) {
                            match val["type"].as_str() {
                                Some("progress") => {
                                    *tracker.last_activity.lock().unwrap() = Some(Instant::now());
                                    tracker.node_has_progress.store(true, Ordering::SeqCst);
                                    if let (Some(cur), Some(max)) = (val["data"]["value"].as_u64(), val["data"]["max"].as_u64()) {
                                        on_progress(cur as u32, max as u32);
                                        let node_id = val["data"]["node"]
                                            .as_str()
                                            .map(|s| s.to_string())
                                            .or_else(|| tracker.executing_node.lock().unwrap().clone());
                                        if let Some(frac) = spec.progress_for_node_step(node_id.as_deref(), cur as u32, max as u32) {
                                            emit_total(frac);
                                        }
                                    }
                                }
                                Some("progress_state") => {
                                    *tracker.last_activity.lock().unwrap() = Some(Instant::now());
                                }
                                Some("executing") => {
                                    *tracker.last_activity.lock().unwrap() = Some(Instant::now());
                                    let node = val["data"]["node"].as_str().map(|s| s.to_string());
                                    if let Some(ref n) = node {
                                        if let Some(frac) = spec.progress_for_executing_node(n) {
                                            emit_total(frac);
                                        }
                                    }
                                    let mut exec = tracker.executing_node.lock().unwrap();
                                    if *exec != node {
                                        tracker.node_has_progress.store(false, Ordering::SeqCst);
                                        *exec = node;
                                    }
                                }
                                Some("execution_error") => {
                                    *tracker.last_activity.lock().unwrap() = Some(Instant::now());
                                    let err = format!(
                                        "{}: {}",
                                        val["data"]["node_type"].as_str().unwrap_or("Engine"),
                                        val["data"]["exception_message"].as_str().unwrap_or("Execution error")
                                    );
                                    *tracker.execution_error.lock().unwrap() = Some(err);
                                }
                                Some("execution_interrupted") => {
                                    tracker.execution_interrupted.store(true, Ordering::SeqCst);
                                }
                                Some("executed") => {
                                    *tracker.last_activity.lock().unwrap() = Some(Instant::now());
                                }
                                _ => {}
                            }
                        }
                    }
                } else if opcode == 0x2 && payload.len() > 8 {
                    *tracker.last_activity.lock().unwrap() = Some(Instant::now());
                    tracker.node_has_progress.store(true, Ordering::SeqCst);
                    let event = u32::from_be_bytes(payload[0..4].try_into().unwrap_or_default());
                    let img_bytes = if event == 1 {
                        Some(&payload[8..])
                    } else if event == 4 {
                        let meta_len = u32::from_be_bytes(payload[4..8].try_into().unwrap_or_default()) as usize;
                        if payload.len() > 8 + meta_len {
                            Some(&payload[8 + meta_len..])
                        } else {
                            None
                        }
                    } else {
                        Some(&payload[8..])
                    };
                    if let Some(bytes) = img_bytes {
                        on_preview(bytes.to_vec());
                    }
                }
            }
        }
    });
}

pub fn run(
    p: &Params,
    imgs: Vec<Vec<u8>>,
    cancel: &AtomicBool,
    status: &dyn Fn(String),
    progress: Arc<dyn Fn(u32, u32) + Send + Sync>,
    total_progress: Arc<dyn Fn(f32) + Send + Sync>,
    preview: Arc<dyn Fn(Vec<u8>) + Send + Sync>,
) -> Result<GenResult, String> {
    let ws_stop = Arc::new(AtomicBool::new(false));
    let tracker = Arc::new(WsTracker::default());
    spawn_ws_progress(
        ws_stop.clone(),
        tracker.clone(),
        WorkflowProgressSpec::SingleStage { sampler_node: "8" },
        progress,
        total_progress,
        preview,
    );

    let c = Client::builder().timeout(Duration::from_secs(120)).build().map_err(|e| e.to_string())?;
    let tag = p.seed % 100000;
    let mut names = vec![];
    let mut temp_files = vec![];
    for (i, png) in imgs.into_iter().enumerate() {
        status(format!("Uploading image {}...", i + 1));
        let filename = format!("studio_{tag}_{i}.png");
        names.push(upload(&c, png, &filename)?);
        temp_files.push(filename);
    }
    prepare_clean_slate(&c, status);
    status("Queueing...".into());
    let body = json!({"prompt": workflow(p, &names), "client_id": "image-studio"});
    let r = c.post(format!("{}/prompt", base_url())).json(&body).send().map_err(|e| format!("queue: {e}"))?;
    let ok = r.status().is_success();
    let v: Value = r.json().map_err(|e| format!("queue reply: {e}"))?;
    if !ok {
        ws_stop.store(true, Ordering::SeqCst);
        return Err(explain(&v));
    }
    let id = v["prompt_id"].as_str().ok_or("no prompt_id")?.to_string();
    let t0 = Instant::now();
    let mut consecutive_errs = 0;
    let mut missing_queue_ticks = 0;
    loop {
        if cancel.load(Ordering::SeqCst) || tracker.execution_interrupted.load(Ordering::SeqCst) {
            let _ = c.post(format!("{}/interrupt", base_url())).send();
            let _ = c.post(format!("{}/queue", base_url())).json(&serde_json::json!({"clear": true})).send();
            let _ = c.post(format!("{}/free", base_url())).json(&serde_json::json!({"unload_models": true, "free_memory": true})).send();
            ws_stop.store(true, Ordering::SeqCst);
            return Err("Cancelled".into());
        }

        if let Some(err) = tracker.execution_error.lock().unwrap().take() {
            let _ = c.post(format!("{}/interrupt", base_url())).send();
            let _ = c.post(format!("{}/queue", base_url())).json(&serde_json::json!({"clear": true})).send();
            let _ = c.post(format!("{}/free", base_url())).json(&serde_json::json!({"unload_models": true, "free_memory": true})).send();
            ws_stop.store(true, Ordering::SeqCst);
            return Err(err);
        }

        std::thread::sleep(Duration::from_millis(500));
        let secs = t0.elapsed().as_secs();

        // Stall watchdog: if no activity received for >120s during image generation, abort cleanly
        let last_act = tracker.last_activity.lock().unwrap().unwrap_or(t0);
        let inactive_secs = last_act.elapsed().as_secs();
        if inactive_secs > 120 {
            let _ = c.post(format!("{}/interrupt", base_url())).send();
            let _ = c.post(format!("{}/queue", base_url())).json(&serde_json::json!({"clear": true})).send();
            let _ = c.post(format!("{}/free", base_url())).json(&serde_json::json!({"unload_models": true, "free_memory": true})).send();
            ws_stop.store(true, Ordering::SeqCst);
            return Err(format!("Engine stalled (no progress for {inactive_secs}s). Queue cleared."));
        }

        let h_res = c.get(format!("{}/history/{id}", base_url())).send();
        let h: Value = match h_res {
            Ok(r) => {
                consecutive_errs = 0;
                r.json().unwrap_or(Value::Null)
            }
            Err(e) => {
                consecutive_errs += 1;
                if consecutive_errs > 10 {
                    ws_stop.store(true, Ordering::SeqCst);
                    return Err(format!("poll: {e}"));
                }
                continue;
            }
        };
        if let Some(e) = h.get(&id) {
            let st = &e["status"];
            if st["status_str"] == "error" {
                ws_stop.store(true, Ordering::SeqCst);
                if let Some(msgs) = st["messages"].as_array() {
                    for m in msgs {
                        if m[0] == "execution_error" {
                            return Err(format!("{}: {}", m[1]["node_type"].as_str().unwrap_or(""), m[1]["exception_message"].as_str().unwrap_or("error")));
                        }
                        if m[0] == "execution_interrupted" {
                            return Err("Cancelled".into());
                        }
                    }
                }
                return Err("Generation failed".into());
            }
            if let Some(img) = e["outputs"]["11"]["images"][0].as_object() {
                let out_fn = img["filename"].as_str().unwrap_or("").to_string();
                let out_sub = img["subfolder"].as_str().unwrap_or("").to_string();
                let url = format!("{}/view", base_url());
                let bytes = c
                    .get(url)
                    .query(&[
                        ("filename", out_fn.as_str()),
                        ("subfolder", out_sub.as_str()),
                        ("type", "output"),
                    ])
                    .send()
                    .and_then(|r| r.bytes())
                    .map_err(|e| format!("fetch result: {e}"))?;
                ws_stop.store(true, Ordering::SeqCst);
                return Ok(GenResult {
                    png: bytes.to_vec(),
                    output_filename: out_fn,
                    output_subfolder: out_sub,
                    temp_files,
                });
            }
            if st["completed"] == true {
                ws_stop.store(true, Ordering::SeqCst);
                return Err("Finished but produced no image".into());
            }
        }

        let q: Value = c.get(format!("{}/queue", base_url())).send().ok().and_then(|r| r.json().ok()).unwrap_or(Value::Null);
        let running = q["queue_running"].as_array().map(|a| a.iter().any(|x| x[1] == id.as_str())).unwrap_or(false);
        let pending = q["queue_pending"].as_array().map(|a| a.iter().any(|x| x[1] == id.as_str())).unwrap_or(false);

        if !running && !pending {
            missing_queue_ticks += 1;
            if missing_queue_ticks >= 4 {
                ws_stop.store(true, Ordering::SeqCst);
                return Err("Generation stopped: job left queue without producing image output".into());
            }
        } else {
            missing_queue_ticks = 0;
        }

        status(if running {
            format!("Generating... {secs}s")
        } else {
            format!("Waiting in queue... {secs}s")
        });

        if secs > 1800 {
            ws_stop.store(true, Ordering::SeqCst);
            return Err("Timed out".into());
        }
    }
}

pub fn run_video(
    p: &VideoParams,
    img: Option<Vec<u8>>,
    cancel: &AtomicBool,
    status: &dyn Fn(String),
    progress: Arc<dyn Fn(u32, u32) + Send + Sync>,
    total_progress: Arc<dyn Fn(f32) + Send + Sync>,
    preview: Arc<dyn Fn(Vec<u8>) + Send + Sync>,
) -> Result<VideoGenResult, String> {
    let ws_stop = Arc::new(AtomicBool::new(false));
    let tracker = Arc::new(WsTracker::default());
    let spec = match p.arch {
        VideoArch::MiniMaxH3 => WorkflowProgressSpec::SingleStage { sampler_node: "115" },
        VideoArch::Ltx25 => WorkflowProgressSpec::TwoStage {
            stage1_node: "215",
            stage2_node: "246",
            stage1_steps: 8,
            stage2_steps: 3,
        },
        VideoArch::Ltx23 => WorkflowProgressSpec::TwoStage {
            stage1_node: "283",
            stage2_node: "308",
            stage1_steps: 8,
            stage2_steps: 3,
        },
    };
    spawn_ws_progress(ws_stop.clone(), tracker.clone(), spec, progress, total_progress, preview);

    let c = Client::builder().timeout(Duration::from_secs(120)).build().map_err(|e| e.to_string())?;
    let tag = p.seed % 100000;
    let mut image_name = None;
    let mut last_frame_name = None;
    let mut ref_names = Vec::new();
    let mut temp_files = vec![];
    if let Some(png) = img {
        status("Uploading initial frame...".into());
        let filename = format!("studio_vid_{tag}.png");
        let uploaded = upload(&c, png, &filename)?;
        image_name = Some(uploaded);
        temp_files.push(filename);
    }
    if !p.use_r2v {
        if let Some(ref lf) = p.last_frame {
            status("Uploading end frame...".into());
            let filename = format!("studio_vid_last_{tag}.png");
            let uploaded = upload(&c, lf.clone(), &filename)?;
            last_frame_name = Some(uploaded);
            temp_files.push(filename);
        }
    } else {
        for (idx, rpng) in p.ref_images.iter().take(9).enumerate() {
            status(format!("Uploading reference image {}...", idx + 1));
            let filename = format!("studio_vid_ref_{tag}_{idx}.png");
            let uploaded = upload(&c, rpng.clone(), &filename)?;
            ref_names.push(uploaded);
            temp_files.push(filename);
        }
    }

    prepare_clean_slate(&c, status);

    status("Queueing video generation...".into());
    let prompt_graph = match p.arch {
        VideoArch::MiniMaxH3 => workflow_minimax_h3(
            p,
            image_name.as_deref(),
            last_frame_name.as_deref(),
            &ref_names,
        ),
        VideoArch::Ltx25 => workflow_ltx25(p, image_name.as_deref()),
        VideoArch::Ltx23 => workflow_video(p, image_name.as_deref()),
    };
    let body = json!({ "prompt": prompt_graph, "client_id": "image-studio" });
    let r = c.post(format!("{}/prompt", base_url())).json(&body).send().map_err(|e| format!("queue: {e}"))?;
    let ok = r.status().is_success();
    let v: Value = r.json().map_err(|e| format!("queue reply: {e}"))?;
    if !ok {
        ws_stop.store(true, Ordering::SeqCst);
        return Err(explain(&v));
    }
    let id = v["prompt_id"].as_str().ok_or("no prompt_id")?.to_string();
    let t0 = Instant::now();
    let mut consecutive_errs = 0;
    let mut missing_queue_ticks = 0;
    loop {
        if cancel.load(Ordering::SeqCst) || tracker.execution_interrupted.load(Ordering::SeqCst) {
            let _ = c.post(format!("{}/interrupt", base_url())).send();
            let _ = c.post(format!("{}/queue", base_url())).json(&serde_json::json!({"clear": true})).send();
            let _ = c.post(format!("{}/free", base_url())).json(&serde_json::json!({"unload_models": true, "free_memory": true})).send();
            ws_stop.store(true, Ordering::SeqCst);
            return Err("Cancelled".into());
        }

        if let Some(err) = tracker.execution_error.lock().unwrap().take() {
            let _ = c.post(format!("{}/interrupt", base_url())).send();
            let _ = c.post(format!("{}/queue", base_url())).json(&serde_json::json!({"clear": true})).send();
            let _ = c.post(format!("{}/free", base_url())).json(&serde_json::json!({"unload_models": true, "free_memory": true})).send();
            ws_stop.store(true, Ordering::SeqCst);
            return Err(err);
        }

        std::thread::sleep(Duration::from_millis(500));
        let secs = t0.elapsed().as_secs();

        // Stall Watchdog: detect genuine hangs per stage and automatically abort & clear queue
        let active_node = tracker.executing_node.lock().unwrap().clone();
        let node_has_progress = tracker.node_has_progress.load(Ordering::SeqCst);
        let last_act = tracker.last_activity.lock().unwrap().unwrap_or(t0);
        let inactive_secs = last_act.elapsed().as_secs();
        let max_inactive = video_stage_timeout_secs(active_node.as_deref(), node_has_progress);
        if inactive_secs > max_inactive {
            let _ = c.post(format!("{}/interrupt", base_url())).send();
            let _ = c.post(format!("{}/queue", base_url())).json(&serde_json::json!({"clear": true})).send();
            let _ = c.post(format!("{}/free", base_url())).json(&serde_json::json!({"unload_models": true, "free_memory": true})).send();
            ws_stop.store(true, Ordering::SeqCst);
            let stage_name = video_stage_name(active_node.as_deref());
            return Err(format!("Engine stalled during {stage_name} (no activity for {inactive_secs}s). Queue cleared automatically."));
        }

        let h_res = c.get(format!("{}/history/{id}", base_url())).send();
        let h: Value = match h_res {
            Ok(r) => {
                consecutive_errs = 0;
                r.json().unwrap_or(Value::Null)
            }
            Err(e) => {
                consecutive_errs += 1;
                if consecutive_errs > 10 {
                    ws_stop.store(true, Ordering::SeqCst);
                    return Err(format!("poll: {e}"));
                }
                continue;
            }
        };
        if let Some(e) = h.get(&id) {
            let st = &e["status"];
            if st["status_str"] == "error" {
                ws_stop.store(true, Ordering::SeqCst);
                if let Some(msgs) = st["messages"].as_array() {
                    for m in msgs {
                        if m[0] == "execution_error" {
                            return Err(format!("{}: {}", m[1]["node_type"].as_str().unwrap_or(""), m[1]["exception_message"].as_str().unwrap_or("error")));
                        }
                        if m[0] == "execution_interrupted" {
                            return Err("Cancelled".into());
                        }
                    }
                }
                return Err("Generation failed".into());
            }

            // Look for SaveVideo output in node 75 (videos or gifs or images)
            let out_node = e["outputs"].get("75");
            let vid_obj = out_node.and_then(|o| {
                o["videos"].as_array().and_then(|a| a.first()).and_then(|v| v.as_object())
                    .or_else(|| o["gifs"].as_array().and_then(|a| a.first()).and_then(|v| v.as_object()))
                    .or_else(|| o["images"].as_array().and_then(|a| a.first()).and_then(|v| v.as_object()))
            });

            if let Some(vid) = vid_obj {
                let out_fn = vid["filename"].as_str().unwrap_or("").to_string();
                let out_sub = vid["subfolder"].as_str().unwrap_or("").to_string();
                let url = format!("{}/view", base_url());
                let bytes = c
                    .get(url)
                    .query(&[
                        ("filename", out_fn.as_str()),
                        ("subfolder", out_sub.as_str()),
                        ("type", "output"),
                    ])
                    .send()
                    .and_then(|r| r.bytes())
                    .map_err(|e| format!("fetch video result: {e}"))?;
                ws_stop.store(true, Ordering::SeqCst);
                return Ok(VideoGenResult {
                    mp4: bytes.to_vec(),
                    output_filename: out_fn,
                    output_subfolder: out_sub,
                    temp_files,
                });
            }

            if st["completed"] == true {
                ws_stop.store(true, Ordering::SeqCst);
                return Err("Finished but produced no video output".into());
            }
        }

        let q: Value = c.get(format!("{}/queue", base_url())).send().ok().and_then(|r| r.json().ok()).unwrap_or(Value::Null);
        let running = q["queue_running"].as_array().map(|a| a.iter().any(|x| x[1] == id.as_str())).unwrap_or(false);
        let pending = q["queue_pending"].as_array().map(|a| a.iter().any(|x| x[1] == id.as_str())).unwrap_or(false);

        if !running && !pending {
            missing_queue_ticks += 1;
            // If the prompt is gone from the queue for 2s without producing output, it failed/aborted
            if missing_queue_ticks >= 4 {
                ws_stop.store(true, Ordering::SeqCst);
                return Err("Generation stopped: engine completed job without producing video output".into());
            }
        } else {
            missing_queue_ticks = 0;
        }

        // Informative live stage reporting
        let status_text = match active_node.as_deref() {
            Some("325") => format!("Expanding prompt with Gemma 3 12B... {secs}s"),
            Some("316") | Some("317") | Some("324") | Some("285") | Some("286") | Some("279") | Some("311")
            | Some("100") | Some("101") | Some("102") | Some("103") | Some("105")
            | Some("236") | Some("237") | Some("248") | Some("212") | Some("233") => {
                format!("Staging model weights into VRAM... {secs}s")
            }
            Some("303") | Some("313") | Some("240") | Some("247") => {
                format!("Encoding prompt conditionings... {secs}s")
            }
            Some("110") => format!("Encoding multimodal video/audio prompt... {secs}s"),
            Some("304") | Some("314") | Some("284") | Some("282") | Some("239") | Some("241") => {
                format!("Configuring conditioning & guiders... {secs}s")
            }
            Some("295") | Some("305") | Some("318") | Some("278") | Some("277") | Some("276")
            | Some("291") | Some("280") | Some("306") | Some("281")
            | Some("111") | Some("112") | Some("113") | Some("114") => {
                format!("Initializing latents & noise schedules... {secs}s")
            }
            Some("269") | Some("290") | Some("330") | Some("289") | Some("296") | Some("288")
            | Some("200") | Some("201") | Some("202") | Some("203") | Some("204") | Some("205") => {
                format!("Processing input frames & references... {secs}s")
            }
            Some("115") => format!("Generating MiniMax H3 audio+video diffusion... {secs}s"),
            Some("283") | Some("215") => format!("Generating video (Stage 1 diffusion)... {secs}s"),
            Some("307") => format!("Splitting AV latents... {secs}s"),
            Some("287") => format!("Upscaling video latents (2x spatial)... {secs}s"),
            Some("308") | Some("246") => format!("Refining video details (Stage 2 diffusion)... {secs}s"),
            Some("309") => format!("Extracting refined video & audio latents... {secs}s"),
            Some("315") | Some("116") => format!("Decoding video frames (VAE)... {secs}s"),
            Some("297") | Some("117") => format!("Decoding audio track... {secs}s"),
            Some("310") | Some("118") | Some("75") => format!("Assembling final MP4 video... {secs}s"),
            _ => {
                if pending {
                    format!("Waiting in queue... {secs}s")
                } else if running {
                    format!("Generating video... {secs}s")
                } else {
                    format!("Finalizing output... {secs}s")
                }
            }
        };
        status(status_text);

        if secs > 1800 {
            ws_stop.store(true, Ordering::SeqCst);
            return Err("Timed out (video generation took > 30m)".into());
        }
    }
}

fn prepare_clean_slate(c: &Client, status: &dyn Fn(String)) {
    let base = base_url();
    // 1. Clear residual queue
    let _ = c.post(format!("{base}/queue")).json(&serde_json::json!({"clear": true})).send();

    // 2. Free unused models & memory cache from previous runs
    let _ = c.post(format!("{base}/free")).json(&serde_json::json!({"unload_models": true, "free_memory": true})).send();

    // 3. Wait up to 3 seconds if an old job is currently finishing its abort/unwind
    let t0 = Instant::now();
    let mut notified = false;
    while t0.elapsed() < Duration::from_millis(3000) {
        let q: Value = c.get(format!("{base}/queue")).send().ok().and_then(|r| r.json().ok()).unwrap_or(Value::Null);
        let running_count = q["queue_running"].as_array().map(|a| a.len()).unwrap_or(0);
        if running_count == 0 {
            break;
        }
        if !notified {
            status("Waiting for engine to settle...".into());
            notified = true;
        }
        let _ = c.post(format!("{base}/interrupt")).send();
        let _ = c.post(format!("{base}/queue")).json(&serde_json::json!({"clear": true})).send();
        let _ = c.post(format!("{base}/free")).json(&serde_json::json!({"unload_models": true, "free_memory": true})).send();
        std::thread::sleep(Duration::from_millis(150));
    }
}

/// Returns the inactivity watchdog threshold (in seconds) for a video workflow node.
/// Sampler nodes (`283`, `308`) get a longer grace period before their first `progress` event
/// because ComfyUI must offload Gemma 3, cold-load/stage the 22B FP8 weights, initialize
/// CUDA/cuDNN kernels, and complete step 1 before emitting progress.
fn video_stage_timeout_secs(active_node: Option<&str>, node_has_progress: bool) -> u64 {
    match active_node {
        Some("325") => 180, // Gemma 3 prompt expansion (takes ~90-120s)
        Some("316") | Some("317") | Some("324") | Some("285") | Some("286") | Some("279") | Some("311")
        | Some("100") | Some("101") | Some("102") | Some("103") | Some("105")
        | Some("236") | Some("237") | Some("248") | Some("212") | Some("233") => 180, // Model weight staging
        Some("303") | Some("313") | Some("240") | Some("247") => 60, // Prompt conditioning encoding
        Some("110") => 240, // MiniMax H3 Qwen3-VL 32B prompt + reference encoding (first load + inference)
        Some("304") | Some("314") | Some("284") | Some("282") | Some("239") | Some("241") => 45, // Conditioning and guiders setup
        Some("295") | Some("305") | Some("318") | Some("278") | Some("277") | Some("276")
        | Some("291") | Some("280") | Some("306") | Some("281")
        | Some("111") | Some("112") | Some("113") | Some("114") => 45, // Latent initialization
        Some("269") | Some("290") | Some("330") | Some("289") | Some("296") | Some("288")
        | Some("200") | Some("201") | Some("202") | Some("203") | Some("204") | Some("205") => 60, // Reference frame processing
        Some("115") | Some("283") | Some("308") | Some("215") | Some("246") => {
            if node_has_progress {
                180 // Active diffusion steps (each step updates last_activity)
            } else {
                300 // Cold start / text-encoder->DiT swap / weight staging + step 1
            }
        }
        Some("287") => 180, // Spatial 2x upscaling
        Some("315") | Some("297") | Some("116") | Some("117") => 180, // VAE decoding
        Some("310") | Some("118") | Some("75") => 120, // Video encoding & saving
        _ => 120,
    }
}

fn video_stage_name(active_node: Option<&str>) -> &'static str {
    match active_node {
        Some("325") => "prompt expansion (Gemma 3 12B)",
        Some("316") | Some("317") | Some("324") | Some("285") | Some("286") | Some("279") | Some("311")
        | Some("100") | Some("101") | Some("102") | Some("103") | Some("105")
        | Some("236") | Some("237") | Some("248") | Some("212") | Some("233") => "model weight staging",
        Some("303") | Some("313") | Some("240") | Some("247") => "prompt conditioning encoding",
        Some("110") => "multimodal prompt & reference encoding",
        Some("304") | Some("314") | Some("284") | Some("282") | Some("239") | Some("241") => "conditioning and guidance setup",
        Some("295") | Some("305") | Some("318") | Some("278")
        | Some("111") | Some("112") | Some("113") | Some("114") => "latent initialization",
        Some("269") | Some("290") | Some("330") | Some("289") | Some("296") | Some("288")
        | Some("200") | Some("201") | Some("202") | Some("203") | Some("204") | Some("205") => "reference frame processing",
        Some("115") => "MiniMax H3 video diffusion",
        Some("283") | Some("215") => "Stage 1 video diffusion",
        Some("287") => "spatial 2x upscaling",
        Some("308") | Some("246") => "Stage 2 detail refinement",
        Some("315") | Some("116") => "VAE video decoding",
        Some("297") | Some("117") => "audio VAE decoding",
        Some("310") | Some("118") | Some("75") => "MP4 video encoding",
        _ => "generation pipeline",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_cold_start_gets_extended_timeout() {
        assert_eq!(video_stage_timeout_secs(Some("283"), false), 300);
        assert_eq!(video_stage_timeout_secs(Some("283"), true), 180);
        assert_eq!(video_stage_timeout_secs(Some("308"), false), 300);
        assert_eq!(video_stage_timeout_secs(Some("308"), true), 180);
        assert_eq!(video_stage_timeout_secs(Some("115"), false), 300);
        assert_eq!(video_stage_timeout_secs(Some("115"), true), 180);
        assert_eq!(video_stage_timeout_secs(Some("215"), false), 300);
    }

    #[test]
    fn stage_names_match_expected_nodes() {
        assert_eq!(video_stage_name(Some("283")), "Stage 1 video diffusion");
        assert_eq!(video_stage_name(Some("308")), "Stage 2 detail refinement");
        assert_eq!(video_stage_name(Some("115")), "MiniMax H3 video diffusion");
        assert_eq!(video_stage_name(None), "generation pipeline");
    }

    #[test]
    fn workflow_minimax_h3_builds_t2v_i2v_continuation_and_r2v() {
        let mut p = VideoParams::default();
        p.prompt = "a glowing neon city".into();
        p.lora_name = Some("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors".into());
        p.lora_strength = 0.85;

        // 1. T2V + Turbo LoRA
        let t2v = workflow_minimax_h3(&p, None, None, &[]);
        assert_eq!(t2v["100"]["class_type"], "UnetLoaderGGUF");
        assert_eq!(t2v["101"]["class_type"], "CLIPLoader");
        assert_eq!(t2v["101"]["inputs"]["type"], "minimax");
        assert_eq!(t2v["105"]["class_type"], "LoraLoaderModelOnly");
        assert_eq!(t2v["110"]["class_type"], "MiniMaxH3ImageToVideo");
        assert_eq!(t2v["110"]["inputs"]["length"], 124);
        assert_eq!(t2v["113"]["inputs"]["sampler_name"], "er_sde");
        assert_eq!(t2v["114"]["inputs"]["scheduler"], "sgm_uniform");
        assert_eq!(t2v["75"]["class_type"], "SaveVideo");

        // 2. I2V + End Frame Continuation
        p.lora_name = None;
        let cont = workflow_minimax_h3(&p, Some("first.png [temp]"), Some("last.png [temp]"), &[]);
        assert!(cont.get("105").is_none());
        assert_eq!(cont["110"]["inputs"]["first_frame"], json!(["201", 0]));
        assert_eq!(cont["110"]["inputs"]["last_frame"], json!(["203", 0]));

        // 3. R2V with Ref2VA weights & reference images
        p.use_r2v = true;
        let refs = vec!["ref0.png [temp]".to_string(), "ref1.png [temp]".to_string()];
        let r2v = workflow_minimax_h3(&p, None, None, &refs);
        assert_eq!(r2v["100"]["inputs"]["unet_name"], "MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf");
        assert_eq!(r2v["110"]["class_type"], "MiniMaxH3ReferenceToVideo");
        assert_eq!(r2v["110"]["inputs"]["ref_images.ref_image_0"], json!(["200", 0]));
        assert_eq!(r2v["110"]["inputs"]["ref_images.ref_image_1"], json!(["201", 0]));
    }

    #[test]
    fn workflow_ltx25_builds_gguf_and_safetensors_graphs() {
        let mut p = VideoParams::default();
        p.arch = VideoArch::Ltx25;
        p.unet_name = "ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf".into();
        p.clip_name = "gemma4-12b-ltx25-comfy-w4a8.safetensors".into();
        p.vae_name = "ltx-2.5-video-vae-bf16.safetensors".into();
        p.audio_vae_name = "ltx-2.5-audio-vae-bf16.safetensors".into();
        p.latent_upscale_model = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors".into();

        let g = workflow_ltx25(&p, Some("start.png [temp]"));
        assert_eq!(g["236"]["class_type"], "UnetLoaderGGUF");
        assert_eq!(g["237"]["class_type"], "CLIPLoader");
        assert_eq!(g["237"]["inputs"]["type"], "ltxv");
        assert_eq!(g["296"]["inputs"]["bypass"], false);
        assert_eq!(g["291"]["inputs"]["sampler_name"], "er_sde");
        assert_eq!(g["75"]["class_type"], "SaveVideo");
    }

    #[test]
    fn workflow_progress_spec_computes_total_progress_across_stages() {
        let two_stage = WorkflowProgressSpec::TwoStage {
            stage1_node: "215",
            stage2_node: "246",
            stage1_steps: 8,
            stage2_steps: 3,
        };
        let p_s1_mid = two_stage.progress_for_node_step(Some("215"), 4, 8).unwrap();
        let p_s1_end = two_stage.progress_for_node_step(Some("215"), 8, 8).unwrap();
        let p_s2_start = two_stage.progress_for_node_step(Some("246"), 1, 3).unwrap();
        let p_s2_end = two_stage.progress_for_node_step(Some("246"), 3, 3).unwrap();
        let p_vae = two_stage.progress_for_node_step(Some("315"), 1, 4).unwrap();

        assert!(p_s1_mid > 0.05 && p_s1_mid < p_s1_end);
        assert!(p_s1_end < p_s2_start, "Stage 2 step 1 must advance past Stage 1 end");
        assert!(p_s2_start < p_s2_end);
        assert!(p_s2_end <= p_vae && p_vae < 1.0);

        let single = WorkflowProgressSpec::SingleStage { sampler_node: "115" };
        let s_mid = single.progress_for_node_step(Some("115"), 10, 20).unwrap();
        assert!((s_mid - 0.50).abs() < 0.01);
    }
}


