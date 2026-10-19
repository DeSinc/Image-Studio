//! ComfyUI HTTP client: upload, queue (verified Qwen Image 2.1 edit graph), poll, fetch.
use crate::engine::base_url;
use reqwest::blocking::{multipart, Client};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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

pub struct VideoParams {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub duration: u32,
    pub fps: u32,
    pub seed: u64,
    pub prompt_enhance: bool,
    pub ckpt_name: String,
    pub text_encoder: String,
    pub latent_upscale_model: String,
    pub distilled_lora: String,
    pub gemma_lora: String,
}

impl Default for VideoParams {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1280,
            height: 720,
            duration: 5,
            fps: 24,
            seed: 42,
            prompt_enhance: false,
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

fn workflow_video(p: &VideoParams, image_name: Option<&str>) -> Value {
    let mut g = json!({
        "316": { "class_type": "CheckpointLoaderSimple", "inputs": { "ckpt_name": p.ckpt_name } },
        "317": { "class_type": "LTXAVTextEncoderLoader", "inputs": { "text_encoder": p.text_encoder, "ckpt_name": p.ckpt_name, "device": "default" } },
        "279": { "class_type": "LTXVAudioVAELoader", "inputs": { "ckpt_name": p.ckpt_name } },
        "285": { "class_type": "LoraLoaderModelOnly", "inputs": { "model": ["316", 0], "lora_name": p.distilled_lora, "strength_model": 0.5 } },
        "324": { "class_type": "LoraLoader", "inputs": { "model": ["285", 0], "clip": ["317", 0], "lora_name": p.gemma_lora, "strength_model": 1.0, "strength_clip": 1.0 } },

        "277": { "class_type": "RandomNoise", "inputs": { "noise_seed": p.seed } },
        "276": { "class_type": "RandomNoise", "inputs": { "noise_seed": 42 } },
        "291": { "class_type": "KSamplerSelect", "inputs": { "sampler_name": "euler" } },
        "280": { "class_type": "KSamplerSelect", "inputs": { "sampler_name": "euler" } },
        "306": { "class_type": "ManualSigmas", "inputs": { "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0" } },
        "281": { "class_type": "ManualSigmas", "inputs": { "sigmas": "0.85, 0.7250, 0.4219, 0.0" } },

        "313": { "class_type": "CLIPTextEncode", "inputs": { "clip": ["317", 0], "text": p.negative } },
        "304": { "class_type": "LTXVConditioning", "inputs": { "positive": ["303", 0], "negative": ["313", 0], "frame_rate": p.fps as f32 } },
        "314": { "class_type": "CFGGuider", "inputs": { "model": ["285", 0], "positive": ["304", 0], "negative": ["304", 1], "cfg": 1.0 } },

        "295": { "class_type": "EmptyLTXVLatentVideo", "inputs": { "width": p.width / 2, "height": p.height / 2, "length": (p.duration * p.fps + 1) as i64, "batch_size": 1 } },
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

    if let Some(img_name) = image_name {
        g["269"] = json!({ "class_type": "LoadImage", "inputs": { "image": img_name } });
        g["290"] = json!({ "class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["269", 0], "resize_type": "scale dimensions", "resize_type.width": p.width, "resize_type.height": p.height, "resize_type.crop": "center", "scale_method": "lanczos"
        }});
        g["330"] = json!({ "class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["290", 0], "resize_type": "scale longer dimension", "resize_type.longer_size": 512, "scale_method": "lanczos"
        }});
        g["289"] = json!({ "class_type": "LTXVPreprocess", "inputs": { "image": ["330", 0], "img_compression": 18 } });

        g["296"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["316", 2], "image": ["289", 0], "latent": ["295", 0], "bypass": false, "strength": 0.7 } });
        g["288"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["316", 2], "image": ["289", 0], "latent": ["287", 0], "bypass": false, "strength": 1.0 } });

        if p.prompt_enhance {
            g["325"] = json!({ "class_type": "TextGenerateLTX2Prompt", "inputs": {
                "clip": ["324", 1], "image": ["330", 0], "prompt": p.prompt, "max_length": 2048, "sampling_mode": "on",
                "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64, "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
                "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": (p.seed % 1000000) as i64, "sampling_mode.presence_penalty": 0, "thinking": false, "use_default_template": true
            }});
        }
    } else {
        g["269"] = json!({ "class_type": "EmptyImage", "inputs": { "width": 512, "height": 512, "batch_size": 1, "color": 0 } });
        g["290"] = json!({ "class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["269", 0], "resize_type": "scale dimensions", "resize_type.width": p.width, "resize_type.height": p.height, "resize_type.crop": "center", "scale_method": "lanczos"
        }});
        g["330"] = json!({ "class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["290", 0], "resize_type": "scale longer dimension", "resize_type.longer_size": 512, "scale_method": "lanczos"
        }});
        g["289"] = json!({ "class_type": "LTXVPreprocess", "inputs": { "image": ["330", 0], "img_compression": 18 } });

        g["296"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["316", 2], "image": ["289", 0], "latent": ["295", 0], "bypass": true, "strength": 0.7 } });
        g["288"] = json!({ "class_type": "LTXVImgToVideoInplace", "inputs": { "vae": ["316", 2], "image": ["289", 0], "latent": ["287", 0], "bypass": true, "strength": 1.0 } });

        if p.prompt_enhance {
            g["325"] = json!({ "class_type": "TextGenerateLTX2Prompt", "inputs": {
                "clip": ["324", 1], "prompt": p.prompt, "max_length": 2048, "sampling_mode": "on",
                "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 64, "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0.05,
                "sampling_mode.repetition_penalty": 1.05, "sampling_mode.seed": (p.seed % 1000000) as i64, "sampling_mode.presence_penalty": 0, "thinking": false, "use_default_template": true
            }});
        }
    }

    if p.prompt_enhance {
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

fn spawn_ws_progress(
    stop: Arc<AtomicBool>,
    on_progress: Arc<dyn Fn(u32, u32) + Send + Sync>,
    on_preview: Arc<dyn Fn(Vec<u8>) + Send + Sync>,
) {
    std::thread::spawn(move || {
        let stream = match std::net::TcpStream::connect(format!("127.0.0.1:{}", crate::engine::PORT)) {
            Ok(s) => s,
            Err(_) => return,
        };
        let _ = stream.set_read_timeout(Some(Duration::from_millis(350)));
        use std::io::{BufRead, Read, Write};
        let mut reader = std::io::BufReader::new(stream);
        let req = format!(
            "GET /ws?clientId=image-studio HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n",
            crate::engine::PORT
        );
        if reader.get_mut().write_all(req.as_bytes()).is_err() {
            return;
        }
        let mut line = String::new();
        while let Ok(n) = reader.read_line(&mut line) {
            if n == 0 || line == "\r\n" || line == "\n" {
                break;
            }
            line.clear();
        }
        while !stop.load(Ordering::SeqCst) {
            let mut header = [0u8; 2];
            if reader.read_exact(&mut header).is_err() {
                continue;
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
            if reader.read_exact(&mut payload).is_err() {
                break;
            }
            if opcode == 0x1 {
                if let Ok(text) = std::str::from_utf8(&payload) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(text) {
                        if val["type"] == "progress" {
                            if let (Some(cur), Some(max)) = (val["data"]["value"].as_u64(), val["data"]["max"].as_u64()) {
                                on_progress(cur as u32, max as u32);
                            }
                        }
                    }
                }
            } else if opcode == 0x2 && payload.len() > 8 {
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
    });
}

pub fn run(
    p: &Params,
    imgs: Vec<Vec<u8>>,
    cancel: &AtomicBool,
    status: &dyn Fn(String),
    progress: Arc<dyn Fn(u32, u32) + Send + Sync>,
    preview: Arc<dyn Fn(Vec<u8>) + Send + Sync>,
) -> Result<GenResult, String> {
    let ws_stop = Arc::new(AtomicBool::new(false));
    spawn_ws_progress(ws_stop.clone(), progress, preview);

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
    loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = c.post(format!("{}/interrupt", base_url())).send();
            let _ = c.post(format!("{}/queue", base_url())).json(&serde_json::json!({"clear": true})).send();
            ws_stop.store(true, Ordering::SeqCst);
            return Err("Cancelled".into());
        }
        std::thread::sleep(Duration::from_millis(500));
        let secs = t0.elapsed().as_secs();
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
    preview: Arc<dyn Fn(Vec<u8>) + Send + Sync>,
) -> Result<VideoGenResult, String> {
    let ws_stop = Arc::new(AtomicBool::new(false));
    spawn_ws_progress(ws_stop.clone(), progress, preview);

    let c = Client::builder().timeout(Duration::from_secs(120)).build().map_err(|e| e.to_string())?;
    let tag = p.seed % 100000;
    let mut image_name = None;
    let mut temp_files = vec![];
    if let Some(png) = img {
        status("Uploading initial frame...".into());
        let filename = format!("studio_vid_{tag}.png");
        let uploaded = upload(&c, png, &filename)?;
        image_name = Some(uploaded);
        temp_files.push(filename);
    }

    prepare_clean_slate(&c, status);

    status("Queueing video generation...".into());
    let body = json!({ "prompt": workflow_video(p, image_name.as_deref()), "client_id": "image-studio" });
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
    loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = c.post(format!("{}/interrupt", base_url())).send();
            let _ = c.post(format!("{}/queue", base_url())).json(&serde_json::json!({"clear": true})).send();
            ws_stop.store(true, Ordering::SeqCst);
            return Err("Cancelled".into());
        }
        std::thread::sleep(Duration::from_millis(500));
        let secs = t0.elapsed().as_secs();
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
        status(if running {
            format!("Generating video (high-load diffusion)... {secs}s")
        } else {
            format!("Waiting in queue... {secs}s")
        });
        if secs > 3600 {
            ws_stop.store(true, Ordering::SeqCst);
            return Err("Timed out (video generation took > 1hr)".into());
        }
    }
}

fn prepare_clean_slate(c: &Client, status: &dyn Fn(String)) {
    let base = base_url();
    // 1. Clear residual queue
    let _ = c.post(format!("{base}/queue")).json(&serde_json::json!({"clear": true})).send();

    // 2. Wait up to 3 seconds if an old job is currently finishing its abort/unwind
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
        std::thread::sleep(Duration::from_millis(150));
    }
}

