//! ComfyUI HTTP client: upload, queue (verified Qwen Image 2.1 edit graph), poll, fetch.
use crate::engine::base_url;
use reqwest::blocking::{multipart, Client};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    g["7"] = json!({"class_type": "ComfySwitchNode", "inputs": {"switch": p.custom.is_some(), "on_false": ["5", 2], "on_true": ["6", 0]}});
    g["8"] = json!({"class_type": "KSampler", "inputs": {
        "model": ["4", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["7", 0],
        "seed": p.seed, "steps": p.steps, "cfg": p.cfg,
        "sampler_name": p.sampler, "scheduler": p.scheduler, "denoise": 1.0}});
    g["9"] = json!({"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}});
    g["11"] = json!({"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "ImageStudio/edit"}});
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
) -> Result<Vec<u8>, String> {
    let ws_stop = Arc::new(AtomicBool::new(false));
    spawn_ws_progress(ws_stop.clone(), progress, preview);

    let c = Client::builder().timeout(Duration::from_secs(120)).build().map_err(|e| e.to_string())?;
    let tag = p.seed % 100000;
    let mut names = vec![];
    for (i, png) in imgs.into_iter().enumerate() {
        status(format!("Uploading image {}...", i + 1));
        names.push(upload(&c, png, &format!("studio_{tag}_{i}.png"))?);
    }
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
    loop {
        if cancel.load(Ordering::SeqCst) {
            let _ = c.post(format!("{}/interrupt", base_url())).send();
            ws_stop.store(true, Ordering::SeqCst);
            return Err("Cancelled".into());
        }
        std::thread::sleep(Duration::from_millis(500));
        let secs = t0.elapsed().as_secs();
        let h: Value = c.get(format!("{}/history/{id}", base_url())).send().map_err(|e| format!("poll: {e}"))?.json().unwrap_or(Value::Null);
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
                let url = format!("{}/view", base_url());
                let bytes = c
                    .get(url)
                    .query(&[
                        ("filename", img["filename"].as_str().unwrap_or("")),
                        ("subfolder", img["subfolder"].as_str().unwrap_or("")),
                        ("type", img["type"].as_str().unwrap_or("output")),
                    ])
                    .send()
                    .and_then(|r| r.bytes())
                    .map_err(|e| format!("fetch result: {e}"))?;
                ws_stop.store(true, Ordering::SeqCst);
                return Ok(bytes.to_vec());
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
