//! Finds, downloads (if needed), launches (hidden) and supervises ComfyUI.
//! Runs on its own thread and never touches the UI thread; the UI only reads `Shared`.
use serde::{Deserialize, Serialize};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const PORT: u16 = 8188;
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub fn base_url() -> String {
    format!("http://127.0.0.1:{PORT}")
}

pub fn data_dir() -> PathBuf {
    let d = dirs::data_local_dir().unwrap_or_else(std::env::temp_dir).join("ImageStudio");
    let _ = std::fs::create_dir_all(&d);
    d
}

pub fn log(msg: &str) {
    use std::io::Write;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(data_dir().join("app.log")) {
        let _ = writeln!(f, "[{t}] {msg}");
    }
}

pub fn tail(path: &Path, lines: usize) -> String {
    let s = std::fs::read(path).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
    let v: Vec<&str> = s.lines().collect();
    v[v.len().saturating_sub(lines)..].join("\n")
}

pub fn format_bytes(b: u64) -> String {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;
    if b >= GB {
        format!("{:.2} GB", b as f64 / GB as f64)
    } else if b >= MB {
        format!("{:.1} MB", b as f64 / MB as f64)
    } else {
        format!("{:.0} KB", b as f64 / 1024.0)
    }
}

// ---------------------------------------------------------------- config
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct Config {
    pub comfy_dir: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
}

impl Config {
    pub fn load() -> Self {
        std::fs::read_to_string(data_dir().join("config.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
    pub fn save(&self) {
        if let Ok(s) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(data_dir().join("config.json"), s);
        }
    }
}

// ---------------------------------------------------------------- profiles
#[derive(Deserialize, Clone)]
pub struct ModelFile {
    pub category: String,
    pub filename: String,
    #[serde(default)]
    pub size_bytes: u64,
    #[serde(default)]
    pub hf_url: String,
    #[serde(default)]
    pub modelscope_url: String,
}
#[derive(Deserialize, Clone)]
#[allow(dead_code)]
pub struct Profile {
    pub name: String,
    pub default_steps: u32,
    pub default_cfg: f32,
    pub sampler: String,
    pub scheduler: String,
    pub models: Vec<ModelFile>,
}
#[derive(Deserialize)]
struct ProfileFile {
    active_profile: String,
    profiles: Vec<serde_json::Value>,
}

impl Profile {
    /// `models.json` next to the exe overrides the embedded default.
    pub fn load() -> Profile {
        let exe_json = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("models.json")))
            .and_then(|p| std::fs::read_to_string(p).ok());
        let text = exe_json.unwrap_or_else(|| include_str!("../models.json").to_string());
        let pf: ProfileFile = serde_json::from_str(&text)
            .or_else(|_| serde_json::from_str(include_str!("../models.json")))
            .expect("models.json");
        pf.profiles
            .into_iter()
            .find(|p| p["id"] == pf.active_profile.as_str())
            .and_then(|p| serde_json::from_value(p).ok())
            .expect("active profile")
    }
    pub fn file(&self, cat: &str) -> String {
        self.models.iter().find(|m| m.category == cat).map(|m| m.filename.clone()).unwrap_or_default()
    }
}

// ---------------------------------------------------------------- state
#[derive(Clone)]
pub enum EngState {
    Detecting,
    DownloadingEngine {
        step: String,
        progress: f32,
        bytes_done: u64,
        bytes_total: u64,
        speed_mb: f32,
    },
    ExtractingEngine {
        step: String,
    },
    DownloadingModels {
        current_file: String,
        file_index: usize,
        total_files: usize,
        file_progress: f32,
        bytes_done: u64,
        bytes_total: u64,
        speed_mb: f32,
        overall_progress: f32,
    },
    Starting {
        since: Instant,
        attempt: usize,
    },
    Ready {
        ours: bool,
        gpu: String,
    },
    Failed(String),
}

#[derive(Default, Clone)]
pub struct Paths {
    pub comfy: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub models: PathBuf,
}

pub struct Shared {
    pub state: Mutex<EngState>,
    pub paths: Mutex<Paths>,
    pub models_found: Mutex<Vec<(String, bool)>>,
    pub restart: AtomicBool,
    pub retry: AtomicBool,
    pub cancel: AtomicBool,
    pub cfg: Mutex<Config>,
}

impl Shared {
    pub fn new() -> Arc<Shared> {
        Arc::new(Shared {
            state: Mutex::new(EngState::Detecting),
            paths: Mutex::new(Paths::default()),
            models_found: Mutex::new(vec![]),
            restart: AtomicBool::new(false),
            retry: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            cfg: Mutex::new(Config::load()),
        })
    }
}

fn detect_gpu_vendor() -> &'static str {
    let out = Command::new("powershell")
        .args(["-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
    if let Ok(o) = out {
        let s = String::from_utf8_lossy(&o.stdout).to_lowercase();
        if s.contains("nvidia") || s.contains("geforce") || s.contains("quadro") || s.contains("rtx") {
            return "nvidia";
        } else if s.contains("amd") || s.contains("radeon") {
            return "amd";
        } else if s.contains("intel") || s.contains("arc") {
            return "intel";
        }
    }
    "nvidia"
}

fn get_comfy_portable_url() -> (&'static str, &'static str) {
    let vendor = detect_gpu_vendor();
    match vendor {
        "amd" => ("https://github.com/Comfy-Org/ComfyUI/releases/download/v0.39.0/ComfyUI_windows_portable_amd.7z", "AMD"),
        "intel" => ("https://github.com/Comfy-Org/ComfyUI/releases/download/v0.39.0/ComfyUI_windows_portable_intel.7z", "Intel"),
        _ => ("https://github.com/Comfy-Org/ComfyUI/releases/download/v0.39.0/ComfyUI_windows_portable_nvidia.7z", "NVIDIA"),
    }
}

pub fn find_comfy(cfg: &Config) -> Option<PathBuf> {
    let mut c: Vec<PathBuf> = vec![];
    if let Some(p) = &cfg.comfy_dir {
        c.push(p.clone());
        c.push(p.join("ComfyUI"));
    }

    // Default automated bootstrap locations
    let d = data_dir();
    c.push(d.join("engine/ComfyUI_windows_portable/ComfyUI"));
    c.push(d.join("engine/ComfyUI"));
    c.push(d.join("ComfyUI_windows_portable/ComfyUI"));
    c.push(d.join("ComfyUI"));

    // Next to current exe (portable folder layout)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            c.push(parent.join("ComfyUI_windows_portable/ComfyUI"));
            c.push(parent.join("ComfyUI"));
        }
    }

    // Common user home folders
    if let Some(h) = dirs::home_dir() {
        c.push(h.join("ComfyUI-Installs/ComfyUI/ComfyUI"));
        c.push(h.join("ComfyUI_windows_portable/ComfyUI"));
        c.push(h.join("ComfyUI"));
        c.push(h.join("Documents/ComfyUI"));
    }
    c.into_iter().find(|p| p.join("main.py").exists())
}

fn find_pythons(comfy: &Path) -> Vec<PathBuf> {
    let parent = comfy.parent().map(Path::to_path_buf).unwrap_or_default();
    let grand = parent.parent().map(Path::to_path_buf).unwrap_or_default();
    [
        comfy.join(".venv/Scripts/python.exe"),
        comfy.join("venv/Scripts/python.exe"),
        parent.join(".venv/Scripts/python.exe"),
        parent.join("python_embeded/python.exe"),
        grand.join("python_embeded/python.exe"),
        parent.join("standalone-env/python.exe"),
    ]
    .into_iter()
    .filter(|p| p.exists())
    .collect()
}

pub fn find_models(cfg: &Config, comfy: &Path) -> PathBuf {
    if let Some(m) = &cfg.models_dir {
        return m.clone();
    }
    if let Some(h) = dirs::home_dir() {
        let s = h.join("ComfyUI-Shared/models");
        if s.exists() {
            return s;
        }
    }
    comfy.join("models")
}

fn ping(client: &reqwest::blocking::Client) -> Option<String> {
    let v: serde_json::Value = client.get(format!("{}/system_stats", base_url())).send().ok()?.json().ok()?;
    Some(v["devices"][0]["name"].as_str().unwrap_or("GPU").to_string())
}

pub fn kill_tree(c: &mut Child) {
    let _ = Command::new("taskkill")
        .args(["/F", "/T", "/PID", &c.id().to_string()])
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let _ = c.kill();
    let _ = c.wait();
}

fn spawn(py: &Path, comfy: &Path, models: &Path) -> Result<Child, String> {
    let log = data_dir().join("comfyui.log");
    let out = std::fs::File::create(&log).map_err(|e| e.to_string())?;
    let err = out.try_clone().map_err(|e| e.to_string())?;
    let mut cmd = Command::new(py);
    cmd.args([
        "-u",
        "main.py",
        "--port",
        &PORT.to_string(),
        "--listen",
        "127.0.0.1",
        "--disable-auto-launch",
        "--preview-method",
        "auto",
        "--models-directory",
    ])
    .arg(models)
    .current_dir(comfy)
    .env("PYTHONUNBUFFERED", "1")
    .env("PYTHONIOENCODING", "utf-8")
    .stdin(Stdio::null())
    .stdout(out)
    .stderr(err)
    .creation_flags(CREATE_NO_WINDOW);
    cmd.spawn().map_err(|e| format!("could not start {}: {e}", py.display()))
}

fn download_file<F>(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
    expected_size: u64,
    mut on_progress: F,
    cancel: &AtomicBool,
    ctx: &eframe::egui::Context,
) -> Result<(), String>
where
    F: FnMut(u64, u64, f32),
{
    if let Some(p) = dest.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    let part_path = dest.with_extension(format!(
        "{}.part",
        dest.extension().and_then(|e| e.to_str()).unwrap_or("tmp")
    ));

    let mut existing_len = if part_path.exists() {
        std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    if expected_size > 0 && existing_len >= expected_size {
        let _ = std::fs::rename(&part_path, dest);
        return Ok(());
    }

    let mut req = client.get(url);
    if existing_len > 0 {
        req = req.header("Range", format!("bytes={existing_len}-"));
    }

    let mut resp = req.send().map_err(|e| format!("HTTP request failed: {e}"))?;
    let status = resp.status();

    let (mut file, total_bytes) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        let content_len = resp.content_length().unwrap_or(0);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part_path)
            .map_err(|e| format!("failed to open file for append: {e}"))?;
        let tot = if expected_size > 0 {
            expected_size
        } else {
            existing_len + content_len
        };
        (file, tot)
    } else if status.is_success() {
        existing_len = 0;
        let content_len = resp.content_length().unwrap_or(0);
        let file = std::fs::File::create(&part_path).map_err(|e| format!("failed to create file: {e}"))?;
        let tot = if expected_size > 0 { expected_size } else { content_len };
        (file, tot)
    } else if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        let _ = std::fs::rename(&part_path, dest);
        return Ok(());
    } else {
        return Err(format!("server returned HTTP status {status}"));
    };

    use std::io::{Read, Write};
    let mut buf = vec![0u8; 1024 * 1024]; // 1MB buffer
    let mut downloaded = existing_len;
    let mut last_tick = Instant::now();
    let mut bytes_since_tick = 0u64;
    let mut speed_mb = 0.0f32;

    on_progress(downloaded, total_bytes, speed_mb);
    ctx.request_repaint();

    loop {
        if cancel.load(Ordering::SeqCst) {
            return Err("download cancelled".into());
        }
        let n = match resp.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("download read error: {e}")),
        };
        file.write_all(&buf[..n]).map_err(|e| format!("write error: {e}"))?;
        downloaded += n as u64;
        bytes_since_tick += n as u64;

        let elapsed = last_tick.elapsed();
        if elapsed >= Duration::from_millis(250) {
            let secs = elapsed.as_secs_f32();
            if secs > 0.0 {
                speed_mb = (bytes_since_tick as f32 / 1024.0 / 1024.0) / secs;
            }
            last_tick = Instant::now();
            bytes_since_tick = 0;
            on_progress(downloaded, total_bytes, speed_mb);
            ctx.request_repaint();
        }
    }
    file.flush().map_err(|e| format!("flush error: {e}"))?;
    drop(file);

    if dest.exists() {
        let _ = std::fs::remove_file(dest);
    }
    std::fs::rename(&part_path, dest).map_err(|e| format!("rename error: {e}"))?;
    on_progress(downloaded, total_bytes, speed_mb);
    ctx.request_repaint();
    Ok(())
}

fn bootstrap_engine(
    sh: &Shared,
    ctx: &eframe::egui::Context,
    dl_client: &reqwest::blocking::Client,
) -> Result<PathBuf, String> {
    let engine_dir = data_dir().join("engine");
    let _ = std::fs::create_dir_all(&engine_dir);
    let archive_path = data_dir().join("comfyui_portable.7z");

    let (comfy_url, vendor_name) = get_comfy_portable_url();
    log(&format!("bootstrap: downloading {vendor_name} portable from {comfy_url}"));

    let mut downloaded_archive = false;
    if archive_path.exists() {
        if let Ok(meta) = std::fs::metadata(&archive_path) {
            if meta.len() > 1_500_000_000 {
                downloaded_archive = true;
            }
        }
    }

    if !downloaded_archive {
        let archive_size = 2_003_629_308u64;
        let step_desc = format!("Downloading ComfyUI Portable ({vendor_name})...");
        let step_clone = step_desc.clone();
        let res = download_file(
            dl_client,
            comfy_url,
            &archive_path,
            archive_size,
            |done, total, speed| {
                let prog = if total > 0 { (done as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
                *sh.state.lock().unwrap() = EngState::DownloadingEngine {
                    step: step_clone.clone(),
                    progress: prog,
                    bytes_done: done,
                    bytes_total: total,
                    speed_mb: speed,
                };
            },
            &sh.cancel,
            ctx,
        );
        if let Err(e) = res {
            return Err(format!("Engine download failed: {e}"));
        }
    }

    *sh.state.lock().unwrap() = EngState::ExtractingEngine {
        step: format!("Extracting ComfyUI Portable ({vendor_name}) components..."),
    };
    ctx.request_repaint();
    log("bootstrap: extracting 7z archive into engine directory");

    sevenz_rust2::decompress_file(&archive_path, &engine_dir)
        .map_err(|e| format!("Failed to extract ComfyUI 7z archive: {e}"))?;

    let _ = std::fs::remove_file(&archive_path);
    log("bootstrap: extracted successfully");

    let cfg = sh.cfg.lock().unwrap().clone();
    if let Some(c) = find_comfy(&cfg) {
        Ok(c)
    } else {
        Err("Extraction finished but main.py was not found in extracted directory.".into())
    }
}

fn bootstrap_models(
    sh: &Shared,
    ctx: &eframe::egui::Context,
    dl_client: &reqwest::blocking::Client,
    models_dir: &Path,
    profile: &Profile,
) -> Result<(), String> {
    let missing: Vec<_> = profile
        .models
        .iter()
        .filter(|m| !models_dir.join(&m.category).join(&m.filename).exists())
        .cloned()
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    let total_all_bytes: u64 = missing.iter().map(|m| m.size_bytes).sum();
    let mut completed_bytes: u64 = 0;

    for (idx, m) in missing.iter().enumerate() {
        let dest = models_dir.join(&m.category).join(&m.filename);
        log(&format!("bootstrap: downloading model {} -> {}", m.filename, dest.display()));

        let cur_file = m.filename.clone();
        let cur_idx = idx + 1;
        let total_files = missing.len();
        let file_expected = m.size_bytes;

        let mut download_result = download_file(
            dl_client,
            &m.hf_url,
            &dest,
            file_expected,
            |done, total, speed| {
                let f_prog = if total > 0 { (done as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
                let overall = if total_all_bytes > 0 {
                    ((completed_bytes + done) as f32 / total_all_bytes as f32).clamp(0.0, 1.0)
                } else {
                    f_prog
                };
                *sh.state.lock().unwrap() = EngState::DownloadingModels {
                    current_file: cur_file.clone(),
                    file_index: cur_idx,
                    total_files,
                    file_progress: f_prog,
                    bytes_done: done,
                    bytes_total: total,
                    speed_mb: speed,
                    overall_progress: overall,
                };
            },
            &sh.cancel,
            ctx,
        );

        if download_result.is_err() && !m.modelscope_url.is_empty() {
            log(&format!("HF download failed for {}, falling back to ModelScope", m.filename));
            download_result = download_file(
                dl_client,
                &m.modelscope_url,
                &dest,
                file_expected,
                |done, total, speed| {
                    let f_prog = if total > 0 { (done as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
                    let overall = if total_all_bytes > 0 {
                        ((completed_bytes + done) as f32 / total_all_bytes as f32).clamp(0.0, 1.0)
                    } else {
                        f_prog
                    };
                    *sh.state.lock().unwrap() = EngState::DownloadingModels {
                        current_file: cur_file.clone(),
                        file_index: cur_idx,
                        total_files,
                        file_progress: f_prog,
                        bytes_done: done,
                        bytes_total: total,
                        speed_mb: speed,
                        overall_progress: overall,
                    };
                },
                &sh.cancel,
                ctx,
            );
        }

        if let Err(e) = download_result {
            return Err(format!("Failed to download {}: {e}", m.filename));
        }

        completed_bytes += file_expected;
    }

    Ok(())
}

pub fn start_supervisor(sh: Arc<Shared>, profile: Profile, ctx: eframe::egui::Context) {
    std::thread::spawn(move || {
        let ping_client = reqwest::blocking::Client::builder().timeout(Duration::from_millis(1200)).build().unwrap();
        let dl_client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(7200))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        let mut child: Option<Child> = None;
        let mut attempt = 0usize;
        let mut started = Instant::now();
        let mut failed = false;

        loop {
            if sh.retry.swap(false, Ordering::SeqCst) || sh.restart.swap(false, Ordering::SeqCst) {
                if let Some(c) = child.as_mut() {
                    kill_tree(c);
                }
                child = None;
                attempt = 0;
                failed = false;
                log("retry/restart supervisor cycle");
            }

            let cfg = sh.cfg.lock().unwrap().clone();
            let mut comfy = find_comfy(&cfg);

            // 1. Is ComfyUI already running and responding?
            if let Some(gpu) = ping(&ping_client) {
                attempt = 0;
                failed = false;
                *sh.state.lock().unwrap() = EngState::Ready { ours: child.is_some(), gpu };
                let models = comfy.as_ref().map(|c| find_models(&cfg, c)).unwrap_or_default();
                let pys = comfy.as_ref().map(|c| find_pythons(c)).unwrap_or_default();
                *sh.paths.lock().unwrap() = Paths { comfy: comfy.clone(), python: pys.first().cloned(), models: models.clone() };
                *sh.models_found.lock().unwrap() = profile
                    .models
                    .iter()
                    .map(|m| (m.filename.clone(), models.join(&m.category).join(&m.filename).exists()))
                    .collect();
                ctx.request_repaint();
                std::thread::sleep(Duration::from_millis(600));
                continue;
            }

            if failed {
                ctx.request_repaint();
                std::thread::sleep(Duration::from_millis(600));
                continue;
            }

            // 2. ComfyUI missing? Bootstrap automatically!
            if comfy.is_none() {
                match bootstrap_engine(&sh, &ctx, &dl_client) {
                    Ok(c) => {
                        comfy = Some(c.clone());
                        sh.cfg.lock().unwrap().comfy_dir = Some(c);
                        sh.cfg.lock().unwrap().save();
                    }
                    Err(e) => {
                        failed = true;
                        *sh.state.lock().unwrap() = EngState::Failed(e);
                        ctx.request_repaint();
                        std::thread::sleep(Duration::from_millis(600));
                        continue;
                    }
                }
            }

            let cd = comfy.as_ref().unwrap();
            let models = find_models(&cfg, cd);
            {
                let pys = find_pythons(cd);
                *sh.paths.lock().unwrap() = Paths { comfy: Some(cd.clone()), python: pys.first().cloned(), models: models.clone() };
            }

            // 3. Are all profile models present?
            match bootstrap_models(&sh, &ctx, &dl_client, &models, &profile) {
                Ok(()) => {
                    *sh.models_found.lock().unwrap() = profile
                        .models
                        .iter()
                        .map(|m| (m.filename.clone(), models.join(&m.category).join(&m.filename).exists()))
                        .collect();
                }
                Err(e) => {
                    failed = true;
                    *sh.state.lock().unwrap() = EngState::Failed(e);
                    ctx.request_repaint();
                    std::thread::sleep(Duration::from_millis(600));
                    continue;
                }
            }

            // 4. Launch ComfyUI process
            if child.is_none() {
                let pys = find_pythons(cd);
                if pys.is_empty() {
                    failed = true;
                    *sh.state.lock().unwrap() = EngState::Failed("No Python environment found inside ComfyUI directory.".into());
                } else if attempt > pys.len() {
                    failed = true;
                    let t = tail(&data_dir().join("comfyui.log"), 14);
                    *sh.state.lock().unwrap() = EngState::Failed(format!("ComfyUI failed to start.\n{t}"));
                } else {
                    let py = &pys[attempt % pys.len()];
                    log(&format!("launching {} (attempt {attempt})", py.display()));
                    match spawn(py, cd, &models) {
                        Ok(c) => {
                            child = Some(c);
                            started = Instant::now();
                            *sh.state.lock().unwrap() = EngState::Starting { since: started, attempt };
                        }
                        Err(e) => {
                            failed = true;
                            *sh.state.lock().unwrap() = EngState::Failed(e);
                        }
                    }
                }
            } else {
                if let Some(c) = child.as_mut() {
                    let exited = matches!(c.try_wait(), Ok(Some(_)));
                    if exited || started.elapsed() > Duration::from_secs(180) {
                        log(&format!("comfyui attempt {attempt} ended (exited={exited})"));
                        if !exited {
                            kill_tree(c);
                        }
                        child = None;
                        attempt += 1;
                    } else {
                        *sh.state.lock().unwrap() = EngState::Starting { since: started, attempt };
                    }
                }
            }

            ctx.request_repaint();
            std::thread::sleep(Duration::from_millis(600));
        }
    });
}

// ---------------------------------------------------------------- job object
/// Puts *this* process in a kill-on-close job. Children (ComfyUI and its python
/// grandchildren) inherit it, so they die whenever the app exits or crashes.
pub fn adopt_into_kill_job() {
    use windows_sys::Win32::System::JobObjects::*;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            log("CreateJobObject failed");
            return;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok1 = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        let ok2 = AssignProcessToJobObject(job, GetCurrentProcess());
        log(&format!("kill-on-close job: set={ok1} assign={ok2}"));
    }
}
