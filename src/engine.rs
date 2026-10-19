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
fn default_true() -> bool { true }

#[derive(Serialize, Deserialize, Clone)]
pub struct Config {
    pub comfy_dir: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
    #[serde(default)]
    pub ignore_paths: Vec<PathBuf>,
    #[serde(default)]
    pub active_profile: Option<String>,
    #[serde(default = "default_true")]
    pub dl_image_model: bool,
    #[serde(default)]
    pub dl_video_model: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            comfy_dir: None,
            models_dir: None,
            ignore_paths: Vec::new(),
            active_profile: None,
            dl_image_model: true,
            dl_video_model: false,
        }
    }
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
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub recommended_vram_gb: u32,
    pub default_steps: u32,
    pub default_cfg: f32,
    pub sampler: String,
    pub scheduler: String,
    pub models: Vec<ModelFile>,
}
#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ProfileFile {
    pub active_profile: String,
    pub profiles: Vec<Profile>,
}

pub fn detect_gpu_vram_gb() -> Option<u32> {
    static CACHE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let mut cmd = Command::new("nvidia-smi");
        cmd.args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .creation_flags(CREATE_NO_WINDOW);
        if let Ok(out) = cmd.output() {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout);
                if let Some(first_line) = s.lines().next() {
                    if let Ok(mib) = first_line.trim().parse::<u32>() {
                        let gb = (mib as f32 / 1024.0).round() as u32;
                        log(&format!("detect_gpu_vram: nvidia-smi reported {mib} MiB ({gb} GB)"));
                        return Some(gb);
                    }
                }
            }
        }
        None
    })
}

pub fn candidate_models_dirs(cfg: &Config) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(m) = &cfg.models_dir {
        dirs.push(m.clone());
    }
    if let Some(h) = dirs::home_dir() {
        dirs.push(h.join("ComfyUI-Shared").join("models"));
    }
    if let Some(cd) = find_comfy(cfg) {
        dirs.push(cd.join("models"));
        if let Some(parent) = cd.parent() {
            dirs.push(parent.join("models"));
        }
    }
    let d = data_dir();
    dirs.push(d.join("engine").join("ComfyUI").join("models"));
    dirs.push(d.join("engine").join("ComfyUI_windows_portable").join("ComfyUI").join("models"));
    dirs.push(d.join("ComfyUI").join("models"));
    dirs.push(d.join("models"));
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("models"));
            dirs.push(parent.join("ComfyUI").join("models"));
        }
    }
    dirs
}

impl Profile {
    pub fn all() -> Vec<Profile> {
        let exe_json = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join("models.json")))
            .and_then(|p| std::fs::read_to_string(p).ok());
        let text = exe_json.unwrap_or_else(|| include_str!("../models.json").to_string());
        let pf: ProfileFile = serde_json::from_str(&text)
            .or_else(|_| serde_json::from_str(include_str!("../models.json")))
            .expect("models.json");
        pf.profiles
    }

    pub fn load_with_cfg(cfg: &Config) -> Profile {
        let profiles = Self::all();
        // 1. Explicit user choice in config.json
        if let Some(ref aid) = cfg.active_profile {
            if let Some(p) = profiles.iter().find(|p| &p.id == aid) {
                return p.clone();
            }
        }

        // 2. Check existing models on disk: NEVER force redownload for existing installs!
        let candidates = candidate_models_dirs(cfg);
        let has_int8 = candidates.iter().any(|d| {
            d.join("text_encoders").join("qwen3vl_8b_int8_convrot.safetensors").is_file()
        });
        let has_w4a8 = candidates.iter().any(|d| {
            d.join("text_encoders").join("qwen3vl_8b_w4a8.safetensors").is_file()
        });

        if has_int8 {
            if let Some(p) = profiles.iter().find(|p| p.id == "qwen_2.1_int8") {
                log("Profile: detected existing INT8 models on disk; preserving qwen_2.1_int8 profile");
                return p.clone();
            }
        } else if has_w4a8 {
            if let Some(p) = profiles.iter().find(|p| p.id == "qwen_2.1_8gb") {
                log("Profile: detected existing W4A8 models on disk; preserving qwen_2.1_8gb profile");
                return p.clone();
            }
        }

        // 3. New install with no text encoder downloaded yet: auto-detect based on VRAM
        if let Some(vram) = detect_gpu_vram_gb() {
            if vram <= 8 {
                if let Some(p) = profiles.iter().find(|p| p.id == "qwen_2.1_8gb") {
                    log(&format!("Profile: fresh install auto-selected 8GB quant profile for {vram}GB GPU"));
                    return p.clone();
                }
            }
        }

        // 4. Default to standard int8 profile
        if let Some(p) = profiles.iter().find(|p| p.id == "qwen_2.1_int8") {
            p.clone()
        } else {
            profiles.first().cloned().expect("at least one profile")
        }
    }

    pub fn load() -> Profile {
        let cfg = Config::load();
        Self::load_with_cfg(&cfg)
    }

    pub fn file(&self, cat: &str) -> String {
        self.models.iter().find(|m| m.category == cat).map(|m| m.filename.clone()).unwrap_or_default()
    }

    pub fn missing_models(sh: &Shared) -> Vec<ModelFile> {
        let all_profiles = Self::all();
        let ltx_profile = all_profiles.iter().find(|p| p.id == "ltx_2.3").cloned();
        let profile = sh.active_profile.lock().unwrap().clone();
        let models_dir = sh.paths.lock().unwrap().models.clone();

        let mut target_models: Vec<ModelFile> = Vec::new();
        if sh.dl_image_model.load(Ordering::SeqCst) {
            target_models.extend(profile.models.clone());
        }
        if sh.dl_video_model.load(Ordering::SeqCst) {
            if let Some(ref vid_p) = ltx_profile {
                target_models.extend(vid_p.models.clone());
            }
        }

        target_models
            .into_iter()
            .filter(|m| !safe_model_path(&models_dir, &m.category, &m.filename).exists())
            .collect()
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
    ModelDownloadFailed {
        current_file: String,
        error: String,
    },
    RepairingEngine {
        step: String,
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
    pub clean_engine: AtomicBool,
    pub cancel: AtomicBool,
    pub cancel_model_dl: AtomicBool,
    pub dl_image_model: AtomicBool,
    pub dl_video_model: AtomicBool,
    pub cfg: Mutex<Config>,
    pub pending_models_dir: Mutex<Option<PathBuf>>,
    pub migration_status: Mutex<Option<String>>,
    pub active_profile: Mutex<Profile>,
}

impl Shared {
    pub fn new() -> Arc<Shared> {
        let cfg = Config::load();
        let dl_img = cfg.dl_image_model;
        let dl_vid = cfg.dl_video_model;
        Arc::new(Shared {
            state: Mutex::new(EngState::Detecting),
            paths: Mutex::new(Paths::default()),
            models_found: Mutex::new(vec![]),
            restart: AtomicBool::new(false),
            retry: AtomicBool::new(false),
            clean_engine: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            cancel_model_dl: AtomicBool::new(false),
            dl_image_model: AtomicBool::new(dl_img),
            dl_video_model: AtomicBool::new(dl_vid),
            cfg: Mutex::new(cfg),
            pending_models_dir: Mutex::new(None),
            migration_status: Mutex::new(None),
            active_profile: Mutex::new(Profile::load()),
        })
    }
}



pub fn move_file_robust(from: &Path, to: &Path) -> Result<(), String> {
    if let Some(p) = to.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to).map_err(|e| format!("Copy failed: {e}"))?;
    let _ = std::fs::remove_file(from);
    Ok(())
}

pub fn migrate_storage(
    sh: Arc<Shared>,
    from_models: PathBuf,
    to_models: PathBuf,
    move_files: bool,
    ctx: eframe::egui::Context,
) {
    std::thread::spawn(move || {
        *sh.migration_status.lock().unwrap() = Some(format!("Preparing migration to {}...", to_models.display()));
        ctx.request_repaint();

        if move_files && from_models.exists() {
            let mut files_to_move = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&from_models) {
                for cat in entries.flatten() {
                    let cat_path = cat.path();
                    if cat_path.is_dir() {
                        if let Ok(files) = std::fs::read_dir(&cat_path) {
                            for f in files.flatten() {
                                let fp = f.path();
                                if fp.is_file() {
                                    let rel = fp.strip_prefix(&from_models).unwrap_or(&fp);
                                    files_to_move.push((fp.clone(), to_models.join(rel)));
                                }
                            }
                        }
                    } else if cat_path.is_file() {
                        let rel = cat_path.strip_prefix(&from_models).unwrap_or(&cat_path);
                        files_to_move.push((cat_path.clone(), to_models.join(rel)));
                    }
                }
            }

            let total = files_to_move.len();
            for (idx, (src, dst)) in files_to_move.into_iter().enumerate() {
                let name = src.file_name().and_then(|n| n.to_str()).unwrap_or("file");
                *sh.migration_status.lock().unwrap() = Some(format!("Moving file {}/{} ({name})...", idx + 1, total));
                ctx.request_repaint();
                if let Err(e) = move_file_robust(&src, &dst) {
                    *sh.migration_status.lock().unwrap() = Some(format!("Migration warning: {e}"));
                }
            }
        }

        {
            let mut cfg = sh.cfg.lock().unwrap();
            cfg.models_dir = Some(to_models.clone());
            cfg.save();
        }
        sh.paths.lock().unwrap().models = to_models.clone();
        *sh.pending_models_dir.lock().unwrap() = Some(to_models);
        *sh.migration_status.lock().unwrap() = Some("Storage migration completed successfully.".into());
        ctx.request_repaint();
    });
}

pub fn open_output_folder(sh: &Shared) {
    let p = {
        let paths = sh.paths.lock().unwrap();
        if let Some(ref c) = paths.comfy {
            c.join("output").join("ImageStudio")
        } else {
            data_dir().join("output")
        }
    };
    let _ = std::fs::create_dir_all(&p);
    let _ = Command::new("explorer.exe")
        .arg(&p)
        .creation_flags(CREATE_NO_WINDOW)
        .spawn();
}

pub fn open_models_folder(sh: &Shared) {
    let p = sh.paths.lock().unwrap().models.clone();
    let _ = std::fs::create_dir_all(&p);
    let _ = Command::new("explorer.exe")
        .arg(&p)
        .creation_flags(CREATE_NO_WINDOW)
        .spawn();
}

pub fn reset_to_clean_engine(sh: &Shared) {
    let mut cfg = sh.cfg.lock().unwrap();
    let paths = sh.paths.lock().unwrap();
    if cfg.models_dir.is_none() && paths.models.is_dir() {
        cfg.models_dir = Some(paths.models.clone());
    }
    cfg.save();
    sh.clean_engine.store(true, Ordering::SeqCst);
    log("reset to clean managed engine requested");
}

pub fn repair_environment(sh: Arc<Shared>, ctx: eframe::egui::Context) {
    std::thread::spawn(move || {
        let paths = sh.paths.lock().unwrap().clone();
        let py = match paths.python {
            Some(p) => p,
            None => {
                *sh.state.lock().unwrap() = EngState::Failed("Cannot repair: No Python executable found in detected folder.".into());
                ctx.request_repaint();
                return;
            }
        };
        let comfy = match paths.comfy {
            Some(c) => c,
            None => {
                *sh.state.lock().unwrap() = EngState::Failed("Cannot repair: No ComfyUI folder found.".into());
                ctx.request_repaint();
                return;
            }
        };

        *sh.state.lock().unwrap() = EngState::RepairingEngine {
            step: "Running pip install to fix dependencies (async-timeout, aiohttp, requirements.txt)...".into(),
        };
        ctx.request_repaint();

        log(&format!("repair: running pip via {} in {}", py.display(), comfy.display()));
        let mut cmd = Command::new(&py);
        cmd.args(["-m", "pip", "install", "async-timeout", "aiohttp", "-r", "requirements.txt"])
            .current_dir(&comfy)
            .creation_flags(CREATE_NO_WINDOW);

        match cmd.output() {
            Ok(out) => {
                if out.status.success() {
                    log("repair: pip finished successfully");
                    sh.retry.store(true, Ordering::SeqCst);
                } else {
                    let err = String::from_utf8_lossy(&out.stderr);
                    let out_txt = String::from_utf8_lossy(&out.stdout);
                    log(&format!("repair failed: {err}\n{out_txt}"));
                    *sh.state.lock().unwrap() = EngState::Failed(format!("Pip repair exited with error:\n{}", if err.is_empty() { out_txt } else { err }));
                }
            }
            Err(e) => {
                log(&format!("repair: failed to spawn pip: {e}"));
                *sh.state.lock().unwrap() = EngState::Failed(format!("Failed to execute pip: {e}"));
            }
        }
        ctx.request_repaint();
    });
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

    c.into_iter().find(|p| {
        !cfg.ignore_paths.iter().any(|ign| ign == p || p.starts_with(ign)) && p.join("main.py").exists()
    })
}

fn find_pythons(comfy: &Path) -> Vec<PathBuf> {
    let parent = comfy.parent().map(Path::to_path_buf).unwrap_or_default();
    let grand = parent.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut candidates: Vec<PathBuf> = [
        comfy.join(".venv/Scripts/python.exe"),
        comfy.join("venv/Scripts/python.exe"),
        parent.join(".venv/Scripts/python.exe"),
        parent.join("python_embeded/python.exe"),
        grand.join("python_embeded/python.exe"),
        parent.join("standalone-env/python.exe"),
    ]
    .into_iter()
    .filter(|p| p.exists())
    .collect();
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let python = dir.join("python.exe");
            if python.is_file() && !candidates.contains(&python) {
                candidates.push(python);
            }
        }
    }
    candidates
}

pub fn find_models(cfg: &Config, comfy: &Path) -> PathBuf {
    if let Some(m) = &cfg.models_dir {
        return m.clone();
    }
    if let Some(h) = dirs::home_dir() {
        let s = h.join("ComfyUI-Shared").join("models");
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
    if log.exists() {
        let prev = data_dir().join("comfyui.prev.log");
        let _ = std::fs::rename(&log, &prev);
    }
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
    .env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    .stdin(Stdio::null())
    .stdout(out)
    .stderr(err)
    .creation_flags(CREATE_NO_WINDOW);
    cmd.spawn().map_err(|e| format!("could not start {}: {e}", py.display()))
}

const DL_SEGMENTS: u64 = 8;

/// Parallel ranged download via several curl processes. Each segment streams into its own
/// `<part>.N` file and resumes independently; segment boundaries are persisted in `<part>.segs`
/// so stop/restart keeps working. Hides per-connection ramp-up/stalls (e.g. HF Xet bridge).
#[allow(clippy::too_many_arguments)]
fn download_segmented<F>(
    curl_exe: &Path,
    url: &str,
    dest: &Path,
    part_path: &Path,
    total: u64,
    mut on_progress: F,
    cancel: &AtomicBool,
    ctx: &eframe::egui::Context,
    speed_watchdog: bool,
) -> Result<(), String>
where
    F: FnMut(u64, u64, f32),
{
    use std::io::{Read, Write};
    use std::sync::atomic::AtomicU64;

    let seg_path = |i: usize| PathBuf::from(format!("{}.{i}", part_path.display()));
    let meta_path = PathBuf::from(format!("{}.segs", part_path.display()));

    // Load or create segment boundaries.
    let mut bounds: Vec<u64> = std::fs::read_to_string(&meta_path)
        .ok()
        .map(|s| s.split_whitespace().filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_default();
    if bounds.len() < 2 || bounds[0] != 0 || *bounds.last().unwrap() != total {
        // Fresh layout. Reuse a legacy single .part file as the start of segment 0.
        let existing = std::fs::metadata(part_path).map(|m| m.len()).unwrap_or(0).min(total);
        let rest = total - existing;
        let chunk = rest.div_ceil(DL_SEGMENTS).max(1);
        bounds = vec![0];
        let mut b = existing;
        while b < total {
            b = (b + chunk).min(total);
            bounds.push(b);
        }
        if bounds.len() < 2 {
            bounds.push(total);
        }
        for i in 0..bounds.len() - 1 {
            let _ = std::fs::remove_file(seg_path(i));
        }
        if existing > 0 {
            std::fs::rename(part_path, seg_path(0)).map_err(|e| format!("rename error: {e}"))?;
        }
        std::fs::write(&meta_path, bounds.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(" "))
            .map_err(|e| format!("failed to write segment map: {e}"))?;
    }
    let n = bounds.len() - 1;

    let stop = Arc::new(AtomicBool::new(false));
    let pids: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
    let mut have = Vec::with_capacity(n);
    let mut counters = Vec::with_capacity(n);
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let (lo, hi) = (bounds[i], bounds[i + 1]);
        let path = seg_path(i);
        let mut len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if len > hi - lo {
            let _ = std::fs::remove_file(&path);
            len = 0;
        }
        have.push(len);
        let counter = Arc::new(AtomicU64::new(len));
        counters.push(counter.clone());
        let (stop, pids, curl_exe, url) = (stop.clone(), pids.clone(), curl_exe.to_path_buf(), url.to_string());
        handles.push(std::thread::spawn(move || -> Result<(), String> {
            let need = hi - lo;
            let mut attempts = 0;
            loop {
                let got = counter.load(Ordering::SeqCst);
                if got >= need || stop.load(Ordering::SeqCst) {
                    return Ok(());
                }
                let mut child = Command::new(&curl_exe)
                    .args(["-L", "--fail", "-s", "-S", "--connect-timeout", "30", "--speed-limit", "1", "--speed-time", "60", "-r"])
                    .arg(format!("{}-{}", lo + got, hi - 1))
                    .arg(&url)
                    .creation_flags(CREATE_NO_WINDOW)
                    .stdin(Stdio::null())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("failed to start curl: {e}"))?;
                pids.lock().unwrap().push(child.id());
                let mut out = child.stdout.take().unwrap();
                let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&path)
                    .map_err(|e| format!("failed to open segment: {e}"))?;
                let mut buf = vec![0u8; 512 * 1024];
                let mut overflow = false;
                loop {
                    let r = match out.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(r) => r,
                    };
                    let cur = counter.load(Ordering::SeqCst);
                    if cur + r as u64 > need {
                        overflow = true; // server ignored Range
                        break;
                    }
                    if file.write_all(&buf[..r]).is_err() {
                        break;
                    }
                    counter.fetch_add(r as u64, Ordering::SeqCst);
                }
                let _ = file.flush();
                if overflow {
                    kill_tree(&mut child);
                    return Err("server does not support ranged downloads".into());
                }
                let status = child.wait().ok();
                if counter.load(Ordering::SeqCst) >= need || stop.load(Ordering::SeqCst) {
                    return Ok(());
                }
                attempts += 1;
                if attempts >= 6 {
                    let mut msg = String::new();
                    if let Some(mut e) = child.stderr.take() {
                        let _ = e.read_to_string(&mut msg);
                    }
                    return Err(format!("segment {i} failed ({status:?}): {}", msg.trim()));
                }
                std::thread::sleep(Duration::from_secs(2));
            }
        }));
    }

    let kill_all = |pids: &Mutex<Vec<u32>>| {
        for pid in pids.lock().unwrap().iter() {
            let _ = Command::new("taskkill")
                .args(["/F", "/T", "/PID", &pid.to_string()])
                .creation_flags(CREATE_NO_WINDOW)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    };

    let start_time = Instant::now();
    let mut last_done: u64 = have.iter().sum();
    let mut last_tick = Instant::now();
    let mut speed_mb = 0.0f32;
    on_progress(last_done, total, 0.0);
    ctx.request_repaint();
    let mut abort: Option<String> = None;
    while handles.iter().any(|h| !h.is_finished()) {
        if cancel.load(Ordering::SeqCst) {
            abort = Some("download cancelled".into());
            break;
        }
        std::thread::sleep(Duration::from_millis(250));
        let done: u64 = counters.iter().map(|c| c.load(Ordering::SeqCst)).sum();
        let secs = last_tick.elapsed().as_secs_f32();
        if secs > 0.0 {
            let inst = (done.saturating_sub(last_done) as f32 / 1024.0 / 1024.0) / secs;
            speed_mb = if speed_mb == 0.0 { inst } else { speed_mb * 0.7 + inst * 0.3 };
        }
        last_done = done;
        last_tick = Instant::now();
        on_progress(done, total, speed_mb);
        ctx.request_repaint();
        let el = start_time.elapsed();
        if speed_watchdog && el >= Duration::from_secs(10) && el <= Duration::from_secs(22) && speed_mb < 0.6 {
            log(&format!("Download watchdog: current stream is throttled at {speed_mb:.2} MB/s, switching to faster mirror"));
            abort = Some(format!("Download throttled at {speed_mb:.2} MB/s; switching mirror"));
            break;
        }
    }
    if abort.is_some() {
        stop.store(true, Ordering::SeqCst);
        kill_all(&pids);
    }
    let mut first_err = None;
    for h in handles {
        match h.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => { first_err.get_or_insert(e); }
            Err(_) => { first_err.get_or_insert("segment thread panicked".into()); }
        }
    }
    if let Some(e) = abort.or(first_err) {
        stop.store(true, Ordering::SeqCst);
        kill_all(&pids);
        return Err(e);
    }

    // Verify and stitch segments into segment 0, then rename to dest.
    for i in 0..n {
        let len = std::fs::metadata(seg_path(i)).map(|m| m.len()).unwrap_or(0);
        if len != bounds[i + 1] - bounds[i] {
            return Err(format!("Incomplete segment {i}: {len} of {} bytes.", bounds[i + 1] - bounds[i]));
        }
    }
    {
        let mut out = std::fs::OpenOptions::new().append(true).open(seg_path(0))
            .map_err(|e| format!("failed to open segment 0: {e}"))?;
        for i in 1..n {
            let mut inp = std::fs::File::open(seg_path(i)).map_err(|e| format!("failed to open segment {i}: {e}"))?;
            std::io::copy(&mut inp, &mut out).map_err(|e| format!("failed to join segments: {e}"))?;
        }
        out.flush().map_err(|e| e.to_string())?;
    }
    let final_len = std::fs::metadata(seg_path(0)).map(|m| m.len()).unwrap_or(0);
    if final_len != total {
        return Err(format!("Incomplete download: received {final_len} of {total} bytes."));
    }
    if dest.exists() {
        let _ = std::fs::remove_file(dest);
    }
    std::fs::rename(seg_path(0), dest).map_err(|e| format!("rename error: {e}"))?;
    for i in 1..n {
        let _ = std::fs::remove_file(seg_path(i));
    }
    let _ = std::fs::remove_file(&meta_path);
    on_progress(total, total, speed_mb);
    ctx.request_repaint();
    Ok(())
}

fn download_stream<F>(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
    expected_size: u64,
    mut on_progress: F,
    cancel: &AtomicBool,
    ctx: &eframe::egui::Context,
    speed_watchdog: bool,
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

    if expected_size > 0 && existing_len > expected_size {
        existing_len = 0;
        let _ = std::fs::remove_file(&part_path);
    }

    // Prefer Windows' bundled curl.exe: it is verified to saturate the link where the
    // in-process client plateaus (~14 Mb/s). Falls back to reqwest if curl is unavailable.
    let curl_exe = std::env::var_os("SystemRoot")
        .map(|r| PathBuf::from(r).join("System32").join("curl.exe"))
        .filter(|p| p.is_file());
    if let Some(curl_exe) = curl_exe {
        if expected_size > 64 * 1024 * 1024 {
            return download_segmented(&curl_exe, url, dest, &part_path, expected_size, on_progress, cancel, ctx, speed_watchdog);
        }
        let mut child = Command::new(&curl_exe)
            .args(["-L", "--fail", "-s", "-S", "-C", "-", "--connect-timeout", "30", "-o"])
            .arg(&part_path)
            .arg(url)
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to start curl: {e}"))?;
        let total_bytes = expected_size;
        let start_time = Instant::now();
        let mut last_len = existing_len;
        let mut last_tick = Instant::now();
        let mut speed_mb = 0.0f32;
        on_progress(existing_len, total_bytes, 0.0);
        ctx.request_repaint();
        let status = loop {
            if let Some(st) = child.try_wait().map_err(|e| e.to_string())? {
                break st;
            }
            if cancel.load(Ordering::SeqCst) {
                kill_tree(&mut child);
                return Err("download cancelled".into());
            }
            std::thread::sleep(Duration::from_millis(250));
            let len = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(last_len);
            let secs = last_tick.elapsed().as_secs_f32();
            if secs > 0.0 {
                let inst = (len.saturating_sub(last_len) as f32 / 1024.0 / 1024.0) / secs;
                speed_mb = if speed_mb == 0.0 { inst } else { speed_mb * 0.7 + inst * 0.3 };
            }
            last_len = len;
            last_tick = Instant::now();
            on_progress(len, total_bytes, speed_mb);
            ctx.request_repaint();
            let el = start_time.elapsed();
            if speed_watchdog && el >= Duration::from_secs(10) && el <= Duration::from_secs(22)
                && speed_mb < 0.6 && total_bytes > 50 * 1024 * 1024 {
                log(&format!("Download watchdog: current stream is throttled at {speed_mb:.2} MB/s, switching to faster mirror"));
                kill_tree(&mut child);
                return Err(format!("Download throttled at {speed_mb:.2} MB/s; switching mirror"));
            }
        };
        if !status.success() {
            let mut msg = String::new();
            if let Some(mut e) = child.stderr.take() {
                use std::io::Read;
                let _ = e.read_to_string(&mut msg);
            }
            return Err(format!("curl failed ({status}): {}", msg.trim()));
        }
        let len = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
        if total_bytes > 0 && len != total_bytes {
            return Err(format!("Incomplete download: received {len} of {total_bytes} bytes."));
        }
        if dest.exists() {
            let _ = std::fs::remove_file(dest);
        }
        std::fs::rename(&part_path, dest).map_err(|e| format!("rename error: {e}"))?;
        on_progress(len, len, speed_mb);
        ctx.request_repaint();
        return Ok(());
    }

    let mut req = client.get(url);
    if existing_len > 0 {
        req = req.header("Range", format!("bytes={existing_len}-"));
    }

    let mut resp = req.send().map_err(|e| format!("HTTP request failed: {e}"))?;
    let status = resp.status();

    let (file, total_bytes) = if status == reqwest::StatusCode::PARTIAL_CONTENT {
        let range_start = resp.headers().get(reqwest::header::CONTENT_RANGE)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.strip_prefix("bytes "))
            .and_then(|s| s.split_once('-'))
            .and_then(|(start, _)| start.parse::<u64>().ok());
        if range_start != Some(existing_len) {
            return Err("Server returned an unexpected download range.".into());
        }
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
        let remote_size = resp.headers().get(reqwest::header::CONTENT_RANGE)
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.strip_prefix("bytes */"))
            .and_then(|s| s.parse::<u64>().ok());
        if existing_len > 0 && remote_size == Some(existing_len)
            && (expected_size == 0 || expected_size == existing_len) {
            std::fs::rename(&part_path, dest).map_err(|e| format!("rename error: {e}"))?;
            on_progress(existing_len, existing_len, 0.0);
            return Ok(());
        }
        return Err("Download resume size does not match the server; remove the partial download and retry.".into());
    } else {
        return Err(format!("server returned HTTP status {status}"));
    };

    use std::io::{BufWriter, Read, Write};
    use std::sync::mpsc::sync_channel;

    // Decouple network reading from disk writing with a 32 MB in-memory ring buffer (64 x 512KB).
    // This guarantees the network socket read loop NEVER pauses when disk I/O, NTFS allocation,
    // or Windows Defender antivirus holds the filesystem lock.
    let (tx, rx) = sync_channel::<Vec<u8>>(64);
    let writer_thread = std::thread::spawn(move || -> Result<(), String> {
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        while let Ok(chunk) = rx.recv() {
            writer.write_all(&chunk).map_err(|e| format!("disk write error: {e}"))?;
        }
        writer.flush().map_err(|e| format!("disk flush error: {e}"))?;
        Ok(())
    });

    let mut buf = vec![0u8; 256 * 1024]; // 256 KB read buffer
    let mut downloaded = existing_len;
    let start_time = Instant::now();
    let mut last_tick = Instant::now();
    let mut bytes_since_tick = 0u64;
    let mut speed_mb = 0.0f32;

    on_progress(downloaded, total_bytes, speed_mb);
    ctx.request_repaint();

    let mut download_err: Option<String> = None;

    loop {
        if cancel.load(Ordering::SeqCst) {
            download_err = Some("download cancelled".into());
            break;
        }
        let n = match resp.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                download_err = Some(format!("download read error: {e}"));
                break;
            }
        };
        if tx.send(buf[..n].to_vec()).is_err() {
            download_err = Some("disk writer channel closed".into());
            break;
        }
        downloaded += n as u64;
        bytes_since_tick += n as u64;

        let elapsed = last_tick.elapsed();
        if elapsed >= Duration::from_millis(250) {
            let secs = elapsed.as_secs_f32();
            if secs > 0.0 {
                let inst = (bytes_since_tick as f32 / 1024.0 / 1024.0) / secs;
                speed_mb = if speed_mb == 0.0 { inst } else { speed_mb * 0.7 + inst * 0.3 };
            }
            last_tick = Instant::now();
            bytes_since_tick = 0;
            on_progress(downloaded, total_bytes, speed_mb);
            ctx.request_repaint();

            // Speed Watchdog: During the initial qualification window (10s to 22s), if sustained speed is choked under 0.6 MB/s, switch mirror.
            let elapsed_total = start_time.elapsed();
            if speed_watchdog && elapsed_total >= Duration::from_secs(10) && elapsed_total <= Duration::from_secs(22) && speed_mb < 0.6 && total_bytes > 50 * 1024 * 1024 {
                log(&format!("Download watchdog: current stream is throttled at {speed_mb:.2} MB/s, switching to faster mirror"));
                download_err = Some(format!("Download throttled at {speed_mb:.2} MB/s; switching mirror"));
                break;
            }
        }
    }

    drop(tx);
    let writer_res = writer_thread.join().map_err(|_| "disk writer thread panicked".to_string())?;
    writer_res?;

    if let Some(err) = download_err {
        return Err(err);
    }

    if total_bytes > 0 && downloaded != total_bytes {
        return Err(format!("Incomplete download: received {downloaded} of {total_bytes} bytes."));
    }

    if dest.exists() {
        let _ = std::fs::remove_file(dest);
    }
    std::fs::rename(&part_path, dest).map_err(|e| format!("rename error: {e}"))?;
    on_progress(downloaded, total_bytes, speed_mb);
    ctx.request_repaint();
    Ok(())
}

fn download_file<F>(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
    expected_size: u64,
    on_progress: F,
    cancel: &AtomicBool,
    ctx: &eframe::egui::Context,
) -> Result<(), String>
where
    F: FnMut(u64, u64, f32),
{
    download_stream(client, url, dest, expected_size, on_progress, cancel, ctx, false)
}

fn bootstrap_engine(
    sh: &Shared,
    ctx: &eframe::egui::Context,
    dl_client: &reqwest::blocking::Client,
    recovering: bool,
) -> Result<PathBuf, String> {
    let engine_dir = fresh_engine_dir(&data_dir().join("engine"))?;

    let (comfy_url, vendor_name) = get_comfy_portable_url();
    let archive_path = data_dir().join(format!("comfyui_portable_{vendor_name}.7z"));
    log(&format!("bootstrap: downloading {vendor_name} portable from {comfy_url}"));

    {
        let step_desc = if recovering {
            format!("Existing engine failed; creating an isolated {vendor_name} instance...")
        } else {
            format!("Downloading ComfyUI Portable ({vendor_name})...")
        };
        let step_clone = step_desc.clone();
        let res = download_file(
            dl_client,
            comfy_url,
            &archive_path,
            0,
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

    let extraction = sevenz_rust2::decompress_file(&archive_path, &engine_dir);
    let _ = std::fs::remove_file(&archive_path);
    extraction.map_err(|e| format!("Failed to extract ComfyUI 7z archive: {e}"))?;

    log("bootstrap: extracted successfully");

    if let Some(c) = [engine_dir.join("ComfyUI_windows_portable/ComfyUI"), engine_dir.join("ComfyUI")]
        .into_iter().find(|c| c.join("main.py").is_file()) {
        Ok(c)
    } else {
        Err("Extraction finished but main.py was not found in extracted directory.".into())
    }
}

fn fresh_engine_dir(root: &Path) -> Result<PathBuf, String> {
    std::fs::create_dir_all(root).map_err(|e| format!("Cannot create engine directory: {e}"))?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?.as_nanos();
    for n in 0..100 {
        let dir = root.join(format!("isolated-{stamp}-{}-{n}", std::process::id()));
        match std::fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("Cannot create isolated engine: {e}")),
        }
    }
    Err("Cannot allocate an isolated engine directory.".into())
}

pub fn safe_model_path(base: &Path, category: &str, filename: &str) -> PathBuf {
    let safe_cat = Path::new(category).file_name().unwrap_or_else(|| std::ffi::OsStr::new(category));
    let safe_file = Path::new(filename).file_name().unwrap_or_else(|| std::ffi::OsStr::new(filename));
    base.join(safe_cat).join(safe_file)
}

fn bootstrap_models(
    sh: &Shared,
    ctx: &eframe::egui::Context,
    dl_client: &reqwest::blocking::Client,
    models_dir: &Path,
    profile: &Profile,
) -> Result<(), (String, String)> {
    let all_profiles = Profile::all();
    let ltx_profile = all_profiles.iter().find(|p| p.id == "ltx_2.3").cloned();

    loop {
        let mut current_models_dir = if let Some(p) = sh.pending_models_dir.lock().unwrap().clone() {
            p
        } else {
            models_dir.to_path_buf()
        };

        let mut target_models: Vec<ModelFile> = Vec::new();
        if sh.dl_image_model.load(Ordering::SeqCst) {
            target_models.extend(profile.models.clone());
        }
        if sh.dl_video_model.load(Ordering::SeqCst) {
            if let Some(ref vid_p) = ltx_profile {
                target_models.extend(vid_p.models.clone());
            }
        }

        let missing: Vec<_> = target_models
            .iter()
            .filter(|m| !safe_model_path(&current_models_dir, &m.category, &m.filename).exists())
            .cloned()
            .collect();

        if missing.is_empty() {
            return Ok(());
        }

        let total_all_bytes: u64 = missing.iter().map(|m| m.size_bytes).sum();
        let mut completed_bytes: u64 = 0;
        let mut any_interrupted = false;

        for (idx, m) in missing.iter().enumerate() {
            let is_video_model = ltx_profile.as_ref().map(|vp| vp.models.iter().any(|vm| vm.filename == m.filename)).unwrap_or(false);
            if is_video_model && !sh.dl_video_model.load(Ordering::SeqCst) {
                continue;
            }
            if !is_video_model && !sh.dl_image_model.load(Ordering::SeqCst) {
                continue;
            }

            if let Some(p) = sh.pending_models_dir.lock().unwrap().clone() {
                current_models_dir = p;
            }
            let dest = safe_model_path(&current_models_dir, &m.category, &m.filename);
            if dest.exists() {
                completed_bytes += m.size_bytes;
                continue;
            }

            log(&format!("bootstrap: downloading model {} -> {}", m.filename, dest.display()));

            let cur_file = m.filename.clone();
            let cur_idx = idx + 1;
            let total_files = missing.len();
            let file_expected = m.size_bytes;

            let file_cancel = AtomicBool::new(false);
            let cancel_ref = &file_cancel;

            let hf_mirror_url = if m.hf_url.starts_with("https://huggingface.co/") {
                m.hf_url.replace("https://huggingface.co/", "https://hf-mirror.com/")
            } else {
                String::new()
            };

            // Download candidates in priority order:
            // 1. huggingface.co direct (CDN; verified >80 Mb/s with curl)
            // 2. ModelScope, 3. hf-mirror.com — fallbacks only. The mirrors are China-hosted and
            //    gave spiky ~15 Mb/s throughput that the 0.6 MB/s watchdog never flagged.
            let mut candidates = Vec::new();
            if !m.hf_url.is_empty() {
                candidates.push((m.hf_url.clone(), "HuggingFace"));
            }
            if !m.modelscope_url.is_empty() {
                candidates.push((m.modelscope_url.clone(), "ModelScope"));
            }
            if !hf_mirror_url.is_empty() {
                candidates.push((hf_mirror_url, "hf-mirror"));
            }

            let total_candidates = candidates.len();
            let mut download_result = Err("No download candidates".to_string());
            for (c_idx, (url, source_name)) in candidates.into_iter().enumerate() {
                if cancel_ref.load(Ordering::SeqCst) || sh.cancel.load(Ordering::SeqCst) || sh.cancel_model_dl.load(Ordering::SeqCst) {
                    break;
                }
                let is_last_candidate = c_idx + 1 >= total_candidates;
                // Only watchdog-abort if there is another mirror to switch to!
                let enable_watchdog = !is_last_candidate;
                log(&format!("bootstrap: attempting download for {} via {} ({url}) [watchdog={enable_watchdog}]", m.filename, source_name));
                download_result = download_stream(
                    dl_client,
                    &url,
                    &dest,
                    file_expected,
                    |done, total, speed| {
                        if sh.cancel.load(Ordering::SeqCst) || sh.cancel_model_dl.load(Ordering::SeqCst) {
                            cancel_ref.store(true, Ordering::SeqCst);
                        }
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
                    cancel_ref,
                    ctx,
                    enable_watchdog,
                );
                if download_result.is_ok() {
                    log(&format!("bootstrap: successfully downloaded {} via {}", m.filename, source_name));
                    break;
                } else {
                    log(&format!("bootstrap: download from {} failed or throttled: {:?}", source_name, download_result.as_ref().err()));
                }
            }

            if sh.cancel_model_dl.swap(false, Ordering::SeqCst) {
                log(&format!("bootstrap: model download interrupted by checkbox change for {}", m.filename));
                any_interrupted = true;
                break;
            }

            if let Err(e) = download_result {
                if sh.cancel.load(Ordering::SeqCst) {
                    return Err((m.filename.clone(), "Setup cancelled by user".into()));
                }
                return Err((m.filename.clone(), format!("Failed to download {}: {e}", m.filename)));
            }

            if let Some(target) = sh.pending_models_dir.lock().unwrap().clone() {
                if target != current_models_dir {
                    let target_dest = safe_model_path(&target, &m.category, &m.filename);
                    log(&format!("bootstrap: moving finished model file to new storage: {} -> {}", dest.display(), target_dest.display()));
                    let _ = move_file_robust(&dest, &target_dest);
                    current_models_dir = target;
                }
            }

            completed_bytes += file_expected;
        }

        if !any_interrupted {
            break;
        }
    }

    Ok(())
}


pub fn start_supervisor(sh: Arc<Shared>, _profile: Profile, ctx: eframe::egui::Context) {
    std::thread::spawn(move || {
        let ping_client = reqwest::blocking::Client::builder().timeout(Duration::from_millis(1200)).build().unwrap();
        let dl_client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(7200))
            .connect_timeout(Duration::from_secs(30))
            .tcp_nodelay(true)
            .no_proxy()
            .http1_only()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
            .build()
            .unwrap();

        let mut child: Option<Child> = None;
        let mut attempt = 0usize;
        let mut started = Instant::now();
        let mut failed = false;
        let mut isolated_attempted = false;
        let mut ever_ready = false;
        let mut consecutive_crashes = 0usize;

        loop {
            let profile = sh.active_profile.lock().unwrap().clone();
            let clean_requested = sh.clean_engine.swap(false, Ordering::SeqCst);
            let retry_requested = sh.retry.swap(false, Ordering::SeqCst);
            let restart_requested = sh.restart.swap(false, Ordering::SeqCst);
            if clean_requested || retry_requested || restart_requested {
                if let Some(c) = child.as_mut() {
                    kill_tree(c);
                }
                child = None;
                attempt = 0;
                consecutive_crashes = 0;
                failed = false;
                sh.cancel.store(false, Ordering::SeqCst);
                log("retry/restart supervisor cycle");
            }

            let mut cfg = sh.cfg.lock().unwrap().clone();
            let mut comfy = if clean_requested { None } else { find_comfy(&cfg) };

            // 1. Is ComfyUI already running and responding?
            if let Some(gpu) = (!clean_requested).then(|| ping(&ping_client)).flatten() {
                attempt = 0;
                failed = false;
                ever_ready = true;
                consecutive_crashes = 0;
                *sh.state.lock().unwrap() = EngState::Ready { ours: child.is_some(), gpu };
                let models = comfy.as_ref().map(|c| find_models(&cfg, c)).unwrap_or_default();
                let pys = comfy.as_ref().map(|c| find_pythons(c)).unwrap_or_default();
                if child.is_none() {
                    *sh.paths.lock().unwrap() = Paths { comfy: comfy.clone(), python: pys.first().cloned(), models: models.clone() };
                }
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
                isolated_attempted = true;
                match bootstrap_engine(&sh, &ctx, &dl_client, clean_requested) {
                    Ok(c) => {
                        comfy = Some(c.clone());
                        cfg.comfy_dir = Some(c);
                        cfg.save();
                        *sh.cfg.lock().unwrap() = cfg.clone();
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
            if child.is_none() {
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
                Err((file, e)) => {
                    failed = true;
                    *sh.state.lock().unwrap() = EngState::ModelDownloadFailed {
                        current_file: file,
                        error: e,
                    };
                    ctx.request_repaint();
                    std::thread::sleep(Duration::from_millis(600));
                    continue;
                }
            }

            // 4. Launch ComfyUI process
            if child.is_none() {
                let pys = find_pythons(cd);
                if attempt >= pys.len() {
                    let t = tail(&data_dir().join("comfyui.log"), 14);
                    let is_managed = cd.starts_with(data_dir());
                    if !isolated_attempted && !is_managed {
                        isolated_attempted = true;
                        log(&format!("all Python candidates failed in {}; creating isolated engine\n{t}", cd.display()));
                        match bootstrap_engine(&sh, &ctx, &dl_client, true) {
                            Ok(c) => {
                                cfg.comfy_dir = Some(c);
                                if cfg.models_dir.is_none() {
                                    cfg.models_dir = Some(models.clone());
                                }
                                cfg.save();
                                *sh.cfg.lock().unwrap() = cfg;
                                attempt = 0;
                            }
                            Err(e) => {
                                failed = true;
                                *sh.state.lock().unwrap() = EngState::Failed(format!("Existing engine failed and isolated setup failed: {e}"));
                            }
                        }
                    } else {
                        failed = true;
                        *sh.state.lock().unwrap() = EngState::Failed(format!("ComfyUI failed to start.\n{t}"));
                    }
                } else {
                    let py = &pys[attempt];
                    sh.paths.lock().unwrap().python = Some(py.clone());
                    log(&format!("launching {} (attempt {attempt})", py.display()));
                    match spawn(py, cd, &models) {
                        Ok(c) => {
                            child = Some(c);
                            started = Instant::now();
                            *sh.state.lock().unwrap() = EngState::Starting { since: started, attempt };
                        }
                        Err(e) => {
                            log(&e);
                            attempt += 1;
                        }
                    }
                }
            } else {
                if let Some(c) = child.as_mut() {
                    let exited = matches!(c.try_wait(), Ok(Some(_)));
                    if exited || started.elapsed() > Duration::from_secs(180) {
                        log(&format!("comfyui attempt {attempt} ended (exited={exited}, ever_ready={ever_ready})"));
                        if !exited {
                            kill_tree(c);
                        }
                        child = None;
                        if ever_ready {
                            consecutive_crashes += 1;
                            log(&format!("comfyui process exited after reaching Ready (crash #{consecutive_crashes}/5) - relaunching existing engine"));
                            if consecutive_crashes <= 5 {
                                std::thread::sleep(Duration::from_millis(1000));
                                continue;
                            } else {
                                failed = true;
                                let t = tail(&data_dir().join("comfyui.log"), 16);
                                *sh.state.lock().unwrap() = EngState::Failed(format!("ComfyUI crashed repeatedly (likely CUDA Out Of Memory or driver reset).\n{t}"));
                                continue;
                            }
                        } else {
                            attempt += 1;
                        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn sandbox() -> PathBuf {
        fresh_engine_dir(&std::env::temp_dir().join("image-studio-tests")).unwrap()
    }

    fn serve(response: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/archive", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut request = [0; 4096];
            socket.read(&mut request).unwrap();
            socket.write_all(response.as_bytes()).unwrap();
        });
        url
    }

    fn download(url: &str, dest: &Path, expected: u64) -> Result<(), String> {
        let client = reqwest::blocking::Client::builder().no_proxy()
            .timeout(Duration::from_secs(5)).build().unwrap();
        download_file(&client, url, dest, expected, |_, _, _| {},
            &AtomicBool::new(false), &eframe::egui::Context::default())
    }

    #[test]
    fn isolated_install_does_not_reuse_existing_directory() {
        let root = sandbox();
        let old = root.join("ComfyUI");
        std::fs::create_dir(&old).unwrap();
        std::fs::write(old.join("main.py"), "existing install").unwrap();
        let first = fresh_engine_dir(&root).unwrap();
        let second = fresh_engine_dir(&root).unwrap();
        assert_ne!(first, second);
        assert_ne!(first, old);
        assert!(first.starts_with(&root));
        assert_eq!(std::fs::read_to_string(old.join("main.py")).unwrap(), "existing install");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incomplete_download_preserves_existing_file() {
        let dir = sandbox();
        let dest = dir.join("engine.7z");
        std::fs::write(&dest, "existing").unwrap();
        let url = serve("HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\nabc");
        assert!(download(&url, &dest, 10).is_err());
        assert_eq!(std::fs::read_to_string(&dest).unwrap(), "existing");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn server_ignoring_resume_restarts_download() {
        let dir = sandbox();
        let dest = dir.join("engine.7z");
        std::fs::write(dest.with_extension("7z.part"), "old").unwrap();
        let url = serve("HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nabcdef");
        download(&url, &dest, 6).unwrap();
        assert_eq!(std::fs::read_to_string(&dest).unwrap(), "abcdef");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn partial_download_resumes_at_correct_offset() {
        let dir = sandbox();
        let dest = dir.join("engine.7z");
        std::fs::write(dest.with_extension("7z.part"), "abc").unwrap();
        let url = serve("HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 3-5/6\r\nConnection: close\r\n\r\ndef");
        download(&url, &dest, 6).unwrap();
        assert_eq!(std::fs::read_to_string(&dest).unwrap(), "abcdef");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn mismatched_resume_range_is_not_appended() {
        let dir = sandbox();
        let dest = dir.join("engine.7z");
        let partial = dest.with_extension("7z.part");
        std::fs::write(&partial, "abc").unwrap();
        let url = serve("HTTP/1.1 206 Partial Content\r\nContent-Length: 3\r\nContent-Range: bytes 0-2/6\r\nConnection: close\r\n\r\ndef");
        assert!(download(&url, &dest, 6).is_err());
        assert_eq!(std::fs::read_to_string(&partial).unwrap(), "abc");
        assert!(!dest.exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unsatisfiable_range_does_not_promote_wrong_sized_file() {
        let dir = sandbox();
        let dest = dir.join("engine.7z");
        let url = serve("HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nContent-Range: bytes */6\r\nConnection: close\r\n\r\n");
        assert!(download(&url, &dest, 6).is_err());
        assert!(!dest.exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

