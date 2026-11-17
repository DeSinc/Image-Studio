//! Finds, downloads (if needed), launches (hidden) and supervises ComfyUI.
//! Runs on its own thread and never touches the UI thread; the UI only reads `Shared`.
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(not(windows))]
pub trait CommandExtCompat {
    fn creation_flags(&mut self, flags: u32) -> &mut Self;
}

#[cfg(not(windows))]
impl CommandExtCompat for Command {
    fn creation_flags(&mut self, _flags: u32) -> &mut Self {
        self
    }
}

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

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
pub struct PersistedProfileDownload {
    pub profile_id: String,
    #[serde(default)]
    pub paused: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Config {
    pub comfy_dir: Option<PathBuf>,
    pub models_dir: Option<PathBuf>,
    #[serde(default)]
    pub python_path: Option<PathBuf>,
    #[serde(default)]
    pub ignore_paths: Vec<PathBuf>,
    #[serde(default)]
    pub active_profile: Option<String>,
    #[serde(default)]
    pub active_video_profile: Option<String>,
    #[serde(default = "default_true")]
    pub dl_image_model: bool,
    #[serde(default)]
    pub dl_video_model: bool,
    #[serde(default)]
    pub dl_r2v_model: bool,
    #[serde(default)]
    pub allow_multi_video_loras: bool,
    #[serde(default)]
    pub auto_switch_mode_on_input: bool,
    #[serde(default)]
    pub remember_last_mode: bool,
    #[serde(default)]
    pub last_studio_mode: Option<String>,
    #[serde(default = "default_true")]
    pub keep_last_project_loaded: bool,
    #[serde(default)]
    pub pending_profile_downloads: Vec<PersistedProfileDownload>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            comfy_dir: None,
            models_dir: None,
            python_path: None,
            ignore_paths: Vec::new(),
            active_profile: None,
            active_video_profile: None,
            dl_image_model: true,
            dl_video_model: false,
            dl_r2v_model: false,
            allow_multi_video_loras: false,
            auto_switch_mode_on_input: false,
            remember_last_mode: false,
            last_studio_mode: None,
            keep_last_project_loaded: true,
            pending_profile_downloads: Vec::new(),
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
fn deserialize_null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    T: Default + serde::Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    let opt = Option::deserialize(deserializer)?;
    Ok(opt.unwrap_or_default())
}

#[derive(Deserialize, Clone)]
pub struct ModelFile {
    pub category: String,
    pub filename: String,
    #[serde(default)]
    pub size_bytes: u64,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub hf_url: String,
    #[serde(default, deserialize_with = "deserialize_null_default")]
    pub modelscope_url: String,
    #[serde(default)]
    pub optional: bool,
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
    pub fn is_video_profile_id(id: &str) -> bool {
        id.starts_with("minimax_h3") || id.starts_with("ltx_")
    }

    pub fn all_raw() -> Vec<Profile> {
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

    pub fn all() -> Vec<Profile> {
        Self::all_raw()
            .into_iter()
            .filter(|p| !Self::is_video_profile_id(&p.id))
            .collect()
    }

    pub fn all_video_profiles() -> Vec<Profile> {
        Self::all_raw()
            .into_iter()
            .filter(|p| Self::is_video_profile_id(&p.id))
            .collect()
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

    pub fn load_video_with_cfg(cfg: &Config) -> Profile {
        let profiles = Self::all_video_profiles();
        // 1. Explicit user choice in config.json
        if let Some(ref aid) = cfg.active_video_profile {
            if let Some(p) = profiles.iter().find(|p| &p.id == aid) {
                return p.clone();
            }
        }

        // 2. Preserve existing MiniMax H3 or LTX-2.5 models if already downloaded on disk
        let candidates = candidate_models_dirs(cfg);
        for p in &profiles {
            if p.id == "ltx_2.3" {
                continue;
            }
            if let Some(primary) = p.models.iter().find(|m| m.category == "diffusion_models") {
                if candidates.iter().any(|d| safe_model_path(d, &primary.category, &primary.filename).is_file()) {
                    log(&format!("Video Profile: detected existing {} on disk; preserving {}", primary.filename, p.id));
                    return p.clone();
                }
            }
        }

        // 3. Auto-detect GPU VRAM and choose optimal default video quantization
        // >= 18GB: INT8 ConvRot (listed as 22-24GB+); >= 15GB: Q6_K; >= 13GB: Q4_K_M;
        // >= 11GB (12GB): UD-Q3_K_XL; 9-10GB: Q3_K_M; < 9GB: LTX-2.5 UD-Q2_K_XL
        let target_id = match detect_gpu_vram_gb() {
            Some(vram) if vram >= 18 => "minimax_h3_24gb",
            Some(vram) if vram >= 15 => "minimax_h3_16gb",
            Some(vram) if vram >= 13 => "minimax_h3_12gb",
            Some(vram) if vram >= 11 => "minimax_h3_ud_q3_k_xl",
            Some(vram) if vram >= 9 => "minimax_h3_10gb",
            Some(_) => "ltx_2.5_q2_k_xl",
            None => "minimax_h3_ud_q3_k_xl",
        };
        if let Some(p) = profiles.iter().find(|p| p.id == target_id) {
            log(&format!("Video Profile: auto-selected {target_id} for detected GPU VRAM"));
            return p.clone();
        }
        profiles.first().cloned().expect("at least one video profile")
    }

    pub fn load() -> Profile {
        let cfg = Config::load();
        Self::load_with_cfg(&cfg)
    }

    pub fn file(&self, cat: &str) -> String {
        self.models.iter().find(|m| m.category == cat).map(|m| m.filename.clone()).unwrap_or_default()
    }

    pub fn total_size_gb(&self) -> f64 {
        self.models.iter().filter(|m| !m.optional).map(|m| m.size_bytes).sum::<u64>() as f64 / 1_000_000_000.0
    }

    pub fn is_minimax_h3(&self) -> bool {
        self.id.starts_with("minimax_h3")
    }

    pub fn is_ltx25(&self) -> bool {
        self.id.starts_with("ltx_2.5")
    }

    pub fn fl2va_unet_name(&self) -> String {
        if self.is_minimax_h3() {
            if let Some(m) = self.models.iter().find(|m| {
                m.category == "diffusion_models" && m.filename.to_ascii_lowercase().contains("fl2va")
            }) {
                return m.filename.clone();
            }
        }
        let d = self.file("diffusion_models");
        if !d.is_empty() { d } else { self.file("checkpoints") }
    }

    pub fn ref2va_unet_name(&self) -> String {
        self.models
            .iter()
            .find(|m| m.category == "diffusion_models" && m.filename.to_ascii_lowercase().contains("ref2va"))
            .map(|m| m.filename.clone())
            .unwrap_or_default()
    }

    pub fn clip_name(&self) -> String {
        self.file("text_encoders")
    }

    pub fn video_vae_name(&self) -> String {
        self.models
            .iter()
            .find(|m| m.category == "vae" && m.filename.to_ascii_lowercase().contains("video"))
            .map(|m| m.filename.clone())
            .unwrap_or_else(|| self.file("vae"))
    }

    pub fn audio_vae_name(&self) -> String {
        self.models
            .iter()
            .find(|m| m.category == "vae" && m.filename.to_ascii_lowercase().contains("audio"))
            .map(|m| m.filename.clone())
            .unwrap_or_default()
    }

    pub fn upscaler_name(&self) -> String {
        self.file("latent_upscale_models")
    }

    pub fn r2v_model_file(&self) -> Option<ModelFile> {
        self.models
            .iter()
            .find(|m| m.optional && m.category == "diffusion_models")
            .cloned()
    }

    pub fn has_r2v_model(&self, models_dir: &Path) -> bool {
        self.r2v_model_file()
            .map(|m| safe_model_path(models_dir, &m.category, &m.filename).is_file())
            .unwrap_or(false)
    }

    pub fn missing_in_dir(&self, models_dir: &Path) -> Vec<ModelFile> {
        self.models
            .iter()
            .filter(|m| !m.optional && !safe_model_path(models_dir, &m.category, &m.filename).exists())
            .cloned()
            .collect()
    }

    /// Returns the best downloaded video profile whose files are all present on disk in `models_dir`.
    pub fn find_downloaded_video_profile(models_dir: &Path) -> Option<Profile> {
        Self::find_best_downloaded_video_profile(models_dir, false, None)
    }

    /// Picks the highest-quality downloaded video profile in `models_dir`.
    /// - If `preferred_id` is Some and that profile is downloaded (and matches `ltx_only`), returns it.
    /// - Otherwise picks the highest `recommended_vram_gb` that fits detected GPU VRAM (or smallest if none fit).
    pub fn find_best_downloaded_video_profile(
        models_dir: &Path,
        ltx_only: bool,
        preferred_id: Option<&str>,
    ) -> Option<Profile> {
        let downloaded: Vec<Profile> = Self::all_video_profiles()
            .into_iter()
            .filter(|p| (!ltx_only || !p.is_minimax_h3()) && p.missing_in_dir(models_dir).is_empty())
            .collect();
        if downloaded.is_empty() {
            return None;
        }
        if let Some(pref) = preferred_id {
            if let Some(p) = downloaded.iter().find(|p| p.id == pref) {
                return Some(p.clone());
            }
        }
        Self::pick_best_from_downloaded(&downloaded, detect_gpu_vram_gb())
    }

    pub fn find_best_downloaded_h3_profile(models_dir: &Path) -> Option<Profile> {
        let downloaded: Vec<Profile> = Self::all_video_profiles()
            .into_iter()
            .filter(|p| p.is_minimax_h3() && p.missing_in_dir(models_dir).is_empty())
            .collect();
        Self::pick_best_from_downloaded(&downloaded, detect_gpu_vram_gb())
    }

    pub fn pick_best_from_downloaded(downloaded: &[Profile], vram: Option<u32>) -> Option<Profile> {
        if downloaded.is_empty() {
            return None;
        }
        let modern: Vec<&Profile> = downloaded
            .iter()
            .filter(|p| p.is_minimax_h3() || p.is_ltx25())
            .collect();
        let pool: Vec<&Profile> = if modern.is_empty() {
            downloaded.iter().collect()
        } else {
            modern
        };
        if let Some(gpu_vram) = vram {
            if let Some(best_fit) = pool
                .iter()
                .filter(|p| p.recommended_vram_gb <= gpu_vram)
                .max_by_key(|p| (p.recommended_vram_gb, (p.total_size_gb() * 1000.0) as u64))
            {
                return Some((*best_fit).clone());
            }
            return pool
                .iter()
                .min_by_key(|p| (p.recommended_vram_gb, (p.total_size_gb() * 1000.0) as u64))
                .map(|p| (*p).clone());
        }
        pool.iter()
            .max_by_key(|p| (p.recommended_vram_gb, (p.total_size_gb() * 1000.0) as u64))
            .map(|p| (*p).clone())
    }

    pub fn target_download_models(sh: &Shared) -> Vec<ModelFile> {
        let profile = sh.active_profile.lock().unwrap().clone();
        let vid_profile = sh.active_video_profile.lock().unwrap().clone();

        let mut target_models: Vec<ModelFile> = Vec::new();
        if sh.dl_image_model.load(Ordering::SeqCst) {
            target_models.extend(profile.models.iter().filter(|m| !m.optional).cloned());
        }
        if sh.dl_video_model.load(Ordering::SeqCst) {
            target_models.extend(vid_profile.models.iter().filter(|m| !m.optional).cloned());
        }
        // Optional models (e.g. MiniMax H3 Ref2VA) are only queued if explicitly requested,
        // and always placed very last after all required image/video weights.
        if sh.dl_r2v_model.load(Ordering::SeqCst) {
            target_models.extend(vid_profile.models.iter().filter(|m| m.optional).cloned());
        }
        target_models
    }

    pub fn missing_models(sh: &Shared) -> Vec<ModelFile> {
        let models_dir = sh.paths.lock().unwrap().models.clone();
        Self::target_download_models(sh)
            .into_iter()
            .filter(|m| !safe_model_path(&models_dir, &m.category, &m.filename).exists())
            .collect()
    }
}

// ---------------------------------------------------------------- state
#[derive(Clone)]
pub enum EngState {
    Detecting,
    AwaitingDownloadConfirmation {
        metered: bool,
    },
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
    ModelDownloadPaused {
        current_file: String,
        file_index: usize,
        total_files: usize,
        file_progress: f32,
        bytes_done: u64,
        bytes_total: u64,
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

#[derive(Clone, Debug)]
pub struct ModelDownloadProgress {
    pub current_file: String,
    pub file_index: usize,
    pub total_files: usize,
    pub file_progress: f32,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub speed_mb: f32,
    pub overall_progress: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProfileDownloadStatus {
    Downloading,
    Paused,
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct ProfileDownloadInfo {
    pub profile_name: String,
    pub status: ProfileDownloadStatus,
    pub progress: f32,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub speed_mb: f32,
    pub current_file: String,
}

pub struct ProfileDownloadTask {
    pub profile: Profile,
    pub status: ProfileDownloadStatus,
    pub progress: f32,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub speed_mb: f32,
    pub current_file: String,
    pub pause_flag: Arc<AtomicBool>,
    pub cancel_flag: Arc<AtomicBool>,
    pub stop_flag: Arc<AtomicBool>,
    pub running: Arc<AtomicBool>,
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
    pub model_dl_progress: Mutex<Option<ModelDownloadProgress>>,
    pub model_dl_running: AtomicBool,
    pub restart: AtomicBool,
    pub retry: AtomicBool,
    pub clean_engine: AtomicBool,
    pub cancel: AtomicBool,
    pub cancel_model_dl: AtomicBool,
    pub download_confirmed: AtomicBool,
    pub download_paused: AtomicBool,
    pub metered_allowed: AtomicBool,
    pub metered_switch_detected: AtomicBool,
    pub dl_image_model: AtomicBool,
    pub dl_video_model: AtomicBool,
    pub dl_r2v_model: AtomicBool,
    pub cfg: Mutex<Config>,
    pub pending_models_dir: Mutex<Option<PathBuf>>,
    pub migration_status: Mutex<Option<String>>,
    pub active_profile: Mutex<Profile>,
    pub active_video_profile: Mutex<Profile>,
    pub profile_downloads: Mutex<HashMap<String, ProfileDownloadTask>>,
    pub lora_downloads: Mutex<HashMap<String, crate::lora::LoraDownloadStatus>>,
    pub active_file_downloads: Mutex<HashSet<PathBuf>>,
    pub restored_profile_downloads: AtomicBool,
}

impl Shared {
    pub fn new() -> Arc<Shared> {
        let cfg = Config::load();
        let dl_img = cfg.dl_image_model;
        let dl_vid = cfg.dl_video_model;
        let dl_r2v = cfg.dl_r2v_model;
        let img_profile = Profile::load_with_cfg(&cfg);
        let vid_profile = Profile::load_video_with_cfg(&cfg);
        Arc::new(Shared {
            state: Mutex::new(EngState::Detecting),
            paths: Mutex::new(Paths::default()),
            models_found: Mutex::new(vec![]),
            model_dl_progress: Mutex::new(None),
            model_dl_running: AtomicBool::new(false),
            restart: AtomicBool::new(false),
            retry: AtomicBool::new(false),
            clean_engine: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            cancel_model_dl: AtomicBool::new(false),
            download_confirmed: AtomicBool::new(false),
            download_paused: AtomicBool::new(false),
            metered_allowed: AtomicBool::new(false),
            metered_switch_detected: AtomicBool::new(false),
            dl_image_model: AtomicBool::new(dl_img),
            dl_video_model: AtomicBool::new(dl_vid),
            dl_r2v_model: AtomicBool::new(dl_r2v),
            cfg: Mutex::new(cfg),
            pending_models_dir: Mutex::new(None),
            migration_status: Mutex::new(None),
            active_profile: Mutex::new(img_profile),
            active_video_profile: Mutex::new(vid_profile),
            profile_downloads: Mutex::new(HashMap::new()),
            lora_downloads: Mutex::new(HashMap::new()),
            active_file_downloads: Mutex::new(HashSet::new()),
            restored_profile_downloads: AtomicBool::new(false),
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

pub fn open_path(p: &Path) {
    #[cfg(windows)]
    {
        let _ = Command::new("explorer.exe")
            .arg(p)
            .creation_flags(CREATE_NO_WINDOW)
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(p).spawn();
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = Command::new("xdg-open").arg(p).spawn();
    }
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
    open_path(&p);
}

pub fn open_models_folder(sh: &Shared) {
    let p = sh.paths.lock().unwrap().models.clone();
    let _ = std::fs::create_dir_all(&p);
    open_path(&p);
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
        let comfy = match paths.comfy.or_else(|| {
            let cfg = sh.cfg.lock().unwrap();
            find_comfy(&cfg)
        }) {
            Some(c) => c,
            None => {
                *sh.state.lock().unwrap() = EngState::Failed("Cannot repair: No ComfyUI folder found.".into());
                ctx.request_repaint();
                return;
            }
        };
        let py = match paths.python.or_else(|| {
            let cfg = sh.cfg.lock().unwrap();
            find_pythons(&cfg, &comfy).into_iter().next()
        }) {
            Some(p) => p,
            None => {
                *sh.state.lock().unwrap() = EngState::Failed("Cannot repair: No Python executable found in detected folder.".into());
                ctx.request_repaint();
                return;
            }
        };

        {
            let cfg = sh.cfg.lock().unwrap().clone();
            let models_dir = find_models(&cfg, &comfy);
            let mut p = sh.paths.lock().unwrap();
            if p.models.as_os_str().is_empty() {
                p.models = models_dir;
            }
        }
        spawn_model_downloads_if_needed(&sh, &ctx);

        match install_or_repair_dependencies(&py, &comfy, &sh, &ctx) {
            Ok(()) => {
                log("repair: dependencies installed successfully; restarting supervisor");
                sh.retry.store(true, Ordering::SeqCst);
            }
            Err(e) => {
                log(&format!("repair: failed to install dependencies: {e}"));
                *sh.state.lock().unwrap() = EngState::Failed(format!("Environment repair failed:\n{e}"));
            }
        }
        ctx.request_repaint();
    });
}

pub fn install_or_repair_dependencies(
    py: &Path,
    comfy: &Path,
    sh: &Shared,
    ctx: &eframe::egui::Context,
) -> Result<(), String> {
    let (_, vendor_name) = get_comfy_portable_url();
    log(&format!("repair: inspecting/repairing dependencies for {vendor_name} via {} in {}", py.display(), comfy.display()));

    *sh.state.lock().unwrap() = EngState::RepairingEngine {
        step: "Configuring pip package manager in virtual environment...".into(),
    };
    ctx.request_repaint();

    // Ensure pip exists and is functional
    let _ = Command::new(py)
        .args(["-m", "ensurepip", "--default-pip"])
        .current_dir(comfy)
        .creation_flags(CREATE_NO_WINDOW)
        .status();

    let _ = Command::new(py)
        .args(["-m", "pip", "install", "--upgrade", "pip"])
        .current_dir(comfy)
        .creation_flags(CREATE_NO_WINDOW)
        .status();

    // Check if PyTorch can be imported
    let torch_probe = Command::new(py)
        .args(["-c", "import torch"])
        .current_dir(comfy)
        .creation_flags(CREATE_NO_WINDOW)
        .output();
    let torch_ok = torch_probe.map(|o| o.status.success()).unwrap_or(false);

    if !torch_ok {
        let (torch_args, torch_desc) = match vendor_name {
            "AMD" => (
                vec!["install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/rocm6.2"],
                "PyTorch (ROCm 6.2)",
            ),
            "Intel" => (
                vec!["install", "torch", "torchvision", "torchaudio", "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
                "PyTorch (Intel / CPU)",
            ),
            _ => (
                vec!["install", "torch", "torchvision", "torchaudio", "--extra-index-url", "https://download.pytorch.org/whl/cu124"],
                "PyTorch (CUDA 12.4)",
            ),
        };

        let base_step = format!("Installing {torch_desc} (may take several min)");
        log(&format!("repair: installing {torch_desc} via pip"));

        let (ok, out_txt, err) = run_pip_with_progress(py, comfy, &torch_args, &base_step, sh, ctx)
            .map_err(|e| format!("Failed to spawn pip for PyTorch: {e}"))?;
        if !ok {
            log(&format!("repair: PyTorch installation warning/error:\n{err}\n{out_txt}"));
        } else {
            log(&format!("repair: {torch_desc} installed successfully"));
        }
    }

    // Install ComfyUI requirements from PyPI
    log("repair: installing ComfyUI requirements and comfy-aimdo via pip");
    let req_args = ["install", "-r", "requirements.txt", "comfy-aimdo", "aiohttp", "async-timeout"];
    let (ok, out_txt, err) = run_pip_with_progress(
        py,
        comfy,
        &req_args,
        "Installing ComfyUI packages & comfy-aimdo",
        sh,
        ctx,
    )
    .map_err(|e| format!("Failed to spawn pip for requirements: {e}"))?;
    if !ok {
        log(&format!("repair: pip requirements warning/error:\n{err}\n{out_txt}"));
    } else {
        log("repair: requirements installed successfully");
    }

    // Explicitly ensure comfy-aimdo is installed & upgraded
    let aimdo_probe = Command::new(py)
        .args(["-c", "import comfy_aimdo"])
        .current_dir(comfy)
        .creation_flags(CREATE_NO_WINDOW)
        .output();
    let aimdo_ok = aimdo_probe.map(|o| o.status.success()).unwrap_or(false);

    if !aimdo_ok {
        log("repair: installing comfy-aimdo directly via pip");
        let aimdo_args = ["install", "--upgrade", "comfy-aimdo"];
        let (ok, out_txt, err) = run_pip_with_progress(
            py,
            comfy,
            &aimdo_args,
            "Installing comfy-aimdo VRAM manager",
            sh,
            ctx,
        )
        .map_err(|e| format!("Failed to install comfy-aimdo: {e}"))?;
        if !ok {
            log(&format!("repair: comfy-aimdo install failed:\n{err}\n{out_txt}"));
            return Err(format!("Failed to install comfy-aimdo:\n{}", if !err.is_empty() { err } else { out_txt }));
        }
    }

    ensure_gguf_custom_node(comfy, Some(py));
    ensure_viggle_remake_custom_node(comfy);

    // Verify both comfy_aimdo and torch are working
    let verify = Command::new(py)
        .args(["-c", "import comfy_aimdo; import torch"])
        .current_dir(comfy)
        .creation_flags(CREATE_NO_WINDOW)
        .output();

    if let Ok(o) = verify {
        if o.status.success() {
            log("repair: dependencies verified successfully (comfy_aimdo and torch ready)");
            Ok(())
        } else {
            let err = String::from_utf8_lossy(&o.stderr);
            log(&format!("repair: dependency verification failed: {err}"));
            Err(format!("Python dependency verification failed:\n{err}"))
        }
    } else {
        Err("Failed to execute verification probe.".into())
    }
}

/// Writes the LTX-2.5 transformer metadata sidecar into `models/diffusion_models/`
/// so `ComfyUI-GGUF-Loader` detects the 48-layer, 9-parameter AdaLN architecture
/// without leaking onto MiniMax H3 or other diffusion GGUFs.
pub fn ensure_ltx25_gguf_sidecar(models_dir: &Path) {
    if models_dir.as_os_str().is_empty() {
        return;
    }
    let diff_dir = models_dir.join("diffusion_models");
    let _ = std::fs::create_dir_all(&diff_dir);
    let sidecar_path = diff_dir.join("ltx-2.5-transformer-metadata.json");
    if sidecar_path.is_file() {
        if let Ok(existing) = std::fs::read_to_string(&sidecar_path) {
            if existing.contains("\"applies_to\"")
                && existing.contains("\"rope_type\"")
                && existing.contains("\"frequencies_precision\"")
                && existing.contains("\"causal_temporal_positioning\"")
                && existing.contains("\"use_middle_indices_grid\"")
            {
                return;
            }
        }
    }
    let sidecar_raw = r#"{
  "applies_to": ["*ltx*2.5*.gguf", "*ltx25*.gguf"],
  "config": {
    "transformer": {
      "_class_name": "AVTransformer3DModel",
      "activation_fn": "gelu-approximate",
      "attention_bias": true,
      "attention_head_dim": 128,
      "attention_type": "default",
      "caption_channels": 3840,
      "cross_attention_dim": 4096,
      "double_self_attention": false,
      "dropout": 0.0,
      "in_channels": 128,
      "norm_elementwise_affine": false,
      "norm_eps": 1e-06,
      "norm_num_groups": 32,
      "num_attention_heads": 32,
      "num_embeds_ada_norm": 1000,
      "num_layers": 48,
      "num_vector_embeds": null,
      "only_cross_attention": false,
      "cross_attention_norm": true,
      "out_channels": 128,
      "upcast_attention": false,
      "use_linear_projection": false,
      "qk_norm": "rms_norm",
      "standardization_norm": "rms_norm",
      "positional_embedding_type": "rope",
      "positional_embedding_theta": 10000.0,
      "positional_embedding_max_pos": [20, 2048, 2048],
      "timestep_scale_multiplier": 1000,
      "av_ca_timestep_scale_multiplier": 1000.0,
      "causal_temporal_positioning": true,
      "audio_num_attention_heads": 32,
      "audio_attention_head_dim": 64,
      "use_audio_video_cross_attention": true,
      "ff_bias": false,
      "audio_ff_bias": false,
      "share_ff": false,
      "audio_out_channels": 128,
      "audio_cross_attention_dim": 2048,
      "audio_positional_embedding_max_pos": [20],
      "av_cross_ada_norm": true,
      "use_embeddings_connector": true,
      "connector_attention_head_dim": 128,
      "connector_num_attention_heads": 32,
      "connector_num_layers": 8,
      "connector_positional_embedding_max_pos": [4096],
      "connector_num_learnable_registers": 128,
      "connector_norm_output": true,
      "use_middle_indices_grid": true,
      "apply_gated_attention": true,
      "connector_apply_gated_attention": true,
      "caption_projection_first_linear": false,
      "caption_projection_second_linear": false,
      "caption_proj_input_norm": false,
      "connector_learnable_registers_std": 1,
      "caption_proj_before_connector": true,
      "audio_connector_attention_head_dim": 64,
      "audio_connector_num_attention_heads": 32,
      "audio_connector_num_layers": 8,
      "cross_attention_adaln": true,
      "rope_type": "split",
      "frequencies_precision": "float64",
      "text_encoder_norm_type": "PER_TOKEN_RMS",
      "use_keyframes_abs_pos_embedding": true
    }
  }
}"#;
    if let Ok(sidecar_json) = serde_json::from_str::<serde_json::Value>(sidecar_raw) {
        if let Ok(pretty) = serde_json::to_string_pretty(&sidecar_json) {
            if let Err(e) = std::fs::write(&sidecar_path, pretty) {
                log(&format!("gguf sidecar: failed writing {}: {e}", sidecar_path.display()));
            } else {
                log(&format!("gguf sidecar: wrote {}", sidecar_path.display()));
            }
        }
    }
}

/// Ensures `Viggle/Viggle-Remake` ComfyUI custom nodes (`RemakeConditioning`, `RemakeNoise`, `RemakeKeyframeUntil`)
/// are installed inside `ComfyUI/custom_nodes/ComfyUI-Viggle-Remake` and patched for ComfyUI versions without `comfy.audio`.
/// Returns `true` if files were newly installed or patched (requiring a ComfyUI restart if already running).
pub fn ensure_viggle_remake_custom_node(comfy: &Path) -> bool {
    let custom_nodes = comfy.join("custom_nodes");
    let node_dir = custom_nodes.join("ComfyUI-Viggle-Remake");
    let init_py = node_dir.join("__init__.py");
    let nodes_py = node_dir.join("nodes.py");
    let mut changed = false;

    if !init_py.is_file() || !nodes_py.is_file() {
        log("viggle_remake: installing ComfyUI-Viggle-Remake custom node");
        if std::fs::create_dir_all(&node_dir).is_err() {
            return false;
        }
        let client = build_dl_client();
        let base = "https://huggingface.co/Viggle/Viggle-Remake/resolve/main/comfyui/ComfyUI-Viggle-Remake";
        for (fname, dst) in [("__init__.py", &init_py), ("nodes.py", &nodes_py)] {
            if dst.is_file() {
                continue;
            }
            if let Ok(resp) = client.get(format!("{base}/{fname}")).send() {
                if resp.status().is_success() {
                    if let Ok(bytes) = resp.bytes() {
                        if std::fs::write(dst, &bytes).is_ok() {
                            changed = true;
                        }
                    }
                }
            }
        }
        if init_py.is_file() && nodes_py.is_file() {
            log("viggle_remake: ComfyUI-Viggle-Remake installed successfully");
        } else {
            log("viggle_remake: WARNING - failed to install ComfyUI-Viggle-Remake");
        }
    }

    if let Ok(txt) = std::fs::read_to_string(&nodes_py) {
        if txt.contains("import comfy.audio") && !txt.contains("_comfy_audio") {
            let patched = txt
                .replace(
                    "import comfy.audio",
                    "try:\n    import comfy.audio as _comfy_audio\nexcept ImportError:\n    _comfy_audio = None",
                )
                .replace(
                    "wav = comfy.audio.resample(wav, sr, AUDIO_SR) if sr != AUDIO_SR else wav",
                    "if sr != AUDIO_SR:\n                if _comfy_audio is not None:\n                    wav = _comfy_audio.resample(wav, sr, AUDIO_SR)\n                else:\n                    try:\n                        import torchaudio\n                        wav = torchaudio.functional.resample(wav, sr, AUDIO_SR)\n                    except ImportError:\n                        new_len = max(1, round(wav.shape[-1] * AUDIO_SR / sr))\n                        wav = torch.nn.functional.interpolate(wav.unsqueeze(0), size=new_len, mode=\"linear\", align_corners=False).squeeze(0)",
                );
            if patched != txt && std::fs::write(&nodes_py, patched).is_ok() {
                log("viggle_remake: patched nodes.py for ComfyUI without comfy.audio");
                changed = true;
            }
        }
    }

    changed
}

/// Ensures `ChrisColeTech/ComfyUI-GGUF-Loader` and its Python dependencies
/// (`gguf>=0.13.0`, `sentencepiece`, `protobuf`, `opencv-python-headless`) are installed inside `ComfyUI/custom_nodes`.
pub fn ensure_gguf_custom_node(comfy: &Path, py: Option<&Path>) {
    let custom_nodes = comfy.join("custom_nodes");
    let loader_dir = custom_nodes.join("ComfyUI-GGUF-Loader");
    let gguf_py = loader_dir.join("nodes").join("gguf.py");

    if !gguf_py.is_file() {
        log("gguf: installing ChrisColeTech/ComfyUI-GGUF-Loader custom node");
        let _ = std::fs::create_dir_all(&custom_nodes);

        // Move legacy city96 ComfyUI-GGUF aside if present so node registrations do not collide
        let legacy_dir = custom_nodes.join("ComfyUI-GGUF");
        if legacy_dir.exists() {
            let disabled_root = comfy.join("custom_nodes_disabled");
            let _ = std::fs::create_dir_all(&disabled_root);
            let backup = disabled_root.join("ComfyUI-GGUF");
            if backup.exists() {
                let _ = std::fs::remove_dir_all(&backup);
            }
            if let Err(e) = std::fs::rename(&legacy_dir, &backup) {
                log(&format!("gguf: could not move legacy ComfyUI-GGUF aside: {e}"));
                let _ = std::fs::remove_dir_all(&legacy_dir);
            } else {
                log("gguf: moved legacy ComfyUI-GGUF to custom_nodes_disabled");
            }
        }

        if loader_dir.exists() {
            let _ = std::fs::remove_dir_all(&loader_dir);
        }

        let clone_ok = Command::new("git")
            .args(["clone", "--depth", "1", "https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader.git"])
            .arg(&loader_dir)
            .creation_flags(CREATE_NO_WINDOW)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if !clone_ok || !gguf_py.is_file() {
            log("gguf: git clone unavailable or incomplete; downloading GitHub archive zip");
            let zip_path = custom_nodes.join("ComfyUI-GGUF-Loader-main.zip");
            let client = build_dl_client();
            if let Ok(resp) = client
                .get("https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader/archive/refs/heads/main.zip")
                .send()
            {
                if resp.status().is_success() {
                    if let Ok(bytes) = resp.bytes() {
                        if std::fs::write(&zip_path, &bytes).is_ok() {
                            #[cfg(windows)]
                            let ext_ok = Command::new("powershell")
                                .args([
                                    "-NoProfile",
                                    "-Command",
                                    &format!(
                                        "Expand-Archive -Force -Path '{}' -DestinationPath '{}'",
                                        zip_path.display().to_string().replace('\'', "''"),
                                        custom_nodes.display().to_string().replace('\'', "''")
                                    ),
                                ])
                                .creation_flags(CREATE_NO_WINDOW)
                                .status()
                                .map(|s| s.success())
                                .unwrap_or(false);
                            #[cfg(not(windows))]
                            let ext_ok = Command::new("python3")
                                .args(["-m", "zipfile", "-e"])
                                .arg(&zip_path)
                                .arg(&custom_nodes)
                                .status()
                                .map(|s| s.success())
                                .unwrap_or(false);

                            let _ = std::fs::remove_file(&zip_path);
                            let extracted = custom_nodes.join("ComfyUI-GGUF-Loader-main");
                            if ext_ok && extracted.is_dir() {
                                if loader_dir.exists() {
                                    let _ = std::fs::remove_dir_all(&loader_dir);
                                }
                                let _ = std::fs::rename(&extracted, &loader_dir);
                            }
                        }
                    }
                }
            }
        }

        if gguf_py.is_file() {
            log("gguf: ComfyUI-GGUF-Loader installed successfully");
        } else {
            log("gguf: WARNING - failed to install ComfyUI-GGUF-Loader");
        }
    }

    if gguf_py.is_file() {
        let deps_marker = loader_dir.join(".studio_deps_v2_ok");
        if !deps_marker.is_file() {
            if let Some(py_exe) = py {
                let probe = Command::new(py_exe)
                    .args(["-c", "import gguf, sentencepiece, google.protobuf, cv2"])
                    .current_dir(comfy)
                    .creation_flags(CREATE_NO_WINDOW)
                    .output();
                let deps_present = probe.map(|o| o.status.success()).unwrap_or(false);
                if !deps_present {
                    log("gguf: installing Python dependencies (gguf>=0.13.0 sentencepiece protobuf opencv-python-headless)");
                    let _ = Command::new(py_exe)
                        .args([
                            "-m",
                            "pip",
                            "install",
                            "--no-warn-script-location",
                            "gguf>=0.13.0",
                            "sentencepiece",
                            "protobuf",
                            "opencv-python-headless",
                        ])
                        .current_dir(comfy)
                        .creation_flags(CREATE_NO_WINDOW)
                        .status();
                }
                let verify = Command::new(py_exe)
                    .args(["-c", "import gguf, sentencepiece, google.protobuf, cv2"])
                    .current_dir(comfy)
                    .creation_flags(CREATE_NO_WINDOW)
                    .output()
                    .map(|o| o.status.success())
                    .unwrap_or(false);
                if verify {
                    let _ = std::fs::write(&deps_marker, "ok\n");
                    log("gguf: Python dependencies verified");
                } else {
                    log("gguf: WARNING - gguf/sentencepiece/protobuf/cv2 probe failed after pip install");
                }
            }
        }
    }
}

fn collect_safetensors_rel(base: &Path, dir: &Path, depth: usize, out: &mut Vec<String>) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_safetensors_rel(base, &p, depth + 1, out);
        } else if p
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("safetensors"))
            .unwrap_or(false)
        {
            if let Ok(rel) = p.strip_prefix(base) {
                let s = rel.to_string_lossy().replace('\\', "/");
                if !s.is_empty() && !out.contains(&s) {
                    out.push(s);
                }
            }
        }
    }
}

/// Lists all `.safetensors` LoRA filenames found in configured `models/loras` folders.
#[allow(dead_code)]
pub fn list_available_loras(sh: &Shared) -> Vec<String> {
    let mut dirs = Vec::new();
    let active_models = sh.paths.lock().unwrap().models.clone();
    if !active_models.as_os_str().is_empty() {
        dirs.push(active_models.join("loras"));
    }
    let cfg = sh.cfg.lock().unwrap().clone();
    for base in candidate_models_dirs(&cfg) {
        let d = base.join("loras");
        if d.is_dir() && !dirs.contains(&d) {
            dirs.push(d);
        }
    }
    let mut out = Vec::new();
    for d in dirs {
        collect_safetensors_rel(&d, &d, 0, &mut out);
    }
    out.sort_by(|a, b| a.to_ascii_lowercase().cmp(&b.to_ascii_lowercase()));
    out
}

/// Extracts a concise status summary from a `pip` output line for UI display.
fn summarize_pip_line(line: &str) -> Option<String> {
    let t = line.trim();
    if let Some(rest) = t.strip_prefix("Downloading ") {
        let pkg = rest.split_whitespace().next().unwrap_or(rest);
        let short = pkg.rsplit('/').next().unwrap_or(pkg);
        return Some(format!("Downloading {short}"));
    }
    if let Some(rest) = t.strip_prefix("Collecting ") {
        let pkg = rest.split([' ', '>', '<', '=', ';']).next().unwrap_or(rest);
        return Some(format!("Collecting {pkg}"));
    }
    if t.starts_with("Installing collected packages:") {
        return Some("Unpacking & installing packages...".to_string());
    }
    if t.starts_with("Successfully installed ") {
        return Some("Finalizing packages...".to_string());
    }
    None
}

/// Runs `py -m pip <args>` while streaming stdout lines into `EngState::RepairingEngine`.
fn run_pip_with_progress(
    py: &Path,
    comfy: &Path,
    args: &[&str],
    base_desc: &str,
    sh: &Shared,
    ctx: &eframe::egui::Context,
) -> std::io::Result<(bool, String, String)> {
    use std::io::{BufRead, BufReader, Read};
    use std::process::Stdio;

    *sh.state.lock().unwrap() = EngState::RepairingEngine {
        step: format!("{base_desc}..."),
    };
    ctx.request_repaint();

    let mut child = Command::new(py)
        .args(["-m", "pip"])
        .args(args)
        .env("PYTHONUNBUFFERED", "1")
        .current_dir(comfy)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()?;

    let stderr_handle = child.stderr.take().map(|mut se| {
        std::thread::spawn(move || {
            let mut s = String::new();
            let _ = se.read_to_string(&mut s);
            s
        })
    });

    let mut stdout_buf = String::new();
    if let Some(so) = child.stdout.take() {
        let reader = BufReader::new(so);
        let t0 = Instant::now();
        for line in reader.lines().map_while(Result::ok) {
            stdout_buf.push_str(&line);
            stdout_buf.push('\n');
            if let Some(detail) = summarize_pip_line(&line) {
                let secs = t0.elapsed().as_secs();
                *sh.state.lock().unwrap() = EngState::RepairingEngine {
                    step: format!("{base_desc}: {detail} ({secs}s)"),
                };
                ctx.request_repaint();
            }
        }
    }

    let status = child.wait()?;
    let stderr_buf = stderr_handle.and_then(|h| h.join().ok()).unwrap_or_default();
    Ok((status.success(), stdout_buf, stderr_buf))
}

fn detect_gpu_vendor() -> &'static str {
    #[cfg(windows)]
    {
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
    }

    #[cfg(not(windows))]
    {
        if let Ok(o) = Command::new("lspci").output() {
            let s = String::from_utf8_lossy(&o.stdout).to_lowercase();
            if s.contains("nvidia") || s.contains("geforce") || s.contains("quadro") || s.contains("rtx") {
                return "nvidia";
            } else if s.contains("amd") || s.contains("radeon") || s.contains("advanced micro devices") {
                return "amd";
            } else if s.contains("intel") || s.contains("arc") {
                return "intel";
            }
        }
        if let Ok(o) = Command::new("nvidia-smi").output() {
            if o.status.success() {
                return "nvidia";
            }
        }
        if let Ok(o) = Command::new("rocm-smi").output() {
            if o.status.success() {
                return "amd";
            }
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
        c.push(h.join(".local/share/ComfyUI"));
        c.push(h.join("src/ComfyUI"));
        c.push(h.join("comfyui"));
    }

    #[cfg(not(windows))]
    {
        c.push(PathBuf::from("/opt/ComfyUI"));
        c.push(PathBuf::from("/opt/comfyui"));
    }

    c.into_iter().find(|p| {
        !cfg.ignore_paths.iter().any(|ign| ign == p || p.starts_with(ign)) && p.join("main.py").exists()
    })
}

fn find_in_path(bin_name: &str) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let p = dir.join(bin_name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn find_pythons(cfg: &Config, comfy: &Path) -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    // 1. Explicitly configured python in settings
    if let Some(ref p) = cfg.python_path {
        if p.is_file() {
            candidates.push(p.clone());
        }
    }

    let parent = comfy.parent().map(Path::to_path_buf).unwrap_or_default();
    let grand = parent.parent().map(Path::to_path_buf).unwrap_or_default();

    // 2. Comfy-local virtual environments
    let local_envs = [
        comfy.join(".venv/Scripts/python.exe"),
        comfy.join("venv/Scripts/python.exe"),
        parent.join(".venv/Scripts/python.exe"),
        parent.join("python_embeded/python.exe"),
        grand.join("python_embeded/python.exe"),
        parent.join("standalone-env/python.exe"),
        comfy.join(".venv/bin/python"),
        comfy.join(".venv/bin/python3"),
        comfy.join("venv/bin/python"),
        comfy.join("venv/bin/python3"),
        parent.join(".venv/bin/python"),
        parent.join(".venv/bin/python3"),
        parent.join("bin/python"),
        parent.join("bin/python3"),
    ];

    for p in local_envs {
        if p.is_file() && !candidates.contains(&p) {
            candidates.push(p);
        }
    }

    // 3. User conda / pyenv virtual environments
    if let Some(h) = dirs::home_dir() {
        let env_dirs = [
            h.join("miniconda3/envs"),
            h.join("anaconda3/envs"),
            h.join(".conda/envs"),
            h.join("miniforge3/envs"),
            h.join(".virtualenvs"),
        ];
        let comfy_env_names = ["comfy", "comfyui", "ComfyUI", "comfy-env", "ai", "torch"];
        for ed in env_dirs {
            for name in comfy_env_names {
                let py_unix = ed.join(name).join("bin/python");
                if py_unix.is_file() && !candidates.contains(&py_unix) {
                    candidates.push(py_unix);
                }
                let py_win = ed.join(name).join("python.exe");
                if py_win.is_file() && !candidates.contains(&py_win) {
                    candidates.push(py_win);
                }
            }
        }
    }

    // 4. PATH binaries
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            #[cfg(windows)]
            let names = ["python.exe"];
            #[cfg(not(windows))]
            let names = ["python3", "python"];

            for name in names {
                let python = dir.join(name);
                if python.is_file() && !candidates.contains(&python) {
                    candidates.push(python);
                }
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

fn kill_pid(pid: u32) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("pkill")
            .args(["-9", "-P", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = Command::new("kill")
            .args(["-9", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

pub fn kill_tree(c: &mut Child) {
    kill_pid(c.id());
    let _ = c.kill();
    let _ = c.wait();
}

fn supports_ck_attention(py: &Path, comfy: &Path) -> bool {
    let cli_args = comfy.join("comfy").join("cli_args.py");
    let has_flag = std::fs::read_to_string(&cli_args)
        .map(|s| s.contains("--use-ck-attention"))
        .unwrap_or(false);
    if !has_flag {
        return false;
    }

    let mut check = Command::new(py);
    check
        .args([
            "-c",
            "import comfy_kitchen; raise SystemExit(0 if comfy_kitchen.int8_attention_is_available() else 1)",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW);
    check.status().map(|s| s.success()).unwrap_or(false)
}

fn spawn(py: &Path, comfy: &Path, models: &Path) -> Result<Child, String> {
    let log_file = data_dir().join("comfyui.log");
    if log_file.exists() {
        let prev = data_dir().join("comfyui.prev.log");
        let _ = std::fs::rename(&log_file, &prev);
    }
    let out = std::fs::File::create(&log_file).map_err(|e| e.to_string())?;
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
        "--disable-smart-memory",
        "--cache-none",
        "--reserve-vram",
        "1.5",
        "--vram-headroom",
        "1.0",
    ]);
    if supports_ck_attention(py, comfy) {
        log("ComfyUI: enabling built-in INT8 attention (--use-ck-attention)");
        cmd.arg("--use-ck-attention");
    }
    cmd.arg("--models-directory")
    .arg(models)
    .current_dir(comfy)
    .env("PYTHONUNBUFFERED", "1")
    .env("PYTHONIOENCODING", "utf-8")
    .stdin(Stdio::null())
    .stdout(out)
    .stderr(err)
    .creation_flags(CREATE_NO_WINDOW);

    #[cfg(not(windows))]
    {
        if std::env::var("HSA_OVERRIDE_GFX_VERSION").is_err() {
            if let Ok(o) = Command::new("rocminfo").output() {
                let s = String::from_utf8_lossy(&o.stdout);
                if s.contains("gfx1031") || s.contains("gfx1032") || s.contains("gfx1033") || s.contains("gfx1034") || s.contains("gfx1035") || s.contains("gfx1036") {
                    cmd.env("HSA_OVERRIDE_GFX_VERSION", "10.3.0");
                    log("ROCm: auto-set HSA_OVERRIDE_GFX_VERSION=10.3.0 for RDNA2 target");
                } else if s.contains("gfx1102") || s.contains("gfx1103") {
                    cmd.env("HSA_OVERRIDE_GFX_VERSION", "11.0.0");
                    log("ROCm: auto-set HSA_OVERRIDE_GFX_VERSION=11.0.0 for RDNA3 target");
                }
            }
        }
    }

    cmd.spawn().map_err(|e| format!("could not start {}: {e}", py.display()))
}

static PEAK_DOWNLOAD_SPEED_BPS: AtomicU64 = AtomicU64::new(0);

fn update_peak_speed(speed_mb: f32) {
    if speed_mb <= 0.0 {
        return;
    }
    let bps = (speed_mb * 1024.0 * 1024.0) as u64;
    let mut cur = PEAK_DOWNLOAD_SPEED_BPS.load(Ordering::Relaxed);
    while bps > cur {
        match PEAK_DOWNLOAD_SPEED_BPS.compare_exchange_weak(cur, bps, Ordering::SeqCst, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

/// Lightweight parallel probe that measures curl's network download speed into memory without disk writes.
struct CurlSpeedProbe {
    stop: Arc<AtomicBool>,
    child_id: Arc<AtomicU32>,
    speed_bps: Arc<AtomicU64>,
    start_time: Instant,
}

impl CurlSpeedProbe {
    fn start(curl_exe: &Path, url: &str, probe_start: u64, probe_len: u64) -> Option<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let child_id = Arc::new(AtomicU32::new(0));
        let speed_bps = Arc::new(AtomicU64::new(0));

        let stop_clone = stop.clone();
        let child_id_clone = child_id.clone();
        let speed_bps_clone = speed_bps.clone();
        let curl_path = curl_exe.to_path_buf();
        let url_str = url.to_string();

        let _ = std::thread::spawn(move || {
            use std::io::Read;
            let mut cmd = Command::new(&curl_path);
            cmd.args(["-L", "--fail", "-s", "-S", "--connect-timeout", "10", "--max-time", "16", "-o", "-"]);
            if probe_len > 0 {
                cmd.arg("-r").arg(format!("{}-{}", probe_start, probe_start + probe_len));
            }
            cmd.arg(&url_str)
                .creation_flags(CREATE_NO_WINDOW)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null());

            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(_) => return,
            };
            child_id_clone.store(child.id(), Ordering::SeqCst);

            let mut stdout = match child.stdout.take() {
                Some(s) => s,
                None => {
                    kill_tree(&mut child);
                    return;
                }
            };

            let mut buf = [0u8; 64 * 1024];
            let mut last_tick = Instant::now();
            let mut bytes_since = 0u64;
            let mut smoothed_bps = 0.0f32;

            loop {
                if stop_clone.load(Ordering::SeqCst) {
                    break;
                }
                match stdout.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        bytes_since += n as u64;
                        let elapsed = last_tick.elapsed();
                        if elapsed >= Duration::from_millis(250) {
                            let secs = elapsed.as_secs_f32();
                            if secs > 0.0 {
                                let inst = bytes_since as f32 / secs;
                                smoothed_bps = if smoothed_bps == 0.0 { inst } else { smoothed_bps * 0.7 + inst * 0.3 };
                                speed_bps_clone.store(smoothed_bps as u64, Ordering::SeqCst);
                            }
                            last_tick = Instant::now();
                            bytes_since = 0;
                        }
                    }
                    Err(_) => break,
                }
            }
            kill_tree(&mut child);
        });

        Some(Self {
            stop,
            child_id,
            speed_bps,
            start_time: Instant::now(),
        })
    }

    fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        let pid = self.child_id.load(Ordering::SeqCst);
        if pid > 0 {
            kill_pid(pid);
        }
    }
}

impl Drop for CurlSpeedProbe {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Seamless single-stream curl download that resumes on an existing partial file (`-C -`).
fn download_curl_resume<F>(
    curl_exe: &Path,
    url: &str,
    dest: &Path,
    part_path: &Path,
    total_bytes: u64,
    existing_len: u64,
    mut on_progress: F,
    cancel: &AtomicBool,
    ctx: &eframe::egui::Context,
    speed_watchdog: bool,
    sh: Option<&Shared>,
) -> Result<(), String>
where
    F: FnMut(u64, u64, f32),
{
    let mut child = Command::new(curl_exe)
        .args(["-L", "--fail", "-s", "-S", "-C", "-", "--connect-timeout", "30", "-o"])
        .arg(part_path)
        .arg(url)
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to start curl: {e}"))?;

    let start_time = Instant::now();
    let mut last_len = existing_len;
    let mut last_tick = Instant::now();
    let mut last_metered_check = Instant::now();
    let mut speed_mb = 0.0f32;

    if let Some(s) = sh {
        if s.download_paused.load(Ordering::SeqCst) {
            kill_tree(&mut child);
            return Err("download paused".into());
        }
    }

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
        if let Some(s) = sh {
            if s.download_paused.load(Ordering::SeqCst) {
                kill_tree(&mut child);
                return Err("download paused".into());
            }
            if last_metered_check.elapsed() >= Duration::from_millis(1500) {
                last_metered_check = Instant::now();
                if is_metered_connection() && !s.metered_allowed.load(Ordering::SeqCst) {
                    log("metered network switch detected during curl stream");
                    s.metered_switch_detected.store(true, Ordering::SeqCst);
                    s.download_paused.store(true, Ordering::SeqCst);
                    flash_taskbar();
                    ctx.send_viewport_cmd(eframe::egui::ViewportCommand::RequestUserAttention(eframe::egui::UserAttentionType::Critical));
                    ctx.request_repaint();
                    kill_tree(&mut child);
                    return Err("download paused (metered network)".into());
                }
            }
        }
        std::thread::sleep(Duration::from_millis(250));
        let len = std::fs::metadata(part_path).map(|m| m.len()).unwrap_or(last_len);
        let secs = last_tick.elapsed().as_secs_f32();
        if secs > 0.0 {
            let inst = (len.saturating_sub(last_len) as f32 / 1024.0 / 1024.0) / secs;
            speed_mb = if speed_mb == 0.0 { inst } else { speed_mb * 0.7 + inst * 0.3 };
            update_peak_speed(speed_mb);
        }
        last_len = len;
        last_tick = Instant::now();
        on_progress(len, total_bytes, speed_mb);
        ctx.request_repaint();

        let el = start_time.elapsed();
        if speed_watchdog && el >= Duration::from_secs(10) && el <= Duration::from_secs(22)
            && speed_mb < 0.6 && total_bytes > 50 * 1024 * 1024 {
            log(&format!("Download watchdog: curl stream is throttled at {speed_mb:.2} MB/s, switching mirror"));
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

    let len = std::fs::metadata(part_path).map(|m| m.len()).unwrap_or(0);
    if total_bytes > 0 && len != total_bytes {
        return Err(format!("Incomplete download: received {len} of {total_bytes} bytes."));
    }
    if dest.exists() {
        let _ = std::fs::remove_file(dest);
    }
    std::fs::rename(part_path, dest).map_err(|e| format!("rename error: {e}"))?;
    on_progress(len, len, speed_mb);
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
    sh: Option<&Shared>,
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

    let curl_exe = if cfg!(windows) {
        std::env::var_os("SystemRoot")
            .map(|r| PathBuf::from(r).join("System32").join("curl.exe"))
            .filter(|p| p.is_file())
            .or_else(|| find_in_path("curl.exe"))
    } else {
        find_in_path("curl").or_else(|| {
            let p = PathBuf::from("/usr/bin/curl");
            if p.is_file() { Some(p) } else { None }
        })
    };

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
    let mut last_metered_check = Instant::now();
    let mut bytes_since_tick = 0u64;
    let mut speed_mb = 0.0f32;

    if let Some(s) = sh {
        if s.download_paused.load(Ordering::SeqCst) {
            return Err("download paused".into());
        }
    }

    on_progress(downloaded, total_bytes, speed_mb);
    ctx.request_repaint();

    let mut download_err: Option<String> = None;
    let mut should_swap_to_curl = false;
    let mut probe_attempted = false;
    let mut curl_probe: Option<CurlSpeedProbe> = None;
    let mut consecutive_faster_time = Duration::ZERO;

    loop {
        if cancel.load(Ordering::SeqCst) {
            download_err = Some("download cancelled".into());
            break;
        }
        if let Some(s) = sh {
            if s.download_paused.load(Ordering::SeqCst) {
                download_err = Some("download paused".into());
                break;
            }
            if last_metered_check.elapsed() >= Duration::from_millis(1500) {
                last_metered_check = Instant::now();
                if is_metered_connection() && !s.metered_allowed.load(Ordering::SeqCst) {
                    log("metered network switch detected during download stream");
                    s.metered_switch_detected.store(true, Ordering::SeqCst);
                    s.download_paused.store(true, Ordering::SeqCst);
                    flash_taskbar();
                    ctx.send_viewport_cmd(eframe::egui::ViewportCommand::RequestUserAttention(eframe::egui::UserAttentionType::Critical));
                    ctx.request_repaint();
                    download_err = Some("download paused (metered network)".into());
                    break;
                }
            }
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
                update_peak_speed(speed_mb);
            }
            last_tick = Instant::now();
            bytes_since_tick = 0;
            on_progress(downloaded, total_bytes, speed_mb);
            ctx.request_repaint();

            let elapsed_total = start_time.elapsed();

            // Parallel curl speed test:
            // If download is large (> 50 MB), has been running >= 15s to let TCP ramp up,
            // and is failing to break max speed observed on this connection (or slow < 2.5 MB/s),
            // start a 15-second parallel test of curl in the background without interrupting reqwest.
            if !probe_attempted && curl_probe.is_none() && curl_exe.is_some() && total_bytes > 50 * 1024 * 1024 && elapsed_total >= Duration::from_secs(15) {
                let peak_bps = PEAK_DOWNLOAD_SPEED_BPS.load(Ordering::Relaxed);
                let cur_bps = (speed_mb * 1024.0 * 1024.0) as u64;
                let is_slow = if peak_bps >= 5 * 1024 * 1024 {
                    cur_bps < peak_bps / 2
                } else {
                    speed_mb < 2.5
                };
                if is_slow {
                    probe_attempted = true;
                    let probe_start = (downloaded + 50 * 1024 * 1024).min(total_bytes.saturating_sub(150 * 1024 * 1024));
                    let probe_len = 100 * 1024 * 1024;
                    log(&format!(
                        "Download probe: reqwest running at {speed_mb:.2} MB/s (connection peak: {:.2} MB/s). Starting 15s curl probe in parallel...",
                        peak_bps as f32 / 1024.0 / 1024.0
                    ));
                    curl_probe = CurlSpeedProbe::start(curl_exe.as_ref().unwrap(), url, probe_start, probe_len);
                }
            }

            // Monitor parallel curl probe
            if let Some(ref probe) = curl_probe {
                let curl_bps = probe.speed_bps.load(Ordering::SeqCst);
                let curl_mb = curl_bps as f32 / 1024.0 / 1024.0;
                let cur_bps = (speed_mb * 1024.0 * 1024.0) as u64;

                // Check if curl exceeds reqwest speed significantly (> 1.25x and > 1.5 MB/s)
                if curl_bps > (cur_bps as f32 * 1.25) as u64 && curl_mb > 1.5 {
                    consecutive_faster_time += elapsed;
                    if consecutive_faster_time >= Duration::from_secs(3) {
                        log(&format!(
                            "Download live swap: curl exceeded reqwest ({curl_mb:.2} MB/s vs {speed_mb:.2} MB/s) for 3 consecutive seconds. Swapping over to curl live!"
                        ));
                        should_swap_to_curl = true;
                        break;
                    }
                } else {
                    consecutive_faster_time = Duration::ZERO;
                }

                if probe.start_time.elapsed() >= Duration::from_secs(15) {
                    log(&format!(
                        "Download probe: curl 15s test completed (curl maxed at {curl_mb:.2} MB/s vs reqwest {speed_mb:.2} MB/s). Retaining original reqwest method."
                    ));
                    probe.stop();
                    curl_probe = None;
                }
            }

            // Speed Watchdog: During the initial qualification window (10s to 22s), if sustained speed is choked under 0.6 MB/s, switch mirror.
            if speed_watchdog && curl_probe.is_none() && elapsed_total >= Duration::from_secs(10) && elapsed_total <= Duration::from_secs(22) && speed_mb < 0.6 && total_bytes > 50 * 1024 * 1024 {
                log(&format!("Download watchdog: current stream is throttled at {speed_mb:.2} MB/s, switching to faster mirror"));
                download_err = Some(format!("Download throttled at {speed_mb:.2} MB/s; switching mirror"));
                break;
            }
        }
    }

    if let Some(probe) = curl_probe.take() {
        probe.stop();
    }

    drop(tx);
    let writer_res = writer_thread.join().map_err(|_| "disk writer thread panicked".to_string())?;
    writer_res?;

    if should_swap_to_curl {
        let flushed_len = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(downloaded);
        log(&format!("Download live swap: resuming with curl from byte offset {flushed_len} / {total_bytes}"));
        return download_curl_resume(
            curl_exe.as_ref().unwrap(),
            url,
            dest,
            &part_path,
            total_bytes,
            flushed_len,
            on_progress,
            cancel,
            ctx,
            speed_watchdog,
            sh,
        );
    }

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
    sh: Option<&Shared>,
) -> Result<(), String>
where
    F: FnMut(u64, u64, f32),
{
    download_stream(client, url, dest, expected_size, on_progress, cancel, ctx, false, sh)
}

fn bootstrap_engine(
    sh: &Arc<Shared>,
    ctx: &eframe::egui::Context,
    dl_client: &reqwest::blocking::Client,
    recovering: bool,
) -> Result<PathBuf, String> {
    let engine_dir = fresh_engine_dir(&data_dir().join("engine"))?;

    #[cfg(not(windows))]
    {
        let (_, vendor_name) = get_comfy_portable_url();
        log(&format!("bootstrap: setting up ComfyUI for Linux ({vendor_name}) in {}", engine_dir.display()));

        let step_desc = if recovering {
            "Existing engine failed; setting up a fresh ComfyUI instance...".to_string()
        } else {
            "Setting up ComfyUI engine for Linux...".to_string()
        };

        *sh.state.lock().unwrap() = EngState::DownloadingEngine {
            step: format!("{step_desc} (cloning repository)"),
            progress: 0.1,
            bytes_done: 0,
            bytes_total: 0,
            speed_mb: 0.0,
        };
        ctx.request_repaint();

        let comfy_target = engine_dir.join("ComfyUI");
        let git_status = Command::new("git")
            .args(["clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git"])
            .arg(&comfy_target)
            .status();

        let clone_ok = git_status.map(|s| s.success()).unwrap_or(false);
        if !clone_ok {
            log("bootstrap: git clone failed or git missing; downloading repository source archive");
            let archive_path = engine_dir.join("comfyui_master.tar.gz");
            let dl_res = download_file(
                dl_client,
                "https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.tar.gz",
                &archive_path,
                0,
                |done, total, speed| {
                    let prog = if total > 0 { (done as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
                    *sh.state.lock().unwrap() = EngState::DownloadingEngine {
                        step: "Downloading ComfyUI source archive...".into(),
                        progress: prog,
                        bytes_done: done,
                        bytes_total: total,
                        speed_mb: speed,
                    };
                },
                &sh.cancel,
                ctx,
                Some(sh.as_ref()),
            );
            if let Err(e) = dl_res {
                return Err(format!("Could not acquire ComfyUI on Linux: {e}\nPlease install ComfyUI and set its path in Settings."));
            }

            let _ = Command::new("tar")
                .args(["-xzf", archive_path.to_str().unwrap_or(""), "-C", engine_dir.to_str().unwrap_or("")])
                .status();
            let _ = std::fs::remove_file(&archive_path);

            if engine_dir.join("ComfyUI-master").exists() {
                let _ = std::fs::rename(engine_dir.join("ComfyUI-master"), &comfy_target);
            }
        }

        if !comfy_target.join("main.py").is_file() {
            return Err("ComfyUI download finished but main.py was not found. Please specify ComfyUI directory in Settings.".into());
        }

        // Engine source files are downloaded; start ticked model downloads immediately while venv & deps set up
        {
            let cfg = sh.cfg.lock().unwrap().clone();
            let models_dir = find_models(&cfg, &comfy_target);
            sh.paths.lock().unwrap().models = models_dir;
        }
        spawn_model_downloads_if_needed(sh, ctx);

        // Initialize virtualenv
        *sh.state.lock().unwrap() = EngState::ExtractingEngine {
            step: "Creating Python virtual environment (.venv)...".into(),
        };
        ctx.request_repaint();

        let venv_status = Command::new("python3")
            .args(["-m", "venv", ".venv"])
            .current_dir(&comfy_target)
            .status()
            .or_else(|_| {
                Command::new("python")
                    .args(["-m", "venv", ".venv"])
                    .current_dir(&comfy_target)
                    .status()
            });

        let venv_ok = venv_status.map(|s| s.success()).unwrap_or(false);
        if !venv_ok {
            return Err("Failed to create Python virtual environment (.venv). Please ensure python3 and python3-venv are installed (e.g. 'sudo apt install python3 python3-venv python3-pip').".into());
        }

        let py_venv = if comfy_target.join(".venv/bin/python").is_file() {
            comfy_target.join(".venv/bin/python")
        } else if comfy_target.join(".venv/bin/python3").is_file() {
            comfy_target.join(".venv/bin/python3")
        } else {
            return Err("Virtual environment created but .venv/bin/python was not found.".into());
        };

        log(&format!("bootstrap: virtual environment created at {}; configuring dependencies", py_venv.display()));
        install_or_repair_dependencies(&py_venv, &comfy_target, sh, ctx)
            .map_err(|e| format!("ComfyUI dependency installation failed on Linux ({vendor_name}):\n{e}"))?;

        return Ok(comfy_target);
    }

    #[cfg(windows)]
    {
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
                Some(sh.as_ref()),
            );
            if let Err(e) = res {
                return Err(format!("Engine download failed: {e}"));
            }
        }

        // Engine archive is downloaded; start ticked model downloads immediately while extracting
        {
            let cfg = sh.cfg.lock().unwrap().clone();
            let expected_comfy = engine_dir.join("ComfyUI_windows_portable/ComfyUI");
            let models_dir = find_models(&cfg, &expected_comfy);
            sh.paths.lock().unwrap().models = models_dir;
        }
        spawn_model_downloads_if_needed(sh, ctx);

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
    loop {
        let vid_profile = sh.active_video_profile.lock().unwrap().clone();
        let mut current_models_dir = if let Some(p) = sh.pending_models_dir.lock().unwrap().clone() {
            p
        } else {
            models_dir.to_path_buf()
        };

        if vid_profile.is_ltx25() {
            ensure_ltx25_gguf_sidecar(&current_models_dir);
        }

        let target_models = Profile::target_download_models(sh);

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
            let is_optional_r2v = m.optional && vid_profile.models.iter().any(|vm| vm.filename == m.filename);
            let is_video_model = !m.optional && vid_profile.models.iter().any(|vm| vm.filename == m.filename);
            if is_optional_r2v && !sh.dl_r2v_model.load(Ordering::SeqCst) {
                continue;
            }
            if is_video_model && !sh.dl_video_model.load(Ordering::SeqCst) {
                continue;
            }
            if !is_optional_r2v && !is_video_model && !sh.dl_image_model.load(Ordering::SeqCst) {
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

            if sh.download_paused.load(Ordering::SeqCst) {
                log(&format!("bootstrap: model download paused for {}", m.filename));
                let part_path = dest.with_extension(format!(
                    "{}.part",
                    dest.extension().and_then(|e| e.to_str()).unwrap_or("tmp")
                ));
                let part_len = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
                let f_prog = if file_expected > 0 { (part_len as f32 / file_expected as f32).clamp(0.0, 1.0) } else { 0.0 };
                let overall = if total_all_bytes > 0 {
                    ((completed_bytes + part_len) as f32 / total_all_bytes as f32).clamp(0.0, 1.0)
                } else {
                    f_prog
                };
                set_model_dl_state(
                    sh,
                    None,
                    Some(EngState::ModelDownloadPaused {
                        current_file: cur_file.clone(),
                        file_index: cur_idx,
                        total_files,
                        file_progress: f_prog,
                        bytes_done: part_len,
                        bytes_total: file_expected,
                        overall_progress: overall,
                    }),
                );
                ctx.request_repaint();
                return Ok(());
            }

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
                if sh.download_paused.load(Ordering::SeqCst) || cancel_ref.load(Ordering::SeqCst) || sh.cancel.load(Ordering::SeqCst) || sh.cancel_model_dl.load(Ordering::SeqCst) {
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
                        if sh.download_paused.load(Ordering::SeqCst) {
                            return;
                        }
                        if sh.cancel.load(Ordering::SeqCst) || sh.cancel_model_dl.load(Ordering::SeqCst) {
                            cancel_ref.store(true, Ordering::SeqCst);
                        }
                        let f_prog = if total > 0 { (done as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
                        let overall = if total_all_bytes > 0 {
                            ((completed_bytes + done) as f32 / total_all_bytes as f32).clamp(0.0, 1.0)
                        } else {
                            f_prog
                        };
                        set_model_dl_state(
                            sh,
                            Some(ModelDownloadProgress {
                                current_file: cur_file.clone(),
                                file_index: cur_idx,
                                total_files,
                                file_progress: f_prog,
                                bytes_done: done,
                                bytes_total: total,
                                speed_mb: speed,
                                overall_progress: overall,
                            }),
                            Some(EngState::DownloadingModels {
                                current_file: cur_file.clone(),
                                file_index: cur_idx,
                                total_files,
                                file_progress: f_prog,
                                bytes_done: done,
                                bytes_total: total,
                                speed_mb: speed,
                                overall_progress: overall,
                            }),
                        );
                    },
                    cancel_ref,
                    ctx,
                    enable_watchdog,
                    Some(sh),
                );
                if download_result.is_ok() || sh.download_paused.load(Ordering::SeqCst) {
                    if download_result.is_ok() {
                        log(&format!("bootstrap: successfully downloaded {} via {}", m.filename, source_name));
                    }
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

            if sh.download_paused.load(Ordering::SeqCst) {
                log(&format!("bootstrap: model download paused for {}", m.filename));
                let part_path = dest.with_extension(format!(
                    "{}.part",
                    dest.extension().and_then(|e| e.to_str()).unwrap_or("tmp")
                ));
                let part_len = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
                let f_prog = if file_expected > 0 { (part_len as f32 / file_expected as f32).clamp(0.0, 1.0) } else { 0.0 };
                let overall = if total_all_bytes > 0 {
                    ((completed_bytes + part_len) as f32 / total_all_bytes as f32).clamp(0.0, 1.0)
                } else {
                    f_prog
                };
                set_model_dl_state(
                    sh,
                    None,
                    Some(EngState::ModelDownloadPaused {
                        current_file: cur_file.clone(),
                        file_index: cur_idx,
                        total_files,
                        file_progress: f_prog,
                        bytes_done: part_len,
                        bytes_total: file_expected,
                        overall_progress: overall,
                    }),
                );
                ctx.request_repaint();
                return Ok(());
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
            *sh.models_found.lock().unwrap() = profile
                .models
                .iter()
                .map(|pm| (pm.filename.clone(), current_models_dir.join(&pm.category).join(&pm.filename).exists()))
                .collect();
        }

        if sh.download_paused.load(Ordering::SeqCst) {
            return Ok(());
        }

        if !any_interrupted {
            break;
        }
    }

    *sh.model_dl_progress.lock().unwrap() = None;
    Ok(())
}

fn build_dl_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(7200))
        .connect_timeout(Duration::from_secs(30))
        .tcp_nodelay(true)
        .no_proxy()
        .http1_only()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        .build()
        .unwrap()
}

fn set_model_dl_state(sh: &Shared, progress: Option<ModelDownloadProgress>, eng_state: Option<EngState>) {
    *sh.model_dl_progress.lock().unwrap() = progress;
    if let Some(new_st) = eng_state {
        let mut st = sh.state.lock().unwrap();
        let engine_busy_or_ready = matches!(
            *st,
            EngState::DownloadingEngine { .. }
                | EngState::ExtractingEngine { .. }
                | EngState::RepairingEngine { .. }
                | EngState::Starting { .. }
                | EngState::Ready { .. }
                | EngState::Failed(_)
        );
        if !engine_busy_or_ready {
            *st = new_st;
        }
    }
}

struct ModelDlRunningGuard(Arc<Shared>);
impl Drop for ModelDlRunningGuard {
    fn drop(&mut self) {
        self.0.model_dl_running.store(false, Ordering::SeqCst);
    }
}

/// Spawns a background model downloader thread if any ticked models are missing and no
/// model download worker is currently active. Safe to call during engine extraction/repair.
/// Cancels any in-flight model download and starts downloading missing models
/// for the newly selected profile in the background, without restarting ComfyUI
/// or interrupting any active generation in VRAM.
pub fn restart_model_downloads_for_profile_change(sh: &Arc<Shared>, ctx: &eframe::egui::Context) {
    sh.cancel_model_dl.store(true, Ordering::SeqCst);
    sh.download_paused.store(false, Ordering::SeqCst);
    let sh = sh.clone();
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        for _ in 0..40 {
            if !sh.model_dl_running.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        sh.cancel_model_dl.store(false, Ordering::SeqCst);
        spawn_model_downloads_if_needed(&sh, &ctx);
    });
}

pub fn spawn_model_downloads_if_needed(sh: &Arc<Shared>, ctx: &eframe::egui::Context) {
    if sh.download_paused.load(Ordering::SeqCst) {
        return;
    }
    if Profile::missing_models(sh).is_empty() {
        return;
    }
    if is_metered_connection() && !sh.metered_allowed.load(Ordering::SeqCst) && !sh.download_confirmed.load(Ordering::SeqCst) {
        return;
    }
    if sh.model_dl_running.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        return;
    }

    let sh = sh.clone();
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let _guard = ModelDlRunningGuard(sh.clone());
        sh.cancel_model_dl.store(false, Ordering::SeqCst);

        let models_dir = sh
            .pending_models_dir
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| sh.paths.lock().unwrap().models.clone());
        let profile = sh.active_profile.lock().unwrap().clone();
        let dl_client = build_dl_client();

        match bootstrap_models(&sh, &ctx, &dl_client, &models_dir, &profile) {
            Ok(()) => {
                *sh.model_dl_progress.lock().unwrap() = None;
                let cur_dir = sh
                    .pending_models_dir
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| sh.paths.lock().unwrap().models.clone());
                *sh.models_found.lock().unwrap() = profile
                    .models
                    .iter()
                    .map(|m| (m.filename.clone(), cur_dir.join(&m.category).join(&m.filename).exists()))
                    .collect();
            }
            Err((file, e)) => {
                set_model_dl_state(
                    &sh,
                    None,
                    Some(EngState::ModelDownloadFailed {
                        current_file: file,
                        error: e,
                    }),
                );
            }
        }
        ctx.request_repaint();
    });
}

pub fn toggle_model_download(sh: &Arc<Shared>, ctx: &eframe::egui::Context, is_image: bool, enabled: bool) {
    if is_image {
        sh.dl_image_model.store(enabled, Ordering::SeqCst);
    } else {
        sh.dl_video_model.store(enabled, Ordering::SeqCst);
    }
    {
        let mut cfg = sh.cfg.lock().unwrap();
        if is_image {
            cfg.dl_image_model = enabled;
        } else {
            cfg.dl_video_model = enabled;
        }
        cfg.save();
    }
    if !enabled {
        let dl_active = sh.model_dl_running.load(Ordering::SeqCst)
            || matches!(*sh.state.lock().unwrap(), EngState::DownloadingModels { .. });
        if dl_active {
            sh.cancel_model_dl.store(true, Ordering::SeqCst);
        }
        if !sh.dl_image_model.load(Ordering::SeqCst)
            && !sh.dl_video_model.load(Ordering::SeqCst)
            && !sh.dl_r2v_model.load(Ordering::SeqCst)
        {
            *sh.model_dl_progress.lock().unwrap() = None;
            let mut st = sh.state.lock().unwrap();
            if matches!(*st, EngState::DownloadingModels { .. } | EngState::ModelDownloadPaused { .. }) {
                *st = EngState::Detecting;
            }
        }
    } else {
        sh.cancel_model_dl.store(false, Ordering::SeqCst);
        sh.download_paused.store(false, Ordering::SeqCst);
        let paths = sh.paths.lock().unwrap().clone();
        if !paths.models.as_os_str().is_empty() {
            spawn_model_downloads_if_needed(sh, ctx);
        }
    }
}

pub fn toggle_r2v_download(sh: &Arc<Shared>, ctx: &eframe::egui::Context, enabled: bool) {
    sh.dl_r2v_model.store(enabled, Ordering::SeqCst);
    {
        let mut cfg = sh.cfg.lock().unwrap();
        cfg.dl_r2v_model = enabled;
        cfg.save();
    }
    if !enabled {
        let dl_r2v_active = sh.model_dl_running.load(Ordering::SeqCst)
            && sh.model_dl_progress.lock().unwrap().as_ref().is_some_and(|p| {
                p.current_file.to_ascii_lowercase().contains("ref2va")
            });
        if dl_r2v_active {
            sh.cancel_model_dl.store(true, Ordering::SeqCst);
        }
        if !sh.dl_image_model.load(Ordering::SeqCst)
            && !sh.dl_video_model.load(Ordering::SeqCst)
            && !sh.dl_r2v_model.load(Ordering::SeqCst)
        {
            *sh.model_dl_progress.lock().unwrap() = None;
            let mut st = sh.state.lock().unwrap();
            if matches!(*st, EngState::DownloadingModels { .. } | EngState::ModelDownloadPaused { .. }) {
                *st = EngState::Detecting;
            }
        }
    } else {
        sh.cancel_model_dl.store(false, Ordering::SeqCst);
        sh.download_paused.store(false, Ordering::SeqCst);
        let paths = sh.paths.lock().unwrap().clone();
        if !paths.models.as_os_str().is_empty() {
            spawn_model_downloads_if_needed(sh, ctx);
        }
    }
}


pub fn part_path_for(dest: &Path) -> PathBuf {
    dest.with_extension(format!(
        "{}.part",
        dest.extension().and_then(|e| e.to_str()).unwrap_or("tmp")
    ))
}

pub fn get_profile_download_info(sh: &Shared, profile_id: &str) -> Option<ProfileDownloadInfo> {
    let map = sh.profile_downloads.lock().unwrap();
    map.get(profile_id).map(|t| ProfileDownloadInfo {
        profile_name: t.profile.name.clone(),
        status: t.status.clone(),
        progress: t.progress,
        bytes_done: t.bytes_done,
        bytes_total: t.bytes_total,
        speed_mb: t.speed_mb,
        current_file: t.current_file.clone(),
    })
}

pub fn all_profile_download_infos(sh: &Shared) -> Vec<ProfileDownloadInfo> {
    let map = sh.profile_downloads.lock().unwrap();
    map.values()
        .map(|t| ProfileDownloadInfo {
            profile_name: t.profile.name.clone(),
            status: t.status.clone(),
            progress: t.progress,
            bytes_done: t.bytes_done,
            bytes_total: t.bytes_total,
            speed_mb: t.speed_mb,
            current_file: t.current_file.clone(),
        })
        .collect()
}

fn effective_models_dir(sh: &Shared) -> PathBuf {
    sh.pending_models_dir
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(|| sh.paths.lock().unwrap().models.clone())
}

fn profile_download_metrics(profile: &Profile, models_dir: &Path) -> (u64, u64, u64, f32, String) {
    let required: Vec<&ModelFile> = profile.models.iter().filter(|m| !m.optional).collect();
    let mut missing_total_bytes: u64 = 0;
    let mut part_bytes: u64 = 0;
    let mut first_missing: Option<String> = None;

    for m in required {
        let dest = safe_model_path(models_dir, &m.category, &m.filename);
        if !dest.exists() {
            missing_total_bytes += m.size_bytes;
            if first_missing.is_none() {
                first_missing = Some(m.filename.clone());
            }
            let p = part_path_for(&dest);
            let plen = std::fs::metadata(&p).map(|md| md.len()).unwrap_or(0);
            part_bytes += plen.min(m.size_bytes);
        }
    }

    let progress = if missing_total_bytes > 0 {
        (part_bytes as f32 / missing_total_bytes as f32).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0, part_bytes, missing_total_bytes, progress, first_missing.unwrap_or_default())
}

fn persist_profile_download_state(sh: &Shared, profile_id: &str, paused: Option<bool>) {
    let mut cfg = sh.cfg.lock().unwrap();
    let changed = match paused {
        Some(paused_val) => {
            if let Some(entry) = cfg.pending_profile_downloads.iter_mut().find(|e| e.profile_id == profile_id) {
                if entry.paused != paused_val {
                    entry.paused = paused_val;
                    true
                } else {
                    false
                }
            } else {
                cfg.pending_profile_downloads.push(PersistedProfileDownload {
                    profile_id: profile_id.to_string(),
                    paused: paused_val,
                });
                true
            }
        }
        None => {
            let before = cfg.pending_profile_downloads.len();
            cfg.pending_profile_downloads.retain(|e| e.profile_id != profile_id);
            cfg.pending_profile_downloads.len() != before
        }
    };
    if changed {
        #[cfg(not(test))]
        cfg.save();
    }
}

pub fn restore_profile_downloads_if_needed(sh: &Arc<Shared>, ctx: &eframe::egui::Context) {
    let models_dir = effective_models_dir(sh);
    if models_dir.as_os_str().is_empty() {
        return;
    }
    if sh.restored_profile_downloads.swap(true, Ordering::SeqCst) {
        return;
    }

    let pending = sh.cfg.lock().unwrap().pending_profile_downloads.clone();
    if pending.is_empty() {
        return;
    }

    let all_profiles = Profile::all_raw();
    let mut any_ui_update = false;

    for entry in pending {
        let Some(profile) = all_profiles.iter().find(|p| p.id == entry.profile_id) else {
            persist_profile_download_state(sh, &entry.profile_id, None);
            continue;
        };
        if profile.missing_in_dir(&models_dir).is_empty() {
            persist_profile_download_state(sh, &entry.profile_id, None);
            continue;
        }
        if entry.paused {
            let (_, bytes_done, bytes_total, progress, current_file) =
                profile_download_metrics(profile, &models_dir);
            let mut map = sh.profile_downloads.lock().unwrap();
            map.entry(profile.id.clone()).or_insert_with(|| ProfileDownloadTask {
                profile: profile.clone(),
                status: ProfileDownloadStatus::Paused,
                progress,
                bytes_done,
                bytes_total,
                speed_mb: 0.0,
                current_file,
                pause_flag: Arc::new(AtomicBool::new(true)),
                cancel_flag: Arc::new(AtomicBool::new(false)),
                stop_flag: Arc::new(AtomicBool::new(false)),
                running: Arc::new(AtomicBool::new(false)),
            });
            any_ui_update = true;
        } else {
            start_profile_download(sh, profile, ctx);
        }
    }

    if any_ui_update {
        ctx.request_repaint();
    }
}

pub fn start_profile_download(sh: &Arc<Shared>, profile: &Profile, ctx: &eframe::egui::Context) {
    let models_dir = effective_models_dir(sh);
    if models_dir.as_os_str().is_empty() {
        return;
    }
    if profile.missing_in_dir(&models_dir).is_empty() {
        persist_profile_download_state(sh, &profile.id, None);
        return;
    }

    let (_, bytes_done, total_bytes, init_prog, current_file) =
        profile_download_metrics(profile, &models_dir);

    let (pause_flag, cancel_flag, stop_flag, running) = {
        let mut map = sh.profile_downloads.lock().unwrap();
        if let Some(existing) = map.get_mut(&profile.id) {
            if existing.running.load(Ordering::SeqCst) {
                return;
            }
            existing.pause_flag.store(false, Ordering::SeqCst);
            existing.cancel_flag.store(false, Ordering::SeqCst);
            existing.stop_flag.store(false, Ordering::SeqCst);
            existing.running.store(true, Ordering::SeqCst);
            existing.status = ProfileDownloadStatus::Downloading;
            existing.bytes_done = bytes_done;
            existing.bytes_total = total_bytes;
            existing.progress = init_prog;
            existing.current_file = current_file;
            (
                existing.pause_flag.clone(),
                existing.cancel_flag.clone(),
                existing.stop_flag.clone(),
                existing.running.clone(),
            )
        } else {
            let pause_flag = Arc::new(AtomicBool::new(false));
            let cancel_flag = Arc::new(AtomicBool::new(false));
            let stop_flag = Arc::new(AtomicBool::new(false));
            let running = Arc::new(AtomicBool::new(true));
            map.insert(
                profile.id.clone(),
                ProfileDownloadTask {
                    profile: profile.clone(),
                    status: ProfileDownloadStatus::Downloading,
                    progress: init_prog,
                    bytes_done,
                    bytes_total: total_bytes,
                    speed_mb: 0.0,
                    current_file,
                    pause_flag: pause_flag.clone(),
                    cancel_flag: cancel_flag.clone(),
                    stop_flag: stop_flag.clone(),
                    running: running.clone(),
                },
            );
            (pause_flag, cancel_flag, stop_flag, running)
        }
    };

    persist_profile_download_state(sh, &profile.id, Some(false));

    let sh_c = sh.clone();
    let prof_c = profile.clone();
    let ctx_c = ctx.clone();
    std::thread::spawn(move || {
        run_profile_download_worker(
            sh_c,
            prof_c,
            ctx_c,
            pause_flag,
            cancel_flag,
            stop_flag,
            running,
        );
    });
    ctx.request_repaint();
}

pub fn toggle_profile_download_pause(sh: &Arc<Shared>, profile_id: &str, ctx: &eframe::egui::Context) {
    enum Action {
        None,
        Pause,
        Resume(Profile),
    }

    let action = {
        let mut map = sh.profile_downloads.lock().unwrap();
        if let Some(task) = map.get_mut(profile_id) {
            match task.status {
                ProfileDownloadStatus::Downloading => {
                    task.pause_flag.store(true, Ordering::SeqCst);
                    task.stop_flag.store(true, Ordering::SeqCst);
                    task.status = ProfileDownloadStatus::Paused;
                    task.speed_mb = 0.0;
                    Action::Pause
                }
                ProfileDownloadStatus::Paused | ProfileDownloadStatus::Failed(_) => {
                    Action::Resume(task.profile.clone())
                }
            }
        } else {
            Action::None
        }
    };

    match action {
        Action::Pause => {
            persist_profile_download_state(sh, profile_id, Some(true));
            ctx.request_repaint();
        }
        Action::Resume(profile) => {
            // Wait briefly if previous worker is still winding down
            for _ in 0..40 {
                let is_running = sh
                    .profile_downloads
                    .lock()
                    .unwrap()
                    .get(profile_id)
                    .map(|t| t.running.load(Ordering::SeqCst))
                    .unwrap_or(false);
                if !is_running {
                    break;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            sh.download_paused.store(false, Ordering::SeqCst);
            start_profile_download(sh, &profile, ctx);
        }
        Action::None => {}
    }
}

pub fn cancel_profile_download(sh: &Arc<Shared>, profile_id: &str) {
    let running_arc = {
        let mut map = sh.profile_downloads.lock().unwrap();
        if let Some(task) = map.get_mut(profile_id) {
            task.cancel_flag.store(true, Ordering::SeqCst);
            task.stop_flag.store(true, Ordering::SeqCst);
            Some(task.running.clone())
        } else {
            None
        }
    };

    if let Some(running) = running_arc {
        for _ in 0..80 {
            if !running.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    sh.profile_downloads.lock().unwrap().remove(profile_id);
    persist_profile_download_state(sh, profile_id, None);
}

struct ActiveFileGuard(Arc<Shared>, PathBuf);
impl Drop for ActiveFileGuard {
    fn drop(&mut self) {
        self.0.active_file_downloads.lock().unwrap().remove(&self.1);
    }
}

struct RunningFlagGuard(Arc<AtomicBool>);
impl Drop for RunningFlagGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

fn run_profile_download_worker(
    sh: Arc<Shared>,
    profile: Profile,
    ctx: eframe::egui::Context,
    pause_flag: Arc<AtomicBool>,
    cancel_flag: Arc<AtomicBool>,
    stop_flag: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
) {
    let _running_guard = RunningFlagGuard(running);

    let models_dir = effective_models_dir(&sh);

    if models_dir.as_os_str().is_empty() {
        if let Some(task) = sh.profile_downloads.lock().unwrap().get_mut(&profile.id) {
            task.status = ProfileDownloadStatus::Failed("Models directory not configured".into());
        }
        ctx.request_repaint();
        return;
    }

    if profile.is_ltx25() {
        ensure_ltx25_gguf_sidecar(&models_dir);
    }

    let missing = profile.missing_in_dir(&models_dir);
    if missing.is_empty() {
        sh.profile_downloads.lock().unwrap().remove(&profile.id);
        persist_profile_download_state(&sh, &profile.id, None);
        ctx.request_repaint();
        return;
    }

    let (mut completed_bytes, _, total_all_bytes, _, _) =
        profile_download_metrics(&profile, &models_dir);
    let dl_client = build_dl_client();

    for m in &missing {
        if cancel_flag.load(Ordering::SeqCst) || sh.cancel.load(Ordering::SeqCst) {
            sh.profile_downloads.lock().unwrap().remove(&profile.id);
            if cancel_flag.load(Ordering::SeqCst) {
                persist_profile_download_state(&sh, &profile.id, None);
            }
            ctx.request_repaint();
            return;
        }
        if pause_flag.load(Ordering::SeqCst) || sh.download_paused.load(Ordering::SeqCst) {
            if let Some(task) = sh.profile_downloads.lock().unwrap().get_mut(&profile.id) {
                task.status = ProfileDownloadStatus::Paused;
                task.speed_mb = 0.0;
            }
            ctx.request_repaint();
            return;
        }

        let dest = safe_model_path(&models_dir, &m.category, &m.filename);

        // Coordinate shared files across parallel profile downloads
        let _file_guard = loop {
            if dest.exists() {
                break None;
            }
            if cancel_flag.load(Ordering::SeqCst) || sh.cancel.load(Ordering::SeqCst) {
                sh.profile_downloads.lock().unwrap().remove(&profile.id);
                if cancel_flag.load(Ordering::SeqCst) {
                    persist_profile_download_state(&sh, &profile.id, None);
                }
                ctx.request_repaint();
                return;
            }
            if pause_flag.load(Ordering::SeqCst) || sh.download_paused.load(Ordering::SeqCst) {
                if let Some(task) = sh.profile_downloads.lock().unwrap().get_mut(&profile.id) {
                    task.status = ProfileDownloadStatus::Paused;
                    task.speed_mb = 0.0;
                }
                ctx.request_repaint();
                return;
            }
            let claimed = sh.active_file_downloads.lock().unwrap().insert(dest.clone());
            if claimed {
                break Some(ActiveFileGuard(sh.clone(), dest.clone()));
            }
            // Another worker is downloading this shared file; reflect its .part progress
            let part_len = std::fs::metadata(part_path_for(&dest))
                .map(|md| md.len())
                .unwrap_or(0);
            let done_total = completed_bytes + part_len.min(m.size_bytes);
            let prog = if total_all_bytes > 0 {
                (done_total as f32 / total_all_bytes as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
            if let Some(task) = sh.profile_downloads.lock().unwrap().get_mut(&profile.id) {
                task.current_file = m.filename.clone();
                task.bytes_done = done_total;
                task.bytes_total = total_all_bytes;
                task.progress = prog;
            }
            ctx.request_repaint();
            std::thread::sleep(Duration::from_millis(250));
        };

        if dest.exists() {
            completed_bytes += m.size_bytes;
            let prog = if total_all_bytes > 0 {
                (completed_bytes as f32 / total_all_bytes as f32).clamp(0.0, 1.0)
            } else {
                1.0
            };
            if let Some(task) = sh.profile_downloads.lock().unwrap().get_mut(&profile.id) {
                task.bytes_done = completed_bytes;
                task.bytes_total = total_all_bytes;
                task.progress = prog;
            }
            continue;
        }

        log(&format!("profile_dl [{}]: downloading {} -> {}", profile.id, m.filename, dest.display()));

        let hf_mirror_url = if m.hf_url.starts_with("https://huggingface.co/") {
            m.hf_url.replace("https://huggingface.co/", "https://hf-mirror.com/")
        } else {
            String::new()
        };
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
            if stop_flag.load(Ordering::SeqCst) || sh.cancel.load(Ordering::SeqCst) || sh.download_paused.load(Ordering::SeqCst) {
                break;
            }
            let enable_watchdog = c_idx + 1 < total_candidates;
            download_result = download_stream(
                &dl_client,
                &url,
                &dest,
                m.size_bytes,
                |done, _total, speed| {
                    let done_total = completed_bytes + done.min(m.size_bytes);
                    let prog = if total_all_bytes > 0 {
                        (done_total as f32 / total_all_bytes as f32).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    if let Some(task) = sh.profile_downloads.lock().unwrap().get_mut(&profile.id) {
                        task.current_file = m.filename.clone();
                        task.bytes_done = done_total;
                        task.bytes_total = total_all_bytes;
                        task.progress = prog;
                        task.speed_mb = speed;
                    }
                },
                stop_flag.as_ref(),
                &ctx,
                enable_watchdog,
                Some(&sh),
            );
            if download_result.is_ok() || stop_flag.load(Ordering::SeqCst) || sh.download_paused.load(Ordering::SeqCst) {
                if download_result.is_ok() {
                    log(&format!("profile_dl [{}]: downloaded {} via {}", profile.id, m.filename, source_name));
                }
                break;
            } else {
                log(&format!("profile_dl [{}]: {} failed from {}: {:?}", profile.id, m.filename, source_name, download_result.as_ref().err()));
            }
        }

        if cancel_flag.load(Ordering::SeqCst) || sh.cancel.load(Ordering::SeqCst) {
            sh.profile_downloads.lock().unwrap().remove(&profile.id);
            if cancel_flag.load(Ordering::SeqCst) {
                persist_profile_download_state(&sh, &profile.id, None);
            }
            ctx.request_repaint();
            return;
        }

        if pause_flag.load(Ordering::SeqCst) || sh.download_paused.load(Ordering::SeqCst) {
            let part_len = std::fs::metadata(part_path_for(&dest))
                .map(|md| md.len())
                .unwrap_or(0);
            let done_total = completed_bytes + part_len.min(m.size_bytes);
            let prog = if total_all_bytes > 0 {
                (done_total as f32 / total_all_bytes as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
            if let Some(task) = sh.profile_downloads.lock().unwrap().get_mut(&profile.id) {
                task.status = ProfileDownloadStatus::Paused;
                task.bytes_done = done_total;
                task.bytes_total = total_all_bytes;
                task.progress = prog;
                task.speed_mb = 0.0;
            }
            ctx.request_repaint();
            return;
        }

        if let Err(e) = download_result {
            if let Some(task) = sh.profile_downloads.lock().unwrap().get_mut(&profile.id) {
                task.status = ProfileDownloadStatus::Failed(e);
                task.speed_mb = 0.0;
            }
            ctx.request_repaint();
            return;
        }

        completed_bytes += m.size_bytes;
    }

    sh.profile_downloads.lock().unwrap().remove(&profile.id);
    persist_profile_download_state(&sh, &profile.id, None);
    ctx.request_repaint();
}

pub fn start_supervisor(sh: Arc<Shared>, _profile: Profile, ctx: eframe::egui::Context) {
    std::thread::spawn(move || {
        let ping_client = reqwest::blocking::Client::builder().timeout(Duration::from_millis(3500)).build().unwrap();
        let dl_client = build_dl_client();

        let mut child: Option<Child> = None;
        let mut attempt = 0usize;
        let mut started = Instant::now();
        let mut failed = false;
        let mut isolated_attempted = false;
        let mut auto_repair_attempted = false;
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
                auto_repair_attempted = false;
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
                let pys = comfy.as_ref().map(|c| find_pythons(&cfg, c)).unwrap_or_default();
                if child.is_none() {
                    *sh.paths.lock().unwrap() = Paths { comfy: comfy.clone(), python: pys.first().cloned(), models: models.clone() };
                }
                restore_profile_downloads_if_needed(&sh, &ctx);
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

            if sh.download_paused.load(Ordering::SeqCst) {
                ctx.request_repaint();
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }

            // 2. ComfyUI missing? Bootstrap automatically (requires confirmation if on metered connection)
            if comfy.is_none() {
                if is_metered_connection() && !sh.metered_allowed.load(Ordering::SeqCst) && !sh.download_confirmed.load(Ordering::SeqCst) {
                    *sh.state.lock().unwrap() = EngState::AwaitingDownloadConfirmation { metered: true };
                    ctx.request_repaint();
                    std::thread::sleep(Duration::from_millis(200));
                    continue;
                }
                isolated_attempted = true;
                match bootstrap_engine(&sh, &ctx, &dl_client, clean_requested) {
                    Ok(c) => {
                        comfy = Some(c.clone());
                        cfg.comfy_dir = Some(c);
                        cfg.save();
                        *sh.cfg.lock().unwrap() = cfg.clone();
                    }
                    Err(e) => {
                        if sh.download_paused.load(Ordering::SeqCst) {
                            ctx.request_repaint();
                            std::thread::sleep(Duration::from_millis(200));
                            continue;
                        }
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
                let pys = find_pythons(&cfg, cd);
                *sh.paths.lock().unwrap() = Paths { comfy: Some(cd.clone()), python: pys.first().cloned(), models: models.clone() };
            }
            restore_profile_downloads_if_needed(&sh, &ctx);

            // 3. Start ticked model downloads in background (requires confirmation if on metered connection)
            let missing_models = Profile::missing_models(&sh);
            if !missing_models.is_empty() {
                if is_metered_connection() && !sh.metered_allowed.load(Ordering::SeqCst) && !sh.download_confirmed.load(Ordering::SeqCst) {
                    *sh.state.lock().unwrap() = EngState::AwaitingDownloadConfirmation { metered: true };
                    ctx.request_repaint();
                    std::thread::sleep(Duration::from_millis(200));
                    continue;
                }
                spawn_model_downloads_if_needed(&sh, &ctx);
            }

            *sh.models_found.lock().unwrap() = profile
                .models
                .iter()
                .map(|m| (m.filename.clone(), models.join(&m.category).join(&m.filename).exists()))
                .collect();

            // 4. Launch ComfyUI process
            if child.is_none() {
                let pys = find_pythons(&cfg, cd);
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
                    } else if !auto_repair_attempted && !pys.is_empty() {
                        log(&format!("all Python candidates failed in {}; attempting dependency repair on primary python\n{t}", cd.display()));
                        auto_repair_attempted = true;
                        let primary_py = &pys[0];
                        match install_or_repair_dependencies(primary_py, cd, &sh, &ctx) {
                            Ok(()) => {
                                log("automated repair succeeded; retrying primary python");
                                attempt = 0;
                                continue;
                            }
                            Err(e) => {
                                failed = true;
                                *sh.state.lock().unwrap() = EngState::Failed(format!("ComfyUI failed to start and repair failed: {e}\n{t}"));
                            }
                        }
                    } else {
                        failed = true;
                        *sh.state.lock().unwrap() = EngState::Failed(format!("ComfyUI failed to start.\n{t}"));
                    }
                } else {
                    let py = &pys[attempt];
                    sh.paths.lock().unwrap().python = Some(py.clone());

                    // If attempt 0, check if this is a managed/local environment and probe for critical dependencies
                    let is_managed = cd.starts_with(data_dir());
                    let is_local_env = py.starts_with(cd);
                    if attempt == 0 && !auto_repair_attempted && (is_managed || is_local_env) {
                        let probe = Command::new(py)
                            .args(["-c", "import comfy_aimdo; import torch"])
                            .current_dir(cd)
                            .creation_flags(CREATE_NO_WINDOW)
                            .output();
                        let deps_ok = probe.map(|o| o.status.success()).unwrap_or(false);
                        if !deps_ok {
                            log(&format!("dependency probe in {} detected missing modules (comfy_aimdo/torch); initiating auto-repair", py.display()));
                            auto_repair_attempted = true;
                            if let Err(e) = install_or_repair_dependencies(py, cd, &sh, &ctx) {
                                log(&format!("pre-launch dependency repair failed: {e}"));
                            }
                        }
                    }

                    ensure_gguf_custom_node(cd, Some(py));
                    ensure_viggle_remake_custom_node(cd);
                    ensure_ltx25_gguf_sidecar(&models);

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
                    if exited || (!ever_ready && started.elapsed() > Duration::from_secs(180)) {
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
                            let log_tail = tail(&data_dir().join("comfyui.log"), 25);
                            let has_module_err = log_tail.contains("ModuleNotFoundError") || log_tail.contains("No module named") || log_tail.contains("ImportError");
                            let pys = find_pythons(&cfg, cd);
                            if has_module_err && !auto_repair_attempted && !pys.is_empty() {
                                log(&format!("detected missing Python module in comfyui.log; attempting dependency repair on primary python:\n{log_tail}"));
                                auto_repair_attempted = true;
                                let primary_py = &pys[0];
                                if let Ok(()) = install_or_repair_dependencies(primary_py, cd, &sh, &ctx) {
                                    log("repair succeeded after module crash; retrying primary python");
                                    attempt = 0;
                                    continue;
                                }
                            }
                            attempt += 1;
                        }
                    } else if !ever_ready {
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
#[cfg(windows)]
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
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK;
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

#[cfg(not(windows))]
pub fn adopt_into_kill_job() {
    // No-op on non-Windows
}

/// Checks if the primary network connection is metered to prevent unconfirmed large downloads.
#[cfg(windows)]
pub fn is_metered_connection() -> bool {
    use windows_sys::Win32::System::Com::*;
    use windows_sys::core::GUID;

    const CLSID_NETWORK_LIST_MANAGER: GUID = GUID {
        data1: 0xDCB00C01,
        data2: 0x570F,
        data3: 0x4A9B,
        data4: [0x8D, 0x69, 0x19, 0x9F, 0xDB, 0xA5, 0x72, 0x3B],
    };
    const IID_INETWORK_COST_MANAGER: GUID = GUID {
        data1: 0xDCB00008,
        data2: 0x570F,
        data3: 0x4A9B,
        data4: [0x8D, 0x69, 0x19, 0x9F, 0xDB, 0xA5, 0x72, 0x3B],
    };

    unsafe {
        let _ = CoInitializeEx(std::ptr::null(), COINIT_MULTITHREADED as u32);
        let mut obj: *mut std::ffi::c_void = std::ptr::null_mut();
        let hr = CoCreateInstance(
            &CLSID_NETWORK_LIST_MANAGER,
            std::ptr::null_mut(),
            CLSCTX_ALL,
            &IID_INETWORK_COST_MANAGER,
            &mut obj,
        );
        if hr == 0 && !obj.is_null() {
            type GetCostFn = unsafe extern "system" fn(*mut std::ffi::c_void, *mut u32, *const std::ffi::c_void) -> i32;
            type ReleaseFn = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;

            let vtbl = *(obj as *const *const usize);
            let get_cost: GetCostFn = std::mem::transmute(*vtbl.add(3));
            let release: ReleaseFn = std::mem::transmute(*vtbl.add(2));

            let mut cost: u32 = 0;
            let hr_cost = get_cost(obj, &mut cost, std::ptr::null());
            release(obj);
            if hr_cost == 0 {
                const METERED_MASK: u32 = 0x2 | 0x4 | 0x10000 | 0x20000 | 0x40000 | 0x80000;
                return (cost & METERED_MASK) != 0;
            }
        }
    }
    false
}

#[cfg(not(windows))]
pub fn is_metered_connection() -> bool {
    false
}

/// Flashes the application taskbar button to draw immediate user attention.
#[cfg(windows)]
pub fn flash_taskbar() {
    use windows_sys::Win32::UI::WindowsAndMessaging::*;
    unsafe {
        let hwnd = GetForegroundWindow();
        if !hwnd.is_null() {
            let mut fwi = FLASHWINFO {
                cbSize: std::mem::size_of::<FLASHWINFO>() as u32,
                hwnd,
                dwFlags: FLASHW_ALL | FLASHW_TIMERNOFG,
                uCount: 6,
                dwTimeout: 0,
            };
            FlashWindowEx(&mut fwi);
        }
    }
}

#[cfg(not(windows))]
pub fn flash_taskbar() {}

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
            &AtomicBool::new(false), &eframe::egui::Context::default(), None)
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

    #[test]
    fn test_metered_connection_detection() {
        let is_metered = is_metered_connection();
        println!("Metered connection: {is_metered}");
    }

    #[test]
    fn test_config_python_path_serialization() {
        let mut cfg = Config::default();
        cfg.python_path = Some(PathBuf::from("/usr/bin/custom-python3"));
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("/usr/bin/custom-python3"));

        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.python_path, Some(PathBuf::from("/usr/bin/custom-python3")));
    }

    #[test]
    fn test_find_pythons_prioritizes_configured_path() {
        let root = sandbox();
        let comfy = root.join("ComfyUI");
        std::fs::create_dir_all(&comfy).unwrap();
        std::fs::write(comfy.join("main.py"), "print('comfy')").unwrap();

        let custom_py = root.join("my_python.exe");
        std::fs::write(&custom_py, "#!/bin/sh\n").unwrap();

        let mut cfg = Config::default();
        cfg.python_path = Some(custom_py.clone());

        let candidates = find_pythons(&cfg, &comfy);
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0], custom_py);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn test_set_model_dl_state_preserves_engine_setup_states() {
        let sh = Shared::new();
        *sh.state.lock().unwrap() = EngState::RepairingEngine {
            step: "Installing PyTorch...".into(),
        };

        let prog = ModelDownloadProgress {
            current_file: "qwen.safetensors".into(),
            file_index: 1,
            total_files: 3,
            file_progress: 0.4,
            bytes_done: 400,
            bytes_total: 1000,
            speed_mb: 42.0,
            overall_progress: 0.2,
        };
        set_model_dl_state(
            &sh,
            Some(prog),
            Some(EngState::DownloadingModels {
                current_file: "qwen.safetensors".into(),
                file_index: 1,
                total_files: 3,
                file_progress: 0.4,
                bytes_done: 400,
                bytes_total: 1000,
                speed_mb: 42.0,
                overall_progress: 0.2,
            }),
        );

        assert!(matches!(*sh.state.lock().unwrap(), EngState::RepairingEngine { .. }));
        assert_eq!(sh.model_dl_progress.lock().unwrap().as_ref().unwrap().current_file, "qwen.safetensors");
    }

    #[test]
    fn test_untick_image_model_with_video_unticked_cancels_cleanly() {
        let sh = Shared::new();
        sh.dl_image_model.store(true, Ordering::SeqCst);
        sh.dl_video_model.store(false, Ordering::SeqCst);
        sh.model_dl_running.store(true, Ordering::SeqCst);
        *sh.state.lock().unwrap() = EngState::DownloadingModels {
            current_file: "qwen.safetensors".into(),
            file_index: 1,
            total_files: 3,
            file_progress: 0.1,
            bytes_done: 100,
            bytes_total: 1000,
            speed_mb: 10.0,
            overall_progress: 0.05,
        };

        let ctx = eframe::egui::Context::default();
        toggle_model_download(&sh, &ctx, true, false);

        assert!(!sh.dl_image_model.load(Ordering::SeqCst));
        assert!(sh.cancel_model_dl.load(Ordering::SeqCst));
        assert!(sh.model_dl_progress.lock().unwrap().is_none());
        assert!(matches!(*sh.state.lock().unwrap(), EngState::Detecting));
    }

    #[test]
    fn summarize_pip_line_extracts_progress() {
        assert_eq!(
            summarize_pip_line("Collecting torch>=2.0"),
            Some("Collecting torch".to_string())
        );
        assert_eq!(
            summarize_pip_line("  Downloading https://example.com/torch-2.6.0-cp312.whl (2.5 GB)"),
            Some("Downloading torch-2.6.0-cp312.whl".to_string())
        );
        assert_eq!(
            summarize_pip_line("Installing collected packages: torch, comfy-aimdo"),
            Some("Unpacking & installing packages...".to_string())
        );
        assert_eq!(summarize_pip_line("Requirement already satisfied: pip"), None);
    }

    #[test]
    fn video_and_image_profiles_partition_cleanly() {
        let img = Profile::all();
        let vid = Profile::all_video_profiles();
        assert!(!img.is_empty());
        assert!(img.iter().all(|p| !Profile::is_video_profile_id(&p.id)));
        assert!(vid.iter().all(|p| Profile::is_video_profile_id(&p.id)));
        for expected in [
            "minimax_h3_10gb",
            "minimax_h3_ud_q3_k_xl",
            "minimax_h3_12gb",
            "minimax_h3_q5_k_m",
            "minimax_h3_16gb",
            "minimax_h3_24gb",
            "ltx_2.5_q2_k_xl",
            "ltx_2.5_10gb",
            "ltx_2.5_12gb",
            "ltx_2.5_16gb",
            "ltx_2.5_24gb",
            "ltx_2.3",
        ] {
            assert!(vid.iter().any(|p| p.id == expected), "missing video profile {expected}");
        }

        let h3_16 = vid.iter().find(|p| p.id == "minimax_h3_16gb").unwrap();
        assert!(h3_16.is_minimax_h3());
        assert_eq!(h3_16.fl2va_unet_name(), "MiniMax-H3-FL2VA-Pruned-Q6_K.gguf");
        assert_eq!(h3_16.ref2va_unet_name(), "MiniMax-H3-Ref2VA-Pruned-Q6_K.gguf");
        assert_eq!(h3_16.clip_name(), "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors");
        assert_eq!(h3_16.video_vae_name(), "minimax_h3_video_vae_int8_convrot.safetensors");
        assert_eq!(h3_16.audio_vae_name(), "minimax_h3_audio_vae_fp32.safetensors");
        assert_eq!(h3_16.default_steps, 20);
        assert_eq!(h3_16.sampler, "er_sde");
        assert_eq!(h3_16.scheduler, "sgm_uniform");

        // Ref2VA is optional and excluded from required size & missing_in_dir
        let r2v_file = h3_16.r2v_model_file().expect("h3_16 has optional r2v file");
        assert!(r2v_file.optional);
        assert_eq!(r2v_file.filename, "MiniMax-H3-Ref2VA-Pruned-Q6_K.gguf");
        let root = sandbox();
        assert!(!h3_16.missing_in_dir(&root).iter().any(|m| m.filename == r2v_file.filename));
        assert!(!h3_16.has_r2v_model(&root));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn optional_r2v_model_downloads_only_when_requested_and_very_last() {
        let sh = Shared::new();
        *sh.active_video_profile.lock().unwrap() = Profile::all_video_profiles()
            .into_iter()
            .find(|p| p.id == "minimax_h3_16gb")
            .unwrap();
        sh.dl_image_model.store(true, Ordering::SeqCst);
        sh.dl_video_model.store(true, Ordering::SeqCst);
        sh.dl_r2v_model.store(false, Ordering::SeqCst);

        let without_r2v = Profile::target_download_models(&sh);
        assert!(!without_r2v.iter().any(|m| m.optional));

        sh.dl_r2v_model.store(true, Ordering::SeqCst);
        let with_r2v = Profile::target_download_models(&sh);
        assert!(with_r2v.last().is_some_and(|m| m.optional && m.filename.contains("Ref2VA")));
        assert_eq!(with_r2v.iter().filter(|m| m.optional).count(), 1);
    }

    #[test]
    fn ltx25_sidecar_writes_valid_schema_and_lora_scan_works() {
        let root = sandbox();
        ensure_ltx25_gguf_sidecar(&root);
        let sidecar = root.join("diffusion_models").join("ltx-2.5-transformer-metadata.json");
        assert!(sidecar.is_file());
        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&sidecar).unwrap()).unwrap();
        assert!(parsed["applies_to"].is_array());
        assert_eq!(parsed["config"]["transformer"]["num_layers"], 48);
        assert_eq!(parsed["config"]["transformer"]["cross_attention_adaln"], true);
        assert_eq!(parsed["config"]["transformer"]["rope_type"], "split");
        assert_eq!(parsed["config"]["transformer"]["causal_temporal_positioning"], true);
        assert_eq!(parsed["config"]["transformer"]["use_middle_indices_grid"], true);

        let loras_dir = root.join("loras");
        std::fs::create_dir_all(loras_dir.join("sub")).unwrap();
        std::fs::write(loras_dir.join("turbo.safetensors"), b"x").unwrap();
        std::fs::write(loras_dir.join("sub").join("style.safetensors"), b"x").unwrap();
        std::fs::write(loras_dir.join("ignore.txt"), b"x").unwrap();
        let mut found = Vec::new();
        collect_safetensors_rel(&loras_dir, &loras_dir, 0, &mut found);
        found.sort();
        assert_eq!(found, vec!["sub/style.safetensors", "turbo.safetensors"]);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn config_persists_pending_profile_downloads_and_defaults_safely() {
        let legacy_json = r#"{"comfy_dir":null,"models_dir":null}"#;
        let legacy: Config = serde_json::from_str(legacy_json).unwrap();
        assert!(legacy.pending_profile_downloads.is_empty());

        let mut cfg = Config::default();
        cfg.pending_profile_downloads.push(PersistedProfileDownload {
            profile_id: "flux_schnell".into(),
            paused: false,
        });
        cfg.pending_profile_downloads.push(PersistedProfileDownload {
            profile_id: "minimax_h3_16gb".into(),
            paused: true,
        });
        let raw = serde_json::to_string(&cfg).unwrap();
        let roundtrip: Config = serde_json::from_str(&raw).unwrap();
        assert_eq!(roundtrip.pending_profile_downloads, cfg.pending_profile_downloads);
    }

    #[test]
    fn restore_profile_downloads_restores_paused_progress_and_prunes_invalid() {
        let root = sandbox();
        let profiles = Profile::all_raw();
        let target = profiles
            .iter()
            .find(|p| p.models.iter().filter(|m| !m.optional).count() >= 2)
            .expect("need a multi-file profile")
            .clone();
        let req: Vec<&ModelFile> = target.models.iter().filter(|m| !m.optional).collect();
        let first = req[0];
        let second = req[1];

        let first_dest = safe_model_path(&root, &first.category, &first.filename);
        std::fs::create_dir_all(first_dest.parent().unwrap()).unwrap();
        std::fs::write(&first_dest, b"done").unwrap();

        let second_dest = safe_model_path(&root, &second.category, &second.filename);
        std::fs::create_dir_all(second_dest.parent().unwrap()).unwrap();
        let partial_len = (second.size_bytes / 2).min(1024).max(1);
        std::fs::write(part_path_for(&second_dest), vec![0u8; partial_len as usize]).unwrap();

        let (already_done, bytes_done, bytes_total, prog, current_file) =
            profile_download_metrics(&target, &root);
        assert_eq!(already_done, 0);
        assert_eq!(bytes_done, partial_len);
        assert!(bytes_total >= bytes_done);
        assert!(prog > 0.0 && prog < 1.0);
        assert_eq!(current_file, second.filename);

        let sh = Shared::new();
        *sh.pending_models_dir.lock().unwrap() = Some(root.clone());
        {
            let mut cfg = sh.cfg.lock().unwrap();
            cfg.pending_profile_downloads = vec![
                PersistedProfileDownload {
                    profile_id: target.id.clone(),
                    paused: true,
                },
                PersistedProfileDownload {
                    profile_id: "nonexistent_profile_id".into(),
                    paused: false,
                },
            ];
        }

        let ctx = eframe::egui::Context::default();
        restore_profile_downloads_if_needed(&sh, &ctx);

        let info = get_profile_download_info(&sh, &target.id).expect("paused download restored");
        assert_eq!(info.status, ProfileDownloadStatus::Paused);
        assert_eq!(info.bytes_done, bytes_done);
        assert_eq!(info.bytes_total, bytes_total);
        assert!((info.progress - prog).abs() < 1e-6);
        assert_eq!(info.current_file, second.filename);
        assert!(get_profile_download_info(&sh, "nonexistent_profile_id").is_none());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pick_best_from_downloaded_prefers_higher_quality_fitting_vram_over_q2() {
        let vid = Profile::all_video_profiles();
        let q2 = vid.iter().find(|p| p.id == "ltx_2.5_q2_k_xl").unwrap().clone();
        let q3 = vid.iter().find(|p| p.id == "ltx_2.5_10gb").unwrap().clone();
        let q6 = vid.iter().find(|p| p.id == "ltx_2.5_16gb").unwrap().clone();

        let best_12gb = Profile::pick_best_from_downloaded(&[q2.clone(), q3.clone(), q6.clone()], Some(12))
            .expect("should pick best fitting");
        assert_eq!(best_12gb.id, "ltx_2.5_10gb");

        let best_8gb = Profile::pick_best_from_downloaded(&[q2, q3], Some(8))
            .expect("should pick q2 on 8gb");
        assert_eq!(best_8gb.id, "ltx_2.5_q2_k_xl");
    }
}
