//! Finds, launches (hidden) and supervises ComfyUI. Runs on its own thread and
//! never touches the UI thread; the UI only reads `Shared`.
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
    NotInstalled,
    Starting { since: Instant, attempt: usize },
    Ready { ours: bool, gpu: String },
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
    pub cfg: Mutex<Config>,
}

impl Shared {
    pub fn new() -> Arc<Shared> {
        Arc::new(Shared {
            state: Mutex::new(EngState::Detecting),
            paths: Mutex::new(Paths::default()),
            models_found: Mutex::new(vec![]),
            restart: AtomicBool::new(false),
            cfg: Mutex::new(Config::load()),
        })
    }
}

fn find_comfy(cfg: &Config) -> Option<PathBuf> {
    let mut c: Vec<PathBuf> = vec![];
    if let Some(p) = &cfg.comfy_dir {
        c.push(p.clone());
        c.push(p.join("ComfyUI"));
    }
    if let Some(h) = dirs::home_dir() {
        c.push(h.join("ComfyUI-Installs/ComfyUI/ComfyUI"));
        c.push(h.join("ComfyUI"));
        c.push(h.join("Documents/ComfyUI"));
        c.push(h.join("ComfyUI_windows_portable/ComfyUI"));
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

fn find_models(cfg: &Config, comfy: &Path) -> PathBuf {
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
    cmd.args(["-u", "main.py", "--port", &PORT.to_string(), "--listen", "127.0.0.1", "--disable-auto-launch", "--preview-method", "auto", "--models-directory"])
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

pub fn start_supervisor(sh: Arc<Shared>, profile: Profile, ctx: eframe::egui::Context) {
    std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder().timeout(Duration::from_millis(1200)).build().unwrap();
        let mut child: Option<Child> = None;
        let mut attempt = 0usize;
        let mut started = Instant::now();
        let mut failed = false;
        loop {
            if sh.restart.swap(false, Ordering::SeqCst) {
                if let Some(c) = child.as_mut() {
                    kill_tree(c);
                }
                child = None;
                attempt = 0;
                failed = false;
                log("restart requested");
            }
            let cfg = sh.cfg.lock().unwrap().clone();
            let comfy = find_comfy(&cfg);
            let models = comfy.as_ref().map(|c| find_models(&cfg, c)).unwrap_or_default();
            {
                let pys = comfy.as_ref().map(|c| find_pythons(c)).unwrap_or_default();
                *sh.paths.lock().unwrap() = Paths { comfy: comfy.clone(), python: pys.first().cloned(), models: models.clone() };
                *sh.models_found.lock().unwrap() = profile
                    .models
                    .iter()
                    .map(|m| (m.filename.clone(), models.join(&m.category).join(&m.filename).exists()))
                    .collect();
            }

            let new_state: Option<EngState> = if let Some(gpu) = ping(&client) {
                attempt = 0;
                failed = false;
                Some(EngState::Ready { ours: child.is_some(), gpu })
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
                    }
                }
                if child.is_none() && !failed {
                    match &comfy {
                        None => Some(EngState::NotInstalled),
                        Some(cd) => {
                            let pys = find_pythons(cd);
                            if pys.is_empty() {
                                failed = true;
                                Some(EngState::Failed("No Python environment found next to ComfyUI.".into()))
                            } else if attempt > pys.len() {
                                failed = true;
                                let t = tail(&data_dir().join("comfyui.log"), 14);
                                Some(EngState::Failed(format!("ComfyUI failed to start.\n{t}")))
                            } else {
                                let py = &pys[attempt % pys.len()];
                                log(&format!("launching {} (attempt {attempt})", py.display()));
                                match spawn(py, cd, &models) {
                                    Ok(c) => {
                                        child = Some(c);
                                        started = Instant::now();
                                        Some(EngState::Starting { since: started, attempt })
                                    }
                                    Err(e) => {
                                        failed = true;
                                        Some(EngState::Failed(e))
                                    }
                                }
                            }
                        }
                    }
                } else if child.is_some() {
                    Some(EngState::Starting { since: started, attempt })
                } else {
                    None
                }
            };
            if let Some(s) = new_state {
                *sh.state.lock().unwrap() = s;
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
        // intentionally leak the handle: closing it (process exit) kills children
    }
}
