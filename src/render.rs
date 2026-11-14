use crate::presets::QualityPreset;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Clone, Debug)]
pub struct RenderJobRequest {
    pub id: u64,
    pub input_path: PathBuf,
    pub trim_start: Option<f64>,
    pub trim_end: Option<f64>,
    pub total_duration: f64,
    pub compress: bool,
    pub preset: QualityPreset,
    pub replace_in_place: bool,
    pub normalise_audio: bool,
    pub audio_normalise_ratio: f32,
    pub audio_normalise_thresh_db: f32,
    pub audio_normalise_boost_db: f32,
    pub display_name: Option<String>,
    pub explicit_output_path: Option<PathBuf>,
    pub temp_input_guard: Option<Arc<crate::video::TempFileGuard>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum JobState {
    Queued,
    Running { progress: f32, speed: String },
    Finished,
    Deleted,
    Failed(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobProgressInfo {
    pub job_id: u64,
    pub file_name: String,
    pub input_path: PathBuf,
    #[serde(default)]
    pub output_path: Option<PathBuf>,
    pub backup_file_name: Option<String>,
    pub state: JobState,
    #[serde(skip)]
    pub finished_at: Option<std::time::Instant>,
    pub is_undone: bool,
    #[serde(skip)]
    pub undone_at: Option<std::time::Instant>,
}

#[allow(dead_code)]
pub enum WorkerMessage {
    Enqueue(RenderJobRequest),
    Cancel(u64),
}

#[allow(dead_code)]
pub enum WorkerEvent {
    JobStarted(u64, String),
    JobProgress(u64, f32, String),
    JobFinished(u64, PathBuf, Option<String>),
    JobFailed(u64, String),
}

pub struct RenderManager {
    sender: Sender<WorkerMessage>,
    pub events_rx: Receiver<WorkerEvent>,
    pub active_jobs: Arc<Mutex<Vec<JobProgressInfo>>>,
    next_id: AtomicU64,
}

impl RenderManager {
    pub fn new() -> Self {
        let (tx_msg, rx_msg) = channel::<WorkerMessage>();
        let (tx_evt, rx_evt) = channel::<WorkerEvent>();
        let active_jobs = Arc::new(Mutex::new(Vec::new()));

        let active_jobs_worker = Arc::clone(&active_jobs);
        thread::spawn(move || {
            worker_loop(rx_msg, tx_evt, active_jobs_worker);
        });

        Self {
            sender: tx_msg,
            events_rx: rx_evt,
            active_jobs,
            next_id: AtomicU64::new(1),
        }
    }

    pub fn submit_job(&self, mut req: RenderJobRequest) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        req.id = id;

        let file_name = req.display_name.clone().unwrap_or_else(|| {
            req.input_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("video")
                .to_string()
        });

        if let Ok(mut jobs) = self.active_jobs.lock() {
            jobs.push(JobProgressInfo {
                job_id: id,
                file_name,
                input_path: req.input_path.clone(),
                output_path: req.explicit_output_path.clone(),
                backup_file_name: None,
                state: JobState::Queued,
                finished_at: None,
                is_undone: false,
                undone_at: None,
            });
        }

        let _ = self.sender.send(WorkerMessage::Enqueue(req));
        id
    }

    pub fn record_deleted_job(&self, path: PathBuf) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("video")
            .to_string();

        if let Ok(mut jobs) = self.active_jobs.lock() {
            jobs.push(JobProgressInfo {
                job_id: id,
                file_name,
                input_path: path,
                output_path: None,
                backup_file_name: None,
                state: JobState::Deleted,
                finished_at: Some(std::time::Instant::now()),
                is_undone: false,
                undone_at: None,
            });
        }
        id
    }

    pub fn queue_history_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("ImageStudio").join("queue_history.json"))
    }

    pub fn load_queue_history() -> Vec<JobProgressInfo> {
        if let Some(path) = Self::queue_history_path() {
            if let Ok(data) = std::fs::read_to_string(path) {
                if let Ok(mut jobs) = serde_json::from_str::<Vec<JobProgressInfo>>(&data) {
                    for j in &mut jobs {
                        if matches!(j.state, JobState::Running { .. } | JobState::Queued) {
                            j.state = JobState::Failed("Interrupted by previous app shutdown".into());
                        }
                    }
                    return jobs;
                }
            }
        }
        Vec::new()
    }

    pub fn save_queue_history(&self) {
        if let Some(path) = Self::queue_history_path() {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(jobs) = self.active_jobs.lock() {
                if let Ok(json) = serde_json::to_string_pretty(&*jobs) {
                    let _ = std::fs::write(path, json);
                }
            }
        }
    }

    pub fn delete_queue_history_file() {
        if let Some(path) = Self::queue_history_path() {
            if path.exists() {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    pub fn load_persisted_jobs(&self, jobs: Vec<JobProgressInfo>) {
        let max_id = jobs.iter().map(|j| j.job_id).max().unwrap_or(0);
        self.next_id.store(max_id + 1, Ordering::SeqCst);
        if let Ok(mut lock) = self.active_jobs.lock() {
            *lock = jobs;
        }
    }

    pub fn remove_job(&self, job_id: u64) {
        if let Ok(mut jobs) = self.active_jobs.lock() {
            jobs.retain(|j| j.job_id != job_id);
        }
    }

    pub fn update(&self) -> bool {
        let mut changed = false;
        while let Ok(evt) = self.events_rx.try_recv() {
            changed = true;
            if let Ok(mut jobs) = self.active_jobs.lock() {
                match &evt {
                    WorkerEvent::JobStarted(id, _) => {
                        if let Some(j) = jobs.iter_mut().find(|j| j.job_id == *id) {
                            j.state = JobState::Running {
                                progress: 0.0,
                                speed: "starting".into(),
                            };
                        }
                    }
                    WorkerEvent::JobProgress(id, prog, speed) => {
                        if let Some(j) = jobs.iter_mut().find(|j| j.job_id == *id) {
                            j.state = JobState::Running {
                                progress: *prog,
                                speed: speed.clone(),
                            };
                        }
                    }
                    WorkerEvent::JobFinished(id, out_path, backup_name) => {
                        if let Some(j) = jobs.iter_mut().find(|j| j.job_id == *id) {
                            j.state = JobState::Finished;
                            j.output_path = Some(out_path.clone());
                            j.finished_at = Some(std::time::Instant::now());
                            j.backup_file_name = backup_name.clone();
                        }
                    }
                    WorkerEvent::JobFailed(id, err) => {
                        if let Some(j) = jobs.iter_mut().find(|j| j.job_id == *id) {
                            j.state = JobState::Failed(err.clone());
                        }
                    }
                }
            }
        }
        changed
    }
}

fn worker_loop(
    rx_msg: Receiver<WorkerMessage>,
    tx_evt: Sender<WorkerEvent>,
    _active_jobs: Arc<Mutex<Vec<JobProgressInfo>>>,
) {
    while let Ok(msg) = rx_msg.recv() {
        match msg {
            WorkerMessage::Enqueue(req) => {
                let file_name = req
                    .input_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("video")
                    .to_string();

                let _ = tx_evt.send(WorkerEvent::JobStarted(req.id, file_name));
                match run_single_job(&req, &tx_evt) {
                    Ok((out_path, backup_name)) => {
                        let _ = tx_evt.send(WorkerEvent::JobFinished(req.id, out_path, backup_name));
                    }
                    Err(e) => {
                        let _ = tx_evt.send(WorkerEvent::JobFailed(req.id, e));
                    }
                }
            }
            WorkerMessage::Cancel(_) => {}
        }
    }
}

fn run_single_job(req: &RenderJobRequest, tx_evt: &Sender<WorkerEvent>) -> Result<(PathBuf, Option<String>), String> {
    let input = &req.input_path;
    if !input.exists() {
        return Err(format!("Input file does not exist: {}", input.display()));
    }

    let parent_dir = input.parent().unwrap_or_else(|| Path::new("."));
    let file_stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("video");
    let ext = input
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("mp4");

    // Output to a temp file in target folder for fast rename
    let temp_dir = if !req.replace_in_place {
        if let Some(ref out_p) = req.explicit_output_path {
            if let Some(p) = out_p.parent() {
                let _ = std::fs::create_dir_all(p);
                p.to_path_buf()
            } else {
                parent_dir.to_path_buf()
            }
        } else {
            parent_dir.to_path_buf()
        }
    } else {
        parent_dir.to_path_buf()
    };
    let temp_output = temp_dir.join(format!("{}.imagestudio_temp.{}.{}", file_stem, req.id, ext));

    let duration_to_render = match (req.trim_start, req.trim_end) {
        (Some(start), Some(end)) if end > start => end - start,
        (Some(start), None) if req.total_duration > start => req.total_duration - start,
        (None, Some(end)) if end > 0.0 => end,
        _ => req.total_duration.max(1.0),
    };

    let mut args = Vec::new();
    args.push("-y".to_string());

    // Fast seek before input
    if let Some(start) = req.trim_start {
        if start > 0.001 {
            args.push("-ss".to_string());
            args.push(format!("{:.3}", start));
        }
    }

    // Input file
    args.push("-i".to_string());
    args.push(input.to_string_lossy().into_owned());

    // Trim duration
    if let Some(end) = req.trim_end {
        let start = req.trim_start.unwrap_or(0.0);
        if end > start {
            args.push("-t".to_string());
            args.push(format!("{:.3}", end - start));
        }
    }

    let audio_filter = if req.normalise_audio {
        Some(crate::video::build_audio_normalise_filter(
            req.audio_normalise_thresh_db,
            req.audio_normalise_ratio,
            req.audio_normalise_boost_db,
        ))
    } else {
        None
    };

    if req.compress {
        // NVENC / Quality VBR Compression mode
        // 1. Map video stream 0:v:0 and ALL audio streams 0:a? (retains Shadowplay mic + game tracks!)
        args.push("-map".to_string());
        args.push("0:v:0".to_string());
        args.push("-map".to_string());
        args.push("0:a?".to_string());

        let codec = req.preset.codec;
        args.push("-c:v".to_string());
        args.push(codec.ffmpeg_codec_name().to_string());

        if codec.is_nvenc() {
            // High-quality NVIDIA NVENC VBR matching Vegas Pro NVENC profile
            args.push("-rc:v".to_string());
            args.push("vbr".to_string());
            args.push("-b:v".to_string());
            args.push(format!("{:.1}M", req.preset.avg_bitrate_mbps));
            args.push("-maxrate:v".to_string());
            args.push(format!("{:.1}M", req.preset.max_bitrate_mbps));
            args.push("-bufsize:v".to_string());
            args.push(format!("{:.1}M", req.preset.max_bitrate_mbps * 2.0));
            args.push("-preset".to_string());
            args.push("p6".to_string());
            args.push("-tune".to_string());
            args.push("hq".to_string());
        } else {
            // CPU x264/x265 fallback
            args.push("-b:v".to_string());
            args.push(format!("{:.1}M", req.preset.avg_bitrate_mbps));
            args.push("-maxrate".to_string());
            args.push(format!("{:.1}M", req.preset.max_bitrate_mbps));
            args.push("-bufsize".to_string());
            args.push(format!("{:.1}M", req.preset.max_bitrate_mbps * 2.0));
            args.push("-preset".to_string());
            args.push("fast".to_string());
        }

        // Target Resolution scaling if needed
        if let Some(max_h) = req.preset.resolution.max_height() {
            args.push("-vf".to_string());
            args.push(format!("scale=-2:'min({},ih)'", max_h));
        }

        // Audio encoding - applies to ALL mapped audio streams (game audio and microphone!)
        args.push("-c:a".to_string());
        args.push("aac".to_string());
        args.push("-b:a".to_string());
        args.push(format!("{}k", req.preset.audio_bitrate_kbps));

        if let Some(ref af) = audio_filter {
            args.push("-af".to_string());
            args.push(af.clone());
        }
    } else if req.normalise_audio {
        // Stream copy video losslessly (-c:v copy), while all audio streams
        // are individually normalised and encoded to AAC
        args.push("-map".to_string());
        args.push("0:v:0".to_string());
        args.push("-map".to_string());
        args.push("0:a?".to_string());

        args.push("-c:v".to_string());
        args.push("copy".to_string());

        args.push("-c:a".to_string());
        args.push("aac".to_string());
        args.push("-b:a".to_string());
        args.push(format!("{}k", req.preset.audio_bitrate_kbps));

        if let Some(ref af) = audio_filter {
            args.push("-af".to_string());
            args.push(af.clone());
        }

        args.push("-avoid_negative_ts".to_string());
        args.push("make_zero".to_string());
    } else {
        // Lossless Cut Mode (-c copy)
        // Map ALL streams (all video, all audio tracks, subtitles)
        args.push("-map".to_string());
        args.push("0".to_string());
        args.push("-c".to_string());
        args.push("copy".to_string());
        args.push("-avoid_negative_ts".to_string());
        args.push("make_zero".to_string());
    }

    // Output file
    args.push(temp_output.to_string_lossy().into_owned());

    let mut cmd = Command::new("ffmpeg");
    cmd.args(&args);
    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::piped());
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn ffmpeg: {e}"))?;

    // Read stderr for progress updates
    if let Some(stderr) = child.stderr.take() {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            if let Some(pos) = line.find("time=") {
                let rest = &line[pos + 5..];
                if let Some(end_pos) = rest.find(' ') {
                    let time_str = &rest[..end_pos];
                    if let Some(cur_secs) = parse_time_to_seconds(time_str) {
                        let progress = (cur_secs / duration_to_render).clamp(0.0, 0.99) as f32;
                        let speed = if let Some(sp_pos) = line.find("speed=") {
                            let sp_rest = &line[sp_pos + 6..];
                            sp_rest.split_whitespace().next().unwrap_or("").to_string()
                        } else {
                            "".to_string()
                        };
                        let _ = tx_evt.send(WorkerEvent::JobProgress(req.id, progress, speed));
                    }
                }
            }
        }
    }

    let status = child
        .wait()
        .map_err(|e| format!("Failed to wait on ffmpeg: {e}"))?;

    if !status.success() {
        let _ = std::fs::remove_file(&temp_output);
        return Err(format!("FFmpeg exited with error code: {:?}", status.code()));
    }

    if !temp_output.exists() || std::fs::metadata(&temp_output).map(|m| m.len()).unwrap_or(0) == 0 {
        let _ = std::fs::remove_file(&temp_output);
        return Err("Rendered file was empty or missing".into());
    }

    // Default replace in-place or explicit output behavior
    if req.replace_in_place {
        let original_target = input.clone();
        let backup_name = replace_file_with_retry(&temp_output, &original_target, req.id)?;
        Ok((original_target, Some(backup_name)))
    } else if let Some(ref out_p) = req.explicit_output_path {
        if out_p.exists() {
            let _ = std::fs::remove_file(out_p);
        }
        std::fs::rename(&temp_output, out_p)
            .map_err(|e| format!("Failed to move rendered video to {}: {e}", out_p.display()))?;
        Ok((out_p.clone(), None))
    } else {
        Ok((temp_output, None))
    }
}

fn replace_file_with_retry(temp_output: &Path, original_target: &Path, job_id: u64) -> Result<String, String> {
    let parent_dir = original_target.parent().unwrap_or_else(|| Path::new("."));
    let file_stem = original_target
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("video");
    let ext = original_target
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("mp4");

    let backup_name = format!("{}.orig_bak_{}.{}", file_stem, job_id, ext);
    let backup_path = parent_dir.join(&backup_name);
    if backup_path.exists() {
        let _ = std::fs::remove_file(&backup_path);
    }

    // Give Windows kernel a brief moment (50ms) to settle handles after render process exit
    std::thread::sleep(std::time::Duration::from_millis(50));

    let mut last_err = String::new();
    let max_retries = 35; // 35 * 100ms = 3.5 seconds of retries for any lingering file locks

    // Step 1: Move original_target to backup_path
    let mut original_moved = false;
    for attempt in 1..=max_retries {
        match std::fs::rename(original_target, &backup_path) {
            Ok(_) => {
                original_moved = true;
                break;
            }
            Err(e) => {
                last_err = format!("Cannot move original to backup (attempt {attempt}/{max_retries}): {e}");
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }

    if !original_moved {
        // Fallback: Attempt direct overwrite rename
        for _ in 1..=10 {
            if std::fs::rename(temp_output, original_target).is_ok() {
                return Ok(backup_name);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        return Err(format!("Could not access or rename original file (file locked by system/player): {last_err}"));
    }

    // Step 2: Move temp_output to original_target
    let mut temp_moved = false;
    for attempt in 1..=max_retries {
        match std::fs::rename(temp_output, original_target) {
            Ok(_) => {
                temp_moved = true;
                break;
            }
            Err(e) => {
                last_err = format!("Cannot move temp to original (attempt {attempt}/{max_retries}): {e}");
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }

    if !temp_moved {
        // Rollback: try to restore original from backup_path
        for _ in 1..=10 {
            if std::fs::rename(&backup_path, original_target).is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        return Err(format!("Failed to place trimmed video in position: {last_err}"));
    }

    // Step 3: Remove backup_path after successful in-place replacement
    let _ = std::fs::remove_file(&backup_path);

    Ok(backup_name)
}

fn parse_time_to_seconds(s: &str) -> Option<f64> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() == 3 {
        let hours: f64 = parts[0].parse().ok()?;
        let mins: f64 = parts[1].parse().ok()?;
        let secs: f64 = parts[2].parse().ok()?;
        Some(hours * 3600.0 + mins * 60.0 + secs)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_deleted_job() {
        let mgr = RenderManager::new();
        let path = PathBuf::from("C:/Videos/sample_clip.mp4");
        let id = mgr.record_deleted_job(path.clone());

        let jobs = mgr.active_jobs.lock().unwrap();
        let job = jobs.iter().find(|j| j.job_id == id).expect("job should exist");
        assert_eq!(job.file_name, "sample_clip.mp4");
        assert_eq!(job.input_path, path);
        assert_eq!(job.state, JobState::Deleted);
        assert!(!job.is_undone);
        assert!(job.finished_at.is_some());
    }

    #[test]
    fn test_persisted_jobs_management() {
        let mgr = RenderManager::new();
        let sample = vec![
            JobProgressInfo {
                job_id: 10,
                file_name: "test1.mp4".into(),
                input_path: PathBuf::from("C:/test1.mp4"),
                output_path: None,
                backup_file_name: Some("test1.orig_bak_10.mp4".into()),
                state: JobState::Finished,
                finished_at: None,
                is_undone: false,
                undone_at: None,
            },
            JobProgressInfo {
                job_id: 11,
                file_name: "test2.mp4".into(),
                input_path: PathBuf::from("C:/test2.mp4"),
                output_path: None,
                backup_file_name: None,
                state: JobState::Running {
                    progress: 0.5,
                    speed: "1x".into(),
                },
                finished_at: None,
                is_undone: false,
                undone_at: None,
            },
        ];
        mgr.load_persisted_jobs(sample);

        let next_id = mgr.next_id.load(Ordering::SeqCst);
        assert_eq!(next_id, 12);

        mgr.remove_job(10);
        let jobs = mgr.active_jobs.lock().unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].job_id, 11);
    }

    #[test]
    fn test_audio_normalise_filter_generation() {
        let filter = crate::video::build_audio_normalise_filter(-25.0, 6.0, 10.0);
        assert!(filter.contains("threshold=-25.0dB"));
        assert!(filter.contains("ratio=6.0"));
        assert!(filter.contains("makeup=3.162"));
        assert!(filter.contains("knee=2.82843"));
        assert!(filter.contains("alimiter=limit=0.95"));
    }
}
