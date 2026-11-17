//! All UI logic.
//! Guaranteed rules:
//! 1. The UI thread never blocks, never decodes images, and never makes network calls.
//! 2. Heavy work happens in worker threads, sending events via `mpsc::channel`.
//! 3. Textures are created strictly on the UI thread inside `drain()`, outside any input locks.
use crate::comfy;
use crate::engine::{self, EngState, Profile, Shared};
use crate::history::TrimHistory;
use crate::presets::{AutoQualityLevel, QualityPreset, TargetResolution, VideoCodec};
use crate::render::{JobState, RenderJobRequest, RenderManager};
use eframe::egui::{self, pos2, vec2, Color32, ColorImage, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, TextureHandle, Vec2};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

pub fn estimate_video_vram_gb_with_lora(w: u32, h: u32, duration: u32, fps: u32, lora_vram_gb: f32) -> f32 {
    let frames = (duration * fps) as f32;
    let mp = (w * h) as f32 / 1_000_000.0;
    let load_factor = mp * (frames / 24.0);
    9.5 + load_factor * 0.55 + lora_vram_gb.max(0.0)
}

const ABS_MIN_RES: u32 = 256;
const ABS_MAX_RES: u32 = 4096;

/// Returns `(min_scale, max_scale)` relative to `(base_w, base_h)` so the shorter side
/// reaches `min_dim` at the left end of the slider and the longer side reaches `max_dim` at the right end.
fn absolute_scale_range(base_w: u32, base_h: u32, min_dim: u32, max_dim: u32) -> (f32, f32) {
    let w = base_w.max(1) as f32;
    let h = base_h.max(1) as f32;
    let k_min = min_dim as f32 / w.min(h);
    let k_max = max_dim as f32 / w.max(h);
    if k_min <= k_max {
        (k_min, k_max)
    } else {
        (k_max, k_max)
    }
}

/// Scales `(base_w, base_h)` by `scale`, rounding each axis to the nearest `step` and clamping to `min_dim..=max_dim`.
fn scale_dimensions_stepped(
    base_w: u32,
    base_h: u32,
    scale: f32,
    step: u32,
    min_dim: u32,
    max_dim: u32,
) -> (u32, u32) {
    let step = step.max(1);
    let step_f = step as f32;
    let w = (((base_w.max(1) as f32 * scale) / step_f).round() as u32 * step).clamp(min_dim, max_dim);
    let h = (((base_h.max(1) as f32 * scale) / step_f).round() as u32 * step).clamp(min_dim, max_dim);
    (w, h)
}

pub fn build_timestamp_display() -> String {
    if let Ok(epoch_secs) = option_env!("BUILD_EPOCH").unwrap_or("").parse::<i64>() {
        if let Some(dt) = chrono::DateTime::from_timestamp(epoch_secs, 0) {
            return dt.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M:%S").to_string();
        }
    }
    env!("BUILD_STAMP").to_string()
}

const BG: Color32 = Color32::from_rgb(12, 14, 19);
const PANEL: Color32 = Color32::from_rgb(18, 21, 29);
const CARD: Color32 = Color32::from_rgb(25, 29, 40);
const CARD_HI: Color32 = Color32::from_rgb(33, 38, 53);
const LINE: Color32 = Color32::from_rgb(52, 60, 82);
const ACCENT: Color32 = Color32::from_rgb(99, 102, 241);
const ACCENT_GLOW: Color32 = Color32::from_rgb(129, 140, 248);
const TEXT: Color32 = Color32::from_rgb(235, 238, 248);
const DIM: Color32 = Color32::from_rgb(168, 177, 202);
const GREEN: Color32 = Color32::from_rgb(52, 211, 153);
const BLUE: Color32 = Color32::from_rgb(59, 130, 246);
const AMBER: Color32 = Color32::from_rgb(251, 191, 36);
const RED: Color32 = Color32::from_rgb(248, 113, 113);
const DISCORD_GREEN: Color32 = Color32::from_rgb(35, 165, 90);
const DISCORD_GREEN_HOVER: Color32 = Color32::from_rgb(43, 190, 104);
const MAX_REFS: usize = 15;
const FULL_UV: Rect = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub enum StudioMode {
    Image,
    Video,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum VideoWorkMode {
    Edit,
    Generate,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub enum VideoOrientation {
    Landscape,
    Portrait,
}

#[derive(Clone, Copy, PartialEq)]
enum Target {
    Base,
    Ref,
}

enum Src {
    Path(PathBuf),
    Rgba(usize, usize, Vec<u8>),
}

enum Ev {
    Loaded {
        target: Target,
        name: String,
        png: Vec<u8>,
        img: ColorImage,
        width: u32,
        height: u32,
    },
    LoadErr(String),
    Status(String),
    Progress { value: u32, max: u32 },
    TotalProgress(f32),
    Preview(ColorImage),
    Done {
        png: Vec<u8>,
        img: ColorImage,
        output_filename: String,
        output_subfolder: String,
        temp_files: Vec<String>,
    },
    VideoDone {
        mp4_bytes: Vec<u8>,
        output_filename: String,
        output_subfolder: String,
        temp_files: Vec<String>,
    },
    GenErr(String),
}


struct Item {
    name: String,
    png: Vec<u8>,
    tex: TextureHandle,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug)]
pub struct QueuedItem {
    pub prompt: String,
}

#[derive(Clone)]
pub struct HistoryNode {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub label: String,
    pub prompt: String,
    pub png: Vec<u8>,
    pub tex: TextureHandle,
    pub width: u32,
    pub height: u32,
    pub step_num: usize,
    pub output_filename: String,
    pub output_subfolder: String,
    pub temp_files: Vec<String>,
    pub mp4_bytes: Option<Vec<u8>>,
    pub source_video_path: Option<PathBuf>,
    pub video_last_sec: Option<f64>,
    pub video_trim_start: Option<f64>,
    pub video_trim_end: Option<f64>,
    pub video_work_mode: Option<VideoWorkMode>,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum HistoryDock {
    Left,
    Right,
    Floating,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq)]
pub enum SizePreset {
    Original,
    P720,
    P1024,
    Custom,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct UserPreset {
    pub name: String,
    pub steps: u32,
    pub size_preset: SizePreset,
    pub target_w: u32,
    pub target_h: u32,
    pub resolution: u32,
    pub cfg: f32,
}

impl UserPreset {
    pub fn defaults() -> Vec<UserPreset> {
        vec![
            UserPreset {
                name: "Quality".to_string(),
                steps: 21,
                size_preset: SizePreset::Original,
                target_w: 1024,
                target_h: 1024,
                resolution: 1024,
                cfg: 1.0,
            },
            UserPreset {
                name: "Balanced".to_string(),
                steps: 10,
                size_preset: SizePreset::Original,
                target_w: 1024,
                target_h: 1024,
                resolution: 1024,
                cfg: 1.0,
            },
            UserPreset {
                name: "Fast".to_string(),
                steps: 10,
                size_preset: SizePreset::P720,
                target_w: 720,
                target_h: 720,
                resolution: 1024,
                cfg: 1.0,
            },
        ]
    }
}

fn presets_path() -> PathBuf {
    engine::data_dir().join("presets.json")
}

fn load_presets() -> Vec<UserPreset> {
    if let Ok(data) = std::fs::read_to_string(presets_path()) {
        if let Ok(mut list) = serde_json::from_str::<Vec<UserPreset>>(&data) {
            if !list.is_empty() {
                // Ensure Balanced exists
                if !list.iter().any(|p| p.name.eq_ignore_ascii_case("Balanced")) {
                    let quality_pos = list.iter().position(|p| p.name.eq_ignore_ascii_case("Quality"));
                    let insert_pos = quality_pos.map(|i| i + 1).unwrap_or(0);
                    let balanced = UserPreset {
                        name: "Balanced".to_string(),
                        steps: 10,
                        size_preset: SizePreset::Original,
                        target_w: 1024,
                        target_h: 1024,
                        resolution: 1024,
                        cfg: 1.0,
                    };
                    if insert_pos <= list.len() {
                        list.insert(insert_pos, balanced);
                    } else {
                        list.push(balanced);
                    }
                    save_presets(&list);
                }
                return list;
            }
        }
    }
    let defs = UserPreset::defaults();
    save_presets(&defs);
    defs
}

fn save_presets(presets: &[UserPreset]) {
    if let Ok(data) = serde_json::to_string_pretty(presets) {
        let _ = std::fs::write(presets_path(), data);
    }
}

const RESOLUTIONS: [(u32, &str); 4] = [
    (0, "Native"),
    (768, "768px"),
    (1024, "1024px"),
    (1536, "1536px"),
];

const LEFT_HISTORY_DEFAULT_H: f32 = 300.0;
const LEFT_HISTORY_MIN_H: f32 = 220.0;
const LEFT_LORA_MIN_H: f32 = 120.0;
const LEFT_SPLIT_BAR_H: f32 = 10.0;

struct PendingHistoryDelete {
    node_id: usize,
    label: String,
    file_desc: String,
    error: Option<String>,
}

pub struct App {
    sh: Arc<Shared>,
    profile: Profile,
    tx: mpsc::Sender<Ev>,
    rx: mpsc::Receiver<Ev>,
    nodes: Vec<HistoryNode>,
    active_node_id: Option<usize>,
    next_node_id: usize,
    generating_parent_id: Option<usize>,
    generating_prompt: String,
    history_dock: HistoryDock,
    show_history_window: bool,
    left_history_h: f32,
    refs: Vec<Item>,
    prompt: String,
    focus_prompt: bool,
    progress: Option<(u32, u32)>,
    total_gen_progress: Option<f32>,
    taskbar_progress: crate::taskbar::TaskbarProgress,
    preview_tex: Option<TextureHandle>,
    negative: String,
    steps: u32,
    cfg: f32,
    seed: u64,
    random_seed: bool,
    presets: Vec<UserPreset>,
    active_preset_idx: Option<usize>,
    size_preset: SizePreset,
    custom_w: u32,
    custom_h: u32,
    ref_w: u32,
    ref_h: u32,
    scale: f32,
    resolution: u32,
    generating: bool,
    generating_dims: Option<(u32, u32)>,
    cancel: Arc<AtomicBool>,
    status: String,
    error: Option<String>,
    compare_mode: bool,
    split: f32,
    canvas_zoom: f32,
    canvas_pan: Vec2,
    video_pan_drag_active: bool,
    last_effective_video_profile_id: String,
    last_ltx_video_profile_id: Option<String>,
    show_setup: bool,
    pub model_picker: crate::model_picker::ModelPickerState,
    loading: usize,
    queue: Vec<QueuedItem>,
    last_ctrl_v_down: bool,
    pub updater: Arc<std::sync::Mutex<crate::updater::UpdaterState>>,
    pub mode: StudioMode,
    pub video_work_mode: VideoWorkMode,
    edit_lora_needs_video_until: Option<Instant>,
    pub video_orientation: VideoOrientation,
    pub video_scale: f32,
    pub video_base_w: u32,
    pub video_base_h: u32,
    pub video_player: Option<crate::video::VideoPlayer>,
    pub video_duration: u32,
    pub video_fps: u32,
    pub video_width: u32,
    pub video_height: u32,
    pub video_prompt_enhance: bool,
    pub video_negative: String,
    pub video_steps: u32,
    pub video_sampler: String,
    pub video_scheduler: String,
    pub video_loras: Vec<crate::lora::ActiveLora>,
    pub video_prompt_tags: Vec<crate::lora::PromptTag>,
    pub image_loras: Vec<crate::lora::ActiveLora>,
    pub image_prompt_tags: Vec<crate::lora::PromptTag>,
    pub lora_browser: crate::lora::LoraBrowserState,
    pub video_r2v_mode: bool,
    pub show_video_dl_modal: bool,
    pending_history_delete: Option<PendingHistoryDelete>,
    pub download_pane: crate::download_ui::DownloadPaneState,
    pub refs_drop_rect: Option<Rect>,
    pub canvas_drop_rect: Option<Rect>,
    pub last_progress_time: Option<Instant>,
    pub last_progress_val: u32,
    pub step_duration_ema: f32,
    pub is_swapping_detected: bool,
    pub generation_retries: u32,
    pub stall_prompt: Arc<AtomicBool>,
    pub stall_keep_waiting: Arc<AtomicBool>,
    /// Cached `(mp4_bytes, snapped_frames)` for IC-LoRA edit retries if `video_player` was cleared.
    pub last_edit_video_source: Option<(Vec<u8>, u32)>,
    // Video player trim, compression & history state (ported from ztrim)
    pub trim_history: TrimHistory,
    pub render_mgr: RenderManager,
    pub video_presets: Vec<QualityPreset>,
    pub selected_video_preset_idx: usize,
    pub auto_quality_level: AutoQualityLevel,
    pub compress_enabled: bool,
    pub avg_bitrate: f32,
    pub max_bitrate: f32,
    pub selected_codec: VideoCodec,
    pub selected_res: TargetResolution,
    pub normalise_audio: bool,
    pub audio_normalise_ratio: f32,
    pub audio_normalise_thresh_db: f32,
    pub audio_normalise_boost_db: f32,
    pub new_video_preset_modal_open: bool,
    pub new_video_preset_name: String,
    pub active_timeline_drag: TimelineDrag,
    pub timeline_drag_occurred: bool,
    pub was_playing_before_scrub: bool,
    pub trim_drag_initial: Option<TrimStateSnapshot>,
    pub trim_undo_stack: Vec<TrimStateSnapshot>,
    pub trim_redo_stack: Vec<TrimStateSnapshot>,
    pub end_trim_error_flash_start: Option<Instant>,
    pub has_seen_trim_hint: bool,
    pub persist_trim_history: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimelineDrag {
    None,
    Playhead,
    TrimStart,
    TrimEnd,
}

pub const END_TRIM_PAD_PX: f32 = 25.0;
pub const START_TRIM_PAD_PX: f32 = 26.0;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TrimClickDecision {
    SetStartTrim(f64),
    SetEndTrim(f64),
    BlockRightClickFlashRed,
}

pub fn evaluate_trim_click(
    is_right_click: bool,
    pos_x: f32,
    trim_x_min: f32,
    trim_x_max: f32,
    target_sec: f64,
    cur_start: f64,
    cur_end: f64,
    dur: f64,
) -> TrimClickDecision {
    if is_right_click {
        let is_after_end_marker = pos_x >= trim_x_max;
        let min_trim_gap_sec = 0.05;
        let is_inside_or_before_start = !is_after_end_marker
            && (pos_x <= trim_x_min + START_TRIM_PAD_PX || target_sec <= cur_start + min_trim_gap_sec);

        if is_inside_or_before_start {
            TrimClickDecision::BlockRightClickFlashRed
        } else {
            let new_e = target_sec.clamp(cur_start + min_trim_gap_sec, dur);
            TrimClickDecision::SetEndTrim(new_e)
        }
    } else {
        let d_start = (pos_x - trim_x_min).abs();
        let d_end = (pos_x - trim_x_max).abs();

        let is_after_end = pos_x >= trim_x_max;
        let is_in_end_padding = pos_x >= trim_x_max - END_TRIM_PAD_PX;
        let start_overrides = d_start <= START_TRIM_PAD_PX && d_start < d_end;
        let controls_end_trim = is_after_end || (is_in_end_padding && !start_overrides);

        if controls_end_trim {
            let new_e = target_sec.clamp(cur_start + 0.05, dur);
            TrimClickDecision::SetEndTrim(new_e)
        } else {
            let new_s = target_sec.clamp(0.0, (cur_end - 0.05).max(0.0));
            TrimClickDecision::SetStartTrim(new_s)
        }
    }
}

pub fn is_end_trim_flashing_red_at(flash_start: Option<Instant>, now: Instant) -> (bool, bool) {
    if let Some(start) = flash_start {
        let elapsed = now.saturating_duration_since(start).as_millis();
        if elapsed < 400 {
            (true, true)
        } else if elapsed < 600 {
            (false, true)
        } else if elapsed < 1000 {
            (true, true)
        } else {
            (false, false)
        }
    } else {
        (false, false)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrimStateSnapshot {
    pub start: Option<f64>,
    pub end: Option<f64>,
}

fn rand_u64() -> u64 {
    let mut x = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1) | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn draw_generating_overlay(
    p: &egui::Painter,
    image_rect: Rect,
    progress: Option<(u32, u32)>,
    is_swapping_detected: bool,
    step_duration_ema: f32,
) {
    p.rect_stroke(image_rect, 0, Stroke::new(1.0, ACCENT_GLOW), StrokeKind::Outside);

    let badge_text = if is_swapping_detected {
        if let Some((v, max)) = progress {
            format!("[!] VRAM SWAPPING SLOWDOWN ({}/{} • {:.1}s/step)", v, max, step_duration_ema)
        } else {
            "[!] VRAM SWAPPING SLOWDOWN".to_string()
        }
    } else if let Some((v, max)) = progress {
        format!("GENERATING (Step {}/{})", v, max)
    } else {
        "GENERATING...".to_string()
    };
    let g = p.layout_no_wrap(badge_text, egui::FontId::proportional(11.0), Color32::WHITE);
    let badge_pos = image_rect.left_top() + vec2(10.0, 10.0);
    let r = egui::Align2::LEFT_TOP.anchor_size(badge_pos, g.size()).expand2(vec2(8.0, 4.0));
    p.rect_filled(r, 4, Color32::from_black_alpha(190));
    p.rect_stroke(r, 4, Stroke::new(1.0, if is_swapping_detected { AMBER } else { ACCENT }), StrokeKind::Inside);
    p.galley(r.shrink2(vec2(8.0, 4.0)).min, g, Color32::WHITE);
}

fn fit(src: Vec2, max: Vec2) -> Vec2 {
    if src.x <= 0.0 || src.y <= 0.0 || max.x <= 0.0 || max.y <= 0.0 {
        return vec2(1.0, 1.0);
    }
    let s = (max.x / src.x).min(max.y / src.y).min(1.0);
    vec2((src.x * s).max(1.0), (src.y * s).max(1.0))
}

fn card<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0, LINE))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, add)
        .inner
}

fn heading(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).size(12.0).strong().color(DIM));
    ui.add_space(2.0);
}

fn draw_update_arrow_icon(p: &egui::Painter, center: egui::Pos2, color: Color32) {
    let stroke = Stroke::new(1.8, color);
    // Vertical stem:
    p.line_segment([center - vec2(0.0, 5.0), center + vec2(0.0, 2.0)], stroke);
    // Chevron pointing down:
    p.line_segment([center + vec2(-4.0, -1.5), center + vec2(0.0, 2.5)], stroke);
    p.line_segment([center + vec2(4.0, -1.5), center + vec2(0.0, 2.5)], stroke);
    // Horizontal tray bar beneath arrow:
    p.line_segment([center + vec2(-5.0, 6.0), center + vec2(5.0, 6.0)], Stroke::new(1.5, color));
}

fn draw_restart_icon(p: &egui::Painter, center: egui::Pos2, color: Color32) {
    let stroke = Stroke::new(1.6, color);
    let r = 5.0;
    let n_pts = 12;
    let start_angle = 0.8f32;
    let end_angle = 5.6f32;
    let pts: Vec<egui::Pos2> = (0..=n_pts)
        .map(|i| {
            let t = i as f32 / n_pts as f32;
            let a = start_angle + t * (end_angle - start_angle);
            center + vec2(r * a.cos(), r * a.sin())
        })
        .collect();
    for w in pts.windows(2) {
        p.line_segment([w[0], w[1]], stroke);
    }
    if let Some(&tip) = pts.last() {
        p.line_segment([tip, tip + vec2(2.5, -2.5)], stroke);
        p.line_segment([tip, tip + vec2(-2.5, -1.0)], stroke);
    }
}

pub fn draw_scissors_icon(p: &egui::Painter, center: Pos2, color: Color32) {
    let stroke = Stroke::new(1.6, color);
    p.circle_stroke(center + vec2(-5.0, -3.0), 2.5, stroke);
    p.circle_stroke(center + vec2(-5.0, 3.0), 2.5, stroke);
    p.line_segment([center + vec2(-2.5, -2.5), center + vec2(6.5, 3.5)], stroke);
    p.line_segment([center + vec2(-2.5, 2.5), center + vec2(6.5, -3.5)], stroke);
    p.circle_filled(center + vec2(0.5, 0.0), 1.2, color);
}

pub fn draw_play_icon(p: &egui::Painter, center: Pos2, color: Color32) {
    let c = center + vec2(0.75, 0.0);
    let p1 = c + vec2(-4.5, -5.5);
    let p2 = c + vec2(-4.5, 5.5);
    let p3 = c + vec2(5.5, 0.0);
    p.add(egui::Shape::convex_polygon(vec![p1, p2, p3], color, Stroke::NONE));
}

pub fn draw_pause_icon(p: &egui::Painter, center: Pos2, color: Color32) {
    let h = 5.5;
    let w = 2.4;
    let gap = 2.0;
    let rect_left = Rect::from_min_max(center + vec2(-gap - w, -h), center + vec2(-gap, h));
    let rect_right = Rect::from_min_max(center + vec2(gap, -h), center + vec2(gap + w, h));
    p.rect_filled(rect_left, 1.0, color);
    p.rect_filled(rect_right, 1.0, color);
}

pub fn draw_trash_icon(p: &egui::Painter, center: Pos2, color: Color32) {
    let stroke = Stroke::new(1.25, color);
    // Lid + handle
    p.line_segment([center + vec2(-4.5, -3.5), center + vec2(4.5, -3.5)], stroke);
    p.line_segment([center + vec2(-2.0, -5.2), center + vec2(2.0, -5.2)], stroke);
    p.line_segment([center + vec2(-2.0, -5.2), center + vec2(-2.0, -3.5)], stroke);
    p.line_segment([center + vec2(2.0, -5.2), center + vec2(2.0, -3.5)], stroke);
    // Bin body
    let body = Rect::from_min_max(center + vec2(-3.5, -2.5), center + vec2(3.5, 5.0));
    p.rect_stroke(body, 1.2, stroke, StrokeKind::Inside);
    // Vertical ribs
    p.line_segment([center + vec2(-1.2, -1.0), center + vec2(-1.2, 3.5)], Stroke::new(1.0, color));
    p.line_segment([center + vec2(1.2, -1.0), center + vec2(1.2, 3.5)], Stroke::new(1.0, color));
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ctx = cc.egui_ctx.clone();

        // Enforce dark mode visuals unconditionally to prevent black-on-black text on systems running light theme
        let mut visuals = egui::Visuals::dark();
        visuals.dark_mode = true;
        visuals.override_text_color = Some(TEXT);
        visuals.panel_fill = PANEL;
        visuals.window_fill = PANEL;
        visuals.window_stroke = Stroke::new(1.0, LINE);
        visuals.widgets.noninteractive.bg_fill = CARD;
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, LINE);
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        visuals.widgets.inactive.bg_fill = CARD_HI;
        visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, LINE);
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
        visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(46, 52, 72);
        visuals.widgets.hovered.bg_fill = Color32::from_rgb(46, 52, 72);
        visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
        visuals.widgets.active.weak_bg_fill = ACCENT;
        visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
        ctx.set_visuals(visuals);
        ctx.set_theme(egui::Theme::Dark);

        let apply_style = |s: &mut egui::Style| {
            s.visuals.dark_mode = true;
            s.visuals.override_text_color = Some(TEXT);
            s.spacing.item_spacing = vec2(6.0, 6.0);
            s.spacing.button_padding = vec2(8.0, 5.0);
            for (ts, sz) in [
                (egui::TextStyle::Body, 13.5),
                (egui::TextStyle::Button, 13.0),
                (egui::TextStyle::Small, 12.0),
                (egui::TextStyle::Heading, 18.0),
                (egui::TextStyle::Monospace, 12.5),
            ] {
                if let Some(f) = s.text_styles.get_mut(&ts) {
                    f.size = sz;
                }
            }
        };
        ctx.style_mut_of(egui::Theme::Dark, apply_style);
        ctx.style_mut_of(egui::Theme::Light, apply_style);
        let sh = Shared::new();
        let profile = Profile::load();
        engine::start_supervisor(sh.clone(), profile.clone(), ctx.clone());
        let (tx, rx) = mpsc::channel();
        let updater = Arc::new(std::sync::Mutex::new(crate::updater::UpdaterState::default()));
        crate::updater::check_for_updates(updater.clone(), ctx.clone());
        let presets = load_presets();
        let default_idx = presets
            .iter()
            .position(|p| p.name.eq_ignore_ascii_case("Quality"))
            .unwrap_or(0);
        let initial_steps = presets.get(default_idx).map(|p| p.steps).unwrap_or(21);
        let initial_size = presets.get(default_idx).map(|p| p.size_preset).unwrap_or(SizePreset::Original);
        let initial_w = presets.get(default_idx).map(|p| p.target_w).unwrap_or(1024);
        let initial_h = presets.get(default_idx).map(|p| p.target_h).unwrap_or(1024);
        let initial_cfg = presets.get(default_idx).map(|p| p.cfg).unwrap_or(profile.default_cfg);

        let video_presets = QualityPreset::load_presets();
        let trim_history = TrimHistory::load();
        let render_mgr = RenderManager::new();
        let persisted_jobs = RenderManager::load_queue_history();
        if !persisted_jobs.is_empty() {
            render_mgr.load_persisted_jobs(persisted_jobs);
        }
        let default_preset = video_presets
            .first()
            .cloned()
            .unwrap_or_else(|| QualityPreset::default_presets()[0].clone());

        let initial_vid_prof = sh.active_video_profile.lock().unwrap().clone();
        let initial_vsteps = initial_vid_prof.default_steps.max(4);
        let initial_vsampler = if initial_vid_prof.sampler.is_empty() {
            "er_sde".to_string()
        } else {
            initial_vid_prof.sampler.clone()
        };
        let initial_vscheduler = if initial_vid_prof.scheduler.is_empty() {
            "sgm_uniform".to_string()
        } else {
            initial_vid_prof.scheduler.clone()
        };
        let mut lora_browser = crate::lora::LoraBrowserState::default();
        lora_browser.items = crate::lora::scan_local_loras(&sh);

        let (initial_mode, keep_last_project) = {
            let cfg = sh.cfg.lock().unwrap();
            let m = if cfg.remember_last_mode && cfg.last_studio_mode.as_deref() == Some("video") {
                StudioMode::Video
            } else {
                StudioMode::Image
            };
            (m, cfg.keep_last_project_loaded)
        };

        let mut app = App {
            sh,
            tx,
            rx,
            nodes: vec![],
            active_node_id: None,
            next_node_id: 1,
            generating_parent_id: None,
            generating_prompt: String::new(),
            history_dock: HistoryDock::Left,
            show_history_window: true,
            left_history_h: LEFT_HISTORY_DEFAULT_H,
            refs: vec![],
            prompt: String::new(),
            focus_prompt: false,
            progress: None,
            total_gen_progress: None,
            taskbar_progress: Default::default(),
            preview_tex: None,
            negative: String::new(),
            steps: initial_steps,
            cfg: initial_cfg,
            profile,
            seed: rand_u64(),
            random_seed: true,
            presets,
            active_preset_idx: Some(default_idx),
            size_preset: initial_size,
            custom_w: initial_w,
            custom_h: initial_h,
            ref_w: initial_w,
            ref_h: initial_h,
            scale: 1.0,
            resolution: 1024,
            generating: false,
            generating_dims: None,
            cancel: Arc::new(AtomicBool::new(false)),
            status: String::new(),
            error: None,
            compare_mode: false,
            split: 0.5,
            canvas_zoom: 1.0,
            canvas_pan: vec2(0.0, 0.0),
            video_pan_drag_active: false,
            last_effective_video_profile_id: initial_vid_prof.id.clone(),
            last_ltx_video_profile_id: if !initial_vid_prof.is_minimax_h3() {
                Some(initial_vid_prof.id.clone())
            } else {
                None
            },
            show_setup: false,
            model_picker: crate::model_picker::ModelPickerState::new(&initial_vid_prof),
            loading: 0,
            queue: Vec::new(),
            last_ctrl_v_down: false,
            updater,
            mode: initial_mode,
            video_work_mode: VideoWorkMode::Edit,
            edit_lora_needs_video_until: None,
            video_orientation: VideoOrientation::Landscape,
            video_scale: 1.0,
            video_base_w: 1280,
            video_base_h: 720,
            video_player: None,
            video_duration: 5,
            video_fps: 24,
            video_width: 1280,
            video_height: 720,
            video_prompt_enhance: true,
            video_negative: String::new(),
            video_steps: initial_vsteps,
            video_sampler: initial_vsampler,
            video_scheduler: initial_vscheduler,
            video_loras: Vec::new(),
            video_prompt_tags: Vec::new(),
            image_loras: Vec::new(),
            image_prompt_tags: Vec::new(),
            lora_browser,
            video_r2v_mode: false,
            show_video_dl_modal: false,
            pending_history_delete: None,
            download_pane: crate::download_ui::DownloadPaneState::default(),
            refs_drop_rect: None,
            canvas_drop_rect: None,
            last_progress_time: None,
            last_progress_val: 0,
            step_duration_ema: 0.0,
            is_swapping_detected: false,
            generation_retries: 0,
            stall_prompt: Arc::new(AtomicBool::new(false)),
            stall_keep_waiting: Arc::new(AtomicBool::new(false)),
            last_edit_video_source: None,
            trim_history,
            render_mgr,
            video_presets,
            selected_video_preset_idx: 0,
            auto_quality_level: AutoQualityLevel::Med,
            compress_enabled: false,
            avg_bitrate: default_preset.avg_bitrate_mbps,
            max_bitrate: default_preset.max_bitrate_mbps,
            selected_codec: default_preset.codec,
            selected_res: default_preset.resolution,
            normalise_audio: false,
            audio_normalise_ratio: 6.0,
            audio_normalise_thresh_db: -25.0,
            audio_normalise_boost_db: 10.0,
            new_video_preset_modal_open: false,
            new_video_preset_name: String::new(),
            active_timeline_drag: TimelineDrag::None,
            timeline_drag_occurred: false,
            was_playing_before_scrub: false,
            trim_drag_initial: None,
            trim_undo_stack: Vec::new(),
            trim_redo_stack: Vec::new(),
            end_trim_error_flash_start: None,
            has_seen_trim_hint: false,
            persist_trim_history: true,
        };
        if keep_last_project {
            app.restore_last_project(&ctx);
        }
        app
    }

    fn remove_history_node(&mut self, node_id: usize) {
        if let Some(pos) = self.nodes.iter().position(|n| n.id == node_id) {
            let node = self.nodes.remove(pos);
            if self.active_node_id == Some(node_id) {
                let next_id = node
                    .parent_id
                    .filter(|pid| self.nodes.iter().any(|n| n.id == *pid))
                    .or_else(|| self.nodes.last().map(|n| n.id));
                if let Some(nid) = next_id {
                    self.select_node(nid);
                } else {
                    self.active_node_id = None;
                    self.video_player = None;
                    self.compare_mode = false;
                }
            }
        }
    }

    fn move_to_recycle_bin(path: &std::path::Path) -> Result<(), String> {
        if !path.exists() {
            return Ok(());
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            use windows_sys::Win32::UI::Shell::{
                FO_DELETE, FOF_ALLOWUNDO, FOF_NOCONFIRMATION, FOF_NOERRORUI, FOF_SILENT,
                SHFILEOPSTRUCTW, SHFileOperationW,
            };
            let abs = if path.is_absolute() {
                path.to_path_buf()
            } else {
                std::env::current_dir()
                    .map(|d| d.join(path))
                    .unwrap_or_else(|_| path.to_path_buf())
            };
            let mut wide: Vec<u16> = abs.as_os_str().encode_wide().collect();
            wide.push(0);
            wide.push(0);
            let mut op = SHFILEOPSTRUCTW {
                hwnd: std::ptr::null_mut(),
                wFunc: FO_DELETE,
                pFrom: wide.as_ptr(),
                pTo: std::ptr::null(),
                fFlags: (FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT) as u16,
                fAnyOperationsAborted: 0,
                hNameMappings: std::ptr::null_mut(),
                lpszProgressTitle: std::ptr::null(),
            };
            let res = unsafe { SHFileOperationW(&mut op) };
            if res == 0 && op.fAnyOperationsAborted == 0 {
                return Ok(());
            }
            return Err(format!("Failed to move {} to Recycle Bin (code {res})", abs.display()));
        }
        #[cfg(not(windows))]
        {
            std::fs::remove_file(path).map_err(|e| format!("Failed to remove {}: {e}", path.display()))
        }
    }

    fn cleanup_unused_temp_videos(keep_path: Option<&std::path::Path>) {
        let tmp_dir = std::env::temp_dir();
        let pid_prefix = format!("image_studio_vid_{}_", std::process::id());
        if let Ok(entries) = std::fs::read_dir(&tmp_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if keep_path.is_some_and(|k| k == path.as_path()) {
                    continue;
                }
                let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
                    continue;
                };
                if fname.starts_with(&pid_prefix) || fname.starts_with("imgstudio_trim_") {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }

    fn request_delete_history_node(&mut self, node_id: usize) {
        let Some(node) = self.find_node(node_id) else {
            return;
        };
        let file_desc = if let Some(ref p) = node.source_video_path {
            p.display().to_string()
        } else if !node.output_filename.is_empty() {
            node.output_filename.clone()
        } else {
            node.label.clone()
        };
        self.pending_history_delete = Some(PendingHistoryDelete {
            node_id,
            label: node.label.clone(),
            file_desc,
            error: None,
        });
    }

    fn history_delete_confirm_modal(&mut self, ctx: &egui::Context) {
        let Some(ref mut del) = self.pending_history_delete else {
            return;
        };
        let mut do_delete = false;
        let mut cancel = false;

        egui::Window::new("Move File to Recycle Bin?")
            .id(egui::Id::new("history_delete_confirm_modal"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .frame(
                egui::Frame::NONE
                    .fill(Color32::from_rgb(24, 28, 39))
                    .stroke(Stroke::new(1.0, Color32::from_rgb(44, 50, 68)))
                    .corner_radius(8.0)
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ctx, |ui| {
                ui.set_width(370.0);
                ui.label(RichText::new(&del.label).size(13.0).strong().color(TEXT));
                ui.add_space(4.0);
                ui.label(
                    RichText::new(&del.file_desc)
                        .size(11.0)
                        .color(DIM),
                );
                ui.add_space(6.0);
                ui.label(
                    RichText::new("This will remove the item from history, send its file(s) on disk to the Recycle Bin, and clear cached copies in Temp and last_project_assets.")
                        .size(11.5)
                        .color(Color32::from_rgb(175, 182, 198)),
                );
                if let Some(ref err) = del.error {
                    ui.add_space(6.0);
                    ui.label(RichText::new(err).size(11.0).color(RED));
                }
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let btn = egui::Button::new(
                            RichText::new("Move to Recycle Bin").strong().color(Color32::WHITE),
                        )
                        .fill(Color32::from_rgb(175, 42, 42));
                        if ui.add(btn).clicked() {
                            do_delete = true;
                        }
                    });
                });
            });

        if do_delete {
            let id = del.node_id;
            match self.delete_history_node(id) {
                Ok(()) => {
                    self.pending_history_delete = None;
                }
                Err(e) => {
                    if let Some(ref mut d) = self.pending_history_delete {
                        d.error = Some(e);
                    }
                }
            }
        } else if cancel {
            self.pending_history_delete = None;
        }
    }

    fn delete_history_node(&mut self, node_id: usize) -> Result<(), String> {
        let Some(pos) = self.nodes.iter().position(|n| n.id == node_id) else {
            return Ok(());
        };

        let was_active = self.active_node_id == Some(node_id);
        let node = &self.nodes[pos];

        // Drop active VideoPlayer before deleting so file handles & ffmpeg workers are released.
        let player_matches_node = self.video_player.as_ref().is_some_and(|vp| {
            was_active
                || node.source_video_path.as_ref() == Some(&vp.file_path)
                || (!node.output_filename.is_empty() && vp.filename == node.output_filename)
        });
        let dropped_temp_path = if player_matches_node {
            let p = self.video_player.as_ref().map(|vp| vp.file_path.clone());
            self.video_player = None;
            p
        } else {
            None
        };

        let node = self.nodes.remove(pos);
        let tmp_dir = std::env::temp_dir();
        let assets_dir = Self::last_project_assets_dir();

        // Recycle or remove the user/source video path if present.
        if let Some(ref src_path) = node.source_video_path {
            if src_path.is_file() {
                if src_path.starts_with(&tmp_dir) || src_path.starts_with(&assets_dir) {
                    let _ = std::fs::remove_file(src_path);
                } else {
                    Self::move_to_recycle_bin(src_path)?;
                    engine::log(&format!("Moved source file to Recycle Bin: {}", src_path.display()));
                }
            }
        }

        let comfy_dir = self.sh.paths.lock().unwrap().comfy.clone();
        let mut recycle_dirs = Vec::new();
        if let Some(ref cp) = comfy_dir {
            recycle_dirs.push(cp.clone());
            if let Some(parent) = cp.parent() {
                recycle_dirs.push(parent.to_path_buf());
            }
        }
        recycle_dirs.push(engine::data_dir());

        // Move output files generated by ComfyUI to the Recycle Bin.
        if !node.output_filename.is_empty() {
            for dir in &recycle_dirs {
                let candidates = [
                    dir.join("output").join(&node.output_subfolder).join(&node.output_filename),
                    dir.join("output").join(&node.output_filename),
                    dir.join(&node.output_subfolder).join(&node.output_filename),
                    dir.join(&node.output_filename),
                ];
                for c in candidates {
                    if c.is_file() {
                        Self::move_to_recycle_bin(&c)?;
                        engine::log(&format!("Moved output file to Recycle Bin: {}", c.display()));
                    }
                }
            }
            // Remove any copy in AppData\Local\Temp directly.
            for c in [
                tmp_dir.join(&node.output_subfolder).join(&node.output_filename),
                tmp_dir.join(&node.output_filename),
            ] {
                if c.is_file() {
                    let _ = std::fs::remove_file(&c);
                    engine::log(&format!("Removed temp output file: {}", c.display()));
                }
            }
        }

        // Remove internal temp input images uploaded to ComfyUI.
        let mut temp_search_dirs = recycle_dirs;
        temp_search_dirs.push(tmp_dir.clone());
        for tf in &node.temp_files {
            for dir in &temp_search_dirs {
                let candidates = [
                    dir.join("temp").join(tf),
                    dir.join("input").join(tf),
                    dir.join(tf),
                ];
                for c in candidates {
                    if c.is_file() {
                        let _ = std::fs::remove_file(&c);
                        engine::log(&format!("Deleted temp file: {}", c.display()));
                    }
                }
            }
        }

        // Remove any temp player file that belonged to this node and clean up unused session temp videos.
        if let Some(ref tp) = dropped_temp_path {
            if tp.starts_with(&tmp_dir) && tp.is_file() {
                let _ = std::fs::remove_file(tp);
            }
        }
        Self::cleanup_unused_temp_videos(self.video_player.as_ref().map(|p| p.file_path.as_path()));

        // Remove cached copies in last_project_assets and refresh last_project.json if present.
        for ext in ["png", "mp4"] {
            let asset_path = assets_dir.join(format!("node_{}.{ext}", node_id));
            if asset_path.is_file() {
                let _ = std::fs::remove_file(&asset_path);
                engine::log(&format!("Deleted last_project_assets file: {}", asset_path.display()));
            }
        }
        if Self::last_project_json_path().is_file() {
            if self.nodes.is_empty() && self.refs.is_empty() {
                Self::clear_last_project();
            } else {
                self.save_last_project();
            }
        }

        // If the deleted node was currently active, switch to parent or last remaining node.
        if was_active {
            let next_id = node
                .parent_id
                .filter(|pid| self.nodes.iter().any(|n| n.id == *pid))
                .or_else(|| self.nodes.last().map(|n| n.id));
            if let Some(nid) = next_id {
                self.select_node(nid);
            } else {
                self.active_node_id = None;
                self.video_player = None;
                self.compare_mode = false;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------ History helpers
    fn active_node(&self) -> Option<&HistoryNode> {
        self.active_node_id.and_then(|id| self.find_node(id))
    }

    fn find_node(&self, id: usize) -> Option<&HistoryNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    fn parent_node(&self, id: usize) -> Option<&HistoryNode> {
        self.find_node(id).and_then(|n| n.parent_id).and_then(|pid| self.find_node(pid))
    }

    fn child_nodes(&self, id: usize) -> Vec<&HistoryNode> {
        self.nodes.iter().filter(|n| n.parent_id == Some(id)).collect()
    }

    fn can_undo(&self) -> bool {
        self.active_node().and_then(|n| n.parent_id).is_some()
    }

    fn can_redo(&self) -> bool {
        if let Some(id) = self.active_node_id {
            !self.child_nodes(id).is_empty()
        } else {
            false
        }
    }

    fn select_node(&mut self, id: usize) {
        if !self.generating {
            if let Some(curr_id) = self.active_node_id {
                if curr_id != id {
                    if let Some(curr_node) = self.nodes.iter_mut().find(|n| n.id == curr_id) {
                        curr_node.prompt = self.prompt.clone();
                    }
                }
            }
        }

        // Save current video player position, trim markers, and work mode before switching nodes.
        if !self.generating {
            if let Some(prev_id) = self.active_node_id {
                if prev_id != id {
                    if let Some(ref player) = self.video_player {
                        let cur_sec = player.current_sec;
                        let trim_s = player.trim_start_sec;
                        let trim_e = player.trim_end_sec;
                        let wm = self.video_work_mode;
                        if let Some(prev_node) = self.nodes.iter_mut().find(|n| n.id == prev_id) {
                            if prev_node.mp4_bytes.is_some() || prev_node.source_video_path.is_some() {
                                prev_node.video_last_sec = Some(cur_sec);
                                prev_node.video_trim_start = trim_s;
                                prev_node.video_trim_end = trim_e;
                                prev_node.video_work_mode = Some(wm);
                            }
                        }
                    }
                }
            }
        }

        self.active_node_id = Some(id);
        self.compare_mode = false;
        if !self.generating {
            let node_info = self.find_node(id).map(|n| (
                n.prompt.clone(),
                n.mp4_bytes.clone(),
                n.source_video_path.clone(),
                n.output_filename.clone(),
                n.output_subfolder.clone(),
                n.video_last_sec,
                n.video_trim_start,
                n.video_trim_end,
                n.video_work_mode,
            ));
            if let Some((
                prompt,
                mp4_bytes,
                source_video_path,
                out_fn,
                out_sub,
                last_sec,
                trim_s,
                trim_e,
                saved_work_mode,
            )) = node_info
            {
                if !prompt.is_empty() {
                    self.prompt = crate::lora::strip_prompt_tags(self.active_prompt_tags(), &prompt);
                    self.focus_prompt = true;
                }

                let has_disk_video = source_video_path.as_ref().is_some_and(|p| p.is_file());
                let is_video_node = mp4_bytes.is_some() || has_disk_video;

                // If selecting a video history node, load it in the video player
                if is_video_node {
                    let already_loaded = self.video_player.as_ref()
                        .map(|p| {
                            (has_disk_video && source_video_path.as_ref() == Some(&p.file_path))
                                || (p.filename == out_fn && !out_fn.is_empty())
                        })
                        .unwrap_or(false);
                    if !already_loaded {
                        let norm = self.current_audio_normalise();
                        let loaded_player = if let Some(ref src_path) = source_video_path.filter(|p| p.is_file()) {
                            crate::video::VideoPlayer::load_from_file(src_path, norm).map(|mut p| {
                                p.filename = out_fn.clone();
                                p.subfolder = out_sub.clone();
                                if p.mp4_bytes.is_none() {
                                    p.mp4_bytes = mp4_bytes.clone();
                                }
                                p
                            })
                        } else if let Some(mp4) = mp4_bytes {
                            crate::video::VideoPlayer::from_mp4_bytes(mp4, out_fn, out_sub, norm)
                        } else {
                            Err("No video source available".into())
                        };
                        if let Ok(mut player) = loaded_player {
                            player.trim_start_sec = trim_s;
                            player.trim_end_sec = trim_e;
                            if let Some(sec) = last_sec {
                                if sec > 0.001 {
                                    player.seek_to_time(sec);
                                }
                            }
                            self.video_player = Some(player);
                            self.canvas_zoom = 1.0;
                            self.canvas_pan = vec2(0.0, 0.0);
                            self.video_pan_drag_active = false;
                            self.trim_undo_stack.clear();
                            self.trim_redo_stack.clear();
                        }
                    } else if let Some(ref mut player) = self.video_player {
                        player.trim_start_sec = trim_s;
                        player.trim_end_sec = trim_e;
                        if let Some(sec) = last_sec {
                            player.seek_to_time(sec);
                        }
                    }
                    if let Some(wm) = saved_work_mode {
                        self.set_video_work_mode(wm);
                    }
                } else {
                    // Selecting an image history node restores image canvas view
                    self.video_player = None;
                }
            }
        }
        self.update_dimensions_from_active_node();
    }

    fn undo(&mut self) {
        if let Some(n) = self.active_node() {
            if let Some(pid) = n.parent_id {
                self.select_node(pid);
            }
        }
    }

    fn redo(&mut self) {
        if let Some(id) = self.active_node_id {
            let children = self.child_nodes(id);
            if let Some(child) = children.last() {
                self.select_node(child.id);
            }
        }
    }

    /// Returns the active linear sequence of node IDs from root to current tip.
    fn active_thread_ids(&self) -> Vec<usize> {
        let active_id = match self.active_node_id {
            Some(id) => id,
            None => return vec![],
        };

        let mut path_up = vec![active_id];
        let mut curr = active_id;
        while let Some(parent) = self.find_node(curr).and_then(|n| n.parent_id) {
            path_up.push(parent);
            curr = parent;
        }
        path_up.reverse();

        let mut curr = active_id;
        while let Some(child) = self.child_nodes(curr).last() {
            path_up.push(child.id);
            curr = child.id;
        }

        path_up
    }

    fn leaf_nodes(&self) -> Vec<&HistoryNode> {
        self.nodes.iter().filter(|n| self.child_nodes(n.id).is_empty()).collect()
    }

    fn get_branches(&self) -> Vec<(usize, String, bool)> {
        let active_thread = self.active_thread_ids();
        let active_tip = active_thread.last().copied();
        let leaves = self.leaf_nodes();
        if leaves.len() <= 1 {
            return vec![];
        }

        let mut branches = vec![];
        for (i, leaf) in leaves.iter().enumerate() {
            let is_curr = Some(leaf.id) == active_tip;
            let name = if is_curr {
                format!("Current (Step {})", leaf.step_num)
            } else {
                format!("Branch {} (Step {})", i + 1, leaf.step_num)
            };
            branches.push((leaf.id, name, is_curr));
        }
        branches
    }

    fn update_dimensions_from_active_node(&mut self) {
        if let Some((n_w, n_h, is_video_node)) = self
            .active_node()
            .map(|n| (n.width, n.height, n.mp4_bytes.is_some() || n.source_video_path.is_some()))
        {
            match self.size_preset {
                SizePreset::Original => {
                    self.custom_w = n_w;
                    self.custom_h = n_h;
                    self.ref_w = n_w;
                    self.ref_h = n_h;
                    self.scale = 1.0;
                }
                SizePreset::P720 => {
                    self.custom_w = 720;
                    self.custom_h = (((720.0 * n_h as f32 / n_w as f32) / 16.0).round() as u32 * 16).clamp(64, 4096);
                    self.ref_w = self.custom_w;
                    self.ref_h = self.custom_h;
                    self.scale = 1.0;
                }
                SizePreset::P1024 => {
                    self.custom_w = 1024;
                    self.custom_h = (((1024.0 * n_h as f32 / n_w as f32) / 16.0).round() as u32 * 16).clamp(64, 4096);
                    self.ref_w = self.custom_w;
                    self.ref_h = self.custom_h;
                    self.scale = 1.0;
                }
                SizePreset::Custom => {
                    // Retain user's custom aspect ratio and dimensions
                }
            }
            if self.mode == StudioMode::Video || is_video_node {
                self.set_video_resolution_from_source(n_w, n_h);
            }
        }
    }

    fn apply_preset(&mut self, idx: usize) {
        if let Some(p) = self.presets.get(idx).cloned() {
            self.active_preset_idx = Some(idx);
            self.steps = p.steps;
            self.size_preset = p.size_preset;
            self.resolution = p.resolution;
            self.cfg = p.cfg;
            self.custom_w = p.target_w;
            self.custom_h = p.target_h;

            if let Some((n_w, n_h)) = self.active_node().map(|n| (n.width, n.height)) {
                self.ref_w = n_w;
                self.ref_h = n_h;
                let (k_min, k_max) = absolute_scale_range(n_w, n_h, ABS_MIN_RES, ABS_MAX_RES);
                match p.size_preset {
                    SizePreset::Original => {
                        self.custom_w = n_w;
                        self.custom_h = n_h;
                        self.scale = 1.0;
                    }
                    SizePreset::P720 => {
                        self.custom_w = 720;
                        self.custom_h = (((720.0 * n_h as f32 / n_w as f32) / 16.0).round() as u32 * 16).clamp(64, 4096);
                        self.scale = (720.0 / n_w.max(1) as f32).clamp(k_min, k_max);
                    }
                    SizePreset::P1024 => {
                        self.custom_w = 1024;
                        self.custom_h = (((1024.0 * n_h as f32 / n_w as f32) / 16.0).round() as u32 * 16).clamp(64, 4096);
                        self.scale = (1024.0 / n_w.max(1) as f32).clamp(k_min, k_max);
                    }
                    SizePreset::Custom => {
                        if n_w > 0 {
                            self.scale = (p.target_w as f32 / n_w as f32).clamp(k_min, k_max);
                        }
                    }
                }
            }
        }
    }

    fn save_current_to_active_preset(&mut self) {
        if let Some(idx) = self.active_preset_idx {
            if idx < self.presets.len() {
                self.presets[idx].steps = self.steps;
                self.presets[idx].size_preset = self.size_preset;
                self.presets[idx].target_w = self.custom_w;
                self.presets[idx].target_h = self.custom_h;
                self.presets[idx].resolution = self.resolution;
                self.presets[idx].cfg = self.cfg;
                save_presets(&self.presets);
            }
        }
    }

    fn add_new_preset(&mut self) {
        let name = format!("Preset {}", self.presets.len() + 1);
        let new_preset = UserPreset {
            name,
            steps: self.steps,
            size_preset: self.size_preset,
            target_w: self.custom_w,
            target_h: self.custom_h,
            resolution: self.resolution,
            cfg: self.cfg,
        };
        self.presets.push(new_preset);
        self.active_preset_idx = Some(self.presets.len() - 1);
        save_presets(&self.presets);
    }

    fn delete_active_preset(&mut self) {
        if let Some(idx) = self.active_preset_idx {
            if self.presets.len() > 1 && idx < self.presets.len() {
                self.presets.remove(idx);
                self.active_preset_idx = Some(idx.min(self.presets.len() - 1));
                save_presets(&self.presets);
            }
        }
    }

    fn apply_size_preset(&mut self, preset: SizePreset) {
        self.size_preset = preset;
        if let Some((n_w, n_h)) = self.active_node().map(|n| (n.width, n.height)) {
            match preset {
                SizePreset::Original => {
                    self.custom_w = n_w;
                    self.custom_h = n_h;
                    self.ref_w = n_w;
                    self.ref_h = n_h;
                    self.scale = 1.0;
                }
                SizePreset::P720 => {
                    self.custom_w = 720;
                    self.custom_h = (((720.0 * n_h as f32 / n_w as f32) / 16.0).round() as u32 * 16).clamp(64, 4096);
                    self.ref_w = self.custom_w;
                    self.ref_h = self.custom_h;
                    self.scale = 1.0;
                }
                SizePreset::P1024 => {
                    self.custom_w = 1024;
                    self.custom_h = (((1024.0 * n_h as f32 / n_w as f32) / 16.0).round() as u32 * 16).clamp(64, 4096);
                    self.ref_w = self.custom_w;
                    self.ref_h = self.custom_h;
                    self.scale = 1.0;
                }
                SizePreset::Custom => {}
            }
        } else {
            match preset {
                SizePreset::P720 => {
                    self.custom_w = 720;
                    self.custom_h = 720;
                    self.ref_w = 720;
                    self.ref_h = 720;
                    self.scale = 1.0;
                }
                SizePreset::P1024 => {
                    self.custom_w = 1024;
                    self.custom_h = 1024;
                    self.ref_w = 1024;
                    self.ref_h = 1024;
                    self.scale = 1.0;
                }
                _ => {}
            }
        }
    }

    fn update_scale_dimensions(&mut self) {
        self.size_preset = SizePreset::Custom;
        if self.ref_w == 0 { self.ref_w = self.custom_w.max(ABS_MIN_RES); }
        if self.ref_h == 0 { self.ref_h = self.custom_h.max(ABS_MIN_RES); }
        let (w, h) = scale_dimensions_stepped(self.ref_w, self.ref_h, self.scale, 16, ABS_MIN_RES, ABS_MAX_RES);
        self.custom_w = w;
        self.custom_h = h;
    }

    fn set_video_resolution_from_source(&mut self, src_w: u32, src_h: u32) {
        if src_w == 0 || src_h == 0 {
            return;
        }
        let vw = (((src_w as f32) / 32.0).round() as u32 * 32).clamp(256, 4096);
        let vh = (((src_h as f32) / 32.0).round() as u32 * 32).clamp(256, 4096);
        self.video_orientation = if vh > vw {
            VideoOrientation::Portrait
        } else {
            VideoOrientation::Landscape
        };
        self.video_base_w = vw;
        self.video_base_h = vh;
        self.video_width = vw;
        self.video_height = vh;
        self.video_scale = 1.0;
    }

    fn load_video_file(&mut self, ctx: &egui::Context, path: PathBuf) {
        let norm = self.current_audio_normalise();
        match crate::video::VideoPlayer::load_from_file(&path, norm) {
            Ok(mut player) => {
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("video.mp4")
                    .to_string();
                player.filename = filename.clone();

                // Read bytes for HistoryNode if <= 256 MB
                let mp4_bytes = std::fs::metadata(&path)
                    .ok()
                    .filter(|m| m.len() <= 256 * 1024 * 1024)
                    .and_then(|_| std::fs::read(&path).ok());
                player.mp4_bytes = mp4_bytes.clone();

                if let Some(first) = player.current_image.as_ref() {
                    let id = self.next_node_id;
                    self.next_node_id += 1;
                    let tex = ctx.load_texture(format!("video_frame_{id}"), first.clone(), egui::TextureOptions::LINEAR);
                    let mut png_bytes = Vec::new();
                    if let Some(rgba) = image::RgbaImage::from_raw(
                        first.size[0] as u32,
                        first.size[1] as u32,
                        first.as_raw().to_vec(),
                    ) {
                        let _ = image::DynamicImage::ImageRgba8(rgba)
                            .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png);
                    }
                    let node = HistoryNode {
                        id,
                        parent_id: None,
                        label: filename.clone(),
                        prompt: String::new(),
                        png: png_bytes,
                        tex,
                        width: player.original_width,
                        height: player.original_height,
                        step_num: 0,
                        output_filename: filename,
                        output_subfolder: String::new(),
                        temp_files: Vec::new(),
                        mp4_bytes,
                        source_video_path: Some(path.clone()),
                        video_last_sec: None,
                        video_trim_start: None,
                        video_trim_end: None,
                        video_work_mode: Some(VideoWorkMode::Edit),
                    };
                    self.nodes.push(node);
                    self.active_node_id = Some(id);
                }

                let src_w = player.original_width;
                let src_h = player.original_height;
                let auto_switch = self.sh.cfg.lock().unwrap().auto_switch_mode_on_input;
                if auto_switch {
                    self.mode = StudioMode::Video;
                }
                self.video_player = Some(player);
                if self.mode == StudioMode::Video {
                    self.set_video_work_mode(VideoWorkMode::Edit);
                }
                self.set_video_resolution_from_source(src_w, src_h);
                self.trim_undo_stack.clear();
                self.trim_redo_stack.clear();
                self.error = None;
                self.status = format!("Loaded video: {}", path.display());
                ctx.request_repaint();
            }
            Err(e) => {
                self.error = Some(format!("Failed to load video {}: {e}", path.display()));
            }
        }
    }

    fn load(&mut self, ctx: &egui::Context, target: Target, src: Src) {
        self.loading += 1;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let res = (|| -> Result<(String, Vec<u8>, ColorImage, u32, u32), String> {
                let (name, mut dyn_img) = match src {
                    Src::Path(p) => {
                        let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("image").to_string();
                        let bytes = std::fs::read(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
                        let img = image::load_from_memory(&bytes).map_err(|e| format!("decode {}: {e}", p.display()))?;
                        (n, img)
                    }
                    Src::Rgba(w, h, raw) => {
                        let buf = image::RgbaImage::from_raw(w as u32, h as u32, raw).ok_or("invalid clipboard RGBA buffer")?;
                        ("clipboard".to_string(), image::DynamicImage::ImageRgba8(buf))
                    }
                };

                // Clamp dimensions to prevent excessive GPU texture allocations, upload payload limits (413), and CUDA OOM:
                // Reference images: max 2048px (the multimodal vision encoder processes image patches; >2048px causes extreme token load & OOM)
                // Base images: max 4096px (diffusion model ceiling and OpenGL texture limits)
                let max_dim = match target {
                    Target::Ref => 2048u32,
                    Target::Base => 4096u32,
                };
                if dyn_img.width() > max_dim || dyn_img.height() > max_dim {
                    dyn_img = dyn_img.resize(max_dim, max_dim, image::imageops::FilterType::Lanczos3);
                }

                let w = dyn_img.width();
                let h = dyn_img.height();
                let rgba = dyn_img.to_rgba8();
                let mut png = Vec::new();
                let enc = image::codecs::png::PngEncoder::new_with_quality(
                    &mut png,
                    image::codecs::png::CompressionType::Fast,
                    image::codecs::png::FilterType::NoFilter,
                );
                image::ImageEncoder::write_image(
                    enc,
                    rgba.as_raw(),
                    w,
                    h,
                    image::ExtendedColorType::Rgba8,
                )
                .map_err(|e| format!("encode png: {e}"))?;
                let ci = ColorImage::from_rgba_unmultiplied([w as usize, h as usize], rgba.as_raw());
                Ok((name, png, ci, w, h))
            })();

            match res {
                Ok((name, png, img, width, height)) => {
                    let _ = tx.send(Ev::Loaded {
                        target,
                        name,
                        png,
                        img,
                        width,
                        height,
                    });
                }
                Err(e) => {
                    let _ = tx.send(Ev::LoadErr(e));
                }
            }
            ctx.request_repaint();
        });
    }

    pub fn active_video_models_dir(&self) -> PathBuf {
        let models_dir = self.sh.paths.lock().unwrap().models.clone();
        if !models_dir.as_os_str().is_empty() {
            return models_dir;
        }
        let cfg = self.sh.cfg.lock().unwrap().clone();
        engine::candidate_models_dirs(&cfg)
            .into_iter()
            .find(|d| d.exists())
            .unwrap_or_default()
    }

    /// Returns the active video profile for the current state:
    /// - When a source video is loaded or an LTX-only / IC-LoRA is active, uses an LTX profile
    ///   (preferring the active profile if LTX, then the last used LTX profile, then the best downloaded LTX).
    /// - Otherwise uses the user's selected `active_video_profile` if downloaded (never falling back),
    ///   or falls back to the best downloaded profile if the selected one is still downloading.
    pub fn effective_video_profile(&self) -> Option<engine::Profile> {
        let dir = self.active_video_models_dir();
        if dir.as_os_str().is_empty() {
            return None;
        }
        let vprof = self.sh.active_video_profile.lock().unwrap().clone();
        let needs_h3 = crate::lora::has_active_h3_lora(&self.video_loras, &self.lora_browser.items);
        if needs_h3 {
            if vprof.is_minimax_h3() && vprof.missing_in_dir(&dir).is_empty() {
                return Some(vprof);
            }
            return engine::Profile::find_best_downloaded_h3_profile(&dir);
        }

        let needs_ltx = crate::lora::has_active_ltx_lora(&self.video_loras, &self.lora_browser.items);

        if needs_ltx {
            if !vprof.is_minimax_h3() && vprof.missing_in_dir(&dir).is_empty() {
                return Some(vprof);
            }
            return engine::Profile::find_best_downloaded_video_profile(
                &dir,
                true,
                self.last_ltx_video_profile_id.as_deref(),
            );
        }

        if vprof.missing_in_dir(&dir).is_empty() {
            return Some(vprof);
        }

        let same_family: Vec<engine::Profile> = engine::Profile::all_video_profiles()
            .into_iter()
            .filter(|p| p.is_minimax_h3() == vprof.is_minimax_h3() && p.missing_in_dir(&dir).is_empty())
            .collect();
        if !same_family.is_empty() {
            if let Some(best) = engine::Profile::pick_best_from_downloaded(&same_family, engine::detect_gpu_vram_gb()) {
                return Some(best);
            }
        }
        engine::Profile::find_best_downloaded_video_profile(
            &dir,
            false,
            self.last_ltx_video_profile_id.as_deref(),
        )
    }

    pub fn has_video_models(&self) -> bool {
        self.effective_video_profile().is_some()
    }

    pub fn is_edit_video_mode(&self) -> bool {
        self.mode == StudioMode::Video
            && self.video_player.is_some()
            && self.video_work_mode == VideoWorkMode::Edit
    }

    pub fn set_video_work_mode(&mut self, mode: VideoWorkMode) {
        if self.video_work_mode != mode {
            self.video_work_mode = mode;
            self.video_loras.clear();
            self.video_prompt_tags.clear();
            self.sync_effective_video_profile();
        }
    }

    pub fn active_video_lora_vram_gb(&self) -> f32 {
        let base: f32 = self.video_loras
            .iter()
            .map(|active| {
                let size_bytes = self
                    .lora_browser
                    .items
                    .iter()
                    .find(|it| it.filename == active.filename)
                    .map(|it| it.size_bytes)
                    .unwrap_or(0);
                crate::lora::estimate_lora_vram_gb(&active.filename, size_bytes)
            })
            .sum();
        if self.is_edit_video_mode() {
            // IC-LoRA doubles Stage 1 sequence length (target + guide tokens); scale overhead with frame/pixel load.
            let frames = (self.video_duration * self.video_fps) as f32;
            let mp = (self.video_width * self.video_height) as f32 / 1_000_000.0;
            let load_factor = mp * (frames / 24.0);
            base + (0.35 + load_factor * 0.25).clamp(0.4, 1.8)
        } else {
            base
        }
    }

    pub fn active_prompt_tags(&self) -> &[crate::lora::PromptTag] {
        if self.mode == StudioMode::Video {
            &self.video_prompt_tags
        } else {
            &self.image_prompt_tags
        }
    }

    pub fn has_prompt_content(&self) -> bool {
        !self.prompt.trim().is_empty() || !self.active_prompt_tags().is_empty()
    }

    pub fn composed_prompt(&self) -> String {
        crate::lora::compose_prompt_with_tags(self.active_prompt_tags(), &self.prompt)
    }

    fn start_generation(&mut self, ctx: &egui::Context) {
        self.generation_retries = 0;
        let p = self.composed_prompt();
        self.prompt.clear();
        if self.mode == StudioMode::Video {
            self.start_video_generation_with_prompt(ctx, p);
        } else {
            self.start_generation_with_prompt(ctx, p);
        }
    }

fn extract_trimmed_mp4_bytes(
    vp: &crate::video::VideoPlayer,
    trim_start: f64,
    trim_end: f64,
    target_fps: u32,
    max_frames: u32,
) -> Option<Vec<u8>> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_dir = std::env::temp_dir();
    let mut temp_in: Option<PathBuf> = None;
    let in_path = if vp.file_path.is_file() {
        vp.file_path.clone()
    } else if let Some(ref raw) = vp.mp4_bytes {
        let p = tmp_dir.join(format!("imgstudio_trim_in_{stamp}.mp4"));
        std::fs::write(&p, raw).ok()?;
        temp_in = Some(p.clone());
        p
    } else {
        return None;
    };
    let out_path = tmp_dir.join(format!("imgstudio_trim_out_{stamp}.mp4"));

    let fps_str = target_fps.max(1).to_string();
    let max_frames_str = max_frames.max(1).to_string();
    let mut cmd = std::process::Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-ss",
        &format!("{trim_start:.3}"),
        "-to",
        &format!("{trim_end:.3}"),
        "-i",
        in_path.to_str()?,
        "-r",
        &fps_str,
        "-frames:v",
        &max_frames_str,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "16",
        "-c:a",
        "aac",
        out_path.to_str()?,
    ]);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }
    let ok = cmd.output().ok().map(|o| o.status.success()).unwrap_or(false);
    if let Some(p) = temp_in {
        let _ = std::fs::remove_file(p);
    }
    let bytes = if ok { std::fs::read(&out_path).ok() } else { None };
    let _ = std::fs::remove_file(&out_path);
    bytes
}

    fn start_video_generation_with_prompt(&mut self, ctx: &egui::Context, prompt: String) {
        let active_vprof = self.sh.active_video_profile.lock().unwrap().clone();
        let vprof = self.effective_video_profile().unwrap_or_else(|| active_vprof.clone());
        let is_fallback = vprof.id != active_vprof.id;
        let arch = if vprof.is_minimax_h3() {
            comfy::VideoArch::MiniMaxH3
        } else if vprof.is_ltx25() {
            comfy::VideoArch::Ltx25
        } else {
            comfy::VideoArch::Ltx23
        };
        let model_label = if vprof.is_minimax_h3() {
            if is_fallback { "MiniMax H3 (fallback)" } else { "MiniMax H3" }
        } else if vprof.is_ltx25() {
            if is_fallback { "LTX-Video 2.5 (fallback)" } else { "LTX-Video 2.5" }
        } else {
            if is_fallback { "LTX-Video 2.3 (fallback)" } else { "LTX-Video 2.3" }
        };

        let is_edit_mode = self.is_edit_video_mode();
        let is_viggle_remake = is_edit_mode
            && self
                .video_loras
                .iter()
                .any(|l| l.filename.to_ascii_lowercase().contains("viggle_remake"));

        if is_viggle_remake {
            if let Some(comfy_dir) = self.sh.paths.lock().unwrap().comfy.clone() {
                if engine::ensure_viggle_remake_custom_node(&comfy_dir) {
                    self.sh.restart.store(true, Ordering::SeqCst);
                    self.prompt = crate::lora::strip_prompt_tags(self.active_prompt_tags(), &prompt);
                    self.status = "Updated Viggle Remake custom node — restarting ComfyUI, please click Edit Video again in a moment.".to_string();
                    return;
                }
            }
        }

        // If a video is currently loaded in the player, automatically extract the frame
        // the user has scrubbed to as the start frame for the next shot!
        // In H3 V2V edit mode, also extract the current frame as fallback ref_1 if no refs are attached.
        let extract_player_frame = |player: &crate::video::VideoPlayer| -> Option<(Vec<u8>, ColorImage)> {
            let cur_img = player.current_image.as_ref()?.clone();
            let [f_w, f_h] = cur_img.size;
            let mut png_bytes = Vec::new();
            if let Some(rgba) = image::RgbaImage::from_raw(f_w as u32, f_h as u32, cur_img.as_raw().to_vec()) {
                let _ = image::DynamicImage::ImageRgba8(rgba)
                    .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png);
            }
            if !png_bytes.is_empty() {
                Some((png_bytes, cur_img))
            } else {
                None
            }
        };

        let (base_png_opt, scrubbed_frame_ci) = if is_edit_mode {
            if arch == comfy::VideoArch::MiniMaxH3 && self.refs.is_empty() {
                if let Some((png, _)) = self.video_player.as_ref().and_then(extract_player_frame) {
                    (Some(png), None)
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        } else if let Some(ref player) = self.video_player {
            if let Some((png, ci)) = extract_player_frame(player) {
                (Some(png), Some(ci))
            } else {
                (self.active_node().map(|n| n.png.clone()), None)
            }
        } else {
            (self.active_node().map(|n| n.png.clone()), None)
        };

        let use_r2v = !is_edit_mode && arch == comfy::VideoArch::MiniMaxH3 && self.video_r2v_mode && !self.refs.is_empty();
        if use_r2v {
            let models_dir = self.active_video_models_dir();
            if !vprof.has_r2v_model(&models_dir) {
                self.prompt = crate::lora::strip_prompt_tags(self.active_prompt_tags(), &prompt);
                self.status = "Reference-to-Video requires the Ref2VA model. Download it in the top-left References panel.".to_string();
                return;
            }
        }
        let ref_images = if use_r2v || (is_edit_mode && arch == comfy::VideoArch::MiniMaxH3) {
            self.refs.iter().map(|r| r.png.clone()).collect()
        } else {
            Vec::new()
        };
        let last_frame = if !is_edit_mode && arch == comfy::VideoArch::MiniMaxH3 && !use_r2v && base_png_opt.is_some() && !self.refs.is_empty() {
            Some(self.refs[0].png.clone())
        } else {
            None
        };
        let is_t2v = base_png_opt.is_none() && !use_r2v;

        let (source_video_bytes, source_video_frames) = if is_edit_mode {
            if let Some(ref vp) = self.video_player {
                let trim_s = vp.trim_start();
                let trim_e = vp.trim_end();
                let trim_dur = (trim_e - trim_s).max(0.2);
                let target_fps = if arch == comfy::VideoArch::MiniMaxH3 { 24 } else { self.video_fps };
                let raw_frames = (trim_dur * target_fps as f64).round() as u32;
                let snapped_frames = if is_viggle_remake {
                    let clamped = raw_frames.clamp(90, 345);
                    ((clamped - 5) / 17) * 17 + 5
                } else if arch == comfy::VideoArch::MiniMaxH3 {
                    let rem = (5 - (raw_frames.max(5) % 17) as i32).rem_euclid(17) as u32;
                    raw_frames.max(5) + rem
                } else {
                    ((raw_frames.saturating_sub(1)) / 8).max(1) * 8 + 1
                };
                let is_trimmed = vp.trim_start_sec.is_some() || vp.trim_end_sec.is_some();
                let fps_mismatch = (vp.original_fps - target_fps as f32).abs() > 0.5;
                let bytes = if is_trimmed || fps_mismatch || is_viggle_remake {
                    Self::extract_trimmed_mp4_bytes(vp, trim_s, trim_e, target_fps, snapped_frames)
                        .or_else(|| vp.mp4_bytes.clone())
                        .or_else(|| std::fs::read(&vp.file_path).ok())
                } else {
                    vp.mp4_bytes
                        .clone()
                        .or_else(|| std::fs::read(&vp.file_path).ok())
                };
                let Some(b) = bytes else {
                    self.error = Some("Load or drop a source video to edit with this IC-LoRA.".into());
                    return;
                };
                self.last_edit_video_source = Some((b.clone(), snapped_frames));
                (Some(b), Some(snapped_frames))
            } else if let Some((ref cached_bytes, cached_frames)) = self.last_edit_video_source {
                (Some(cached_bytes.clone()), Some(cached_frames))
            } else {
                self.error = Some("Load or drop a source video to edit with this IC-LoRA.".into());
                return;
            }
        } else {
            self.last_edit_video_source = None;
            (None, None)
        };

        self.generating = true;
        self.status = if is_edit_mode {
            format!("Editing video with IC-LoRA ({model_label})...")
        } else if use_r2v {
            format!("Generating Reference-to-Video with {} ref(s) ({model_label})...", ref_images.len())
        } else if last_frame.is_some() {
            format!("Generating First+Last frame continuation ({model_label})...")
        } else if is_t2v {
            format!("Generating video from text ({model_label})...")
        } else {
            format!("Generating next shot from start frame ({model_label})...")
        };
        self.error = None;
        self.generating_parent_id = self.active_node_id;
        self.generating_prompt = prompt.trim().to_string();

        if let Some(pid) = self.active_node_id {
            let cur_sec = self.video_player.as_ref().map(|p| p.current_sec);
            let cur_trim_s = self.video_player.as_ref().and_then(|p| p.trim_start_sec);
            let cur_trim_e = self.video_player.as_ref().and_then(|p| p.trim_end_sec);
            let cur_wm = self.video_work_mode;
            if let Some(parent) = self.nodes.iter_mut().find(|n| n.id == pid) {
                parent.prompt = self.generating_prompt.clone();
                if let Some(sec) = cur_sec {
                    if parent.mp4_bytes.is_some() || parent.source_video_path.is_some() {
                        parent.video_last_sec = Some(sec);
                        parent.video_trim_start = cur_trim_s;
                        parent.video_trim_end = cur_trim_e;
                        parent.video_work_mode = Some(cur_wm);
                    }
                }
            }
        }

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = cancel.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();

        let seed = if self.random_seed { rand_u64() } else { self.seed };
        if self.random_seed {
            self.seed = seed;
        }

        self.generating_dims = Some((self.video_width, self.video_height));

        let has_day_to_night = self
            .video_loras
            .iter()
            .any(|l| l.filename.to_ascii_lowercase().contains("day-to-night") || l.filename.to_ascii_lowercase().contains("day_to_night"));
        let negative = if has_day_to_night && !self.video_negative.to_ascii_lowercase().contains("daytime") {
            let neg_prefix = "daytime, bright sunlight, blue sky, overexposed, ";
            if self.video_negative.trim().is_empty() {
                format!("{neg_prefix}worst quality, inconsistent motion, blurry, jittery, distorted")
            } else {
                format!("{neg_prefix}{}", self.video_negative.trim())
            }
        } else {
            self.video_negative.clone()
        };

        let vparams = comfy::VideoParams {
            arch,
            prompt: self.generating_prompt.clone(),
            negative,
            width: self.video_width,
            height: self.video_height,
            duration: self.video_duration,
            fps: self.video_fps,
            seed,
            steps: self.video_steps,
            cfg: vprof.default_cfg,
            sampler: self.video_sampler.clone(),
            scheduler: self.video_scheduler.clone(),
            denoise: 1.0,
            prompt_enhance: self.video_prompt_enhance,
            last_frame,
            ref_images,
            use_r2v,
            source_video_bytes,
            source_video_frames,
            stall_prompt: Some(self.stall_prompt.clone()),
            stall_keep_waiting: Some(self.stall_keep_waiting.clone()),
            loras: self.video_loras.clone(),
            unet_name: vprof.fl2va_unet_name(),
            ref2va_unet_name: vprof.ref2va_unet_name(),
            clip_name: vprof.clip_name(),
            vae_name: vprof.video_vae_name(),
            audio_vae_name: vprof.audio_vae_name(),
            ckpt_name: vprof.file("checkpoints"),
            text_encoder: vprof.clip_name(),
            latent_upscale_model: vprof.upscaler_name(),
            ..Default::default()
        };

        self.progress = None;
        self.total_gen_progress = Some(0.02);
        if let Some(ci) = scrubbed_frame_ci {
            self.preview_tex = Some(ctx.load_texture("locked_start_frame", ci, egui::TextureOptions::LINEAR));
        } else {
            self.preview_tex = None;
        }
        if let Some(ref mut player) = self.video_player {
            player.pause();
        }
        self.last_progress_time = None;
        self.last_progress_val = 0;
        self.step_duration_ema = 0.0;
        self.is_swapping_detected = false;
        self.stall_prompt.store(false, Ordering::SeqCst);
        self.stall_keep_waiting.store(false, Ordering::SeqCst);

        let tx_p = tx.clone();
        let ctx_p = ctx.clone();
        let prog_cb = Arc::new(move |value: u32, max: u32| {
            let _ = tx_p.send(Ev::Progress { value, max });
            ctx_p.request_repaint();
        });

        let tx_tp = tx.clone();
        let ctx_tp = ctx.clone();
        let total_prog_cb = Arc::new(move |frac: f32| {
            let _ = tx_tp.send(Ev::TotalProgress(frac));
            ctx_tp.request_repaint();
        });

        let tx_prev = tx.clone();
        let ctx_prev = ctx.clone();
        let prev_cb = Arc::new(move |bytes: Vec<u8>| {
            if let Ok(dyn_img) = image::load_from_memory(&bytes) {
                let rgba = dyn_img.to_rgba8();
                let ci = ColorImage::from_rgba_unmultiplied(
                    [dyn_img.width() as usize, dyn_img.height() as usize],
                    rgba.as_raw(),
                );
                let _ = tx_prev.send(Ev::Preview(ci));
                ctx_prev.request_repaint();
            }
        });

        std::thread::spawn(move || {
            let res = comfy::run_video(
                &vparams,
                base_png_opt,
                &cancel,
                &|st: String| {
                    let _ = tx.send(Ev::Status(st));
                    ctx.request_repaint();
                },
                prog_cb,
                total_prog_cb,
                prev_cb,
            );
            match res {
                Ok(gen_res) => {
                    let _ = tx.send(Ev::VideoDone {
                        mp4_bytes: gen_res.mp4,
                        output_filename: gen_res.output_filename,
                        output_subfolder: gen_res.output_subfolder,
                        temp_files: gen_res.temp_files,
                    });
                }
                Err(e) => {
                    let _ = tx.send(Ev::GenErr(e));
                }
            }
            ctx.request_repaint();
        });
    }

    fn start_generation_with_prompt(&mut self, ctx: &egui::Context, prompt: String) {
        let base_png_opt = if let Some(ref player) = self.video_player {
            player
                .current_image
                .as_ref()
                .and_then(|cur_img| {
                    let [f_w, f_h] = cur_img.size;
                    let mut png_bytes = Vec::new();
                    if let Some(rgba) = image::RgbaImage::from_raw(
                        f_w as u32,
                        f_h as u32,
                        cur_img.as_raw().to_vec(),
                    ) {
                        let _ = image::DynamicImage::ImageRgba8(rgba)
                            .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png);
                    }
                    if !png_bytes.is_empty() {
                        Some(png_bytes)
                    } else {
                        None
                    }
                })
                .or_else(|| self.active_node().map(|n| n.png.clone()))
        } else {
            self.active_node().map(|n| n.png.clone())
        };
        let is_t2i = base_png_opt.is_none();

        self.generating = true;
        self.status = if is_t2i { "Generating image...".into() } else { "Uploading images...".into() };
        self.error = None;
        self.generating_parent_id = self.active_node_id;
        self.generating_prompt = prompt.trim().to_string();

        // Save the prompt on the parent node that created this generation
        if let Some(pid) = self.active_node_id {
            let cur_sec = self.video_player.as_ref().map(|p| p.current_sec);
            let cur_trim_s = self.video_player.as_ref().and_then(|p| p.trim_start_sec);
            let cur_trim_e = self.video_player.as_ref().and_then(|p| p.trim_end_sec);
            let cur_wm = self.video_work_mode;
            if let Some(parent) = self.nodes.iter_mut().find(|n| n.id == pid) {
                parent.prompt = self.generating_prompt.clone();
                if let Some(sec) = cur_sec {
                    if parent.mp4_bytes.is_some() || parent.source_video_path.is_some() {
                        parent.video_last_sec = Some(sec);
                        parent.video_trim_start = cur_trim_s;
                        parent.video_trim_end = cur_trim_e;
                        parent.video_work_mode = Some(cur_wm);
                    }
                }
            }
        }

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = cancel.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();

        let mut pngs = Vec::new();
        if let Some(base_png) = base_png_opt {
            pngs.push(base_png);
        }
        for r in &self.refs {
            pngs.push(r.png.clone());
        }

        let seed = if self.random_seed { rand_u64() } else { self.seed };
        if self.random_seed {
            self.seed = seed;
        }

        let custom = match self.size_preset {
            SizePreset::Original if !is_t2i => None,
            _ => Some((self.custom_w.max(64), self.custom_h.max(64))),
        };

        let (target_w, target_h) = if let Some((cw, ch)) = custom {
            (cw, ch)
        } else if let Some(n) = self.active_node() {
            (n.width, n.height)
        } else {
            (self.custom_w.max(64), self.custom_h.max(64))
        };
        self.generating_dims = Some((target_w, target_h));

        let params = comfy::Params {
            prompt: self.generating_prompt.clone(),
            negative: self.negative.clone(),
            steps: self.steps,
            cfg: self.cfg,
            seed,
            resolution: self.resolution,
            custom,
            unet: {
                let f = self.profile.file("diffusion_models");
                if f.is_empty() { self.profile.file("unet") } else { f }
            },
            clip: {
                let f = self.profile.file("text_encoders");
                if f.is_empty() { self.profile.file("clip") } else { f }
            },
            vae: {
                let f = self.profile.file("vae");
                if f.is_empty() { self.profile.file("vae_models") } else { f }
            },
            sampler: self.profile.sampler.clone(),
            scheduler: self.profile.scheduler.clone(),
            loras: self.image_loras.clone(),
        };

        self.progress = None;
        self.total_gen_progress = Some(0.02);
        self.preview_tex = None;
        self.last_progress_time = None;
        self.last_progress_val = 0;
        self.step_duration_ema = 0.0;
        self.is_swapping_detected = false;
        self.stall_prompt.store(false, Ordering::SeqCst);
        self.stall_keep_waiting.store(false, Ordering::SeqCst);
        let tx_p = tx.clone();
        let ctx_p = ctx.clone();
        let prog_cb = Arc::new(move |value: u32, max: u32| {
            let _ = tx_p.send(Ev::Progress { value, max });
            ctx_p.request_repaint();
        });

        let tx_tp = tx.clone();
        let ctx_tp = ctx.clone();
        let total_prog_cb = Arc::new(move |frac: f32| {
            let _ = tx_tp.send(Ev::TotalProgress(frac));
            ctx_tp.request_repaint();
        });

        let tx_prev = tx.clone();
        let ctx_prev = ctx.clone();
        let prev_cb = Arc::new(move |bytes: Vec<u8>| {
            if let Ok(dyn_img) = image::load_from_memory(&bytes) {
                let rgba = dyn_img.to_rgba8();
                let ci = ColorImage::from_rgba_unmultiplied(
                    [dyn_img.width() as usize, dyn_img.height() as usize],
                    rgba.as_raw(),
                );
                let _ = tx_prev.send(Ev::Preview(ci));
                ctx_prev.request_repaint();
            }
        });

        std::thread::spawn(move || {
            let res = comfy::run(
                &params,
                pngs,
                &cancel,
                &|st: String| {
                    let _ = tx.send(Ev::Status(st));
                    ctx.request_repaint();
                },
                prog_cb,
                total_prog_cb,
                prev_cb,
            );
            match res {
                Ok(gen_res) => {
                    if let Ok(dyn_img) = image::load_from_memory(&gen_res.png) {
                        let rgba = dyn_img.to_rgba8();
                        let ci = ColorImage::from_rgba_unmultiplied(
                            [dyn_img.width() as usize, dyn_img.height() as usize],
                            rgba.as_raw(),
                        );
                        let _ = tx.send(Ev::Done {
                            png: gen_res.png,
                            img: ci,
                            output_filename: gen_res.output_filename,
                            output_subfolder: gen_res.output_subfolder,
                            temp_files: gen_res.temp_files,
                        });
                    } else {
                        let _ = tx.send(Ev::GenErr("Failed to decode output PNG".into()));
                    }
                }
                Err(e) => {
                    let _ = tx.send(Ev::GenErr(e));
                }
            }
            ctx.request_repaint();
        });
    }

    fn drain(&mut self, ctx: &egui::Context) {
        while let Ok(ev) = self.rx.try_recv() {
            match ev {
                Ev::Loaded {
                    target,
                    name,
                    png,
                    img,
                    width,
                    height,
                } => {
                    self.loading = self.loading.saturating_sub(1);
                    let tex = ctx.load_texture(&name, img, egui::TextureOptions::LINEAR);
                    match target {
                        Target::Base => {
                            if self.sh.cfg.lock().unwrap().auto_switch_mode_on_input {
                                self.mode = StudioMode::Image;
                            }
                            let id = self.next_node_id;
                            self.next_node_id += 1;
                            let label = if name.is_empty() || name == "clipboard" {
                                "Original Image".to_string()
                            } else {
                                name
                            };
                            let node = HistoryNode {
                                id,
                                parent_id: None,
                                label,
                                prompt: String::new(),
                                png,
                                tex,
                                width,
                                height,
                                step_num: 0,
                                output_filename: String::new(),
                                output_subfolder: String::new(),
                                temp_files: Vec::new(),
                                mp4_bytes: None,
                                source_video_path: None,
                                video_last_sec: None,
                                video_trim_start: None,
                                video_trim_end: None,
                                video_work_mode: None,
                            };
                            self.nodes.push(node);
                            self.ref_w = width;
                            self.ref_h = height;
                            self.custom_w = width;
                            self.custom_h = height;
                            self.scale = 1.0;
                            self.size_preset = SizePreset::Original;
                            if self.mode == StudioMode::Video {
                                self.set_video_resolution_from_source(width, height);
                            }
                            self.select_node(id);
                            self.canvas_zoom = 1.0;
                            self.canvas_pan = vec2(0.0, 0.0);
                        }
                        Target::Ref if self.refs.len() < MAX_REFS => {
                            self.refs.push(Item {
                                name,
                                png,
                                tex,
                                width,
                                height,
                            });
                        }
                        Target::Ref => self.error = Some(format!("Up to {MAX_REFS} reference images are supported.")),
                    }
                }
                Ev::LoadErr(e) => {
                    self.loading = self.loading.saturating_sub(1);
                    self.error = Some(e);
                }
                Ev::Status(s) => self.status = s,
                Ev::Progress { value, max } => {
                    self.progress = Some((value, max));
                    let now = Instant::now();
                    if let Some(last_t) = self.last_progress_time {
                        let elapsed = last_t.elapsed().as_secs_f32();
                        let step_diff = value.saturating_sub(self.last_progress_val);
                        if step_diff > 0 {
                            let per_step = elapsed / step_diff as f32;
                            if self.step_duration_ema <= 0.0 {
                                self.step_duration_ema = per_step;
                            } else {
                                self.step_duration_ema = self.step_duration_ema * 0.7 + per_step * 0.3;
                            }
                            if self.mode == StudioMode::Video {
                                let lora_vram = self.active_video_lora_vram_gb();
                                let vram_exceeded = if let Some(gpu_vram) = engine::detect_gpu_vram_gb() {
                                    estimate_video_vram_gb_with_lora(self.video_width, self.video_height, self.video_duration, self.video_fps, lora_vram) > gpu_vram as f32
                                } else {
                                    false
                                };
                                // Normal video diffusion step for LTX-2.3 is ~6s - 10s.
                                // Severe shared system memory swapping causes steps to exceed 40s or stall.
                                if (per_step > 40.0 || (self.step_duration_ema > 35.0 && value > 2)) && (vram_exceeded || per_step > 60.0) {
                                    self.is_swapping_detected = true;
                                }
                            } else {
                                // Normal image diffusion step is ~0.2s - 1.5s.
                                if per_step > 8.0 || (self.step_duration_ema > 6.0 && value > 2) {
                                    self.is_swapping_detected = true;
                                }
                            }
                        }
                    }
                    self.last_progress_time = Some(now);
                    self.last_progress_val = value;
                }
                Ev::TotalProgress(frac) => {
                    self.total_gen_progress = Some(frac.clamp(0.0, 1.0));
                }
                Ev::Preview(img) => {
                    if self.generating {
                        self.preview_tex = Some(ctx.load_texture("live_preview", img, egui::TextureOptions::LINEAR));
                    }
                }
                Ev::Done { png, img, output_filename, output_subfolder, temp_files } => {
                    self.generating = false;
                    self.generating_dims = None;
                    self.status = "Idle".into();
                    self.progress = None;
                    self.total_gen_progress = None;
                    self.preview_tex = None;
                    self.is_swapping_detected = false;
        self.stall_prompt.store(false, Ordering::SeqCst);
        self.stall_keep_waiting.store(false, Ordering::SeqCst);
                    let [out_w, out_h] = img.size;
                    let id = self.next_node_id;
                    self.next_node_id += 1;
                    let tex = ctx.load_texture(format!("edit_{id}"), img, egui::TextureOptions::LINEAR);

                    let parent_id = self.generating_parent_id;
                    let parent_step = parent_id.and_then(|pid| self.find_node(pid)).map(|n| n.step_num).unwrap_or(0);
                    let step_num = if parent_id.is_some() { parent_step + 1 } else { 0 };
                    let prompt_label = if self.generating_prompt.is_empty() {
                        format!("Image {}", id)
                    } else {
                        let p = &self.generating_prompt;
                        if p.chars().count() > 22 {
                            format!("{}...", p.chars().take(22).collect::<String>())
                        } else {
                            p.clone()
                        }
                    };

                    // Ensure parent node keeps this prompt
                    if let Some(pid) = parent_id {
                        if let Some(parent) = self.nodes.iter_mut().find(|n| n.id == pid) {
                            parent.prompt = self.generating_prompt.clone();
                        }
                    }

                    let node = HistoryNode {
                        id,
                        parent_id,
                        label: prompt_label,
                        prompt: String::new(),
                        png,
                        tex,
                        width: out_w as u32,
                        height: out_h as u32,
                        step_num,
                        output_filename,
                        output_subfolder,
                        temp_files,
                        mp4_bytes: None,
                        source_video_path: None,
                        video_last_sec: None,
                        video_trim_start: None,
                        video_trim_end: None,
                        video_work_mode: None,
                    };
                    self.nodes.push(node);
                    self.active_node_id = Some(id);
                    self.video_player = None;
                    self.compare_mode = false;
                    self.generation_retries = 0;

                    // Maintain newly generated resolution while resetting scale slider to 100% (1.0x) in the middle
                    self.custom_w = out_w as u32;
                    self.custom_h = out_h as u32;
                    self.ref_w = out_w as u32;
                    self.ref_h = out_h as u32;
                    self.scale = 1.0;
                    self.size_preset = SizePreset::Original;

                    self.split = 0.5;

                    // Sequentially process next queued prompt in the chain
                    if !self.queue.is_empty() {
                        let next = self.queue.remove(0);
                        if self.mode == StudioMode::Video {
                            self.start_video_generation_with_prompt(ctx, next.prompt);
                        } else {
                            self.start_generation_with_prompt(ctx, next.prompt);
                        }
                    }
                }
                Ev::VideoDone { mp4_bytes, output_filename, output_subfolder, temp_files } => {
                    self.generating = false;
                    self.generating_dims = None;
                    self.status = "Idle".into();
                    self.progress = None;
                    self.total_gen_progress = None;
                    self.preview_tex = None;
                    self.is_swapping_detected = false;
        self.stall_prompt.store(false, Ordering::SeqCst);
        self.stall_keep_waiting.store(false, Ordering::SeqCst);
                    self.generation_retries = 0;

                    let norm = self.current_audio_normalise();
                    let disk_video_path = (!output_filename.is_empty()).then(|| {
                        let comfy_dir = self.sh.paths.lock().unwrap().comfy.clone();
                        let mut dirs = Vec::new();
                        if let Some(ref cp) = comfy_dir {
                            dirs.push(cp.clone());
                            if let Some(parent) = cp.parent() {
                                dirs.push(parent.to_path_buf());
                            }
                        }
                        dirs.into_iter().find_map(|dir| {
                            [
                                dir.join("output").join(&output_subfolder).join(&output_filename),
                                dir.join("output").join(&output_filename),
                            ]
                            .into_iter()
                            .find(|c| c.is_file())
                        })
                    }).flatten();
                    let player_res = if let Some(ref disk_path) = disk_video_path {
                        crate::video::VideoPlayer::load_from_file(disk_path, norm).map(|mut p| {
                            p.filename = output_filename.clone();
                            p.subfolder = output_subfolder.clone();
                            p.mp4_bytes = Some(mp4_bytes);
                            p
                        })
                    } else {
                        crate::video::VideoPlayer::from_mp4_bytes(mp4_bytes, output_filename.clone(), output_subfolder.clone(), norm)
                    };
                    match player_res {
                        Ok(player) => {
                            if let Some(first) = player.current_image.as_ref() {
                                let id = self.next_node_id;
                                self.next_node_id += 1;
                                let tex = ctx.load_texture(format!("video_frame_{id}"), first.clone(), egui::TextureOptions::LINEAR);
                                let parent_id = self.generating_parent_id;
                                let parent_step = parent_id.and_then(|pid| self.find_node(pid)).map(|n| n.step_num).unwrap_or(0);
                                let step_num = if parent_id.is_some() { parent_step + 1 } else { 0 };
                                let prompt_label = if self.generating_prompt.is_empty() {
                                    format!("Video {}", id)
                                } else {
                                    let p = &self.generating_prompt;
                                    if p.chars().count() > 22 {
                                        format!("{}...", p.chars().take(22).collect::<String>())
                                    } else {
                                        p.clone()
                                    }
                                };

                                let mut png_bytes = Vec::new();
                                if let Some(rgba) = image::RgbaImage::from_raw(
                                    first.size[0] as u32,
                                    first.size[1] as u32,
                                    first.as_raw().to_vec(),
                                ) {
                                    let _ = image::DynamicImage::ImageRgba8(rgba)
                                        .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png);
                                }

                                let node = HistoryNode {
                                    id,
                                    parent_id,
                                    label: prompt_label,
                                    prompt: self.generating_prompt.clone(),
                                    png: png_bytes,
                                    tex,
                                    width: player.original_width,
                                    height: player.original_height,
                                    step_num,
                                    output_filename,
                                    output_subfolder,
                                    temp_files,
                                    mp4_bytes: player.mp4_bytes.clone(),
                                    source_video_path: disk_video_path.clone(),
                                    video_last_sec: None,
                                    video_trim_start: None,
                                    video_trim_end: None,
                                    video_work_mode: Some(self.video_work_mode),
                                };
                                self.nodes.push(node);
                                self.active_node_id = Some(id);
                            }
                            self.video_player = Some(player);
                            self.canvas_zoom = 1.0;
                            self.canvas_pan = vec2(0.0, 0.0);
                            self.video_pan_drag_active = false;
                            self.trim_undo_stack.clear();
                            self.trim_redo_stack.clear();
                        }
                        Err(e) => {
                            self.error = Some(format!("Failed to decode video: {e}"));
                        }
                    }

                    if !self.queue.is_empty() {
                        let next = self.queue.remove(0);
                        if self.mode == StudioMode::Video {
                            self.start_video_generation_with_prompt(ctx, next.prompt);
                        } else {
                            self.start_generation_with_prompt(ctx, next.prompt);
                        }
                    }
                }
                Ev::GenErr(e) => {
                    self.generating = false;
                    self.generating_dims = None;
                    self.progress = None;
                    self.total_gen_progress = None;
                    self.preview_tex = None;
                    self.is_swapping_detected = false;
        self.stall_prompt.store(false, Ordering::SeqCst);
        self.stall_keep_waiting.store(false, Ordering::SeqCst);
                    self.queue.clear();
                    if e == "Cancelled" {
                        self.status = "Cancelled".into();
                        self.generation_retries = 0;
                    } else if (e.contains("stopped") || e.contains("without producing")) && self.generation_retries < 1 {
                        self.generation_retries += 1;
                        self.status = "Engine stalled; automatically retrying generation (1/1)...".into();
                        let retry_prompt = if !self.generating_prompt.trim().is_empty() {
                            self.generating_prompt.clone()
                        } else {
                            self.prompt.clone()
                        };
                        if self.mode == StudioMode::Video {
                            self.start_video_generation_with_prompt(ctx, retry_prompt);
                        } else {
                            self.start_generation_with_prompt(ctx, retry_prompt);
                        }
                        return;
                    } else {
                        self.generation_retries = 0;
                        self.status = "Error".into();
                        self.error = Some(e);
                    }
                    if self.prompt.trim().is_empty() && !self.generating_prompt.trim().is_empty() {
                        self.prompt = crate::lora::strip_prompt_tags(self.active_prompt_tags(), &self.generating_prompt);
                    }
                }
            }
        }

        let is_ready = matches!(*self.sh.state.lock().unwrap(), EngState::Ready { .. });
        if is_ready && !self.generating && !self.queue.is_empty() {
            let next = self.queue.remove(0);
            if self.mode == StudioMode::Video {
                self.start_video_generation_with_prompt(ctx, next.prompt);
            } else {
                self.start_generation_with_prompt(ctx, next.prompt);
            }
        }

        if self.render_mgr.update() {
            if self.persist_trim_history {
                self.render_mgr.save_queue_history();
            }
            ctx.request_repaint();
        }
    }

    fn current_audio_normalise(&self) -> Option<crate::video::AudioNormaliseConfig> {
        if self.normalise_audio {
            Some(crate::video::AudioNormaliseConfig {
                ratio: self.audio_normalise_ratio,
                threshold_db: self.audio_normalise_thresh_db,
                boost_db: self.audio_normalise_boost_db,
            })
        } else {
            None
        }
    }

    fn is_auto_video_preset(&self) -> bool {
        self.video_presets
            .get(self.selected_video_preset_idx)
            .map(|p| p.id == "auto")
            .unwrap_or(false)
    }

    fn apply_auto_bitrate(&mut self) {
        let source_h = self
            .video_player
            .as_ref()
            .map(|p| p.original_height)
            .unwrap_or(self.video_height.max(1));
        let effective_h = match self.selected_res.max_height() {
            Some(cap_h) => source_h.min(cap_h),
            None => source_h,
        };
        let (avg, max) = crate::presets::auto_bitrate_range(effective_h, self.auto_quality_level);
        self.avg_bitrate = avg;
        self.max_bitrate = max;
    }

    fn sync_from_video_preset_idx(&mut self, idx: usize) {
        if idx < self.video_presets.len() {
            self.selected_video_preset_idx = idx;
            let p = &self.video_presets[idx];
            if p.id == "auto" {
                self.selected_res = TargetResolution::Source;
                self.selected_codec = p.codec;
                self.apply_auto_bitrate();
            } else {
                self.avg_bitrate = p.avg_bitrate_mbps;
                self.max_bitrate = p.max_bitrate_mbps;
                self.selected_codec = p.codec;
                self.selected_res = p.resolution;
            }
        }
    }

    fn record_trim_undo(&mut self, snapshot: TrimStateSnapshot) {
        self.trim_undo_stack.push(snapshot);
        if self.trim_undo_stack.len() > 64 {
            self.trim_undo_stack.remove(0);
        }
        self.trim_redo_stack.clear();
    }

    fn perform_trim_undo(&mut self) {
        if let Some(prev) = self.trim_undo_stack.pop() {
            if let Some(ref mut player) = self.video_player {
                let current = TrimStateSnapshot {
                    start: player.trim_start_sec,
                    end: player.trim_end_sec,
                };
                self.trim_redo_stack.push(current);
                player.trim_start_sec = prev.start;
                player.trim_end_sec = prev.end;
                let target = prev.start.unwrap_or(0.0);
                player.seek_to_time(target);
                self.status = "Undid trim change".into();
            }
        }
    }

    fn perform_trim_redo(&mut self) {
        if let Some(next) = self.trim_redo_stack.pop() {
            if let Some(ref mut player) = self.video_player {
                let current = TrimStateSnapshot {
                    start: player.trim_start_sec,
                    end: player.trim_end_sec,
                };
                self.trim_undo_stack.push(current);
                player.trim_start_sec = next.start;
                player.trim_end_sec = next.end;
                let target = next.start.unwrap_or(0.0);
                player.seek_to_time(target);
                self.status = "Redid trim change".into();
            }
        }
    }

    fn trigger_trim_and_compress(&mut self) {
        let Some(ref player) = self.video_player else {
            return;
        };
        let input_path = player.file_path.clone();
        let temp_guard = player.temp_file_guard.clone();
        let trim_start = if player.trim_start() > 0.01 {
            Some(player.trim_start())
        } else {
            None
        };
        let trim_end = if player.trim_end() < (player.duration_sec - 0.01) {
            Some(player.trim_end())
        } else {
            None
        };
        let total_duration = player.duration_sec;
        let base_name = if !player.filename.is_empty() {
            player.filename.clone()
        } else {
            input_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("video.mp4")
                .to_string()
        };

        let stem = PathBuf::from(&base_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("video")
            .to_string();
        let ext = PathBuf::from(&base_name)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("mp4")
            .to_string();
        let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let out_filename = format!("{stem}_trimmed_{stamp}.{ext}");
        let renders_dir = engine::data_dir().join("renders");
        let _ = std::fs::create_dir_all(&renders_dir);
        let explicit_output_path = renders_dir.join(&out_filename);

        let mut preset = self
            .video_presets
            .get(self.selected_video_preset_idx)
            .cloned()
            .unwrap_or_else(|| QualityPreset::default_presets()[0].clone());
        preset.avg_bitrate_mbps = self.avg_bitrate;
        preset.max_bitrate_mbps = self.max_bitrate;
        preset.codec = self.selected_codec;
        preset.resolution = self.selected_res;

        let req = RenderJobRequest {
            id: 0,
            input_path: input_path.clone(),
            trim_start,
            trim_end,
            total_duration,
            compress: self.compress_enabled,
            preset,
            replace_in_place: false,
            normalise_audio: self.normalise_audio,
            audio_normalise_ratio: self.audio_normalise_ratio,
            audio_normalise_thresh_db: self.audio_normalise_thresh_db,
            audio_normalise_boost_db: self.audio_normalise_boost_db,
            display_name: Some(out_filename.clone()),
            explicit_output_path: Some(explicit_output_path.clone()),
            temp_input_guard: temp_guard,
        };

        self.render_mgr.submit_job(req);
        self.trim_history.mark_trimmed(&explicit_output_path);
        if self.persist_trim_history {
            self.render_mgr.save_queue_history();
        }
        self.status = format!("Queued trim/compress: {out_filename}");
    }

    fn render_trim_history_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            heading(ui, "TRIM HISTORY");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.checkbox(&mut self.persist_trim_history, "Persist")
                    .on_hover_text("Keep trim & render history across app launches")
                    .changed()
                {
                    if self.persist_trim_history {
                        self.render_mgr.save_queue_history();
                    } else {
                        RenderManager::delete_queue_history_file();
                    }
                }
            });
        });

        let mut job_to_remove: Option<u64> = None;
        let mut path_to_load: Option<(PathBuf, String)> = None;
        let mut path_to_save: Option<(PathBuf, String)> = None;

        card(ui, |ui| {
            let max_w = ui.available_width();
            ui.set_max_width(max_w);
            egui::ScrollArea::vertical()
                .id_salt("trim_history_scroll")
                .max_height(220.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    if let Ok(jobs) = self.render_mgr.active_jobs.lock() {
                        if jobs.is_empty() {
                            ui.label(RichText::new("No trim or compression jobs yet.").size(11.0).color(DIM));
                        } else {
                            for j in jobs.iter().rev() {
                                egui::Frame::new()
                                    .fill(CARD_HI)
                                    .stroke(Stroke::new(1.0, LINE))
                                    .corner_radius(5)
                                    .inner_margin(egui::Margin::symmetric(6, 4))
                                    .show(ui, |ui| {
                                        ui.set_width(ui.available_width());
                                        ui.add(
                                            egui::Label::new(
                                                RichText::new(&j.file_name).size(11.0).strong().color(TEXT),
                                            )
                                            .truncate(),
                                        )
                                        .on_hover_text(&j.file_name);

                                        ui.horizontal(|ui| {
                                            match &j.state {
                                                JobState::Queued => {
                                                    ui.label(RichText::new("Queued").size(10.5).color(AMBER));
                                                }
                                                JobState::Running { progress, speed } => {
                                                    ui.ctx().request_repaint_after(Duration::from_millis(100));
                                                    ui.label(
                                                        RichText::new(format!("{:.0}% ({speed})", progress * 100.0))
                                                            .size(10.5)
                                                            .color(BLUE),
                                                    );
                                                }
                                                JobState::Finished => {
                                                    if let Some(ref out_p) = j.output_path {
                                                        if out_p.exists() {
                                                            if ui.small_button("Load").on_hover_text("Open in player").clicked() {
                                                                path_to_load = Some((out_p.clone(), j.file_name.clone()));
                                                            }
                                                            if ui.small_button("Save").on_hover_text("Save trimmed MP4 as...").clicked() {
                                                                path_to_save = Some((out_p.clone(), j.file_name.clone()));
                                                            }
                                                        } else {
                                                            ui.label(RichText::new("Done").size(10.5).color(GREEN));
                                                        }
                                                    } else {
                                                        ui.label(RichText::new("Done").size(10.5).color(GREEN));
                                                    }
                                                }
                                                JobState::Deleted => {
                                                    ui.label(RichText::new("Deleted").size(10.5).color(DIM));
                                                }
                                                JobState::Failed(err) => {
                                                    ui.label(RichText::new("Failed").size(10.5).color(RED))
                                                        .on_hover_text(err);
                                                }
                                            }

                                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                let x_btn = egui::Button::new(RichText::new("x").size(10.5).color(DIM))
                                                    .frame(false)
                                                    .min_size(vec2(16.0, 16.0));
                                                if ui.add(x_btn).on_hover_text("Remove from trim history").clicked() {
                                                    job_to_remove = Some(j.job_id);
                                                }
                                            });
                                        });
                                    });
                                ui.add_space(3.0);
                            }
                        }
                    }
                });
        });

        if let Some(id) = job_to_remove {
            self.render_mgr.remove_job(id);
            if self.persist_trim_history {
                self.render_mgr.save_queue_history();
            }
        }
        if let Some((path, name)) = path_to_load {
            let norm = self.current_audio_normalise();
            match crate::video::VideoPlayer::load_from_file(&path, norm) {
                Ok(mut player) => {
                    player.filename = name;
                    if let Ok(bytes) = std::fs::read(&path) {
                        player.mp4_bytes = Some(bytes);
                    }
                    self.video_player = Some(player);
                    self.trim_undo_stack.clear();
                    self.trim_redo_stack.clear();
                    ctx.request_repaint();
                }
                Err(e) => {
                    self.error = Some(format!("Could not load trimmed video: {e}"));
                }
            }
        }
        if let Some((src_path, name)) = path_to_save {
            if let Some(dst) = rfd::FileDialog::new()
                .set_file_name(&name)
                .add_filter("MP4 Video", &["mp4"])
                .save_file()
            {
                if let Err(e) = std::fs::copy(&src_path, &dst) {
                    self.error = Some(format!("Could not save trimmed MP4: {e}"));
                } else {
                    self.status = format!("Saved {}", dst.display());
                }
            }
        }
    }

    fn new_video_preset_modal(&mut self, ctx: &egui::Context) {
        if !self.new_video_preset_modal_open {
            return;
        }
        let mut open = true;
        let mut should_close = false;
        egui::Window::new("Save Video Compression Preset")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.label("Preset Name:");
                ui.text_edit_singleline(&mut self.new_video_preset_name);
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() && !self.new_video_preset_name.trim().is_empty() {
                        let name = self.new_video_preset_name.trim().to_string();
                        let id = format!(
                            "custom_{}",
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0)
                        );
                        let new_p = QualityPreset {
                            id,
                            name,
                            resolution: self.selected_res,
                            avg_bitrate_mbps: self.avg_bitrate,
                            max_bitrate_mbps: self.max_bitrate,
                            codec: self.selected_codec,
                            audio_bitrate_kbps: 192,
                            is_builtin: false,
                        };
                        self.video_presets.push(new_p);
                        self.selected_video_preset_idx = self.video_presets.len() - 1;
                        QualityPreset::save_presets(&self.video_presets);
                        self.new_video_preset_name.clear();
                        should_close = true;
                    }
                    if ui.button("Cancel").clicked() {
                        should_close = true;
                    }
                });
            });
        self.new_video_preset_modal_open = open && !should_close;
    }

    
fn fit_single_line_middle(p: &egui::Painter, text: &str, font: egui::FontId, max_w: f32) -> String {
    let g = p.layout_no_wrap(text.to_string(), font.clone(), Color32::WHITE);
    if g.size().x <= max_w {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    if total <= 6 {
        return text.to_string();
    }
    let mut low = 3;
    let mut high = total - 1;
    let mut best = format!("{}...", &chars[0..2].iter().collect::<String>());
    while low <= high {
        let keep = (low + high) / 2;
        let front_len = keep / 2;
        let back_len = keep - front_len;
        let candidate = format!(
            "{}...{}",
            chars[0..front_len].iter().collect::<String>(),
            chars[total - back_len..total].iter().collect::<String>()
        );
        let g = p.layout_no_wrap(candidate.clone(), font.clone(), Color32::WHITE);
        if g.size().x <= max_w {
            best = candidate;
            low = keep + 1;
        } else {
            if keep == 0 { break; }
            high = keep - 1;
        }
    }
    best
}

    
#[cfg(target_os = "windows")]
fn is_win_ctrl_v() -> bool {
    unsafe extern "system" {
        fn GetAsyncKeyState(v_key: i32) -> i16;
        fn GetForegroundWindow() -> isize;
        fn GetCurrentProcessId() -> u32;
        fn GetWindowThreadProcessId(h_wnd: isize, lpdw_process_id: *mut u32) -> u32;
    }
    let fg = unsafe { GetForegroundWindow() };
    if fg == 0 {
        return false;
    }
    let mut pid = 0u32;
    unsafe { GetWindowThreadProcessId(fg, &mut pid as *mut u32) };
    let my_pid = unsafe { GetCurrentProcessId() };
    if pid != my_pid {
        return false;
    }

    let ctrl = unsafe { (GetAsyncKeyState(0x11) as u16 & 0x8000) != 0 };
    let v = unsafe { (GetAsyncKeyState(0x56) as u16 & 0x8000) != 0 };
    ctrl && v
}

#[cfg(not(target_os = "windows"))]
fn is_win_ctrl_v() -> bool {
    false
}

    
    fn paste_from_clipboard(&mut self, ctx: &egui::Context) -> bool {
        for _ in 0..3 {
            if let Ok(mut cb) = arboard::Clipboard::new() {
                if let Ok(img) = cb.get_image() {
                    self.load(ctx, Target::Base, Src::Rgba(img.width, img.height, img.bytes.into_owned()));
                    return true;
                } else if let Ok(text) = cb.get_text() {
                    let trimmed = text.trim().trim_matches('"');
                    let path = PathBuf::from(trimmed);
                    if path.is_file() {
                        if let Some(ext) = path.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase()) {
                            if matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif") {
                                self.load(ctx, Target::Base, Src::Path(path));
                                return true;
                            }
                        }
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(15));
        }
        false
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        if !dropped.is_empty() {
            let drop_pos = ctx.input(|i| i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()));
            let dropped_in_refs = drop_pos.is_some_and(|pos| {
                self.refs_drop_rect
                    .is_some_and(|r| r.expand(6.0).contains(pos))
            });
            let dropped_in_canvas = drop_pos.is_some_and(|pos| {
                self.canvas_drop_rect.is_some_and(|r| r.contains(pos))
            });
            let mut need_base = self.active_node().is_none();
            for p in dropped {
                let is_vid = p
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "mp4" | "mov" | "mkv" | "webm" | "avi" | "m4v"))
                    .unwrap_or(false);
                if is_vid {
                    self.load_video_file(ctx, p);
                } else if dropped_in_refs {
                    self.load(ctx, Target::Ref, Src::Path(p));
                } else if dropped_in_canvas {
                    self.load(ctx, Target::Base, Src::Path(p));
                    need_base = false;
                } else {
                    let t = if need_base { Target::Base } else { Target::Ref };
                    need_base = false;
                    self.load(ctx, t, Src::Path(p));
                }
            }
        }
        let win_down = Self::is_win_ctrl_v();
        let win_paste = win_down && !self.last_ctrl_v_down;
        self.last_ctrl_v_down = win_down;

        let egui_paste = ctx.input(|i| {
            i.raw.events.iter().any(|e| matches!(e, egui::Event::Paste(_)))
                || (i.modifiers.command && i.key_pressed(egui::Key::V))
        });

        if win_paste || egui_paste {
            let pasted = self.paste_from_clipboard(ctx);
            if pasted {
                if let Ok(mut cb) = arboard::Clipboard::new() {
                    if let Ok(text) = cb.get_text() {
                        let t = text.trim().trim_matches('"');
                        if !t.is_empty() && self.prompt.ends_with(t) {
                            let new_len = self.prompt.len() - t.len();
                            self.prompt.truncate(new_len);
                        }
                    }
                }
                while self.prompt.ends_with('\n') || self.prompt.ends_with('\r') {
                    self.prompt.pop();
                }
            }
        }

        // Undo / Redo keyboard shortcuts (image/node history; video trim undo is handled in render_video_canvas)
        if self.video_player.is_none() {
            let undo = ctx.input(|i| i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::Z));
            let redo = ctx.input(|i| (i.modifiers.command && i.key_pressed(egui::Key::Y)) || (i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::Z)));
            if undo && !ctx.egui_wants_keyboard_input() && self.can_undo() {
                self.undo();
            }
            if redo && !ctx.egui_wants_keyboard_input() && self.can_redo() {
                self.redo();
            }
        }
    }

    #[allow(dead_code)]
    fn engine_ready(&self) -> bool {
        matches!(*self.sh.state.lock().unwrap(), EngState::Ready { .. })
    }

    fn can_queue(&self) -> Result<(), &'static str> {
        let st = self.sh.state.lock().unwrap().clone();
        match st {
            EngState::Ready { .. }
            | EngState::Starting { .. }
            | EngState::Detecting
            | EngState::AwaitingDownloadConfirmation { .. }
            | EngState::DownloadingEngine { .. }
            | EngState::ExtractingEngine { .. }
            | EngState::DownloadingModels { .. }
            | EngState::ModelDownloadPaused { .. }
            | EngState::RepairingEngine { .. } => {}
            EngState::ModelDownloadFailed { .. } => return Err("Model download interrupted - check downloads"),
            EngState::Failed(_) => return Err("Engine setup failed"),
        }
        if self.mode == StudioMode::Video {
            if !self.has_video_models() && !self.sh.dl_video_model.load(Ordering::SeqCst) {
                return Err("Video model download disabled in settings");
            }
        } else {
            if !self.sh.dl_image_model.load(Ordering::SeqCst) && self.sh.models_found.lock().unwrap().iter().any(|m| !m.1) {
                return Err("Image model download disabled in settings");
            }
        }
        Ok(())
    }

    fn can_generate(&self) -> Result<(), &'static str> {
        if self.generating {
            return Err("Generating...");
        }
        if !self.has_prompt_content() && !self.is_edit_video_mode() {
            return Err("Type a prompt or pick an Edit LoRA");
        }
        let st = self.sh.state.lock().unwrap().clone();
        if !matches!(st, EngState::Ready { .. }) {
            return Err("Engine starting - prompt will be queued");
        }
        self.can_queue()?;
        if self.mode == StudioMode::Video {
            if crate::lora::has_active_h3_lora(&self.video_loras, &self.lora_browser.items)
                && self.effective_video_profile().is_none()
            {
                return Err("MiniMax H3 model required for Viggle Remake / Style Transfer");
            }
            if !self.has_video_models() {
                return Err("Downloading video models - prompt will be queued");
            }
        } else if self.sh.models_found.lock().unwrap().iter().any(|m| !m.1) {
            return Err("Downloading image models - prompt will be queued");
        }
        Ok(())
    }

    fn pick_images(&mut self, ctx: &egui::Context, target: Target, multi: bool) {
        let d = rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp", "gif"]);
        let files = if multi { d.pick_files().unwrap_or_default() } else { d.pick_file().into_iter().collect() };
        for f in files {
            self.load(ctx, target, Src::Path(f));
        }
    }

    // ------------------------------------------------------------ UI pieces
    fn header(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .fill(PANEL)
            .inner_margin(egui::Margin::symmetric(16, 8))
            .stroke(Stroke::new(1.0, LINE))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Image Studio").size(17.0).strong());
                    ui.label(RichText::new(&self.profile.name).color(DIM).size(12.0));



                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let setup_resp = ui.button("Setup");
                        if setup_resp.clicked() {
                            self.show_setup = !self.show_setup;
                        }
                        if self.updater.lock().unwrap().update_available {
                            let badge_pos = setup_resp.rect.right_top() + vec2(-3.0, 3.0);
                            let p = ui.painter();
                            p.circle_filled(badge_pos, 4.0, Color32::from_rgb(35, 165, 90));
                            p.circle_stroke(badge_pos, 4.0, Stroke::new(1.0, Color32::from_rgb(18, 21, 29)));
                        }

                        self.render_update_button(ui, &ui.ctx().clone());

                        if ui.button("Outputs").on_hover_text("Open Image Studio output folder in Windows File Explorer").clicked() {
                            engine::open_output_folder(&self.sh);
                        }
                        let (txt, col) = match &*self.sh.state.lock().unwrap() {
                            EngState::Detecting => ("Detecting engine...".to_string(), AMBER),
                            EngState::AwaitingDownloadConfirmation { .. } => ("Download confirmation required".to_string(), AMBER),
                            EngState::DownloadingEngine { .. } => ("Downloading engine...".to_string(), AMBER),
                            EngState::ExtractingEngine { .. } => ("Extracting engine...".to_string(), AMBER),
                            EngState::DownloadingModels { file_index, total_files, .. } => (format!("Downloading model ({}/{})", file_index, total_files), AMBER),
                            EngState::ModelDownloadPaused { .. } => ("Model download paused".to_string(), AMBER),
                            EngState::RepairingEngine { step } => (step.clone(), AMBER),
                            EngState::Starting { since, .. } => (format!("Starting engine... {}s", since.elapsed().as_secs()), AMBER),
                            EngState::Ready { ours, .. } => (
                                if *ours {
                                    "Engine ready".to_string()
                                } else {
                                    "Engine ready (external)".to_string()
                                },
                                GREEN,
                            ),
                            EngState::ModelDownloadFailed { .. } => ("Model download interrupted".to_string(), AMBER),
                            EngState::Failed(_) => ("Engine setup failed".to_string(), RED),
                        };
                        let pill_resp = egui::Frame::new()
                            .fill(col.linear_multiply(0.14))
                            .corner_radius(20)
                            .inner_margin(egui::Margin::symmetric(10, 3))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    let (r, _) = ui.allocate_exact_size(vec2(8.0, 8.0), Sense::hover());
                                    ui.painter().circle_filled(r.center(), 4.0, col);
                                    ui.label(RichText::new(&txt).color(col).strong().size(12.0));
                                });
                            });
                        let pill_rect = pill_resp.response.rect;
                        let pill_click = ui.interact(pill_rect, egui::Id::new("dl_status_pill"), Sense::click());
                        if pill_click.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        if pill_click.clicked() {
                            self.download_pane.toggle(&self.sh);
                        }
                        pill_click.on_hover_text("Click to toggle download manager");
                        let local_time = build_timestamp_display();
                        ui.label(RichText::new(format!("build {local_time}")).size(11.0).color(DIM))
                            .on_hover_text(format!("Build time (local): {local_time}\nBuild UTC: {}", env!("BUILD_STAMP")));
                    });
                });
            });
    }

    fn render_update_button(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let (up_avail, latest, is_dl, dl_prog, applied, checked) = {
            let st = self.updater.lock().unwrap();
            (
                st.update_available,
                st.latest_version.clone(),
                st.downloading,
                st.download_progress,
                st.update_applied,
                st.checked,
            )
        };
        let model_dl = crate::download_ui::active_download_progress(&self.sh);

        if applied {
            let (rect, resp) = ui.allocate_exact_size(vec2(80.0, 24.0), Sense::click());
            let bg = if resp.hovered() {
                DISCORD_GREEN_HOVER
            } else {
                DISCORD_GREEN
            };
            let p = ui.painter();
            p.rect_filled(rect, 12.0, bg);
            draw_restart_icon(&p, rect.left_center() + vec2(14.0, 0.0), Color32::WHITE);
            p.text(
                rect.center() + vec2(8.0, 0.0),
                egui::Align2::CENTER_CENTER,
                "Restart",
                egui::FontId::proportional(12.0),
                Color32::WHITE,
            );
            let resp = resp.on_hover_text("Update ready! Click to restart Image Studio now");
            if resp.clicked() {
                crate::updater::restart_application();
            }
        } else if is_dl {
            let (rect, resp) = ui.allocate_exact_size(vec2(72.0, 24.0), Sense::click());
            let p = ui.painter();
            p.rect_filled(rect, 12.0, Color32::from_rgb(20, 24, 34));
            p.rect_stroke(rect, 12.0, Stroke::new(1.0, DISCORD_GREEN), StrokeKind::Inside);
            draw_update_arrow_icon(&p, rect.left_center() + vec2(14.0, 0.0), DISCORD_GREEN);
            p.text(
                rect.center() + vec2(8.0, 0.0),
                egui::Align2::CENTER_CENTER,
                format!("{:.0}%", dl_prog * 100.0),
                egui::FontId::proportional(11.5),
                Color32::WHITE,
            );
            let resp = resp.on_hover_text(format!("Downloading app update: {:.0}%\nClick to open Updates & Downloads", dl_prog * 100.0));
            if resp.clicked() {
                self.download_pane.toggle(&self.sh);
            }
        } else if let Some((m_prog, paused, summary)) = model_dl {
            let border_col = if paused { AMBER } else { ACCENT };
            let (rect, resp) = ui.allocate_exact_size(vec2(72.0, 24.0), Sense::click());
            let p = ui.painter();
            p.rect_filled(rect, 12.0, Color32::from_rgb(20, 24, 34));
            p.rect_stroke(rect, 12.0, Stroke::new(1.0, border_col), StrokeKind::Inside);
            draw_update_arrow_icon(&p, rect.left_center() + vec2(14.0, 0.0), border_col);
            p.text(
                rect.center() + vec2(8.0, 0.0),
                egui::Align2::CENTER_CENTER,
                format!("{:.0}%", m_prog * 100.0),
                egui::FontId::proportional(11.5),
                Color32::WHITE,
            );
            let resp = resp.on_hover_text(format!("{summary}\nClick to open Updates & Downloads"));
            if resp.clicked() {
                self.download_pane.toggle(&self.sh);
            }
        } else if up_avail {
            let (rect, resp) = ui.allocate_exact_size(vec2(28.0, 24.0), Sense::click());
            let bg = if resp.hovered() {
                DISCORD_GREEN_HOVER
            } else {
                DISCORD_GREEN
            };
            let p = ui.painter();
            p.rect_filled(rect, 12.0, bg);
            draw_update_arrow_icon(&p, rect.center(), Color32::WHITE);
            let resp = resp.on_hover_text(format!("Update v{latest} available! Click to open Updates & Downloads"));
            if resp.clicked() {
                self.download_pane.toggle(&self.sh);
            }
        } else {
            let (rect, resp) = ui.allocate_exact_size(vec2(26.0, 24.0), Sense::click());
            let bg = if resp.hovered() || self.download_pane.is_expanded { CARD_HI } else { Color32::TRANSPARENT };
            let p = ui.painter();
            if resp.hovered() || self.download_pane.is_expanded {
                p.rect_filled(rect, 4.0, bg);
            }
            let icon_col = if resp.hovered() || self.download_pane.is_expanded { TEXT } else { DIM };
            draw_update_arrow_icon(&p, rect.center(), icon_col);
            let cur_ver = env!("CARGO_PKG_VERSION");
            let tip = if checked {
                format!("Image Studio v{cur_ver} (Up to date)\nClick for Updates & Downloads")
            } else {
                format!("Image Studio v{cur_ver}\nClick for Updates & Downloads")
            };
            let resp = resp.on_hover_text(tip);
            if resp.clicked() {
                self.download_pane.toggle(&self.sh);
                if self.download_pane.is_expanded {
                    crate::updater::check_for_updates(self.updater.clone(), ctx.clone());
                }
            }
        }
    }

    fn render_refs_list(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, max_h: Option<f32>, subtitle: &str, compact: bool) {
        let mut remove = None;
        let refs_resp = ui.vertical(|ui| {
        ui.horizontal(|ui| {
            heading(ui, "REFERENCES");
            ui.label(RichText::new(format!("{}/{}", self.refs.len(), MAX_REFS)).color(DIM).size(11.0));
        });
        ui.label(RichText::new(subtitle).color(DIM).size(if compact { 10.5 } else { 11.0 }));
        ui.add_space(if compact { 2.0 } else { 3.0 });

        let thumb_sz = if compact { 44.0 } else { 50.0 };
        let mut area = egui::ScrollArea::vertical().id_salt("refs_scroll");
        if let Some(h) = max_h {
            area = area.max_height(h);
        }
        area.show(ui, |ui| {
            for (i, r) in self.refs.iter().enumerate() {
                egui::Frame::new()
                    .fill(CARD)
                    .stroke(Stroke::new(1.0, LINE))
                    .corner_radius(6)
                    .inner_margin(egui::Margin::same(if compact { 5 } else { 6 }))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let (rect, _) = ui.allocate_exact_size(vec2(thumb_sz, thumb_sz), Sense::hover());
                            let p = ui.painter_at(rect);
                            p.rect_filled(rect, 4, Color32::from_rgb(15, 17, 24));
                            let s = fit(r.tex.size_vec2(), rect.shrink(2.0).size());
                            p.image(r.tex.id(), Rect::from_center_size(rect.center(), s), FULL_UV, Color32::WHITE);
                            p.rect_stroke(rect, 4, Stroke::new(1.0, LINE), StrokeKind::Inside);

                            ui.vertical(|ui| {
                                ui.add(egui::Label::new(RichText::new(&r.name).size(if compact { 10.5 } else { 11.0 }).strong()).truncate());
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 4.0;
                                    ui.label(RichText::new(format!("{}x{}", r.width, r.height)).size(if compact { 10.0 } else { 10.5 }).color(DIM));
                                    if ui.small_button("X").on_hover_text("Remove reference").clicked() {
                                        remove = Some(i);
                                    }
                                });
                            });
                        });
                    });
                ui.add_space(if compact { 2.0 } else { 3.0 });
            }

            let add_btn = egui::Button::new(RichText::new("+ Add Reference").size(if compact { 11.5 } else { 12.0 }).strong().color(ACCENT_GLOW))
                .fill(CARD_HI)
                .stroke(Stroke::new(1.0, ACCENT.linear_multiply(0.5)))
                .corner_radius(6)
                .min_size(vec2(ui.available_width(), if compact { 26.0 } else { 30.0 }));
            if ui.add(add_btn).clicked() {
                self.pick_images(ctx, Target::Ref, true);
            }
        });
        });
        self.refs_drop_rect = Some(refs_resp.response.rect);

        if let Some(i) = remove {
            self.refs.remove(i);
        }
    }

    // Left Panel: Compact Reference Strip (+ Docked History / Video Trim History)
    fn refs_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if self.mode == StudioMode::Video {
            let vprof = self
                .effective_video_profile()
                .unwrap_or_else(|| self.sh.active_video_profile.lock().unwrap().clone());
            let is_h3 = vprof.is_minimax_h3();
            let is_h3_v2v_edit = is_h3 && self.is_edit_video_mode();
            let models_dir = self.active_video_models_dir();
            let has_r2v = !is_h3 || (!models_dir.as_os_str().is_empty() && vprof.has_r2v_model(&models_dir));
            let avail_h = ui.available_height();
            let refs_max_h = (avail_h * 0.30).clamp(90.0, 200.0);
            let sub = if is_h3_v2v_edit {
                "Optional keyframe/style refs (<Picture 1>..) — works with base H3 (no Ref2VA needed)"
            } else if is_h3 {
                "R2V refs (<Picture 1>..) or End Frame"
            } else {
                "Reference images (switch to MiniMax H3 for R2V)"
            };
            self.render_refs_list(ui, ctx, Some(refs_max_h), sub, true);
            if is_h3 && !is_h3_v2v_edit && !self.refs.is_empty() {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.video_r2v_mode, false, "End Frame");
                    ui.selectable_value(&mut self.video_r2v_mode, true, "R2V Multi-Ref");
                });
            }
            if is_h3 && !is_h3_v2v_edit && !has_r2v {
                if let Some(r2v_file) = vprof.r2v_model_file() {
                    let r2v_gb = r2v_file.size_bytes as f64 / 1_000_000_000.0;
                    let r2v_dl = self.sh.dl_r2v_model.load(Ordering::SeqCst);
                    let is_dl_this = self
                        .sh
                        .model_dl_progress
                        .lock()
                        .unwrap()
                        .as_ref()
                        .is_some_and(|p| p.current_file == r2v_file.filename);
                    ui.add_space(4.0);
                    egui::Frame::new()
                        .fill(CARD)
                        .stroke(Stroke::new(1.0, AMBER.linear_multiply(0.5)))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::same(6))
                        .show(ui, |ui| {
                            ui.label(
                                RichText::new(format!(
                                    "Reference-to-Video (R2V) requires a separate Ref2VA model (~{r2v_gb:.1} GB). End Frame continuation uses the base FL2VA model."
                                ))
                                .size(10.5)
                                .color(DIM),
                            );
                            ui.add_space(3.0);
                            if is_dl_this {
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new("Downloading Ref2VA...").size(10.5).color(ACCENT_GLOW).strong());
                                    if ui.small_button("Cancel").clicked() {
                                        engine::toggle_r2v_download(&self.sh, ctx, false);
                                    }
                                });
                            } else if r2v_dl {
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new("Ref2VA queued (downloads last)").size(10.5).color(ACCENT_GLOW));
                                    if ui.small_button("Cancel").clicked() {
                                        engine::toggle_r2v_download(&self.sh, ctx, false);
                                    }
                                });
                            } else {
                                let btn = egui::Button::new(
                                    RichText::new(format!("Download R2V Model (~{r2v_gb:.1} GB)"))
                                        .size(11.0)
                                        .strong()
                                        .color(TEXT),
                                )
                                .fill(CARD_HI)
                                .stroke(Stroke::new(1.0, ACCENT.linear_multiply(0.6)))
                                .min_size(vec2(ui.available_width(), 24.0));
                                if ui.add(btn).clicked() {
                                    engine::toggle_r2v_download(&self.sh, ctx, true);
                                }
                            }
                        });
                }
            }
            ui.add_space(4.0);
            self.render_left_lora_history_split(ui, ctx, true);
            return;
        }
        if !self.lora_browser.remote_fetched && !self.lora_browser.remote_fetching {
            crate::lora::fetch_comfy_lora_browser_async(&mut self.lora_browser, ctx);
        }
        let avail_h = ui.available_height();
        if self.history_dock == HistoryDock::Left {
            let refs_max_h = (avail_h * 0.28).clamp(90.0, 180.0);
            self.render_refs_list(ui, ctx, Some(refs_max_h), "Drag extra images for reference", true);

            ui.add_space(4.0);
            self.render_left_lora_history_split(ui, ctx, false);
        } else {
            let refs_max_h = (avail_h * 0.36).clamp(100.0, 240.0);
            self.render_refs_list(ui, ctx, Some(refs_max_h), "Drag extra images here for reference", true);

            let lora_max_h = (ui.available_height() - 48.0).max(160.0);
            ui.add_space(4.0);
            self.render_image_lora_card(ui, ctx, Some(lora_max_h));
        }
    }

    /// Renders the squishy middle LoRA (and optional Video Trim History) pane, a draggable
    /// horizontal resizing bar, and the bottom History pane locked at `self.left_history_h`.
    fn render_left_lora_history_split(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, is_video: bool) {
        let rem_h = ui.available_height();
        let usable_h = (rem_h - LEFT_SPLIT_BAR_H).max(0.0);
        let min_hist_h = LEFT_HISTORY_MIN_H.min((usable_h - LEFT_LORA_MIN_H).max(0.0));
        let max_hist_h = (usable_h - LEFT_LORA_MIN_H).max(min_hist_h);
        let hist_box_h = self.left_history_h.clamp(min_hist_h, max_hist_h);
        let middle_box_h = (usable_h - hist_box_h).max(0.0);

        ui.allocate_ui_with_layout(
            vec2(ui.available_width(), middle_box_h),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                ui.set_max_height(middle_box_h);
                egui::ScrollArea::vertical()
                    .id_salt(if is_video { "left_video_mid_scroll" } else { "left_image_mid_scroll" })
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if is_video {
                            let has_trim = self.video_player.is_some();
                            let trim_reserve = if has_trim { 110.0 } else { 0.0 };
                            let lora_overhead = 52.0 + self.video_loras.len() as f32 * 48.0 + trim_reserve;
                            let lora_max_h = (middle_box_h - lora_overhead).max(64.0);
                            self.render_video_lora_card(ui, ctx, Some(lora_max_h));
                            if has_trim {
                                ui.add_space(4.0);
                                ui.separator();
                                ui.add_space(4.0);
                                self.render_trim_history_ui(ui, ctx);
                            }
                        } else {
                            let lora_overhead = 48.0 + self.image_loras.len() as f32 * 48.0;
                            let lora_max_h = (middle_box_h - lora_overhead).max(64.0);
                            self.render_image_lora_card(ui, ctx, Some(lora_max_h));
                        }
                    });
            },
        );

        let (bar_rect, bar_resp) = ui.allocate_exact_size(
            vec2(ui.available_width(), LEFT_SPLIT_BAR_H),
            Sense::drag(),
        );
        if bar_resp.hovered() || bar_resp.dragged() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
        }
        if bar_resp.dragged() {
            let dy = bar_resp.drag_delta().y;
            let drag_max = max_hist_h.max(LEFT_HISTORY_MIN_H);
            self.left_history_h = (hist_box_h - dy).clamp(LEFT_HISTORY_MIN_H, drag_max);
        }
        if ui.is_rect_visible(bar_rect) {
            let p = ui.painter();
            let active = bar_resp.hovered() || bar_resp.dragged();
            let line_col = if active { ACCENT_GLOW } else { LINE };
            let grip_col = if active { ACCENT_GLOW } else { DIM };
            let y = bar_rect.center().y;
            p.line_segment(
                [pos2(bar_rect.left(), y), pos2(bar_rect.right(), y)],
                Stroke::new(1.0, line_col),
            );
            let pill = Rect::from_center_size(bar_rect.center(), vec2(34.0, 3.0));
            p.rect_filled(pill, 1.5, grip_col);
        }
        bar_resp.on_hover_text("Drag to resize LoRA and History panes");

        ui.allocate_ui_with_layout(
            vec2(ui.available_width(), hist_box_h),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                ui.set_max_height(hist_box_h);
                self.render_history_ui(ui);
            },
        );
    }

    fn render_orientation_button(
        ui: &mut egui::Ui,
        selected: bool,
        is_landscape: bool,
        text: &str,
        width: f32,
    ) -> egui::Response {
        let desired_size = vec2(width, 28.0);
        let (rect, resp) = ui.allocate_exact_size(desired_size, egui::Sense::click());
        if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if ui.is_rect_visible(rect) {
            let p = ui.painter();
            let bg_col = if selected {
                ACCENT
            } else if resp.hovered() {
                CARD_HI
            } else {
                CARD
            };
            let stroke_col = if selected {
                ACCENT_GLOW
            } else {
                LINE
            };
            p.rect_filled(rect, 6.0, bg_col);
            p.rect_stroke(rect, 6.0, Stroke::new(1.0, stroke_col), StrokeKind::Inside);

            // Programmatic vector rectangle icon (16:9 for landscape, 9:16 for portrait)
            let icon_center = pos2(rect.left() + 20.0, rect.center().y);
            let icon_size = if is_landscape { vec2(18.0, 11.0) } else { vec2(11.0, 18.0) };
            let icon_rect = Rect::from_center_size(icon_center, icon_size);
            let icon_stroke = if selected { Color32::WHITE } else { TEXT };
            let icon_fill = if selected { Color32::from_white_alpha(50) } else { Color32::from_white_alpha(20) };
            p.rect_filled(icon_rect, 2.0, icon_fill);
            p.rect_stroke(icon_rect, 2.0, Stroke::new(1.2, icon_stroke), StrokeKind::Inside);

            // Text
            let text_col = if selected { Color32::WHITE } else { TEXT };
            let text_pos = pos2(rect.left() + 38.0, rect.center().y);
            let font_id = egui::FontId::proportional(12.0);
            let galley = p.layout_no_wrap(text.to_string(), font_id, text_col);
            p.galley(pos2(text_pos.x, text_pos.y - galley.size().y * 0.5), galley, text_col);
        }
        resp
    }

    fn render_image_lora_card(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, max_scroll_h: Option<f32>) {
        card(ui, |ui| {
            ui.horizontal(|ui| {
                heading(ui, "IMAGE LORA");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Refresh").on_hover_text("Rescan models/loras and ComfyUI LoRA browser thumbnails").clicked() {
                        self.lora_browser.items = crate::lora::scan_local_loras(&self.sh);
                        self.lora_browser.thumbs.clear();
                        self.lora_browser.loading_thumbs.clear();
                        self.lora_browser.remote_fetched = false;
                        crate::lora::fetch_comfy_lora_browser_async(&mut self.lora_browser, ctx);
                    }
                });
            });
            crate::lora::render_lora_carousel_filtered(
                ui,
                ctx,
                &self.sh,
                &mut self.lora_browser,
                &mut self.image_loras,
                &mut self.image_prompt_tags,
                crate::lora::LoraModelFamily::QwenImage21,
                None,
                None,
                max_scroll_h,
            );
        });
    }

    fn render_video_lora_card(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, max_scroll_h: Option<f32>) {
        let vprof = self
            .effective_video_profile()
            .unwrap_or_else(|| self.sh.active_video_profile.lock().unwrap().clone());
        let lora_family = crate::lora::LoraModelFamily::from_video_profile(&vprof);

        card(ui, |ui| {
            ui.horizontal(|ui| {
                let is_edit = self.video_work_mode == VideoWorkMode::Edit;
                heading(ui, if is_edit { "EDIT EFFECTS" } else { "GENERATE LORAS" });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Refresh").on_hover_text("Rescan models/loras and ComfyUI LoRA browser thumbnails").clicked() {
                        self.lora_browser.items = crate::lora::scan_local_loras(&self.sh);
                        self.lora_browser.thumbs.clear();
                        self.lora_browser.loading_thumbs.clear();
                        self.lora_browser.remote_fetched = false;
                        crate::lora::fetch_comfy_lora_browser_async(&mut self.lora_browser, ctx);
                    }
                    if ui
                        .selectable_label(!is_edit, "Generate")
                        .on_hover_text("Generate a new video from text or current frame (prefers MiniMax H3)")
                        .clicked()
                        && is_edit
                    {
                        self.set_video_work_mode(VideoWorkMode::Generate);
                    }
                    if ui
                        .selectable_label(is_edit, "Edit")
                        .on_hover_text("Edit loaded video with H3 (Viggle Remake / Style Transfer) or LTX IC-LoRAs")
                        .clicked()
                        && !is_edit
                    {
                        self.set_video_work_mode(VideoWorkMode::Edit);
                    }
                });
            });
            let (carousel_family, ic_filter) = if self.video_work_mode == VideoWorkMode::Edit {
                (crate::lora::LoraModelFamily::VideoEditAll, Some(true))
            } else {
                (lora_family, Some(false))
            };
            let allow_multi = self.sh.cfg.lock().unwrap().allow_multi_video_loras;
            let lora_vram = self.active_video_lora_vram_gb();
            let current_est_vram_gb = estimate_video_vram_gb_with_lora(
                self.video_width,
                self.video_height,
                self.video_duration,
                self.video_fps,
                lora_vram,
            );
            let multi_policy = crate::lora::MultiLoraPolicy {
                allow_multi,
                current_est_vram_gb,
                gpu_vram_gb: engine::detect_gpu_vram_gb(),
            };
            let edit_loras_disabled = self.video_work_mode == VideoWorkMode::Edit && self.video_player.is_none();
            let carousel_resp = ui.add_enabled_ui(!edit_loras_disabled, |ui| {
                crate::lora::render_lora_carousel_with_policy(
                    ui,
                    ctx,
                    &self.sh,
                    &mut self.lora_browser,
                    &mut self.video_loras,
                    &mut self.video_prompt_tags,
                    carousel_family,
                    Some((vprof.default_steps, &mut self.video_steps)),
                    ic_filter,
                    max_scroll_h,
                    Some(multi_policy),
                );
            });
            if edit_loras_disabled {
                carousel_resp
                    .response
                    .on_disabled_hover_text("Video editing only — load or drop a video clip to use Edit LoRAs");
            }

            if self.video_player.is_none()
                && crate::lora::remove_ic_loras(
                    &mut self.video_loras,
                    &mut self.video_prompt_tags,
                    &self.lora_browser.items,
                )
            {
                self.edit_lora_needs_video_until =
                    Some(Instant::now() + std::time::Duration::from_secs_f32(2.0));
                ctx.request_repaint();
            }

            let now = Instant::now();
            let flash_rem = self
                .edit_lora_needs_video_until
                .and_then(|until| until.checked_duration_since(now));
            if flash_rem.is_none() {
                self.edit_lora_needs_video_until = None;
            } else {
                ctx.request_repaint();
            }

            if self.video_work_mode == VideoWorkMode::Edit
                && crate::lora::has_active_h3_lora(&self.video_loras, &self.lora_browser.items)
            {
                let vdir = self.active_video_models_dir();
                let h3_ready = !vdir.as_os_str().is_empty()
                    && engine::Profile::find_best_downloaded_h3_profile(&vdir).is_some();
                if !h3_ready {
                    ui.add_space(6.0);
                    let mut download_rec_h3 = false;
                    egui::Frame::NONE
                        .fill(Color32::from_rgb(56, 36, 18))
                        .stroke(Stroke::new(1.0, AMBER))
                        .corner_radius(5.0)
                        .inner_margin(egui::Margin::symmetric(8, 6))
                        .show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    RichText::new("MiniMax H3 model required for Viggle Remake / Style Transfer.")
                                        .size(10.5)
                                        .strong()
                                        .color(AMBER),
                                );
                                if ui.small_button("Download Recommended H3").clicked() {
                                    download_rec_h3 = true;
                                }
                            });
                        });
                    if download_rec_h3 {
                        let rec_id = crate::model_picker::recommended_video_profile_id(
                            crate::model_picker::VideoFamily::MiniMaxH3,
                            engine::detect_gpu_vram_gb(),
                        );
                        if let Some(rec_prof) = engine::Profile::all_video_profiles()
                            .into_iter()
                            .find(|p| p.id == rec_id)
                        {
                            {
                                let mut c = self.sh.cfg.lock().unwrap();
                                c.dl_video_model = true;
                                c.save();
                            }
                            self.sh.dl_video_model.store(true, Ordering::SeqCst);
                            self.apply_video_profile(&rec_prof, ctx);
                            self.download_pane.is_expanded = true;
                            self.download_pane.user_opened = true;
                        }
                    }
                }
            }

            if self.video_work_mode == VideoWorkMode::Edit || self.is_edit_video_mode() || flash_rem.is_some() {
                ui.add_space(6.0);
                let has_vid = self.video_player.is_some();
                let is_flashing = !has_vid
                    && flash_rem.is_some_and(|d| ((d.as_millis() / 220) % 2) == 0);
                let border_col = if has_vid {
                    DISCORD_GREEN
                } else if is_flashing {
                    Color32::from_rgb(248, 113, 113)
                } else {
                    AMBER
                };
                let fill_col = if is_flashing {
                    Color32::from_rgb(68, 28, 34)
                } else {
                    Color32::from_rgb(24, 29, 42)
                };
                let mut pick_vid = false;
                egui::Frame::NONE
                    .fill(fill_col)
                    .stroke(Stroke::new(if is_flashing { 1.8 } else { 1.0 }, border_col.linear_multiply(0.85)))
                    .corner_radius(5.0)
                    .inner_margin(egui::Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                RichText::new("VIDEO EDIT MODE (IC-LoRA)")
                                    .size(10.0)
                                    .strong()
                                    .color(border_col),
                            );
                            if let Some(ref vp) = self.video_player {
                                ui.label(
                                    RichText::new(format!("Source: {}", vp.filename))
                                        .size(10.5)
                                        .color(TEXT),
                                );
                            } else {
                                let msg = if flash_rem.is_some() {
                                    "Choose or drop a video first!"
                                } else {
                                    "Load a video to edit"
                                };
                                ui.label(
                                    RichText::new(msg)
                                        .size(10.5)
                                        .strong()
                                        .color(border_col),
                                );
                                if ui.small_button("Load Video...").clicked() {
                                    pick_vid = true;
                                }
                            }
                        });
                    });
                if pick_vid {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Video", &["mp4", "mov", "mkv", "webm", "avi"])
                        .pick_file()
                    {
                        self.load_video_file(ctx, path);
                    }
                }
            }
        });
    }

    fn render_video_settings(&mut self, ui: &mut egui::Ui) {
        let active_vprof = self.sh.active_video_profile.lock().unwrap().clone();
        let vprof = self.effective_video_profile().unwrap_or_else(|| active_vprof.clone());
        let is_fallback = vprof.id != active_vprof.id;
        let is_h3 = vprof.is_minimax_h3();
        let is_ltx25 = vprof.is_ltx25();
        let _lora_family = crate::lora::LoraModelFamily::from_video_profile(&vprof);

        if !self.lora_browser.remote_fetched && !self.lora_browser.remote_fetching {
            crate::lora::fetch_comfy_lora_browser_async(&mut self.lora_browser, ui.ctx());
        }

        egui::ScrollArea::vertical().id_salt("video_settings_scroll").show(ui, |ui| {
            card(ui, |ui| {
                ui.horizontal(|ui| {
                    heading(ui, "VIDEO MODEL");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("Change in Setup...").clicked() {
                            self.show_setup = true;
                        }
                    });
                });
                ui.label(RichText::new(&vprof.name).size(11.5).strong().color(TEXT));
                if is_fallback {
                    let vdir = self.active_video_models_dir();
                    let msg = if !vdir.as_os_str().is_empty() && active_vprof.missing_in_dir(&vdir).is_empty() {
                        format!("Using {} for video edit / LTX LoRA (active: {}).", vprof.name, active_vprof.name)
                    } else {
                        format!("Using {} while {} downloads in background.", vprof.name, active_vprof.name)
                    };
                    ui.label(
                        RichText::new(msg)
                            .size(10.5)
                            .color(AMBER),
                    );
                }
                if let Some(gpu_vram) = engine::detect_gpu_vram_gb() {
                    if vprof.recommended_vram_gb > gpu_vram {
                        ui.label(
                            RichText::new(format!(
                                "[!] Profile recommends {}GB VRAM (detected {}GB) - may page RAM or OOM",
                                vprof.recommended_vram_gb, gpu_vram
                            ))
                            .size(10.5)
                            .color(AMBER),
                        );
                    }
                }
            });

            ui.add_space(6.0);

            card(ui, |ui| {
                heading(ui, "WORKFLOW");
                if is_h3 {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.video_r2v_mode, false, "T2V / I2V / Continuation");
                        ui.selectable_value(&mut self.video_r2v_mode, true, "Reference (R2V)");
                    });
                    ui.add_space(2.0);
                    if self.video_r2v_mode {
                        let models_dir = self.active_video_models_dir();
                        let has_r2v = !models_dir.as_os_str().is_empty() && vprof.has_r2v_model(&models_dir);
                        if !has_r2v {
                            ui.label(RichText::new("Ref2VA model required for R2V. Download it in the top-left References panel.").size(10.5).color(AMBER));
                        } else if self.refs.is_empty() {
                            ui.label(RichText::new("Add 1-9 reference images in the left References panel, then cite them in your prompt as <Picture 1>, <Picture 2>, etc.").size(10.5).color(AMBER));
                        } else {
                            ui.label(RichText::new(format!("Reference-to-Video Active ({} ref image(s) linked as <Picture 1>..)", self.refs.len())).size(11.0).strong().color(ACCENT_GLOW));
                        }
                    } else if self.active_node().is_some() && !self.refs.is_empty() {
                        ui.label(RichText::new("First + Last Frame Continuation").strong().color(ACCENT_GLOW));
                        ui.label(RichText::new("Canvas frame is Start Frame; Reference #1 is End Frame.").size(10.5).color(DIM));
                    } else if self.active_node().is_some() {
                        ui.label(RichText::new("Image-to-Video").strong().color(ACCENT_GLOW));
                        ui.label(RichText::new("Canvas image/scrubbed frame is Start Frame. Add a Reference to set an End Frame.").size(10.5).color(DIM));
                        if ui.button("Switch to Text-to-Video (Deselect Image)").clicked() {
                            self.active_node_id = None;
                        }
                    } else {
                        ui.label(RichText::new("Text-to-Video").strong().color(GREEN));
                        ui.label(RichText::new("Generates video + synced stereo audio from your text prompt.").size(10.5).color(DIM));
                        if ui.button("Load Image for Image-to-Video...").clicked() {
                            self.pick_images(ui.ctx(), Target::Base, false);
                        }
                    }
                } else if self.active_node().is_some() {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Image-to-Video").strong().color(ACCENT_GLOW));
                    });
                    ui.label(RichText::new("Active image/frame on canvas will be used as the starting frame.").size(11.0).color(DIM));
                    ui.add_space(4.0);
                    if ui.button("Switch to Text-to-Video (Deselect Image)").clicked() {
                        self.active_node_id = None;
                    }
                } else {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Text-to-Video").strong().color(GREEN));
                    });
                    ui.label(RichText::new("Generates video entirely from your text prompt.").size(11.0).color(DIM));
                    ui.add_space(4.0);
                    if ui.button("Load Image for Image-to-Video...").clicked() {
                        self.pick_images(ui.ctx(), Target::Base, false);
                    }
                }
            });

            ui.add_space(6.0);

            card(ui, |ui| {
                heading(ui, "ORIENTATION & RESOLUTION");

                // 1. Orientation buttons with programmatic vector rectangles (16:9 Landscape / 9:16 Portrait)
                ui.horizontal(|ui| {
                    let is_land = self.video_orientation == VideoOrientation::Landscape;
                    let is_port = self.video_orientation == VideoOrientation::Portrait;
                    let half_w = (ui.available_width() - 4.0) * 0.5;

                    let land_btn = Self::render_orientation_button(ui, is_land, true, "Landscape", half_w);
                    if land_btn.on_hover_text("16:9 Landscape Orientation (Desktop, YouTube, TV)").clicked() && !is_land {
                        self.video_orientation = VideoOrientation::Landscape;
                        if self.video_width < self.video_height {
                            std::mem::swap(&mut self.video_width, &mut self.video_height);
                            std::mem::swap(&mut self.video_base_w, &mut self.video_base_h);
                        }
                    }

                    let port_btn = Self::render_orientation_button(ui, is_port, false, "Portrait", ui.available_width());
                    if port_btn.on_hover_text("9:16 Vertical Orientation (Reels, Shorts, TikTok)").clicked() && !is_port {
                        self.video_orientation = VideoOrientation::Portrait;
                        if self.video_width > self.video_height {
                            std::mem::swap(&mut self.video_width, &mut self.video_height);
                            std::mem::swap(&mut self.video_base_w, &mut self.video_base_h);
                        }
                    }
                });

                ui.add_space(6.0);

                // 2. Clean Presets: 360p, 480p, 720p, 1080p, 1440p, 4K
                let is_portrait = self.video_orientation == VideoOrientation::Portrait;
                let presets: [(&str, u32, u32, &str); 6] = [
                    ("360p", 640, 384, "640x384 - Fast preview, minimal VRAM"),
                    ("480p", 832, 448, "832x448 - Smooth preview, standard definition"),
                    ("720p", 1280, 704, "1280x704 - Standard HD (Recommended standard)"),
                    ("1080p", 1920, 1088, "1920x1088 - Full HD (Requires ~16GB+ VRAM)"),
                    ("1440p", 2560, 1408, "2560x1408 - 2K Quad HD (Requires ~22GB+ VRAM)"),
                    ("4K", 3840, 2176, "3840x2176 - Ultra HD (Requires 32GB+ VRAM or heavy paging)"),
                ];

                ui.horizontal_wrapped(|ui| {
                    for (label, w_land, h_land, desc) in presets {
                        let (target_w, target_h) = if is_portrait { (h_land, w_land) } else { (w_land, h_land) };
                        let is_sel = self.video_base_w == target_w && self.video_base_h == target_h;
                        let btn = egui::Button::new(RichText::new(label).strong().color(if is_sel { Color32::WHITE } else { TEXT }))
                            .fill(if is_sel { ACCENT } else { CARD_HI })
                            .stroke(Stroke::new(1.0, if is_sel { ACCENT_GLOW } else { LINE }))
                            .corner_radius(5);
                        if ui.add(btn).on_hover_text(format!("{}x{} px - {}", target_w, target_h, desc)).clicked() {
                            self.video_base_w = target_w;
                            self.video_base_h = target_h;
                            self.video_scale = 1.0;
                            self.video_width = target_w;
                            self.video_height = target_h;
                        }
                    }
                });

                ui.add_space(6.0);

                // 3. Resolution scale slider (keeps aspect ratio)
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Resolution scale:").size(11.5).color(DIM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!("{:.2}x ({:.0}%)", self.video_scale, self.video_scale * 100.0)).size(11.5).strong());
                    });
                });
                let (v_min, v_max) = absolute_scale_range(self.video_base_w, self.video_base_h, ABS_MIN_RES, ABS_MAX_RES);
                let slider = egui::Slider::new(&mut self.video_scale, v_min..=v_max)
                    .show_value(false);
                let s_resp = ui.add(slider);
                if s_resp.changed() {
                    let (vw, vh) = scale_dimensions_stepped(
                        self.video_base_w,
                        self.video_base_h,
                        self.video_scale,
                        32,
                        ABS_MIN_RES,
                        ABS_MAX_RES,
                    );
                    self.video_width = vw;
                    self.video_height = vh;
                }

                ui.add_space(4.0);

                // 4. Custom dimensions & live display
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Custom:").size(11.0).color(DIM));
                    let w_resp = ui.add(egui::DragValue::new(&mut self.video_width).speed(32).range(ABS_MIN_RES..=ABS_MAX_RES).prefix("W: "));
                    let h_resp = ui.add(egui::DragValue::new(&mut self.video_height).speed(32).range(ABS_MIN_RES..=ABS_MAX_RES).prefix("H: "));
                    if w_resp.changed() || h_resp.changed() {
                        self.video_width = ((self.video_width as f32 / 32.0).round() as u32 * 32).clamp(ABS_MIN_RES, ABS_MAX_RES);
                        self.video_height = ((self.video_height as f32 / 32.0).round() as u32 * 32).clamp(ABS_MIN_RES, ABS_MAX_RES);
                        self.video_base_w = self.video_width;
                        self.video_base_h = self.video_height;
                        self.video_scale = 1.0;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Output: {} x {} px", self.video_width, self.video_height)).size(11.0).color(TEXT).strong());
                    let ratio = self.video_width as f32 / self.video_height.max(1) as f32;
                    ui.label(RichText::new(format!("({:.2}:1)", ratio)).size(10.5).color(DIM));
                });

                // 5. Dynamic VRAM estimate and excessive swap warning card
                let vram_detected = engine::detect_gpu_vram_gb();
                let lora_vram = self.active_video_lora_vram_gb();
                let est_vram = estimate_video_vram_gb_with_lora(self.video_width, self.video_height, self.video_duration, self.video_fps, lora_vram);
                let lora_note = if lora_vram > 0.0 {
                    format!(", +{lora_vram:.1} GB LoRA")
                } else {
                    String::new()
                };

                if let Some(gpu_vram) = vram_detected {
                    if est_vram > gpu_vram as f32 {
                        ui.add_space(4.0);
                        egui::Frame::new()
                            .fill(Color32::from_rgb(42, 22, 18))
                            .stroke(Stroke::new(1.0, Color32::from_rgb(220, 90, 40)))
                            .corner_radius(6)
                            .inner_margin(egui::Margin::same(8))
                            .show(ui, |ui| {
                                ui.label(RichText::new("[!] High VRAM Load Warning").strong().color(AMBER));
                                ui.label(
                                    RichText::new(format!("Est: {:.1} GB / {} GB VRAM{}", est_vram, gpu_vram, lora_note))
                                        .size(10.5)
                                        .color(AMBER),
                                );
                                ui.add_space(2.0);
                                ui.label(
                                    RichText::new("This resolution & duration combination exceeds your GPU's dedicated VRAM. Windows will use shared system RAM paging, causing generation steps to slow down by 10x-100x or appear stalled.\n\nTip: Reduce duration, lower the Resolution Scale slider, or choose 480p / 360p for fast generation.")
                                        .size(10.5)
                                        .color(Color32::from_rgb(235, 205, 185)),
                                );
                            });
                    } else if est_vram > (gpu_vram as f32 - 1.5) {
                        ui.add_space(2.0);
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new("[i] Near VRAM capacity").size(10.5).color(AMBER));
                            ui.label(RichText::new(format!("(Est. {:.1} GB / {} GB{})", est_vram, gpu_vram, lora_note)).size(10.5).color(DIM));
                        });
                    } else {
                        ui.add_space(2.0);
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new("[OK] Within VRAM budget").size(10.5).color(GREEN));
                            ui.label(RichText::new(format!("(Est. {:.1} GB / {} GB{})", est_vram, gpu_vram, lora_note)).size(10.5).color(DIM));
                        });
                    }
                } else if est_vram >= 14.0 {
                    ui.add_space(4.0);
                    ui.label(RichText::new(format!("[!] Heavy load (Est. {:.1} GB VRAM required{}). May cause RAM swapping on GPUs with <16GB.", est_vram, lora_note)).size(10.5).color(AMBER));
                }
            });

            ui.add_space(6.0);

            card(ui, |ui| {
                let edit_mode = self.is_edit_video_mode();
                heading(ui, if edit_mode { "CLIP TRIM & FRAMERATE" } else { "DURATION & FRAMERATE" });
                if !edit_mode {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Duration:").size(11.5).color(TEXT));
                        ui.add(egui::Slider::new(&mut self.video_duration, 1..=15).suffix("s"));
                    });
                }

                if is_h3 {
                    self.video_fps = 24;
                    let raw_frames = (self.video_duration * 24).max(5);
                    let rem = (5 - (raw_frames % 17) as i32).rem_euclid(17) as u32;
                    let snapped = raw_frames + rem;
                    ui.label(RichText::new(format!("Output grid: {} frames ({}s @ 24 fps)", snapped, self.video_duration)).size(11.0).color(DIM));
                } else {
                    if !edit_mode {
                        ui.add_space(4.0);
                    }
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Framerate:").size(11.5).color(TEXT));
                        for fps_opt in [24, 25, 30] {
                            let sel = self.video_fps == fps_opt;
                            if ui.selectable_label(sel, format!("{fps_opt} fps")).clicked() {
                                self.video_fps = fps_opt;
                            }
                        }
                    });

                    ui.add_space(2.0);
                    if edit_mode {
                        if let Some(ref vp) = self.video_player {
                            let ts = vp.trim_start();
                            let te = vp.trim_end();
                            let tdur = (te - ts).max(0.2);
                            let raw_frames = (tdur * self.video_fps as f64).round() as u32;
                            let snapped = ((raw_frames.saturating_sub(1)) / 8).max(1) * 8 + 1;
                            ui.label(
                                RichText::new(format!(
                                    "Editing trim: {:.2}s ({:.2}s – {:.2}s) · {} frames @ {} fps",
                                    tdur, ts, te, snapped, self.video_fps
                                ))
                                .size(11.0)
                                .color(DIM),
                            );
                        } else {
                            ui.label(
                                RichText::new("Uses loaded clip length (set trim markers on timeline)")
                                    .size(11.0)
                                    .color(DIM),
                            );
                        }
                    } else {
                        let total_frames = self.video_duration * self.video_fps;
                        ui.label(RichText::new(format!("Total frames: {} frames ({}s @ {} fps)", total_frames, self.video_duration, self.video_fps)).size(11.0).color(DIM));
                    }
                }
            });

            if !is_h3 && !is_ltx25 {
                ui.add_space(6.0);

                card(ui, |ui| {
                    heading(ui, "AI PROMPT ENHANCER");
                    let chk = egui::Checkbox::new(&mut self.video_prompt_enhance, RichText::new("Gemma 3 12B Enhancer").strong());
                    ui.add(chk);
                    ui.label(
                        RichText::new("Expands short prompts into detailed cinematic visual scripts using local Gemma 3 12B before rendering.")
                            .size(10.5)
                            .color(DIM),
                    );
                });
            }

            ui.add_space(6.0);

            card(ui, |ui| {
                heading(ui, "SAMPLING & SEED");
                if is_h3 {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Steps:").size(11.0).color(TEXT));
                        ui.add(egui::Slider::new(&mut self.video_steps, 4..=40));
                    });
                    let (step_note, step_color) = match self.video_steps {
                        ..=7 => ("Quality starts to be compromised at 7 steps or below.", AMBER),
                        8..=13 => ("Faster", DIM),
                        14..=20 => ("Quality", DIM),
                        _ => ("Diminishing returns kick in at 21 steps or above.", AMBER),
                    };
                    ui.label(RichText::new(step_note).size(10.5).color(step_color));
                    ui.add_space(2.0);
                    egui::CollapsingHeader::new(RichText::new("Advanced").size(11.0).color(DIM))
                        .id_salt("h3_sampling_advanced")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Sampler:").size(11.0).color(TEXT));
                                egui::ComboBox::from_id_salt("video_sampler_combo")
                                    .selected_text(&self.video_sampler)
                                    .show_ui(ui, |ui| {
                                        for s in ["er_sde", "seeds_2", "euler", "res_multistep", "euler_ancestral", "dpmpp_2m_sde"] {
                                            ui.selectable_value(&mut self.video_sampler, s.to_string(), s);
                                        }
                                    });
                            });
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Scheduler:").size(11.0).color(TEXT));
                                egui::ComboBox::from_id_salt("video_scheduler_combo")
                                    .selected_text(&self.video_scheduler)
                                    .show_ui(ui, |ui| {
                                        for sc in ["sgm_uniform", "ddim_uniform", "simple", "beta", "normal"] {
                                            ui.selectable_value(&mut self.video_scheduler, sc.to_string(), sc);
                                        }
                                    });
                            });
                            if ui.small_button("Return to default").clicked() {
                                let eff = self.effective_video_profile();
                                self.video_sampler = eff
                                    .as_ref()
                                    .map(|p| p.sampler.clone())
                                    .filter(|s| !s.is_empty())
                                    .unwrap_or_else(|| "er_sde".to_string());
                                self.video_scheduler = eff
                                    .as_ref()
                                    .map(|p| p.scheduler.clone())
                                    .filter(|s| !s.is_empty())
                                    .unwrap_or_else(|| "sgm_uniform".to_string());
                            }
                        });
                } else {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Sampler:").size(11.0).color(TEXT));
                        egui::ComboBox::from_id_salt("video_sampler_combo")
                            .selected_text(&self.video_sampler)
                            .show_ui(ui, |ui| {
                                for s in ["er_sde", "euler", "res_multistep", "euler_ancestral", "dpmpp_2m_sde"] {
                                    ui.selectable_value(&mut self.video_sampler, s.to_string(), s);
                                }
                            });
                    });
                    ui.label(RichText::new("Distilled 2-stage schedule (8 + 3 steps @ CFG 1.0)").size(10.5).color(DIM));
                }
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.random_seed, "Random");
                    if !self.random_seed {
                        ui.add(egui::DragValue::new(&mut self.seed).speed(1));
                    }
                    if ui.small_button("Rnd").on_hover_text("Generate random seed").clicked() {
                        self.seed = rand_u64();
                        self.random_seed = false;
                    }
                });
            });

            if !is_h3 {
                ui.add_space(6.0);

                card(ui, |ui| {
                    heading(ui, "NEGATIVE PROMPT");
                    ui.add(
                        egui::TextEdit::multiline(&mut self.video_negative)
                            .desired_rows(3)
                            .desired_width(ui.available_width())
                            .hint_text("e.g. video game, cartoon, CGI, 3D render, ugly, blurry, deformed"),
                    );
                });
            }

            ui.add_space(6.0);

            // Compression & Export Settings (ported from ztrim)
            card(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.compress_enabled, "");
                    ui.label(
                        RichText::new("Compress Video (Trim & Export)")
                            .size(12.5)
                            .strong()
                            .color(if self.compress_enabled { TEXT } else { DIM }),
                    );
                });
            });

            if self.compress_enabled {
                ui.add_space(6.0);
                card(ui, |ui| {
                    heading(ui, "COMPRESSION PRESET");
                    let mut new_preset_idx = None;
                    egui::ComboBox::from_id_salt("video_compress_preset_dropdown")
                        .width(ui.available_width() - 8.0)
                        .selected_text(
                            self.video_presets
                                .get(self.selected_video_preset_idx)
                                .map(|p| p.name.as_str())
                                .unwrap_or("Custom"),
                        )
                        .show_ui(ui, |ui| {
                            for (i, p) in self.video_presets.iter().enumerate() {
                                if ui.selectable_value(&mut self.selected_video_preset_idx, i, &p.name).clicked() {
                                    new_preset_idx = Some(i);
                                }
                            }
                        });
                    if let Some(idx) = new_preset_idx {
                        self.sync_from_video_preset_idx(idx);
                    }

                    if self.is_auto_video_preset() {
                        ui.add_space(6.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Auto Range:").size(11.5).color(TEXT));
                            let mut level_idx = self.auto_quality_level.to_index();
                            let slider = egui::Slider::new(&mut level_idx, 0..=2)
                                .custom_formatter(|val, _| match val.round() as usize {
                                    0 => "Low".to_string(),
                                    1 => "Med".to_string(),
                                    _ => "High".to_string(),
                                })
                                .show_value(true);
                            if ui.add(slider).changed() {
                                self.auto_quality_level = AutoQualityLevel::from_index(level_idx);
                                self.apply_auto_bitrate();
                            }
                        });
                    }

                    ui.add_space(8.0);
                    heading(ui, "VARIABLE BITRATE (VBR)");

                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Average Bitrate:").size(11.5).color(TEXT));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new("Mb/s").size(11.0).color(DIM));
                            ui.add(egui::DragValue::new(&mut self.avg_bitrate).range(1.0..=120.0).speed(0.2));
                        });
                    });
                    ui.add(egui::Slider::new(&mut self.avg_bitrate, 1.0..=60.0).show_value(false));

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Maximum Bitrate:").size(11.5).color(TEXT));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new("Mb/s").size(11.0).color(DIM));
                            ui.add(egui::DragValue::new(&mut self.max_bitrate).range(1.0..=200.0).speed(0.5));
                        });
                    });
                    ui.add(egui::Slider::new(&mut self.max_bitrate, 2.0..=100.0).show_value(false));
                    if self.max_bitrate < self.avg_bitrate {
                        self.max_bitrate = self.avg_bitrate;
                    }

                    ui.add_space(8.0);
                    heading(ui, "ENCODER & RESOLUTION");

                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Codec:").size(11.5).color(TEXT));
                        egui::ComboBox::from_id_salt("video_codec_select")
                            .selected_text(self.selected_codec.display_name())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.selected_codec, VideoCodec::NvencHevc, VideoCodec::NvencHevc.display_name());
                                ui.selectable_value(&mut self.selected_codec, VideoCodec::NvencH264, VideoCodec::NvencH264.display_name());
                                ui.selectable_value(&mut self.selected_codec, VideoCodec::NvencAv1, VideoCodec::NvencAv1.display_name());
                                ui.selectable_value(&mut self.selected_codec, VideoCodec::CpuX264, VideoCodec::CpuX264.display_name());
                                ui.selectable_value(&mut self.selected_codec, VideoCodec::CpuX265, VideoCodec::CpuX265.display_name());
                            });
                    });

                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Target Res:").size(11.5).color(TEXT));
                        let prev_res = self.selected_res;
                        egui::ComboBox::from_id_salt("video_res_select")
                            .selected_text(self.selected_res.display_name())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.selected_res, TargetResolution::Source, TargetResolution::Source.display_name());
                                ui.selectable_value(&mut self.selected_res, TargetResolution::P1440, TargetResolution::P1440.display_name());
                                ui.selectable_value(&mut self.selected_res, TargetResolution::P1080, TargetResolution::P1080.display_name());
                                ui.selectable_value(&mut self.selected_res, TargetResolution::P4K, TargetResolution::P4K.display_name());
                                ui.selectable_value(&mut self.selected_res, TargetResolution::P720, TargetResolution::P720.display_name());
                            });
                        if self.selected_res != prev_res && self.is_auto_video_preset() {
                            self.apply_auto_bitrate();
                        }
                    });

                    ui.add_space(6.0);
                    let (est_dur, src_bytes_opt) = if let Some(ref player) = self.video_player {
                        let dur = (player.trim_end_sec.unwrap_or(player.duration_sec) - player.trim_start_sec.unwrap_or(0.0)).max(0.1);
                        let full_dur = player.duration_sec.max(0.001);
                        let src_bytes = std::fs::metadata(&player.file_path).map(|m| m.len()).unwrap_or(0);
                        let scaled_src = ((src_bytes as f64) * (dur / full_dur).clamp(0.0, 1.0)) as u64;
                        (dur, if scaled_src > 0 { Some(scaled_src) } else { None })
                    } else {
                        (self.video_duration.max(1) as f64, None)
                    };
                    let est_bytes = (((self.avg_bitrate as f64) + 0.128) * 1_000_000.0 / 8.0 * est_dur) as u64;
                    let fmt_size = |b: u64| -> String {
                        if b >= 1_000_000 {
                            format!("{:.2} MB", b as f64 / 1_000_000.0)
                        } else {
                            format!("{} KB", (b / 1024).max(1))
                        }
                    };
                    if let Some(src_bytes) = src_bytes_opt {
                        if est_bytes > src_bytes {
                            ui.label(
                                RichText::new(format!(
                                    "Warning: Estimated output ({}) is larger than source ({}). Lower bitrate or uncheck compression.",
                                    fmt_size(est_bytes),
                                    fmt_size(src_bytes)
                                ))
                                .size(10.5)
                                .color(AMBER),
                            );
                        } else {
                            ui.label(
                                RichText::new(format!(
                                    "Estimated output: ~{} (source: {})",
                                    fmt_size(est_bytes),
                                    fmt_size(src_bytes)
                                ))
                                .size(10.5)
                                .color(DIM),
                            );
                        }
                    } else {
                        ui.label(
                            RichText::new(format!("Estimated output: ~{}", fmt_size(est_bytes)))
                                .size(10.5)
                                .color(DIM),
                        );
                    }

                    ui.add_space(6.0);
                    if ui.button("+ Save as New Preset...").clicked() {
                        self.new_video_preset_modal_open = true;
                    }
                });
            }

            ui.add_space(6.0);
            card(ui, |ui| {
                let mut audio_toggled = false;
                ui.horizontal(|ui| {
                    if ui.checkbox(&mut self.normalise_audio, "").changed() {
                        audio_toggled = true;
                    }
                    ui.label(
                        RichText::new("Normalise Audio Volume")
                            .size(12.5)
                            .strong()
                            .color(if self.normalise_audio { TEXT } else { DIM }),
                    );
                });
                if self.normalise_audio {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Threshold:").size(11.0).color(DIM));
                        if ui.add(egui::Slider::new(&mut self.audio_normalise_thresh_db, -45.0..=-5.0).suffix(" dB")).changed() {
                            audio_toggled = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Ratio:").size(11.0).color(DIM));
                        if ui.add(egui::Slider::new(&mut self.audio_normalise_ratio, 1.5..=20.0).suffix(":1")).changed() {
                            audio_toggled = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Makeup:").size(11.0).color(DIM));
                        if ui.add(egui::Slider::new(&mut self.audio_normalise_boost_db, 0.0..=24.0).suffix(" dB")).changed() {
                            audio_toggled = true;
                        }
                    });
                }
                if audio_toggled {
                    let norm = self.current_audio_normalise();
                    if let Some(ref mut player) = self.video_player {
                        player.reload_audio(norm);
                    }
                }
            });
        });
    }

    // Right Panel: Compact, power-user styled settings
    fn settings_panel(&mut self, ui: &mut egui::Ui) {
        if self.mode == StudioMode::Video {
            self.render_video_settings(ui);
            return;
        }
        egui::ScrollArea::vertical().id_salt("settings_scroll").show(ui, |ui| {
            card(ui, |ui| {
                ui.horizontal(|ui| {
                    heading(ui, "PRESETS");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let add_btn = egui::Button::new(RichText::new("+ New").size(11.0).strong().color(GREEN))
                            .fill(CARD_HI)
                            .stroke(Stroke::new(1.0, GREEN.linear_multiply(0.4)))
                            .corner_radius(4);
                        if ui.add(add_btn).on_hover_text("Create a new preset from current settings").clicked() {
                            self.add_new_preset();
                        }
                    });
                });

                let mut select_preset = None;
                ui.horizontal_wrapped(|ui| {
                    for (i, p) in self.presets.iter().enumerate() {
                        let is_active = self.active_preset_idx == Some(i);
                        let tip = match p.name.to_lowercase().as_str() {
                            "quality" => "Quality (Default): 21 samples @ Original resolution. Maximum detail and cleanest output.",
                            "balanced" => "Balanced: 10 samples @ Original resolution. Much faster while retaining full native resolution.",
                            "fast" => "Fast: 10 samples @ 720px width. Rapid draft mode. Notice: downscaling degrades resolution over repeated sequential edits.",
                            _ => "Custom user preset",
                        };
                        let btn = egui::Button::new(
                            RichText::new(&p.name)
                                .size(11.5)
                                .strong()
                                .color(if is_active { Color32::WHITE } else { TEXT }),
                        )
                        .fill(if is_active { ACCENT } else { CARD_HI })
                        .stroke(Stroke::new(1.0, if is_active { ACCENT_GLOW } else { LINE }))
                        .corner_radius(4);

                        if ui.add(btn).on_hover_text(tip).clicked() {
                            select_preset = Some(i);
                        }
                    }
                });

                if let Some(i) = select_preset {
                    self.apply_preset(i);
                }

                if let Some(idx) = self.active_preset_idx {
                    if let Some(preset) = self.presets.get(idx) {
                        if preset.name.eq_ignore_ascii_case("Fast") || preset.size_preset == SizePreset::P720 {
                            ui.add_space(2.0);
                            ui.label(
                                RichText::new("Notice: Fast downscales to 720px width. Repeated edits will progressively lose source detail. Use Balanced or Quality to keep native resolution.")
                                    .size(10.5)
                                    .color(Color32::from_rgb(225, 175, 75))
                            );
                        }
                    }
                }

                if let Some(idx) = self.active_preset_idx {
                    if idx < self.presets.len() {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            let mut name = self.presets[idx].name.clone();
                            let w = (ui.available_width() - 110.0).max(60.0);
                            let r = ui.add(egui::TextEdit::singleline(&mut name).desired_width(w));
                            if r.changed() {
                                self.presets[idx].name = name;
                                save_presets(&self.presets);
                            }

                            let save_btn = egui::Button::new(RichText::new("Save").size(11.0).strong().color(Color32::WHITE))
                                .fill(Color32::from_rgb(45, 85, 125))
                                .corner_radius(4);
                            if ui.add(save_btn).on_hover_text("Update preset with current sliders and resolution").clicked() {
                                self.save_current_to_active_preset();
                            }

                            if self.presets.len() > 1 {
                                let del_btn = egui::Button::new(RichText::new("Delete").size(11.0).color(RED))
                                    .fill(CARD_HI)
                                    .stroke(Stroke::new(1.0, RED.linear_multiply(0.4)))
                                    .corner_radius(4);
                                if ui.add(del_btn).on_hover_text("Delete this preset").clicked() {
                                    self.delete_active_preset();
                                }
                            }
                        });
                    }
                }
            });

            ui.add_space(6.0);

            card(ui, |ui| {
                heading(ui, "TARGET RESOLUTION");

                let row1 = [
                    (SizePreset::Original, "Original (1x)"),
                    (SizePreset::P720, "720px width"),
                ];
                let row2 = [
                    (SizePreset::P1024, "1024px width"),
                    (SizePreset::Custom, "Custom"),
                ];

                for row in [row1, row2] {
                    ui.columns(2, |cols| {
                        for (i, (preset, label)) in row.iter().enumerate() {
                            let active = self.size_preset == *preset;
                            let b = egui::Button::new(
                                RichText::new(*label).size(11.5).color(if active { Color32::WHITE } else { TEXT })
                            )
                            .fill(if active { ACCENT } else { CARD_HI })
                            .stroke(Stroke::new(1.0, if active { ACCENT_GLOW } else { LINE }))
                            .corner_radius(4)
                            .min_size(vec2(cols[i].available_width(), 24.0));
                            if cols[i].add(b).clicked() {
                                self.apply_size_preset(*preset);
                            }
                        }
                    });
                    ui.add_space(2.0);
                }

                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label(RichText::new("Resolution scale:").size(11.5).color(DIM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!("{:.2}x ({:.0}%)", self.scale, self.scale * 100.0)).size(11.5).strong());
                    });
                });
                let base_w = if self.ref_w > 0 { self.ref_w } else { self.custom_w.max(ABS_MIN_RES) };
                let base_h = if self.ref_h > 0 { self.ref_h } else { self.custom_h.max(ABS_MIN_RES) };
                let (i_min, i_max) = absolute_scale_range(base_w, base_h, ABS_MIN_RES, ABS_MAX_RES);
                let slider = egui::Slider::new(&mut self.scale, i_min..=i_max)
                    .show_value(false);
                let resp = ui.add(slider);
                if resp.changed() {
                    self.update_scale_dimensions();
                }

                ui.add_space(2.0);

                ui.horizontal(|ui| {
                    ui.label(RichText::new("W:").size(11.5).color(DIM));
                    let w_resp = ui.add(egui::DragValue::new(&mut self.custom_w).speed(8).range(64..=4096).suffix(" px"));
                    ui.label(RichText::new("H:").size(11.5).color(DIM));
                    let h_resp = ui.add(egui::DragValue::new(&mut self.custom_h).speed(8).range(64..=4096).suffix(" px"));
                    if w_resp.changed() {
                        self.custom_w = ((self.custom_w as f32 / 16.0).round() as u32 * 16).clamp(64, 4096);
                        self.ref_w = self.custom_w;
                        self.ref_h = self.custom_h;
                        self.scale = 1.0;
                        self.size_preset = SizePreset::Custom;
                    }
                    if h_resp.changed() {
                        self.custom_h = ((self.custom_h as f32 / 16.0).round() as u32 * 16).clamp(64, 4096);
                        self.ref_w = self.custom_w;
                        self.ref_h = self.custom_h;
                        self.scale = 1.0;
                        self.size_preset = SizePreset::Custom;
                    }
                    if let Some((n_w, n_h)) = self.active_node().map(|n| (n.width, n.height)) {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(format!("(base: {}x{})", n_w, n_h)).size(11.0).color(DIM));
                        });
                    }
                });

                // Dynamic load warning algorithm based on MP-steps
                let (tw, th) = match self.size_preset {
                    SizePreset::Original => {
                        if let Some(n) = self.active_node() {
                            (n.width, n.height)
                        } else {
                            (1024, 1024)
                        }
                    }
                    _ => (self.custom_w, self.custom_h),
                };
                let mp = (tw as f64 * th as f64) / 1_000_000.0;
                let work_metric = mp * (self.steps as f64);
                if work_metric >= 32.0 || mp >= 4.0 {
                    ui.add_space(3.0);
                    ui.label(
                        RichText::new(format!(
                            "Notice: Heavy generation load ({}x{} @ {} steps = {:.1} MP-steps). Generation will take longer and requires high VRAM.",
                            tw, th, self.steps, work_metric
                        ))
                        .size(10.5)
                        .color(Color32::from_rgb(230, 165, 60))
                    );
                }
            });

            ui.add_space(6.0);

            card(ui, |ui| {
                heading(ui, "SAMPLING & GUIDANCE");

                ui.horizontal(|ui| {
                    ui.label(RichText::new("Sampling steps").size(11.5).color(DIM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add(egui::DragValue::new(&mut self.steps).range(4..=50));
                    });
                });
                ui.add(egui::Slider::new(&mut self.steps, 4..=50).show_value(false));

                ui.add_space(4.0);

                ui.label(RichText::new("Reference detail").size(11.5).color(DIM));
                ui.columns(4, |cols| {
                    for (i, (val, label)) in RESOLUTIONS.iter().enumerate() {
                        let active = self.resolution == *val;
                        let b = egui::Button::new(RichText::new(*label).size(11.0).color(if active { Color32::WHITE } else { TEXT }))
                            .fill(if active { ACCENT } else { CARD_HI })
                            .stroke(Stroke::new(1.0, if active { ACCENT_GLOW } else { LINE }))
                            .corner_radius(4)
                            .min_size(vec2(cols[i].available_width(), 22.0));
                        if cols[i].add(b).clicked() {
                            self.resolution = *val;
                        }
                    }
                });

                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label(RichText::new("Guidance (CFG)").size(11.5).color(DIM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add(egui::DragValue::new(&mut self.cfg).speed(0.05).range(1.0..=10.0));
                    });
                });
                ui.add(egui::Slider::new(&mut self.cfg, 1.0..=10.0).step_by(0.05).show_value(false));

                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    ui.label(RichText::new("Seed").size(11.5).color(DIM));
                    ui.checkbox(&mut self.random_seed, "Random");
                    if !self.random_seed {
                        ui.add(egui::DragValue::new(&mut self.seed));
                    }
                });

                ui.add_space(2.0);
                ui.collapsing("Negative prompt (optional)", |ui| {
                    ui.add(egui::TextEdit::multiline(&mut self.negative).desired_rows(2).desired_width(f32::INFINITY));
                });
            });

            if self.history_dock == HistoryDock::Right {
                ui.add_space(6.0);
                card(ui, |ui| {
                    self.render_history_ui(ui);
                });
            }
        });
    }

    // Unified History UI (used when docked Left, Right, or Floating)
    fn render_history_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            heading(ui, "HISTORY");
            ui.label(RichText::new(format!("({})", self.nodes.len())).color(DIM).size(11.0));
            if ui.small_button("Outputs").on_hover_text("Open generated outputs folder in Windows File Explorer").clicked() {
                engine::open_output_folder(&self.sh);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if self.history_dock != HistoryDock::Floating {
                    let font_id = egui::FontId::proportional(10.5);
                    let galley = ui.painter().layout_no_wrap("Drag to undock".to_string(), font_id.clone(), DIM);
                    let grip_w = 10.0;
                    let gap = 4.0;
                    let total_size = vec2(grip_w + gap + galley.size().x, galley.size().y.max(14.0));
                    let (drag_rect, drag_resp) = ui.allocate_exact_size(total_size, egui::Sense::drag());
                    if drag_resp.hovered() || drag_resp.dragged() {
                        ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::Grab);
                    }
                    if ui.is_rect_visible(drag_rect) {
                        let p = ui.painter();
                        let col = if drag_resp.hovered() || drag_resp.dragged() { TEXT } else { DIM };
                        let grip_cx = drag_rect.left() + grip_w * 0.5;
                        let grip_cy = drag_rect.center().y;
                        for col_idx in [-1.0_f32, 0.0, 1.0] {
                            for row_idx in [-0.5_f32, 0.5] {
                                p.circle_filled(
                                    pos2(grip_cx + col_idx * 3.2, grip_cy + row_idx * 3.5),
                                    1.05,
                                    col,
                                );
                            }
                        }
                        let text_galley = p.layout_no_wrap("Drag to undock".to_string(), font_id, col);
                        p.galley(
                            pos2(drag_rect.left() + grip_w + gap, drag_rect.center().y - text_galley.size().y * 0.5),
                            text_galley,
                            col,
                        );
                    }
                    if drag_resp.drag_started() {
                        self.history_dock = HistoryDock::Floating;
                        self.show_history_window = true;
                    }
                }
            });
        });

        ui.horizontal(|ui| {
            let can_u = self.can_undo();
            let u_btn = egui::Button::new(RichText::new("Undo").size(11.0).color(if can_u { TEXT } else { DIM }))
                .fill(if can_u { CARD_HI } else { CARD })
                .corner_radius(4);
            if ui.add_enabled(can_u, u_btn).on_hover_text("Undo to parent edit (Ctrl+Z)").clicked() {
                self.undo();
            }

            let can_r = self.can_redo();
            let r_btn = egui::Button::new(RichText::new("Redo").size(11.0).color(if can_r { TEXT } else { DIM }))
                .fill(if can_r { CARD_HI } else { CARD })
                .corner_radius(4);
            if ui.add_enabled(can_r, r_btn).on_hover_text("Redo to next edit (Ctrl+Y)").clicked() {
                self.redo();
            }

            if !self.nodes.is_empty() {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Clear").on_hover_text("Clear all entries from history (keeps files on disk)").clicked() {
                        self.nodes.clear();
                        self.active_node_id = None;
                        self.video_player = None;
                        self.compare_mode = false;
                    }
                });
            }
        });

        let branches = self.get_branches();
        let mut switch_branch_tip = None;
        if branches.len() > 1 {
            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Branches:").size(10.5).color(DIM));
                for (tip_id, name, is_curr) in &branches {
                    let btn = egui::Button::new(
                        RichText::new(name)
                            .size(10.5)
                            .strong()
                            .color(if *is_curr { Color32::WHITE } else { TEXT }),
                    )
                    .fill(if *is_curr { ACCENT } else { CARD_HI })
                    .stroke(Stroke::new(1.0, if *is_curr { ACCENT_GLOW } else { LINE }))
                    .corner_radius(4);
                    if ui.add(btn).clicked() && !is_curr {
                        switch_branch_tip = Some(*tip_id);
                    }
                }
            });
        }
        if let Some(tip_id) = switch_branch_tip {
            self.select_node(tip_id);
        }

        ui.add_space(3.0);

        let active_thread = self.active_thread_ids();
        let mut to_select = None;
        let mut to_remove = None;
        let mut to_delete = None;

        egui::ScrollArea::vertical().id_salt("history_scroll").show(ui, |ui| {
            if active_thread.is_empty() {
                ui.label(RichText::new("No edits yet. Drop an image or enter a prompt.").color(DIM).size(11.0));
            } else {
                for &node_id in &active_thread {
                    if let Some(node) = self.find_node(node_id) {
                        let is_active = self.active_node_id == Some(node.id);
                        let children = self.child_nodes(node.id);
                        let has_fork = children.len() > 1;

                        let card_w = ui.available_width();
                        let (card_rect, card_resp) = ui.allocate_exact_size(vec2(card_w, 38.0), Sense::click());
                        
                        let rm_btn_rect = Rect::from_center_size(pos2(card_rect.max.x - 13.0, card_rect.center().y), vec2(18.0, 18.0));
                        let del_btn_rect = Rect::from_center_size(pos2(card_rect.max.x - 33.0, card_rect.center().y), vec2(18.0, 18.0));
                        let rm_resp = ui.allocate_rect(rm_btn_rect, Sense::click());
                        let del_resp = ui.allocate_rect(del_btn_rect, Sense::click());
                        let rm_hovered = rm_resp.hovered();
                        let del_hovered = del_resp.hovered();

                        if rm_resp.on_hover_text("Remove from history (keeps files on disk)").clicked() {
                            to_remove = Some(node.id);
                        } else if del_resp.on_hover_text("Move file to Recycle Bin and remove from history").clicked() {
                            to_delete = Some(node.id);
                        } else if card_resp.clicked() {
                            to_select = Some(node.id);
                        }
                        if rm_hovered || del_hovered || card_resp.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }

                        let p = ui.painter_at(card_rect);
                        let bg_col = if is_active {
                            ACCENT.linear_multiply(0.28)
                        } else if card_resp.hovered() {
                            CARD_HI
                        } else {
                            CARD
                        };
                        let border_col = if is_active {
                            ACCENT_GLOW
                        } else if card_resp.hovered() {
                            ACCENT.linear_multiply(0.6)
                        } else {
                            LINE
                        };
                        p.rect_filled(card_rect, 6, bg_col);
                        p.rect_stroke(card_rect, 6, Stroke::new(1.0, border_col), StrokeKind::Inside);

                        let thumb_rect = Rect::from_min_size(card_rect.min + vec2(4.0, 4.0), vec2(30.0, 30.0));
                        p.rect_filled(thumb_rect, 3, Color32::from_rgb(15, 17, 24));
                        let s = fit(node.tex.size_vec2(), thumb_rect.shrink(1.0).size());
                        p.image(node.tex.id(), Rect::from_center_size(thumb_rect.center(), s), FULL_UV, Color32::WHITE);
                        p.rect_stroke(thumb_rect, 3, Stroke::new(1.0, if is_active { ACCENT } else { LINE }), StrokeKind::Inside);

                        let text_left = thumb_rect.max.x + 6.0;
                        let text_max_w = (del_btn_rect.min.x - text_left - 6.0).max(10.0);
                        let title = if is_active {
                            format!("> {}", node.label)
                        } else {
                            node.label.clone()
                        };
                        let font_title = egui::FontId::proportional(11.0);
                        let fitted_title = Self::fit_single_line_middle(&p, &title, font_title.clone(), text_max_w);
                        let title_galley = p.layout_no_wrap(
                            fitted_title,
                            font_title,
                            if is_active { Color32::WHITE } else { TEXT },
                        );
                        p.galley(pos2(text_left, card_rect.min.y + 5.0), title_galley, Color32::WHITE);

                        let sub = if node.mp4_bytes.is_some() || node.source_video_path.is_some() {
                            format!("Video | {}x{}", node.width, node.height)
                        } else {
                            format!("Step {} | {}x{}", node.step_num, node.width, node.height)
                        };
                        let sub_font = egui::FontId::proportional(10.0);
                        let fitted_sub = Self::fit_single_line_middle(&p, &sub, sub_font.clone(), text_max_w);
                        let sub_galley = p.layout_no_wrap(
                            fitted_sub,
                            sub_font,
                            DIM,
                        );
                        p.galley(pos2(text_left, card_rect.min.y + 21.0), sub_galley, DIM);

                        if has_fork {
                            let fork_txt = format!("{} branches", children.len());
                            let fork_g = p.layout_no_wrap(fork_txt, egui::FontId::proportional(9.5), AMBER);
                            let fork_pos = pos2(del_btn_rect.min.x - fork_g.size().x - 4.0, card_rect.min.y + 20.0);
                            p.galley(fork_pos, fork_g, AMBER);
                        }

                        // Render Delete Bin SVG & Remove (x) Buttons
                        if del_hovered {
                            p.rect_filled(del_btn_rect, 3, Color32::from_rgb(180, 45, 45));
                            draw_trash_icon(&p, del_btn_rect.center(), Color32::WHITE);
                        } else {
                            p.rect_filled(del_btn_rect, 3, Color32::from_white_alpha(15));
                            draw_trash_icon(&p, del_btn_rect.center(), DIM);
                        }
                        if rm_hovered {
                            p.rect_filled(rm_btn_rect, 3, Color32::from_rgb(72, 78, 96));
                            p.text(rm_btn_rect.center(), egui::Align2::CENTER_CENTER, "×", egui::FontId::proportional(13.0), Color32::WHITE);
                        } else {
                            p.rect_filled(rm_btn_rect, 3, Color32::from_white_alpha(15));
                            p.text(rm_btn_rect.center(), egui::Align2::CENTER_CENTER, "×", egui::FontId::proportional(12.0), DIM);
                        }

                        if !node.prompt.is_empty() && !del_hovered && !rm_hovered {
                            card_resp.on_hover_text(&node.prompt);
                        }

                        ui.add_space(2.0);
                    }
                }
            }
        });

        if let Some(id) = to_remove {
            self.remove_history_node(id);
        } else if let Some(id) = to_delete {
            self.request_delete_history_node(id);
        } else if let Some(id) = to_select {
            self.select_node(id);
        }
    }

    fn floating_history_window(&mut self, ctx: &egui::Context) {
        if self.history_dock != HistoryDock::Floating || !self.show_history_window {
            return;
        }
        let mut open = self.show_history_window;
        let screen_rect = ctx.content_rect();
        let left_dock_thresh = screen_rect.min.x + 280.0;
        let right_dock_thresh = screen_rect.max.x - 300.0;

        let win_resp = egui::Window::new("History")
            .open(&mut open)
            .id(egui::Id::new("floating_history_window"))
            .default_width(280.0)
            .default_height(400.0)
            .resizable(true)
            .collapsible(true)
            .show(ctx, |ui| {
                self.render_history_ui(ui);
            });
        self.show_history_window = open;

        if let Some(resp) = win_resp {
            let pointer_pos = ctx.input(|i| i.pointer.latest_pos());
            let is_dragged = resp.response.dragged();
            let is_down = ctx.input(|i| i.pointer.primary_down());
            let just_released = ctx.input(|i| i.pointer.any_released());

            if let Some(pos) = pointer_pos {
                let hover_left = pos.x < left_dock_thresh;
                let hover_right = pos.x > right_dock_thresh;

                if is_dragged || (is_down && (hover_left || hover_right)) {
                    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("dock_overlay")));
                    if hover_left {
                        let r = Rect::from_min_max(screen_rect.min, pos2(left_dock_thresh, screen_rect.max.y));
                        painter.rect_filled(r, 8.0, Color32::from_rgba_unmultiplied(20, 100, 180, 50));
                        painter.rect_stroke(r.shrink(4.0), 8.0, Stroke::new(2.0, ACCENT), StrokeKind::Inside);
                        painter.text(r.center(), egui::Align2::CENTER_CENTER, "Drop to dock Left", egui::FontId::proportional(15.0), Color32::WHITE);
                    } else if hover_right {
                        let r = Rect::from_min_max(pos2(right_dock_thresh, screen_rect.min.y), screen_rect.max);
                        painter.rect_filled(r, 8.0, Color32::from_rgba_unmultiplied(20, 100, 180, 50));
                        painter.rect_stroke(r.shrink(4.0), 8.0, Stroke::new(2.0, ACCENT), StrokeKind::Inside);
                        painter.text(r.center(), egui::Align2::CENTER_CENTER, "Drop to dock Right", egui::FontId::proportional(15.0), Color32::WHITE);
                    }
                }

                if just_released {
                    if hover_left {
                        self.history_dock = HistoryDock::Left;
                    } else if hover_right {
                        self.history_dock = HistoryDock::Right;
                    }
                }
            }
        }
    }

    fn render_model_checkboxes(&self, ui: &mut egui::Ui) {
        let sh = &self.sh;
        let vid_prof = sh.active_video_profile.lock().unwrap().clone();
        let vid_gb = vid_prof.total_size_gb();
        let vid_short = if vid_prof.is_minimax_h3() {
            "MiniMax H3"
        } else if vid_prof.is_ltx25() {
            "LTX-2.5"
        } else {
            "LTX-2.3"
        };
        egui::Frame::new()
            .fill(Color32::from_rgb(20, 23, 32))
            .stroke(Stroke::new(1.0, LINE))
            .corner_radius(6)
            .inner_margin(egui::Margin::symmetric(12, 8))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    let mut dl_img = sh.dl_image_model.load(Ordering::SeqCst);
                    let mut dl_vid = sh.dl_video_model.load(Ordering::SeqCst);

                    let resp_img = ui.checkbox(&mut dl_img, RichText::new("Image Model (Qwen 2.1 - ~16 GB)").size(12.0).strong());
                    if resp_img.changed() {
                        engine::toggle_model_download(sh, ui.ctx(), true, dl_img);
                    }

                    ui.add_space(16.0);

                    let resp_vid = ui.checkbox(&mut dl_vid, RichText::new(format!("Video Model ({vid_short} - ~{vid_gb:.0} GB)")).size(12.0).strong());
                    if resp_vid.changed() {
                        engine::toggle_model_download(sh, ui.ctx(), false, dl_vid);
                    }
                });
            });
    }



    fn render_video_canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, max_size: Vec2) {
        let wants_kb = ctx.egui_wants_keyboard_input();

        let (space_pressed, left_pressed, right_pressed, i_pressed, o_pressed, undo_pressed, redo_pressed) = if !wants_kb {
            ctx.input(|i| {
                let ctrl = i.modifiers.ctrl || i.modifiers.command;
                (
                    i.key_pressed(egui::Key::Space),
                    i.key_pressed(egui::Key::ArrowLeft),
                    i.key_pressed(egui::Key::ArrowRight),
                    i.key_pressed(egui::Key::I),
                    i.key_pressed(egui::Key::O),
                    ctrl && i.key_pressed(egui::Key::Z) && !i.modifiers.shift,
                    ctrl && (i.key_pressed(egui::Key::Y) || (i.key_pressed(egui::Key::Z) && i.modifiers.shift)),
                )
            })
        } else {
            (false, false, false, false, false, false, false)
        };

        if undo_pressed {
            self.perform_trim_undo();
        } else if redo_pressed {
            self.perform_trim_redo();
        }

        let mut pending_undo_snapshot: Option<TrimStateSnapshot> = None;
        let mut close_video = false;
        let mut save_video = false;
        let is_edit_mode = self.is_edit_video_mode();
        let mut toggle_work_mode: Option<VideoWorkMode> = None;
        let mut extract_frame = false;
        let mut trigger_trim = false;
        let mut interacted_with_trim = false;
        let video_canvas_rect;

        {
            let player = match self.video_player.as_mut() {
                Some(p) => p,
                None => return,
            };

            player.update(ctx);

            if space_pressed {
                player.toggle_play();
            }
            if left_pressed {
                player.step_frames(-1);
            }
            if right_pressed {
                player.step_frames(1);
            }
            if is_edit_mode && i_pressed {
                let snap = TrimStateSnapshot {
                    start: player.trim_start_sec,
                    end: player.trim_end_sec,
                };
                player.set_trim_start_to_current();
                if player.trim_start_sec != snap.start {
                    pending_undo_snapshot = Some(snap);
                    interacted_with_trim = true;
                }
            }
            if is_edit_mode && o_pressed {
                let snap = TrimStateSnapshot {
                    start: player.trim_start_sec,
                    end: player.trim_end_sec,
                };
                player.set_trim_end_to_current();
                if player.trim_end_sec != snap.end {
                    pending_undo_snapshot = Some(snap);
                    interacted_with_trim = true;
                }
            }

            let dur = player.duration_sec.max(0.001);
            let cur_s = player.current_sec;
            let fps = player.original_fps.max(1.0);
            let p_w = player.original_width;
            let p_h = player.original_height;

            ui.horizontal(|ui| {
                heading(ui, &format!("Video Player   {} × {} px   •   {:.1}s @ {:.0} fps", p_w, p_h, dur, fps));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close Video").on_hover_text("Close player and return to image canvas").clicked() {
                        close_video = true;
                    }
                    if ui.button("Save MP4 as...").on_hover_text("Export raw video file").clicked() {
                        save_video = true;
                    }
                    if self.mode == StudioMode::Video {
                        if ui.button("Use Frame as Base Image").on_hover_text("Switch to Image mode while keeping this video scrubber to edit frames").clicked() {
                            extract_frame = true;
                        }
                    }

                    if is_edit_mode {
                        // Drawn Scissors SVG + Trim & Compress button
                        let trim_btn_label = if self.compress_enabled {
                            "Trim & Compress"
                        } else {
                            "Trim / Export"
                        };
                        let (trim_rect, trim_resp) = ui.allocate_exact_size(vec2(132.0, 24.0), Sense::click());
                        if trim_resp.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        if trim_resp.clicked() {
                            trigger_trim = true;
                        }
                        let trim_bg = if trim_resp.is_pointer_button_down_on() {
                            Color32::from_rgb(109, 40, 217)
                        } else if trim_resp.hovered() {
                            Color32::from_rgb(139, 92, 246)
                        } else {
                            ACCENT
                        };
                        let tp = ui.painter();
                        tp.rect_filled(trim_rect, 5.0, trim_bg);
                        tp.rect_stroke(trim_rect, 5.0, Stroke::new(1.0, ACCENT_GLOW), StrokeKind::Inside);
                        draw_scissors_icon(tp, Pos2::new(trim_rect.left() + 14.0, trim_rect.center().y), Color32::WHITE);
                        tp.text(
                            Pos2::new(trim_rect.left() + 26.0, trim_rect.center().y),
                            egui::Align2::LEFT_CENTER,
                            trim_btn_label,
                            egui::FontId::proportional(11.5),
                            Color32::WHITE,
                        );
                        let _ = trim_resp.on_hover_text("Render trimmed & compressed MP4 to Trim History");
                    }

                    if self.mode == StudioMode::Video {
                        if is_edit_mode {
                            if ui
                                .button("Use First Frame for Generation")
                                .on_hover_text("Switch to Generate mode and use the current scrubbed frame as the first frame for I2V")
                                .clicked()
                            {
                                toggle_work_mode = Some(VideoWorkMode::Generate);
                            }
                        } else if ui
                            .button("Back to Edit Mode")
                            .on_hover_text("Switch back to Edit mode with trim markers and LTX IC-LoRAs")
                            .clicked()
                        {
                            toggle_work_mode = Some(VideoWorkMode::Edit);
                        }
                    }

                    let zoom_txt = format!("{:.0}%", self.canvas_zoom * 100.0);
                    if ui.button(zoom_txt).on_hover_text("Click to reset zoom and center video").clicked() {
                        self.canvas_zoom = 1.0;
                        self.canvas_pan = vec2(0.0, 0.0);
                    }
                });
            });

            if self.canvas_zoom < 1.0 {
                self.canvas_zoom = 1.0;
                self.canvas_pan = vec2(0.0, 0.0);
                self.video_pan_drag_active = false;
            }

            let canvas_area = max_size - vec2(0.0, 128.0);
            let canvas_sense = if self.canvas_zoom > 1.001 {
                Sense::click_and_drag()
            } else {
                Sense::click()
            };
            let (canvas_rect, resp) = ui.allocate_exact_size(canvas_area, canvas_sense);
            video_canvas_rect = canvas_rect;

            let target_size = if self.generating {
                if let Some((gw, gh)) = self.generating_dims {
                    vec2(gw as f32, gh as f32)
                } else {
                    vec2(p_w as f32, p_h as f32)
                }
            } else {
                vec2(p_w as f32, p_h as f32)
            };
            let fit_size = fit(target_size, canvas_area);

            // Zoom with mouse scroll (Min 100%, no zooming out below 100%)
            if resp.hovered() {
                let (scroll_steps, pinch_delta) = ctx.input(|i| {
                    let mut steps = 0.0f32;
                    for ev in &i.raw.events {
                        if let egui::Event::MouseWheel { unit, delta, .. } = ev {
                            match unit {
                                egui::MouseWheelUnit::Line => steps += delta.y,
                                egui::MouseWheelUnit::Point => steps += delta.y / 60.0,
                                egui::MouseWheelUnit::Page => steps += delta.y * 3.0,
                            }
                        }
                    }
                    (steps, i.zoom_delta())
                });

                let mut zoom_factor = 1.0f32;
                if scroll_steps != 0.0 {
                    let clamped = scroll_steps.clamp(-4.0, 4.0);
                    zoom_factor *= 1.12f32.powf(clamped);
                }
                if (pinch_delta - 1.0).abs() > 0.001 {
                    zoom_factor *= pinch_delta;
                }

                if (zoom_factor - 1.0).abs() > 0.0005 {
                    let old_zoom = self.canvas_zoom;
                    let new_zoom = (old_zoom * zoom_factor).clamp(1.0, 10.0);
                    if new_zoom <= 1.001 {
                        self.canvas_zoom = 1.0;
                        self.canvas_pan = vec2(0.0, 0.0);
                        self.video_pan_drag_active = false;
                    } else {
                        if let Some(mouse) = resp.hover_pos() {
                            let center = canvas_rect.center();
                            let mouse_rel = mouse - (center + self.canvas_pan);
                            self.canvas_pan -= mouse_rel * (new_zoom / old_zoom - 1.0);
                        }
                        self.canvas_zoom = new_zoom;
                    }
                }
            }

            // Lock center to default position at 100% zoom (no dragging/panning);
            // when zoomed in, require a 6px drag deadzone before panning so clicks cleanly play/pause.
            if self.canvas_zoom > 1.001 {
                let primary_down = ctx.input(|i| i.pointer.primary_down());
                if resp.is_pointer_button_down_on() {
                    if let (Some(origin), Some(pos)) = ctx.input(|i| (i.pointer.press_origin(), i.pointer.interact_pos())) {
                        if (pos - origin).length() > 6.0 {
                            self.video_pan_drag_active = true;
                        }
                    }
                }
                if self.video_pan_drag_active && resp.dragged() {
                    self.canvas_pan += resp.drag_delta();
                }
                if resp.double_clicked() {
                    self.canvas_zoom = 1.0;
                    self.canvas_pan = vec2(0.0, 0.0);
                    self.video_pan_drag_active = false;
                } else if !self.generating && !self.video_pan_drag_active && resp.hovered() && ctx.input(|i| i.pointer.primary_released()) {
                    player.toggle_play();
                }
                if !primary_down {
                    self.video_pan_drag_active = false;
                }
            } else {
                self.canvas_zoom = 1.0;
                self.canvas_pan = vec2(0.0, 0.0);
                self.video_pan_drag_active = false;
                if !self.generating && !resp.double_clicked() && resp.clicked() {
                    player.toggle_play();
                }
            }

            let cur_size = fit_size * self.canvas_zoom;
            let cur_center = canvas_rect.center() + self.canvas_pan;
            let image_rect = Rect::from_center_size(cur_center, cur_size);

            let p = ui.painter().with_clip_rect(canvas_rect);
            p.rect_filled(canvas_rect, 8, Color32::from_rgb(14, 16, 22));
            p.rect_stroke(canvas_rect, 8, Stroke::new(1.0, LINE), StrokeKind::Inside);

            if self.generating {
                if let Some(ref prev) = self.preview_tex {
                    p.image(prev.id(), image_rect, FULL_UV, Color32::WHITE);
                } else if let Some(tex) = player.current_texture(ctx) {
                    p.image(tex.id(), image_rect, FULL_UV, Color32::from_white_alpha(140));
                } else {
                    p.rect_filled(image_rect, 6, Color32::from_rgb(18, 20, 28));
                }
                draw_generating_overlay(&p, image_rect, self.progress, self.is_swapping_detected, self.step_duration_ema);
            } else if let Some(tex) = player.current_texture(ctx) {
                p.image(tex.id(), image_rect, FULL_UV, Color32::WHITE);
                p.rect_stroke(image_rect, 0, Stroke::new(1.0, Color32::from_white_alpha(30)), StrokeKind::Outside);
            }

            ui.add_space(6.0);

            // ztrim Dual-Track Vegas-Style Timeline & Transport Controls
            card(ui, |ui| {
                ui.vertical(|ui| {
                    let full_w = ui.available_width();
                    let show_hint = is_edit_mode && !self.has_seen_trim_hint;
                    let top_margin = if is_edit_mode {
                        if show_hint { 20.0 } else { 10.0 }
                    } else {
                        4.0
                    };
                    let total_h = if is_edit_mode { top_margin + 28.0 } else { 24.0 };
                    let (track_rect, scrub_resp) = ui.allocate_exact_size(vec2(full_w, total_h), Sense::click_and_drag());
                    let p = ui.painter();

                    let trim_s = player.trim_start();
                    let trim_e = player.trim_end();

                    let start_ratio = (trim_s / dur).clamp(0.0, 1.0) as f32;
                    let end_ratio = (trim_e / dur).clamp(0.0, 1.0) as f32;
                    let trim_x_min = track_rect.left() + track_rect.width() * start_ratio;
                    let trim_x_max = track_rect.left() + track_rect.width() * end_ratio;

                    let trim_track_y = track_rect.top() + top_margin;
                    let trim_zone_bottom = trim_track_y + 4.0;
                    let play_bar_top = if is_edit_mode { trim_zone_bottom + 3.0 } else { track_rect.top() };
                    let split_y = if is_edit_mode { trim_zone_bottom + 1.0 } else { track_rect.top() };

                    // 1. Playback Bar
                    let play_track_h = 15.0;
                    let play_track_y = play_bar_top + (track_rect.bottom() - play_bar_top) * 0.5;
                    let play_track_bg = Rect::from_center_size(
                        Pos2::new(track_rect.center().x, play_track_y),
                        vec2(track_rect.width(), play_track_h),
                    );
                    if is_edit_mode {
                        let scrub_trimmed_color = Color32::from_rgb(82, 86, 98);
                        let scrub_stroke_color = Color32::from_rgb(125, 131, 148);
                        p.rect_filled(play_track_bg, 4.0, scrub_trimmed_color);
                        p.rect_stroke(play_track_bg, 4.0, Stroke::new(1.0, scrub_stroke_color), StrokeKind::Inside);

                        let play_trim_rect = Rect::from_min_max(
                            Pos2::new(trim_x_min, play_track_y - play_track_h * 0.5),
                            Pos2::new(trim_x_max, play_track_y + play_track_h * 0.5),
                        );
                        p.rect_filled(play_trim_rect, 3.0, Color32::from_rgb(37, 99, 235));
                        p.rect_stroke(play_trim_rect, 3.0, Stroke::new(1.0, Color32::from_rgb(96, 165, 250)), StrokeKind::Inside);
                    } else {
                        p.rect_filled(play_track_bg, 4.0, Color32::from_rgb(37, 99, 235));
                        p.rect_stroke(play_track_bg, 4.0, Stroke::new(1.0, Color32::from_rgb(96, 165, 250)), StrokeKind::Inside);
                    }

                    if scrub_resp.hovered() {
                        if let Some(pos) = scrub_resp.hover_pos() {
                            if pos.y >= play_bar_top {
                                p.line_segment(
                                    [Pos2::new(pos.x, play_track_bg.top() - 1.0), Pos2::new(pos.x, play_track_bg.bottom() + 1.0)],
                                    Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 255, 255, 140)),
                                );
                            }
                        }
                    }

                    // 2. Trim Bar
                    if is_edit_mode {
                        p.line_segment(
                            [Pos2::new(track_rect.left(), trim_track_y), Pos2::new(track_rect.right(), trim_track_y)],
                            Stroke::new(1.0, Color32::from_rgb(46, 46, 52)),
                        );
                        p.line_segment(
                            [Pos2::new(trim_x_min, trim_track_y), Pos2::new(trim_x_max, trim_track_y)],
                            Stroke::new(3.0, ACCENT),
                        );

                        let tab_size = 6.0;
                        let start_tri = vec![
                            Pos2::new(trim_x_min - tab_size, trim_track_y - 7.0),
                            Pos2::new(trim_x_min + tab_size, trim_track_y - 7.0),
                            Pos2::new(trim_x_min, trim_track_y),
                        ];
                        p.add(egui::Shape::convex_polygon(start_tri, BLUE, Stroke::new(1.0, Color32::WHITE)));
                        p.line_segment(
                            [Pos2::new(trim_x_min, trim_track_y), Pos2::new(trim_x_min, track_rect.bottom())],
                            Stroke::new(1.5, BLUE),
                        );

                        let (is_end_trim_flashing_red, should_repaint) =
                            is_end_trim_flashing_red_at(self.end_trim_error_flash_start, Instant::now());
                        if should_repaint {
                            ui.ctx().request_repaint();
                        } else if self.end_trim_error_flash_start.is_some() {
                            self.end_trim_error_flash_start = None;
                        }

                        let end_tri = vec![
                            Pos2::new(trim_x_max - tab_size, trim_track_y - 7.0),
                            Pos2::new(trim_x_max + tab_size, trim_track_y - 7.0),
                            Pos2::new(trim_x_max, trim_track_y),
                        ];
                        let end_marker_color = if is_end_trim_flashing_red { Color32::from_rgb(255, 60, 60) } else { AMBER };
                        let end_tab_stroke = Stroke::new(if is_end_trim_flashing_red { 1.5 } else { 1.0 }, Color32::WHITE);
                        let end_line_stroke = if is_end_trim_flashing_red {
                            Stroke::new(2.5, Color32::from_rgb(255, 60, 60))
                        } else {
                            Stroke::new(1.5, AMBER)
                        };
                        p.add(egui::Shape::convex_polygon(end_tri, end_marker_color, end_tab_stroke));
                        p.line_segment(
                            [Pos2::new(trim_x_max, trim_track_y), Pos2::new(trim_x_max, track_rect.bottom())],
                            end_line_stroke,
                        );
                    }

                    // 3. Playhead
                    let cur_ratio = (cur_s / dur).clamp(0.0, 1.0) as f32;
                    let playhead_x = track_rect.left() + track_rect.width() * cur_ratio;
                    let marker_cap = vec![
                        Pos2::new(playhead_x - 4.5, play_track_bg.top() - 4.0),
                        Pos2::new(playhead_x + 4.5, play_track_bg.top() - 4.0),
                        Pos2::new(playhead_x, play_track_bg.top()),
                    ];
                    p.add(egui::Shape::convex_polygon(marker_cap, Color32::WHITE, Stroke::NONE));
                    p.line_segment(
                        [Pos2::new(playhead_x, play_track_bg.top() - 4.0), Pos2::new(playhead_x, play_track_bg.bottom() + 2.0)],
                        Stroke::new(2.0, Color32::WHITE),
                    );

                    // 4. Onboarding Badges
                    if show_hint {
                        let mut l_badge_center_x = trim_x_min.clamp(track_rect.left() + 48.0, track_rect.right() - 48.0);
                        let mut r_badge_center_x = trim_x_max.clamp(track_rect.left() + 48.0, track_rect.right() - 48.0);
                        if (r_badge_center_x - l_badge_center_x).abs() < 98.0 {
                            if l_badge_center_x > track_rect.left() + 52.0 {
                                l_badge_center_x -= 24.0;
                            }
                            if r_badge_center_x < track_rect.right() - 52.0 {
                                r_badge_center_x += 24.0;
                            }
                        }
                        let badge_y = track_rect.top() + 8.0;
                        let l_badge_rect = Rect::from_center_size(Pos2::new(l_badge_center_x, badge_y), vec2(90.0, 15.0));
                        p.rect_filled(l_badge_rect, 3.0, Color32::from_rgb(18, 25, 45));
                        p.rect_stroke(l_badge_rect, 3.0, Stroke::new(1.0, BLUE), StrokeKind::Inside);
                        p.text(l_badge_rect.center(), egui::Align2::CENTER_CENTER, "L-Click: Start", egui::FontId::proportional(10.0), TEXT);

                        let r_badge_rect = Rect::from_center_size(Pos2::new(r_badge_center_x, badge_y), vec2(90.0, 15.0));
                        p.rect_filled(r_badge_rect, 3.0, Color32::from_rgb(42, 32, 18));
                        p.rect_stroke(r_badge_rect, 3.0, Stroke::new(1.0, AMBER), StrokeKind::Inside);
                        p.text(r_badge_rect.center(), egui::Align2::CENTER_CENTER, "R-Click: End", egui::FontId::proportional(10.0), TEXT);
                    }

                    let is_drag_stopped = scrub_resp.drag_stopped()
                        || scrub_resp.drag_stopped_by(egui::PointerButton::Secondary)
                        || !ui.input(|i| i.pointer.primary_down() || i.pointer.secondary_down());
                    if is_drag_stopped {
                        if self.active_timeline_drag == TimelineDrag::Playhead {
                            if self.was_playing_before_scrub {
                                self.was_playing_before_scrub = false;
                                player.is_playing = true;
                                player.seek_to_time(player.current_sec);
                            } else if self.timeline_drag_occurred && !player.is_playing {
                                player.seek_to_time(player.current_sec);
                            }
                        } else if self.active_timeline_drag == TimelineDrag::TrimStart
                            || self.active_timeline_drag == TimelineDrag::TrimEnd
                        {
                            if let Some(initial) = self.trim_drag_initial.take() {
                                let current = TrimStateSnapshot {
                                    start: player.trim_start_sec,
                                    end: player.trim_end_sec,
                                };
                                if current != initial {
                                    pending_undo_snapshot = Some(initial);
                                }
                            }
                        }
                        self.active_timeline_drag = TimelineDrag::None;
                        self.timeline_drag_occurred = false;
                    }

                    let right_clicked = is_edit_mode
                        && (scrub_resp.secondary_clicked()
                            || scrub_resp.drag_started_by(egui::PointerButton::Secondary)
                            || (scrub_resp.hovered() && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary) || i.pointer.button_clicked(egui::PointerButton::Secondary))));

                    if right_clicked {
                        let origin_pos = ui.input(|i| i.pointer.press_origin())
                            .or_else(|| scrub_resp.interact_pointer_pos())
                            .or_else(|| scrub_resp.hover_pos())
                            .or_else(|| ui.input(|i| i.pointer.latest_pos()));

                        if let Some(pos) = origin_pos {
                            if pos.y < split_y {
                                let rel = ((pos.x - track_rect.left()) / track_rect.width()).clamp(0.0, 1.0) as f64;
                                let clicked_sec = rel * dur;
                                match evaluate_trim_click(true, pos.x, trim_x_min, trim_x_max, clicked_sec, player.trim_start(), player.trim_end(), dur) {
                                    TrimClickDecision::BlockRightClickFlashRed => {
                                        self.end_trim_error_flash_start = Some(Instant::now());
                                        ui.ctx().request_repaint();
                                        self.status = "Blocked: Right-click moves End Trim only (minimum trim reached)".into();
                                    }
                                    TrimClickDecision::SetEndTrim(new_e) => {
                                        self.trim_drag_initial = Some(TrimStateSnapshot {
                                            start: player.trim_start_sec,
                                            end: player.trim_end_sec,
                                        });
                                        player.trim_end_sec = Some(new_e);
                                        self.active_timeline_drag = TimelineDrag::TrimEnd;
                                        player.seek_to_time(new_e);
                                        interacted_with_trim = true;
                                    }
                                    TrimClickDecision::SetStartTrim(_) => {}
                                }
                            }
                        }
                    }

                    let left_clicked = scrub_resp.clicked()
                        || scrub_resp.drag_started()
                        || (scrub_resp.hovered() && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary) || i.pointer.button_clicked(egui::PointerButton::Primary)));

                    if !right_clicked && self.active_timeline_drag == TimelineDrag::None && left_clicked {
                        let origin_pos = ui.input(|i| i.pointer.press_origin())
                            .or_else(|| scrub_resp.interact_pointer_pos())
                            .or_else(|| scrub_resp.hover_pos())
                            .or_else(|| ui.input(|i| i.pointer.latest_pos()));

                        if let Some(pos) = origin_pos {
                            let rel = ((pos.x - track_rect.left()) / track_rect.width()).clamp(0.0, 1.0) as f64;
                            let target_sec = rel * dur;

                            if pos.y < split_y {
                                self.trim_drag_initial = Some(TrimStateSnapshot {
                                    start: player.trim_start_sec,
                                    end: player.trim_end_sec,
                                });
                                match evaluate_trim_click(false, pos.x, trim_x_min, trim_x_max, target_sec, player.trim_start(), player.trim_end(), dur) {
                                    TrimClickDecision::SetEndTrim(new_e) => {
                                        player.trim_end_sec = Some(new_e);
                                        self.active_timeline_drag = TimelineDrag::TrimEnd;
                                        player.seek_to_time(new_e);
                                        interacted_with_trim = true;
                                    }
                                    TrimClickDecision::SetStartTrim(new_s) => {
                                        player.trim_start_sec = Some(new_s);
                                        self.active_timeline_drag = TimelineDrag::TrimStart;
                                        player.seek_to_time(new_s);
                                        interacted_with_trim = true;
                                    }
                                    TrimClickDecision::BlockRightClickFlashRed => {}
                                }
                            } else {
                                self.active_timeline_drag = TimelineDrag::Playhead;
                                self.timeline_drag_occurred = false;
                                if player.is_playing {
                                    if scrub_resp.drag_started() {
                                        self.was_playing_before_scrub = true;
                                        player.pause();
                                        player.scrub_to_time(target_sec, true);
                                    } else {
                                        self.was_playing_before_scrub = false;
                                        player.seek_to_time(target_sec);
                                    }
                                } else {
                                    self.was_playing_before_scrub = false;
                                    player.seek_to_time(target_sec);
                                }
                            }
                        }
                    }

                    let is_dragged = scrub_resp.dragged()
                        || scrub_resp.dragged_by(egui::PointerButton::Secondary)
                        || (self.active_timeline_drag != TimelineDrag::None && ui.input(|i| i.pointer.primary_down() || i.pointer.secondary_down()));

                    if is_dragged {
                        if let Some(pos) = scrub_resp.interact_pointer_pos().or_else(|| scrub_resp.hover_pos()).or_else(|| ui.input(|i| i.pointer.latest_pos())) {
                            let rel = ((pos.x - track_rect.left()) / track_rect.width()).clamp(0.0, 1.0) as f64;
                            let target_sec = rel * dur;
                            match self.active_timeline_drag {
                                TimelineDrag::TrimStart => {
                                    self.timeline_drag_occurred = true;
                                    let end_s = player.trim_end();
                                    let new_s = target_sec.clamp(0.0, (end_s - 0.05).max(0.0));
                                    player.trim_start_sec = Some(new_s);
                                    player.scrub_to_time(new_s, false);
                                    interacted_with_trim = true;
                                }
                                TimelineDrag::TrimEnd => {
                                    self.timeline_drag_occurred = true;
                                    let start_s = player.trim_start();
                                    let new_e = target_sec.clamp((start_s + 0.05).min(dur), dur);
                                    player.trim_end_sec = Some(new_e);
                                    player.scrub_to_time(new_e, false);
                                    interacted_with_trim = true;
                                }
                                TimelineDrag::Playhead => {
                                    self.timeline_drag_occurred = true;
                                    if player.is_playing {
                                        self.was_playing_before_scrub = true;
                                        player.pause();
                                    }
                                    player.scrub_to_time(target_sec, self.was_playing_before_scrub);
                                }
                                TimelineDrag::None => {}
                            }
                        }
                    }

                    ui.add_space(3.0);

                    // Transport Controls Row with drawn SVG Play/Pause button
                    ui.horizontal(|ui| {
                        let is_playing = player.is_playing;
                        let (play_rect, play_resp) = ui.allocate_exact_size(vec2(48.0, 26.0), Sense::click());
                        if play_resp.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        if play_resp.clicked() {
                            player.toggle_play();
                        }
                        let play_resp = play_resp.on_hover_text(if is_playing { "Pause (Space)" } else { "Play (Space)" });

                        let is_pressed = play_resp.is_pointer_button_down_on();
                        let is_hovered = play_resp.hovered();
                        let bg = if is_playing {
                            if is_pressed {
                                Color32::from_rgb(109, 40, 217)
                            } else if is_hovered {
                                Color32::from_rgb(139, 92, 246)
                            } else {
                                ACCENT
                            }
                        } else if is_pressed {
                            Color32::from_rgb(26, 28, 34)
                        } else if is_hovered {
                            Color32::from_rgb(46, 50, 60)
                        } else {
                            CARD_HI
                        };
                        let border = if is_hovered {
                            Stroke::new(1.0, if is_playing { Color32::from_rgb(196, 181, 253) } else { Color32::from_rgb(90, 95, 110) })
                        } else {
                            Stroke::new(1.0, LINE)
                        };
                        let pp = ui.painter();
                        pp.rect_filled(play_rect, 6.0, bg);
                        pp.rect_stroke(play_rect, 6.0, border, StrokeKind::Inside);
                        if is_playing {
                            draw_pause_icon(pp, play_rect.center(), Color32::WHITE);
                        } else {
                            draw_play_icon(pp, play_rect.center(), Color32::WHITE);
                        }

                        if ui.button("< 1f").on_hover_text("Step 1 frame back (Left Arrow)").clicked() {
                            player.step_frames(-1);
                        }
                        if ui.button("1f >").on_hover_text("Step 1 frame forward (Right Arrow)").clicked() {
                            player.step_frames(1);
                        }

                        if is_edit_mode {
                            ui.add_space(4.0);
                            if ui.button("[ Set Start (I)").clicked() {
                                let snap = TrimStateSnapshot {
                                    start: player.trim_start_sec,
                                    end: player.trim_end_sec,
                                };
                                player.set_trim_start_to_current();
                                if player.trim_start_sec != snap.start {
                                    pending_undo_snapshot = Some(snap);
                                    interacted_with_trim = true;
                                }
                            }
                            if ui.button("Set End (O) ]").clicked() {
                                let snap = TrimStateSnapshot {
                                    start: player.trim_start_sec,
                                    end: player.trim_end_sec,
                                };
                                player.set_trim_end_to_current();
                                if player.trim_end_sec != snap.end {
                                    pending_undo_snapshot = Some(snap);
                                    interacted_with_trim = true;
                                }
                            }
                            if player.has_custom_trim() && ui.button("Reset Trim").clicked() {
                                let snap = TrimStateSnapshot {
                                    start: player.trim_start_sec,
                                    end: player.trim_end_sec,
                                };
                                player.reset_trim();
                                pending_undo_snapshot = Some(snap);
                                self.end_trim_error_flash_start = None;
                            }
                        }

                        ui.add_space(6.0);
                        let time_str = format!(
                            "{:02}:{:04.1} / {:02}:{:04.1}",
                            (cur_s / 60.0) as u32, cur_s % 60.0,
                            (dur / 60.0) as u32, dur % 60.0
                        );
                        ui.label(RichText::new(time_str).size(12.0).monospace().color(TEXT));

                        if is_edit_mode {
                            if player.has_custom_trim() {
                                let trimmed_len = player.trimmed_duration();
                                let pct_saved = ((1.0 - (trimmed_len / dur)) * 100.0).clamp(0.0, 100.0);
                                let cut_info = format!("Trim: {:.1}s (-{:.0}%)", trimmed_len, pct_saved);
                                ui.label(RichText::new(cut_info).size(11.0).color(GREEN));
                            }
                        } else if self.mode == StudioMode::Image {
                            ui.label(
                                RichText::new("Scrubbed frame used as input image")
                                    .size(11.0)
                                    .strong()
                                    .color(Color32::from_rgb(96, 165, 250)),
                            )
                            .on_hover_text("Scrub to any frame — it will be used as the input image when you click Generate in Image mode");
                        } else {
                            ui.label(
                                RichText::new("First frame for generated video")
                                    .size(11.0)
                                    .strong()
                                    .color(Color32::from_rgb(96, 165, 250)),
                            )
                            .on_hover_text("Scrub to any frame — it will be used as the first frame in the I2V workflow");
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let loop_txt = if player.loop_playback { "Loop: On" } else { "Loop: Off" };
                            let loop_btn = egui::Button::new(
                                RichText::new(loop_txt)
                                    .size(11.5)
                                    .strong()
                                    .color(if player.loop_playback { Color32::WHITE } else { DIM }),
                            )
                            .fill(if player.loop_playback { ACCENT } else { CARD_HI })
                            .stroke(Stroke::new(1.0, if player.loop_playback { ACCENT_GLOW } else { LINE }))
                            .corner_radius(6);
                            if ui.add(loop_btn).clicked() {
                                player.loop_playback = !player.loop_playback;
                            }

                            if let Some(ref mut audio) = player.audio {
                                let mut vol = audio.volume;
                                let vol_slider = egui::Slider::new(&mut vol, 0.0..=1.0).show_value(false);
                                if ui.add_sized([64.0, 16.0], vol_slider).changed() {
                                    audio.set_volume(vol);
                                }
                                let mute_txt = if audio.is_muted || audio.volume == 0.0 { "Unmute" } else { "Mute" };
                                if ui.button(mute_txt).on_hover_text("Mute / Unmute audio").clicked() {
                                    audio.toggle_mute();
                                }
                                ui.label(RichText::new("Vol:").size(10.5).color(DIM));
                            } else {
                                ui.label(RichText::new("No Audio").size(10.5).color(DIM));
                            }
                        });
                    });
                });
            });
        }

        if let Some(new_mode) = toggle_work_mode {
            self.set_video_work_mode(new_mode);
        }

        if let Some(snap) = pending_undo_snapshot {
            self.record_trim_undo(snap);
        }
        if interacted_with_trim && !self.has_seen_trim_hint {
            self.has_seen_trim_hint = true;
        }

        if close_video {
            self.video_player = None;
            return;
        }

        self.render_floating_generation_status(ctx, video_canvas_rect);

        if trigger_trim {
            self.trigger_trim_and_compress();
        }

        if save_video {
            if let Some(ref player) = self.video_player {
                let filename = player.filename.clone();
                let mp4_data = player.mp4_bytes.clone().or_else(|| std::fs::read(&player.file_path).ok());
                if let Some(mp4_data) = mp4_data {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_file_name(&filename)
                        .add_filter("MP4 Video", &["mp4"])
                        .save_file()
                    {
                        if let Err(e) = std::fs::write(&path, &mp4_data) {
                            self.error = Some(format!("Could not save MP4: {e}"));
                        }
                    }
                }
            }
        }

        if extract_frame {
            self.mode = StudioMode::Image;
        }
    }

    // Center Hero Canvas (Centered Image + Interactive Zoom & Pan)
    fn center_canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let max_size = ui.available_size();

        if self.video_player.is_some() {
            self.render_video_canvas(ui, ctx, max_size);
            return;
        }

        if self.active_node().is_some() || self.generating {
            let (active_id, active_w, active_h, active_step, active_label, active_png, active_tex_opt) =
                if let Some(n) = self.active_node() {
                    (n.id, n.width, n.height, n.step_num, n.label.clone(), n.png.clone(), Some(n.tex.clone()))
                } else {
                    (0, self.custom_w, self.custom_h, 0, "Generating...".to_string(), Vec::new(), None)
                };
            let parent_opt = if active_tex_opt.is_some() {
                self.parent_node(active_id).map(|p| (p.step_num, p.tex.clone()))
            } else {
                None
            };
            let has_active = active_tex_opt.is_some();

            let mut action: Option<&str> = None;
            let mut toggle_compare = false;
            let mut do_undo = false;
            let mut do_redo = false;
            let mut do_clear = false;

            ui.horizontal(|ui| {
                if self.compare_mode && parent_opt.is_some() {
                    let (p_step, _) = parent_opt.as_ref().unwrap();
                    heading(ui, &format!("Comparison (drag divider)   Step {p_step} vs Step {active_step}"));
                } else if self.generating && !has_active {
                    if self.mode == StudioMode::Video {
                        heading(ui, &format!("Generating video from scratch   ({} x {} px)", self.video_width, self.video_height));
                    } else {
                        heading(ui, &format!("Generating from scratch   ({} x {} px)", self.custom_w, self.custom_h));
                    }
                } else if self.mode == StudioMode::Video {
                    if self.generating {
                        heading(ui, &format!("Generating video from image   Step {}   \"{}\"   ({} x {} px)", active_step, active_label, self.video_width, self.video_height));
                    } else {
                        heading(ui, &format!("Initial Frame (Image-to-Video)   Step {}   \"{}\"   ({} x {} px)", active_step, active_label, active_w, active_h));
                    }
                } else {
                    heading(ui, &format!("Step {}   \"{}\"   ({} x {} px)", active_step, active_label, active_w, active_h));
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Clear Canvas").on_hover_text("Start fresh").clicked() {
                        do_clear = true;
                    }
                    if has_active {
                        if ui.button("Copy").clicked() {
                            action = Some("copy");
                        }
                        if ui.button("Save as...").clicked() {
                            action = Some("save");
                        }
                    }

                    if parent_opt.is_some() && !self.generating {
                        let btn_txt = if self.compare_mode { "Result view" } else { "Compare" };
                        let tip = if self.compare_mode { "Show the finished image directly" } else { "Split slider against the parent image before this edit" };
                        if ui.button(btn_txt).on_hover_text(tip).clicked() {
                            toggle_compare = true;
                        }
                    }

                    // Zoom indicator / reset button
                    let zoom_txt = format!("{:.0}%", self.canvas_zoom * 100.0);
                    if ui.button(zoom_txt).on_hover_text("Click to reset zoom and center image").clicked() {
                        self.canvas_zoom = 1.0;
                        self.canvas_pan = vec2(0.0, 0.0);
                    }

                    let can_r = self.can_redo();
                    let r_btn = egui::Button::new(RichText::new("Redo").size(11.5).color(if can_r { TEXT } else { DIM }))
                        .fill(if can_r { CARD_HI } else { CARD })
                        .corner_radius(4);
                    if ui.add_enabled(can_r, r_btn).on_hover_text("Redo to next edit (Ctrl+Y)").clicked() {
                        do_redo = true;
                    }

                    let can_u = self.can_undo();
                    let u_btn = egui::Button::new(RichText::new("Undo").size(11.5).color(if can_u { TEXT } else { DIM }))
                        .fill(if can_u { CARD_HI } else { CARD })
                        .corner_radius(4);
                    if ui.add_enabled(can_u, u_btn).on_hover_text("Undo to parent edit (Ctrl+Z)").clicked() {
                        do_undo = true;
                    }
                });
            });

            if do_clear {
                self.nodes.clear();
                self.active_node_id = None;
                self.video_player = None;
                self.compare_mode = false;
                self.prompt.clear();
                return;
            }
            if do_undo {
                self.undo();
                return;
            }
            if do_redo {
                self.redo();
                return;
            }
            if toggle_compare {
                self.compare_mode = !self.compare_mode;
            }

            let canvas_area = max_size - vec2(0.0, 36.0);
            let (canvas_rect, resp) = ui.allocate_exact_size(canvas_area, Sense::click_and_drag());

            // Target aspect ratio for canvas fitting:
            // When generating, update immediately to the generating image ratio!
            let target_size = if self.generating {
                if let Some((gw, gh)) = self.generating_dims {
                    vec2(gw as f32, gh as f32)
                } else {
                    vec2(self.custom_w as f32, self.custom_h as f32)
                }
            } else if let Some(ref tex) = active_tex_opt {
                tex.size_vec2()
            } else {
                vec2(self.custom_w as f32, self.custom_h as f32)
            };

            let fit_size = fit(target_size, canvas_area);

            let is_video_canvas = self.mode == StudioMode::Video;
            if is_video_canvas && self.canvas_zoom < 1.0 {
                self.canvas_zoom = 1.0;
                self.canvas_pan = vec2(0.0, 0.0);
            }

            // Zoom into cursor position with mouse wheel & trackpad pinch
            if resp.hovered() {
                let (scroll_steps, pinch_delta) = ctx.input(|i| {
                    let mut steps = 0.0f32;
                    for ev in &i.raw.events {
                        if let egui::Event::MouseWheel { unit, delta, .. } = ev {
                            match unit {
                                egui::MouseWheelUnit::Line => steps += delta.y,
                                egui::MouseWheelUnit::Point => steps += delta.y / 60.0,
                                egui::MouseWheelUnit::Page => steps += delta.y * 3.0,
                            }
                        }
                    }
                    (steps, i.zoom_delta())
                });

                let mut zoom_factor = 1.0f32;
                if scroll_steps != 0.0 {
                    let clamped = scroll_steps.clamp(-4.0, 4.0);
                    zoom_factor *= 1.12f32.powf(clamped);
                }
                if (pinch_delta - 1.0).abs() > 0.001 {
                    zoom_factor *= pinch_delta;
                }

                if (zoom_factor - 1.0).abs() > 0.0005 {
                    let old_zoom = self.canvas_zoom;
                    let min_zoom = if is_video_canvas { 1.0 } else { 0.15 };
                    let new_zoom = (old_zoom * zoom_factor).clamp(min_zoom, 25.0);

                    if is_video_canvas && new_zoom <= 1.001 {
                        self.canvas_zoom = 1.0;
                        self.canvas_pan = vec2(0.0, 0.0);
                    } else {
                        if let Some(mouse) = resp.hover_pos() {
                            let center = canvas_rect.center();
                            let mouse_rel = mouse - (center + self.canvas_pan);
                            self.canvas_pan -= mouse_rel * (new_zoom / old_zoom - 1.0);
                        }
                        self.canvas_zoom = new_zoom;
                    }
                }
            }

            // Pan by dragging (when not dragging compare split; locked at 100% in video mode)
            if !self.compare_mode && resp.dragged() && (!is_video_canvas || self.canvas_zoom > 1.001) {
                self.canvas_pan += resp.drag_delta();
            }
            if is_video_canvas && self.canvas_zoom <= 1.001 {
                self.canvas_pan = vec2(0.0, 0.0);
            }
            if resp.double_clicked() {
                self.canvas_zoom = 1.0;
                self.canvas_pan = vec2(0.0, 0.0);
            }

            let cur_size = fit_size * self.canvas_zoom;
            let cur_center = canvas_rect.center() + self.canvas_pan;
            let image_rect = Rect::from_center_size(cur_center, cur_size);

            let p = ui.painter().with_clip_rect(canvas_rect);
            p.rect_filled(canvas_rect, 8, Color32::from_rgb(14, 16, 22));
            p.rect_stroke(canvas_rect, 8, Stroke::new(1.0, LINE), StrokeKind::Inside);

            if self.compare_mode && parent_opt.is_some() && !self.generating {
                let (p_step, p_tex) = parent_opt.unwrap();
                let active_tex = active_tex_opt.as_ref().unwrap();
                let mut split = self.split;
                if resp.dragged() || resp.clicked() {
                    if let Some(m) = resp.interact_pointer_pos() {
                        if image_rect.width() > 0.0 {
                            split = ((m.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0);
                        }
                    }
                }

                p.image(active_tex.id(), image_rect, FULL_UV, Color32::WHITE);
                let x = image_rect.left() + image_rect.width() * split;
                p.image(
                    p_tex.id(),
                    Rect::from_min_max(image_rect.min, pos2(x, image_rect.bottom())),
                    Rect::from_min_max(pos2(0.0, 0.0), pos2(split, 1.0)),
                    Color32::WHITE,
                );
                p.line_segment([pos2(x, image_rect.top()), pos2(x, image_rect.bottom())], Stroke::new(2.0, Color32::WHITE));
                p.circle_filled(pos2(x, image_rect.center().y), 13.0, ACCENT);
                p.circle_stroke(pos2(x, image_rect.center().y), 13.0, Stroke::new(1.5, Color32::WHITE));
                p.text(pos2(x, image_rect.center().y), egui::Align2::CENTER_CENTER, "<>", egui::FontId::proportional(11.0), Color32::WHITE);

                for (t, pos, al) in [
                    (format!("BEFORE (Step {p_step})"), image_rect.left_top() + vec2(10.0, 10.0), egui::Align2::LEFT_TOP),
                    (format!("AFTER (Step {active_step})"), image_rect.right_top() + vec2(-10.0, 10.0), egui::Align2::RIGHT_TOP),
                ] {
                    let g = p.layout_no_wrap(t, egui::FontId::proportional(11.0), Color32::WHITE);
                    let r = al.anchor_size(pos, g.size()).expand2(vec2(6.0, 2.5));
                    p.rect_filled(r, 4, Color32::from_black_alpha(160));
                    p.galley(r.shrink2(vec2(6.0, 2.5)).min, g, Color32::WHITE);
                }
                self.split = split;
            } else if self.generating {
                if let Some(ref prev) = self.preview_tex {
                    p.image(prev.id(), image_rect, FULL_UV, Color32::WHITE);
                } else if let Some(ref a_tex) = active_tex_opt {
                    let base_fit = fit(a_tex.size_vec2(), image_rect.size());
                    let base_r = Rect::from_center_size(image_rect.center(), base_fit);
                    p.image(a_tex.id(), base_r, FULL_UV, Color32::from_white_alpha(120));
                } else {
                    p.rect_filled(image_rect, 6, Color32::from_rgb(18, 20, 28));
                }

                draw_generating_overlay(&p, image_rect, self.progress, self.is_swapping_detected, self.step_duration_ema);
            } else if let Some(ref a_tex) = active_tex_opt {
                p.image(a_tex.id(), image_rect, FULL_UV, Color32::WHITE);
                p.rect_stroke(image_rect, 0, Stroke::new(1.0, Color32::from_white_alpha(20)), StrokeKind::Outside);

                let (target_w, target_h) = match self.size_preset {
                    SizePreset::Original => (active_w, active_h),
                    _ => (self.custom_w, self.custom_h),
                };

                // Live visual aspect ratio box showing relative output width and height
                if target_w != active_w || target_h != active_h {
                    let pw = image_rect.width() * (target_w as f32 / active_w.max(1) as f32);
                    let ph = image_rect.height() * (target_h as f32 / active_h.max(1) as f32);
                    let preview_rect = Rect::from_center_size(image_rect.center(), vec2(pw, ph));

                    p.rect_stroke(
                        preview_rect,
                        2,
                        Stroke::new(1.5, Color32::from_rgba_premultiplied(139, 92, 246, 200)),
                        StrokeKind::Outside,
                    );

                    let b_len = 16.0f32.min(preview_rect.width() * 0.25).min(preview_rect.height() * 0.25);
                    let b_stroke = Stroke::new(2.5, ACCENT);
                    p.line_segment([preview_rect.left_top(), preview_rect.left_top() + vec2(b_len, 0.0)], b_stroke);
                    p.line_segment([preview_rect.left_top(), preview_rect.left_top() + vec2(0.0, b_len)], b_stroke);
                    p.line_segment([preview_rect.right_top(), preview_rect.right_top() + vec2(-b_len, 0.0)], b_stroke);
                    p.line_segment([preview_rect.right_top(), preview_rect.right_top() + vec2(0.0, b_len)], b_stroke);
                    p.line_segment([preview_rect.left_bottom(), preview_rect.left_bottom() + vec2(b_len, 0.0)], b_stroke);
                    p.line_segment([preview_rect.left_bottom(), preview_rect.left_bottom() + vec2(0.0, -b_len)], b_stroke);
                    p.line_segment([preview_rect.right_bottom(), preview_rect.right_bottom() + vec2(-b_len, 0.0)], b_stroke);
                    p.line_segment([preview_rect.right_bottom(), preview_rect.right_bottom() + vec2(0.0, -b_len)], b_stroke);

                    fn gcd(a: u32, b: u32) -> u32 { if b == 0 { a } else { gcd(b, a % b) } }
                    let d = gcd(target_w, target_h).max(1);
                    let tag_text = format!("Target: {}x{} ({}:{})", target_w, target_h, target_w / d, target_h / d);
                    let g = p.layout_no_wrap(tag_text, egui::FontId::proportional(11.0), Color32::WHITE);
                    let badge_pos = preview_rect.center_bottom() + vec2(0.0, 10.0);
                    let r = egui::Align2::CENTER_TOP.anchor_size(badge_pos, g.size()).expand2(vec2(8.0, 3.5));
                    p.rect_filled(r, 4, Color32::from_black_alpha(200));
                    p.rect_stroke(r, 4, Stroke::new(1.0, ACCENT), StrokeKind::Inside);
                    p.galley(r.shrink2(vec2(8.0, 3.5)).min, g, Color32::WHITE);
                }
            }

            match action {
                Some("save") => {
                    if let Some(path) = rfd::FileDialog::new().set_file_name("edit.png").add_filter("PNG", &["png"]).save_file() {
                        if let Err(e) = std::fs::write(&path, &active_png) {
                            self.error = Some(format!("Could not save: {e}"));
                        }
                    }
                }
                Some("copy") => {
                    if let Ok(img) = image::load_from_memory(&active_png) {
                        let rgba = img.to_rgba8();
                        let data = arboard::ImageData {
                            width: rgba.width() as usize,
                            height: rgba.height() as usize,
                            bytes: std::borrow::Cow::Owned(rgba.into_raw()),
                        };
                        if let Err(e) = arboard::Clipboard::new().and_then(|mut c| c.set_image(data)) {
                            self.error = Some(format!("Could not copy: {e}"));
                        }
                    }
                }
                _ => {}
            }

        } else {
            let (rect, resp) = ui.allocate_exact_size(max_size, Sense::click());
            let hover = resp.hovered();
            let p = ui.painter_at(rect);
            p.rect_filled(rect, 10, Color32::from_rgb(15, 17, 24));
            p.rect_stroke(rect, 10, Stroke::new(1.5, if hover { ACCENT } else { LINE }), StrokeKind::Inside);

            let c = rect.center();
            if self.mode == StudioMode::Video {
                let vid_prof = self.sh.active_video_profile.lock().unwrap().clone();
                let vid_title = if vid_prof.is_minimax_h3() {
                    "Video Generation Mode (MiniMax H3)"
                } else if vid_prof.is_ltx25() {
                    "Video Generation Mode (LTX-Video 2.5)"
                } else {
                    "Video Generation Mode (LTX-Video 2.3)"
                };
                p.text(
                    c - vec2(0.0, 36.0),
                    egui::Align2::CENTER_CENTER,
                    vid_title,
                    egui::FontId::proportional(19.0),
                    TEXT,
                );
                p.text(
                    c - vec2(0.0, 10.0),
                    egui::Align2::CENTER_CENTER,
                    "Enter a prompt below for Text-to-Video, or paste an image for Image-to-Video.",
                    egui::FontId::proportional(13.0),
                    ACCENT_GLOW,
                );
            } else {
                p.text(
                    c - vec2(0.0, 36.0),
                    egui::Align2::CENTER_CENTER,
                    if self.loading > 0 { "Loading image..." } else { "Start by Pasting or Dropping an Image" },
                    egui::FontId::proportional(19.0),
                    TEXT,
                );
                p.text(
                    c - vec2(0.0, 10.0),
                    egui::Align2::CENTER_CENTER,
                    "Or enter a prompt below to generate an image from scratch.",
                    egui::FontId::proportional(13.0),
                    ACCENT_GLOW,
                );
            }

            let btn_rect = Rect::from_center_size(c + vec2(0.0, 26.0), vec2(if self.mode == StudioMode::Video { 340.0 } else { 270.0 }, 34.0));
            let paste_hover = btn_rect.contains(ui.ctx().input(|i| i.pointer.hover_pos().unwrap_or_default()));
            p.rect_filled(btn_rect, 6, if paste_hover { ACCENT } else { CARD_HI });
            p.rect_stroke(btn_rect, 6, Stroke::new(1.0, if paste_hover { ACCENT_GLOW } else { LINE }), StrokeKind::Inside);
            let btn_label = if self.mode == StudioMode::Video {
                "Paste Image for Image-to-Video (Ctrl+V)"
            } else {
                "Paste Image from Clipboard (Ctrl+V)"
            };
            p.text(btn_rect.center(), egui::Align2::CENTER_CENTER, btn_label, egui::FontId::proportional(12.5), Color32::WHITE);

            if paste_hover && ui.ctx().input(|i| i.pointer.any_pressed()) {
                let _ = self.paste_from_clipboard(ctx);
            } else if resp.clicked() {
                self.pick_images(ctx, Target::Base, false);
            }
        }

        self.render_floating_generation_status(ctx, ui.max_rect());
    }

    fn render_floating_generation_status(&mut self, ctx: &egui::Context, anchor_rect: Rect) {
        if self.generating {
            let status_txt = if self.is_swapping_detected {
                if let Some((v, max)) = self.progress {
                    format!("{}/{} • VRAM paging active (~{:.1}s/step)", v, max, self.step_duration_ema)
                } else {
                    "VRAM paging active".to_string()
                }
            } else if self.status.contains("diffusion") || self.status.contains("Generating...") {
                if let Some((v, max)) = self.progress {
                    format!("{}/{} • {}", v, max, self.status)
                } else {
                    self.status.clone()
                }
            } else {
                self.status.clone()
            };
            let status_col = if self.is_swapping_detected { AMBER } else { TEXT };

            let margin = 14.0;
            egui::Area::new(egui::Id::new("floating_canvas_generation_status"))
                .fixed_pos(pos2(anchor_rect.right() - margin, anchor_rect.bottom() - margin))
                .pivot(egui::Align2::RIGHT_BOTTOM)
                .order(egui::Order::Foreground)
                .interactable(true)
                .show(ctx, |ui| {
                    egui::Frame::new()
                        .fill(CARD_HI)
                        .stroke(Stroke::new(1.0, if self.is_swapping_detected { AMBER } else { ACCENT }))
                        .corner_radius(8)
                        .inner_margin(egui::Margin::symmetric(14, 9))
                        .shadow(egui::epaint::Shadow {
                            offset: [0, 2],
                            blur: 10,
                            spread: 0,
                            color: Color32::from_black_alpha(150),
                        })
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label(RichText::new(status_txt).size(12.0).color(status_col).strong());
                                ui.add_space(8.0);
                                let cancel_btn = egui::Button::new(RichText::new("Cancel").size(12.0).strong().color(RED))
                                    .fill(CARD)
                                    .stroke(Stroke::new(1.0, RED.linear_multiply(0.45)))
                                    .corner_radius(6)
                                    .min_size(vec2(66.0, 24.0));
                                if ui.add(cancel_btn).clicked() {
                                    self.cancel.store(true, Ordering::SeqCst);
                                    self.queue.clear();
                                    self.generation_retries = 0;
                                    self.status = "Cancelling...".into();
                                    if self.prompt.trim().is_empty() && !self.generating_prompt.trim().is_empty() {
                                        self.prompt = crate::lora::strip_prompt_tags(self.active_prompt_tags(), &self.generating_prompt);
                                    }
                                    std::thread::spawn(move || {
                                        let client = reqwest::blocking::Client::builder()
                                            .timeout(Duration::from_secs(2))
                                            .build();
                                        if let Ok(c) = client {
                                            let base = engine::base_url();
                                            let _ = c.post(format!("{base}/interrupt")).send();
                                            let _ = c.post(format!("{base}/queue")).json(&serde_json::json!({"clear": true})).send();
                                        }
                                    });
                                }
                            });
                            if self.stall_prompt.load(Ordering::Relaxed) {
                                ui.add_space(4.0);
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new("Slow step (>3m, VRAM swapping) — auto-cancels after 20m")
                                            .size(10.5)
                                            .color(AMBER),
                                    );
                                    let keep_btn = egui::Button::new(RichText::new("Continue Waiting").size(11.0).strong().color(DISCORD_GREEN))
                                        .fill(CARD)
                                        .stroke(Stroke::new(1.0, DISCORD_GREEN.linear_multiply(0.5)))
                                        .corner_radius(6);
                                    if ui.add(keep_btn).clicked() {
                                        self.stall_keep_waiting.store(true, Ordering::SeqCst);
                                        self.stall_prompt.store(false, Ordering::SeqCst);
                                    }
                                });
                            }
                        });
                });
        }
    }

    // Bottom Docked ChatGPT-style Prompt & Action Bar
    fn bottom_dock(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let err_clone = self.error.clone();
        if let Some(err) = err_clone {
            let mut dismiss = false;
            ui.horizontal_wrapped(|ui| {
                ui.add(egui::Label::new(RichText::new(format!("Error: {err}")).color(RED).size(12.0)).selectable(true));
                if ui.small_button("Copy").on_hover_text("Copy error message to clipboard").clicked() {
                    if let Ok(mut cb) = arboard::Clipboard::new() {
                        let _ = cb.set_text(err.clone());
                    }
                }
                if ui.small_button("Dismiss").clicked() {
                    dismiss = true;
                }
            });
            if dismiss {
                self.error = None;
            }
            ui.add_space(2.0);
        }

        let st = self.sh.state.lock().unwrap().clone();
        let is_ready = matches!(st, EngState::Ready { .. });
        if !is_ready && !self.queue.is_empty() {
            let attempt_str = match &st {
                EngState::DownloadingModels { file_index, total_files, file_progress, .. } => {
                    format!("Downloading models ({}/{}, {:.0}%) — queued tasks will run once ready...", file_index, total_files, file_progress * 100.0)
                }
                EngState::DownloadingEngine { step, .. } => {
                    format!("{step} — queued tasks will run once ready...")
                }
                EngState::ExtractingEngine { step } => {
                    format!("{step} — queued tasks will run once ready...")
                }
                EngState::ModelDownloadPaused { .. } => {
                    "Download paused — queued tasks waiting...".to_string()
                }
                EngState::Starting { since, attempt } => {
                    format!("Starting engine in background (attempt {}, {}s)...", attempt + 1, since.elapsed().as_secs())
                }
                _ => "Preparing engine in background...".to_string(),
            };
            card(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(RichText::new(attempt_str).size(11.5).color(AMBER));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!("{} prompt queued", self.queue.len())).size(11.0).color(GREEN).strong());
                    });
                });
            });
            ui.add_space(3.0);
        }

        egui::Frame::new()
            .fill(CARD_HI)
            .stroke(Stroke::new(1.0, LINE))
            .corner_radius(10)
            .inner_margin(egui::Margin::symmetric(12, 7))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let paste_icon_btn = egui::Button::new(RichText::new("Paste").size(12.0).color(TEXT))
                        .fill(CARD)
                        .stroke(Stroke::new(1.0, LINE))
                        .corner_radius(6)
                        .min_size(vec2(64.0, 36.0));
                    if ui.add(paste_icon_btn).on_hover_text("Paste image from clipboard (Ctrl+V)").clicked() {
                        let _ = self.paste_from_clipboard(ctx);
                    }
                    ui.add_space(4.0);

                    let prompt_id = egui::Id::new("main_prompt_text_edit");
                    let hint = if self.mode == StudioMode::Video {
                        if self.is_edit_video_mode() {
                            "Optional: add extra details for this edit... Enter to edit"
                        } else if self.video_player.is_some() {
                            "Describe motion, camera action, or animation from this frame... Enter to generate"
                        } else if self.active_node().is_some() {
                            "Describe motion, camera action, or animation for this image... Enter to generate"
                        } else {
                            "Describe a video scene, camera motion, and action... Enter to generate"
                        }
                    } else if self.active_node().is_some() {
                        "Describe your edit instructions... Enter to generate, Shift+Enter for newline"
                    } else {
                        "Describe an image to generate from scratch, or paste/drop an image..."
                    };
                    let st = self.sh.state.lock().unwrap().clone();
                    let is_ready = matches!(st, EngState::Ready { .. });
                    let fallback_info: Option<(String, String)> = if self.mode == StudioMode::Video && !self.generating && is_ready {
                        let active_vprof = self.sh.active_video_profile.lock().unwrap().clone();
                        self.effective_video_profile().and_then(|eff| {
                            if eff.id != active_vprof.id {
                                let short = if eff.is_minimax_h3() {
                                    "MiniMax H3"
                                } else if eff.is_ltx25() {
                                    "LTX 2.5"
                                } else {
                                    "LTX 2.3"
                                };
                                let vdir = self.active_video_models_dir();
                                let tip = if !vdir.as_os_str().is_empty() && active_vprof.missing_in_dir(&vdir).is_empty() {
                                    format!(
                                        "Using {} for video edit / LTX LoRA (active model {} is ready).",
                                        eff.name, active_vprof.name
                                    )
                                } else {
                                    format!(
                                        "Using {} (best available downloaded model) while {} finishes downloading.",
                                        eff.name, active_vprof.name
                                    )
                                };
                                Some((format!("Using fallback {short}"), tip))
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    };
                    let controls_w = if fallback_info.is_some() { 304.0 } else { 266.0 };
                    let prompt_w = (ui.available_width() - controls_w).max(80.0);
                    let text_resp = ui.vertical(|ui| {
                        ui.set_max_width(prompt_w);
                        if self.mode == StudioMode::Video {
                            crate::lora::render_prompt_tags_bar(ui, ctx, &mut self.video_prompt_tags);
                        } else {
                            crate::lora::render_prompt_tags_bar(ui, ctx, &mut self.image_prompt_tags);
                        }
                        let text_edit = egui::TextEdit::multiline(&mut self.prompt)
                            .id(prompt_id)
                            .desired_rows(2)
                            .desired_width(prompt_w)
                            .hint_text(hint);
                        ui.add(text_edit)
                    }).inner;

                    if self.focus_prompt {
                        self.focus_prompt = false;
                        text_resp.request_focus();
                        let char_count = self.prompt.chars().count();
                        let range = egui::text::CCursorRange::two(
                            egui::text::CCursor::new(0),
                            egui::text::CCursor::new(char_count),
                        );
                        let mut state = egui::text_edit::TextEditState::load(ctx, prompt_id).unwrap_or_default();
                        state.cursor.set_char_range(Some(range));
                        state.store(ctx, prompt_id);
                    }

                    let should_queue = self.generating || !is_ready;

                    // Submit on Enter (without Shift)
                    let enter_submit = text_resp.has_focus()
                        && ctx.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                    if enter_submit {
                        // egui inserts a newline at the caret on Enter (and advances the caret past
                        // it) regardless of caret position: text end, mid-text, or over a selection.
                        // So the Enter newline is NOT necessarily trailing; it lands where the caret
                        // was. Remove exactly that inserted newline at caret-1. A real newline typed
                        // with Shift+Enter never reaches this branch (enter_submit requires no shift).
                        let caret = egui::text_edit::TextEditState::load(ctx, prompt_id)
                            .and_then(|s| s.cursor.char_range())
                            .map(|r| r.primary.index.0)
                            .unwrap_or(0);
                        if caret >= 1 {
                            let i = caret - 1;
                            let inserted = self.prompt.chars().nth(i);
                            if inserted == Some('\n') || inserted == Some('\r') {
                                self.prompt = self.prompt.chars()
                                    .enumerate()
                                    .filter(|(k, _)| *k != i)
                                    .map(|(_, c)| c)
                                    .collect::<String>();
                            }
                        }
                        let p_text = self.composed_prompt();
                        if !p_text.is_empty() || self.is_edit_video_mode() {
                            if should_queue {
                                if self.can_queue().is_ok() {
                                    self.queue.push(QueuedItem { prompt: p_text });
                                    self.prompt.clear();
                                    if !is_ready {
                                        self.status = "Queued (will start automatically once engine is ready)".into();
                                    }
                                }
                            } else if self.can_generate().is_ok() {
                                self.start_generation(ctx);
                            }
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {

                        let edit_mode = self.is_edit_video_mode();
                        let (is_ok, btn_text, tip) = if self.generating {
                            let q_can = self.can_queue();
                            let ok = q_can.is_ok() && (self.has_prompt_content() || edit_mode);
                            (ok, "Queue", q_can.err().map(|s| s.to_string()))
                        } else if !is_ready {
                            let q_can = self.can_queue();
                            let ok = q_can.is_ok() && (self.has_prompt_content() || edit_mode);
                            (ok, "Queue", Some("Engine setup in progress; prompt will be queued and will generate automatically once ready.".to_string()))
                        } else {
                            let g_can = self.can_generate();
                            let ok = g_can.is_ok() && (self.has_prompt_content() || edit_mode);
                            let fallback_tip = fallback_info.as_ref().map(|(_, t)| t.clone());
                            let label = if edit_mode { "Edit Video" } else { "Generate" };
                            (ok, label, g_can.err().map(|s| s.to_string()).or(fallback_tip))
                        };

                        let btn_w = if fallback_info.is_some() { 138.0 } else { 100.0 };
                        let btn_h = 38.0;
                        let sense = if is_ok { Sense::click() } else { Sense::hover() };
                        let (btn_rect, resp) = ui.allocate_exact_size(vec2(btn_w, btn_h), sense);
                        if is_ok && resp.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        let btn_fill = if !is_ok {
                            CARD
                        } else if resp.is_pointer_button_down_on() {
                            Color32::from_rgb(79, 70, 229)
                        } else if resp.hovered() {
                            Color32::from_rgb(112, 115, 248)
                        } else {
                            ACCENT
                        };
                        let btn_stroke = Stroke::new(1.0, if is_ok { ACCENT_GLOW } else { LINE });
                        let bp = ui.painter();
                        bp.rect_filled(btn_rect, 8.0, btn_fill);
                        bp.rect_stroke(btn_rect, 8.0, btn_stroke, StrokeKind::Inside);

                        let title_color = if is_ok { Color32::WHITE } else { DIM };
                        if let Some((ref sub, ref fallback_tip)) = fallback_info {
                            bp.text(
                                Pos2::new(btn_rect.center().x, btn_rect.center().y - 6.0),
                                egui::Align2::CENTER_CENTER,
                                btn_text,
                                egui::FontId::proportional(13.0),
                                title_color,
                            );
                            bp.text(
                                Pos2::new(btn_rect.center().x, btn_rect.center().y + 8.5),
                                egui::Align2::CENTER_CENTER,
                                sub,
                                egui::FontId::proportional(9.5),
                                if is_ok { Color32::from_rgb(224, 231, 255) } else { DIM },
                            );

                            // Show fallback tooltip immediately on mouseover without hover delay
                            if resp.hovered() {
                                egui::Area::new(egui::Id::new("fallback_generate_tooltip"))
                                    .order(egui::Order::Tooltip)
                                    .interactable(false)
                                    .constrain(true)
                                    .fixed_pos(Pos2::new(btn_rect.center().x, btn_rect.top() - 6.0))
                                    .pivot(egui::Align2::CENTER_BOTTOM)
                                    .show(ctx, |ui| {
                                        egui::Frame::new()
                                            .fill(Color32::from_rgb(24, 28, 40))
                                            .stroke(Stroke::new(1.0, ACCENT_GLOW))
                                            .corner_radius(6)
                                            .inner_margin(egui::Margin::symmetric(10, 6))
                                            .show(ui, |ui| {
                                                ui.set_max_width(280.0);
                                                ui.label(RichText::new(fallback_tip).size(11.0).color(TEXT));
                                            });
                                    });
                            }
                        } else {
                            bp.text(
                                btn_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                btn_text,
                                egui::FontId::proportional(13.5),
                                title_color,
                            );
                        }

                        if is_ok && resp.clicked() {
                            let p_text = self.composed_prompt();
                            if !p_text.is_empty() || edit_mode {
                                if should_queue {
                                    if self.can_queue().is_ok() {
                                        self.queue.push(QueuedItem { prompt: p_text });
                                        self.prompt.clear();
                                        if !is_ready {
                                            self.status = "Queued (will start automatically once engine is ready)".into();
                                        }
                                    }
                                } else if self.can_generate().is_ok() {
                                    self.start_generation(ctx);
                                }
                            }
                        }
                        if fallback_info.is_none() {
                            if let Some(why) = tip {
                                resp.on_hover_text(why);
                            }
                        }

                        // Segmented pill buttons [ Image | Video ] placed to the LEFT of the generate button:
                        ui.add_space(8.0);
                        let is_img = self.mode == StudioMode::Image;
                        let is_vid = self.mode == StudioMode::Video;

                        egui::Frame::new()
                            .fill(CARD)
                            .stroke(Stroke::new(1.0, LINE))
                            .corner_radius(8)
                            .inner_margin(egui::Margin::symmetric(3, 2))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing = vec2(2.0, 0.0);
                                    let img_btn = egui::Button::new(
                                        RichText::new("Image")
                                            .size(12.0)
                                            .strong()
                                            .color(if is_img { Color32::WHITE } else { DIM }),
                                    )
                                    .fill(if is_img { ACCENT } else { Color32::TRANSPARENT })
                                    .corner_radius(6)
                                    .min_size(vec2(64.0, 28.0));
                                    if ui.add(img_btn).on_hover_text("Switch to Image Editing (Qwen Image 2.1)").clicked() {
                                        self.mode = StudioMode::Image;
                                    }

                                    let vid_btn = egui::Button::new(
                                        RichText::new("Video")
                                            .size(12.0)
                                            .strong()
                                            .color(if is_vid { Color32::WHITE } else { DIM }),
                                    )
                                    .fill(if is_vid { ACCENT } else { Color32::TRANSPARENT })
                                    .corner_radius(6)
                                    .min_size(vec2(64.0, 28.0));
                                    let vid_hover = {
                                        let vp = self.sh.active_video_profile.lock().unwrap();
                                        if vp.is_minimax_h3() {
                                            "Switch to Video Generation (MiniMax H3)"
                                        } else if vp.is_ltx25() {
                                            "Switch to Video Generation (LTX-Video 2.5)"
                                        } else {
                                            "Switch to Video Generation (LTX-Video 2.3)"
                                        }
                                    };
                                    if ui.add(vid_btn).on_hover_text(vid_hover).clicked() {
                                        self.mode = StudioMode::Video;
                                        if !self.has_video_models() && !self.sh.dl_video_model.load(Ordering::SeqCst) {
                                            self.show_video_dl_modal = true;
                                        }
                                    }
                                });
                            });

                        if self.generating && !self.queue.is_empty() {
                            ui.add_space(6.0);
                            let queue_btn = egui::Button::new(
                                    RichText::new(format!("{} queued", self.queue.len()))
                                        .size(11.5)
                                        .color(AMBER)
                                        .strong(),
                                )
                                .fill(CARD)
                                .stroke(Stroke::new(1.0, AMBER.linear_multiply(0.4)))
                                .corner_radius(6);
                                let mut tip = String::from("Queued in chain:\n");
                                for (idx, q) in self.queue.iter().enumerate() {
                                    tip.push_str(&format!("{}. {}\n", idx + 1, q.prompt));
                                }
                                tip.push_str("\nClick to clear queue");
                                if ui.add(queue_btn).on_hover_text(tip).clicked() {
                                    self.queue.clear();
                                }
                                ui.add_space(4.0);
                            }
                    });
                });

                // Neat progress bar at bottom of card (always occupies 3.5px height to prevent layout bump)
                ui.add_space(4.0);
                let (bar_rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 3.5), egui::Sense::hover());
                let p = ui.painter_at(bar_rect);
                if self.generating {
                    p.rect_filled(bar_rect, 2, Color32::from_rgb(26, 30, 42));
                    if let Some((v, max)) = self.progress {
                        let frac = (v as f32 / max.max(1) as f32).clamp(0.0, 1.0);
                        let fill_rect = Rect::from_min_size(bar_rect.min, vec2(bar_rect.width() * frac, bar_rect.height()));
                        p.rect_filled(fill_rect, 2, ACCENT);
                    } else {
                        let t = ctx.input(|i| i.time as f32);
                        let pulse = (t * 1.5).fract();
                        let seg_w = bar_rect.width() * 0.28;
                        let x0 = bar_rect.left() + (bar_rect.width() + seg_w) * pulse - seg_w;
                        let fill_rect = Rect::from_min_max(
                            pos2(x0.max(bar_rect.left()), bar_rect.top()),
                            pos2((x0 + seg_w).min(bar_rect.right()), bar_rect.bottom()),
                        );
                        p.rect_filled(fill_rect, 2, ACCENT);
                        ctx.request_repaint();
                    }
                } else {
                    p.rect_filled(bar_rect, 2, Color32::TRANSPARENT);
                }
            });
    }

    fn apply_image_profile(&mut self, p: &Profile, ctx: &egui::Context) {
        let mut cfg = self.sh.cfg.lock().unwrap();
        cfg.active_profile = Some(p.id.clone());
        cfg.save();
        drop(cfg);
        *self.sh.active_profile.lock().unwrap() = p.clone();
        self.profile = p.clone();
        let models_dir = self.sh.paths.lock().unwrap().models.clone();
        if models_dir.as_os_str().is_empty() || !p.missing_in_dir(&models_dir).is_empty() {
            engine::restart_model_downloads_for_profile_change(&self.sh, ctx);
        }
    }

    fn sync_effective_video_profile(&mut self) {
        let eff = self
            .effective_video_profile()
            .unwrap_or_else(|| self.sh.active_video_profile.lock().unwrap().clone());
        if eff.id != self.last_effective_video_profile_id {
            let prev_is_h3 = self.last_effective_video_profile_id.starts_with("minimax_h3");
            let prev_is_ltx25 = self.last_effective_video_profile_id.starts_with("ltx_2.5");
            if prev_is_h3 != eff.is_minimax_h3() || prev_is_ltx25 != eff.is_ltx25() {
                let eff_family = crate::lora::LoraModelFamily::from_video_profile(&eff);
                let items = &self.lora_browser.items;
                self.video_loras.retain(|active| {
                    items
                        .iter()
                        .find(|it| it.filename == active.filename)
                        .map(|it| it.is_compatible_with(eff_family))
                        .unwrap_or(true)
                });
                self.video_prompt_tags.retain(|t| {
                    self.video_loras.iter().any(|l| l.filename == t.lora_filename)
                });
            }
            self.video_steps = eff.default_steps.max(4);
            if !eff.sampler.is_empty() {
                self.video_sampler = eff.sampler.clone();
            }
            if !eff.scheduler.is_empty() {
                self.video_scheduler = eff.scheduler.clone();
            }
            self.last_effective_video_profile_id = eff.id;
        }
    }

    fn apply_video_profile(&mut self, vp: &Profile, ctx: &egui::Context) {
        if !vp.is_minimax_h3() {
            self.last_ltx_video_profile_id = Some(vp.id.clone());
        }
        {
            let mut cfg = self.sh.cfg.lock().unwrap();
            cfg.active_video_profile = Some(vp.id.clone());
            cfg.save();
        }
        *self.sh.active_video_profile.lock().unwrap() = vp.clone();
        self.sync_effective_video_profile();
        let models_dir = self.active_video_models_dir();
        if models_dir.as_os_str().is_empty() || !vp.missing_in_dir(&models_dir).is_empty() {
            engine::restart_model_downloads_for_profile_change(&self.sh, ctx);
        }
    }

    fn handle_models_deleted(&mut self) {
        let models_dir = self.sh.paths.lock().unwrap().models.clone();
        if models_dir.as_os_str().is_empty() {
            return;
        }
        let vram = engine::detect_gpu_vram_gb();

        // Fallback for image profile only if the currently active image profile was deleted
        if !self.profile.missing_in_dir(&models_dir).is_empty() {
            let all_img = Profile::all();
            let rec_id = crate::model_picker::recommended_image_profile_id(vram);
            let fallback = all_img
                .iter()
                .find(|p| p.id == rec_id && p.missing_in_dir(&models_dir).is_empty())
                .or_else(|| all_img.iter().find(|p| p.missing_in_dir(&models_dir).is_empty()))
                .cloned();
            if let Some(p) = fallback {
                let mut cfg = self.sh.cfg.lock().unwrap();
                cfg.active_profile = Some(p.id.clone());
                cfg.save();
                drop(cfg);
                *self.sh.active_profile.lock().unwrap() = p.clone();
                self.profile = p;
            }
        }

        // Fallback for video profile only if the currently active video profile was deleted
        let active_vid = self.sh.active_video_profile.lock().unwrap().clone();
        if !active_vid.missing_in_dir(&models_dir).is_empty() {
            let vdir = self.active_video_models_dir();
            if !vdir.as_os_str().is_empty() {
                if let Some(vp) = Profile::find_downloaded_video_profile(&vdir) {
                    {
                        let mut cfg = self.sh.cfg.lock().unwrap();
                        cfg.active_video_profile = Some(vp.id.clone());
                        cfg.save();
                    }
                    *self.sh.active_video_profile.lock().unwrap() = vp.clone();
                    self.sync_effective_video_profile();
                    self.model_picker.record_active_video(&vp);
                }
            }
        }
    }

    fn handle_picker_action(&mut self, action: crate::model_picker::PickerAction, ctx: &egui::Context) {
        match action {
            crate::model_picker::PickerAction::SwitchImage(p) => {
                self.apply_image_profile(&p, ctx);
            }
            crate::model_picker::PickerAction::SwitchVideo(vp) => {
                self.apply_video_profile(&vp, ctx);
            }
            crate::model_picker::PickerAction::DownloadProfile(p) => {
                engine::start_profile_download(&self.sh, &p, ctx);
            }
            crate::model_picker::PickerAction::ToggleDownloadPause(pid) => {
                engine::toggle_profile_download_pause(&self.sh, &pid, ctx);
            }
            crate::model_picker::PickerAction::ModelsDeleted => {
                self.handle_models_deleted();
            }
            crate::model_picker::PickerAction::None => {}
        }
    }

    fn setup_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_setup;
        egui::Window::new("Setup & diagnostics")
            .open(&mut open)
            .default_width(700.0)
            .min_width(620.0)
            .default_height(620.0)
            .max_height(ctx.content_rect().height() - 60.0)
            .resizable(true)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("setup_window_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let paths = self.sh.paths.lock().unwrap().clone();
                        let state = self.sh.state.lock().unwrap().clone();
                        let st = match &state {
                            EngState::Detecting => "Detecting engine...".to_string(),
                            EngState::AwaitingDownloadConfirmation { .. } => "Download confirmation required (metered connection)".to_string(),
                            EngState::DownloadingEngine { step, progress, bytes_done, bytes_total, speed_mb } => {
                                format!("{step} ({:.1}%) - {} / {} @ {:.1} MB/s", progress * 100.0, engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total), speed_mb)
                            }
                            EngState::ExtractingEngine { step } => step.clone(),
                            EngState::RepairingEngine { step } => format!("Repairing: {step}"),
                            EngState::DownloadingModels { current_file, file_index, total_files, file_progress, bytes_done, bytes_total, speed_mb, .. } => {
                                format!("Downloading Model [{}/{}] {} ({:.1}%) - {} / {} @ {:.1} MB/s", file_index, total_files, current_file, file_progress * 100.0, engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total), speed_mb)
                            }
                            EngState::ModelDownloadPaused { current_file, file_index, total_files, file_progress, bytes_done, bytes_total, .. } => {
                                format!("Model Download Paused [{}/{}] {} ({:.1}%) - {} / {}", file_index, total_files, current_file, file_progress * 100.0, engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total))
                            }
                            EngState::Starting { since, attempt } => format!("Starting (attempt {}, {}s)", attempt + 1, since.elapsed().as_secs()),
                            EngState::Ready { ours, gpu } => {
                                format!("Ready on {gpu} ({})", if *ours { "launched by Image Studio" } else { "already running" })
                            }
                            EngState::ModelDownloadFailed { current_file, error } => format!("Model download failed ({current_file}): {error}"),
                            EngState::Failed(m) => format!("Failed: {m}"),
                        };
                        ui.label(RichText::new(st).strong());
                        ui.separator();

                        heading(ui, "SOFTWARE UPDATES");
                        let cur_ver = env!("CARGO_PKG_VERSION");
                        let (up_avail, latest, is_dl, dl_prog, applied, err) = {
                            let st = self.updater.lock().unwrap();
                            (st.update_available, st.latest_version.clone(), st.downloading, st.download_progress, st.update_applied, st.error.clone())
                        };

                        card(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(format!("Installed version: v{cur_ver}")).size(12.0).color(TEXT));
                                if up_avail {
                                    ui.label(RichText::new(format!("* Update available: v{latest}")).size(12.0).color(DISCORD_GREEN).strong());
                                } else if !latest.is_empty() {
                                    ui.label(RichText::new("(Up to date)").size(12.0).color(DIM));
                                }
                            });

                            if let Some(ref e) = err {
                                ui.label(RichText::new(format!("Update check: {e}")).size(11.5).color(RED));
                            }

                            if applied {
                                ui.label(RichText::new("Update installed successfully! Restart Image Studio to use the new version.").color(DISCORD_GREEN).strong());
                                if ui.button("Restart Image Studio Now").clicked() {
                                    crate::updater::restart_application();
                                }
                            } else if is_dl {
                                ui.label(RichText::new(format!("Downloading update: {:.0}%", dl_prog * 100.0)).size(12.0).color(ACCENT));
                                let bar = egui::ProgressBar::new(dl_prog).show_percentage().fill(ACCENT);
                                ui.add_sized([ui.available_width() - 20.0, 14.0], bar);
                            } else if up_avail {
                                let btn = egui::Button::new(RichText::new(format!("Download & Install Update (v{latest})")).size(12.0).strong().color(Color32::WHITE))
                                    .fill(DISCORD_GREEN);
                                if ui.add(btn).clicked() {
                                    crate::updater::download_and_install_update(self.updater.clone(), ctx.clone());
                                }
                            } else {
                                if ui.button("Check for Updates").clicked() {
                                    crate::updater::check_for_updates(self.updater.clone(), ctx.clone());
                                }
                            }
                        });

                        ui.separator();
                        heading(ui, "HARDWARE & MODEL PROFILE");
                        let vram_detected = engine::detect_gpu_vram_gb();
                        ui.horizontal(|ui| {
                            ui.label("Detected GPU VRAM:");
                            if let Some(vram) = vram_detected {
                                ui.label(RichText::new(format!("~{vram} GB")).strong().color(GREEN));
                                if vram <= 8 {
                                    ui.label(RichText::new("(8GB GPU detected - 8GB W4A8 image & LTX-2.5 recommended)").size(11.0).color(AMBER));
                                }
                            } else {
                                ui.label(RichText::new("Auto-detecting...").color(DIM));
                            }
                        });

                        let cur_img_prof = self.profile.clone();
                        let cur_vid_prof = self.sh.active_video_profile.lock().unwrap().clone();
                        let picker_action = crate::model_picker::render_model_picker(
                            ui,
                            &mut self.model_picker,
                            &self.sh,
                            &cur_img_prof,
                            &cur_vid_prof,
                            vram_detected,
                            &paths.models,
                        );
                        self.handle_picker_action(picker_action, ctx);

                        let active_vid_after = self.sh.active_video_profile.lock().unwrap().clone();
                        if let Some(gpu_vram) = vram_detected {
                            if active_vid_after.recommended_vram_gb > gpu_vram
                                && !crate::model_picker::is_allowed_step_up_variant(&active_vid_after.id, vram_detected)
                                && active_vid_after.id != crate::model_picker::recommended_video_profile_id(
                                    if active_vid_after.is_minimax_h3() {
                                        crate::model_picker::VideoFamily::MiniMaxH3
                                    } else {
                                        crate::model_picker::VideoFamily::LtxVideo
                                    },
                                    vram_detected,
                                )
                            {
                                ui.add_space(4.0);
                                egui::Frame::new()
                                    .fill(Color32::from_rgb(42, 22, 18))
                                    .stroke(Stroke::new(1.0, Color32::from_rgb(220, 90, 40)))
                                    .corner_radius(6)
                                    .inner_margin(egui::Margin::same(8))
                                    .show(ui, |ui| {
                                        ui.label(
                                            RichText::new(format!(
                                                "[!] VRAM Warning: '{}' recommends {} GB VRAM (detected: {} GB). Running it on this GPU will rely heavily on shared system RAM paging, causing 10x-100x slower generation or CUDA Out-Of-Memory (OOM) errors.",
                                                active_vid_after.name, active_vid_after.recommended_vram_gb, gpu_vram
                                            ))
                                            .size(11.0)
                                            .color(AMBER),
                                        );
                                    });
                            }
                        }

                        ui.separator();
                        heading(ui, "DIRECTORIES & PATHS");
                        egui::Grid::new("paths").num_columns(3).spacing([12.0, 8.0]).show(ui, |ui| {
                            ui.label("ComfyUI");
                            ui.label(RichText::new(paths.comfy.as_ref().map(|p| p.display().to_string()).unwrap_or("-".into())).color(DIM));
                            if ui.button("Browse...").clicked() {
                                if let Some(d) = rfd::FileDialog::new().pick_folder() {
                                    let mut c = self.sh.cfg.lock().unwrap();
                                    c.comfy_dir = Some(d);
                                    c.save();
                                    self.sh.restart.store(true, Ordering::SeqCst);
                                }
                            }
                            ui.end_row();

                            ui.label("Python");
                            ui.label(RichText::new(paths.python.as_ref().map(|p| p.display().to_string()).unwrap_or("-".into())).color(DIM));
                            if ui.button("Browse...").clicked() {
                                if let Some(f) = rfd::FileDialog::new().pick_file() {
                                    let mut c = self.sh.cfg.lock().unwrap();
                                    c.python_path = Some(f);
                                    c.save();
                                    self.sh.restart.store(true, Ordering::SeqCst);
                                }
                            }
                            ui.end_row();

                            ui.label("Models");
                            ui.label(RichText::new(paths.models.display().to_string()).color(DIM));
                            if ui.button("Browse...").clicked() {
                                if let Some(d) = rfd::FileDialog::new().pick_folder() {
                                    let mut c = self.sh.cfg.lock().unwrap();
                                    c.models_dir = Some(d);
                                    c.save();
                                }
                            }
                            ui.end_row();
                        });

                        ui.separator();
                        heading(ui, "MODEL DOWNLOAD SELECTION");
                        self.render_model_checkboxes(ui);
                        ui.add_space(4.0);

                        ui.separator();
                        heading(ui, "WORKSPACE & BEHAVIOR");
                        {
                            let mut auto_switch = self.sh.cfg.lock().unwrap().auto_switch_mode_on_input;
                            if ui
                                .checkbox(&mut auto_switch, "Auto-switch to Image or Video mode depending on input")
                                .on_hover_text("When disabled, loading a video in Image mode keeps you in Image mode to edit frames, and loading an image in Video mode keeps you in Video mode for I2V.")
                                .changed()
                            {
                                let mut c = self.sh.cfg.lock().unwrap();
                                c.auto_switch_mode_on_input = auto_switch;
                                c.save();
                            }

                            let mut remember_mode = self.sh.cfg.lock().unwrap().remember_last_mode;
                            if ui
                                .checkbox(&mut remember_mode, "Remember last Image/Video mode on startup")
                                .on_hover_text("Re-open the app in whichever mode (Image or Video) it was closed in.")
                                .changed()
                            {
                                let mut c = self.sh.cfg.lock().unwrap();
                                c.remember_last_mode = remember_mode;
                                if remember_mode {
                                    c.last_studio_mode = Some(
                                        if self.mode == StudioMode::Video { "video" } else { "image" }.to_string(),
                                    );
                                } else {
                                    c.last_studio_mode = None;
                                }
                                c.save();
                            }

                            let mut keep_project = self.sh.cfg.lock().unwrap().keep_last_project_loaded;
                            if ui
                                .checkbox(&mut keep_project, "Keep last project loaded")
                                .on_hover_text("Automatically save and restore your loaded inputs, history, references, and scrubbed video state when closing and reopening the app.")
                                .changed()
                            {
                                let mut c = self.sh.cfg.lock().unwrap();
                                c.keep_last_project_loaded = keep_project;
                                c.save();
                                if !keep_project {
                                    Self::clear_last_project();
                                }
                            }
                        }
                        ui.add_space(4.0);

                        ui.separator();
                        heading(ui, "VIDEO LORA SETTINGS");
                        let mut allow_multi_vid = self.sh.cfg.lock().unwrap().allow_multi_video_loras;
                        if ui
                            .checkbox(&mut allow_multi_vid, "Allow multiple video LoRAs simultaneously")
                            .changed()
                        {
                            let mut c = self.sh.cfg.lock().unwrap();
                            c.allow_multi_video_loras = allow_multi_vid;
                            c.save();
                            if !allow_multi_vid && self.video_loras.len() > 1 {
                                self.video_loras.truncate(1);
                                let keep = self.video_loras[0].filename.clone();
                                self.video_prompt_tags.retain(|t| t.lora_filename == keep);
                            }
                        }
                        if allow_multi_vid {
                            ui.label(
                                RichText::new("[!] Warning: Stacking multiple video LoRAs increases VRAM usage, slows generation, and may degrade quality or cause OOM. Stacking is still blocked if estimated VRAM would approach your GPU limit.")
                                    .size(11.0)
                                    .color(AMBER),
                            );
                        }
                        ui.add_space(4.0);

                        ui.separator();
                        heading(ui, "MODEL FILES STATUS");
                        let vmodels_dir = self.active_video_models_dir();
                        let active_vid = self.sh.active_video_profile.lock().unwrap().clone();
                        ui.label(RichText::new(format!("Video Profile: {}", active_vid.name)).size(11.5).strong().color(ACCENT_GLOW));
                        for vm in &active_vid.models {
                            let ok = !vmodels_dir.as_os_str().is_empty()
                                && engine::safe_model_path(&vmodels_dir, &vm.category, &vm.filename).exists();
                            ui.label(
                                RichText::new(format!("{}  {}", if ok { "found  " } else { "MISSING" }, vm.filename))
                                    .color(if ok { GREEN } else { RED }),
                            );
                        }
                        ui.add_space(4.0);
                        ui.label(RichText::new(format!("Image Profile: {}", self.profile.name)).size(11.5).strong().color(ACCENT_GLOW));
                        for (name, ok) in self.sh.models_found.lock().unwrap().iter() {
                            ui.label(
                                RichText::new(format!("{}  {}", if *ok { "found  " } else { "MISSING" }, name))
                                    .color(if *ok { GREEN } else { RED }),
                            );
                        }
                        ui.horizontal(|ui| {
                            if ui.button("Open Models Folder").clicked() {
                                engine::open_models_folder(&self.sh);
                            }
                            if ui.button("Retry Setup / Restart Engine").clicked() {
                                self.sh.retry.store(true, Ordering::SeqCst);
                                self.sh.restart.store(true, Ordering::SeqCst);
                            }
                            if ui.button("Repair Python Dependencies").on_hover_text("Re-installs missing Python packages (PyTorch, comfy-aimdo, requirements.txt) into the engine environment").clicked() {
                                engine::repair_environment(self.sh.clone(), ui.ctx().clone());
                            }
                            if ui.button("Switch to Clean Managed Engine").on_hover_text("Bypasses custom/old ComfyUI folders and downloads the official portable engine").clicked() {
                                engine::reset_to_clean_engine(&self.sh);
                            }
                        });

                        ui.separator();
                        heading(ui, "COMFYUI LOG (tail)");
                        egui::ScrollArea::vertical()
                            .id_salt("l1")
                            .max_height(160.0)
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(engine::tail(&engine::data_dir().join("comfyui.log"), 40))
                                            .monospace()
                                            .size(12.0),
                                    )
                                    .selectable(true),
                                );
                            });

                        heading(ui, "APP LOG (tail)");
                        egui::ScrollArea::vertical()
                            .id_salt("l2")
                            .max_height(100.0)
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(engine::tail(&engine::data_dir().join("app.log"), 15))
                                            .monospace()
                                            .size(12.0),
                                    )
                                    .selectable(true),
                                );
                            });

                        ui.separator();
                        heading(ui, "STORAGE & DATA LOCATION");
                        if let Some(ref msg) = *self.sh.migration_status.lock().unwrap() {
                            ui.label(RichText::new(msg).color(AMBER).size(12.0));
                        }
                        ui.horizontal(|ui| {
                            if ui.button("Migrate Models to Another Drive...").on_hover_text("Move models (~17GB) to another drive/partition (e.g. D:\\ImageStudio\\models)").clicked() {
                                if let Some(target) = rfd::FileDialog::new().pick_folder() {
                                    let src = self.sh.paths.lock().unwrap().models.clone();
                                    engine::migrate_storage(self.sh.clone(), src, target.join("models"), true, ctx.clone());
                                }
                            }
                            if ui.button("Open Outputs Folder").on_hover_text("Open Image Studio output images folder in Windows File Explorer").clicked() {
                                engine::open_output_folder(&self.sh);
                            }
                        });

                        ui.add_space(4.0);
                        ui.label(RichText::new(format!("Logs folder: {}", engine::data_dir().display())).size(12.0).color(DIM));
                    });
            });
        self.show_setup = open;
        if !self.show_setup {
            self.model_picker.pending_warning = None;
            self.model_picker.pending_delete = None;
        } else {
            let modal_action = crate::model_picker::render_warning_modal(ctx, &mut self.model_picker, &self.sh);
            self.handle_picker_action(modal_action, ctx);
        }
    }

    fn video_dl_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_video_dl_modal;
        let mut should_close = false;
        let center = ctx.content_rect().center();
        let vprof = self.sh.active_video_profile.lock().unwrap().clone();
        let total_gb = vprof.total_size_gb();
        let win_title = format!("Video Model Setup - {}", vprof.name);
        let mut file_lines = Vec::new();
        let mut has_optional_r2v = false;
        for m in &vprof.models {
            if m.optional {
                has_optional_r2v = true;
                continue;
            }
            file_lines.push(format!("- {} ({})", m.filename, engine::format_bytes(m.size_bytes)));
        }
        let optional_note = if has_optional_r2v {
            "\n\nOptional: The separate Ref2VA model for Reference-to-Video (R2V) is excluded by default and can be downloaded from the References panel if needed."
        } else {
            ""
        };
        let details = format!(
            "{}\n\nTotal download size: ~{:.1} GB\n{}{}",
            vprof.description,
            total_gb,
            file_lines.join("\n"),
            optional_note
        );

        egui::Window::new(win_title)
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(center)
            .fixed_size(vec2(540.0, 290.0))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(4.0);
                    ui.label(RichText::new("Download Video Generation Models").size(17.0).strong().color(TEXT));
                    ui.add_space(6.0);
                    egui::ScrollArea::vertical().max_height(145.0).show(ui, |ui| {
                        ui.label(RichText::new(&details).size(12.0).color(DIM));
                    });
                    ui.add_space(10.0);

                    let is_dl = self.sh.dl_video_model.load(Ordering::SeqCst);
                    if is_dl {
                        ui.label(RichText::new("Video models are currently downloading in the background.").color(GREEN).strong());
                        ui.add_space(8.0);
                        if ui.button("Close").clicked() {
                            should_close = true;
                        }
                    } else {
                        ui.horizontal(|ui| {
                            let dl_btn = egui::Button::new(RichText::new(format!("Start Download (~{:.1} GB)", total_gb)).strong().color(Color32::WHITE))
                                .fill(ACCENT)
                                .min_size(vec2(190.0, 32.0));
                            if ui.add(dl_btn).clicked() {
                                engine::toggle_model_download(&self.sh, ctx, false, true);
                                should_close = true;
                            }
                            if ui.button("Choose Another Profile...").clicked() {
                                self.show_setup = true;
                                should_close = true;
                            }
                            let cancel_btn = egui::Button::new("Not Now").min_size(vec2(80.0, 32.0));
                            if ui.add(cancel_btn).clicked() {
                                should_close = true;
                            }
                        });
                    }
                });
            });
        self.show_video_dl_modal = open && !should_close;
    }
}

impl App {
    fn update_taskbar_progress(&mut self) {
        use crate::taskbar::TaskbarState;
        if self.generating {
            let frac = self.total_gen_progress.unwrap_or(0.02);
            self.taskbar_progress.update(TaskbarState::Generating, frac);
        } else if let Some((dl_prog, _paused, _summary)) = crate::download_ui::active_download_progress(&self.sh) {
            self.taskbar_progress.update(TaskbarState::Download, dl_prog);
        } else {
            let upd_prog = {
                let st = self.updater.lock().unwrap();
                if st.downloading { Some(st.download_progress) } else { None }
            };
            if let Some(p) = upd_prog {
                self.taskbar_progress.update(TaskbarState::Download, p);
            } else {
                self.taskbar_progress.update(TaskbarState::None, 0.0);
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.drain(&ctx);
        self.sync_effective_video_profile();
        self.update_taskbar_progress();
        self.handle_input(&ctx);

        let starting = matches!(
            *self.sh.state.lock().unwrap(),
            EngState::Starting { .. }
                | EngState::Detecting
                | EngState::AwaitingDownloadConfirmation { .. }
                | EngState::DownloadingEngine { .. }
                | EngState::ExtractingEngine { .. }
                | EngState::DownloadingModels { .. }
                | EngState::ModelDownloadPaused { .. }
                | EngState::RepairingEngine { .. }
        );
        if starting || self.generating || self.loading > 0 {
            ctx.request_repaint_after(Duration::from_millis(150));
        }

        // Detect excessive stalling during generation
        if self.generating {
            if let Some(last_t) = self.last_progress_time {
                let stall_limit_secs = if self.mode == StudioMode::Video { 50 } else { 15 };
                if last_t.elapsed().as_secs() > stall_limit_secs && self.progress.map(|(v, m)| v > 0 && v < m).unwrap_or(false) {
                    self.is_swapping_detected = true;
                }
            }
        }

        // Top Header
        egui::Panel::top("header").frame(egui::Frame::new()).show(ui, |ui| self.header(ui));

        // Left Panel: Compact References Strip (+ Docked History if Left)
        // Minimum width ensures 2 LoRA tiles per row inside the panel (10px panel margin + 10px card margin per side + 4px stroke/rounding buffer).
        const LEFT_PANEL_MIN_W: f32 = crate::lora::LORA_TWO_COL_MIN_INNER_W + 44.0;
        egui::Panel::left("refs_panel")
            .default_size(LEFT_PANEL_MIN_W)
            .min_size(LEFT_PANEL_MIN_W)
            .max_size(400.0)
            .resizable(true)
            .frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin::same(10)))
            .show(ui, |ui| {
                self.refs_panel(ui, &ctx);
            });

        // Right Panel: Compact, power-user styled settings (resizable, + Docked History if Right)
        egui::Panel::right("settings_panel")
            .default_size(350.0)
            .min_size(260.0)
            .max_size(600.0)
            .resizable(true)
            .frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin::same(12)))
            .show(ui, |ui| {
                self.settings_panel(ui);
            });

        // Bottom Dock: ChatGPT-style Prompt & Action Bar (in the center column)
        egui::Panel::bottom("prompt_dock")
            .frame(egui::Frame::new().fill(PANEL).inner_margin(egui::Margin::symmetric(12, 8)).stroke(Stroke::new(1.0, LINE)))
            .show(ui, |ui| {
                self.bottom_dock(ui, &ctx);
            });

        // Center Hero Canvas: centered, zoomable, pannable
        let central_resp = egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin::same(12)))
            .show(ui, |ui| {
                self.center_canvas(ui, &ctx);
            });
        self.canvas_drop_rect = Some(central_resp.response.rect);

        // Floating Windows: Setup and Floating History (if undocked)
        self.setup_window(&ctx);
        self.floating_history_window(&ctx);
        if self.show_video_dl_modal {
            self.video_dl_window(&ctx);
        }
        self.new_video_preset_modal(&ctx);
        self.history_delete_confirm_modal(&ctx);
        self.download_pane.show_combined(&ctx, &self.sh, &self.updater);

        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("drop_overlay")));
            let full_r = ctx.content_rect();
            painter.rect_filled(full_r, 0, Color32::from_rgba_unmultiplied(12, 14, 19, 175));

            let hover_pos = ctx.input(|i| i.pointer.hover_pos().or_else(|| i.pointer.latest_pos()));
            let over_refs = hover_pos.is_some_and(|pos| {
                self.refs_drop_rect
                    .is_some_and(|r| r.expand(6.0).contains(pos))
            });
            let over_canvas = !over_refs;

            if let Some(refs_r) = self.refs_drop_rect {
                let target_r = refs_r.expand(4.0);
                let fill = if over_refs {
                    ACCENT.linear_multiply(0.22)
                } else {
                    Color32::from_rgba_unmultiplied(20, 24, 36, 210)
                };
                let stroke = Stroke::new(
                    if over_refs { 3.0 } else { 1.5 },
                    if over_refs { ACCENT_GLOW } else { ACCENT.linear_multiply(0.6) },
                );
                painter.rect_filled(target_r, 10, fill);
                painter.rect_stroke(target_r, 10, stroke, StrokeKind::Inside);
                let refs_msg = if self.refs.len() >= MAX_REFS {
                    "References Full"
                } else {
                    "Drop here\nAdd Reference(s)"
                };
                painter.text(
                    target_r.center(),
                    egui::Align2::CENTER_CENTER,
                    refs_msg,
                    egui::FontId::proportional(14.5),
                    if over_refs { Color32::WHITE } else { TEXT },
                );
            }

            let canvas_r = self
                .canvas_drop_rect
                .map(|r| r.shrink(10.0))
                .unwrap_or_else(|| full_r.shrink(18.0));
            let c_fill = if over_canvas {
                ACCENT.linear_multiply(0.16)
            } else {
                Color32::from_rgba_unmultiplied(16, 19, 28, 190)
            };
            let c_stroke = Stroke::new(
                if over_canvas { 3.0 } else { 1.5 },
                if over_canvas { ACCENT_GLOW } else { LINE },
            );
            painter.rect_filled(canvas_r, 14, c_fill);
            painter.rect_stroke(canvas_r, 14, c_stroke, StrokeKind::Inside);
            let canvas_msg = if self.mode == StudioMode::Video {
                "Drop image or video here\nLoad into Video Canvas"
            } else {
                "Drop image or video here\nLoad as Base Input"
            };
            painter.text(
                canvas_r.center(),
                egui::Align2::CENTER_CENTER,
                canvas_msg,
                egui::FontId::proportional(20.0),
                if over_canvas { Color32::WHITE } else { DIM },
            );
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let (remember_mode, keep_project) = {
            let c = self.sh.cfg.lock().unwrap();
            (c.remember_last_mode, c.keep_last_project_loaded)
        };
        if remember_mode {
            let mut c = self.sh.cfg.lock().unwrap();
            c.last_studio_mode = Some(if self.mode == StudioMode::Video { "video" } else { "image" }.to_string());
            c.save();
        }
        if keep_project {
            self.save_last_project();
        } else {
            Self::clear_last_project();
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PersistedHistoryNode {
    id: usize,
    parent_id: Option<usize>,
    label: String,
    prompt: String,
    width: u32,
    height: u32,
    step_num: usize,
    output_filename: String,
    output_subfolder: String,
    temp_files: Vec<String>,
    png_file: Option<String>,
    mp4_file: Option<String>,
    source_video_path: Option<PathBuf>,
    video_last_sec: Option<f64>,
    video_trim_start: Option<f64>,
    video_trim_end: Option<f64>,
    video_work_mode: Option<VideoWorkMode>,
}

#[derive(Serialize, Deserialize)]
struct PersistedRefItem {
    name: String,
    width: u32,
    height: u32,
    png_file: String,
}

#[derive(Serialize, Deserialize)]
struct PersistedProjectState {
    #[serde(default)]
    studio_mode: Option<String>,
    video_work_mode: VideoWorkMode,
    prompt: String,
    negative: String,
    video_negative: String,
    active_node_id: Option<usize>,
    next_node_id: usize,
    nodes: Vec<PersistedHistoryNode>,
    refs: Vec<PersistedRefItem>,
    video_player_active: bool,
    video_player_sec: Option<f64>,
    video_player_trim_start: Option<f64>,
    video_player_trim_end: Option<f64>,
    video_duration: u32,
    video_fps: u32,
    video_width: u32,
    video_height: u32,
    video_scale: f32,
    video_orientation: VideoOrientation,
    custom_w: u32,
    custom_h: u32,
    ref_w: u32,
    ref_h: u32,
    scale: f32,
    steps: u32,
    cfg: f32,
    video_steps: u32,
    video_prompt_enhance: bool,
    image_loras: Vec<crate::lora::ActiveLora>,
    image_prompt_tags: Vec<crate::lora::PromptTag>,
    video_loras: Vec<crate::lora::ActiveLora>,
    video_prompt_tags: Vec<crate::lora::PromptTag>,
}

impl App {
    fn last_project_json_path() -> PathBuf {
        engine::data_dir().join("last_project.json")
    }

    fn last_project_assets_dir() -> PathBuf {
        engine::data_dir().join("last_project_assets")
    }

    fn clear_last_project() {
        let _ = std::fs::remove_file(Self::last_project_json_path());
        let _ = std::fs::remove_dir_all(Self::last_project_assets_dir());
    }

    fn save_last_project(&self) {
        let assets_dir = Self::last_project_assets_dir();
        let _ = std::fs::remove_dir_all(&assets_dir);
        if std::fs::create_dir_all(&assets_dir).is_err() {
            return;
        }

        let cur_player_sec = self.video_player.as_ref().map(|p| p.current_sec);
        let cur_player_trim_s = self.video_player.as_ref().and_then(|p| p.trim_start_sec);
        let cur_player_trim_e = self.video_player.as_ref().and_then(|p| p.trim_end_sec);

        let mut persisted_nodes = Vec::with_capacity(self.nodes.len());
        for n in &self.nodes {
            let png_file = if !n.png.is_empty() {
                let fname = format!("node_{}.png", n.id);
                if std::fs::write(assets_dir.join(&fname), &n.png).is_ok() {
                    Some(fname)
                } else {
                    None
                }
            } else {
                None
            };
            let mp4_file = if let Some(ref mp4) = n.mp4_bytes {
                // If source_video_path exists on disk, no need to duplicate huge bytes unless missing
                if n.source_video_path.as_ref().is_some_and(|p| p.is_file()) {
                    None
                } else {
                    let fname = format!("node_{}.mp4", n.id);
                    if std::fs::write(assets_dir.join(&fname), mp4).is_ok() {
                        Some(fname)
                    } else {
                        None
                    }
                }
            } else {
                None
            };
            let is_active = self.active_node_id == Some(n.id);
            persisted_nodes.push(PersistedHistoryNode {
                id: n.id,
                parent_id: n.parent_id,
                label: n.label.clone(),
                prompt: n.prompt.clone(),
                width: n.width,
                height: n.height,
                step_num: n.step_num,
                output_filename: n.output_filename.clone(),
                output_subfolder: n.output_subfolder.clone(),
                temp_files: n.temp_files.clone(),
                png_file,
                mp4_file,
                source_video_path: n.source_video_path.clone(),
                video_last_sec: if is_active { cur_player_sec.or(n.video_last_sec) } else { n.video_last_sec },
                video_trim_start: if is_active && self.video_player.is_some() { cur_player_trim_s } else { n.video_trim_start },
                video_trim_end: if is_active && self.video_player.is_some() { cur_player_trim_e } else { n.video_trim_end },
                video_work_mode: if is_active { Some(self.video_work_mode) } else { n.video_work_mode },
            });
        }

        let mut persisted_refs = Vec::with_capacity(self.refs.len());
        for (idx, r) in self.refs.iter().enumerate() {
            let fname = format!("ref_{idx}.png");
            if std::fs::write(assets_dir.join(&fname), &r.png).is_ok() {
                persisted_refs.push(PersistedRefItem {
                    name: r.name.clone(),
                    width: r.width,
                    height: r.height,
                    png_file: fname,
                });
            }
        }

        let state = PersistedProjectState {
            studio_mode: Some(if self.mode == StudioMode::Video { "video" } else { "image" }.to_string()),
            video_work_mode: self.video_work_mode,
            prompt: self.prompt.clone(),
            negative: self.negative.clone(),
            video_negative: self.video_negative.clone(),
            active_node_id: self.active_node_id,
            next_node_id: self.next_node_id,
            nodes: persisted_nodes,
            refs: persisted_refs,
            video_player_active: self.video_player.is_some(),
            video_player_sec: cur_player_sec,
            video_player_trim_start: cur_player_trim_s,
            video_player_trim_end: cur_player_trim_e,
            video_duration: self.video_duration,
            video_fps: self.video_fps,
            video_width: self.video_width,
            video_height: self.video_height,
            video_scale: self.video_scale,
            video_orientation: self.video_orientation,
            custom_w: self.custom_w,
            custom_h: self.custom_h,
            ref_w: self.ref_w,
            ref_h: self.ref_h,
            scale: self.scale,
            steps: self.steps,
            cfg: self.cfg,
            video_steps: self.video_steps,
            video_prompt_enhance: self.video_prompt_enhance,
            image_loras: self.image_loras.clone(),
            image_prompt_tags: self.image_prompt_tags.clone(),
            video_loras: self.video_loras.clone(),
            video_prompt_tags: self.video_prompt_tags.clone(),
        };

        if let Ok(json) = serde_json::to_string(&state) {
            let _ = std::fs::write(Self::last_project_json_path(), json);
        }
    }

    fn restore_last_project(&mut self, ctx: &egui::Context) {
        let Ok(json) = std::fs::read_to_string(Self::last_project_json_path()) else {
            return;
        };
        let Ok(state) = serde_json::from_str::<PersistedProjectState>(&json) else {
            return;
        };
        let assets_dir = Self::last_project_assets_dir();

        let load_png_tex = |fname: &str, tex_id: &str| -> Option<(Vec<u8>, TextureHandle)> {
            let png = std::fs::read(assets_dir.join(fname)).ok()?;
            let img = image::load_from_memory(&png).ok()?;
            let rgba = img.to_rgba8();
            let ci = ColorImage::from_rgba_unmultiplied([rgba.width() as usize, rgba.height() as usize], rgba.as_raw());
            let tex = ctx.load_texture(tex_id, ci, egui::TextureOptions::LINEAR);
            Some((png, tex))
        };

        let mut restored_nodes = Vec::new();
        for pn in state.nodes {
            let Some(ref png_file) = pn.png_file else {
                continue;
            };
            let Some((png, tex)) = load_png_tex(png_file, &format!("restored_node_{}", pn.id)) else {
                continue;
            };
            let mp4_bytes = pn
                .mp4_file
                .as_ref()
                .and_then(|f| std::fs::read(assets_dir.join(f)).ok());
            restored_nodes.push(HistoryNode {
                id: pn.id,
                parent_id: pn.parent_id,
                label: pn.label,
                prompt: pn.prompt,
                png,
                tex,
                width: pn.width,
                height: pn.height,
                step_num: pn.step_num,
                output_filename: pn.output_filename,
                output_subfolder: pn.output_subfolder,
                temp_files: pn.temp_files,
                mp4_bytes,
                source_video_path: pn.source_video_path,
                video_last_sec: pn.video_last_sec,
                video_trim_start: pn.video_trim_start,
                video_trim_end: pn.video_trim_end,
                video_work_mode: pn.video_work_mode,
            });
        }

        let mut restored_refs = Vec::new();
        for (idx, pr) in state.refs.into_iter().enumerate() {
            if let Some((png, tex)) = load_png_tex(&pr.png_file, &format!("restored_ref_{idx}")) {
                restored_refs.push(Item {
                    name: pr.name,
                    png,
                    tex,
                    width: pr.width,
                    height: pr.height,
                });
            }
        }

        self.nodes = restored_nodes;
        self.refs = restored_refs;
        self.next_node_id = state.next_node_id.max(self.nodes.iter().map(|n| n.id + 1).max().unwrap_or(1));
        self.prompt = state.prompt;
        self.negative = state.negative;
        self.video_negative = state.video_negative;
        self.video_duration = state.video_duration;
        self.video_fps = state.video_fps;
        self.video_width = state.video_width;
        self.video_height = state.video_height;
        self.video_scale = state.video_scale;
        self.video_orientation = state.video_orientation;
        self.custom_w = state.custom_w;
        self.custom_h = state.custom_h;
        self.ref_w = state.ref_w;
        self.ref_h = state.ref_h;
        self.scale = state.scale;
        self.steps = state.steps;
        self.cfg = state.cfg;
        self.video_steps = state.video_steps;
        self.video_prompt_enhance = state.video_prompt_enhance;
        self.image_loras = state.image_loras;
        self.image_prompt_tags = state.image_prompt_tags;
        self.video_loras = state.video_loras;
        self.video_prompt_tags = state.video_prompt_tags;
        self.video_work_mode = state.video_work_mode;

        if let Some(aid) = state.active_node_id {
            if self.nodes.iter().any(|n| n.id == aid) {
                let allow_video_player = self.mode == StudioMode::Video
                    || state.studio_mode.as_deref() == Some("image");
                let saved_vw = self.video_width;
                let saved_vh = self.video_height;
                let saved_vbw = self.video_base_w;
                let saved_vbh = self.video_base_h;
                let saved_vscale = self.video_scale;
                let saved_vorient = self.video_orientation;
                if state.video_player_active && allow_video_player {
                    let saved_prompt = self.prompt.clone();
                    self.select_node(aid);
                    self.prompt = saved_prompt;
                    if let Some(ref mut player) = self.video_player {
                        player.trim_start_sec = state.video_player_trim_start;
                        player.trim_end_sec = state.video_player_trim_end;
                        if let Some(sec) = state.video_player_sec {
                            if sec > 0.001 {
                                player.seek_to_time(sec);
                            }
                        }
                    }
                } else {
                    self.active_node_id = Some(aid);
                    self.update_dimensions_from_active_node();
                }
                self.video_width = saved_vw;
                self.video_height = saved_vh;
                self.video_base_w = saved_vbw;
                self.video_base_h = saved_vbh;
                self.video_scale = saved_vscale;
                self.video_orientation = saved_vorient;
            }
        }
        self.sync_effective_video_profile();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_large_image_dimension_clamping() {
        let img = image::DynamicImage::new_rgb8(4200, 3000);
        let max_ref = 2048u32;
        let resized_ref = img.resize(max_ref, max_ref, image::imageops::FilterType::Triangle);
        assert!(resized_ref.width() <= 2048);
        assert!(resized_ref.height() <= 2048);
        assert_eq!(resized_ref.width(), 2048);

        let max_base = 4096u32;
        let resized_base = img.resize(max_base, max_base, image::imageops::FilterType::Triangle);
        assert!(resized_base.width() <= 4096);
        assert!(resized_base.height() <= 4096);
        assert_eq!(resized_base.width(), 4096);
    }

    #[test]
    fn test_estimate_video_vram_gb() {
        let estimate_video_vram_gb = |w, h, d, f| super::estimate_video_vram_gb_with_lora(w, h, d, f, 0.0);
        let vram_720p_5s = estimate_video_vram_gb(1280, 720, 5, 24);
        assert!(vram_720p_5s <= 12.5, "720p 5s should fit within ~12GB VRAM");

        let vram_720p_9s = estimate_video_vram_gb(1280, 720, 9, 24);
        assert!(vram_720p_9s > 13.0, "720p 9s should exceed 12GB VRAM and trigger warning");

        let vram_360p_5s = estimate_video_vram_gb(640, 384, 5, 24);
        assert!(vram_360p_5s < 10.5, "360p 5s should require minimal VRAM");

        let vram_4k_5s = estimate_video_vram_gb(3840, 2160, 5, 24);
        assert!(vram_4k_5s > 30.0, "4K 5s should require >30GB VRAM");

        let vram_with_lora = super::estimate_video_vram_gb_with_lora(1280, 720, 5, 24, 0.8);
        assert!((vram_with_lora - (vram_720p_5s + 0.8)).abs() < 0.001);
    }

    #[test]
    fn test_absolute_resolution_scale_range() {
        use super::{absolute_scale_range, scale_dimensions_stepped, ABS_MAX_RES, ABS_MIN_RES};

        // 1440p landscape video: 2560x1440 -> min height 256 (448x256), max width 4096 (4096x2304)
        let (k_min, k_max) = absolute_scale_range(2560, 1440, ABS_MIN_RES, ABS_MAX_RES);
        assert!(k_min < 1.0 && 1.0 < k_max);
        assert_eq!(scale_dimensions_stepped(2560, 1440, k_min, 32, ABS_MIN_RES, ABS_MAX_RES), (448, 256));
        assert_eq!(scale_dimensions_stepped(2560, 1440, k_max, 32, ABS_MIN_RES, ABS_MAX_RES), (4096, 2304));

        // Small square image: 512x512 -> min 256x256, max 4096x4096
        let (k_min, k_max) = absolute_scale_range(512, 512, ABS_MIN_RES, ABS_MAX_RES);
        assert_eq!(scale_dimensions_stepped(512, 512, k_min, 16, ABS_MIN_RES, ABS_MAX_RES), (256, 256));
        assert_eq!(scale_dimensions_stepped(512, 512, k_max, 16, ABS_MIN_RES, ABS_MAX_RES), (4096, 4096));

        // Portrait 9:16 video: 1440x2560 -> min width 256 (256x448), max height 4096 (2304x4096)
        let (k_min, k_max) = absolute_scale_range(1440, 2560, ABS_MIN_RES, ABS_MAX_RES);
        assert_eq!(scale_dimensions_stepped(1440, 2560, k_min, 32, ABS_MIN_RES, ABS_MAX_RES), (256, 448));
        assert_eq!(scale_dimensions_stepped(1440, 2560, k_max, 32, ABS_MIN_RES, ABS_MAX_RES), (2304, 4096));

        // Extreme aspect ratio (>16:1): k_min <= k_max and dimensions stay within bounds
        let (k_min, k_max) = absolute_scale_range(4096, 128, ABS_MIN_RES, ABS_MAX_RES);
        assert!(k_min <= k_max);
        let (w, h) = scale_dimensions_stepped(4096, 128, k_min, 16, ABS_MIN_RES, ABS_MAX_RES);
        assert!((ABS_MIN_RES..=ABS_MAX_RES).contains(&w));
        assert!((ABS_MIN_RES..=ABS_MAX_RES).contains(&h));
    }
}
