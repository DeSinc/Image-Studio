//! All UI logic.
//! Guaranteed rules:
//! 1. The UI thread never blocks, never decodes images, and never makes network calls.
//! 2. Heavy work happens in worker threads, sending events via `mpsc::channel`.
//! 3. Textures are created strictly on the UI thread inside `drain()`, outside any input locks.
use crate::comfy;
use crate::engine::{self, EngState, Profile, Shared};
use eframe::egui::{self, pos2, vec2, Color32, ColorImage, Rect, RichText, Sense, Stroke, StrokeKind, TextureHandle, Vec2};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

pub fn estimate_video_vram_gb(w: u32, h: u32, duration: u32, fps: u32) -> f32 {
    let frames = (duration * fps) as f32;
    let mp = (w * h) as f32 / 1_000_000.0;
    let load_factor = mp * (frames / 24.0);
    9.5 + load_factor * 0.55
}

const BG: Color32 = Color32::from_rgb(12, 14, 19);
const PANEL: Color32 = Color32::from_rgb(18, 21, 29);
const CARD: Color32 = Color32::from_rgb(25, 29, 40);
const CARD_HI: Color32 = Color32::from_rgb(33, 38, 53);
const LINE: Color32 = Color32::from_rgb(45, 51, 70);
const ACCENT: Color32 = Color32::from_rgb(99, 102, 241);
const ACCENT_GLOW: Color32 = Color32::from_rgb(129, 140, 248);
const TEXT: Color32 = Color32::from_rgb(229, 231, 240);
const DIM: Color32 = Color32::from_rgb(138, 146, 170);
const GREEN: Color32 = Color32::from_rgb(52, 211, 153);
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
    refs: Vec<Item>,
    prompt: String,
    focus_prompt: bool,
    progress: Option<(u32, u32)>,
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
    show_setup: bool,
    loading: usize,
    queue: Vec<QueuedItem>,
    last_ctrl_v_down: bool,
    pub updater: Arc<std::sync::Mutex<crate::updater::UpdaterState>>,
    pub mode: StudioMode,
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
    pub show_video_dl_modal: bool,
    pub show_downloads_menu: bool,
    pub last_progress_time: Option<Instant>,
    pub last_progress_val: u32,
    pub step_duration_ema: f32,
    pub is_swapping_detected: bool,
    pub generation_retries: u32,
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

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ctx = cc.egui_ctx.clone();
        ctx.style_mut_of(egui::Theme::Dark, |s| {
            let v = &mut s.visuals;
            v.dark_mode = true;
            v.override_text_color = Some(TEXT);
            v.panel_fill = PANEL;
            v.window_fill = PANEL;
            v.window_stroke = Stroke::new(1.0, LINE);
            v.widgets.noninteractive.bg_fill = CARD;
            v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, LINE);
            v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
            v.widgets.inactive.bg_fill = CARD_HI;
            v.widgets.hovered.weak_bg_fill = Color32::from_rgb(46, 52, 72);
            v.widgets.hovered.bg_fill = Color32::from_rgb(46, 52, 72);
            v.widgets.active.weak_bg_fill = ACCENT;
            s.spacing.item_spacing = vec2(6.0, 6.0);
            s.spacing.button_padding = vec2(8.0, 4.0);
            for (ts, sz) in [
                (egui::TextStyle::Body, 13.0),
                (egui::TextStyle::Button, 12.0),
                (egui::TextStyle::Small, 11.0),
                (egui::TextStyle::Heading, 16.0),
                (egui::TextStyle::Monospace, 11.5),
            ] {
                if let Some(f) = s.text_styles.get_mut(&ts) {
                    f.size = sz;
                }
            }
        });
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

        App {
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
            refs: vec![],
            prompt: String::new(),
            focus_prompt: false,
            progress: None,
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
            show_setup: false,
            loading: 0,
            queue: Vec::new(),
            last_ctrl_v_down: false,
            updater,
            mode: StudioMode::Image,
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
            show_video_dl_modal: false,
            show_downloads_menu: false,
            last_progress_time: None,
            last_progress_val: 0,
            step_duration_ema: 0.0,
            is_swapping_detected: false,
            generation_retries: 0,
        }
    }

    fn delete_history_node(&mut self, node_id: usize) {
        if let Some(pos) = self.nodes.iter().position(|n| n.id == node_id) {
            let node = self.nodes.remove(pos);

            let comfy_dir = self.sh.paths.lock().unwrap().comfy.clone();
            let mut search_dirs = Vec::new();
            if let Some(ref cp) = comfy_dir {
                search_dirs.push(cp.clone());
                if let Some(parent) = cp.parent() {
                    search_dirs.push(parent.to_path_buf());
                }
            }
            search_dirs.push(engine::data_dir());
            search_dirs.push(std::env::temp_dir());

            // Delete output image files generated by ComfyUI
            if !node.output_filename.is_empty() {
                for dir in &search_dirs {
                    let candidates = [
                        dir.join("output").join(&node.output_subfolder).join(&node.output_filename),
                        dir.join("output").join(&node.output_filename),
                        dir.join(&node.output_subfolder).join(&node.output_filename),
                        dir.join(&node.output_filename),
                    ];
                    for c in candidates {
                        if c.is_file() {
                            let _ = std::fs::remove_file(&c);
                            engine::log(&format!("Deleted output file: {}", c.display()));
                        }
                    }
                }
            }

            // Delete temp input images uploaded to ComfyUI
            for tf in &node.temp_files {
                for dir in &search_dirs {
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

            // If the deleted node was currently active, switch to parent or last remaining node
            if self.active_node_id == Some(node_id) {
                self.active_node_id = node.parent_id
                    .filter(|pid| self.nodes.iter().any(|n| n.id == *pid))
                    .or_else(|| self.nodes.last().map(|n| n.id));
                self.update_dimensions_from_active_node();
            }
        }
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

        self.active_node_id = Some(id);
        self.compare_mode = false;
        if !self.generating {
            let node_info = self.find_node(id).map(|n| (
                n.prompt.clone(),
                n.mp4_bytes.clone(),
                n.output_filename.clone(),
                n.output_subfolder.clone(),
            ));
            if let Some((prompt, mp4_bytes, out_fn, out_sub)) = node_info {
                if !prompt.is_empty() {
                    self.prompt = prompt;
                    self.focus_prompt = true;
                }

                // If selecting a video history node, load it in the video player
                if let Some(mp4) = mp4_bytes {
                    let already_loaded = self.video_player.as_ref()
                        .map(|p| p.filename == out_fn && !out_fn.is_empty())
                        .unwrap_or(false);
                    if !already_loaded {
                        if let Ok(player) = crate::video::VideoPlayer::from_mp4_bytes(
                            mp4,
                            out_fn,
                            out_sub,
                        ) {
                            self.video_player = Some(player);
                        }
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
        if let Some((n_w, n_h)) = self.active_node().map(|n| (n.width, n.height)) {
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
                match p.size_preset {
                    SizePreset::Original => {
                        self.custom_w = n_w;
                        self.custom_h = n_h;
                        self.scale = 1.0;
                    }
                    SizePreset::P720 => {
                        self.custom_w = 720;
                        self.custom_h = (((720.0 * n_h as f32 / n_w as f32) / 16.0).round() as u32 * 16).clamp(64, 4096);
                        self.scale = (720.0 / n_w as f32).clamp(0.1, 2.0);
                    }
                    SizePreset::P1024 => {
                        self.custom_w = 1024;
                        self.custom_h = (((1024.0 * n_h as f32 / n_w as f32) / 16.0).round() as u32 * 16).clamp(64, 4096);
                        self.scale = (1024.0 / n_w as f32).clamp(0.1, 2.0);
                    }
                    SizePreset::Custom => {
                        if n_w > 0 {
                            self.scale = (p.target_w as f32 / n_w as f32).clamp(0.1, 2.0);
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
        if self.ref_w == 0 { self.ref_w = self.custom_w.max(64); }
        if self.ref_h == 0 { self.ref_h = self.custom_h.max(64); }
        self.custom_w = (((self.ref_w as f32 * self.scale) / 16.0).round() as u32 * 16).clamp(64, 4096);
        self.custom_h = (((self.ref_h as f32 * self.scale) / 16.0).round() as u32 * 16).clamp(64, 4096);
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

    pub fn has_video_models(&self) -> bool {
        let paths = self.sh.paths.lock().unwrap();
        let models_dir = &paths.models;
        models_dir.join("checkpoints").join("ltx-2.3-22b-dev-fp8.safetensors").exists()
    }

    fn start_generation(&mut self, ctx: &egui::Context) {
        self.generation_retries = 0;
        let p = self.prompt.trim().to_string();
        self.prompt.clear();
        if self.mode == StudioMode::Video {
            self.start_video_generation_with_prompt(ctx, p);
        } else {
            self.start_generation_with_prompt(ctx, p);
        }
    }

    fn start_video_generation_with_prompt(&mut self, ctx: &egui::Context, prompt: String) {
        // If a video is currently loaded in the player, automatically extract the frame
        // the user has scrubbed to as the start frame for the next shot!
        let (base_png_opt, scrubbed_frame_ci) = if let Some(ref player) = self.video_player {
            let cur_idx = player.current_frame.min(player.frames.len().saturating_sub(1));
            let cur_img = player.frames[cur_idx].clone();
            let [f_w, f_h] = cur_img.size;
            let mut png_bytes = Vec::new();
            if let Some(rgba) = image::RgbaImage::from_raw(f_w as u32, f_h as u32, cur_img.as_raw().to_vec()) {
                let _ = image::DynamicImage::ImageRgba8(rgba)
                    .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png);
            }
            if !png_bytes.is_empty() {
                (Some(png_bytes), Some(cur_img))
            } else {
                (self.active_node().map(|n| n.png.clone()), None)
            }
        } else {
            (self.active_node().map(|n| n.png.clone()), None)
        };
        let is_t2v = base_png_opt.is_none();

        self.generating = true;
        self.status = if is_t2v {
            "Generating video from text (LTX-2.3)...".into()
        } else {
            "Generating next shot from start frame (LTX-2.3)...".into()
        };
        self.error = None;
        self.generating_parent_id = self.active_node_id;
        self.generating_prompt = prompt.trim().to_string();

        if let Some(pid) = self.active_node_id {
            if let Some(parent) = self.nodes.iter_mut().find(|n| n.id == pid) {
                parent.prompt = self.generating_prompt.clone();
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

        let vparams = comfy::VideoParams {
            prompt: self.generating_prompt.clone(),
            negative: self.video_negative.clone(),
            width: self.video_width,
            height: self.video_height,
            duration: self.video_duration,
            fps: self.video_fps,
            seed,
            prompt_enhance: self.video_prompt_enhance,
            ..Default::default()
        };

        self.progress = None;
        if let Some(ci) = scrubbed_frame_ci {
            self.preview_tex = Some(ctx.load_texture("locked_start_frame", ci, egui::TextureOptions::LINEAR));
        } else {
            self.preview_tex = None;
        }
        self.video_player = None;
        self.last_progress_time = None;
        self.last_progress_val = 0;
        self.step_duration_ema = 0.0;
        self.is_swapping_detected = false;

        let tx_p = tx.clone();
        let ctx_p = ctx.clone();
        let prog_cb = Arc::new(move |value: u32, max: u32| {
            let _ = tx_p.send(Ev::Progress { value, max });
            ctx_p.request_repaint();
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
        let base_png_opt = self.active_node().map(|n| n.png.clone());
        let is_t2i = base_png_opt.is_none();

        self.generating = true;
        self.status = if is_t2i { "Generating image...".into() } else { "Uploading images...".into() };
        self.error = None;
        self.generating_parent_id = self.active_node_id;
        self.generating_prompt = prompt.trim().to_string();

        // Save the prompt on the parent node that created this generation
        if let Some(pid) = self.active_node_id {
            if let Some(parent) = self.nodes.iter_mut().find(|n| n.id == pid) {
                parent.prompt = self.generating_prompt.clone();
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
        };

        self.progress = None;
        self.preview_tex = None;
        self.last_progress_time = None;
        self.last_progress_val = 0;
        self.step_duration_ema = 0.0;
        self.is_swapping_detected = false;
        let tx_p = tx.clone();
        let ctx_p = ctx.clone();
        let prog_cb = Arc::new(move |value: u32, max: u32| {
            let _ = tx_p.send(Ev::Progress { value, max });
            ctx_p.request_repaint();
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
                            };
                            self.nodes.push(node);
                            self.ref_w = width;
                            self.ref_h = height;
                            self.custom_w = width;
                            self.custom_h = height;
                            self.scale = 1.0;
                            self.size_preset = SizePreset::Original;
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
                                let vram_exceeded = if let Some(gpu_vram) = engine::detect_gpu_vram_gb() {
                                    estimate_video_vram_gb(self.video_width, self.video_height, self.video_duration, self.video_fps) > gpu_vram as f32
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
                    self.preview_tex = None;
                    self.is_swapping_detected = false;
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
                    };
                    self.nodes.push(node);
                    self.active_node_id = Some(id);
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
                    self.preview_tex = None;
                    self.is_swapping_detected = false;
                    self.generation_retries = 0;

                    match crate::video::VideoPlayer::from_mp4_bytes(mp4_bytes, output_filename.clone(), output_subfolder.clone()) {
                        Ok(player) => {
                            if let Some(first) = player.frames.first() {
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
                                    width: player.width,
                                    height: player.height,
                                    step_num,
                                    output_filename,
                                    output_subfolder,
                                    temp_files,
                                    mp4_bytes: Some(player.mp4_bytes.clone()),
                                };
                                self.nodes.push(node);
                                self.active_node_id = Some(id);
                            }
                            self.video_player = Some(player);
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
                    self.preview_tex = None;
                    self.is_swapping_detected = false;
                    self.queue.clear();
                    if e == "Cancelled" {
                        self.status = "Cancelled".into();
                        self.generation_retries = 0;
                    } else if (e.contains("stalled") || e.contains("stopped") || e.contains("without producing")) && self.generation_retries < 1 {
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
                        self.prompt = self.generating_prompt.clone();
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
        let mut need_base = self.active_node().is_none();
        for p in dropped {
            let t = if need_base { Target::Base } else { Target::Ref };
            need_base = false;
            self.load(ctx, t, Src::Path(p));
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

        // Undo / Redo keyboard shortcuts
        let undo = ctx.input(|i| i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::Z));
        let redo = ctx.input(|i| (i.modifiers.command && i.key_pressed(egui::Key::Y)) || (i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::Z)));
        if undo && !ctx.egui_wants_keyboard_input() && self.can_undo() {
            self.undo();
        }
        if redo && !ctx.egui_wants_keyboard_input() && self.can_redo() {
            self.redo();
        }
    }

    #[allow(dead_code)]
    fn engine_ready(&self) -> bool {
        matches!(*self.sh.state.lock().unwrap(), EngState::Ready { .. })
    }

    fn can_queue(&self) -> Result<(), &'static str> {
        let st = self.sh.state.lock().unwrap().clone();
        match st {
            EngState::Ready { .. } | EngState::Starting { .. } | EngState::Detecting => {}
            EngState::DownloadingEngine { .. } => return Err("Downloading engine..."),
            EngState::ExtractingEngine { .. } => return Err("Extracting engine..."),
            EngState::DownloadingModels { .. } => return Err("Downloading models..."),
            EngState::RepairingEngine { .. } => return Err("Repairing engine..."),
            EngState::ModelDownloadFailed { .. } => return Err("Model download paused (see Setup)"),
            EngState::Failed(_) => return Err("Engine setup failed (see Setup)"),
        }
        if self.mode == StudioMode::Video {
            if !self.has_video_models() {
                return Err("Video models not downloaded (check Setup)");
            }
        } else {
            if self.sh.models_found.lock().unwrap().iter().any(|m| !m.1) {
                return Err("Downloading model weights...");
            }
        }
        Ok(())
    }

    fn can_generate(&self) -> Result<(), &'static str> {
        if self.generating {
            return Err("Generating...");
        }
        if self.prompt.trim().is_empty() {
            return Err("Type a prompt to generate or edit");
        }
        let st = self.sh.state.lock().unwrap().clone();
        if !matches!(st, EngState::Ready { .. }) {
            return Err("Engine starting - prompt will be queued");
        }
        self.can_queue()
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
                            EngState::DownloadingEngine { .. } => ("Downloading engine...".to_string(), AMBER),
                            EngState::ExtractingEngine { .. } => ("Extracting engine...".to_string(), AMBER),
                            EngState::DownloadingModels { file_index, total_files, .. } => (format!("Downloading model ({}/{})", file_index, total_files), AMBER),
                            EngState::RepairingEngine { .. } => ("Repairing engine...".to_string(), AMBER),
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
                            self.show_downloads_menu = !self.show_downloads_menu;
                        }
                        pill_click.on_hover_text("Click to view downloads list");
                        ui.label(RichText::new(format!("build {}", env!("BUILD_STAMP"))).size(11.0).color(DIM));
                    });
                });
            });
    }

    fn render_update_button(&self, ui: &mut egui::Ui, ctx: &egui::Context) {
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
            let (rect, resp) = ui.allocate_exact_size(vec2(72.0, 24.0), Sense::hover());
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
            let _ = resp.on_hover_text(format!("Downloading update: {:.0}%", dl_prog * 100.0));
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
            let resp = resp.on_hover_text(format!("Update v{latest} available! Click to update now"));
            if resp.clicked() {
                crate::updater::download_and_install_update(self.updater.clone(), ctx.clone());
            }
        } else {
            let (rect, resp) = ui.allocate_exact_size(vec2(26.0, 24.0), Sense::click());
            let bg = if resp.hovered() { CARD_HI } else { Color32::TRANSPARENT };
            let p = ui.painter();
            if resp.hovered() {
                p.rect_filled(rect, 4.0, bg);
            }
            let icon_col = if resp.hovered() { TEXT } else { DIM };
            draw_update_arrow_icon(&p, rect.center(), icon_col);
            let cur_ver = env!("CARGO_PKG_VERSION");
            let tip = if checked {
                format!("Image Studio v{cur_ver} (Up to date)\nClick to check for updates")
            } else {
                format!("Image Studio v{cur_ver}\nClick to check for updates")
            };
            let resp = resp.on_hover_text(tip);
            if resp.clicked() {
                crate::updater::check_for_updates(self.updater.clone(), ctx.clone());
            }
        }
    }

    // Left Panel: Compact Reference Strip (+ Docked History if Left)
    fn refs_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if self.mode == StudioMode::Video {
            self.render_history_ui(ui);
            return;
        }
        if self.history_dock == HistoryDock::Left {
            let avail_h = ui.available_height();
            let refs_max_h = (avail_h * 0.40).clamp(110.0, 240.0);

            ui.horizontal(|ui| {
                heading(ui, "REFERENCES");
                ui.label(RichText::new(format!("{}/{}", self.refs.len(), MAX_REFS)).color(DIM).size(11.0));
            });
            ui.label(RichText::new("Drag extra images for reference").color(DIM).size(10.5));
            ui.add_space(2.0);

            let mut remove = None;

            egui::ScrollArea::vertical().id_salt("refs_scroll").max_height(refs_max_h).show(ui, |ui| {
                for (i, r) in self.refs.iter().enumerate() {
                    egui::Frame::new()
                        .fill(CARD)
                        .stroke(Stroke::new(1.0, LINE))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::same(5))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let (rect, _) = ui.allocate_exact_size(vec2(44.0, 44.0), Sense::hover());
                                let p = ui.painter_at(rect);
                                p.rect_filled(rect, 4, Color32::from_rgb(15, 17, 24));
                                let s = fit(r.tex.size_vec2(), rect.shrink(2.0).size());
                                p.image(r.tex.id(), Rect::from_center_size(rect.center(), s), FULL_UV, Color32::WHITE);
                                p.rect_stroke(rect, 4, Stroke::new(1.0, LINE), StrokeKind::Inside);

                                ui.vertical(|ui| {
                                    ui.add(egui::Label::new(RichText::new(&r.name).size(10.5).strong()).truncate());
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 4.0;
                                        ui.label(RichText::new(format!("{}x{}", r.width, r.height)).size(10.0).color(DIM));
                                        if ui.small_button("X").on_hover_text("Remove reference").clicked() {
                                            remove = Some(i);
                                        }
                                    });
                                });
                            });
                        });
                    ui.add_space(2.0);
                }

                let add_btn = egui::Button::new(RichText::new("+ Add Reference").size(11.5).strong().color(ACCENT_GLOW))
                    .fill(CARD_HI)
                    .stroke(Stroke::new(1.0, ACCENT.linear_multiply(0.5)))
                    .corner_radius(6)
                    .min_size(vec2(ui.available_width(), 26.0));
                if ui.add(add_btn).clicked() {
                    self.pick_images(ctx, Target::Ref, true);
                }
            });

            if let Some(i) = remove {
                self.refs.remove(i);
            }

            ui.add_space(4.0);
            ui.separator();
            ui.add_space(4.0);

            self.render_history_ui(ui);
        } else {
            ui.horizontal(|ui| {
                heading(ui, "REFERENCES");
                ui.label(RichText::new(format!("{}/{}", self.refs.len(), MAX_REFS)).color(DIM).size(11.0));
            });
            ui.label(RichText::new("Drag extra images here for reference").color(DIM).size(11.0));
            ui.add_space(3.0);

            let mut remove = None;

            egui::ScrollArea::vertical().id_salt("refs_scroll").show(ui, |ui| {
                for (i, r) in self.refs.iter().enumerate() {
                    egui::Frame::new()
                        .fill(CARD)
                        .stroke(Stroke::new(1.0, LINE))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::same(6))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let (rect, _) = ui.allocate_exact_size(vec2(50.0, 50.0), Sense::hover());
                                let p = ui.painter_at(rect);
                                p.rect_filled(rect, 4, Color32::from_rgb(15, 17, 24));
                                let s = fit(r.tex.size_vec2(), rect.shrink(2.0).size());
                                p.image(r.tex.id(), Rect::from_center_size(rect.center(), s), FULL_UV, Color32::WHITE);
                                p.rect_stroke(rect, 4, Stroke::new(1.0, LINE), StrokeKind::Inside);

                                ui.vertical(|ui| {
                                    ui.add(egui::Label::new(RichText::new(&r.name).size(11.0).strong()).truncate());
                                    ui.horizontal(|ui| {
                                        ui.spacing_mut().item_spacing.x = 4.0;
                                        ui.label(RichText::new(format!("{}x{}", r.width, r.height)).size(10.5).color(DIM));
                                        if ui.small_button("X").on_hover_text("Remove reference").clicked() {
                                            remove = Some(i);
                                        }
                                    });
                                });
                            });
                        });
                    ui.add_space(3.0);
                }

                let add_btn = egui::Button::new(RichText::new("+ Add Reference").size(12.0).strong().color(ACCENT_GLOW))
                    .fill(CARD_HI)
                    .stroke(Stroke::new(1.0, ACCENT.linear_multiply(0.5)))
                    .corner_radius(6)
                    .min_size(vec2(ui.available_width(), 30.0));
                if ui.add(add_btn).clicked() {
                    self.pick_images(ctx, Target::Ref, true);
                }
            });

            if let Some(i) = remove {
                self.refs.remove(i);
            }
        }
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

    fn render_video_settings(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().id_salt("video_settings_scroll").show(ui, |ui| {
            card(ui, |ui| {
                heading(ui, "WORKFLOW");
                if self.active_node().is_some() {
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
                    ("360p", 640, 384, "640x384 • Fast preview, minimal VRAM"),
                    ("480p", 832, 448, "832x448 • Smooth preview, standard definition"),
                    ("720p", 1280, 704, "1280x704 • Standard HD (Recommended standard)"),
                    ("1080p", 1920, 1088, "1920x1088 • Full HD (Crisp, requires ~16GB VRAM)"),
                    ("1440p", 2560, 1408, "2560x1408 • 2K Quad HD (Requires ~22GB VRAM)"),
                    ("4K", 3840, 2176, "3840x2176 • Ultra HD (Requires 32GB+ VRAM or heavy paging)"),
                ];

                ui.horizontal_wrapped(|ui| {
                    for (label, w_land, h_land, desc) in presets {
                        let (target_w, target_h) = if is_portrait { (h_land, w_land) } else { (w_land, h_land) };
                        let is_sel = self.video_base_w == target_w && self.video_base_h == target_h;
                        let btn = egui::Button::new(RichText::new(label).strong().color(if is_sel { Color32::WHITE } else { TEXT }))
                            .fill(if is_sel { ACCENT } else { CARD_HI })
                            .stroke(Stroke::new(1.0, if is_sel { ACCENT_GLOW } else { LINE }))
                            .corner_radius(5);
                        if ui.add(btn).on_hover_text(format!("{}x{} px • {}", target_w, target_h, desc)).clicked() {
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
                let slider = egui::Slider::new(&mut self.video_scale, 0.25..=2.0)
                    .step_by(0.05)
                    .show_value(false);
                let s_resp = ui.add(slider);
                if s_resp.changed() {
                    self.video_width = (((self.video_base_w as f32 * self.video_scale) / 32.0).round() as u32 * 32).clamp(256, 4096);
                    self.video_height = (((self.video_base_h as f32 * self.video_scale) / 32.0).round() as u32 * 32).clamp(256, 4096);
                }

                ui.add_space(4.0);

                // 4. Custom dimensions & live display
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Custom:").size(11.0).color(DIM));
                    let w_resp = ui.add(egui::DragValue::new(&mut self.video_width).speed(32).range(256..=4096).prefix("W: "));
                    let h_resp = ui.add(egui::DragValue::new(&mut self.video_height).speed(32).range(256..=4096).prefix("H: "));
                    if w_resp.changed() || h_resp.changed() {
                        self.video_width = (self.video_width / 32) * 32;
                        self.video_height = (self.video_height / 32) * 32;
                        self.video_base_w = self.video_width;
                        self.video_base_h = self.video_height;
                        self.video_scale = 1.0;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Output: {} × {} px", self.video_width, self.video_height)).size(11.0).color(TEXT).strong());
                    let ratio = self.video_width as f32 / self.video_height.max(1) as f32;
                    ui.label(RichText::new(format!("({:.2}:1)", ratio)).size(10.5).color(DIM));
                });

                // 5. Dynamic VRAM estimate and excessive swap warning card
                let vram_detected = engine::detect_gpu_vram_gb();
                let est_vram = estimate_video_vram_gb(self.video_width, self.video_height, self.video_duration, self.video_fps);

                if let Some(gpu_vram) = vram_detected {
                    if est_vram > gpu_vram as f32 {
                        ui.add_space(4.0);
                        egui::Frame::new()
                            .fill(Color32::from_rgb(42, 22, 18))
                            .stroke(Stroke::new(1.0, Color32::from_rgb(220, 90, 40)))
                            .corner_radius(6)
                            .inner_margin(egui::Margin::same(8))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new("[!] High VRAM Load Warning").strong().color(AMBER));
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        ui.label(RichText::new(format!("Est: {:.1} GB / {} GB VRAM", est_vram, gpu_vram)).size(10.5).color(AMBER));
                                    });
                                });
                                ui.label(
                                    RichText::new("This resolution & duration combination exceeds your GPU's dedicated VRAM. Windows will use shared system RAM paging, causing generation steps to slow down by 10x–100x or appear stalled.\n\nTip: Reduce duration, lower the Resolution Scale slider, or choose 480p / 360p for fast generation.")
                                        .size(10.5)
                                        .color(Color32::from_rgb(235, 205, 185)),
                                );
                            });
                    } else if est_vram > (gpu_vram as f32 - 1.5) {
                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("[i] Near VRAM capacity").size(10.5).color(AMBER));
                            ui.label(RichText::new(format!("(Est. {:.1} GB / {} GB)", est_vram, gpu_vram)).size(10.5).color(DIM));
                        });
                    } else {
                        ui.add_space(2.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("[OK] Within VRAM budget").size(10.5).color(GREEN));
                            ui.label(RichText::new(format!("(Est. {:.1} GB / {} GB)", est_vram, gpu_vram)).size(10.5).color(DIM));
                        });
                    }
                } else if est_vram >= 14.0 {
                    ui.add_space(4.0);
                    ui.label(RichText::new(format!("[!] Heavy load (Est. {:.1} GB VRAM required). May cause RAM swapping on GPUs with <16GB.", est_vram)).size(10.5).color(AMBER));
                }
            });

            ui.add_space(6.0);

            card(ui, |ui| {
                heading(ui, "DURATION & FRAMERATE");
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Duration:").size(11.5).color(TEXT));
                    ui.add(egui::Slider::new(&mut self.video_duration, 1..=15).suffix("s"));
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Framerate:").size(11.5).color(TEXT));
                    for fps_opt in [24, 25, 30] {
                        let sel = self.video_fps == fps_opt;
                        if ui.selectable_label(sel, format!("{fps_opt} fps")).clicked() {
                            self.video_fps = fps_opt;
                        }
                    }
                });

                let total_frames = self.video_duration * self.video_fps;
                ui.add_space(2.0);
                ui.label(RichText::new(format!("Total frames: {} frames ({}s @ {} fps)", total_frames, self.video_duration, self.video_fps)).size(11.0).color(DIM));
            });

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

            ui.add_space(6.0);

            card(ui, |ui| {
                heading(ui, "SAMPLING & SEED");
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
                let slider = egui::Slider::new(&mut self.scale, 0.10..=2.00)
                    .step_by(0.05)
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
                    let drag_resp = ui.add(
                        egui::Label::new(RichText::new("⠿ Drag to undock").size(10.5).color(DIM))
                            .sense(egui::Sense::drag())
                    );
                    if drag_resp.hovered() {
                        ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::Grab);
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

            if self.nodes.len() > 1 {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("Clear").on_hover_text("Reset history and delete generated files").clicked() {
                        let to_remove: Vec<usize> = self.nodes.iter().skip(1).map(|n| n.id).collect();
                        for id in to_remove {
                            self.delete_history_node(id);
                        }
                        if !self.nodes.is_empty() {
                            self.select_node(self.nodes[0].id);
                        }
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
                        
                        let del_btn_rect = Rect::from_center_size(pos2(card_rect.max.x - 14.0, card_rect.center().y), vec2(18.0, 18.0));
                        let del_resp = ui.allocate_rect(del_btn_rect, Sense::click());
                        let del_hovered = del_resp.hovered();

                        if del_resp.on_hover_text("Delete from history & disk").clicked() {
                            to_delete = Some(node.id);
                        } else if card_resp.clicked() {
                            to_select = Some(node.id);
                        }
                        if card_resp.hovered() && !del_hovered {
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

                        let sub = if node.mp4_bytes.is_some() {
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

                        // Render Delete Button
                        if del_hovered {
                            p.rect_filled(del_btn_rect, 3, Color32::from_rgb(180, 45, 45));
                            p.text(del_btn_rect.center(), egui::Align2::CENTER_CENTER, "×", egui::FontId::proportional(13.0), Color32::WHITE);
                        } else {
                            p.rect_filled(del_btn_rect, 3, Color32::from_white_alpha(15));
                            p.text(del_btn_rect.center(), egui::Align2::CENTER_CENTER, "×", egui::FontId::proportional(12.0), DIM);
                        }

                        if !node.prompt.is_empty() && !del_hovered {
                            card_resp.on_hover_text(&node.prompt);
                        }

                        ui.add_space(2.0);
                    }
                }
            }
        });

        if let Some(id) = to_delete {
            self.delete_history_node(id);
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
        egui::Frame::new()
            .fill(Color32::from_rgb(20, 23, 32))
            .stroke(Stroke::new(1.0, LINE))
            .corner_radius(6)
            .inner_margin(egui::Margin::symmetric(10, 6))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let mut dl_img = sh.dl_image_model.load(Ordering::SeqCst);
                    let mut dl_vid = sh.dl_video_model.load(Ordering::SeqCst);

                    let resp_img = ui.checkbox(&mut dl_img, RichText::new("Image Model (Qwen 2.1 • ~16 GB)").size(11.0).strong());
                    if resp_img.changed() {
                        sh.dl_image_model.store(dl_img, Ordering::SeqCst);
                        let mut cfg = sh.cfg.lock().unwrap();
                        cfg.dl_image_model = dl_img;
                        cfg.save();
                        if !dl_img {
                            sh.cancel_model_dl.store(true, Ordering::SeqCst);
                        } else {
                            sh.retry.store(true, Ordering::SeqCst);
                        }
                    }

                    ui.add_space(10.0);

                    let resp_vid = ui.checkbox(&mut dl_vid, RichText::new("Video Model (LTX-2.3 • ~42 GB)").size(11.0).strong());
                    if resp_vid.changed() {
                        sh.dl_video_model.store(dl_vid, Ordering::SeqCst);
                        let mut cfg = sh.cfg.lock().unwrap();
                        cfg.dl_video_model = dl_vid;
                        cfg.save();
                        if !dl_vid {
                            sh.cancel_model_dl.store(true, Ordering::SeqCst);
                        } else {
                            sh.retry.store(true, Ordering::SeqCst);
                        }
                    }
                });
            });
    }

    fn render_setup_card(&mut self, ui: &mut egui::Ui, rect: Rect, state: &EngState) {
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 10, Color32::from_rgb(15, 17, 24));
        p.rect_stroke(rect, 10, Stroke::new(1.5, LINE), StrokeKind::Inside);

        let is_failed = matches!(state, EngState::Failed(_) | EngState::ModelDownloadFailed { .. });
        let is_model_failed = matches!(state, EngState::ModelDownloadFailed { .. });
        let card_w = (if is_failed { 620.0f32 } else { 560.0f32 }).min(rect.width() - 32.0);
        let card_h = if is_model_failed { 320.0f32 } else if is_failed { 380.0f32 } else { 300.0f32 };
        let card_rect = Rect::from_center_size(rect.center(), vec2(card_w, card_h));

        p.rect_filled(card_rect, 12, CARD);
        let stroke_color = if is_model_failed {
            AMBER.linear_multiply(0.5)
        } else if is_failed {
            RED.linear_multiply(0.4)
        } else {
            CARD_HI
        };
        p.rect_stroke(card_rect, 12, Stroke::new(1.0, stroke_color), StrokeKind::Inside);

        let mut card_ui = ui.new_child(egui::UiBuilder::new().max_rect(card_rect.shrink(22.0)));
        card_ui.vertical_centered(|ui| {
            match state {
                EngState::DownloadingEngine { step, progress, bytes_done, bytes_total, speed_mb } => {
                    ui.label(RichText::new("SETTING UP IMAGE STUDIO").size(11.0).color(ACCENT).strong());
                    ui.add_space(4.0);
                    ui.label(RichText::new("Downloading AI Image Engine").size(18.0).strong().color(TEXT));
                    ui.label(RichText::new(step).size(12.0).color(DIM));
                    ui.label(RichText::new(format!("Step 1 of 2: Portable ComfyUI environment • {:.1} MB/s", speed_mb)).size(12.0).color(DIM));
                    ui.add_space(12.0);

                    let bar = egui::ProgressBar::new(*progress)
                        .show_percentage()
                        .fill(ACCENT)
                        .animate(true);
                    ui.add_sized([card_w - 44.0, 14.0], bar);
                    ui.add_space(6.0);

                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{} / {}", engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total))).size(11.5).color(DIM));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(format!("{:.1} MB/s", speed_mb)).size(11.5).color(TEXT).strong());
                        });
                    });
                    ui.add_space(8.0);
                    self.render_model_checkboxes(ui);
                }
                EngState::ExtractingEngine { step } => {
                    ui.label(RichText::new("SETTING UP IMAGE STUDIO").size(11.0).color(ACCENT).strong());
                    ui.add_space(4.0);
                    ui.label(RichText::new("Extracting Portable Engine").size(18.0).strong().color(TEXT));
                    ui.label(RichText::new("Unpacking runtime components (this takes 1-2 minutes)...").size(12.0).color(DIM));
                    ui.add_space(12.0);

                    let spinner_bar = egui::ProgressBar::new(0.5).fill(ACCENT).animate(true);
                    ui.add_sized([card_w - 44.0, 14.0], spinner_bar);
                    ui.add_space(6.0);
                    ui.label(RichText::new(step).size(11.5).color(DIM));
                    ui.add_space(8.0);
                    self.render_model_checkboxes(ui);
                }
                EngState::DownloadingModels { current_file, file_index, total_files, file_progress, bytes_done, bytes_total, speed_mb, overall_progress } => {
                    ui.label(RichText::new("SETTING UP IMAGE STUDIO").size(11.0).color(ACCENT).strong());
                    ui.add_space(4.0);
                    ui.label(RichText::new(format!("Downloading AI Model Weights ({}/{})", file_index, total_files)).size(18.0).strong().color(TEXT));
                    ui.label(RichText::new(format!("File: {}", current_file)).size(11.5).color(DIM));
                    ui.add_space(8.0);

                    let bar = egui::ProgressBar::new(*file_progress)
                        .show_percentage()
                        .fill(ACCENT)
                        .animate(true);
                    ui.add_sized([card_w - 44.0, 14.0], bar);
                    ui.add_space(5.0);

                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{} / {}", engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total))).size(11.5).color(DIM));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(format!("{:.1} MB/s", speed_mb)).size(11.5).color(TEXT).strong());
                        });
                    });
                    ui.add_space(5.0);

                    let ov_bar = egui::ProgressBar::new(*overall_progress)
                        .fill(ACCENT.linear_multiply(0.6))
                        .text(format!("Total Download Progress: {:.0}%", overall_progress * 100.0));
                    ui.add_sized([card_w - 44.0, 10.0], ov_bar);

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        let cur_loc = self.sh.pending_models_dir.lock().unwrap().clone()
                            .unwrap_or_else(|| self.sh.paths.lock().unwrap().models.clone());
                        ui.label(RichText::new("Storage:").size(11.0).color(DIM));
                        let display_str = cur_loc.display().to_string();
                        let truncated = if display_str.len() > 36 {
                            format!("...{}", &display_str[display_str.len() - 33..])
                        } else {
                            display_str
                        };
                        ui.label(RichText::new(truncated).size(11.0).color(TEXT));
                        if ui.small_button("Change Location...").on_hover_text("Change install/model location to another drive (e.g. D:\\ImageStudio). Finished files will automatically move to this location.").clicked() {
                            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                                let target_models = folder.join("models");
                                *self.sh.pending_models_dir.lock().unwrap() = Some(target_models.clone());
                                let mut cfg = self.sh.cfg.lock().unwrap();
                                cfg.models_dir = Some(target_models.clone());
                                cfg.save();
                                self.sh.paths.lock().unwrap().models = target_models;
                            }
                        }
                    });
                    ui.add_space(6.0);
                    self.render_model_checkboxes(ui);
                }
                EngState::Starting { since, attempt } => {
                    ui.label(RichText::new("INITIALIZING BACKEND").size(11.0).color(AMBER).strong());
                    ui.add_space(4.0);
                    ui.label(RichText::new("Starting ComfyUI Engine").size(18.0).strong().color(TEXT));
                    ui.label(RichText::new(format!("Starting background process (attempt {}, {}s)...", attempt + 1, since.elapsed().as_secs())).size(12.0).color(DIM));
                    ui.add_space(14.0);

                    let bar = egui::ProgressBar::new(0.6).fill(AMBER).animate(true);
                    ui.add_sized([card_w - 44.0, 12.0], bar);
                }
                EngState::RepairingEngine { step } => {
                    ui.label(RichText::new("REPAIRING ENVIRONMENT").size(11.0).color(AMBER).strong());
                    ui.add_space(4.0);
                    ui.label(RichText::new("Installing Missing Dependencies").size(18.0).strong().color(TEXT));
                    ui.label(RichText::new(step).size(12.0).color(DIM));
                    ui.add_space(14.0);

                    let spinner_bar = egui::ProgressBar::new(0.5).fill(AMBER).animate(true);
                    ui.add_sized([card_w - 44.0, 14.0], spinner_bar);
                }
                EngState::ModelDownloadFailed { current_file, error } => {
                    ui.label(RichText::new("DOWNLOAD PAUSED").size(11.0).color(AMBER).strong());
                    ui.add_space(3.0);
                    ui.label(RichText::new("Model Download Interrupted").size(18.0).strong().color(TEXT));
                    ui.label(RichText::new("The local ComfyUI engine is working properly. A model file download was interrupted or throttled.").size(11.5).color(DIM));
                    ui.add_space(8.0);

                    egui::Frame::new()
                        .fill(Color32::from_rgb(10, 12, 16))
                        .stroke(Stroke::new(1.0, LINE))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            egui::ScrollArea::vertical()
                                .id_salt("model_dl_err_log_scroll")
                                .max_height(80.0)
                                .stick_to_bottom(true)
                                .show(ui, |ui| {
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new(format!("File: {current_file}\n{error}"))
                                                .monospace()
                                                .size(11.0)
                                                .color(Color32::from_rgb(255, 180, 100)),
                                        )
                                        .selectable(true),
                                    );
                                });
                        });
                    ui.add_space(10.0);

                    let retry_btn = egui::Button::new(
                        RichText::new("Retry Model Download")
                            .size(12.5)
                            .strong()
                            .color(Color32::WHITE),
                    )
                    .fill(ACCENT)
                    .stroke(Stroke::new(1.0, ACCENT_GLOW))
                    .corner_radius(6);
                    if ui.add(retry_btn).on_hover_text("Resumes downloading missing models from the fastest available mirror").clicked() {
                        self.sh.retry.store(true, Ordering::SeqCst);
                    }
                    ui.add_space(6.0);

                    ui.horizontal(|ui| {
                        if ui.button("Open Models Folder").clicked() {
                            engine::open_models_folder(&self.sh);
                        }

                        if ui.button("Dismiss / Check Status").clicked() {
                            self.sh.retry.store(true, Ordering::SeqCst);
                        }
                    });
                }
                EngState::Failed(msg) => {
                    ui.label(RichText::new("SETUP INTERRUPTED").size(11.0).color(RED).strong());
                    ui.add_space(3.0);
                    ui.label(RichText::new("Engine Failed to Start").size(18.0).strong().color(TEXT));
                    ui.label(RichText::new("The local ComfyUI installation failed to launch (missing dependencies or incompatible version).").size(11.5).color(DIM));
                    ui.add_space(8.0);

                    egui::Frame::new()
                        .fill(Color32::from_rgb(10, 12, 16))
                        .stroke(Stroke::new(1.0, LINE))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::same(8))
                        .show(ui, |ui| {
                            egui::ScrollArea::vertical()
                                .id_salt("err_log_scroll")
                                .max_height(105.0)
                                .stick_to_bottom(true)
                                .show(ui, |ui| {
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new(msg)
                                                .monospace()
                                                .size(11.0)
                                                .color(Color32::from_rgb(255, 120, 120)),
                                        )
                                        .selectable(true),
                                    );
                                });
                        });
                    ui.add_space(10.0);

                    let clean_btn = egui::Button::new(
                        RichText::new("Download Clean Dedicated Engine (Recommended)")
                            .size(12.5)
                            .strong()
                            .color(Color32::WHITE),
                    )
                    .fill(ACCENT)
                    .stroke(Stroke::new(1.0, ACCENT_GLOW))
                    .corner_radius(6);
                    if ui.add(clean_btn).on_hover_text("Bypasses broken local installs and automatically downloads a clean, tested portable engine").clicked() {
                        engine::reset_to_clean_engine(&self.sh);
                    }
                    ui.add_space(6.0);

                    ui.horizontal(|ui| {
                        let rep_btn = egui::Button::new(
                            RichText::new("Auto-Repair Environment (pip)")
                                .size(11.5)
                                .color(TEXT),
                        )
                        .fill(CARD_HI)
                        .stroke(Stroke::new(1.0, LINE))
                        .corner_radius(6);
                        if ui.add(rep_btn).on_hover_text("Runs pip install (async-timeout, aiohttp, requirements.txt) in the detected Python environment").clicked() {
                            engine::repair_environment(self.sh.clone(), ui.ctx().clone());
                        }

                        if ui.button("Browse Different Folder").clicked() {
                            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                                let mut cfg = self.sh.cfg.lock().unwrap();
                                cfg.comfy_dir = Some(folder);
                                cfg.save();
                                self.sh.retry.store(true, Ordering::SeqCst);
                            }
                        }

                        if ui.button("Retry").clicked() {
                            self.sh.retry.store(true, Ordering::SeqCst);
                        }
                    });
                }
                EngState::Detecting => {
                    ui.label(RichText::new("DETECTING ENGINE").size(11.0).color(AMBER).strong());
                    ui.add_space(4.0);
                    ui.label(RichText::new("Checking for Local ComfyUI...").size(18.0).strong().color(TEXT));
                    ui.add_space(14.0);
                    let bar = egui::ProgressBar::new(0.3).fill(AMBER).animate(true);
                    ui.add_sized([card_w - 44.0, 12.0], bar);
                }
                EngState::Ready { .. } => {}
            }
        });
    }

    fn render_video_canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, max_size: Vec2) {
        let player = match self.video_player.as_mut() {
            Some(p) => p,
            None => return,
        };

        player.update(ctx);

        let wants_kb = ctx.egui_wants_keyboard_input();

        let space_pressed = !wants_kb && ctx.input(|i| i.key_pressed(egui::Key::Space));
        let left_pressed = !wants_kb && ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft));
        let right_pressed = !wants_kb && ctx.input(|i| i.key_pressed(egui::Key::ArrowRight));

        if space_pressed {
            player.toggle_play();
        }
        if left_pressed {
            let prev = player.current_frame.saturating_sub(1);
            player.seek(prev);
        }
        if right_pressed {
            let next = (player.current_frame + 1).min(player.frames.len().saturating_sub(1));
            player.seek(next);
        }

        let mut close_video = false;
        let mut save_video = false;
        let mut extract_frame = false;

        let cur_frame_idx = player.current_frame;
        let total_frames = player.frames.len();
        let fps = player.fps.max(1.0);
        let cur_sec = cur_frame_idx as f32 / fps;
        let tot_sec = total_frames as f32 / fps;
        let p_w = player.width;
        let p_h = player.height;

        ui.horizontal(|ui| {
            heading(ui, &format!("Video Player   {} × {} px   •   {:.1}s @ {:.0} fps", p_w, p_h, tot_sec, fps));

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Close Video").on_hover_text("Close player and return to image canvas").clicked() {
                    close_video = true;
                }
                if ui.button("Save MP4 as...").on_hover_text("Export video file").clicked() {
                    save_video = true;
                }
                if ui.button("Use Frame as Base Image").on_hover_text("Extract current video frame as base image and switch to Image mode").clicked() {
                    extract_frame = true;
                }

                let zoom_txt = format!("{:.0}%", self.canvas_zoom * 100.0);
                if ui.button(zoom_txt).on_hover_text("Click to reset zoom and center video").clicked() {
                    self.canvas_zoom = 1.0;
                    self.canvas_pan = vec2(0.0, 0.0);
                }
            });
        });

        let canvas_area = max_size - vec2(0.0, 96.0);
        let (canvas_rect, resp) = ui.allocate_exact_size(canvas_area, Sense::click_and_drag());

        let target_size = vec2(p_w as f32, p_h as f32);
        let fit_size = fit(target_size, canvas_area);

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
                let new_zoom = (old_zoom * zoom_factor).clamp(0.15, 25.0);

                if let Some(mouse) = resp.hover_pos() {
                    let center = canvas_rect.center();
                    let mouse_rel = mouse - (center + self.canvas_pan);
                    self.canvas_pan -= mouse_rel * (new_zoom / old_zoom - 1.0);
                }
                self.canvas_zoom = new_zoom;
            }
        }

        // Pan by dragging
        if resp.dragged() {
            self.canvas_pan += resp.drag_delta();
        }
        if resp.double_clicked() {
            self.canvas_zoom = 1.0;
            self.canvas_pan = vec2(0.0, 0.0);
        } else if resp.clicked() {
            player.toggle_play();
        }

        let cur_size = fit_size * self.canvas_zoom;
        let cur_center = canvas_rect.center() + self.canvas_pan;
        let image_rect = Rect::from_center_size(cur_center, cur_size);

        let p = ui.painter().with_clip_rect(canvas_rect);
        p.rect_filled(canvas_rect, 8, Color32::from_rgb(14, 16, 22));
        p.rect_stroke(canvas_rect, 8, Stroke::new(1.0, LINE), StrokeKind::Inside);

        if let Some(tex) = player.current_texture(ctx) {
            p.image(tex.id(), image_rect, FULL_UV, Color32::WHITE);
            p.rect_stroke(image_rect, 0, Stroke::new(1.0, Color32::from_white_alpha(30)), StrokeKind::Outside);
        }

        ui.add_space(6.0);

        // Player Controls Bar at bottom of canvas
        card(ui, |ui| {
            ui.vertical(|ui| {
                // 1. Full-length scrub bar covering the entire width of the player
                let cur_f = player.current_frame;
                let max_f = player.frames.len().saturating_sub(1);
                let full_w = ui.available_width();
                let (track_rect, scrub_resp) = ui.allocate_exact_size(vec2(full_w, 18.0), Sense::click_and_drag());
                if ui.is_rect_visible(track_rect) {
                    let p = ui.painter();
                    let is_hovered = scrub_resp.hovered();
                    let is_dragged = scrub_resp.dragged();
                    if is_hovered || is_dragged {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                    }

                    let track_y = track_rect.center().y;
                    let track_h = if is_hovered || is_dragged { 6.0 } else { 4.0 };
                    let track_bg_rect = Rect::from_center_size(track_rect.center(), vec2(track_rect.width(), track_h));
                    p.rect_filled(track_bg_rect, 2.0, Color32::from_rgb(32, 36, 48));

                    let progress_ratio = if max_f > 0 {
                        (cur_f as f32 / max_f as f32).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let played_w = track_rect.width() * progress_ratio;
                    let played_rect = Rect::from_min_max(
                        pos2(track_rect.left(), track_y - track_h * 0.5),
                        pos2(track_rect.left() + played_w, track_y + track_h * 0.5),
                    );
                    p.rect_filled(played_rect, 2.0, ACCENT);

                    let playhead_x = track_rect.left() + played_w;
                    let thumb_r = if is_dragged { 7.0 } else if is_hovered { 6.0 } else { 4.5 };
                    let thumb_pos = pos2(playhead_x, track_y);
                    p.circle_filled(thumb_pos, thumb_r, Color32::WHITE);
                    p.circle_stroke(thumb_pos, thumb_r, Stroke::new(1.5, ACCENT_GLOW));

                    if scrub_resp.dragged() || scrub_resp.clicked() {
                        if let Some(pos) = scrub_resp.interact_pointer_pos() {
                            let rel_x = ((pos.x - track_rect.left()) / track_rect.width()).clamp(0.0, 1.0);
                            let target_frame = (rel_x * max_f as f32).round() as usize;
                            if scrub_resp.clicked() || target_frame != player.current_frame {
                                player.seek(target_frame);
                            }
                        }
                    }
                }

                ui.add_space(2.0);

                // 2. Transport controls, status counters, audio controls and loop
                ui.horizontal(|ui| {
                    let play_txt = if player.is_playing { "Pause" } else { "Play" };
                    let play_btn = egui::Button::new(RichText::new(play_txt).size(12.0).strong().color(Color32::WHITE))
                        .fill(if player.is_playing { ACCENT } else { CARD_HI })
                        .stroke(Stroke::new(1.0, if player.is_playing { ACCENT_GLOW } else { LINE }))
                        .corner_radius(6)
                        .min_size(vec2(54.0, 28.0));
                    if ui.add(play_btn).on_hover_text("Play / Pause (Space)").clicked() {
                        player.toggle_play();
                    }

                    let step_back_btn = egui::Button::new(RichText::new("< 1f").size(11.5).color(TEXT))
                        .fill(CARD_HI)
                        .stroke(Stroke::new(1.0, LINE))
                        .corner_radius(6)
                        .min_size(vec2(36.0, 28.0));
                    if ui.add(step_back_btn).on_hover_text("Step 1 frame back (Left Arrow)").clicked() {
                        let prev = player.current_frame.saturating_sub(1);
                        player.seek(prev);
                    }

                    let step_fwd_btn = egui::Button::new(RichText::new("1f >").size(11.5).color(TEXT))
                        .fill(CARD_HI)
                        .stroke(Stroke::new(1.0, LINE))
                        .corner_radius(6)
                        .min_size(vec2(36.0, 28.0));
                    if ui.add(step_fwd_btn).on_hover_text("Step 1 frame forward (Right Arrow)").clicked() {
                        let next = (player.current_frame + 1).min(player.frames.len().saturating_sub(1));
                        player.seek(next);
                    }

                    ui.add_space(6.0);

                    let time_str = format!("{:02}:{:04.1} / {:02}:{:04.1}", (cur_sec / 60.0) as u32, cur_sec % 60.0, (tot_sec / 60.0) as u32, tot_sec % 60.0);
                    ui.label(RichText::new(time_str).size(12.0).monospace().color(TEXT));

                    let frame_str = format!("(frame {}/{})", cur_frame_idx + 1, total_frames);
                    ui.label(RichText::new(frame_str).size(10.5).color(DIM));

                    ui.label(RichText::new("• Start frame for next shot").size(10.5).color(ACCENT));

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
                            let v_resp = ui.add_sized([64.0, 16.0], vol_slider);
                            if v_resp.changed() {
                                audio.set_volume(vol);
                            }

                            let mute_txt = if audio.is_muted || audio.volume == 0.0 { "Unmute" } else { "Mute" };
                            let mute_btn = egui::Button::new(RichText::new(mute_txt).size(11.5).color(TEXT))
                                .fill(CARD_HI)
                                .stroke(Stroke::new(1.0, LINE))
                                .corner_radius(6);
                            if ui.add(mute_btn).on_hover_text("Mute / Unmute audio").clicked() {
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

        if close_video {
            self.video_player = None;
            return;
        }

        if save_video {
            let filename = player.filename.clone();
            let mp4_data = player.mp4_bytes.clone();
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

        if extract_frame {
            let cur_img = player.frames[player.current_frame.min(player.frames.len() - 1)].clone();
            let [f_w, f_h] = cur_img.size;
            let mut png_bytes = Vec::new();
            if let Some(rgba) = image::RgbaImage::from_raw(f_w as u32, f_h as u32, cur_img.as_raw().to_vec()) {
                let _ = image::DynamicImage::ImageRgba8(rgba)
                    .write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png);
            }
            let id = self.next_node_id;
            self.next_node_id += 1;
            let tex = ctx.load_texture(format!("extracted_frame_{id}"), cur_img, egui::TextureOptions::LINEAR);
            let node = HistoryNode {
                id,
                parent_id: self.active_node_id,
                label: format!("Video Frame {}", player.current_frame + 1),
                prompt: self.prompt.clone(),
                png: png_bytes,
                tex,
                width: f_w as u32,
                height: f_h as u32,
                step_num: self.active_node().map(|n| n.step_num + 1).unwrap_or(0),
                output_filename: String::new(),
                output_subfolder: String::new(),
                temp_files: Vec::new(),
                mp4_bytes: None,
            };
            self.nodes.push(node);
            self.select_node(id);
            self.mode = StudioMode::Image;
            self.video_player = None;
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
                let ids: Vec<usize> = self.nodes.iter().map(|n| n.id).collect();
                for id in ids {
                    self.delete_history_node(id);
                }
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
                    let new_zoom = (old_zoom * zoom_factor).clamp(0.15, 25.0);

                    if let Some(mouse) = resp.hover_pos() {
                        let center = canvas_rect.center();
                        let mouse_rel = mouse - (center + self.canvas_pan);
                        self.canvas_pan -= mouse_rel * (new_zoom / old_zoom - 1.0);
                    }
                    self.canvas_zoom = new_zoom;
                }
            }

            // Pan by dragging (when not dragging compare split)
            if !self.compare_mode && resp.dragged() {
                self.canvas_pan += resp.drag_delta();
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

                p.rect_stroke(image_rect, 0, Stroke::new(1.0, ACCENT_GLOW), StrokeKind::Outside);

                let badge_text = if self.is_swapping_detected {
                    if let Some((v, max)) = self.progress {
                        format!("[!] VRAM SWAPPING SLOWDOWN ({}/{} • {:.1}s/step)", v, max, self.step_duration_ema)
                    } else {
                        "[!] VRAM SWAPPING SLOWDOWN".to_string()
                    }
                } else if let Some((v, max)) = self.progress {
                    format!("GENERATING (Step {}/{})", v, max)
                } else {
                    "GENERATING...".to_string()
                };
                let g = p.layout_no_wrap(badge_text, egui::FontId::proportional(11.0), Color32::WHITE);
                let badge_pos = image_rect.left_top() + vec2(10.0, 10.0);
                let r = egui::Align2::LEFT_TOP.anchor_size(badge_pos, g.size()).expand2(vec2(8.0, 4.0));
                p.rect_filled(r, 4, Color32::from_black_alpha(190));
                p.rect_stroke(r, 4, Stroke::new(1.0, if self.is_swapping_detected { AMBER } else { ACCENT }), StrokeKind::Inside);
                p.galley(r.shrink2(vec2(8.0, 4.0)).min, g, Color32::WHITE);
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
            let st = self.sh.state.lock().unwrap().clone();
            let is_setup_in_progress = matches!(
                st,
                EngState::DownloadingEngine { .. }
                    | EngState::ExtractingEngine { .. }
                    | EngState::DownloadingModels { .. }
                    | EngState::RepairingEngine { .. }
                    | EngState::ModelDownloadFailed { .. }
                    | EngState::Failed(_)
            );
            if is_setup_in_progress {
                self.render_setup_card(ui, rect, &st);
            } else {
                let hover = resp.hovered();
                let p = ui.painter_at(rect);
                p.rect_filled(rect, 10, Color32::from_rgb(15, 17, 24));
                p.rect_stroke(rect, 10, Stroke::new(1.5, if hover { ACCENT } else { LINE }), StrokeKind::Inside);

                let c = rect.center();
                if self.mode == StudioMode::Video {
                    p.text(
                        c - vec2(0.0, 36.0),
                        egui::Align2::CENTER_CENTER,
                        "Video Generation Mode (LTX-Video 2.3)",
                        egui::FontId::proportional(19.0),
                        TEXT,
                    );
                    p.text(
                        c - vec2(0.0, 10.0),
                        egui::Align2::CENTER_CENTER,
                        "Enter a prompt below for Text-to-Video, or paste an image for Image-to-Video!",
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
                        "Or simply enter a prompt below to generate an image from scratch!",
                        egui::FontId::proportional(13.0),
                        ACCENT_GLOW,
                    );
                }

                let btn_rect = Rect::from_center_size(c + vec2(0.0, 26.0), vec2(if self.mode == StudioMode::Video { 340.0 } else { 270.0 }, 34.0));
                let paste_hover = btn_rect.contains(ui.ctx().input(|i| i.pointer.hover_pos().unwrap_or_default()));
                p.rect_filled(btn_rect, 6, if paste_hover { ACCENT } else { CARD_HI });
                p.rect_stroke(btn_rect, 6, Stroke::new(1.0, if paste_hover { ACCENT_GLOW } else { LINE }), StrokeKind::Inside);
                let btn_label = if self.mode == StudioMode::Video {
                    "📋 Paste Image for Image-to-Video (Ctrl+V)"
                } else {
                    "📋 Paste Image from Clipboard (Ctrl+V)"
                };
                p.text(btn_rect.center(), egui::Align2::CENTER_CENTER, btn_label, egui::FontId::proportional(12.5), Color32::WHITE);

                if paste_hover && ui.ctx().input(|i| i.pointer.any_pressed()) {
                    let _ = self.paste_from_clipboard(ctx);
                } else if resp.clicked() {
                    self.pick_images(ctx, Target::Base, false);
                }
            }
        }

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

            let canvas_rect = ui.max_rect();
            let margin = 14.0;
            egui::Area::new(egui::Id::new("floating_canvas_generation_status"))
                .fixed_pos(pos2(canvas_rect.right() - margin, canvas_rect.bottom() - margin))
                .pivot(egui::Align2::RIGHT_BOTTOM)
                .order(egui::Order::Foreground)
                .interactable(false)
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
                            });
                        });
                });
        }
    }

    // Bottom Docked ChatGPT-style Prompt & Action Bar
    fn bottom_dock(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let err_clone = self.error.clone();
        if let Some(err) = err_clone {
            let mut dismiss = false;
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("Error: {err}")).color(RED).size(12.0));
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
        let is_starting = matches!(st, EngState::Starting { .. } | EngState::Detecting);
        if is_starting && !self.queue.is_empty() {
            let attempt_str = match st {
                EngState::Starting { since, attempt } => {
                    format!("Starting engine in background (attempt {}, {}s)...", attempt + 1, since.elapsed().as_secs())
                }
                _ => "Starting background engine...".to_string(),
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
                    let paste_icon_btn = egui::Button::new(RichText::new("📋 Paste").size(12.0).color(TEXT))
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
                        if self.active_node().is_some() {
                            "Describe motion, camera action, or animation for this image... Enter to generate"
                        } else {
                            "Describe a video scene, camera motion, and action... Enter to generate"
                        }
                    } else if self.active_node().is_some() {
                        "Describe your edit instructions... Enter to generate, Shift+Enter for newline"
                    } else {
                        "Describe an image to generate from scratch, or paste/drop an image..."
                    };
                    let text_edit = egui::TextEdit::multiline(&mut self.prompt)
                        .id(prompt_id)
                        .desired_rows(2)
                        .desired_width((ui.available_width() - 320.0).max(180.0))
                        .hint_text(hint);
                    let text_resp = ui.add(text_edit);

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

                    let st = self.sh.state.lock().unwrap().clone();
                    let is_starting = matches!(st, EngState::Starting { .. } | EngState::Detecting);

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
                        let p_text = self.prompt.trim().to_string();
                        if !p_text.is_empty() {
                            if self.generating || is_starting {
                                if self.can_queue().is_ok() {
                                    self.queue.push(QueuedItem { prompt: p_text });
                                    self.prompt.clear();
                                    if is_starting {
                                        self.status = "Queued (starting engine...)".into();
                                    }
                                }
                            } else if self.can_generate().is_ok() {
                                self.start_generation(ctx);
                            }
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (is_ok, btn_text, tip) = if self.generating {
                            let q_can = self.can_queue();
                            let ok = q_can.is_ok() && !self.prompt.trim().is_empty();
                            (ok, "Queue", q_can.err())
                        } else if is_starting {
                            let ok = !self.prompt.trim().is_empty() && self.can_queue().is_ok();
                            (ok, "Generate", Some("Engine is starting; your prompt will be queued and will generate immediately."))
                        } else {
                            let g_can = self.can_generate();
                            let ok = g_can.is_ok() && !self.prompt.trim().is_empty();
                            (ok, "Generate", g_can.err())
                        };

                        let send_btn = egui::Button::new(
                            RichText::new(btn_text)
                                .size(13.5)
                                .strong()
                                .color(if is_ok { Color32::WHITE } else { DIM }),
                        )
                        .fill(if is_ok { ACCENT } else { CARD })
                        .stroke(Stroke::new(1.0, if is_ok { ACCENT_GLOW } else { LINE }))
                        .corner_radius(8)
                        .min_size(vec2(100.0, 36.0));

                        let resp = ui.add_enabled(is_ok, send_btn);
                        if resp.clicked() {
                            let p_text = self.prompt.trim().to_string();
                            if !p_text.is_empty() {
                                if self.generating || is_starting {
                                    if self.can_queue().is_ok() {
                                        self.queue.push(QueuedItem { prompt: p_text });
                                        self.prompt.clear();
                                        if is_starting {
                                            self.status = "Queued (starting engine...)".into();
                                        }
                                    }
                                } else if self.can_generate().is_ok() {
                                    self.start_generation(ctx);
                                }
                            }
                        }
                        if let Some(why) = tip {
                            resp.on_hover_text(why);
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
                                    if ui.add(vid_btn).on_hover_text("Switch to Video Generation (LTX-Video 2.3)").clicked() {
                                        self.mode = StudioMode::Video;
                                        if !self.has_video_models() && !self.sh.dl_video_model.load(Ordering::SeqCst) {
                                            self.show_video_dl_modal = true;
                                        }
                                    }
                                });
                            });

                        if self.generating {
                            ui.add_space(6.0);
                            let cancel_btn = egui::Button::new(RichText::new("Cancel").size(13.0).strong().color(RED))
                                .fill(CARD)
                                .stroke(Stroke::new(1.0, RED.linear_multiply(0.4)))
                                .corner_radius(8)
                                .min_size(vec2(75.0, 36.0));
                            if ui.add(cancel_btn).clicked() {
                                self.cancel.store(true, Ordering::SeqCst);
                                self.queue.clear();
                                self.generation_retries = 0;
                                self.status = "Cancelling...".into();
                                if self.prompt.trim().is_empty() && !self.generating_prompt.trim().is_empty() {
                                    self.prompt = self.generating_prompt.clone();
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
                            ui.add_space(4.0);

                            if !self.queue.is_empty() {
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

    fn setup_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_setup;
        egui::Window::new("Setup & diagnostics")
            .open(&mut open)
            .default_width(640.0)
            .default_height(580.0)
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
                            EngState::DownloadingEngine { step, progress, bytes_done, bytes_total, speed_mb } => {
                                format!("{step} ({:.1}%) - {} / {} @ {:.1} MB/s", progress * 100.0, engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total), speed_mb)
                            }
                            EngState::ExtractingEngine { step } => step.clone(),
                            EngState::RepairingEngine { step } => format!("Repairing: {step}"),
                            EngState::DownloadingModels { current_file, file_index, total_files, file_progress, bytes_done, bytes_total, speed_mb, .. } => {
                                format!("Downloading Model [{}/{}] {} ({:.1}%) - {} / {} @ {:.1} MB/s", file_index, total_files, current_file, file_progress * 100.0, engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total), speed_mb)
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
                                    ui.label(RichText::new("(8GB GPU detected - 8GB W4A8 profile recommended)").size(11.0).color(AMBER));
                                }
                            } else {
                                ui.label(RichText::new("Auto-detecting...").color(DIM));
                            }
                        });

                        let all_profiles = engine::Profile::all();
                        let cur_profile_id = self.profile.id.clone();
                        for p in all_profiles {
                            let active = p.id == cur_profile_id;
                            ui.horizontal(|ui| {
                                let b = egui::Button::new(
                                    RichText::new(&p.name).size(12.0).color(if active { Color32::WHITE } else { TEXT })
                                )
                                .fill(if active { ACCENT } else { CARD_HI })
                                .stroke(Stroke::new(1.0, if active { ACCENT_GLOW } else { LINE }))
                                .corner_radius(4);

                                if ui.add(b).clicked() && !active {
                                    let mut cfg = self.sh.cfg.lock().unwrap();
                                    cfg.active_profile = Some(p.id.clone());
                                    cfg.save();
                                    *self.sh.active_profile.lock().unwrap() = p.clone();
                                    self.profile = p.clone();
                                    self.sh.retry.store(true, Ordering::SeqCst);
                                }
                                ui.label(RichText::new(format!("(Recommended: {}GB VRAM)", p.recommended_vram_gb)).size(11.0).color(DIM));
                            });
                            ui.label(RichText::new(&p.description).size(10.5).color(DIM));
                            ui.add_space(2.0);
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
                            ui.label("");
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
                        heading(ui, "MODEL FILES STATUS");
                        let vid_ok = self.has_video_models();
                        ui.label(
                            RichText::new(format!("{}  LTX-Video 2.3 Checkpoint (ltx-2.3-22b-dev-fp8.safetensors)", if vid_ok { "found  " } else { "MISSING" }))
                                .color(if vid_ok { GREEN } else { RED }),
                        );
                        for (name, ok) in self.sh.models_found.lock().unwrap().iter() {
                            ui.label(
                                RichText::new(format!("{}  {}", if *ok { "found  " } else { "MISSING" }, name))
                                    .color(if *ok { GREEN } else { RED }),
                            );
                        }
                        ui.horizontal(|ui| {
                            if ui.button("Retry Setup / Restart Engine").clicked() {
                                self.sh.retry.store(true, Ordering::SeqCst);
                                self.sh.restart.store(true, Ordering::SeqCst);
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
    }

    fn video_dl_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_video_dl_modal;
        let mut should_close = false;
        let center = ctx.content_rect().center();
        egui::Window::new("LTX-Video 2.3 Model Setup")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(center)
            .fixed_size(vec2(500.0, 240.0))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(4.0);
                    ui.label(RichText::new("Download Video Generation Models").size(17.0).strong().color(TEXT));
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new("LTX-Video 2.3 enables high-fidelity 720p/1080p video generation from text or images.\n\nTotal download size: ~42 GB\n• LTX-2.3 22B Dev Checkpoint (FP8, ~29 GB)\n• Gemma 3 12B IT Text Encoder (~9.4 GB)\n• Spatial Upscaler x2 & Dynamic LoRAs (~4.3 GB)")
                            .size(12.0)
                            .color(DIM),
                    );
                    ui.add_space(14.0);

                    let is_dl = self.sh.dl_video_model.load(Ordering::SeqCst);
                    if is_dl {
                        ui.label(RichText::new("Video models are currently downloading in the background.").color(GREEN).strong());
                        ui.add_space(8.0);
                        if ui.button("Close").clicked() {
                            should_close = true;
                        }
                    } else {
                        ui.horizontal(|ui| {
                            let dl_btn = egui::Button::new(RichText::new("📥 Start Download (~42 GB)").strong().color(Color32::WHITE))
                                .fill(ACCENT)
                                .min_size(vec2(180.0, 32.0));
                            if ui.add(dl_btn).clicked() {
                                self.sh.dl_video_model.store(true, Ordering::SeqCst);
                                let mut cfg = self.sh.cfg.lock().unwrap();
                                cfg.dl_video_model = true;
                                cfg.save();
                                self.sh.retry.store(true, Ordering::SeqCst);
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

    fn downloads_dropdown_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_downloads_menu;
        let mut should_close = false;
        let screen_rect = ctx.content_rect();
        // Position at top-right near the status pill (e.g., right margin 24px, top 48px)
        let default_pos = egui::pos2((screen_rect.max.x - 390.0).max(10.0), 48.0);

        egui::Window::new("📥 Active Downloads")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .fixed_size(vec2(380.0, 320.0))
            .default_pos(default_pos)
            .show(ctx, |ui| {
                ui.add_space(2.0);
                let state = self.sh.state.lock().unwrap().clone();
                match state {
                    EngState::DownloadingModels { ref current_file, file_index, total_files, file_progress, bytes_done, bytes_total, speed_mb, overall_progress } => {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Downloading Model Files").strong().color(TEXT).size(14.0));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(format!("{:.1} MB/s", speed_mb)).strong().color(ACCENT));
                            });
                        });
                        ui.add_space(4.0);

                        // Overall progress bar
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(format!("Overall: {:.0}% ({}/{})", overall_progress * 100.0, file_index, total_files)).size(11.0).color(DIM));
                        });
                        let bar = egui::ProgressBar::new(overall_progress).show_percentage().fill(ACCENT);
                        ui.add_sized([ui.available_width(), 12.0], bar);
                        ui.add_space(8.0);
                        ui.separator();
                        ui.add_space(4.0);

                        // Current active file card
                        card(ui, |ui| {
                            ui.label(RichText::new(format!("Current: {current_file}")).strong().color(TEXT).size(12.0));
                            ui.add_space(2.0);
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(format!("{} / {}", engine::format_bytes(bytes_done), engine::format_bytes(bytes_total))).size(11.0).color(DIM));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.label(RichText::new(format!("{:.1}%", file_progress * 100.0)).size(11.0).color(AMBER));
                                });
                            });
                            let file_bar = egui::ProgressBar::new(file_progress).fill(AMBER);
                            ui.add_sized([ui.available_width(), 10.0], file_bar);
                        });

                        ui.add_space(6.0);
                        heading(ui, "MODEL QUEUE");
                        egui::ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
                            let missing = engine::Profile::missing_models(&self.sh);
                            if missing.is_empty() {
                                ui.label(RichText::new("All remaining models ready.").size(11.0).color(GREEN));
                            } else {
                                for (i, m) in missing.iter().enumerate() {
                                    let is_active = m.filename == *current_file;
                                    let (col, badge) = if is_active {
                                        (AMBER, "DOWNLOADING")
                                    } else if i < file_index {
                                        (GREEN, "COMPLETED")
                                    } else {
                                        (DIM, "QUEUED")
                                    };
                                    ui.horizontal(|ui| {
                                        ui.label(RichText::new(&m.filename).size(11.0).color(if is_active { TEXT } else { DIM }));
                                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                            ui.label(RichText::new(badge).size(10.0).color(col).strong());
                                            ui.label(RichText::new(engine::format_bytes(m.size_bytes)).size(10.0).color(DIM));
                                        });
                                    });
                                }
                            }
                        });
                    }
                    EngState::DownloadingEngine { ref step, progress, bytes_done, bytes_total, speed_mb } => {
                        ui.label(RichText::new("Downloading Engine Component").strong().color(TEXT).size(14.0));
                        ui.add_space(4.0);
                        ui.label(RichText::new(step).size(11.5).color(DIM));
                        ui.add_space(4.0);
                        let bar = egui::ProgressBar::new(progress).show_percentage().fill(ACCENT);
                        ui.add_sized([ui.available_width(), 14.0], bar);
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(format!("{} / {}", engine::format_bytes(bytes_done), engine::format_bytes(bytes_total))).size(11.0).color(DIM));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(format!("{:.1} MB/s", speed_mb)).strong().color(ACCENT));
                            });
                        });
                    }
                    _ => {
                        ui.vertical_centered(|ui| {
                            ui.add_space(20.0);
                            ui.label(RichText::new("No active background downloads.").size(13.0).color(DIM));
                            ui.add_space(8.0);
                            if ui.button("Open Settings").clicked() {
                                self.show_setup = true;
                                should_close = true;
                            }
                        });
                    }
                }
            });
        self.show_downloads_menu = open && !should_close;
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.drain(&ctx);
        self.handle_input(&ctx);

        let starting = matches!(
            *self.sh.state.lock().unwrap(),
            EngState::Starting { .. }
                | EngState::Detecting
                | EngState::DownloadingEngine { .. }
                | EngState::ExtractingEngine { .. }
                | EngState::DownloadingModels { .. }
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
        egui::Panel::left("refs_panel")
            .default_size(230.0)
            .min_size(170.0)
            .max_size(360.0)
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
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin::same(12)))
            .show(ui, |ui| {
                self.center_canvas(ui, &ctx);
            });

        // Floating Windows: Setup and Floating History (if undocked)
        self.setup_window(&ctx);
        self.floating_history_window(&ctx);
        if self.show_video_dl_modal {
            self.video_dl_window(&ctx);
        }
        if self.show_downloads_menu {
            self.downloads_dropdown_window(&ctx);
        }

        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("drop_overlay")));
            let r = ctx.content_rect();
            painter.rect_filled(r, 0, Color32::from_rgba_unmultiplied(12, 14, 19, 215));
            painter.rect_stroke(r.shrink(18.0), 16, Stroke::new(3.0, ACCENT), StrokeKind::Inside);
            let msg = if self.active_node().is_none() {
                "Drop image here to load onto canvas"
            } else {
                "Drop images to add as references"
            };
            painter.text(r.center(), egui::Align2::CENTER_CENTER, msg, egui::FontId::proportional(22.0), TEXT);
        }
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
        use super::estimate_video_vram_gb;
        let vram_720p_5s = estimate_video_vram_gb(1280, 720, 5, 24);
        assert!(vram_720p_5s <= 12.5, "720p 5s should fit within ~12GB VRAM");

        let vram_720p_9s = estimate_video_vram_gb(1280, 720, 9, 24);
        assert!(vram_720p_9s > 13.0, "720p 9s should exceed 12GB VRAM and trigger warning");

        let vram_360p_5s = estimate_video_vram_gb(640, 384, 5, 24);
        assert!(vram_360p_5s < 10.5, "360p 5s should require minimal VRAM");

        let vram_4k_5s = estimate_video_vram_gb(3840, 2160, 5, 24);
        assert!(vram_4k_5s > 30.0, "4K 5s should require >30GB VRAM");
    }
}
