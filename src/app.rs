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
use std::time::Duration;

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
const MAX_REFS: usize = 15;
const FULL_UV: Rect = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));

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
            if let Some(node) = self.find_node(id) {
                if !node.prompt.is_empty() {
                    self.prompt = node.prompt.clone();
                    self.focus_prompt = true;
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
                let (name, dyn_img) = match src {
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
                let w = dyn_img.width();
                let h = dyn_img.height();
                let rgba = dyn_img.to_rgba8();
                let mut png = Vec::new();
                let enc = image::codecs::png::PngEncoder::new(&mut png);
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

        fn start_generation(&mut self, ctx: &egui::Context) {
        let p = self.prompt.trim().to_string();
        self.prompt.clear();
        self.start_generation_with_prompt(ctx, p);
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
                    };
                    self.nodes.push(node);
                    self.active_node_id = Some(id);
                    self.compare_mode = false;

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
                        self.start_generation_with_prompt(ctx, next.prompt);
                    }
                }
                Ev::GenErr(e) => {
                    self.generating = false;
                    self.generating_dims = None;
                    self.progress = None;
                    self.preview_tex = None;
                    self.queue.clear();
                    if e != "Cancelled" {
                        self.error = Some(e);
                    }
                }
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
            EngState::Ready { .. } => {}
            EngState::DownloadingEngine { .. } => return Err("Downloading engine..."),
            EngState::ExtractingEngine { .. } => return Err("Extracting engine..."),
            EngState::DownloadingModels { .. } => return Err("Downloading models..."),
            EngState::RepairingEngine { .. } => return Err("Repairing engine..."),
            EngState::Starting { .. } => return Err("Starting engine..."),
            EngState::Detecting => return Err("Detecting engine..."),
            EngState::Failed(_) => return Err("Engine setup failed (see Setup)"),
        }
        if self.sh.models_found.lock().unwrap().iter().any(|m| !m.1) {
            return Err("Downloading model weights...");
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
                            EngState::Failed(_) => ("Engine setup failed".to_string(), RED),
                        };
                        egui::Frame::new()
                            .fill(col.linear_multiply(0.14))
                            .corner_radius(20)
                            .inner_margin(egui::Margin::symmetric(10, 3))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    let (r, _) = ui.allocate_exact_size(vec2(8.0, 8.0), Sense::hover());
                                    ui.painter().circle_filled(r.center(), 4.0, col);
                                    ui.label(RichText::new(txt).color(col).strong().size(12.0));
                                });
                            });
                        ui.label(RichText::new(format!("build {}", env!("BUILD_STAMP"))).size(11.0).color(DIM));
                    });
                });
            });
    }

    // Left Panel: Compact Reference Strip (+ Docked History if Left)
    fn refs_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
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

    // Right Panel: Compact, power-user styled settings
    fn settings_panel(&mut self, ui: &mut egui::Ui) {
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
                    ui.label(RichText::new("Aspect scale:").size(11.5).color(DIM));
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

                        let sub = format!("Step {} | {}x{}", node.step_num, node.width, node.height);
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

    fn render_setup_card(&mut self, ui: &mut egui::Ui, rect: Rect, state: &EngState) {
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 10, Color32::from_rgb(15, 17, 24));
        p.rect_stroke(rect, 10, Stroke::new(1.5, LINE), StrokeKind::Inside);

        let is_failed = matches!(state, EngState::Failed(_));
        let card_w = (if is_failed { 620.0f32 } else { 540.0f32 }).min(rect.width() - 32.0);
        let card_h = if is_failed { 380.0f32 } else { 240.0f32 };
        let card_rect = Rect::from_center_size(rect.center(), vec2(card_w, card_h));

        p.rect_filled(card_rect, 12, CARD);
        p.rect_stroke(card_rect, 12, Stroke::new(1.0, if is_failed { RED.linear_multiply(0.4) } else { CARD_HI }), StrokeKind::Inside);

        let mut card_ui = ui.new_child(egui::UiBuilder::new().max_rect(card_rect.shrink(22.0)));
        card_ui.vertical_centered(|ui| {
            match state {
                EngState::DownloadingEngine { step, progress, bytes_done, bytes_total, speed_mb } => {
                    ui.label(RichText::new("SETTING UP IMAGE STUDIO").size(11.0).color(ACCENT).strong());
                    ui.add_space(4.0);
                    ui.label(RichText::new("Downloading AI Image Engine").size(18.0).strong().color(TEXT));
                    ui.label(RichText::new(step).size(12.0).color(DIM));
                    ui.label(RichText::new(format!("Step 1 of 2: Portable ComfyUI environment • {:.1} MB/s", speed_mb)).size(12.0).color(DIM));
                    ui.add_space(14.0);

                    let bar = egui::ProgressBar::new(*progress)
                        .show_percentage()
                        .fill(ACCENT)
                        .animate(true);
                    ui.add_sized([card_w - 44.0, 14.0], bar);
                    ui.add_space(8.0);

                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{} / {}", engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total))).size(11.5).color(DIM));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(format!("{:.1} MB/s", speed_mb)).size(11.5).color(TEXT).strong());
                        });
                    });
                }
                EngState::ExtractingEngine { step } => {
                    ui.label(RichText::new("SETTING UP IMAGE STUDIO").size(11.0).color(ACCENT).strong());
                    ui.add_space(4.0);
                    ui.label(RichText::new("Extracting Portable Engine").size(18.0).strong().color(TEXT));
                    ui.label(RichText::new("Unpacking runtime components (this takes 1-2 minutes)...").size(12.0).color(DIM));
                    ui.add_space(14.0);

                    let spinner_bar = egui::ProgressBar::new(0.5).fill(ACCENT).animate(true);
                    ui.add_sized([card_w - 44.0, 14.0], spinner_bar);
                    ui.add_space(8.0);
                    ui.label(RichText::new(step).size(11.5).color(DIM));
                }
                EngState::DownloadingModels { current_file, file_index, total_files, file_progress, bytes_done, bytes_total, speed_mb, overall_progress } => {
                    ui.label(RichText::new("SETTING UP IMAGE STUDIO").size(11.0).color(ACCENT).strong());
                    ui.add_space(4.0);
                    ui.label(RichText::new(format!("Downloading AI Model Weights ({}/{})", file_index, total_files)).size(18.0).strong().color(TEXT));
                    ui.label(RichText::new(format!("Qwen Image 2.1 Multi-Ref ({})", current_file)).size(11.5).color(DIM));
                    ui.add_space(10.0);

                    let bar = egui::ProgressBar::new(*file_progress)
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
                    ui.add_space(6.0);

                    let ov_bar = egui::ProgressBar::new(*overall_progress)
                        .fill(ACCENT.linear_multiply(0.6))
                        .text(format!("Total Download Progress: {:.0}%", overall_progress * 100.0));
                    ui.add_sized([card_w - 44.0, 10.0], ov_bar);

                    ui.add_space(8.0);
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

    // Center Hero Canvas (Centered Image + Interactive Zoom & Pan)
    fn center_canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let max_size = ui.available_size();

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
                    heading(ui, &format!("Generating from scratch   ({} x {} px)", self.custom_w, self.custom_h));
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

                let badge_text = if let Some((v, max)) = self.progress {
                    format!("GENERATING (Step {}/{})", v, max)
                } else {
                    "GENERATING...".to_string()
                };
                let g = p.layout_no_wrap(badge_text, egui::FontId::proportional(11.0), Color32::WHITE);
                let badge_pos = image_rect.left_top() + vec2(10.0, 10.0);
                let r = egui::Align2::LEFT_TOP.anchor_size(badge_pos, g.size()).expand2(vec2(8.0, 4.0));
                p.rect_filled(r, 4, Color32::from_black_alpha(190));
                p.rect_stroke(r, 4, Stroke::new(1.0, ACCENT), StrokeKind::Inside);
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
            if !matches!(st, EngState::Ready { .. }) {
                self.render_setup_card(ui, rect, &st);
            } else {
                let hover = resp.hovered();
                let p = ui.painter_at(rect);
                p.rect_filled(rect, 10, Color32::from_rgb(15, 17, 24));
                p.rect_stroke(rect, 10, Stroke::new(1.5, if hover { ACCENT } else { LINE }), StrokeKind::Inside);

                let c = rect.center();
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

                let btn_rect = Rect::from_center_size(c + vec2(0.0, 26.0), vec2(270.0, 34.0));
                let paste_hover = btn_rect.contains(ui.ctx().input(|i| i.pointer.hover_pos().unwrap_or_default()));
                p.rect_filled(btn_rect, 6, if paste_hover { ACCENT } else { CARD_HI });
                p.rect_stroke(btn_rect, 6, Stroke::new(1.0, if paste_hover { ACCENT_GLOW } else { LINE }), StrokeKind::Inside);
                p.text(btn_rect.center(), egui::Align2::CENTER_CENTER, "📋 Paste Image from Clipboard (Ctrl+V)", egui::FontId::proportional(12.5), Color32::WHITE);

                if paste_hover && ui.ctx().input(|i| i.pointer.any_pressed()) {
                    let _ = self.paste_from_clipboard(ctx);
                } else if resp.clicked() {
                    self.pick_images(ctx, Target::Base, false);
                }
            }
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
                    let hint = if self.active_node().is_some() {
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

                    // Submit on Enter (without Shift)
                    let enter_submit = text_resp.has_focus()
                        && ctx.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                    if enter_submit {
                        while self.prompt.ends_with('\n') || self.prompt.ends_with('\r') {
                            self.prompt.pop();
                        }
                        let p_text = self.prompt.trim().to_string();
                        if !p_text.is_empty() {
                            if self.generating {
                                if self.can_queue().is_ok() {
                                    self.queue.push(QueuedItem { prompt: p_text });
                                    self.prompt.clear();
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
                                if self.generating {
                                    if self.can_queue().is_ok() {
                                        self.queue.push(QueuedItem { prompt: p_text });
                                        self.prompt.clear();
                                    }
                                } else if self.can_generate().is_ok() {
                                    self.start_generation(ctx);
                                }
                            }
                        }
                        if let Some(why) = tip {
                            resp.on_hover_text(why);
                        }

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
                                self.status = "Cancelling...".into();
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

                            let status_txt = if let Some((v, max)) = self.progress {
                                format!("{}/{} ({})", v, max, self.status)
                            } else {
                                self.status.clone()
                            };
                            ui.label(RichText::new(status_txt).size(11.5).color(DIM));
                            ui.spinner();
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
            .default_width(620.0)
            .show(ctx, |ui| {
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
                    EngState::Failed(m) => format!("Failed: {m}"),
                };
                ui.label(RichText::new(st).strong());
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
                heading(ui, "REQUIRED MODELS");
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

                ui.separator();
                heading(ui, "SOFTWARE UPDATES");
                let cur_ver = env!("CARGO_PKG_VERSION");
                let (up_avail, latest, is_dl, dl_prog, applied, err) = {
                    let st = self.updater.lock().unwrap();
                    (st.update_available, st.latest_version.clone(), st.downloading, st.download_progress, st.update_applied, st.error.clone())
                };

                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Installed version: v{cur_ver}")).size(12.0).color(TEXT));
                    if up_avail {
                        ui.label(RichText::new(format!("* Update available: v{latest}")).size(12.0).color(GREEN).strong());
                    } else if !latest.is_empty() {
                        ui.label(RichText::new("(Up to date)").size(12.0).color(DIM));
                    }
                });

                if let Some(ref e) = err {
                    ui.label(RichText::new(format!("Update check: {e}")).size(11.5).color(RED));
                }

                if applied {
                    ui.label(RichText::new("Update installed successfully! Restart Image Studio to use the new version.").color(GREEN).strong());
                    if ui.button("Restart Image Studio Now").clicked() {
                        if let Ok(exe) = std::env::current_exe() {
                            let _ = std::process::Command::new(exe).spawn();
                            std::process::exit(0);
                        }
                    }
                } else if is_dl {
                    ui.label(RichText::new(format!("Downloading update: {:.0}%", dl_prog * 100.0)).size(12.0).color(ACCENT));
                    let bar = egui::ProgressBar::new(dl_prog).show_percentage().fill(ACCENT);
                    ui.add_sized([ui.available_width() - 20.0, 14.0], bar);
                } else if up_avail {
                    let btn = egui::Button::new("Download & Install Update").fill(ACCENT); if ui.add(btn).clicked() {
                        crate::updater::download_and_install_update(self.updater.clone(), ctx.clone());
                    }
                } else {
                    if ui.button("Check for Updates").clicked() {
                        crate::updater::check_for_updates(self.updater.clone(), ctx.clone());
                    }
                }

                ui.add_space(4.0);
                ui.label(RichText::new(format!("Logs folder: {}", engine::data_dir().display())).size(12.0).color(DIM));
            });
        self.show_setup = open;
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
