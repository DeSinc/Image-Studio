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
    Done { png: Vec<u8>, img: ColorImage },
    GenErr(String),
}

struct Item {
    name: String,
    png: Vec<u8>,
    tex: TextureHandle,
    width: u32,
    height: u32,
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
    scale: f32,
    resolution: u32,
    generating: bool,
    cancel: Arc<AtomicBool>,
    status: String,
    error: Option<String>,
    compare_mode: bool,
    split: f32,
    canvas_zoom: f32,
    canvas_pan: Vec2,
    show_setup: bool,
    loading: usize,
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
            scale: 1.0,
            resolution: 1024,
            generating: false,
            cancel: Arc::new(AtomicBool::new(false)),
            status: String::new(),
            error: None,
            compare_mode: false,
            split: 0.5,
            canvas_zoom: 1.0,
            canvas_pan: vec2(0.0, 0.0),
            show_setup: false,
            loading: 0,
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
        // Save currently typed prompt into the active node before switching away
        if let Some(curr_id) = self.active_node_id {
            if curr_id != id {
                if let Some(curr_node) = self.nodes.iter_mut().find(|n| n.id == curr_id) {
                    curr_node.prompt = self.prompt.clone();
                }
            }
        }

        self.active_node_id = Some(id);
        self.compare_mode = false;
        if let Some(node) = self.find_node(id) {
            self.prompt = node.prompt.clone();
        }
        self.focus_prompt = true;
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
                    if self.scale > 0.0 {
                        self.custom_w = (((n_w as f32 * self.scale) / 16.0).round() as u32 * 16).clamp(64, 4096);
                        self.custom_h = (((n_h as f32 * self.scale) / 16.0).round() as u32 * 16).clamp(64, 4096);
                    }
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
                SizePreset::Custom => {}
            }
        } else {
            match preset {
                SizePreset::P720 => {
                    self.custom_w = 720;
                    self.custom_h = 720;
                    self.scale = 1.0;
                }
                SizePreset::P1024 => {
                    self.custom_w = 1024;
                    self.custom_h = 1024;
                    self.scale = 1.0;
                }
                _ => {}
            }
        }
    }

    fn update_scale_dimensions(&mut self) {
        self.size_preset = SizePreset::Custom;
        if let Some((n_w, n_h)) = self.active_node().map(|n| (n.width, n.height)) {
            self.custom_w = (((n_w as f32 * self.scale) / 16.0).round() as u32 * 16).clamp(64, 4096);
            self.custom_h = (((n_h as f32 * self.scale) / 16.0).round() as u32 * 16).clamp(64, 4096);
        }
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
        let base_png = match self.active_node() {
            Some(n) => n.png.clone(),
            None => {
                self.error = Some("Base image is required".into());
                return;
            }
        };

        self.generating = true;
        self.status = "Uploading images...".into();
        self.error = None;
        self.generating_parent_id = self.active_node_id;
        self.generating_prompt = self.prompt.trim().to_string();

        // Save the prompt on the parent node that created this generation
        if let Some(pid) = self.active_node_id {
            if let Some(parent) = self.nodes.iter_mut().find(|n| n.id == pid) {
                parent.prompt = self.generating_prompt.clone();
            }
        }
        self.prompt.clear();

        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = cancel.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();

        let mut pngs = vec![base_png];
        for r in &self.refs {
            pngs.push(r.png.clone());
        }

        let seed = if self.random_seed { rand_u64() } else { self.seed };
        if self.random_seed {
            self.seed = seed;
        }

        let custom = match self.size_preset {
            SizePreset::Original => None,
            _ => Some((self.custom_w, self.custom_h)),
        };

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
                Ok(png) => {
                    if let Ok(dyn_img) = image::load_from_memory(&png) {
                        let rgba = dyn_img.to_rgba8();
                        let ci = ColorImage::from_rgba_unmultiplied(
                            [dyn_img.width() as usize, dyn_img.height() as usize],
                            rgba.as_raw(),
                        );
                        let _ = tx.send(Ev::Done { png, img: ci });
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
                            };
                            self.nodes.push(node);
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
                Ev::Done { png, img } => {
                    self.generating = false;
                    self.progress = None;
                    self.preview_tex = None;
                    let [out_w, out_h] = img.size;
                    let id = self.next_node_id;
                    self.next_node_id += 1;
                    let tex = ctx.load_texture(format!("edit_{id}"), img, egui::TextureOptions::LINEAR);

                    let parent_id = self.generating_parent_id;
                    let parent_step = parent_id.and_then(|pid| self.find_node(pid)).map(|n| n.step_num).unwrap_or(0);
                    let prompt_label = if self.generating_prompt.is_empty() {
                        format!("Edit {}", id)
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
                        step_num: parent_step + 1,
                    };
                    self.nodes.push(node);
                    self.active_node_id = Some(id);
                    self.compare_mode = false;
                    self.prompt.clear();

                    // Maintain newly generated resolution while resetting scale slider to 100% (1.0x) in the middle
                    self.custom_w = out_w as u32;
                    self.custom_h = out_h as u32;
                    self.scale = 1.0;

                    self.update_dimensions_from_active_node();
                    self.split = 0.5;
                }
                Ev::GenErr(e) => {
                    self.generating = false;
                    self.progress = None;
                    self.preview_tex = None;
                    if e != "Cancelled" {
                        self.error = Some(e);
                    }
                }
            }
        }
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        let mut need_base = self.active_node().is_none();
        for p in dropped {
            let t = if need_base { Target::Base } else { Target::Ref };
            need_base = false;
            self.load(ctx, t, Src::Path(p));
        }
        let paste = ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::V));
        if paste && !ctx.egui_wants_keyboard_input() {
            if let Ok(img) = arboard::Clipboard::new().and_then(|mut c| c.get_image()) {
                let t = if self.active_node().is_none() { Target::Base } else { Target::Ref };
                self.load(ctx, t, Src::Rgba(img.width, img.height, img.bytes.into_owned()));
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

    fn can_generate(&self) -> Result<(), &'static str> {
        if self.generating {
            return Err("Generating...");
        }
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
        if self.active_node().is_none() {
            return Err("Drop or select a base image first");
        }
        if self.prompt.trim().is_empty() {
            return Err("Type an edit instruction");
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

                    ui.add_space(8.0);
                    let hist_txt = match self.history_dock {
                        HistoryDock::Left => format!("History ({}) | Left", self.nodes.len()),
                        HistoryDock::Right => format!("History ({}) | Right", self.nodes.len()),
                        HistoryDock::Floating => format!("History ({}) | Float", self.nodes.len()),
                    };
                    let hist_btn = egui::Button::new(
                        RichText::new(hist_txt)
                            .size(11.5)
                            .color(if self.show_history_window { ACCENT_GLOW } else { TEXT })
                    )
                    .fill(CARD)
                    .stroke(Stroke::new(1.0, LINE))
                    .corner_radius(4);
                    if ui.add(hist_btn).clicked() {
                        if self.history_dock == HistoryDock::Floating {
                            self.show_history_window = !self.show_history_window;
                        } else {
                            self.history_dock = match self.history_dock {
                                HistoryDock::Left => HistoryDock::Right,
                                HistoryDock::Right => HistoryDock::Floating,
                                HistoryDock::Floating => HistoryDock::Left,
                            };
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Setup").clicked() {
                            self.show_setup = !self.show_setup;
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
                    if w_resp.changed() || h_resp.changed() {
                        self.custom_w = ((self.custom_w as f32 / 16.0).round() as u32 * 16).clamp(64, 4096);
                        self.custom_h = ((self.custom_h as f32 / 16.0).round() as u32 * 16).clamp(64, 4096);
                        self.size_preset = SizePreset::Custom;
                        if let Some((n_w, _)) = self.active_node().map(|n| (n.width, n.height)) {
                            if n_w > 0 {
                                self.scale = (self.custom_w as f32 / n_w as f32).clamp(0.1, 2.0);
                            }
                        }
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
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let is_float = self.history_dock == HistoryDock::Floating;
                let is_right = self.history_dock == HistoryDock::Right;
                let is_left = self.history_dock == HistoryDock::Left;

                let float_btn = ui.selectable_label(is_float, "Float").on_hover_text("Float as separate window");
                if float_btn.clicked() {
                    self.history_dock = HistoryDock::Floating;
                    self.show_history_window = true;
                }

                let right_btn = ui.selectable_label(is_right, "Right").on_hover_text("Dock to right panel");
                if right_btn.clicked() {
                    self.history_dock = HistoryDock::Right;
                }

                let left_btn = ui.selectable_label(is_left, "Left").on_hover_text("Dock to left panel");
                if left_btn.clicked() {
                    self.history_dock = HistoryDock::Left;
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
                    if ui.small_button("Clear").on_hover_text("Reset history to start").clicked() {
                        if let Some(root) = self.nodes.first().cloned() {
                            self.nodes = vec![root];
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

        egui::ScrollArea::vertical().id_salt("history_scroll").show(ui, |ui| {
            if active_thread.is_empty() {
                ui.label(RichText::new("No edits yet. Drop an image to start.").color(DIM).size(11.0));
            } else {
                for &node_id in &active_thread {
                    if let Some(node) = self.find_node(node_id) {
                        let is_active = self.active_node_id == Some(node.id);
                        let children = self.child_nodes(node.id);
                        let has_fork = children.len() > 1;

                        let card_w = ui.available_width();
                        let (card_rect, card_resp) = ui.allocate_exact_size(vec2(card_w, 38.0), Sense::click());
                        if card_resp.clicked() {
                            to_select = Some(node.id);
                        }
                        if card_resp.hovered() {
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
                        let text_max_w = (card_rect.max.x - text_left - 6.0).max(10.0);
                        let title = if is_active {
                            format!("> {}", node.label)
                        } else {
                            node.label.clone()
                        };
                        let title_galley = p.layout(
                            title,
                            egui::FontId::proportional(11.0),
                            if is_active { Color32::WHITE } else { TEXT },
                            text_max_w,
                        );
                        p.galley(pos2(text_left, card_rect.min.y + 4.0), title_galley, Color32::WHITE);

                        let sub = format!("Step {} | {}x{}", node.step_num, node.width, node.height);
                        let sub_galley = p.layout(
                            sub,
                            egui::FontId::proportional(10.0),
                            DIM,
                            text_max_w,
                        );
                        p.galley(pos2(text_left, card_rect.min.y + 20.0), sub_galley, DIM);

                        if has_fork {
                            let fork_txt = format!("{} branches", children.len());
                            let fork_g = p.layout_no_wrap(fork_txt, egui::FontId::proportional(9.5), AMBER);
                            let fork_pos = pos2(card_rect.max.x - fork_g.size().x - 6.0, card_rect.min.y + 20.0);
                            p.galley(fork_pos, fork_g, AMBER);
                        }

                        if !node.prompt.is_empty() {
                            card_resp.on_hover_text(&node.prompt);
                        }

                        ui.add_space(2.0);
                    }
                }
            }
        });

        if let Some(id) = to_select {
            self.select_node(id);
        }
    }

    fn floating_history_window(&mut self, ctx: &egui::Context) {
        if self.history_dock != HistoryDock::Floating || !self.show_history_window {
            return;
        }
        let mut open = self.show_history_window;
        egui::Window::new("History")
            .open(&mut open)
            .default_width(280.0)
            .default_height(380.0)
            .resizable(true)
            .collapsible(true)
            .show(ctx, |ui| {
                self.render_history_ui(ui);
            });
        self.show_history_window = open;
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

        if let Some(active) = self.active_node() {
            let active_id = active.id;
            let active_w = active.width;
            let active_h = active.height;
            let active_step = active.step_num;
            let active_label = active.label.clone();
            let active_png = active.png.clone();
            let active_tex = active.tex.clone();
            let parent_opt = self.parent_node(active_id).map(|p| (p.step_num, p.tex.clone()));

            let mut action: Option<&str> = None;
            let mut toggle_compare = false;
            let mut do_undo = false;
            let mut do_redo = false;
            let mut do_clear = false;

            ui.horizontal(|ui| {
                if self.compare_mode && parent_opt.is_some() {
                    let (p_step, _) = parent_opt.as_ref().unwrap();
                    heading(ui, &format!("Comparison (drag divider)   Step {p_step} vs Step {active_step}"));
                } else {
                    heading(ui, &format!("Step {}   \"{}\"   ({} x {} px)", active_step, active_label, active_w, active_h));
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Clear Canvas").on_hover_text("Start fresh").clicked() {
                        do_clear = true;
                    }
                    if ui.button("Copy").clicked() {
                        action = Some("copy");
                    }
                    if ui.button("Save as...").clicked() {
                        action = Some("save");
                    }

                    if parent_opt.is_some() {
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
            let fit_size = fit(active_tex.size_vec2(), canvas_area);

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

            if self.compare_mode && parent_opt.is_some() {
                let (p_step, p_tex) = parent_opt.unwrap();
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
            } else {
                let (tex_to_draw, is_preview) = if self.generating && self.preview_tex.is_some() {
                    (self.preview_tex.as_ref().unwrap(), true)
                } else {
                    (&active_tex, false)
                };

                p.image(tex_to_draw.id(), image_rect, FULL_UV, Color32::WHITE);
                p.rect_stroke(image_rect, 0, Stroke::new(1.0, Color32::from_white_alpha(20)), StrokeKind::Outside);

                if is_preview {
                    let badge_text = if let Some((v, max)) = self.progress {
                        format!("LIVE PREVIEW (Step {}/{})", v, max)
                    } else {
                        "LIVE PREVIEW".to_string()
                    };
                    let g = p.layout_no_wrap(badge_text, egui::FontId::proportional(11.0), Color32::WHITE);
                    let badge_pos = image_rect.left_top() + vec2(10.0, 10.0);
                    let r = egui::Align2::LEFT_TOP.anchor_size(badge_pos, g.size()).expand2(vec2(8.0, 4.0));
                    p.rect_filled(r, 4, Color32::from_black_alpha(190));
                    p.rect_stroke(r, 4, Stroke::new(1.0, ACCENT), StrokeKind::Inside);
                    p.galley(r.shrink2(vec2(8.0, 4.0)).min, g, Color32::WHITE);
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
                    c - vec2(0.0, 14.0),
                    egui::Align2::CENTER_CENTER,
                    if self.loading > 0 { "Loading image..." } else { "Drop an image here to start editing" },
                    egui::FontId::proportional(18.0),
                    TEXT,
                );
                p.text(
                    c + vec2(0.0, 14.0),
                    egui::Align2::CENTER_CENTER,
                    "Click to browse | Ctrl+V to paste from clipboard",
                    egui::FontId::proportional(13.0),
                    DIM,
                );

                if hover {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                if resp.clicked() {
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
                    let prompt_id = egui::Id::new("main_prompt_text_edit");
                    let text_edit = egui::TextEdit::multiline(&mut self.prompt)
                        .id(prompt_id)
                        .desired_rows(2)
                        .desired_width((ui.available_width() - 250.0).max(180.0))
                        .hint_text("Describe your edit instructions... Enter to generate, Shift+Enter for newline");
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

                    if text_resp.changed() {
                        if let Some(curr_id) = self.active_node_id {
                            if let Some(curr_node) = self.nodes.iter_mut().find(|n| n.id == curr_id) {
                                curr_node.prompt = self.prompt.clone();
                            }
                        }
                    }

                    // Submit on Enter (without Shift)
                    let enter_submit = text_resp.has_focus()
                        && ctx.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                    if enter_submit {
                        while self.prompt.ends_with('\n') || self.prompt.ends_with('\r') {
                            self.prompt.pop();
                        }
                        if self.can_generate().is_ok() {
                            self.start_generation(ctx);
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.generating {
                            let cancel_btn = egui::Button::new(RichText::new("Cancel").size(13.0).strong().color(RED))
                                .fill(CARD)
                                .stroke(Stroke::new(1.0, RED.linear_multiply(0.4)))
                                .corner_radius(8)
                                .min_size(vec2(85.0, 36.0));
                            if ui.add(cancel_btn).clicked() {
                                self.cancel.store(true, Ordering::SeqCst);
                                self.status = "Cancelling...".into();
                            }
                            ui.add_space(4.0);
                            let status_txt = if let Some((v, max)) = self.progress {
                                format!("{}/{} ({})", v, max, self.status)
                            } else {
                                self.status.clone()
                            };
                            ui.label(RichText::new(status_txt).size(11.5).color(DIM));
                            ui.spinner();
                        } else {
                            let can = self.can_generate();
                            let is_ok = can.is_ok();
                            let send_btn = egui::Button::new(
                                RichText::new("Generate")
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
                                self.start_generation(ctx);
                            }
                            if let Err(why) = can {
                                resp.on_hover_text(why);
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
