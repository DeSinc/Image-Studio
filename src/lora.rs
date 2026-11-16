use crate::engine::{self, Shared};
use eframe::egui::{self, Color32, RichText, Stroke, StrokeKind, TextureHandle, vec2};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Internal pipeline LoRAs managed automatically by ComfyUI workflows;
/// hidden from the user-facing LoRA carousel to avoid confusion.
const INTERNAL_PIPELINE_LORAS: &[&str] = &[
    "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
    "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoModelFamily {
    MiniMaxH3,
    Ltx25,
    Ltx23,
}

impl VideoModelFamily {
    pub fn from_profile(p: &engine::Profile) -> Self {
        if p.is_minimax_h3() {
            Self::MiniMaxH3
        } else if p.is_ltx25() {
            Self::Ltx25
        } else {
            Self::Ltx23
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::MiniMaxH3 => "MiniMax H3",
            Self::Ltx25 => "LTX-Video 2.5",
            Self::Ltx23 => "LTX-Video 2.3",
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoraItem {
    /// Relative filename inside `models/loras` (used by ComfyUI's LoraLoader).
    pub filename: String,
    /// Human-readable display name for the carousel card.
    pub display_name: String,
    /// Optional short description or trigger-word summary.
    pub description: String,
    /// Which model family this LoRA was built for ("H3", "2.5", "2.3", or "Custom").
    pub compat_tag: String,
    /// Compatible model families.
    pub supports_h3: bool,
    pub supports_ltx25: bool,
    pub supports_ltx23: bool,
    /// Popularity/rating score from ComfyUI LoRA Browser / metadata (higher = top rated).
    pub rating_score: f64,
    /// Local preview image path if found next to the `.safetensors` file.
    pub preview_path: Option<PathBuf>,
    /// Remote preview image URL from ComfyUI LoRA Browser / metadata.
    pub preview_url: Option<String>,
}

impl LoraItem {
    pub fn is_compatible_with(&self, family: VideoModelFamily) -> bool {
        match family {
            VideoModelFamily::MiniMaxH3 => self.supports_h3,
            // LTX 2.5 supports both LTX 2.5 LoRAs and compatible LTX 2.3 LoRAs
            VideoModelFamily::Ltx25 => self.supports_ltx25 || self.supports_ltx23,
            VideoModelFamily::Ltx23 => self.supports_ltx23,
        }
    }
}

pub struct LoraBrowserState {
    pub items: Vec<LoraItem>,
    pub thumbs: HashMap<String, TextureHandle>,
    pub loading_thumbs: HashSet<String>,
    pub remote_fetched: bool,
    pub remote_fetching: bool,
    pub remote_items: Arc<Mutex<Option<Vec<LoraItem>>>>,
    pub thumb_results: Arc<Mutex<Vec<(String, egui::ColorImage)>>>,
}

impl Default for LoraBrowserState {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            thumbs: HashMap::new(),
            loading_thumbs: HashSet::new(),
            remote_fetched: false,
            remote_fetching: false,
            remote_items: Arc::new(Mutex::new(None)),
            thumb_results: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

pub fn is_internal_pipeline_lora(filename: &str) -> bool {
    let base = Path::new(filename)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(filename)
        .to_ascii_lowercase();
    INTERNAL_PIPELINE_LORAS.iter().any(|&p| p == base)
}

fn find_sidecar_preview(lora_path: &Path) -> Option<PathBuf> {
    let parent = lora_path.parent()?;
    let stem = lora_path.file_stem()?.to_str()?;
    let exts = ["preview.png", "preview.jpg", "preview.jpeg", "preview.webp", "png", "jpg", "jpeg", "webp"];
    for ext in exts {
        let cand = parent.join(format!("{stem}.{ext}"));
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

fn find_sidecar_metadata(lora_path: &Path) -> Option<Value> {
    let parent = lora_path.parent()?;
    let stem = lora_path.file_stem()?.to_str()?;
    for ext in ["metadata.json", "civitai.info", "json"] {
        let cand = parent.join(format!("{stem}.{ext}"));
        if cand.is_file() {
            if let Ok(text) = std::fs::read_to_string(&cand) {
                if let Ok(v) = serde_json::from_str::<Value>(&text) {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn pretty_lora_name(filename: &str) -> String {
    let lower = filename.to_ascii_lowercase();
    if lower.contains("minimax_h3") && lower.contains("turbo") && lower.contains("8step") {
        return "Turbo 8-Step".to_string();
    }
    if lower.contains("minimax_h3") && lower.contains("turbo") && lower.contains("4step") {
        return "Turbo 4-Step".to_string();
    }
    let stem = Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename);
    stem.replace(['_', '-'], " ")
}

fn classify_lora(filename: &str, base_model_hint: Option<&str>) -> (String, bool, bool, bool) {
    let combined = format!(
        "{} {}",
        filename.to_ascii_lowercase(),
        base_model_hint.unwrap_or("").to_ascii_lowercase()
    );
    if combined.contains("minimax") || combined.contains("h3") {
        return ("H3".to_string(), true, false, false);
    }
    if combined.contains("2.5") || combined.contains("ltx25") || combined.contains("ltx-2.5") || combined.contains("ltx_2.5") {
        return ("2.5".to_string(), false, true, false);
    }
    if combined.contains("2.3") || combined.contains("ltx23") || combined.contains("ltx-2.3") || combined.contains("ltx_2.3") || combined.contains("ltx") {
        // LTX 2.3 LoRAs are compatible with both LTX 2.3 and LTX 2.5
        return ("2.3".to_string(), false, true, true);
    }
    // Unclassified custom LoRA: show across all video models so user-dropped files are accessible
    ("Custom".to_string(), true, true, true)
}

fn collect_lora_files(root: &Path, dir: &Path, depth: usize, out: &mut Vec<(String, PathBuf)>) {
    if depth > 4 || !dir.is_dir() {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in entries.flatten() {
        let p = ent.path();
        if p.is_dir() {
            collect_lora_files(root, &p, depth + 1, out);
        } else if p.is_file() {
            let is_st = p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("safetensors"))
                .unwrap_or(false);
            if is_st {
                if let Ok(rel) = p.strip_prefix(root) {
                    let rel_s = rel.to_string_lossy().replace('\\', "/");
                    if !is_internal_pipeline_lora(&rel_s) {
                        out.push((rel_s, p));
                    }
                }
            }
        }
    }
}

/// Scans `models/loras` for user LoRAs and reads ComfyUI LoRA Browser sidecar
/// thumbnails (`.preview.png`/`.png`/`.jpg`) and metadata (`.metadata.json`/`.civitai.info`).
pub fn scan_local_loras(sh: &Shared) -> Vec<LoraItem> {
    let mut dirs = Vec::new();
    let active_models = sh.paths.lock().unwrap().models.clone();
    if !active_models.as_os_str().is_empty() {
        dirs.push(active_models.join("loras"));
    }
    let cfg = sh.cfg.lock().unwrap().clone();
    for base in engine::candidate_models_dirs(&cfg) {
        let d = base.join("loras");
        if d.is_dir() && !dirs.contains(&d) {
            dirs.push(d);
        }
    }

    let mut seen = HashSet::new();
    let mut raw_files = Vec::new();
    for d in dirs {
        let mut found = Vec::new();
        collect_lora_files(&d, &d, 0, &mut found);
        for (rel, full) in found {
            if seen.insert(rel.to_ascii_lowercase()) {
                raw_files.push((rel, full));
            }
        }
    }

    let mut items: Vec<LoraItem> = raw_files
        .into_iter()
        .map(|(rel, full)| {
            let meta = find_sidecar_metadata(&full);
            let meta_name = meta.as_ref().and_then(|v| {
                v.get("model_name")
                    .or_else(|| v.get("modelName"))
                    .or_else(|| v.get("name"))
                    .and_then(|s| s.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            });
            let base_model = meta.as_ref().and_then(|v| {
                v.get("base_model")
                    .or_else(|| v.get("baseModel"))
                    .and_then(|s| s.as_str())
            });
            let rating_score = meta
                .as_ref()
                .and_then(|v| {
                    v.pointer("/civitai/stats/thumbsUpCount")
                        .or_else(|| v.pointer("/stats/thumbsUpCount"))
                        .or_else(|| v.pointer("/stats/rating"))
                        .or_else(|| v.get("rating"))
                        .and_then(|n| n.as_f64())
                })
                .unwrap_or_else(|| {
                    if rel.to_ascii_lowercase().contains("turbo") {
                        100.0
                    } else {
                        0.0
                    }
                });
            let preview_url = meta.as_ref().and_then(|v| {
                v.get("preview_url")
                    .or_else(|| v.pointer("/civitai/images/0/url"))
                    .or_else(|| v.pointer("/images/0/url"))
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_string())
            });
            let description = meta
                .as_ref()
                .and_then(|v| {
                    v.get("description")
                        .or_else(|| v.get("notes"))
                        .and_then(|s| s.as_str())
                        .map(|s| s.chars().take(160).collect::<String>())
                })
                .unwrap_or_else(|| {
                    if rel.to_ascii_lowercase().contains("minimax_h3") && rel.to_ascii_lowercase().contains("turbo") {
                        "Official MiniMax H3 8-step speed LoRA".to_string()
                    } else {
                        "Local LoRA from models/loras".to_string()
                    }
                });

            let (compat_tag, supports_h3, supports_ltx25, supports_ltx23) = classify_lora(&rel, base_model);
            let display_name = meta_name.unwrap_or_else(|| pretty_lora_name(&rel));
            let preview_path = find_sidecar_preview(&full);

            LoraItem {
                filename: rel,
                display_name,
                description,
                compat_tag,
                supports_h3,
                supports_ltx25,
                supports_ltx23,
                rating_score,
                preview_path,
                preview_url,
            }
        })
        .collect();

    // Sort top-rated first, then alphabetically
    items.sort_by(|a, b| {
        b.rating_score
            .partial_cmp(&a.rating_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.display_name.to_ascii_lowercase().cmp(&b.display_name.to_ascii_lowercase()))
    });
    items
}

/// Polls ComfyUI's local LoRA Browser / LoRA Manager API (if running) to enrich
/// thumbnails, ratings, and model compatibility metadata.
pub fn fetch_comfy_lora_browser_async(state: &mut LoraBrowserState, ctx: &egui::Context) {
    if state.remote_fetching {
        return;
    }
    state.remote_fetching = true;
    let out = state.remote_items.clone();
    let ctx = ctx.clone();

    std::thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .ok();
        let mut collected = Vec::new();
        if let Some(c) = client {
            // Try common ComfyUI LoRA Browser / LoRA Manager endpoints
            for url in [
                "http://127.0.0.1:8188/api/loras?sort_by=rating&page_size=30",
                "http://127.0.0.1:8188/loras/api/list",
            ] {
                if let Ok(resp) = c.get(url).send() {
                    if resp.status().is_success() {
                        if let Ok(json) = resp.json::<Value>() {
                            let arr = json
                                .get("items")
                                .or_else(|| json.get("loras"))
                                .and_then(|v| v.as_array())
                                .or_else(|| json.as_array());
                            if let Some(list) = arr {
                                for entry in list {
                                    let file = entry
                                        .get("file_name")
                                        .or_else(|| entry.get("filename"))
                                        .or_else(|| entry.get("name"))
                                        .and_then(|s| s.as_str())
                                        .unwrap_or("");
                                    if file.is_empty() || is_internal_pipeline_lora(file) {
                                        continue;
                                    }
                                    let filename = if file.ends_with(".safetensors") {
                                        file.to_string()
                                    } else {
                                        format!("{file}.safetensors")
                                    };
                                    let display_name = entry
                                        .get("model_name")
                                        .or_else(|| entry.get("modelName"))
                                        .or_else(|| entry.get("name"))
                                        .and_then(|s| s.as_str())
                                        .map(|s| s.to_string())
                                        .unwrap_or_else(|| pretty_lora_name(&filename));
                                    let base_model = entry
                                        .get("base_model")
                                        .or_else(|| entry.get("baseModel"))
                                        .and_then(|s| s.as_str());
                                    let preview_url = entry
                                        .get("preview_url")
                                        .or_else(|| entry.get("thumbnail"))
                                        .and_then(|s| s.as_str())
                                        .map(|s| {
                                            if s.starts_with('/') {
                                                format!("http://127.0.0.1:8188{s}")
                                            } else {
                                                s.to_string()
                                            }
                                        });
                                    let rating_score = entry
                                        .get("rating")
                                        .or_else(|| entry.pointer("/civitai/stats/thumbsUpCount"))
                                        .and_then(|n| n.as_f64())
                                        .unwrap_or(0.0);
                                    let (compat_tag, supports_h3, supports_ltx25, supports_ltx23) =
                                        classify_lora(&filename, base_model);
                                    collected.push(LoraItem {
                                        filename,
                                        display_name,
                                        description: "From ComfyUI LoRA Browser".to_string(),
                                        compat_tag,
                                        supports_h3,
                                        supports_ltx25,
                                        supports_ltx23,
                                        rating_score,
                                        preview_path: None,
                                        preview_url,
                                    });
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
        *out.lock().unwrap() = Some(collected);
        ctx.request_repaint();
    });
}

fn load_thumb_async(state: &mut LoraBrowserState, key: &str, path: Option<PathBuf>, url: Option<String>, ctx: &egui::Context) {
    if state.thumbs.contains_key(key) || state.loading_thumbs.contains(key) {
        return;
    }
    if path.is_none() && url.is_none() {
        return;
    }
    state.loading_thumbs.insert(key.to_string());
    let results = state.thumb_results.clone();
    let key_owned = key.to_string();
    let ctx = ctx.clone();

    std::thread::spawn(move || {
        let bytes = if let Some(p) = path {
            std::fs::read(p).ok()
        } else if let Some(u) = url {
            reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .ok()
                .and_then(|c| c.get(&u).send().ok())
                .and_then(|r| if r.status().is_success() { r.bytes().ok() } else { None })
                .map(|b| b.to_vec())
        } else {
            None
        };

        if let Some(b) = bytes {
            if let Ok(img) = image::load_from_memory(&b) {
                let thumb = img.thumbnail(172, 116).to_rgba8();
                let size = [thumb.width() as usize, thumb.height() as usize];
                let ci = egui::ColorImage::from_rgba_unmultiplied(size, thumb.as_raw());
                results.lock().unwrap().push((key_owned, ci));
                ctx.request_repaint();
            }
        }
    });
}

/// Renders a compact, fixed-width horizontal thumbnail carousel for LoRAs
/// compatible with `family`, without ever expanding the sidebar width.
pub fn render_lora_carousel(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    state: &mut LoraBrowserState,
    selected_lora: &mut String,
    lora_strength: &mut f32,
    family: VideoModelFamily,
    default_steps: u32,
    video_steps: &mut u32,
) {
    // Merge any completed remote metadata from ComfyUI LoRA Browser
    if let Some(remote) = state.remote_items.lock().unwrap().take() {
        state.remote_fetching = false;
        state.remote_fetched = true;
        for r in remote {
            if let Some(existing) = state
                .items
                .iter_mut()
                .find(|it| it.filename.eq_ignore_ascii_case(&r.filename))
            {
                if existing.preview_url.is_none() {
                    existing.preview_url = r.preview_url;
                }
                if r.rating_score > existing.rating_score {
                    existing.rating_score = r.rating_score;
                }
                if existing.display_name == pretty_lora_name(&existing.filename) && !r.display_name.is_empty() {
                    existing.display_name = r.display_name;
                }
            }
        }
        state.items.sort_by(|a, b| {
            b.rating_score
                .partial_cmp(&a.rating_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.display_name.to_ascii_lowercase().cmp(&b.display_name.to_ascii_lowercase()))
        });
    }

    // Upload any newly decoded thumbnail textures
    {
        let mut pending = state.thumb_results.lock().unwrap();
        for (key, ci) in pending.drain(..) {
            let tex = ctx.load_texture(format!("lora_thumb_{key}"), ci, egui::TextureOptions::LINEAR);
            state.thumbs.insert(key, tex);
        }
    }

    let compat_items: Vec<LoraItem> = state
        .items
        .iter()
        .filter(|it| it.is_compatible_with(family))
        .cloned()
        .collect();

    // Ensure selected LoRA is not an internal pipeline LoRA
    if *selected_lora != "None" && is_internal_pipeline_lora(selected_lora) {
        *selected_lora = "None".to_string();
    }

    let card_w = 94.0_f32;
    let card_h = 110.0_f32;
    let thumb_w = 84.0_f32;
    let thumb_h = 56.0_f32;

    let accent = Color32::from_rgb(99, 102, 241);
    let card_bg = Color32::from_rgb(26, 30, 42);
    let card_sel = Color32::from_rgb(38, 43, 66);
    let line = Color32::from_rgb(44, 50, 68);
    let dim = Color32::from_rgb(142, 150, 170);
    let text_col = Color32::from_rgb(232, 236, 245);
    let green = Color32::from_rgb(35, 165, 90);

    let prev_lora = selected_lora.clone();

    // Horizontal carousel strictly bounded to available_width so it never stretches the sidebar
    let avail_w = ui.available_width().max(180.0);
    ui.allocate_ui_with_layout(
        vec2(avail_w, card_h + 14.0),
        egui::Layout::top_down(egui::Align::Min),
        |ui| {
            egui::ScrollArea::horizontal()
                .id_salt("video_lora_carousel")
                .max_width(avail_w)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = vec2(6.0, 0.0);

                        // Tile 0: No LoRA (Default model output)
                        let is_none = *selected_lora == "None" || selected_lora.is_empty();
                        let (rect, resp) = ui.allocate_exact_size(vec2(card_w, card_h), egui::Sense::click());
                        let p = ui.painter();
                        p.rect_filled(rect, 6.0, if is_none { card_sel } else { card_bg });
                        p.rect_stroke(
                            rect,
                            6.0,
                            Stroke::new(if is_none { 1.5 } else { 1.0 }, if is_none { accent } else { line }),
                            StrokeKind::Inside,
                        );
                        let thumb_rect = egui::Rect::from_min_size(rect.min + vec2(5.0, 5.0), vec2(thumb_w, thumb_h));
                        p.rect_filled(thumb_rect, 4.0, Color32::from_rgb(19, 22, 31));
                        p.text(
                            thumb_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "DEFAULT",
                            egui::FontId::proportional(11.0),
                            if is_none { text_col } else { dim },
                        );
                        let title_rect = egui::Rect::from_min_size(
                            rect.min + vec2(5.0, 65.0),
                            vec2(thumb_w, 38.0),
                        );
                        ui.scope_builder(egui::UiBuilder::new().max_rect(title_rect), |ui| {
                            ui.add(egui::Label::new(RichText::new("No LoRA").size(11.0).strong().color(text_col)).truncate());
                            ui.add(egui::Label::new(RichText::new(family.label()).size(9.5).color(dim)).truncate());
                        });
                        if resp.clicked() {
                            *selected_lora = "None".to_string();
                        }
                        resp.on_hover_text("Use the selected video model without any additional LoRA.");

                        // LoRA Tiles
                        for item in &compat_items {
                            load_thumb_async(
                                state,
                                &item.filename,
                                item.preview_path.clone(),
                                item.preview_url.clone(),
                                ctx,
                            );
                            let is_sel = *selected_lora == item.filename;
                            let (rect, resp) = ui.allocate_exact_size(vec2(card_w, card_h), egui::Sense::click());
                            let p = ui.painter();
                            p.rect_filled(rect, 6.0, if is_sel { card_sel } else { card_bg });
                            p.rect_stroke(
                                rect,
                                6.0,
                                Stroke::new(if is_sel { 1.5 } else { 1.0 }, if is_sel { accent } else { line }),
                                StrokeKind::Inside,
                            );

                            let thumb_rect = egui::Rect::from_min_size(rect.min + vec2(5.0, 5.0), vec2(thumb_w, thumb_h));
                            if let Some(tex) = state.thumbs.get(&item.filename) {
                                p.image(
                                    tex.id(),
                                    thumb_rect,
                                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                    Color32::WHITE,
                                );
                            } else {
                                p.rect_filled(thumb_rect, 4.0, Color32::from_rgb(20, 24, 35));
                                p.text(
                                    thumb_rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    &item.compat_tag,
                                    egui::FontId::proportional(13.0),
                                    accent,
                                );
                            }

                            // Compatibility badge in top-right of thumbnail
                            let badge_rect = egui::Rect::from_min_size(
                                egui::pos2(thumb_rect.max.x - 30.0, thumb_rect.min.y + 3.0),
                                vec2(27.0, 14.0),
                            );
                            p.rect_filled(badge_rect, 3.0, Color32::from_rgba_unmultiplied(12, 14, 20, 210));
                            p.text(
                                badge_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                &item.compat_tag,
                                egui::FontId::proportional(9.0),
                                if item.compat_tag == "2.3" && family == VideoModelFamily::Ltx25 {
                                    green
                                } else {
                                    text_col
                                },
                            );

                            let title_rect = egui::Rect::from_min_size(
                                rect.min + vec2(5.0, 65.0),
                                vec2(thumb_w, 38.0),
                            );
                            ui.scope_builder(egui::UiBuilder::new().max_rect(title_rect), |ui| {
                                ui.add(
                                    egui::Label::new(RichText::new(&item.display_name).size(10.5).strong().color(text_col))
                                        .truncate(),
                                );
                                let sub = if item.rating_score > 0.0 {
                                    format!("Top ({:.0})", item.rating_score)
                                } else if item.compat_tag == "2.3" && family == VideoModelFamily::Ltx25 {
                                    "2.3 Compat".to_string()
                                } else {
                                    "Installed".to_string()
                                };
                                ui.add(egui::Label::new(RichText::new(sub).size(9.5).color(dim)).truncate());
                            });

                            if resp.clicked() {
                                if is_sel {
                                    *selected_lora = "None".to_string();
                                } else {
                                    *selected_lora = item.filename.clone();
                                }
                            }
                            let compat_note = if item.compat_tag == "2.3" && family == VideoModelFamily::Ltx25 {
                                "LTX 2.3 LoRA (compatible with LTX 2.5)"
                            } else {
                                &item.description
                            };
                            resp.on_hover_text(format!(
                                "{}\n{}\nFile: {}",
                                item.display_name, compat_note, item.filename
                            ));
                        }
                    });
                });
        },
    );

    if *selected_lora != prev_lora {
        let lower = selected_lora.to_ascii_lowercase();
        if lower.contains("turbo") && lower.contains("4step") {
            *video_steps = 4;
        } else if lower.contains("turbo") && lower.contains("8step") {
            *video_steps = 8;
        } else if *selected_lora == "None" && family == VideoModelFamily::MiniMaxH3 {
            *video_steps = default_steps.max(4);
        }
    }

    if *selected_lora != "None" && !selected_lora.is_empty() {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Strength:").size(11.0).color(text_col));
            ui.add(egui::Slider::new(lora_strength, 0.0..=2.0).step_by(0.05));
            if ui.small_button("Clear").clicked() {
                *selected_lora = "None".to_string();
            }
        });
    } else if compat_items.is_empty() {
        let hint = match family {
            VideoModelFamily::MiniMaxH3 => "No MiniMax H3 LoRAs found. Drop .safetensors + preview images in models/loras.",
            VideoModelFamily::Ltx25 => "No LTX 2.5 / 2.3 LoRAs found. Drop .safetensors + preview images in models/loras.",
            VideoModelFamily::Ltx23 => "No LTX 2.3 LoRAs found. Drop .safetensors + preview images in models/loras.",
        };
        ui.label(RichText::new(hint).size(10.0).color(dim));
    } else if family == VideoModelFamily::Ltx25 {
        ui.label(RichText::new("Includes compatible LTX 2.3 LoRAs. Click a thumbnail to apply.").size(10.0).color(dim));
    } else {
        ui.label(RichText::new("Click a thumbnail to apply, or click again to clear.").size(10.0).color(dim));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_pipeline_loras_are_filtered() {
        assert!(is_internal_pipeline_lora(
            "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
        ));
        assert!(is_internal_pipeline_lora(
            "sub/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
        ));
        assert!(!is_internal_pipeline_lora(
            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
        ));
    }

    #[test]
    fn ltx25_includes_compatible_ltx23_loras() {
        let (_, h3, ltx25, ltx23) = classify_lora("cinematic_ltx_2.3_style.safetensors", None);
        let item = LoraItem {
            filename: "cinematic_ltx_2.3_style.safetensors".into(),
            display_name: "Cinematic".into(),
            description: String::new(),
            compat_tag: "2.3".into(),
            supports_h3: h3,
            supports_ltx25: ltx25,
            supports_ltx23: ltx23,
            rating_score: 10.0,
            preview_path: None,
            preview_url: None,
        };
        assert!(!item.is_compatible_with(VideoModelFamily::MiniMaxH3));
        assert!(item.is_compatible_with(VideoModelFamily::Ltx25));
        assert!(item.is_compatible_with(VideoModelFamily::Ltx23));
    }
}
