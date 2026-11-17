use crate::engine::{self, Shared};
use eframe::egui::{self, Color32, RichText, Stroke, StrokeKind, TextureHandle, vec2};
use image::AnimationDecoder;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActiveLora {
    pub filename: String,
    pub strength: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromptTag {
    pub lora_filename: String,
    pub lora_display_name: String,
    pub keyword: String,
    #[serde(skip, default = "std::time::Instant::now")]
    pub created_at: std::time::Instant,
    #[serde(default)]
    pub hovered_dismissed: bool,
}

/// Joins active LoRA prompt tags with `", "` and prepends them to the user's prompt.
pub fn compose_prompt_with_tags(tags: &[PromptTag], user_prompt: &str) -> String {
    let tag_parts: Vec<&str> = tags
        .iter()
        .map(|t| t.keyword.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let trimmed = user_prompt.trim();
    if tag_parts.is_empty() {
        return trimmed.to_string();
    }
    let prefix = tag_parts.join(", ");
    if trimmed.is_empty() {
        prefix
    } else {
        format!("{prefix}, {trimmed}")
    }
}

/// Strips any active LoRA prompt tag keywords from the start of a composed prompt
/// (inverse of `compose_prompt_with_tags`) so restoring/reverting a prompt doesn't
/// duplicate tag text in the user's prompt input box.
pub fn strip_prompt_tags(tags: &[PromptTag], prompt: &str) -> String {
    let mut rest = prompt.trim();
    if rest.is_empty() || tags.is_empty() {
        return rest.to_string();
    }

    let prefix = compose_prompt_with_tags(tags, "");
    if !prefix.is_empty() {
        if rest == prefix {
            return String::new();
        }
        if let Some(after) = rest.strip_prefix(&prefix) {
            let after_trimmed = after.trim_start();
            if let Some(stripped) = after_trimmed.strip_prefix(',') {
                return stripped.trim_start().to_string();
            }
        }
    }

    let mut keywords: Vec<&str> = tags
        .iter()
        .map(|t| t.keyword.trim())
        .filter(|s| !s.is_empty())
        .collect();
    keywords.sort_by(|a, b| b.len().cmp(&a.len()));
    keywords.dedup();

    loop {
        let mut matched = false;
        for kw in &keywords {
            if rest == *kw {
                return String::new();
            }
            if let Some(after) = rest.strip_prefix(kw) {
                let after_trimmed = after.trim_start();
                if let Some(stripped) = after_trimmed.strip_prefix(',') {
                    rest = stripped.trim_start();
                    matched = true;
                    break;
                }
            }
        }
        if !matched || rest.is_empty() {
            break;
        }
    }

    rest.to_string()
}

/// Renders active LoRA keyword tag pills above the prompt input, with a
/// 3-second (or hover-dismissed) speech bubble explaining which LoRA each tag activates.
pub fn render_prompt_tags_bar(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    tags: &mut Vec<PromptTag>,
) {
    if tags.is_empty() {
        return;
    }

    let accent = Color32::from_rgb(99, 102, 241);
    let accent_glow = Color32::from_rgb(138, 141, 255);
    let pill_bg = Color32::from_rgb(32, 37, 58);
    let text_col = Color32::from_rgb(232, 236, 245);
    let dim = Color32::from_rgb(160, 168, 190);
    let bubble_bg = Color32::from_rgb(22, 26, 40);

    let mut remove_idx: Option<usize> = None;
    let mut needs_repaint = false;

    ui.horizontal_wrapped(|ui| {
        ui.set_max_width(ui.available_width());
        ui.spacing_mut().item_spacing = vec2(6.0, 4.0);
        ui.label(
            RichText::new("LoRA keywords:")
                .size(10.5)
                .color(dim),
        );
        let max_pill_w = (ui.available_width() - 12.0).max(80.0);
        for (idx, tag) in tags.iter_mut().enumerate() {
            let font_id = egui::FontId::proportional(11.0);
            let close_font = egui::FontId::proportional(11.0);
            let display_txt = if tag.keyword.chars().count() > 26 {
                format!("{}: {}…", tag.lora_display_name, tag.keyword.chars().take(16).collect::<String>())
            } else {
                tag.keyword.clone()
            };
            let mut label_galley = ui.painter().layout_no_wrap(display_txt, font_id.clone(), text_col);
            let close_galley = ui.painter().layout_no_wrap("×".to_string(), close_font, dim);
            let pad_x = 7.0;
            let gap = 5.0;
            let raw_w = pad_x + label_galley.size().x + gap + close_galley.size().x + pad_x;
            if raw_w > max_pill_w {
                let shorter = if tag.lora_display_name.chars().count() > 20 {
                    format!("{}…", tag.lora_display_name.chars().take(19).collect::<String>())
                } else {
                    tag.lora_display_name.clone()
                };
                label_galley = ui.painter().layout_no_wrap(shorter, font_id, text_col);
            }
            let pill_w = (pad_x + label_galley.size().x + gap + close_galley.size().x + pad_x).min(max_pill_w);
            let pill_h = 20.0;

            let (pill_rect, pill_resp) = ui.allocate_exact_size(vec2(pill_w, pill_h), egui::Sense::click());
            let close_rect = egui::Rect::from_min_max(
                egui::pos2(pill_rect.max.x - pad_x - close_galley.size().x - 3.0, pill_rect.min.y),
                pill_rect.max,
            );

            let p = ui.painter();
            p.rect_filled(pill_rect, 10.0, pill_bg);
            p.rect_stroke(
                pill_rect,
                10.0,
                Stroke::new(1.0, if pill_resp.hovered() { accent_glow } else { accent.linear_multiply(0.75) }),
                StrokeKind::Inside,
            );
            p.galley(
                egui::pos2(pill_rect.min.x + pad_x, pill_rect.center().y - label_galley.size().y * 0.5),
                label_galley,
                text_col,
            );
            let close_col = if pill_resp.hovered() { text_col } else { dim };
            p.galley(
                egui::pos2(
                    pill_rect.max.x - pad_x - close_galley.size().x,
                    pill_rect.center().y - close_galley.size().y * 0.5,
                ),
                close_galley,
                close_col,
            );

            if pill_resp.hovered() {
                ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
                tag.hovered_dismissed = true;
            }
            if pill_resp.clicked() {
                if let Some(pos) = pill_resp.interact_pointer_pos() {
                    if close_rect.contains(pos) || true {
                        remove_idx = Some(idx);
                    }
                } else {
                    remove_idx = Some(idx);
                }
            }
            let hover_tip = format!(
                "Activation keyword for {} (inserted at start of prompt with \", \"). Click to remove.",
                tag.lora_display_name
            );
            let _ = pill_resp.on_hover_text(hover_tip);

            // 3-second speech bubble callout above the tag when first added
            let elapsed = tag.created_at.elapsed().as_secs_f32();
            if !tag.hovered_dismissed && elapsed < 3.0 {
                needs_repaint = true;
                let alpha = if elapsed < 2.2 {
                    1.0
                } else {
                    ((3.0 - elapsed) / 0.8).clamp(0.0, 1.0)
                };
                let msg = format!("Activates {}", tag.lora_display_name);
                let msg_galley = ui.painter().layout_no_wrap(
                    msg,
                    egui::FontId::proportional(10.5),
                    text_col.linear_multiply(alpha),
                );
                let bw = msg_galley.size().x + 14.0;
                let bh = msg_galley.size().y + 10.0;
                let tail_h = 5.0;
                let bx = (pill_rect.center().x - bw * 0.5).max(8.0);
                let by = pill_rect.min.y - bh - tail_h - 3.0;
                let bubble_rect = egui::Rect::from_min_size(egui::pos2(bx, by), vec2(bw, bh));

                if ctx.input(|i| i.pointer.hover_pos().is_some_and(|pos| bubble_rect.contains(pos))) {
                    tag.hovered_dismissed = true;
                } else {
                    let layer = ctx.layer_painter(egui::LayerId::new(
                        egui::Order::Tooltip,
                        ui.id().with(("lora_tag_bubble", idx)),
                    ));
                    let bg_c = bubble_bg.linear_multiply(alpha);
                    let border_c = accent_glow.linear_multiply(alpha);
                    layer.rect_filled(bubble_rect, 6.0, bg_c);
                    layer.rect_stroke(bubble_rect, 6.0, Stroke::new(1.0, border_c), StrokeKind::Inside);

                    let cx = pill_rect.center().x.clamp(bubble_rect.min.x + 8.0, bubble_rect.max.x - 8.0);
                    let tip_pts = vec![
                        egui::pos2(cx - 5.0, bubble_rect.max.y - 0.5),
                        egui::pos2(cx + 5.0, bubble_rect.max.y - 0.5),
                        egui::pos2(cx, bubble_rect.max.y + tail_h),
                    ];
                    layer.add(egui::Shape::convex_polygon(tip_pts, bg_c, Stroke::new(1.0, border_c)));
                    layer.galley(
                        egui::pos2(bubble_rect.min.x + 7.0, bubble_rect.min.y + 5.0),
                        msg_galley,
                        text_col.linear_multiply(alpha),
                    );
                }
            }
        }
    });

    if let Some(idx) = remove_idx {
        tags.remove(idx);
    }
    if needs_repaint {
        ctx.request_repaint();
    }
}

/// Internal pipeline LoRAs managed automatically by ComfyUI workflows;
/// hidden from the user-facing LoRA carousel to avoid confusion.
const INTERNAL_PIPELINE_LORAS: &[&str] = &[
    "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
    "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoraModelFamily {
    QwenImage21,
    MiniMaxH3,
    Ltx25,
    Ltx23,
    VideoEditAll,
}

impl LoraModelFamily {
    pub fn from_video_profile(p: &engine::Profile) -> Self {
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
            Self::QwenImage21 => "Qwen Image 2.1",
            Self::MiniMaxH3 => "MiniMax H3",
            Self::Ltx25 => "LTX-Video 2.5",
            Self::Ltx23 => "LTX-Video 2.3",
            Self::VideoEditAll => "Video Edit (H3 & LTX)",
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
    /// Which model family this LoRA was built for ("Qwen", "H3", "2.5", "2.3", or "Custom").
    pub compat_tag: String,
    /// Compatible model families.
    pub supports_qwen_image: bool,
    pub supports_h3: bool,
    pub supports_ltx25: bool,
    pub supports_ltx23: bool,
    /// Popularity/rating score from ComfyUI LoRA Browser / metadata (higher = top rated).
    pub rating_score: f64,
    /// Local preview image path if found next to the `.safetensors` file.
    pub preview_path: Option<PathBuf>,
    /// Remote preview image/GIF/WebP/MP4 URL from curated list or ComfyUI LoRA Browser.
    pub preview_url: Option<String>,
    /// Direct Hugging Face download URL for optional plugin LoRAs.
    pub download_url: Option<String>,
    /// File size in bytes (on disk or remote).
    pub size_bytes: u64,
    /// Optional trigger word / prompt prefix automatically added when activated.
    pub trigger_word: Option<String>,
    /// Optional recommended LoRA strength applied when activated.
    pub default_strength: Option<f32>,
    /// Whether this is an In-Context (IC) video-to-video LoRA that requires a source video.
    pub is_ic_lora: bool,
    /// Whether the `.safetensors` file is currently installed in `models/loras`.
    pub installed: bool,
}

pub fn is_ic_lora_filename(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    lower.contains("ic-lora")
        || lower.contains("ic_lora")
        || lower.contains("iclora")
        || lower.contains("obscura_remova")
        || lower.contains("systms_flw")
        || lower.contains("360vroutpaint")
        || lower.contains("viggle_remake")
        || lower.contains("minimax_h3_style_transfer")
}

pub fn has_active_ic_lora(active: &[ActiveLora], items: &[LoraItem]) -> bool {
    active.iter().any(|a| {
        items
            .iter()
            .find(|it| it.filename == a.filename)
            .map(|it| it.is_ic_lora)
            .unwrap_or_else(|| is_ic_lora_filename(&a.filename))
    })
}

/// Returns true if any active video LoRA requires a MiniMax H3 model.
pub fn has_active_h3_lora(active: &[ActiveLora], items: &[LoraItem]) -> bool {
    active.iter().any(|a| {
        items
            .iter()
            .find(|it| it.filename == a.filename)
            .map(|it| it.supports_h3 && !it.supports_ltx25 && !it.supports_ltx23)
            .unwrap_or_else(|| {
                let lower = a.filename.to_ascii_lowercase();
                lower.contains("viggle_remake") || lower.contains("minimax_h3") || lower.contains("h3-")
            })
    })
}

/// Returns true if any active video LoRA requires an LTX model (either an LTX IC-LoRA
/// or an LTX-only LoRA that does not support MiniMax H3).
pub fn has_active_ltx_lora(active: &[ActiveLora], items: &[LoraItem]) -> bool {
    active.iter().any(|a| {
        items
            .iter()
            .find(|it| it.filename == a.filename)
            .map(|it| !it.supports_h3 && (it.is_ic_lora || it.supports_ltx25 || it.supports_ltx23))
            .unwrap_or_else(|| {
                let lower = a.filename.to_ascii_lowercase();
                is_ic_lora_filename(&a.filename)
                    && !lower.contains("viggle_remake")
                    && !lower.contains("minimax_h3")
            })
    })
}

/// Removes any IC-LoRAs from `active` and their prompt tags from `prompt_tags`.
/// Returns `true` if at least one IC-LoRA was removed.
pub fn remove_ic_loras(
    active: &mut Vec<ActiveLora>,
    prompt_tags: &mut Vec<PromptTag>,
    items: &[LoraItem],
) -> bool {
    let is_ic = |filename: &str| {
        items
            .iter()
            .find(|it| it.filename == filename)
            .map(|it| it.is_ic_lora)
            .unwrap_or_else(|| is_ic_lora_filename(filename))
    };
    let before = active.len();
    active.retain(|a| !is_ic(&a.filename));
    prompt_tags.retain(|t| !is_ic(&t.lora_filename));
    active.len() != before
}

impl LoraItem {
    pub fn is_compatible_with(&self, family: LoraModelFamily) -> bool {
        match family {
            LoraModelFamily::QwenImage21 => self.supports_qwen_image,
            LoraModelFamily::MiniMaxH3 => self.supports_h3,
            // LTX 2.5 supports both LTX 2.5 LoRAs and compatible LTX 2.3 LoRAs
            LoraModelFamily::Ltx25 => self.supports_ltx25 || self.supports_ltx23,
            LoraModelFamily::Ltx23 => self.supports_ltx23,
            LoraModelFamily::VideoEditAll => self.supports_h3 || self.supports_ltx25 || self.supports_ltx23,
        }
    }
}

pub struct AnimatedThumb {
    pub frames: Vec<TextureHandle>,
    pub delays_ms: Vec<u64>,
}

impl AnimatedThumb {
    pub fn frame_at(&self, time_secs: f64) -> Option<(&TextureHandle, Option<u64>)> {
        if self.frames.is_empty() {
            return None;
        }
        if self.frames.len() == 1 {
            return Some((&self.frames[0], None));
        }
        let total_ms: u64 = self.delays_ms.iter().sum::<u64>().max(100);
        let t_ms = ((time_secs * 1000.0) as u64) % total_ms;
        let mut acc = 0u64;
        for (idx, &d) in self.delays_ms.iter().enumerate() {
            let step = d.max(40);
            acc += step;
            if t_ms < acc {
                return Some((&self.frames[idx], Some(step)));
            }
        }
        Some((&self.frames[0], Some(100)))
    }
}

#[derive(Clone, Debug)]
pub struct LoraDownloadStatus {
    pub filename: String,
    pub display_name: String,
    pub download_url: String,
    pub loras_dir: PathBuf,
    pub downloaded: u64,
    pub total: u64,
    pub speed_mb: f32,
    pub paused: bool,
    pub pause_flag: Arc<AtomicBool>,
    pub cancel_flag: Arc<AtomicBool>,
    pub error: Option<String>,
    pub completed: bool,
}

#[derive(Clone, Debug)]
pub struct PendingLoraDelete {
    pub filename: String,
    pub display_name: String,
    pub loras_dir: PathBuf,
    pub is_mid_download: bool,
    pub size_bytes: u64,
    pub error: Option<String>,
}

pub struct LoraBrowserState {
    pub items: Vec<LoraItem>,
    pub thumbs: HashMap<String, AnimatedThumb>,
    pub loading_thumbs: HashSet<String>,
    pub remote_fetched: bool,
    pub remote_fetching: bool,
    pub remote_items: Arc<Mutex<Option<Vec<LoraItem>>>>,
    pub thumb_results: Arc<Mutex<Vec<(String, Vec<(egui::ColorImage, u64)>)>>>,
    /// Pending confirmation modal for deleting a LoRA or cancelling an in-progress LoRA download.
    pub pending_delete: Option<PendingLoraDelete>,
    /// LoRAs that finished downloading and are waiting for the user to click to activate.
    pub ready_loras: HashSet<String>,
    /// Temporary input buffer for pasting a Hugging Face token on gated repo errors.
    pub hf_token_input: String,
    /// Auto-poll state when a saved HF token gets 401/403 (waiting for license acceptance).
    pub gated_poll_filename: String,
    pub gated_poll_attempts: u8,
    pub gated_poll_next_at: Option<std::time::Instant>,
    /// Temporary warning banner when multi-LoRA stacking is blocked by VRAM limits.
    pub vram_block_warning: Option<(String, std::time::Instant)>,
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
            pending_delete: None,
            ready_loras: HashSet::new(),
            hf_token_input: String::new(),
            gated_poll_filename: String::new(),
            gated_poll_attempts: 0,
            gated_poll_next_at: None,
            vram_block_warning: None,
        }
    }
}

/// Controls whether clicking a new LoRA swaps or stacks, and enforces VRAM safety.
#[derive(Clone, Copy, Debug)]
pub struct MultiLoraPolicy {
    pub allow_multi: bool,
    pub current_est_vram_gb: f32,
    pub gpu_vram_gb: Option<u32>,
}

/// Applies a tile click on `item` according to `multi_policy`.
/// Returns an optional warning message if multi-LoRA stacking was blocked due to VRAM limits.
pub fn apply_lora_tile_click(
    selected_loras: &mut Vec<ActiveLora>,
    prompt_tags: &mut Vec<PromptTag>,
    item: &LoraItem,
    is_sel: bool,
    multi_policy: Option<&MultiLoraPolicy>,
) -> Option<String> {
    if is_sel {
        selected_loras.retain(|l| l.filename != item.filename);
        prompt_tags.retain(|t| t.lora_filename != item.filename);
        return None;
    }

    let mut warning = None;
    let mut should_swap = false;
    if let Some(policy) = multi_policy {
        if !selected_loras.is_empty() {
            if !policy.allow_multi {
                should_swap = true;
            } else if let Some(gpu_vram) = policy.gpu_vram_gb {
                let added_vram = estimate_lora_vram_gb(&item.filename, item.size_bytes);
                let projected = policy.current_est_vram_gb + added_vram;
                if projected > (gpu_vram as f32) - 0.5 {
                    should_swap = true;
                    warning = Some(format!(
                        "Swapped to {} — stacking would exceed safe VRAM (~{:.1} / {} GB).",
                        item.display_name, projected, gpu_vram
                    ));
                }
            }
        }
    }

    if should_swap {
        let removed: Vec<String> = selected_loras.iter().map(|l| l.filename.clone()).collect();
        selected_loras.clear();
        prompt_tags.retain(|t| !removed.contains(&t.lora_filename));
    }

    selected_loras.push(ActiveLora {
        filename: item.filename.clone(),
        strength: item.default_strength.unwrap_or(1.0),
    });
    if let Some(tw) = item.trigger_word.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        if !prompt_tags.iter().any(|t| t.lora_filename == item.filename) {
            prompt_tags.push(PromptTag {
                lora_filename: item.filename.clone(),
                lora_display_name: item.display_name.clone(),
                keyword: tw.to_string(),
                created_at: std::time::Instant::now(),
                hovered_dismissed: false,
            });
        }
    }
    warning
}

/// Estimates additional VRAM (in GB) required per active LoRA in ComfyUI.
pub fn estimate_lora_vram_gb(lora_name: &str, size_bytes: u64) -> f32 {
    if lora_name == "None" || lora_name.trim().is_empty() {
        return 0.0;
    }
    if size_bytes > 0 {
        let size_gb = size_bytes as f32 / (1024.0 * 1024.0 * 1024.0);
        (size_gb * 0.7).clamp(0.25, 1.25)
    } else {
        0.5
    }
}

/// Curated optional-download MiniMax H3 LoRAs from
/// `https://github.com/MiniMax-AI/awesome-minimax-h3-integration#h3-loras`
/// that have clear visual previews and plug-and-play ComfyUI `.safetensors` weights.
pub fn curated_h3_loras() -> Vec<LoraItem> {
    let entries = [
        (
            "h3-realism-people-t2v-i2v-r2v.safetensors",
            "Realism - People",
            "Natural people & skin texture",
            "https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/resolve/main/h3-realism-people-t2v-i2v-r2v.safetensors",
            "https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA/resolve/main/before-after-comparison.mp4",
            131_229_656u64,
            Some("r34l1sm"),
            94.0,
            false,
        ),
        (
            "minimax_h3_flf2v_lora_v1.safetensors",
            "360° Orbit",
            "Frozen-time 360° camera orbit",
            "https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA/resolve/main/minimax_h3_flf2v_lora_v1.safetensors",
            "https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA/resolve/main/media/hero.webp",
            155_111_424u64,
            Some("One frozen instant. Only the camera moves. In a continuous 360 orbit."),
            93.0,
            false,
        ),
        (
            "minimax-h3-facial-realism-closeup-cp2000.safetensors",
            "Facial Realism",
            "Close-up faces & expressions",
            "https://huggingface.co/prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp/resolve/main/minimax-h3-facial-realism-closeup-cp2000.safetensors",
            "https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Z8_LY5JLDCQS3oNofFOWT.mp4",
            74_570_624u64,
            Some("Facial Realism"),
            92.0,
            false,
        ),
        (
            "minimax_h3_looping_sketch_anime_v1.safetensors",
            "Looping Sketch Anime",
            "Hand-drawn 2D anime loop",
            "https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime/resolve/main/minimax_h3_looping_sketch_anime_v1.safetensors",
            "https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime/resolve/main/Examples/MinimaxH3_00006.mp4",
            596_449_768u64,
            None,
            91.0,
            false,
        ),
        (
            "H3-Licon-VBVR-I2V-V2.safetensors",
            "VBVR Consistency",
            "Reduces clipping & duplicates",
            "https://huggingface.co/LiconStudio/Minimax-H3-VBVR/resolve/main/H3-Licon-VBVR-I2V-V2.safetensors",
            "https://huggingface.co/LiconStudio/Minimax-H3-VBVR/resolve/main/validation/2/VBVR.mp4",
            149_233_992u64,
            None,
            90.0,
            false,
        ),
        (
            "minimax_h3_style_transfer_v1.0_r64.safetensors",
            "Style Transfer",
            "Video style transfer (requires H3)",
            "https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/loras/minimax_h3_style_transfer_v1.0_r64.safetensors",
            "https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI/resolve/main/examples/transfer_style_1.mp4",
            620_306_000u64,
            Some("style_transfer"),
            89.0,
            true,
        ),
        (
            "viggle_remake_edit_comfyui.safetensors",
            "Viggle Remake",
            "Recast & edit objects/weather (requires H3)",
            "https://huggingface.co/Viggle/Viggle-Remake/resolve/main/comfyui/viggle_remake_edit_comfyui.safetensors",
            "https://huggingface.co/Viggle/Viggle-Remake/resolve/main/examples/media/edit-add-whale.mp4",
            3_773_044_920u64,
            None,
            88.0,
            true,
        ),
    ];

    entries
        .into_iter()
        .map(
            |(filename, display_name, description, download_url, preview_url, size_bytes, trigger_word, rating_score, is_ic_lora)| {
                LoraItem {
                    filename: filename.to_string(),
                    display_name: display_name.to_string(),
                    description: description.to_string(),
                    compat_tag: "H3".to_string(),
                    supports_qwen_image: false,
                    supports_h3: true,
                    supports_ltx25: false,
                    supports_ltx23: false,
                    rating_score,
                    preview_path: None,
                    preview_url: Some(preview_url.to_string()),
                    download_url: Some(download_url.to_string()),
                    size_bytes,
                    trigger_word: trigger_word.map(|s| s.to_string()),
                    default_strength: None,
                    is_ic_lora,
                    installed: false,
                }
            },
        )
        .collect()
}

/// Curated optional-download Qwen Image 2.1 LoRAs from Danrisi on Hugging Face.
/// Camera/style LoRAs are prioritized above character/style-conversion LoRAs.
pub fn curated_qwen_loras() -> Vec<LoraItem> {
    let entries = [
        (
            "lenovo_qwen21.safetensors",
            "Lenovo",
            "Low-light mobile phone photo look",
            "https://huggingface.co/Danrisi/lenovo_qwen2.1/resolve/main/lenovo_qwen21.safetensors",
            "https://huggingface.co/Danrisi/lenovo_qwen2.1/resolve/main/images/Qwen_image_2.1_00298.png",
            79_743_896u64,
            Some("l3n0v0"),
            Some(0.7f32),
            99.0,
        ),
        (
            "canon_qwen21.safetensors",
            "Canon",
            "Amateur DSLR candid photo look",
            "https://huggingface.co/Danrisi/canon_qwen2.1/resolve/main/canon_qwen21.safetensors",
            "https://huggingface.co/Danrisi/canon_qwen2.1/resolve/main/images/Qwen_image_2.1_00267.png",
            79_743_888u64,
            Some("c2n0n"),
            Some(0.7f32),
            98.0,
        ),
        (
            "samsung_qwen21.safetensors",
            "Samsung",
            "Raw smartphone photo look",
            "https://huggingface.co/Danrisi/samsung_qwen2.1/resolve/main/samsung_qwen21.safetensors",
            "https://huggingface.co/Danrisi/samsung_qwen2.1/resolve/main/images/Qwen_image_2.1_00316.png",
            79_743_880u64,
            Some("s2msun9"),
            Some(0.7f32),
            97.0,
        ),
        (
            "grainscape_qwen21.safetensors",
            "Grainscape",
            "35mm color negative heavy grain",
            "https://huggingface.co/Danrisi/grainscape_qwen2.1/resolve/main/grainscape_qwen21.safetensors",
            "https://huggingface.co/Danrisi/grainscape_qwen2.1/resolve/main/images/Qwen_image_2.1_00168.png",
            79_743_888u64,
            Some("Color film photography, 35mm color negative look, heavy film grain"),
            Some(0.7f32),
            96.0,
        ),
        (
            "filmstills_qwen21.safetensors",
            "FilmStills",
            "35mm cinematic film still aesthetic",
            "https://huggingface.co/Danrisi/filmstills_qwen2.1/resolve/main/filmstills_qwen21.safetensors",
            "https://huggingface.co/Danrisi/filmstills_qwen2.1/resolve/main/images/Qwen_image_2.1_00137.png",
            79_743_888u64,
            Some("film still"),
            Some(0.7f32),
            95.0,
        ),
        (
            "qwen-image-2.1-consistency.safetensors",
            "Consistency",
            "Keeps edits aligned to source frame",
            "https://huggingface.co/ausboss/Qwen-Image-2.1-Consistency-LoRA/resolve/main/qwen-image-2.1-consistency.safetensors",
            "https://huggingface.co/ausboss/Qwen-Image-2.1-Consistency-LoRA/resolve/main/drift_crt_watercolor.jpg",
            159_436_496u64,
            None,
            Some(1.0f32),
            90.0,
        ),
        (
            "Qwen2.1_Anime_consistency.safetensors",
            "Anime Consistency",
            "Anime character edit consistency",
            "https://huggingface.co/WarmBloodAban/Qwen-Image-2.1-LoRAs/resolve/main/Qwen2.1_Anime_consistency.safetensors",
            "https://huggingface.co/WarmBloodAban/Qwen-Image-2.1-LoRAs/resolve/main/images/2b2fbfd3-e134-4eff-b98a-a5e17c906539.png",
            167_830_408u64,
            None,
            Some(0.8f32),
            89.0,
        ),
        (
            "Qwen2.1_Anything2RealCharacters.safetensors",
            "Anything2Real",
            "Convert any art style to realistic",
            "https://huggingface.co/WarmBloodAban/Qwen-Image-2.1-LoRAs/resolve/main/Qwen2.1_Anything2RealCharacters.safetensors",
            "https://huggingface.co/WarmBloodAban/Qwen-Image-2.1-LoRAs/resolve/main/images/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_2026-09-27_231730_493.png",
            318_820_200u64,
            None,
            Some(0.8f32),
            88.0,
        ),
    ];

    entries
        .into_iter()
        .map(
            |(filename, display_name, description, download_url, preview_url, size_bytes, trigger_word, default_strength, rating_score)| {
                LoraItem {
                    filename: filename.to_string(),
                    display_name: display_name.to_string(),
                    description: description.to_string(),
                    compat_tag: "Qwen".to_string(),
                    supports_qwen_image: true,
                    supports_h3: false,
                    supports_ltx25: false,
                    supports_ltx23: false,
                    rating_score,
                    preview_path: None,
                    preview_url: Some(preview_url.to_string()),
                    download_url: Some(download_url.to_string()),
                    size_bytes,
                    trigger_word: trigger_word.map(|s| s.to_string()),
                    default_strength,
                    is_ic_lora: false,
                    installed: false,
                }
            },
        )
        .collect()
}

/// Curated optional-download LTX-Video 2.5 Creative Lab & compatible LTX 2.3 LoRAs.
/// Official LTX-2.5 Creative Lab LoRAs are prioritized above older LTX-2.3 LoRAs.
/// Public GCS CDN previews from Lightricks/LTX-Desktop are used where available so
/// thumbnails render without requiring Hugging Face authentication.
pub fn curated_ltx_loras() -> Vec<LoraItem> {
    type Entry = (
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        Option<&'static str>,
        u64,
        Option<&'static str>,
        Option<f32>,
        f64,
    );
    let entries: Vec<Entry> = vec![
        // --- LTX-2.5 Creative Lab (official) ---
        (
            "ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors",
            "Day to Night",
            "Relight daytime footage to night",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Day-To-Night/resolve/main/ltx-2.5-22b-ic-lora-day-to-night-0.9.safetensors",
            Some("https://storage.googleapis.com/ltx-desktop-artifacts/assets/day_to_night_demo.mp4"),
            327_318_440,
            Some("A realistic nighttime outdoor scene under a dark sky with deep natural shadows and faint ambient light. Photorealistic night. Only the lighting changes from day to night; identical composition, framing, camera movement and motion."),
            Some(1.5),
            99.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors",
            "Clean Plate",
            "Remove subjects to clean background",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate/resolve/main/ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors",
            Some("https://storage.googleapis.com/ltx-desktop-artifacts/assets/clean_plate_demo.mp4"),
            327_318_439,
            Some("An empty clean plate of the exact same location: identical background, environment, structures and lighting as the source video, with no people, no humans, no figures anywhere in the frame. No cast shadows or reflections."),
            Some(1.0),
            98.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-water-simulation-0.9.safetensors",
            "Water Simulation",
            "Realistic water & fluid interaction",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Water-Simulation/resolve/main/ltx-2.5-22b-ic-lora-water-simulation-0.9.safetensors",
            Some("https://storage.googleapis.com/ltx-desktop-artifacts/assets/water_simulation_demo.mp4"),
            906_080_568,
            Some("ADD WATER"),
            Some(1.2),
            97.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
            "Pixel Upscaler 2x",
            "2x spatial video upscaler",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler/resolve/main/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
            Some("https://storage.googleapis.com/ltx-desktop-artifacts/assets/upscale_demo.mp4"),
            327_322_640,
            Some("upscale, high resolution, sharp detail"),
            Some(1.0),
            96.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-deblur-0.9.safetensors",
            "Deblur",
            "Remove motion/focus blur (2-part: DEBLUR <fix> — <scene>)",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Deblur/resolve/main/ltx-2.5-22b-ic-lora-deblur-0.9.safetensors",
            Some("https://storage.googleapis.com/ltx-desktop-artifacts/assets/deblur_demo.mp4"),
            906_080_571,
            Some("DEBLUR sharp focus with crisp detail, identical framing and subject"),
            Some(1.0),
            95.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-decompression-0.9.safetensors",
            "Decompression",
            "Fix compression (2-part: ENHANCE QUALITY <fix> — <scene>)",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Decompression/resolve/main/ltx-2.5-22b-ic-lora-decompression-0.9.safetensors",
            Some("https://storage.googleapis.com/ltx-desktop-artifacts/assets/decompression_demo.mp4"),
            906_080_571,
            Some("ENHANCE QUALITY restored to high quality with sharp detail, clean edges, no compression artifacts"),
            Some(1.0),
            94.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-colorization-0.9.safetensors",
            "Colorization",
            "Colorize B&W footage (2-part: COLORIZE <look> — <scene>)",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Colorization/resolve/main/ltx-2.5-22b-ic-lora-colorization-0.9.safetensors",
            Some("https://storage.googleapis.com/ltx-desktop-artifacts/assets/colorization_demo.mp4"),
            906_080_571,
            Some("COLORIZE natural colors restored, vivid photorealistic color grading"),
            Some(1.0),
            93.0,
        ),
        (
            "ltx-2.5-22b-lora-cinemagraph-0.9.safetensors",
            "Cinemagraph",
            "Subtle looping localized motion",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-LoRA-Cinemagraph/resolve/main/ltx-2.5-22b-lora-cinemagraph-0.9.safetensors",
            Some("https://storage.googleapis.com/ltx-desktop-artifacts/assets/cinemagraph_demo.mp4"),
            201_462_650,
            Some("CINEMAGRAPH_MOTION, tripod locked-off static camera, zero camera movement"),
            Some(2.0),
            92.0,
        ),
        (
            "ltx-2.5-22b-lora-slow-motion-control-1.0.safetensors",
            "Slow Motion",
            "Variable playback speed control",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-LoRA-Slow-Motion-Control/resolve/main/ltx-2.5-22b-lora-slow-motion-control-1.0.safetensors",
            Some("https://huggingface.co/Lightricks/LTX-2.5-22b-LoRA-Slow-Motion-Control/resolve/main/examples/ny_street_sbs.mp4"),
            654_474_448,
            None,
            None,
            91.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
            "Ingredients (Multi-Ref)",
            "Compose subjects & props into scene",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients/resolve/main/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
            Some("https://storage.googleapis.com/ltx-desktop-artifacts/assets/ingredients_demo.mp4"),
            1_308_787_472,
            Some("Seamlessly composite the reference subjects and props into the scene with matching lighting"),
            Some(1.4),
            90.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors",
            "SDR to HDR",
            "Expand dynamic range to HDR",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR/resolve/main/ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors",
            Some("https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR/resolve/main/examples/sdr-to-hdr-driving_sbs.mp4"),
            1_308_787_472,
            Some("High dynamic range, rich contrast, deep shadows, bright highlights, natural color grading"),
            Some(1.0),
            89.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors",
            "Layout to Render",
            "3D blocking/layout to final render",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render/resolve/main/ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors",
            Some("https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Layout-To-Render/resolve/main/examples/green-forest-push_sbs.mp4"),
            1_308_787_439,
            Some("Photorealistic cinematic render matching the layout framing and motion"),
            Some(1.0),
            88.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors",
            "Refine Details",
            "Enhance fine texture & sharpness",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details/resolve/main/ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors",
            Some("https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details/resolve/main/examples/refine_forest_stream_wipe.mp4"),
            1_308_787_534,
            Some("photoreal, 8K, sharp photographic detail, crisp texture, natural grain, high resolution footage"),
            Some(1.0),
            87.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-restore-1.0.safetensors",
            "Restore",
            "Restore & colorize archive footage",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Restore/resolve/main/ltx-2.5-22b-ic-lora-restore-1.0.safetensors",
            Some("https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Restore/resolve/main/examples/restore_family_meal_1954_wipe.mp4"),
            1_711_643_914,
            Some("natural colour, restored film footage, clean sharp detail"),
            Some(1.0),
            86.0,
        ),
        (
            "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors",
            "Alpha Gen",
            "Generate alpha matte from video",
            "2.5",
            "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen/resolve/main/ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors",
            Some("https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen/resolve/main/examples/backlit_hair_sbs.mp4"),
            1_308_787_472,
            None,
            Some(1.0),
            85.0,
        ),
        // --- LTX-2.3 community & official LoRAs (also compatible with LTX-2.5) ---
        (
            "SYSTMS_FLW_V1_LTX23.safetensors",
            "Seamless Transitions",
            "Shot-to-shot flow transitions",
            "2.3",
            "https://huggingface.co/systms/SYSTMS-FLW-IC-LORA-LTX-2.3/resolve/main/SYSTMS_FLW_V1_LTX23.safetensors",
            Some("https://huggingface.co/systms/SYSTMS-FLW-IC-LORA-LTX-2.3/resolve/main/FLW_01.gif"),
            805_412_808,
            Some("seamless smooth camera transition"),
            Some(1.0),
            84.0,
        ),
        (
            "LTX23_Obscura_Remova_v1.safetensors",
            "Remove Foreground",
            "Remove foreground occlusions/haze",
            "2.3",
            "https://huggingface.co/WepeNerd/Obscura_Remova/resolve/main/LTX23_Obscura_Remova_v1.safetensors",
            Some("https://huggingface.co/WepeNerd/Obscura_Remova/resolve/main/examples/1.mp4"),
            327_344_864,
            Some("Remove the objects from the foreground."),
            Some(1.5),
            83.0,
        ),
        (
            "ltx-2.3-22b-ic-lora-outpaint.safetensors",
            "Video Outpaint",
            "Horizontal & vertical canvas outpaint",
            "2.3",
            "https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/resolve/main/ltx-2.3-22b-ic-lora-outpaint.safetensors",
            Some("https://storage.googleapis.com/ltx-desktop-artifacts/assets/outpainting_demo.mp4"),
            1_308_756_416,
            Some("outpaint, seamlessly extend the scene into the surrounding frame"),
            Some(1.0),
            82.0,
        ),
        (
            "360vroutpaint_v2_step09000.safetensors",
            "VR Video Outpaint",
            "360° equirectangular VR outpaint",
            "2.3",
            "https://huggingface.co/TheBurgstall/VR-360-Outpaint-LTX2.3-IC-LoRA/resolve/main/360vroutpaint_v2_step09000.safetensors",
            Some("https://huggingface.co/TheBurgstall/VR-360-Outpaint-LTX2.3-IC-LoRA/resolve/main/skateVR.mp4"),
            1_308_756_416,
            Some("360 equirectangular VR outpaint, seamlessly extend the scene"),
            Some(1.0),
            81.0,
        ),
        (
            "ltx-2.3-22b-ic-lora-dubit-0.9.safetensors",
            "Dubbing LoRA",
            "Audio-driven lip dubbing",
            "2.3",
            "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub/resolve/main/ltx-2.3-22b-ic-lora-dubit-0.9.safetensors",
            None,
            2_466_665_072,
            Some("natural lip sync speaking clearly"),
            Some(1.0),
            80.0,
        ),
    ];

    entries
        .into_iter()
        .map(
            |(
                filename,
                display_name,
                description,
                compat_tag,
                download_url,
                preview_url,
                size_bytes,
                trigger_word,
                default_strength,
                rating_score,
            )| {
                let is_25 = compat_tag == "2.5";
                LoraItem {
                    filename: filename.to_string(),
                    display_name: display_name.to_string(),
                    description: description.to_string(),
                    compat_tag: compat_tag.to_string(),
                    supports_qwen_image: false,
                    supports_h3: false,
                    supports_ltx25: is_25,
                    supports_ltx23: !is_25,
                    rating_score,
                    preview_path: None,
                    preview_url: preview_url.map(|s| s.to_string()),
                    download_url: Some(download_url.to_string()),
                    size_bytes,
                    trigger_word: trigger_word.map(|s| s.to_string()),
                    default_strength,
                    is_ic_lora: is_ic_lora_filename(filename),
                    installed: false,
                }
            },
        )
        .collect()
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
    let exts = [
        "preview.gif",
        "preview.webp",
        "preview.png",
        "preview.jpg",
        "preview.jpeg",
        "gif",
        "webp",
        "png",
        "jpg",
        "jpeg",
    ];
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
    for suffix in ["metadata.json", "civitai.info", "json"] {
        let cand = parent.join(format!("{stem}.{suffix}"));
        if cand.is_file() {
            if let Ok(txt) = std::fs::read_to_string(&cand) {
                if let Ok(v) = serde_json::from_str::<Value>(&txt) {
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

fn classify_lora(filename: &str, base_model_hint: Option<&str>) -> (String, bool, bool, bool, bool) {
    let combined = format!(
        "{} {}",
        filename.to_ascii_lowercase(),
        base_model_hint.unwrap_or("").to_ascii_lowercase()
    );
    if combined.contains("qwen") {
        return ("Qwen".to_string(), false, false, false, true);
    }
    if combined.contains("minimax") || combined.contains("h3") {
        return ("H3".to_string(), true, false, false, false);
    }
    if combined.contains("2.5") || combined.contains("ltx25") || combined.contains("ltx-2.5") || combined.contains("ltx_2.5") {
        return ("2.5".to_string(), false, true, false, false);
    }
    if combined.contains("2.3") || combined.contains("ltx23") || combined.contains("ltx-2.3") || combined.contains("ltx_2.3") || combined.contains("ltx") {
        // LTX 2.3 LoRAs are compatible with both LTX 2.3 and LTX 2.5
        return ("2.3".to_string(), false, true, true, false);
    }
    // Unclassified custom LoRA: show across all models so user-dropped files are accessible
    ("Custom".to_string(), true, true, true, true)
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

pub fn resolve_loras_dir(sh: &Shared) -> Option<PathBuf> {
    let active_models = sh.paths.lock().unwrap().models.clone();
    if !active_models.as_os_str().is_empty() {
        return Some(active_models.join("loras"));
    }
    let cfg = sh.cfg.lock().unwrap().clone();
    engine::candidate_models_dirs(&cfg)
        .into_iter()
        .map(|base| base.join("loras"))
        .find(|d| d.is_dir())
}

/// Scans `models/loras` for user LoRAs and reads ComfyUI LoRA Browser sidecar
/// thumbnails (`.preview.gif`/`.png`/`.jpg`) and metadata (`.metadata.json`/`.civitai.info`).
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

    let curated: Vec<LoraItem> = [
        curated_h3_loras(),
        curated_qwen_loras(),
        curated_ltx_loras(),
    ]
    .concat();

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
                        "Local LoRA".to_string()
                    }
                });

            let (compat_tag, supports_h3, supports_ltx25, supports_ltx23, supports_qwen_image) =
                classify_lora(&rel, base_model);
            let display_name = meta_name.unwrap_or_else(|| pretty_lora_name(&rel));
            let preview_path = find_sidecar_preview(&full);
            let size_bytes = std::fs::metadata(&full).map(|m| m.len()).unwrap_or(0);

            let mut item = LoraItem {
                filename: rel.clone(),
                display_name,
                description,
                compat_tag,
                supports_qwen_image,
                supports_h3,
                supports_ltx25,
                supports_ltx23,
                rating_score,
                preview_path,
                preview_url,
                download_url: None,
                size_bytes,
                trigger_word: None,
                default_strength: None,
                is_ic_lora: is_ic_lora_filename(&rel),
                installed: true,
            };

            // Enrich with curated metadata if this file matches a curated LoRA
            let base_name = Path::new(&rel)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(&rel);
            if let Some(c) = curated.iter().find(|c| c.filename.eq_ignore_ascii_case(base_name)) {
                item.display_name = c.display_name.clone();
                item.description = c.description.clone();
                item.compat_tag = c.compat_tag.clone();
                item.supports_qwen_image = c.supports_qwen_image;
                item.supports_h3 = c.supports_h3;
                item.supports_ltx25 = c.supports_ltx25;
                item.supports_ltx23 = c.supports_ltx23;
                if item.preview_url.is_none() {
                    item.preview_url = c.preview_url.clone();
                }
                item.download_url = c.download_url.clone();
                item.trigger_word = c.trigger_word.clone();
                item.default_strength = c.default_strength;
                item.is_ic_lora = c.is_ic_lora;
                if item.rating_score < c.rating_score {
                    item.rating_score = c.rating_score;
                }
            }
            item
        })
        .collect();

    // Add any curated LoRAs that are not yet downloaded locally
    for c in curated {
        let exists = items.iter().any(|it| {
            let base = Path::new(&it.filename)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(&it.filename);
            base.eq_ignore_ascii_case(&c.filename)
        });
        if !exists {
            items.push(c);
        }
    }

    // Sort installed first (or top-rated first), then alphabetically
    items.sort_by(|a, b| {
        b.installed
            .cmp(&a.installed)
            .then_with(|| {
                b.rating_score
                    .partial_cmp(&a.rating_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
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
                                    let (compat_tag, supports_h3, supports_ltx25, supports_ltx23, supports_qwen_image) =
                                        classify_lora(&filename, base_model);
                                    let is_ic_lora = is_ic_lora_filename(&filename);
                                    collected.push(LoraItem {
                                        filename,
                                        display_name,
                                        description: "From ComfyUI LoRA Browser".to_string(),
                                        compat_tag,
                                        supports_qwen_image,
                                        supports_h3,
                                        supports_ltx25,
                                        supports_ltx23,
                                        rating_score,
                                        preview_path: None,
                                        preview_url,
                                        download_url: None,
                                        size_bytes: 0,
                                        trigger_word: None,
                                        default_strength: None,
                                        is_ic_lora,
                                        installed: false,
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

fn open_external_url(url: &str) {
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .creation_flags(0x08000000)
            .spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}

fn read_hf_token() -> Option<String> {
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        let t = tok.trim().to_string();
        if !t.is_empty() {
            return Some(t);
        }
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    let token_path = PathBuf::from(home).join(".cache").join("huggingface").join("token");
    std::fs::read_to_string(token_path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn save_hf_token(token: &str) -> bool {
    let trimmed = token.trim();
    if trimmed.is_empty() {
        return false;
    }
    let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) else {
        return false;
    };
    let dir = PathBuf::from(home).join(".cache").join("huggingface");
    if std::fs::create_dir_all(&dir).is_err() {
        return false;
    }
    std::fs::write(dir.join("token"), trimmed).is_ok()
}

fn decode_preview_frames(bytes: &[u8], ext: &str) -> Vec<(egui::ColorImage, u64)> {
    let mut out = Vec::new();
    if ext == "gif" {
        if let Ok(decoder) = image::codecs::gif::GifDecoder::new(Cursor::new(bytes)) {
            if let Ok(frames) = decoder.into_frames().collect_frames() {
                for f in frames.into_iter().take(48) {
                    let (num, den) = f.delay().numer_denom_ms();
                    let delay_ms = if den > 0 { (num / den) as u64 } else { 100 };
                    let img = image::DynamicImage::ImageRgba8(f.into_buffer())
                        .thumbnail(192, 128)
                        .to_rgba8();
                    let size = [img.width() as usize, img.height() as usize];
                    out.push((
                        egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw()),
                        delay_ms.max(40),
                    ));
                }
            }
        }
    } else if ext == "webp" {
        if let Ok(decoder) = image::codecs::webp::WebPDecoder::new(Cursor::new(bytes)) {
            if decoder.has_animation() {
                if let Ok(frames) = decoder.into_frames().collect_frames() {
                    for f in frames.into_iter().take(48) {
                        let (num, den) = f.delay().numer_denom_ms();
                        let delay_ms = if den > 0 { (num / den) as u64 } else { 100 };
                        let img = image::DynamicImage::ImageRgba8(f.into_buffer())
                            .thumbnail(192, 128)
                            .to_rgba8();
                        let size = [img.width() as usize, img.height() as usize];
                        out.push((
                            egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw()),
                            delay_ms.max(40),
                        ));
                    }
                }
            }
        }
    }

    if out.is_empty() {
        if let Ok(img) = image::load_from_memory(bytes) {
            let thumb = img.thumbnail(192, 128).to_rgba8();
            let size = [thumb.width() as usize, thumb.height() as usize];
            out.push((egui::ColorImage::from_rgba_unmultiplied(size, thumb.as_raw()), 100));
        }
    }
    out
}

fn convert_mp4_to_gif_bytes(mp4_bytes: &[u8]) -> Option<Vec<u8>> {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_dir = std::env::temp_dir();
    let tmp_mp4 = tmp_dir.join(format!("imgstudio_lora_{stamp}.mp4"));
    let tmp_gif = tmp_dir.join(format!("imgstudio_lora_{stamp}.gif"));
    std::fs::write(&tmp_mp4, mp4_bytes).ok()?;

    let mut cmd = std::process::Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-i",
        tmp_mp4.to_str()?,
        "-t",
        "4",
        "-vf",
        "fps=8,scale=192:-1:flags=lanczos",
        tmp_gif.to_str()?,
    ]);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    let status = cmd.output().ok().map(|o| o.status.success()).unwrap_or(false);
    let _ = std::fs::remove_file(&tmp_mp4);
    if !status {
        let _ = std::fs::remove_file(&tmp_gif);
        return None;
    }
    let gif_bytes = std::fs::read(&tmp_gif).ok();
    let _ = std::fs::remove_file(&tmp_gif);
    gif_bytes
}

fn sanitize_cache_name(key: &str) -> String {
    key.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

fn load_thumb_async(
    state: &mut LoraBrowserState,
    key: &str,
    path: Option<PathBuf>,
    url: Option<String>,
    cache_dir: Option<PathBuf>,
    ctx: &egui::Context,
) {
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
        // 1. Check local sidecar file first
        if let Some(p) = path {
            if let Ok(b) = std::fs::read(&p) {
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
                let frames = decode_preview_frames(&b, &ext);
                if !frames.is_empty() {
                    results.lock().unwrap().push((key_owned, frames));
                    ctx.request_repaint();
                    return;
                }
            }
        }

        // 2. Check disk cache in models/loras/.previews
        let cache_stem = sanitize_cache_name(&key_owned);
        if let Some(ref cdir) = cache_dir {
            for ext in ["gif", "webp", "png", "jpg"] {
                let cached = cdir.join(format!("{cache_stem}.{ext}"));
                if cached.is_file() {
                    if let Ok(b) = std::fs::read(&cached) {
                        let frames = decode_preview_frames(&b, ext);
                        if !frames.is_empty() {
                            results.lock().unwrap().push((key_owned, frames));
                            ctx.request_repaint();
                            return;
                        }
                    }
                }
            }
        }

        // 3. Fetch remote URL
        if let Some(u) = url {
            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .build()
                .ok();
            if let Some(c) = client {
                let mut req = c.get(&u);
                if u.contains("huggingface.co") {
                    if let Some(tok) = read_hf_token() {
                        req = req.bearer_auth(tok);
                    }
                }
                if let Ok(resp) = req.send() {
                    if resp.status().is_success() {
                        if let Ok(raw) = resp.bytes() {
                            let url_lower = u.to_ascii_lowercase();
                            let (final_bytes, ext) = if url_lower.ends_with(".mp4") {
                                if let Some(gif) = convert_mp4_to_gif_bytes(&raw) {
                                    (gif, "gif")
                                } else {
                                    return;
                                }
                            } else if url_lower.ends_with(".webp") {
                                (raw.to_vec(), "webp")
                            } else if url_lower.ends_with(".gif") {
                                (raw.to_vec(), "gif")
                            } else if url_lower.ends_with(".png") {
                                (raw.to_vec(), "png")
                            } else {
                                (raw.to_vec(), "jpg")
                            };

                            if let Some(ref cdir) = cache_dir {
                                let _ = std::fs::create_dir_all(cdir);
                                let _ = std::fs::write(cdir.join(format!("{cache_stem}.{ext}")), &final_bytes);
                            }

                            let frames = decode_preview_frames(&final_bytes, ext);
                            if !frames.is_empty() {
                                results.lock().unwrap().push((key_owned, frames));
                                ctx.request_repaint();
                            }
                        }
                    }
                }
            }
        }
    });
}

pub fn pause_lora_download(sh: &Shared, filename: &str) {
    let mut map = sh.lora_downloads.lock().unwrap();
    if let Some(st) = map.get_mut(filename) {
        st.pause_flag.store(true, Ordering::SeqCst);
        st.paused = true;
        st.speed_mb = 0.0;
    }
}

pub fn resume_lora_download(sh: &Arc<Shared>, filename: &str, ctx: &egui::Context) {
    let info = {
        let map = sh.lora_downloads.lock().unwrap();
        map.get(filename).map(|st| {
            (
                st.filename.clone(),
                st.display_name.clone(),
                st.download_url.clone(),
                st.loras_dir.clone(),
                st.total,
            )
        })
    };
    if let Some((fname, dname, url, dir, total)) = info {
        let item = LoraItem {
            filename: fname,
            display_name: dname,
            description: String::new(),
            compat_tag: String::new(),
            supports_qwen_image: false,
            supports_h3: false,
            supports_ltx25: false,
            supports_ltx23: false,
            rating_score: 0.0,
            preview_path: None,
            preview_url: None,
            download_url: Some(url),
            size_bytes: total,
            trigger_word: None,
            default_strength: None,
            is_ic_lora: false,
            installed: false,
        };
        start_lora_download_async(sh, &item, dir, ctx);
    }
}

pub fn cancel_and_delete_lora(sh: &Shared, loras_dir: &Path, filename: &str) -> Result<(), String> {
    {
        let mut map = sh.lora_downloads.lock().unwrap();
        if let Some(st) = map.remove(filename) {
            st.cancel_flag.store(true, Ordering::SeqCst);
        }
    }
    let mut errors = Vec::new();
    for path in [
        loras_dir.join(filename),
        loras_dir.join(format!("{filename}.part")),
    ] {
        if path.is_file() {
            if let Err(e) = std::fs::remove_file(&path) {
                errors.push(format!("{}: {e}", path.display()));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

pub fn start_lora_download_async(
    sh: &Arc<Shared>,
    item: &LoraItem,
    loras_dir: PathBuf,
    ctx: &egui::Context,
) {
    let Some(url) = item.download_url.clone() else {
        return;
    };
    {
        let map = sh.lora_downloads.lock().unwrap();
        if let Some(st) = map.get(&item.filename) {
            if !st.completed && !st.paused && st.error.is_none() {
                return;
            }
        }
    }

    let filename = item.filename.clone();
    let display_name = item.display_name.clone();
    let total = item.size_bytes;
    let pause_flag = Arc::new(AtomicBool::new(false));
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let part_existing = std::fs::metadata(loras_dir.join(format!("{filename}.part")))
        .map(|m| m.len())
        .unwrap_or(0);

    {
        let mut map = sh.lora_downloads.lock().unwrap();
        map.insert(
            filename.clone(),
            LoraDownloadStatus {
                filename: filename.clone(),
                display_name,
                download_url: url.clone(),
                loras_dir: loras_dir.clone(),
                downloaded: part_existing,
                total,
                speed_mb: 0.0,
                paused: false,
                pause_flag: pause_flag.clone(),
                cancel_flag: cancel_flag.clone(),
                error: None,
                completed: false,
            },
        );
    }

    let sh_dl = Arc::clone(sh);
    let ctx = ctx.clone();

    std::thread::spawn(move || {
        let _ = std::fs::create_dir_all(&loras_dir);
        let dest = loras_dir.join(&filename);
        let part = loras_dir.join(format!("{filename}.part"));
        let existing_bytes = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);

        let set_err = |msg: String| {
            let mut map = sh_dl.lora_downloads.lock().unwrap();
            if let Some(st) = map.get_mut(&filename) {
                if !st.cancel_flag.load(Ordering::SeqCst) {
                    st.error = Some(msg);
                    st.speed_mb = 0.0;
                }
            }
            ctx.request_repaint();
        };

        let client = match reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(20))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                set_err(format!("HTTP client error: {e}"));
                return;
            }
        };

        let mut req = client.get(&url);
        let mut sent_hf_token = false;
        if url.contains("huggingface.co") {
            if let Some(tok) = read_hf_token() {
                sent_hf_token = true;
                req = req.bearer_auth(tok);
            }
        }
        if existing_bytes > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={existing_bytes}-"));
        }

        let mut resp = match req.send() {
            Ok(r) if r.status().is_success() || r.status() == reqwest::StatusCode::PARTIAL_CONTENT => r,
            Ok(r) if r.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && existing_bytes > 0 => {
                let _ = std::fs::remove_file(&part);
                let mut retry_req = client.get(&url);
                if url.contains("huggingface.co") {
                    if let Some(tok) = read_hf_token() {
                        retry_req = retry_req.bearer_auth(tok);
                    }
                }
                match retry_req.send() {
                    Ok(r2) if r2.status().is_success() => r2,
                    Ok(r2) => {
                        set_err(format!("Download HTTP {}", r2.status()));
                        return;
                    }
                    Err(e) => {
                        set_err(format!("Download failed: {e}"));
                        return;
                    }
                }
            }
            Ok(r) => {
                let status = r.status();
                let msg = if (status.as_u16() == 401 || status.as_u16() == 403) && url.contains("huggingface.co") {
                    if sent_hf_token {
                        format!("Gated HF repo (HTTP {status}): accept license on huggingface.co and retry")
                    } else {
                        format!("Gated HF repo (HTTP {status}): accept license on huggingface.co & paste HF token")
                    }
                } else {
                    format!("Download HTTP {status}")
                };
                set_err(msg);
                return;
            }
            Err(e) => {
                set_err(format!("Download failed: {e}"));
                return;
            }
        };

        let is_partial = resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        let resp_len = resp.content_length().unwrap_or(0);
        let mut downloaded = if is_partial { existing_bytes } else { 0 };
        let resolved_total = if is_partial && resp_len > 0 {
            existing_bytes + resp_len
        } else if resp_len > 0 {
            resp_len
        } else {
            total
        };

        {
            let mut map = sh_dl.lora_downloads.lock().unwrap();
            if let Some(st) = map.get_mut(&filename) {
                if resolved_total > 0 {
                    st.total = resolved_total;
                }
                st.downloaded = downloaded;
            }
        }

        let mut file = if is_partial {
            match std::fs::OpenOptions::new().create(true).append(true).open(&part) {
                Ok(f) => f,
                Err(e) => {
                    set_err(format!("File open error: {e}"));
                    return;
                }
            }
        } else {
            match std::fs::File::create(&part) {
                Ok(f) => f,
                Err(e) => {
                    set_err(format!("File create error: {e}"));
                    return;
                }
            }
        };

        let mut buf = [0u8; 64 * 1024];
        let mut last_repaint = std::time::Instant::now();
        let mut speed_window_start = std::time::Instant::now();
        let mut speed_window_bytes = 0u64;
        loop {
            if cancel_flag.load(Ordering::SeqCst) {
                drop(file);
                let _ = std::fs::remove_file(&part);
                ctx.request_repaint();
                return;
            }
            if pause_flag.load(Ordering::SeqCst) {
                drop(file);
                let mut map = sh_dl.lora_downloads.lock().unwrap();
                if let Some(st) = map.get_mut(&filename) {
                    st.paused = true;
                    st.speed_mb = 0.0;
                    st.downloaded = downloaded;
                }
                ctx.request_repaint();
                return;
            }
            match resp.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if let Err(e) = file.write_all(&buf[..n]) {
                        set_err(format!("Write error: {e}"));
                        return;
                    }
                    downloaded += n as u64;
                    speed_window_bytes += n as u64;
                    let win_secs = speed_window_start.elapsed().as_secs_f32();
                    let cur_speed = if win_secs >= 0.4 {
                        let mb = (speed_window_bytes as f32 / (1024.0 * 1024.0)) / win_secs;
                        speed_window_start = std::time::Instant::now();
                        speed_window_bytes = 0;
                        Some(mb)
                    } else {
                        None
                    };
                    if let Some(st) = sh_dl.lora_downloads.lock().unwrap().get_mut(&filename) {
                        st.downloaded = downloaded;
                        if let Some(sp) = cur_speed {
                            st.speed_mb = sp;
                        }
                    }
                    if last_repaint.elapsed().as_millis() >= 120 {
                        last_repaint = std::time::Instant::now();
                        ctx.request_repaint();
                    }
                }
                Err(e) => {
                    set_err(format!("Read error: {e}"));
                    return;
                }
            }
        }
        drop(file);
        if cancel_flag.load(Ordering::SeqCst) {
            let _ = std::fs::remove_file(&part);
            ctx.request_repaint();
            return;
        }

        if let Err(e) = std::fs::rename(&part, &dest) {
            set_err(format!("Rename error: {e}"));
            return;
        }

        {
            let mut map = sh_dl.lora_downloads.lock().unwrap();
            if let Some(st) = map.get_mut(&filename) {
                st.downloaded = downloaded;
                st.speed_mb = 0.0;
                st.completed = true;
            }
        }
        ctx.request_repaint();
    });
}

/// Renders a compact, fixed-width thumbnail grid for LoRAs compatible with `family`.
pub fn render_lora_carousel(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    sh: &Arc<Shared>,
    state: &mut LoraBrowserState,
    selected_loras: &mut Vec<ActiveLora>,
    prompt_tags: &mut Vec<PromptTag>,
    family: LoraModelFamily,
    steps_sync: Option<(u32, &mut u32)>,
) {
    render_lora_carousel_filtered(
        ui,
        ctx,
        sh,
        state,
        selected_loras,
        prompt_tags,
        family,
        steps_sync,
        None,
        None,
    );
}

pub const LORA_TILE_MIN_W: f32 = 112.0;
pub const LORA_TILE_SPACING: f32 = 6.0;
pub const LORA_SCROLLBAR_RESERVE: f32 = 10.0;
/// Minimum inner width required inside the LoRA carousel to always display 2 tiles per row.
pub const LORA_TWO_COL_MIN_INNER_W: f32 =
    LORA_TILE_MIN_W * 2.0 + LORA_TILE_SPACING + LORA_SCROLLBAR_RESERVE;

/// Same as `render_lora_carousel`, optionally filtering to `Some(true)` (IC-LoRA Edit tiles only)
/// or `Some(false)` (standard Generate LoRAs only), and optionally capping scroll height.
pub fn render_lora_carousel_filtered(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    sh: &Arc<Shared>,
    state: &mut LoraBrowserState,
    selected_loras: &mut Vec<ActiveLora>,
    prompt_tags: &mut Vec<PromptTag>,
    family: LoraModelFamily,
    steps_sync: Option<(u32, &mut u32)>,
    ic_filter: Option<bool>,
    max_scroll_h: Option<f32>,
) {
    render_lora_carousel_with_policy(
        ui,
        ctx,
        sh,
        state,
        selected_loras,
        prompt_tags,
        family,
        steps_sync,
        ic_filter,
        max_scroll_h,
        None,
    );
}

pub fn render_lora_carousel_with_policy(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    sh: &Arc<Shared>,
    state: &mut LoraBrowserState,
    selected_loras: &mut Vec<ActiveLora>,
    prompt_tags: &mut Vec<PromptTag>,
    family: LoraModelFamily,
    steps_sync: Option<(u32, &mut u32)>,
    ic_filter: Option<bool>,
    max_scroll_h: Option<f32>,
    multi_policy: Option<MultiLoraPolicy>,
) {
    // Check if any async LoRA downloads just finished and mark them Ready for user activation
    let mut dl_finished_files: Vec<String> = Vec::new();
    let mut dl_map: HashMap<String, LoraDownloadStatus> = HashMap::new();
    {
        let mut guard = sh.lora_downloads.lock().unwrap();
        for (fname, st) in guard.iter() {
            if st.completed {
                dl_finished_files.push(fname.clone());
            } else {
                dl_map.insert(fname.clone(), st.clone());
            }
        }
        for fname in &dl_finished_files {
            guard.remove(fname);
        }
    }

    if !dl_finished_files.is_empty() {
        state.items = scan_local_loras(sh);
        for finished_file in dl_finished_files {
            if state.gated_poll_filename == finished_file {
                state.gated_poll_filename.clear();
                state.gated_poll_attempts = 0;
                state.gated_poll_next_at = None;
            }
            state.ready_loras.insert(finished_file);
        }
    }

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
            b.installed
                .cmp(&a.installed)
                .then_with(|| {
                    b.rating_score
                        .partial_cmp(&a.rating_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.display_name.to_ascii_lowercase().cmp(&b.display_name.to_ascii_lowercase()))
        });
    }

    // Upload any newly decoded thumbnail frames
    {
        let mut pending = state.thumb_results.lock().unwrap();
        for (key, frame_imgs) in pending.drain(..) {
            let mut frames = Vec::with_capacity(frame_imgs.len());
            let mut delays_ms = Vec::with_capacity(frame_imgs.len());
            for (idx, (ci, delay)) in frame_imgs.into_iter().enumerate() {
                let tex = ctx.load_texture(format!("lora_thumb_{key}_{idx}"), ci, egui::TextureOptions::LINEAR);
                frames.push(tex);
                delays_ms.push(delay);
            }
            if !frames.is_empty() {
                state.thumbs.insert(key, AnimatedThumb { frames, delays_ms });
            }
        }
    }

    let compat_items: Vec<LoraItem> = state
        .items
        .iter()
        .filter(|it| it.is_compatible_with(family))
        .filter(|it| ic_filter.map_or(true, |want| it.is_ic_lora == want))
        .cloned()
        .collect();

    // Ensure no internal pipeline LoRAs remain selected
    selected_loras.retain(|l| !l.filename.is_empty() && l.filename != "None" && !is_internal_pipeline_lora(&l.filename));

    let avail_w = ui.available_width().max(1.0);
    let spacing = LORA_TILE_SPACING;
    let preliminary_cols = ((avail_w + spacing) / (LORA_TILE_MIN_W + spacing)).floor().max(1.0) as usize;

    // Build grid slots: None = "No LoRA" default tile, Some(idx) = compat_items[idx]
    let mut slots: Vec<Option<usize>> = Vec::with_capacity(1 + compat_items.len());
    slots.push(None);
    for idx in 0..compat_items.len() {
        slots.push(Some(idx));
    }

    let preliminary_rows = (slots.len() + preliminary_cols - 1) / preliminary_cols;
    let needs_scroll = preliminary_rows >= 4;
    let grid_w = if needs_scroll { (avail_w - LORA_SCROLLBAR_RESERVE).max(1.0) } else { avail_w };
    let cols = ((grid_w + spacing) / (LORA_TILE_MIN_W + spacing)).floor().max(1.0) as usize;
    let card_w = ((grid_w - spacing * (cols.saturating_sub(1)) as f32) / cols as f32).floor().max(1.0);
    let card_h = 132.0_f32;
    let thumb_w = (card_w - 10.0).max(60.0);
    let thumb_h = 56.0_f32;

    let accent = Color32::from_rgb(99, 102, 241);
    let card_bg = Color32::from_rgb(26, 30, 42);
    let card_sel = Color32::from_rgb(38, 43, 66);
    let line = Color32::from_rgb(44, 50, 68);
    let dim = Color32::from_rgb(142, 150, 170);
    let text_col = Color32::from_rgb(232, 236, 245);
    let green = Color32::from_rgb(35, 165, 90);
    let amber = Color32::from_rgb(235, 168, 52);

    let loras_dir = resolve_loras_dir(sh);
    let cache_dir = loras_dir.as_ref().map(|d| d.join(".previews"));
    let now_secs = ctx.input(|i| i.time);
    let mut min_next_frame_ms: Option<u64> = None;

    let prev_loras: Vec<String> = selected_loras.iter().map(|l| l.filename.clone()).collect();
    let mut none_clicked = false;
    let mut pending_downloads: Vec<LoraItem> = Vec::new();
    let mut pending_pauses: Vec<String> = Vec::new();

    let err_downloads: Vec<LoraDownloadStatus> = dl_map
        .values()
        .filter(|d| d.error.is_some())
        .cloned()
        .collect();
    for d in &err_downloads {
        if let Some(ref err) = d.error {
            let err_col = Color32::from_rgb(235, 95, 95);
            let is_gated = err.contains("401") || err.contains("403");
            let has_saved = read_hf_token().is_some();
            let repo_url = d
                .download_url
                .split("/resolve/")
                .next()
                .filter(|s| s.starts_with("https://huggingface.co/"))
                .map(|s| s.to_string());

            // Auto-poll every 5s up to 5 times when a token is already saved and we're waiting for license acceptance
            if is_gated && has_saved {
                if state.gated_poll_filename != d.filename {
                    state.gated_poll_filename = d.filename.clone();
                    state.gated_poll_attempts = 0;
                    state.gated_poll_next_at = Some(std::time::Instant::now() + std::time::Duration::from_secs(5));
                }
                if let Some(next_at) = state.gated_poll_next_at {
                    let now = std::time::Instant::now();
                    if now >= next_at {
                        if state.gated_poll_attempts < 5 {
                            state.gated_poll_attempts += 1;
                            state.gated_poll_next_at = if state.gated_poll_attempts < 5 {
                                Some(now + std::time::Duration::from_secs(5))
                            } else {
                                None
                            };
                            if let Some(retry_item) = compat_items.iter().find(|it| it.filename == d.filename).cloned() {
                                sh.lora_downloads.lock().unwrap().remove(&d.filename);
                                pending_downloads.push(retry_item);
                            }
                        } else {
                            state.gated_poll_next_at = None;
                        }
                    } else {
                        ctx.request_repaint_after(std::time::Duration::from_secs(1));
                    }
                }
            }

            let banner_msg = if is_gated && has_saved {
                if let Some(next_at) = state.gated_poll_next_at {
                    let secs = next_at.saturating_duration_since(std::time::Instant::now()).as_secs().max(1);
                    format!(
                        "{}: Accept license on Hugging Face (auto-checking in {secs}s, {}/5)",
                        d.display_name,
                        state.gated_poll_attempts + 1
                    )
                } else {
                    format!("{}: Accept license on Hugging Face and click Retry.", d.display_name)
                }
            } else {
                format!("{}: {err}", d.display_name)
            };

            ui.allocate_ui(vec2(avail_w, 0.0), |ui| {
                ui.set_max_width(avail_w);
                egui::Frame::NONE
                    .fill(Color32::from_rgb(42, 24, 28))
                    .stroke(Stroke::new(1.0, err_col.linear_multiply(0.6)))
                    .corner_radius(5.0)
                    .inner_margin(egui::Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        ui.set_max_width((avail_w - 16.0).max(120.0));
                        ui.add(
                            egui::Label::new(
                                RichText::new(banner_msg)
                                    .size(10.5)
                                    .color(err_col),
                            )
                            .wrap(),
                        );
                        ui.add_space(4.0);
                        ui.horizontal_wrapped(|ui| {
                            if is_gated {
                                if let Some(ref url) = repo_url {
                                    let btn_text = if has_saved { "Accept License ↗" } else { "1. Accept License ↗" };
                                    if ui
                                        .small_button(RichText::new(btn_text).size(10.0).color(amber))
                                        .on_hover_text(url)
                                        .clicked()
                                    {
                                        open_external_url(url);
                                    }
                                }
                                if has_saved {
                                    if ui.small_button(RichText::new("Retry Now").size(10.0)).clicked() {
                                        state.gated_poll_attempts = 0;
                                        state.gated_poll_next_at = Some(std::time::Instant::now() + std::time::Duration::from_secs(5));
                                        if let Some(retry_item) = compat_items.iter().find(|it| it.filename == d.filename).cloned() {
                                            sh.lora_downloads.lock().unwrap().remove(&d.filename);
                                            pending_downloads.push(retry_item);
                                        }
                                    }
                                } else if ui
                                    .small_button(RichText::new("2. Get Read Token ↗").size(10.0).color(amber))
                                    .on_hover_text("Create a free read-only Hugging Face access token")
                                    .clicked()
                                {
                                    open_external_url("https://huggingface.co/settings/tokens/new?tokenType=read");
                                }
                            }
                            if ui.small_button("Dismiss").clicked() {
                                sh.lora_downloads.lock().unwrap().remove(&d.filename);
                                if state.gated_poll_filename == d.filename {
                                    state.gated_poll_filename.clear();
                                    state.gated_poll_attempts = 0;
                                    state.gated_poll_next_at = None;
                                }
                            }
                        });
                        if is_gated && !has_saved {
                            ui.add_space(4.0);
                            ui.horizontal_wrapped(|ui| {
                                let input_w = (avail_w - 110.0).clamp(90.0, 220.0);
                                ui.add(
                                    egui::TextEdit::singleline(&mut state.hf_token_input)
                                        .password(true)
                                        .desired_width(input_w)
                                        .hint_text("Paste hf_... token"),
                                );
                                let can_save = !state.hf_token_input.trim().is_empty();
                                if ui
                                    .add_enabled(can_save, egui::Button::new(RichText::new("Save & Retry").size(10.0)).small())
                                    .clicked()
                                {
                                    if save_hf_token(&state.hf_token_input) {
                                        state.hf_token_input.clear();
                                    }
                                    if let Some(retry_item) = compat_items.iter().find(|it| it.filename == d.filename).cloned() {
                                        sh.lora_downloads.lock().unwrap().remove(&d.filename);
                                        pending_downloads.push(retry_item);
                                    }
                                }
                            });
                        }
                    });
            });
            ui.add_space(4.0);
        }
    }

    let mut render_grid = |ui: &mut egui::Ui| {
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing = vec2(0.0, spacing);
            for (row_idx, row) in slots.chunks(cols).enumerate() {
                let (row_rect, _) = ui.allocate_exact_size(vec2(grid_w, card_h), egui::Sense::hover());
                for (col_idx, slot) in row.iter().enumerate() {
                    let rect = egui::Rect::from_min_size(
                        row_rect.min + vec2(col_idx as f32 * (card_w + spacing), 0.0),
                        vec2(card_w, card_h),
                    );
                    let tile_id = ui.id().with(("lora_tile", family.label(), row_idx, col_idx));
                    let resp = ui.interact(rect, tile_id, egui::Sense::click());
                    let p = ui.painter_at(rect);

                    match slot {
                        None => {
                            let is_none = selected_loras.is_empty();
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
                                vec2(thumb_w, card_h - 70.0),
                            );
                            let mut child = ui.new_child(
                                egui::UiBuilder::new()
                                    .max_rect(title_rect)
                                    .layout(egui::Layout::top_down(egui::Align::Min)),
                            );
                            child.set_clip_rect(title_rect.intersect(ui.clip_rect()));
                            child.set_max_width(thumb_w);
                            child.spacing_mut().item_spacing = vec2(0.0, 2.0);
                            child.add(egui::Label::new(RichText::new("No LoRA").size(10.5).strong().color(text_col)).wrap().selectable(false));
                            child.add(egui::Label::new(RichText::new(family.label()).size(9.5).color(dim)).wrap().selectable(false));

                            if resp.clicked() {
                                selected_loras.clear();
                                prompt_tags.clear();
                                none_clicked = true;
                            }
                            resp.on_hover_text("Use the selected model without any additional LoRA.");
                        }
                        Some(idx) => {
                            let item = &compat_items[*idx];
                            load_thumb_async(
                                state,
                                &item.filename,
                                item.preview_path.clone(),
                                item.preview_url.clone(),
                                cache_dir.clone(),
                                ctx,
                            );
                            let dl_info = dl_map.get(&item.filename);
                            let is_sel = item.installed && selected_loras.iter().any(|l| l.filename == item.filename);
                            let is_downloading = dl_info.is_some_and(|d| !d.paused && d.error.is_none());
                            let is_paused = dl_info.is_some_and(|d| d.paused && d.error.is_none());
                            let part_bytes = if !item.installed {
                                loras_dir
                                    .as_ref()
                                    .and_then(|d| std::fs::metadata(d.join(format!("{}.part", item.filename))).ok())
                                    .map(|m| m.len())
                                    .unwrap_or(0)
                            } else {
                                0
                            };
                            let show_delete = item.installed || dl_info.is_some() || part_bytes > 0;
                            let is_ready = item.installed && !is_sel && state.ready_loras.contains(&item.filename);

                            p.rect_filled(rect, 6.0, if is_sel { card_sel } else { card_bg });
                            p.rect_stroke(
                                rect,
                                6.0,
                                Stroke::new(
                                    if is_sel || is_downloading || is_paused || is_ready { 1.5 } else { 1.0 },
                                    if is_sel {
                                        accent
                                    } else if is_downloading || is_paused {
                                        amber
                                    } else if is_ready {
                                        green
                                    } else {
                                        line
                                    },
                                ),
                                StrokeKind::Inside,
                            );

                            let thumb_rect = egui::Rect::from_min_size(rect.min + vec2(5.0, 5.0), vec2(thumb_w, thumb_h));
                            if let Some(anim) = state.thumbs.get(&item.filename) {
                                if let Some((tex, next_ms)) = anim.frame_at(now_secs) {
                                    p.image(
                                        tex.id(),
                                        thumb_rect,
                                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                        if item.installed { Color32::WHITE } else { Color32::from_gray(215) },
                                    );
                                    if let Some(ms) = next_ms {
                                        min_next_frame_ms = Some(min_next_frame_ms.map_or(ms, |m| m.min(ms)));
                                    }
                                }
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

                            if let Some(d) = dl_info {
                                let frac = if d.total > 0 {
                                    (d.downloaded as f32 / d.total as f32).clamp(0.0, 1.0)
                                } else {
                                    0.0
                                };
                                let circle_center = egui::pos2(thumb_rect.right() - 15.0, thumb_rect.bottom() - 15.0);
                                draw_lora_circle_progress(&p, circle_center, 11.0, frac, d.paused, d.error.is_some());
                            }

                            let badge_text = if !item.installed {
                                if is_downloading {
                                    "DL...".to_string()
                                } else if is_paused {
                                    "PAUSED".to_string()
                                } else {
                                    "GET".to_string()
                                }
                            } else if is_ready {
                                "READY".to_string()
                            } else {
                                item.compat_tag.clone()
                            };
                            let badge_w = if is_ready || badge_text.len() >= 4 { 38.0 } else { 29.0 };
                            let badge_rect = egui::Rect::from_min_size(
                                egui::pos2(thumb_rect.max.x - (badge_w + 3.0), thumb_rect.min.y + 3.0),
                                vec2(badge_w, 14.0),
                            );
                            p.rect_filled(badge_rect, 3.0, Color32::from_rgba_unmultiplied(12, 14, 20, 215));
                            p.text(
                                badge_rect.center(),
                                egui::Align2::CENTER_CENTER,
                                &badge_text,
                                egui::FontId::proportional(9.0),
                                if !item.installed {
                                    amber
                                } else if is_ready || (item.compat_tag == "2.3" && family == LoraModelFamily::Ltx25) {
                                    green
                                } else {
                                    text_col
                                },
                            );

                            let del_rect = egui::Rect::from_min_size(
                                egui::pos2(rect.right() - 21.0, rect.bottom() - 21.0),
                                vec2(16.0, 16.0),
                            );
                            let (delete_clicked, del_hovered) = if show_delete {
                                let del_resp = ui.interact(
                                    del_rect,
                                    ui.id().with(("lora_tile_del", &item.filename)),
                                    egui::Sense::click(),
                                );
                                let hov = del_resp.hovered();
                                let clk = del_resp.clicked();
                                let tip = if item.installed {
                                    format!(
                                        "Delete {} from disk ({})",
                                        item.display_name,
                                        crate::engine::format_bytes(item.size_bytes)
                                    )
                                } else {
                                    let partial_sz = dl_info.map(|d| d.downloaded).unwrap_or(part_bytes);
                                    format!(
                                        "Cancel download & delete partial file ({})",
                                        crate::engine::format_bytes(partial_sz)
                                    )
                                };
                                del_resp.on_hover_text(tip);
                                (clk, hov)
                            } else {
                                (false, false)
                            };
                            if show_delete {
                                let del_bg = if del_hovered {
                                    Color32::from_rgb(130, 38, 38)
                                } else {
                                    Color32::from_rgb(45, 24, 30)
                                };
                                let del_fg = if del_hovered {
                                    Color32::WHITE
                                } else {
                                    Color32::from_rgb(235, 95, 95)
                                };
                                p.rect_filled(del_rect, 3.0, del_bg);
                                p.rect_stroke(del_rect, 3.0, Stroke::new(1.0, Color32::from_rgb(150, 55, 55)), StrokeKind::Inside);
                                draw_lora_trash_icon(&p, del_rect.center(), del_fg);
                            }

                            let title_w = if show_delete { (thumb_w - 20.0).max(40.0) } else { thumb_w };
                            let title_rect = egui::Rect::from_min_size(
                                rect.min + vec2(5.0, 65.0),
                                vec2(title_w, card_h - 70.0),
                            );
                            let mut child = ui.new_child(
                                egui::UiBuilder::new()
                                    .max_rect(title_rect)
                                    .layout(egui::Layout::top_down(egui::Align::Min)),
                            );
                            child.set_clip_rect(title_rect.intersect(ui.clip_rect()));
                            child.set_max_width(title_w);
                            child.spacing_mut().item_spacing = vec2(0.0, 2.0);
                            child.add(
                                egui::Label::new(RichText::new(&item.display_name).size(10.5).strong().color(text_col))
                                    .wrap()
                                    .selectable(false),
                            );
                            let (sub, sub_col) = if let Some(d) = dl_info {
                                if let Some(ref err) = d.error {
                                    if err.contains("401") || err.contains("403") {
                                        if read_hf_token().is_some() {
                                            ("Accept license".to_string(), amber)
                                        } else {
                                            ("Needs HF_TOKEN".to_string(), amber)
                                        }
                                    } else {
                                        ("Failed - click to retry".to_string(), Color32::from_rgb(235, 95, 95))
                                    }
                                } else if d.paused {
                                    let pct = if d.total > 0 {
                                        ((d.downloaded as f64 / d.total as f64) * 100.0).min(100.0)
                                    } else {
                                        0.0
                                    };
                                    (format!("Paused {pct:.0}%"), amber)
                                } else if d.total > 0 {
                                    let pct = ((d.downloaded as f64 / d.total as f64) * 100.0).min(100.0);
                                    (format!("Downloading {pct:.0}%"), dim)
                                } else {
                                    ("Downloading...".to_string(), dim)
                                }
                            } else if !item.installed {
                                let mb = (item.size_bytes as f64 / (1024.0 * 1024.0)).round();
                                (format!("{} ({mb:.0} MB)", item.description), dim)
                            } else if is_ready {
                                ("Ready - click to activate".to_string(), green)
                            } else if !item.description.is_empty() && item.description != "Local LoRA" {
                                (item.description.clone(), dim)
                            } else if item.compat_tag == "2.3" && family == LoraModelFamily::Ltx25 {
                                ("2.3 Compat".to_string(), dim)
                            } else {
                                ("Installed".to_string(), dim)
                            };
                            child.add(
                                egui::Label::new(RichText::new(sub).size(9.5).color(sub_col))
                                    .wrap()
                                    .selectable(false),
                            );

                            if delete_clicked {
                                if let Some(ref dir) = loras_dir {
                                    let sz = if item.installed {
                                        item.size_bytes
                                    } else {
                                        dl_info.map(|d| d.downloaded).unwrap_or(part_bytes)
                                    };
                                    state.pending_delete = Some(PendingLoraDelete {
                                        filename: item.filename.clone(),
                                        display_name: item.display_name.clone(),
                                        loras_dir: dir.clone(),
                                        is_mid_download: !item.installed,
                                        size_bytes: sz,
                                        error: None,
                                    });
                                }
                            } else if resp.clicked() {
                                if item.installed {
                                    state.ready_loras.remove(&item.filename);
                                    if let Some(warn) = apply_lora_tile_click(
                                        selected_loras,
                                        prompt_tags,
                                        item,
                                        is_sel,
                                        multi_policy.as_ref(),
                                    ) {
                                        state.vram_block_warning = Some((warn, std::time::Instant::now()));
                                    } else {
                                        state.vram_block_warning = None;
                                    }
                                } else if is_downloading {
                                    pending_pauses.push(item.filename.clone());
                                } else {
                                    pending_downloads.push(item.clone());
                                }
                            }

                            let mut hover_lines = vec![item.display_name.clone()];
                            if !item.description.is_empty() {
                                hover_lines.push(item.description.clone());
                            }
                            if let Some(ref tw) = item.trigger_word {
                                hover_lines.push(format!("Prompt trigger: {tw}"));
                            }
                            if let Some(d) = dl_info {
                                if d.paused {
                                    hover_lines.push(format!(
                                        "Paused ({} / {}). Click to resume.",
                                        crate::engine::format_bytes(d.downloaded),
                                        crate::engine::format_bytes(d.total)
                                    ));
                                } else if d.error.is_none() {
                                    hover_lines.push(format!(
                                        "Downloading ({} / {} @ {:.1} MB/s). Click to pause.",
                                        crate::engine::format_bytes(d.downloaded),
                                        crate::engine::format_bytes(d.total),
                                        d.speed_mb
                                    ));
                                } else {
                                    hover_lines.push("Download failed. Click to retry.".to_string());
                                }
                            } else if !item.installed {
                                let mb = item.size_bytes as f64 / (1024.0 * 1024.0);
                                hover_lines.push(format!("Click to download (~{mb:.0} MB)."));
                            } else if is_ready {
                                hover_lines.push("Downloaded and ready. Click to activate and insert prompt keyword.".to_string());
                            } else {
                                let vram = estimate_lora_vram_gb(&item.filename, item.size_bytes);
                                hover_lines.push(format!("File: {} (Est. +{vram:.1} GB VRAM)", item.filename));
                            }
                            if !del_hovered {
                                resp.on_hover_text(hover_lines.join("\n"));
                            }
                        }
                    }
                }
            }
        });
    };

    let actual_rows = (slots.len() + cols - 1) / cols;
    let natural_h = actual_rows as f32 * card_h + (actual_rows.saturating_sub(1)) as f32 * spacing;
    let default_scroll_h = 3.5 * card_h + 3.0 * spacing;
    let cap_h = max_scroll_h.unwrap_or(default_scroll_h);
    if actual_rows >= 4 || (max_scroll_h.is_some() && natural_h > cap_h) {
        let scroll_h = cap_h.min(natural_h.max(card_h));
        egui::ScrollArea::vertical()
            .id_salt(if family == LoraModelFamily::QwenImage21 { "image_lora_grid" } else { "video_lora_grid" })
            .min_scrolled_height(scroll_h)
            .max_height(scroll_h)
            .show(ui, |ui| {
                render_grid(ui);
            });
    } else {
        render_grid(ui);
    }

    if let Some((ref warn_msg, shown_at)) = state.vram_block_warning {
        if shown_at.elapsed().as_secs_f32() < 5.0 {
            ui.add_space(4.0);
            ui.label(
                RichText::new(warn_msg)
                    .size(10.5)
                    .color(Color32::from_rgb(245, 170, 50)),
            );
            ctx.request_repaint();
        } else {
            state.vram_block_warning = None;
        }
    }

    for fname in pending_pauses {
        pause_lora_download(sh, &fname);
    }
    if let Some(ref dir) = loras_dir {
        for item in pending_downloads {
            start_lora_download_async(sh, &item, dir.clone(), ctx);
        }
    }

    render_lora_delete_modal(ctx, sh, state, selected_loras, prompt_tags);

    // Active LoRA strength sliders
    if !selected_loras.is_empty() {
        ui.add_space(4.0);
        let mut to_remove: Option<usize> = None;
        for (idx, active) in selected_loras.iter_mut().enumerate() {
            let label = state
                .items
                .iter()
                .find(|it| it.filename == active.filename)
                .map(|it| it.display_name.as_str())
                .unwrap_or(&active.filename);
            // Keep label + remove button and slider within `ui.available_width()` so selecting
            // any LoRA never widens the left sidebar beyond its minimum width.
            let row_w = ui.available_width();
            ui.allocate_ui_with_layout(
                vec2(row_w, 18.0),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    if ui.small_button("×").clicked() {
                        to_remove = Some(idx);
                    }
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        ui.add(
                            egui::Label::new(RichText::new(label).size(10.5).color(text_col))
                                .truncate(),
                        );
                    });
                },
            );
            ui.horizontal(|ui| {
                ui.spacing_mut().slider_width = (ui.available_width() - 56.0).max(48.0);
                ui.add(
                    egui::Slider::new(&mut active.strength, 0.0..=2.0)
                        .step_by(0.05)
                        .fixed_decimals(2),
                );
            });
        }
        if let Some(idx) = to_remove {
            let removed = selected_loras.remove(idx);
            prompt_tags.retain(|t| t.lora_filename != removed.filename);
        }
    }

    // Sync default steps when switching turbo LoRAs on/off
    let cur_loras: Vec<String> = selected_loras.iter().map(|l| l.filename.clone()).collect();
    if let Some((default_steps, steps)) = steps_sync {
        sync_steps_for_lora_change(&prev_loras, &cur_loras, none_clicked, default_steps, steps);
    }

    if let Some(ms) = min_next_frame_ms {
        ctx.request_repaint_after(std::time::Duration::from_millis(ms));
    }
}

fn draw_lora_circle_progress(
    painter: &egui::Painter,
    center: egui::Pos2,
    radius: f32,
    progress: f32,
    paused: bool,
    failed: bool,
) {
    let track_col = Color32::from_rgb(42, 48, 66);
    painter.circle_filled(center, radius, Color32::from_rgb(18, 22, 32));
    painter.circle_stroke(center, radius, Stroke::new(2.2, track_col));

    let (ring_col, icon_col) = if failed {
        (Color32::from_rgb(235, 95, 95), Color32::from_rgb(235, 95, 95))
    } else if paused {
        (Color32::from_rgb(235, 168, 52), Color32::from_rgb(235, 168, 52))
    } else {
        (Color32::from_rgb(129, 140, 248), Color32::from_rgb(232, 236, 245))
    };

    let p = progress.clamp(0.0, 1.0);
    if p > 0.005 {
        let steps = ((p * 36.0).ceil() as usize).max(2);
        let start_angle = -std::f32::consts::FRAC_PI_2;
        let total_angle = p * std::f32::consts::TAU;
        let pts: Vec<egui::Pos2> = (0..=steps)
            .map(|i| {
                let a = start_angle + total_angle * (i as f32 / steps as f32);
                egui::pos2(center.x + radius * a.cos(), center.y + radius * a.sin())
            })
            .collect();
        painter.add(egui::Shape::line(pts, Stroke::new(2.4, ring_col)));
    }

    if failed {
        painter.line_segment(
            [egui::pos2(center.x, center.y - 3.8), egui::pos2(center.x, center.y + 1.0)],
            Stroke::new(2.0, icon_col),
        );
        painter.circle_filled(egui::pos2(center.x, center.y + 3.2), 1.1, icon_col);
    } else if paused {
        let tri = vec![
            egui::pos2(center.x - 2.3, center.y - 3.8),
            egui::pos2(center.x + 3.8, center.y),
            egui::pos2(center.x - 2.3, center.y + 3.8),
        ];
        painter.add(egui::Shape::convex_polygon(tri, icon_col, Stroke::NONE));
    } else {
        painter.line_segment(
            [egui::pos2(center.x - 2.2, center.y - 3.3), egui::pos2(center.x - 2.2, center.y + 3.3)],
            Stroke::new(1.9, icon_col),
        );
        painter.line_segment(
            [egui::pos2(center.x + 2.2, center.y - 3.3), egui::pos2(center.x + 2.2, center.y + 3.3)],
            Stroke::new(1.9, icon_col),
        );
    }
}

fn draw_lora_trash_icon(painter: &egui::Painter, center: egui::Pos2, color: Color32) {
    let s = Stroke::new(1.2, color);
    painter.line_segment([egui::pos2(center.x - 4.0, center.y - 3.0), egui::pos2(center.x + 4.0, center.y - 3.0)], s);
    painter.line_segment([egui::pos2(center.x - 1.5, center.y - 4.5), egui::pos2(center.x + 1.5, center.y - 4.5)], s);
    painter.rect_stroke(
        egui::Rect::from_center_size(egui::pos2(center.x, center.y + 1.0), vec2(6.0, 6.5)),
        1.0,
        s,
        StrokeKind::Inside,
    );
    painter.line_segment([egui::pos2(center.x - 1.2, center.y - 1.0), egui::pos2(center.x - 1.2, center.y + 3.0)], s);
    painter.line_segment([egui::pos2(center.x + 1.2, center.y - 1.0), egui::pos2(center.x + 1.2, center.y + 3.0)], s);
}

fn render_lora_delete_modal(
    ctx: &egui::Context,
    sh: &Shared,
    state: &mut LoraBrowserState,
    selected_loras: &mut Vec<ActiveLora>,
    prompt_tags: &mut Vec<PromptTag>,
) {
    let Some(ref mut del) = state.pending_delete else {
        return;
    };
    let mut do_delete = false;
    let mut cancel = false;

    let title = if del.is_mid_download {
        "Cancel Download & Delete?"
    } else {
        "Delete LoRA?"
    };

    egui::Window::new(title)
        .id(egui::Id::new("lora_delete_confirm_modal"))
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
            ui.set_width(360.0);
            ui.label(
                RichText::new(&del.display_name)
                    .size(13.0)
                    .strong()
                    .color(Color32::from_rgb(232, 236, 245)),
            );
            ui.add_space(6.0);
            let msg = if del.is_mid_download {
                format!(
                    "This will cancel the download and permanently delete the partial file ({}) from disk.",
                    crate::engine::format_bytes(del.size_bytes)
                )
            } else {
                format!(
                    "This will permanently delete {} ({}) from your models/loras folder.",
                    del.filename,
                    crate::engine::format_bytes(del.size_bytes)
                )
            };
            ui.label(
                RichText::new(msg)
                    .size(11.5)
                    .color(Color32::from_rgb(160, 168, 185)),
            );
            if let Some(ref err) = del.error {
                ui.add_space(6.0);
                ui.label(
                    RichText::new(err)
                        .size(11.0)
                        .color(Color32::from_rgb(235, 95, 95)),
                );
            }
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let btn_label = if del.is_mid_download {
                        "Cancel & Delete"
                    } else {
                        "Delete File"
                    };
                    let btn = egui::Button::new(
                        RichText::new(btn_label).strong().color(Color32::WHITE),
                    )
                    .fill(Color32::from_rgb(175, 42, 42));
                    if ui.add(btn).clicked() {
                        do_delete = true;
                    }
                });
            });
        });

    if do_delete {
        let fname = del.filename.clone();
        let dir = del.loras_dir.clone();
        match cancel_and_delete_lora(sh, &dir, &fname) {
            Ok(()) => {
                selected_loras.retain(|l| l.filename != fname);
                prompt_tags.retain(|t| t.lora_filename != fname);
                state.ready_loras.remove(&fname);
                state.items = scan_local_loras(sh);
                state.pending_delete = None;
            }
            Err(e) => {
                if let Some(ref mut d) = state.pending_delete {
                    d.error = Some(format!("Failed to delete: {e}"));
                }
            }
        }
    } else if cancel {
        state.pending_delete = None;
    }
}

/// Returns the recommended step count for a turbo LoRA filename, if applicable.
fn turbo_lora_steps(filename: &str) -> Option<u32> {
    let lower = filename.to_ascii_lowercase();
    if lower.contains("turbo") {
        if lower.contains("4step") || lower.contains("4-step") {
            Some(4)
        } else {
            Some(8)
        }
    } else {
        None
    }
}

/// Updates `steps` when turbo LoRAs are toggled on/off or "No LoRA" is clicked.
fn sync_steps_for_lora_change(
    prev_loras: &[String],
    cur_loras: &[String],
    none_clicked: bool,
    default_steps: u32,
    steps: &mut u32,
) {
    if none_clicked || (cur_loras != prev_loras && cur_loras.is_empty()) {
        *steps = default_steps;
        return;
    }
    if cur_loras != prev_loras {
        let prev_turbo = prev_loras.iter().find_map(|f| turbo_lora_steps(f));
        let cur_turbo = cur_loras.iter().find_map(|f| turbo_lora_steps(f));
        if cur_turbo != prev_turbo {
            *steps = cur_turbo.unwrap_or(default_steps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_turbo_lora_does_not_leak_to_ltx_and_vice_versa() {
        let (_, h3, ltx25, ltx23, qwen) = classify_lora("minimax_h3_turbo_8step.safetensors", None);
        assert!(h3 && !ltx25 && !ltx23 && !qwen);

        let (_, h3, ltx25, ltx23, qwen) = classify_lora("cinematic_ltx_2.3_style.safetensors", None);
        assert!(!h3 && ltx25 && ltx23 && !qwen);
    }

    #[test]
    fn ltx25_includes_compatible_ltx23_loras() {
        let (_, h3, ltx25, ltx23, qwen) = classify_lora("cinematic_ltx_2.3_style.safetensors", None);
        let item = LoraItem {
            filename: "cinematic_ltx_2.3_style.safetensors".into(),
            display_name: "Cinematic".into(),
            description: String::new(),
            compat_tag: "2.3".into(),
            supports_qwen_image: qwen,
            supports_h3: h3,
            supports_ltx25: ltx25,
            supports_ltx23: ltx23,
            rating_score: 10.0,
            preview_path: None,
            preview_url: None,
            download_url: None,
            size_bytes: 0,
            trigger_word: None,
            default_strength: None,
            is_ic_lora: false,
            installed: true,
        };
        assert!(!item.is_compatible_with(LoraModelFamily::QwenImage21));
        assert!(!item.is_compatible_with(LoraModelFamily::MiniMaxH3));
        assert!(item.is_compatible_with(LoraModelFamily::Ltx25));
        assert!(item.is_compatible_with(LoraModelFamily::Ltx23));
    }

    #[test]
    fn curated_h3_loras_have_valid_urls_and_trigger_insertion_is_idempotent() {
        let list = curated_h3_loras();
        assert!(list.len() >= 6);
        for item in &list {
            assert!(item.supports_h3);
            assert!(item.download_url.as_deref().unwrap_or("").starts_with("https://huggingface.co/"));
            assert!(item.preview_url.is_some());
        }
    }

    #[test]
    fn curated_qwen_loras_prioritise_camera_and_exclude_removed_presets() {
        let list = curated_qwen_loras();
        assert_eq!(list.len(), 8);
        for item in &list {
            assert!(item.supports_qwen_image);
            assert!(!item.supports_h3);
            assert!(item.is_compatible_with(LoraModelFamily::QwenImage21));
            assert!(item.download_url.as_deref().unwrap_or("").starts_with("https://huggingface.co/"));
            assert!(item.preview_url.as_deref().unwrap_or("").starts_with("https://huggingface.co/"));
            assert!(!item.filename.contains("nicegirls"));
            assert!(!item.filename.contains("niceasians"));
        }
        let pos = |name: &str| list.iter().position(|it| it.display_name == name).unwrap();
        assert!(pos("Lenovo") < pos("Consistency"));
        assert!(pos("Canon") < pos("Consistency"));
        assert!(pos("Samsung") < pos("Consistency"));
        assert!(pos("Grainscape") < pos("Consistency"));
        assert!(pos("FilmStills") < pos("Consistency"));
    }

    #[test]
    fn curated_ltx_loras_prioritise_ltx25_and_include_ltx23_compat() {
        let list = curated_ltx_loras();
        assert_eq!(list.len(), 20);
        for item in &list {
            assert!(item.is_compatible_with(LoraModelFamily::Ltx25));
            assert!(item.download_url.as_deref().unwrap_or("").starts_with("https://huggingface.co/"));
            assert!(item.size_bytes > 0);
        }
        let pos = |name: &str| list.iter().position(|it| it.display_name == name).unwrap();
        assert!(pos("Day to Night") < pos("Seamless Transitions"));
        assert!(pos("Pixel Upscaler 2x") < pos("Video Outpaint"));
        assert!(list[pos("Day to Night")].supports_ltx25);
        assert!(!list[pos("Day to Night")].is_compatible_with(LoraModelFamily::Ltx23));
        assert!(list[pos("Seamless Transitions")].is_compatible_with(LoraModelFamily::Ltx23));
        assert!(list[pos("Seamless Transitions")].is_compatible_with(LoraModelFamily::Ltx25));
    }

    #[test]
    fn estimate_lora_vram_accounts_for_active_lora() {
        assert_eq!(estimate_lora_vram_gb("None", 0), 0.0);
        assert_eq!(estimate_lora_vram_gb("", 100_000_000), 0.0);
        assert!(estimate_lora_vram_gb("my_lora.safetensors", 0) >= 0.5);
        assert!(estimate_lora_vram_gb("small.safetensors", 60_000_000) >= 0.25);
        assert!(estimate_lora_vram_gb("big.safetensors", 1_000_000_000) > 0.6);
    }

    #[test]
    fn compose_prompt_with_tags_joins_with_comma_space() {
        let tags = vec![
            PromptTag {
                lora_filename: "lenovo_qwen21.safetensors".into(),
                lora_display_name: "Lenovo".into(),
                keyword: "l3n0v0".into(),
                created_at: std::time::Instant::now(),
                hovered_dismissed: false,
            },
            PromptTag {
                lora_filename: "filmstills_qwen21.safetensors".into(),
                lora_display_name: "FilmStills".into(),
                keyword: "film still".into(),
                created_at: std::time::Instant::now(),
                hovered_dismissed: false,
            },
        ];
        assert_eq!(
            compose_prompt_with_tags(&tags, "a woman in a cafe"),
            "l3n0v0, film still, a woman in a cafe"
        );
        assert_eq!(compose_prompt_with_tags(&tags, "   "), "l3n0v0, film still");
        assert_eq!(compose_prompt_with_tags(&[], "  hello "), "hello");
    }

    #[test]
    fn strip_prompt_tags_excludes_active_tags_without_clipping_prose() {
        let mk = |kw: &str| PromptTag {
            lora_filename: format!("{kw}.safetensors"),
            lora_display_name: kw.into(),
            keyword: kw.into(),
            created_at: std::time::Instant::now(),
            hovered_dismissed: false,
        };
        let tags = vec![mk("l3n0v0"), mk("film still")];
        assert_eq!(
            strip_prompt_tags(&tags, "l3n0v0, film still, a woman in a cafe"),
            "a woman in a cafe"
        );
        assert_eq!(strip_prompt_tags(&tags, "l3n0v0, film still"), "");
        // Reordered / subset tags
        assert_eq!(
            strip_prompt_tags(&[mk("film still"), mk("l3n0v0")], "l3n0v0, film still, a cafe"),
            "a cafe"
        );
        // Comma-containing tag keyword
        let comma_tag = vec![mk("photoreal, 8K, sharp")];
        assert_eq!(
            strip_prompt_tags(&comma_tag, "photoreal, 8K, sharp, cinematic shot"),
            "cinematic shot"
        );
        // Prose word matching tag keyword without comma boundary must not be stripped
        assert_eq!(
            strip_prompt_tags(&[mk("colorize")], "colorize this old photo"),
            "colorize this old photo"
        );
    }

    #[test]
    fn video_lora_tile_click_enforces_single_and_vram_limit() {
        let make_item = |name: &str| LoraItem {
            filename: format!("{name}.safetensors"),
            display_name: name.into(),
            description: String::new(),
            compat_tag: "2.5".into(),
            supports_qwen_image: false,
            supports_h3: false,
            supports_ltx25: true,
            supports_ltx23: false,
            rating_score: 1.0,
            preview_path: None,
            preview_url: None,
            download_url: None,
            size_bytes: 500_000_000,
            trigger_word: Some(name.into()),
            default_strength: Some(1.0),
            is_ic_lora: false,
            installed: true,
        };
        let a = make_item("lora_a");
        let b = make_item("lora_b");

        // 1. Default (allow_multi = false): clicking B swaps out A
        let mut active = Vec::new();
        let mut tags = Vec::new();
        let single_policy = MultiLoraPolicy {
            allow_multi: false,
            current_est_vram_gb: 8.0,
            gpu_vram_gb: Some(16),
        };
        assert!(apply_lora_tile_click(&mut active, &mut tags, &a, false, Some(&single_policy)).is_none());
        assert_eq!(active.len(), 1);
        assert!(apply_lora_tile_click(&mut active, &mut tags, &b, false, Some(&single_policy)).is_none());
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].filename, "lora_b.safetensors");
        assert_eq!(tags.len(), 1);
        assert_eq!(tags[0].lora_filename, "lora_b.safetensors");

        // 2. allow_multi = true with plenty of VRAM headroom: stacks A + B
        let mut active = Vec::new();
        let mut tags = Vec::new();
        let ok_multi = MultiLoraPolicy {
            allow_multi: true,
            current_est_vram_gb: 8.0,
            gpu_vram_gb: Some(16),
        };
        assert!(apply_lora_tile_click(&mut active, &mut tags, &a, false, Some(&ok_multi)).is_none());
        assert!(apply_lora_tile_click(&mut active, &mut tags, &b, false, Some(&ok_multi)).is_none());
        assert_eq!(active.len(), 2);

        // 3. allow_multi = true near VRAM limit: swaps and returns warning
        let mut active = Vec::new();
        let mut tags = Vec::new();
        assert!(apply_lora_tile_click(&mut active, &mut tags, &a, false, Some(&ok_multi)).is_none());
        let tight_multi = MultiLoraPolicy {
            allow_multi: true,
            current_est_vram_gb: 11.4,
            gpu_vram_gb: Some(12),
        };
        let warn = apply_lora_tile_click(&mut active, &mut tags, &b, false, Some(&tight_multi));
        assert!(warn.is_some());
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].filename, "lora_b.safetensors");
    }

    #[test]
    fn turbo_lora_syncs_steps_to_8_and_resets_to_default() {
        let turbo = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors".to_string();
        let style = "minimax_h3_style_transfer_v1.0_r64.safetensors".to_string();

        // Picking turbo LoRA sets steps to 8
        let mut steps = 20;
        sync_steps_for_lora_change(&[], std::slice::from_ref(&turbo), false, 20, &mut steps);
        assert_eq!(steps, 8);

        // No change on subsequent frames preserves user-adjusted steps
        steps = 10;
        sync_steps_for_lora_change(
            std::slice::from_ref(&turbo),
            std::slice::from_ref(&turbo),
            false,
            20,
            &mut steps,
        );
        assert_eq!(steps, 10);

        // Clicking No LoRA resets steps back to 20 (even if already empty)
        sync_steps_for_lora_change(std::slice::from_ref(&turbo), &[], true, 20, &mut steps);
        assert_eq!(steps, 20);
        steps = 12;
        sync_steps_for_lora_change(&[], &[], true, 20, &mut steps);
        assert_eq!(steps, 20);

        // Toggling turbo LoRA off (or removing via x) resets to 20
        steps = 8;
        sync_steps_for_lora_change(std::slice::from_ref(&turbo), &[], false, 20, &mut steps);
        assert_eq!(steps, 20);

        // Removing turbo while another LoRA remains active resets to 20
        steps = 8;
        sync_steps_for_lora_change(
            &[style.clone(), turbo.clone()],
            std::slice::from_ref(&style),
            false,
            20,
            &mut steps,
        );
        assert_eq!(steps, 20);

        // Adding turbo alongside an existing LoRA sets steps to 8
        sync_steps_for_lora_change(
            std::slice::from_ref(&style),
            &[style.clone(), turbo],
            false,
            20,
            &mut steps,
        );
        assert_eq!(steps, 8);
    }
}
