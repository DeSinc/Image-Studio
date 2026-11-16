use std::collections::HashSet;
use std::path::{Path, PathBuf};

use eframe::egui::{self, pos2, vec2, Color32, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Vec2};
use crate::engine::{self, Profile};

const CARD_BG: Color32 = Color32::from_rgb(25, 29, 40);
const CARD_MUTED: Color32 = Color32::from_rgb(19, 22, 30);
const CARD_HOVER: Color32 = Color32::from_rgb(33, 38, 53);
const TRAY_BG: Color32 = Color32::from_rgb(15, 17, 24);
const LINE: Color32 = Color32::from_rgb(52, 60, 82);
const LINE_MUTED: Color32 = Color32::from_rgb(38, 43, 58);
const ACCENT_GLOW: Color32 = Color32::from_rgb(129, 140, 248);
const TEXT: Color32 = Color32::from_rgb(235, 238, 248);
const DIM: Color32 = Color32::from_rgb(168, 177, 202);
const MUTED_TEXT: Color32 = Color32::from_rgb(112, 120, 142);
const GREEN: Color32 = Color32::from_rgb(52, 211, 153);
const GREEN_FILL: Color32 = Color32::from_rgb(19, 44, 36);
const AMBER: Color32 = Color32::from_rgb(251, 191, 36);
const AMBER_FILL: Color32 = Color32::from_rgb(46, 36, 18);
const RED: Color32 = Color32::from_rgb(248, 113, 113);
const BLUE: Color32 = Color32::from_rgb(59, 130, 246);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImageFamily {
    Qwen21,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VideoFamily {
    MiniMaxH3,
    LtxVideo,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VariantIcon {
    Quality,
    Balanced,
    Compact,
    Turbo,
    Legacy,
}

#[derive(Clone)]
pub struct PendingModelWarning {
    pub is_video: bool,
    pub profile: Profile,
    pub title: String,
    pub message: String,
    pub is_severe: bool,
}

#[derive(Clone)]
pub struct PendingModelDelete {
    pub profile_name: String,
    pub files: Vec<(PathBuf, String, u64)>,
    pub error: Option<String>,
}

pub struct ModelPickerState {
    pub selected_image_family: ImageFamily,
    pub selected_video_family: VideoFamily,
    pub last_minimax_id: Option<String>,
    pub last_ltx_id: Option<String>,
    pub pending_warning: Option<PendingModelWarning>,
    pub pending_delete: Option<PendingModelDelete>,
}

impl ModelPickerState {
    pub fn new(active_video_profile: &Profile) -> Self {
        let (selected_video_family, last_minimax_id, last_ltx_id) = if active_video_profile.is_minimax_h3() {
            (VideoFamily::MiniMaxH3, Some(active_video_profile.id.clone()), None)
        } else {
            (VideoFamily::LtxVideo, None, Some(active_video_profile.id.clone()))
        };
        Self {
            selected_image_family: ImageFamily::Qwen21,
            selected_video_family,
            last_minimax_id,
            last_ltx_id,
            pending_warning: None,
            pending_delete: None,
        }
    }

    pub fn record_active_video(&mut self, p: &Profile) {
        if p.is_minimax_h3() {
            self.last_minimax_id = Some(p.id.clone());
        } else {
            self.last_ltx_id = Some(p.id.clone());
        }
    }
}

pub enum PickerAction {
    None,
    SwitchImage(Profile),
    SwitchVideo(Profile),
    ModelsDeleted,
}

pub fn recommended_image_profile_id(vram_gb: Option<u32>) -> &'static str {
    match vram_gb {
        Some(v) if v <= 8 => "qwen_2.1_8gb",
        _ => "qwen_2.1_int8",
    }
}

pub fn recommended_video_family(vram_gb: Option<u32>) -> VideoFamily {
    match vram_gb {
        Some(v) if v < 9 => VideoFamily::LtxVideo,
        _ => VideoFamily::MiniMaxH3,
    }
}

pub fn is_allowed_step_up_variant(id: &str, vram_gb: Option<u32>) -> bool {
    match vram_gb {
        // 12GB cards can pick Q5 variants without caveat popup (yellow border + SLOWER badge)
        Some(v) if (11..15).contains(&v) => matches!(
            id,
            "minimax_h3_q5_k_m" | "minimax_h3_q5_k_s" | "ltx_2.5_q5_k_xl" | "ltx_2.5_q5_k_m"
        ),
        // 10GB cards can pick Q4 variants without caveat popup (yellow border + SLOWER badge)
        Some(v) if (9..11).contains(&v) => matches!(
            id,
            "minimax_h3_12gb" | "minimax_h3_q4_k_s" | "ltx_2.5_12gb" | "ltx_2.5_q4_k_m"
        ),
        _ => false,
    }
}

pub fn recommended_video_profile_id(family: VideoFamily, vram_gb: Option<u32>) -> &'static str {
    match family {
        VideoFamily::MiniMaxH3 => match vram_gb {
            Some(v) if v >= 18 => "minimax_h3_24gb",
            Some(v) if v >= 15 => "minimax_h3_16gb",
            Some(v) if v >= 11 => "minimax_h3_12gb",
            Some(_) => "minimax_h3_10gb",
            None => "minimax_h3_12gb",
        },
        VideoFamily::LtxVideo => match vram_gb {
            Some(v) if v >= 18 => "ltx_2.5_24gb",
            Some(v) if v >= 15 => "ltx_2.5_16gb",
            Some(v) if v >= 11 => "ltx_2.5_12gb",
            Some(_) => "ltx_2.5_10gb",
            None => "ltx_2.5_12gb",
        },
    }
}

fn primary_model_exists(p: &Profile, models_dir: &Path) -> bool {
    if models_dir.as_os_str().is_empty() {
        return false;
    }
    if let Some(m) = p.models.iter().find(|m| {
        !m.optional && (m.category == "diffusion_models" || m.category == "checkpoints")
    }) {
        return engine::safe_model_path(models_dir, &m.category, &m.filename).is_file();
    }
    false
}

fn all_required_models_exist(p: &Profile, models_dir: &Path) -> bool {
    if models_dir.as_os_str().is_empty() {
        return false;
    }
    let required: Vec<_> = p.models.iter().filter(|m| !m.optional).collect();
    !required.is_empty()
        && required
            .iter()
            .all(|m| engine::safe_model_path(models_dir, &m.category, &m.filename).is_file())
}

fn find_downloaded_family_profile(
    family: VideoFamily,
    all_vid: &[Profile],
    last_id: Option<&str>,
    vram_detected: Option<u32>,
    models_dir: &Path,
) -> Option<Profile> {
    let in_family = |p: &&Profile| match family {
        VideoFamily::MiniMaxH3 => p.is_minimax_h3(),
        VideoFamily::LtxVideo => !p.is_minimax_h3(),
    };

    // 1. Prefer the last active model the user chose in this family if still downloaded
    if let Some(lid) = last_id {
        if let Some(p) = all_vid.iter().filter(in_family).find(|p| p.id == lid) {
            if all_required_models_exist(p, models_dir) {
                return Some(p.clone());
            }
        }
    }

    // 2. Otherwise prefer the recommended variant for this family if downloaded
    let rec_id = recommended_video_profile_id(family, vram_detected);
    if let Some(p) = all_vid.iter().filter(in_family).find(|p| p.id == rec_id) {
        if all_required_models_exist(p, models_dir) {
            return Some(p.clone());
        }
    }

    // 3. Otherwise any downloaded variant in this family
    all_vid
        .iter()
        .filter(in_family)
        .find(|p| all_required_models_exist(p, models_dir))
        .cloned()
}

/// Returns files belonging to `target` that exist on disk and are NOT needed by
/// `active_img`, `active_vid`, or any other downloaded profile on disk.
pub fn profile_deletable_files(
    target: &Profile,
    active_img: &Profile,
    active_vid: &Profile,
    all_profiles: &[Profile],
    models_dir: &Path,
) -> Vec<(PathBuf, String, u64)> {
    if models_dir.as_os_str().is_empty()
        || target.id == active_img.id
        || target.id == active_vid.id
    {
        return Vec::new();
    }

    let mut protected: HashSet<(String, String)> = HashSet::new();
    for m in active_img.models.iter().chain(active_vid.models.iter()) {
        protected.insert((m.category.clone(), m.filename.clone()));
    }
    for other in all_profiles {
        if other.id == target.id || other.id == active_img.id || other.id == active_vid.id {
            continue;
        }
        if primary_model_exists(other, models_dir) {
            for m in &other.models {
                protected.insert((m.category.clone(), m.filename.clone()));
            }
        }
    }

    let mut out = Vec::new();
    for m in &target.models {
        if protected.contains(&(m.category.clone(), m.filename.clone())) {
            continue;
        }
        let path = engine::safe_model_path(models_dir, &m.category, &m.filename);
        if path.is_file() {
            let sz = std::fs::metadata(&path).map(|md| md.len()).unwrap_or(m.size_bytes);
            out.push((path, m.filename.clone(), sz));
        }
    }
    out
}

struct VariantMeta<'a> {
    short_title: &'a str,
    subtitle: &'a str,
    vram_label: &'a str,
    icon: VariantIcon,
}

fn image_variant_meta(id: &str, vram_gb: Option<u32>) -> VariantMeta<'static> {
    match id {
        "qwen_2.1_int8" => VariantMeta {
            short_title: "Standard INT8",
            subtitle: if vram_gb.map(|v| v <= 8).unwrap_or(false) {
                "12GB+ VRAM - much slower on 8GB"
            } else {
                "Best quality | 25 steps"
            },
            vram_label: "12GB+ VRAM",
            icon: VariantIcon::Quality,
        },
        "qwen_2.1_8gb" => VariantMeta {
            short_title: "8GB W4A8",
            subtitle: if vram_gb.map(|v| v <= 8).unwrap_or(false) {
                "Optimized 4-bit quant | 25 steps"
            } else {
                "Lower quality, Lower VRAM - for 8GB cards"
            },
            vram_label: "8GB VRAM",
            icon: VariantIcon::Compact,
        },
        "qwen_2.1_turbo_8gb" => VariantMeta {
            short_title: "Turbo 8-Step",
            subtitle: "Faster generation, Lower quality",
            vram_label: "8-Step Fast",
            icon: VariantIcon::Turbo,
        },
        _ => VariantMeta {
            short_title: "Custom",
            subtitle: "Custom image profile",
            vram_label: "Custom",
            icon: VariantIcon::Balanced,
        },
    }
}

fn video_variant_meta(id: &str, is_step_up: bool) -> VariantMeta<'static> {
    match id {
        "minimax_h3_24gb" => VariantMeta {
            short_title: "24GB INT8",
            subtitle: "Native INT8 ConvRot",
            vram_label: "22-24GB+",
            icon: VariantIcon::Quality,
        },
        "minimax_h3_16gb" => VariantMeta {
            short_title: "16GB Q6_K",
            subtitle: "High-precision GGUF",
            vram_label: "16GB+",
            icon: VariantIcon::Quality,
        },
        "minimax_h3_q5_k_m" => VariantMeta {
            short_title: "Q5_K_M",
            subtitle: if is_step_up { "Higher quality Q5_K_M" } else { "5-bit medium GGUF" },
            vram_label: "14GB+",
            icon: VariantIcon::Balanced,
        },
        "minimax_h3_q5_k_s" => VariantMeta {
            short_title: "Q5_K_S",
            subtitle: if is_step_up { "Higher quality Q5_K_S" } else { "5-bit small GGUF" },
            vram_label: "14GB+",
            icon: VariantIcon::Balanced,
        },
        "minimax_h3_12gb" => VariantMeta {
            short_title: "12GB Q4_K_M",
            subtitle: if is_step_up { "Higher quality Q4_K_M" } else { "Balanced GGUF" },
            vram_label: "12GB+",
            icon: VariantIcon::Balanced,
        },
        "minimax_h3_q4_k_s" => VariantMeta {
            short_title: "Q4_K_S",
            subtitle: if is_step_up { "Higher quality Q4_K_S" } else { "Compact 4-bit GGUF" },
            vram_label: "11GB+",
            icon: VariantIcon::Compact,
        },
        "minimax_h3_10gb" => VariantMeta {
            short_title: "10GB Q3_K_M",
            subtitle: "Compact GGUF",
            vram_label: "10-11GB",
            icon: VariantIcon::Compact,
        },
        "ltx_2.5_24gb" => VariantMeta {
            short_title: "24GB INT8",
            subtitle: "Native INT8 + Gemma INT8",
            vram_label: "22-24GB+",
            icon: VariantIcon::Quality,
        },
        "ltx_2.5_16gb" => VariantMeta {
            short_title: "16GB Q6_K",
            subtitle: "Q6_K + W4A8 Gemma",
            vram_label: "16GB+",
            icon: VariantIcon::Quality,
        },
        "ltx_2.5_q5_k_xl" => VariantMeta {
            short_title: "UD-Q5_K_XL",
            subtitle: if is_step_up { "Higher quality Dynamic Q5" } else { "Dynamic 5-bit GGUF" },
            vram_label: "14GB+",
            icon: VariantIcon::Balanced,
        },
        "ltx_2.5_q5_k_m" => VariantMeta {
            short_title: "Q5_K_M",
            subtitle: if is_step_up { "Higher quality Q5_K_M" } else { "5-bit medium GGUF" },
            vram_label: "14GB+",
            icon: VariantIcon::Balanced,
        },
        "ltx_2.5_12gb" => VariantMeta {
            short_title: "12GB UD-Q4",
            subtitle: if is_step_up { "Higher quality Dynamic Q4" } else { "Balanced UD-Q4_K_XL" },
            vram_label: "12GB+",
            icon: VariantIcon::Balanced,
        },
        "ltx_2.5_q4_k_m" => VariantMeta {
            short_title: "Q4_K_M",
            subtitle: if is_step_up { "Higher quality Q4_K_M" } else { "Standard 4-bit GGUF" },
            vram_label: "11GB+",
            icon: VariantIcon::Compact,
        },
        "ltx_2.5_10gb" => VariantMeta {
            short_title: "10GB Q3_K",
            subtitle: "UD-Q3_K_XL + W4A8",
            vram_label: "<= 10GB",
            icon: VariantIcon::Compact,
        },
        "ltx_2.3" => VariantMeta {
            short_title: "Legacy 2.3",
            subtitle: "22B Dev FP8 + Gemma 3",
            vram_label: "16GB Legacy",
            icon: VariantIcon::Legacy,
        },
        _ => VariantMeta {
            short_title: "Custom",
            subtitle: "Custom video profile",
            vram_label: "Custom",
            icon: VariantIcon::Balanced,
        },
    }
}

pub fn render_model_picker(
    ui: &mut egui::Ui,
    state: &mut ModelPickerState,
    active_image_profile: &Profile,
    active_video_profile: &Profile,
    vram_detected: Option<u32>,
    models_dir: &Path,
) -> PickerAction {
    let mut action = PickerAction::None;
    let all_raw = engine::Profile::all_raw();

    // -------------------------------------------------------------------------
    // IMAGE MODEL SECTION
    // -------------------------------------------------------------------------
    ui.add_space(4.0);
    ui.label(RichText::new("Image Model").size(12.5).strong().color(ACCENT_GLOW));
    ui.add_space(4.0);

    let avail_w = ui.available_width();
    let main_gap = 10.0;
    let main_tile_w = ((avail_w - main_gap) * 0.5).max(180.0);
    let img_tile_h = 122.0;

    let rec_img_id = recommended_image_profile_id(vram_detected);
    let active_img_is_rec = active_image_profile.id == rec_img_id;

    ui.horizontal(|ui| {
        let clicked = draw_family_tile(
            ui,
            vec2(main_tile_w, img_tile_h),
            "Qwen Image 2.1",
            &[
                ("Inpainting & mask editing", true),
                ("Multi-reference editing", true),
            ],
            state.selected_image_family == ImageFamily::Qwen21,
            true,
            active_img_is_rec,
            true,
            |painter, center| draw_qwen_logo(painter, center),
        );
        if clicked {
            state.selected_image_family = ImageFamily::Qwen21;
        }
    });

    let all_img = engine::Profile::all();
    let img_order = ["qwen_2.1_int8", "qwen_2.1_8gb", "qwen_2.1_turbo_8gb"];
    let img_variants: Vec<Profile> = img_order
        .iter()
        .filter_map(|id| all_img.iter().find(|p| p.id == *id).cloned())
        .collect();

    if !img_variants.is_empty() {
        ui.add_space(6.0);
        egui::Frame::new()
            .fill(TRAY_BG)
            .stroke(Stroke::new(1.0, LINE_MUTED))
            .corner_radius(8)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                let inner_w = ui.available_width();
                let sub_gap = 8.0;
                let n = img_variants.len() as f32;
                let sub_w = ((inner_w - sub_gap * (n - 1.0)) / n).max(120.0);
                let sub_h = 96.0;

                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = sub_gap;
                    for p in &img_variants {
                        let is_active = p.id == active_image_profile.id;
                        let is_rec = p.id == rec_img_id;
                        let meta = image_variant_meta(&p.id, vram_detected);
                        let size_line = format!("~{:.1} GB", p.total_size_gb());
                        let deletable = profile_deletable_files(
                            p,
                            active_image_profile,
                            active_video_profile,
                            &all_raw,
                            models_dir,
                        );
                        let deletable_bytes: u64 = deletable.iter().map(|(_, _, s)| *s).sum();

                        let res = draw_variant_subtile(
                            ui,
                            vec2(sub_w, sub_h),
                            meta.short_title,
                            meta.subtitle,
                            meta.vram_label,
                            &size_line,
                            meta.icon,
                            is_active,
                            is_rec,
                            false,
                            if deletable.is_empty() { None } else { Some(deletable_bytes) },
                        );
                        if res.delete_clicked {
                            state.pending_delete = Some(PendingModelDelete {
                                profile_name: p.name.clone(),
                                files: deletable,
                                error: None,
                            });
                        } else if res.tile_clicked && !is_active {
                            if is_rec {
                                action = PickerAction::SwitchImage(p.clone());
                            } else {
                                state.pending_warning = Some(build_image_warning(p, vram_detected));
                            }
                        }
                    }
                });
            });
    }

    // -------------------------------------------------------------------------
    // VIDEO MODEL SECTION
    // -------------------------------------------------------------------------
    ui.add_space(12.0);
    ui.label(RichText::new("Video Model").size(12.5).strong().color(ACCENT_GLOW));
    ui.add_space(4.0);

    let rec_vid_family = recommended_video_family(vram_detected);
    let active_vid_family = if active_video_profile.is_minimax_h3() {
        VideoFamily::MiniMaxH3
    } else {
        VideoFamily::LtxVideo
    };
    let rec_vid_id_for_active = recommended_video_profile_id(active_vid_family, vram_detected);
    let active_vid_is_rec = active_video_profile.id == rec_vid_id_for_active;

    let all_vid = engine::Profile::all_video_profiles();

    let vid_tile_h = 158.0;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = main_gap;

        let mm_clicked = draw_family_tile(
            ui,
            vec2(main_tile_w, vid_tile_h),
            "MiniMax H3 Video",
            &[
                ("Prompt adherence & motion", true),
                ("Higher quality faces", true),
                ("T2V, I2V, End-Frame & R2V", true),
                ("Slower generation", false),
            ],
            state.selected_video_family == VideoFamily::MiniMaxH3,
            active_vid_family == VideoFamily::MiniMaxH3,
            active_vid_is_rec,
            rec_vid_family == VideoFamily::MiniMaxH3,
            |painter, center| draw_minimax_logo(painter, center),
        );
        if mm_clicked {
            state.selected_video_family = VideoFamily::MiniMaxH3;
            if active_vid_family != VideoFamily::MiniMaxH3 {
                if let Some(p) = find_downloaded_family_profile(
                    VideoFamily::MiniMaxH3,
                    &all_vid,
                    state.last_minimax_id.as_deref(),
                    vram_detected,
                    models_dir,
                ) {
                    state.record_active_video(&p);
                    action = PickerAction::SwitchVideo(p);
                }
            }
        }

        let ltx_clicked = draw_family_tile(
            ui,
            vec2(main_tile_w, vid_tile_h),
            "LTX-Video 2.5",
            &[
                ("Scenery & environments", true),
                ("2-stage x2 upscaling", true),
                ("Voices & audio clarity", true),
                ("Faster generation", true),
            ],
            state.selected_video_family == VideoFamily::LtxVideo,
            active_vid_family == VideoFamily::LtxVideo,
            active_vid_is_rec,
            rec_vid_family == VideoFamily::LtxVideo,
            |painter, center| draw_ltx_logo(painter, center),
        );
        if ltx_clicked {
            state.selected_video_family = VideoFamily::LtxVideo;
            if active_vid_family != VideoFamily::LtxVideo {
                if let Some(p) = find_downloaded_family_profile(
                    VideoFamily::LtxVideo,
                    &all_vid,
                    state.last_ltx_id.as_deref(),
                    vram_detected,
                    models_dir,
                ) {
                    state.record_active_video(&p);
                    action = PickerAction::SwitchVideo(p);
                }
            }
        }
    });

    // Swappable Video Variant Sub-Tiles (largest on the left, smallest on the right, 4 per row)
    let vid_order: &[&str] = match state.selected_video_family {
        VideoFamily::MiniMaxH3 => &[
            "minimax_h3_24gb",
            "minimax_h3_16gb",
            "minimax_h3_q5_k_m",
            "minimax_h3_q5_k_s",
            "minimax_h3_12gb",
            "minimax_h3_q4_k_s",
            "minimax_h3_10gb",
        ],
        VideoFamily::LtxVideo => &[
            "ltx_2.5_24gb",
            "ltx_2.5_16gb",
            "ltx_2.5_q5_k_xl",
            "ltx_2.5_q5_k_m",
            "ltx_2.5_12gb",
            "ltx_2.5_q4_k_m",
            "ltx_2.5_10gb",
            "ltx_2.3",
        ],
    };
    let vid_variants: Vec<Profile> = vid_order
        .iter()
        .filter_map(|id| all_vid.iter().find(|p| p.id == *id).cloned())
        .collect();
    let rec_vid_id = recommended_video_profile_id(state.selected_video_family, vram_detected);
    let rec_vid_vram = vid_variants
        .iter()
        .find(|p| p.id == rec_vid_id)
        .map(|p| p.recommended_vram_gb)
        .unwrap_or(12);

    if !vid_variants.is_empty() {
        ui.add_space(6.0);
        egui::Frame::new()
            .fill(TRAY_BG)
            .stroke(Stroke::new(1.0, LINE_MUTED))
            .corner_radius(8)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                let inner_w = ui.available_width();
                let sub_gap = 7.0;
                let cols = 4usize;
                let sub_w = ((inner_w - sub_gap * (cols as f32 - 1.0)) / cols as f32).max(115.0);
                let sub_h = 98.0;

                for (row_idx, chunk) in vid_variants.chunks(cols).enumerate() {
                    if row_idx > 0 {
                        ui.add_space(sub_gap);
                    }
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = sub_gap;
                        for vp in chunk {
                            let is_active = vp.id == active_video_profile.id;
                            let is_rec = vp.id == rec_vid_id;
                            let is_step_up = is_allowed_step_up_variant(&vp.id, vram_detected);
                            let meta = video_variant_meta(&vp.id, is_step_up);
                            let size_line = format!("~{:.1} GB", vp.total_size_gb());
                            let deletable = profile_deletable_files(
                                vp,
                                active_image_profile,
                                active_video_profile,
                                &all_raw,
                                models_dir,
                            );
                            let deletable_bytes: u64 = deletable.iter().map(|(_, _, s)| *s).sum();

                            let res = draw_variant_subtile(
                                ui,
                                vec2(sub_w, sub_h),
                                meta.short_title,
                                meta.subtitle,
                                meta.vram_label,
                                &size_line,
                                meta.icon,
                                is_active,
                                is_rec,
                                is_step_up,
                                if deletable.is_empty() { None } else { Some(deletable_bytes) },
                            );
                            if res.delete_clicked {
                                state.pending_delete = Some(PendingModelDelete {
                                    profile_name: vp.name.clone(),
                                    files: deletable,
                                    error: None,
                                });
                            } else if res.tile_clicked && !is_active {
                                if is_rec || is_step_up {
                                    state.record_active_video(vp);
                                    action = PickerAction::SwitchVideo(vp.clone());
                                } else {
                                    state.pending_warning = Some(build_video_warning(
                                        vp,
                                        rec_vid_vram,
                                        vram_detected,
                                    ));
                                }
                            }
                        }
                    });
                }
            });
    }

    action
}

pub fn render_warning_modal(ctx: &egui::Context, state: &mut ModelPickerState) -> PickerAction {
    if let Some(ref mut del) = state.pending_delete {
        let mut do_delete = false;
        let mut cancel = false;
        let center = ctx.content_rect().center();
        let total_bytes: u64 = del.files.iter().map(|(_, _, s)| *s).sum();

        egui::Window::new("Delete Unused Model Files")
            .id(egui::Id::new("model_picker_delete_modal"))
            .collapsible(false)
            .resizable(false)
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(center)
            .fixed_size(vec2(470.0, 250.0))
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.label(RichText::new(&del.profile_name).size(13.0).strong().color(TEXT));
                ui.label(
                    RichText::new(format!(
                        "This will permanently delete {} unused file(s) ({}) from disk. Shared files still used by your active or other downloaded models are preserved.",
                        del.files.len(),
                        engine::format_bytes(total_bytes)
                    ))
                    .size(11.5)
                    .color(DIM),
                );
                ui.add_space(6.0);
                egui::Frame::new()
                    .fill(TRAY_BG)
                    .stroke(Stroke::new(1.0, LINE))
                    .corner_radius(6)
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical().max_height(88.0).show(ui, |ui| {
                            for (_, fname, sz) in &del.files {
                                ui.label(
                                    RichText::new(format!("- {} ({})", fname, engine::format_bytes(*sz)))
                                        .size(11.0)
                                        .color(TEXT),
                                );
                            }
                        });
                    });

                if let Some(ref err) = del.error {
                    ui.add_space(4.0);
                    ui.label(RichText::new(err).size(11.0).color(RED));
                }

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let del_btn = egui::Button::new(
                            RichText::new("Delete Files").strong().color(Color32::WHITE),
                        )
                        .fill(Color32::from_rgb(190, 52, 52))
                        .min_size(vec2(120.0, 30.0));
                        if ui.add(del_btn).clicked() {
                            do_delete = true;
                        }
                        if ui.add(egui::Button::new("Cancel").min_size(vec2(86.0, 30.0))).clicked() {
                            cancel = true;
                        }
                    });
                });
            });

        if do_delete {
            let mut failed = Vec::new();
            for (path, fname, _) in &del.files {
                if path.exists() {
                    if let Err(e) = std::fs::remove_file(path) {
                        failed.push(format!("{fname}: {e}"));
                    }
                }
            }
            if failed.is_empty() {
                state.pending_delete = None;
                return PickerAction::ModelsDeleted;
            } else {
                del.error = Some(format!("Failed to delete: {}", failed.join("; ")));
                return PickerAction::None;
            }
        } else if cancel {
            state.pending_delete = None;
        }
        return PickerAction::None;
    }

    let Some(ref warn) = state.pending_warning else {
        return PickerAction::None;
    };

    let mut confirm = false;
    let mut cancel = false;
    let center = ctx.content_rect().center();
    let accent_col = if warn.is_severe { RED } else { AMBER };

    egui::Window::new(&warn.title)
        .id(egui::Id::new("model_picker_warning_modal"))
        .collapsible(false)
        .resizable(false)
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(center)
        .fixed_size(vec2(440.0, 210.0))
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let (icon_rect, _) = ui.allocate_exact_size(vec2(28.0, 28.0), Sense::hover());
                draw_warning_triangle(ui.painter(), icon_rect.center(), 12.0, accent_col);
                ui.add_space(4.0);
                ui.vertical(|ui| {
                    ui.label(RichText::new(&warn.profile.name).size(13.0).strong().color(TEXT));
                    ui.label(
                        RichText::new(format!(
                            "Recommended VRAM: {} GB | Download: ~{:.1} GB",
                            warn.profile.recommended_vram_gb,
                            warn.profile.total_size_gb()
                        ))
                        .size(11.0)
                        .color(DIM),
                    );
                });
            });

            ui.add_space(8.0);
            egui::Frame::new()
                .fill(if warn.is_severe {
                    Color32::from_rgb(42, 22, 18)
                } else {
                    AMBER_FILL
                })
                .stroke(Stroke::new(1.0, accent_col))
                .corner_radius(6)
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    ui.label(RichText::new(&warn.message).size(12.0).color(TEXT));
                });

            ui.add_space(12.0);
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let ok_btn = egui::Button::new(
                        RichText::new("OK, Switch Anyway").strong().color(Color32::BLACK),
                    )
                    .fill(AMBER)
                    .min_size(vec2(140.0, 30.0));
                    if ui.add(ok_btn).clicked() {
                        confirm = true;
                    }

                    let cancel_btn = egui::Button::new("Cancel").min_size(vec2(86.0, 30.0));
                    if ui.add(cancel_btn).clicked() {
                        cancel = true;
                    }
                });
            });
        });

    if confirm {
        let w = state.pending_warning.take().unwrap();
        if w.is_video {
            state.record_active_video(&w.profile);
            PickerAction::SwitchVideo(w.profile)
        } else {
            PickerAction::SwitchImage(w.profile)
        }
    } else if cancel {
        state.pending_warning = None;
        PickerAction::None
    } else {
        PickerAction::None
    }
}

fn build_image_warning(p: &Profile, vram_gb: Option<u32>) -> PendingModelWarning {
    match p.id.as_str() {
        "qwen_2.1_8gb" => PendingModelWarning {
            is_video: false,
            profile: p.clone(),
            title: "Lower Quality Image Model".to_string(),
            message: format!(
                "Your GPU{} already has enough VRAM for the Standard INT8 model. Switching to the 8GB W4A8 quantized model will not help generation speed much and will just lower output quality.",
                vram_gb.map(|g| format!(" (~{g} GB)")).unwrap_or_default()
            ),
            is_severe: false,
        },
        "qwen_2.1_turbo_8gb" => PendingModelWarning {
            is_video: false,
            profile: p.clone(),
            title: "Turbo Mode: Speed vs Quality".to_string(),
            message: "Turbo uses 8-step distilled inference instead of 25 steps. This will significantly increase generation speed, but at the cost of image quality and fine detail.".to_string(),
            is_severe: false,
        },
        "qwen_2.1_int8" => PendingModelWarning {
            is_video: false,
            profile: p.clone(),
            title: "Exceeds 8GB VRAM".to_string(),
            message: format!(
                "Standard INT8 recommends 12GB+ VRAM (detected: {} GB). Running it on an 8GB GPU will spill weights into system RAM, making generation much slower and risking out-of-memory errors.",
                vram_gb.unwrap_or(8)
            ),
            is_severe: true,
        },
        _ => PendingModelWarning {
            is_video: false,
            profile: p.clone(),
            title: "Non-Recommended Profile".to_string(),
            message: "This profile is not the recommended default for your GPU. Continue anyway?".to_string(),
            is_severe: false,
        },
    }
}

fn build_video_warning(
    vp: &Profile,
    rec_vram: u32,
    vram_gb: Option<u32>,
) -> PendingModelWarning {
    if vp.id == "ltx_2.3" {
        return PendingModelWarning {
            is_video: true,
            profile: vp.clone(),
            title: "Legacy Model (LTX-Video 2.3)".to_string(),
            message: "LTX-Video 2.3 is a legacy ~43 GB model. LTX-Video 2.5 is newer, faster, and produces higher quality video and audio. Switch to legacy LTX-Video 2.3 anyway?".to_string(),
            is_severe: false,
        };
    }

    let exceeds_detected = vram_gb.map(|g| vp.recommended_vram_gb > g).unwrap_or(false);
    if exceeds_detected || vp.recommended_vram_gb > rec_vram {
        let gpu_str = vram_gb
            .map(|g| format!("{g} GB"))
            .unwrap_or_else(|| "unknown".to_string());
        PendingModelWarning {
            is_video: true,
            profile: vp.clone(),
            title: "VRAM Warning: Slower Generation".to_string(),
            message: format!(
                "This variant recommends {} GB VRAM (detected: {}). Running it on this GPU will rely heavily on shared system RAM paging, causing 10x-100x slower generation or CUDA Out-Of-Memory (OOM) errors.",
                vp.recommended_vram_gb, gpu_str
            ),
            is_severe: true,
        }
    } else {
        let gpu_str = vram_gb
            .map(|g| format!(" (~{g} GB VRAM)"))
            .unwrap_or_default();
        PendingModelWarning {
            is_video: true,
            profile: vp.clone(),
            title: "Lower Quality Variant".to_string(),
            message: format!(
                "Your GPU{} already supports a higher-precision variant. Switching to this smaller quantized model will not improve generation speed noticeably and will lower output quality.",
                gpu_str
            ),
            is_severe: false,
        }
    }
}

fn draw_family_tile(
    ui: &mut egui::Ui,
    size: Vec2,
    title: &str,
    points: &[(&str, bool)],
    is_expanded: bool,
    is_active_family: bool,
    active_is_rec: bool,
    is_rec_family: bool,
    draw_logo: impl FnOnce(&egui::Painter, Pos2),
) -> bool {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    let (fill, border_col, border_w) = if is_active_family {
        if active_is_rec {
            (GREEN_FILL, GREEN, 2.0)
        } else {
            (AMBER_FILL, AMBER, 2.0)
        }
    } else if is_expanded {
        (CARD_HOVER, ACCENT_GLOW, 1.5)
    } else if resp.hovered() {
        (CARD_HOVER, LINE, 1.2)
    } else {
        (CARD_BG, LINE_MUTED, 1.0)
    };

    let painter = ui.painter();
    painter.rect_filled(rect, 8.0, fill);
    painter.rect_stroke(rect, 8.0, Stroke::new(border_w, border_col), StrokeKind::Inside);

    let logo_center = pos2(rect.center().x, rect.top() + 28.0);
    draw_logo(painter, logo_center);

    if is_rec_family {
        if is_active_family {
            draw_pill_left(painter, pos2(rect.left() + 8.0, rect.top() + 8.0), "RECOMMENDED", GREEN);
        } else {
            draw_pill(painter, pos2(rect.right() - 8.0, rect.top() + 8.0), "RECOMMENDED", GREEN);
        }
    }
    if is_active_family {
        let pill_col = if active_is_rec { GREEN } else { AMBER };
        let pill_txt = if active_is_rec { "ACTIVE" } else { "CUSTOM" };
        draw_pill(painter, pos2(rect.right() - 8.0, rect.top() + 8.0), pill_txt, pill_col);
    }

    painter.text(
        pos2(rect.center().x, rect.top() + 58.0),
        egui::Align2::CENTER_CENTER,
        title,
        egui::FontId::proportional(13.5),
        if is_active_family || is_expanded || resp.hovered() { TEXT } else { DIM },
    );

    let div_y = rect.top() + 72.0;
    painter.line_segment(
        [pos2(rect.left() + 14.0, div_y), pos2(rect.right() - 14.0, div_y)],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(LINE.r(), LINE.g(), LINE.b(), 120)),
    );

    let mut row_y = div_y + 15.0;
    let left_x = rect.left() + 16.0;
    for &(label, is_pro) in points {
        let badge_center = pos2(left_x + 6.0, row_y);
        let col = if is_pro { GREEN } else { AMBER };
        painter.circle_filled(
            badge_center,
            6.0,
            Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 32),
        );
        painter.line_segment(
            [pos2(badge_center.x - 3.0, badge_center.y), pos2(badge_center.x + 3.0, badge_center.y)],
            Stroke::new(1.5, col),
        );
        if is_pro {
            painter.line_segment(
                [pos2(badge_center.x, badge_center.y - 3.0), pos2(badge_center.x, badge_center.y + 3.0)],
                Stroke::new(1.5, col),
            );
        }
        painter.text(
            pos2(left_x + 17.0, row_y),
            egui::Align2::LEFT_CENTER,
            label,
            egui::FontId::proportional(11.0),
            if is_pro { DIM } else { AMBER },
        );
        row_y += 18.0;
    }

    resp.clicked()
}

struct SubtileResult {
    tile_clicked: bool,
    delete_clicked: bool,
}

fn draw_variant_subtile(
    ui: &mut egui::Ui,
    size: Vec2,
    title: &str,
    subtitle: &str,
    vram_label: &str,
    size_line: &str,
    icon: VariantIcon,
    is_active: bool,
    is_recommended: bool,
    is_step_up: bool,
    deletable_bytes: Option<u64>,
) -> SubtileResult {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());

    // Optional delete button in bottom-right corner for downloaded inactive models
    let mut delete_clicked = false;
    let del_rect = Rect::from_min_size(pos2(rect.right() - 26.0, rect.bottom() - 22.0), vec2(20.0, 16.0));
    let mut del_hovered = false;
    if let Some(bytes) = deletable_bytes {
        let del_resp = ui.interact(del_rect, resp.id.with("del"), Sense::click());
        del_hovered = del_resp.hovered();
        if del_hovered {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        delete_clicked = del_resp.clicked();
        del_resp.on_hover_text(format!(
            "Delete unused model files ({}) to free disk space",
            engine::format_bytes(bytes)
        ));
    }

    if resp.hovered() && !del_hovered {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    let (fill, border_col, border_w, title_col, sub_col, icon_col) = if is_active && is_recommended {
        (GREEN_FILL, GREEN, 2.0, TEXT, DIM, GREEN)
    } else if is_active {
        (AMBER_FILL, AMBER, 2.0, TEXT, DIM, AMBER)
    } else if is_recommended {
        let bg = if resp.hovered() { Color32::from_rgb(24, 48, 40) } else { CARD_BG };
        (bg, Color32::from_rgb(45, 130, 98), 1.5, TEXT, DIM, GREEN)
    } else if is_step_up {
        let bg = if resp.hovered() { Color32::from_rgb(40, 34, 22) } else { CARD_BG };
        (bg, Color32::from_rgb(155, 120, 35), 1.2, TEXT, DIM, AMBER)
    } else if resp.hovered() {
        (CARD_HOVER, LINE, 1.2, TEXT, DIM, DIM)
    } else {
        (CARD_MUTED, LINE_MUTED, 1.0, MUTED_TEXT, Color32::from_rgb(92, 98, 118), MUTED_TEXT)
    };

    let painter = ui.painter();
    painter.rect_filled(rect, 6.0, fill);
    painter.rect_stroke(rect, 6.0, Stroke::new(border_w, border_col), StrokeKind::Inside);

    let pad = 8.0;
    let icon_center = pos2(rect.left() + pad + 7.0, rect.top() + pad + 8.0);
    draw_variant_icon(painter, icon_center, icon, icon_col);

    painter.text(
        pos2(rect.left() + pad + 18.0, rect.top() + pad + 8.0),
        egui::Align2::LEFT_CENTER,
        title,
        egui::FontId::proportional(11.5),
        title_col,
    );

    let badge_y = rect.top() + pad + 24.0;
    if is_recommended {
        draw_pill_left(painter, pos2(rect.left() + pad, badge_y), "RECOMMENDED", GREEN);
    } else if is_step_up {
        draw_pill_left(painter, pos2(rect.left() + pad, badge_y), "SLOWER", AMBER);
    } else if is_active {
        draw_pill_left(painter, pos2(rect.left() + pad, badge_y), "SELECTED", AMBER);
    } else {
        draw_pill_left(painter, pos2(rect.left() + pad, badge_y), vram_label, MUTED_TEXT);
    }

    let galley = painter.layout(
        subtitle.to_string(),
        egui::FontId::proportional(10.0),
        sub_col,
        (rect.width() - pad * 2.0).max(40.0),
    );
    painter.galley(pos2(rect.left() + pad, rect.top() + 50.0), galley, sub_col);

    painter.text(
        pos2(rect.left() + pad, rect.bottom() - 10.0),
        egui::Align2::LEFT_CENTER,
        size_line,
        egui::FontId::proportional(10.0),
        if is_active || is_recommended || is_step_up { DIM } else { MUTED_TEXT },
    );

    if deletable_bytes.is_some() {
        let del_bg = if del_hovered {
            Color32::from_rgb(130, 38, 38)
        } else {
            Color32::from_rgb(55, 28, 32)
        };
        let del_fg = if del_hovered { Color32::WHITE } else { RED };
        painter.rect_filled(del_rect, 3.0, del_bg);
        painter.rect_stroke(del_rect, 3.0, Stroke::new(1.0, Color32::from_rgb(150, 55, 55)), StrokeKind::Inside);
        draw_trash_icon(painter, del_rect.center(), del_fg);
    }

    SubtileResult {
        tile_clicked: resp.clicked() && !delete_clicked,
        delete_clicked,
    }
}

fn draw_pill(painter: &egui::Painter, top_right: Pos2, text: &str, color: Color32) {
    let galley = painter.layout_no_wrap(text.to_string(), egui::FontId::proportional(9.0), color);
    let w = galley.size().x + 10.0;
    let h = 15.0;
    let rect = Rect::from_min_size(pos2(top_right.x - w, top_right.y), vec2(w, h));
    painter.rect_filled(rect, 4.0, Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 28));
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 140)), StrokeKind::Inside);
    painter.galley(pos2(rect.left() + 5.0, rect.center().y - galley.size().y * 0.5), galley, color);
}

fn draw_pill_left(painter: &egui::Painter, top_left: Pos2, text: &str, color: Color32) {
    let galley = painter.layout_no_wrap(text.to_string(), egui::FontId::proportional(9.0), color);
    let w = galley.size().x + 10.0;
    let h = 15.0;
    let rect = Rect::from_min_size(top_left, vec2(w, h));
    painter.rect_filled(rect, 4.0, Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 24));
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 120)), StrokeKind::Inside);
    painter.galley(pos2(rect.left() + 5.0, rect.center().y - galley.size().y * 0.5), galley, color);
}

// =============================================================================
// Custom Vector Logos & Icons (Painter-based, zero emojis)
// =============================================================================

fn draw_qwen_logo(painter: &egui::Painter, center: Pos2) {
    let r = 17.0;
    let mut hex = Vec::with_capacity(6);
    for i in 0..6 {
        let ang = std::f32::consts::FRAC_PI_6 + (i as f32) * std::f32::consts::FRAC_PI_3;
        hex.push(pos2(center.x + r * ang.cos(), center.y + r * ang.sin()));
    }
    painter.add(egui::Shape::convex_polygon(
        hex,
        Color32::from_rgba_unmultiplied(99, 102, 241, 45),
        Stroke::new(1.6, ACCENT_GLOW),
    ));
    painter.circle_stroke(center, 7.5, Stroke::new(2.2, TEXT));
    painter.line_segment(
        [pos2(center.x + 2.5, center.y + 2.5), pos2(center.x + 9.5, center.y + 9.5)],
        Stroke::new(2.4, ACCENT_GLOW),
    );
}

fn draw_minimax_logo(painter: &egui::Painter, center: Pos2) {
    let rect = Rect::from_center_size(center, vec2(34.0, 34.0));
    painter.rect_filled(rect, 9.0, Color32::from_rgba_unmultiplied(59, 130, 246, 40));
    painter.rect_stroke(rect, 9.0, Stroke::new(1.6, BLUE), StrokeKind::Inside);

    let bars: &[(f32, f32, Color32)] = &[
        (-9.0, 7.0, ACCENT_GLOW),
        (-3.0, 12.0, TEXT),
        (3.0, 9.0, ACCENT_GLOW),
        (9.0, 6.0, BLUE),
    ];
    for &(dx, half_h, col) in bars {
        painter.line_segment(
            [pos2(center.x + dx, center.y - half_h), pos2(center.x + dx, center.y + half_h)],
            Stroke::new(2.6, col),
        );
    }
}

fn draw_ltx_logo(painter: &egui::Painter, center: Pos2) {
    let rect = Rect::from_center_size(center, vec2(36.0, 30.0));
    painter.rect_filled(rect, 7.0, Color32::from_rgba_unmultiplied(52, 211, 153, 32));
    painter.rect_stroke(rect, 7.0, Stroke::new(1.6, GREEN), StrokeKind::Inside);

    let tri = vec![
        pos2(center.x - 5.0, center.y - 7.5),
        pos2(center.x + 8.5, center.y),
        pos2(center.x - 5.0, center.y + 7.5),
    ];
    painter.add(egui::Shape::convex_polygon(tri, TEXT, Stroke::NONE));
}

fn draw_variant_icon(painter: &egui::Painter, center: Pos2, icon: VariantIcon, col: Color32) {
    match icon {
        VariantIcon::Quality => {
            let pts = vec![
                pos2(center.x, center.y - 6.0),
                pos2(center.x + 6.0, center.y),
                pos2(center.x, center.y + 6.0),
                pos2(center.x - 6.0, center.y),
            ];
            painter.add(egui::Shape::convex_polygon(pts, Color32::TRANSPARENT, Stroke::new(1.5, col)));
        }
        VariantIcon::Balanced => {
            let r1 = Rect::from_center_size(pos2(center.x - 3.0, center.y - 3.0), vec2(4.0, 4.0));
            let r2 = Rect::from_center_size(pos2(center.x + 3.0, center.y - 3.0), vec2(4.0, 4.0));
            let r3 = Rect::from_center_size(pos2(center.x - 3.0, center.y + 3.0), vec2(4.0, 4.0));
            let r4 = Rect::from_center_size(pos2(center.x + 3.0, center.y + 3.0), vec2(4.0, 4.0));
            for r in [r1, r2, r3, r4] {
                painter.rect_stroke(r, 1.0, Stroke::new(1.2, col), StrokeKind::Inside);
            }
        }
        VariantIcon::Compact => {
            let r = Rect::from_center_size(center, vec2(10.0, 10.0));
            painter.rect_stroke(r, 2.0, Stroke::new(1.5, col), StrokeKind::Inside);
            painter.rect_filled(Rect::from_center_size(center, vec2(4.0, 4.0)), 1.0, col);
        }
        VariantIcon::Turbo => {
            let pts = [
                pos2(center.x + 1.5, center.y - 6.5),
                pos2(center.x - 4.0, center.y + 0.5),
                pos2(center.x + 0.5, center.y + 0.5),
                pos2(center.x - 1.5, center.y + 6.5),
                pos2(center.x + 4.5, center.y - 0.5),
                pos2(center.x, center.y - 0.5),
            ];
            for w in pts.windows(2) {
                painter.line_segment([w[0], w[1]], Stroke::new(1.6, col));
            }
        }
        VariantIcon::Legacy => {
            painter.circle_stroke(center, 5.5, Stroke::new(1.4, col));
            painter.line_segment([center, pos2(center.x, center.y - 3.0)], Stroke::new(1.4, col));
            painter.line_segment([center, pos2(center.x + 2.5, center.y + 1.0)], Stroke::new(1.4, col));
        }
    }
}

fn draw_trash_icon(painter: &egui::Painter, center: Pos2, col: Color32) {
    // Lid
    painter.line_segment(
        [pos2(center.x - 4.5, center.y - 3.5), pos2(center.x + 4.5, center.y - 3.5)],
        Stroke::new(1.3, col),
    );
    painter.line_segment(
        [pos2(center.x - 1.5, center.y - 5.0), pos2(center.x + 1.5, center.y - 5.0)],
        Stroke::new(1.3, col),
    );
    // Bin body
    let body = Rect::from_min_max(pos2(center.x - 3.5, center.y - 2.0), pos2(center.x + 3.5, center.y + 4.5));
    painter.rect_stroke(body, 1.0, Stroke::new(1.2, col), StrokeKind::Inside);
    painter.line_segment(
        [pos2(center.x - 1.2, center.y - 0.5), pos2(center.x - 1.2, center.y + 3.0)],
        Stroke::new(1.0, col),
    );
    painter.line_segment(
        [pos2(center.x + 1.2, center.y - 0.5), pos2(center.x + 1.2, center.y + 3.0)],
        Stroke::new(1.0, col),
    );
}

fn draw_warning_triangle(painter: &egui::Painter, center: Pos2, radius: f32, col: Color32) {
    let pts = vec![
        pos2(center.x, center.y - radius),
        pos2(center.x + radius * 0.95, center.y + radius * 0.75),
        pos2(center.x - radius * 0.95, center.y + radius * 0.75),
    ];
    painter.add(egui::Shape::convex_polygon(
        pts,
        Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 35),
        Stroke::new(1.6, col),
    ));
    painter.line_segment(
        [pos2(center.x, center.y - radius * 0.35), pos2(center.x, center.y + radius * 0.22)],
        Stroke::new(1.8, col),
    );
    painter.circle_filled(pos2(center.x, center.y + radius * 0.48), 1.3, col);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recommended_image_profile_id() {
        assert_eq!(recommended_image_profile_id(Some(8)), "qwen_2.1_8gb");
        assert_eq!(recommended_image_profile_id(Some(6)), "qwen_2.1_8gb");
        assert_eq!(recommended_image_profile_id(Some(12)), "qwen_2.1_int8");
        assert_eq!(recommended_image_profile_id(Some(24)), "qwen_2.1_int8");
        assert_eq!(recommended_image_profile_id(None), "qwen_2.1_int8");
    }

    #[test]
    fn test_recommended_video_profile_id() {
        assert_eq!(recommended_video_profile_id(VideoFamily::MiniMaxH3, Some(24)), "minimax_h3_24gb");
        assert_eq!(recommended_video_profile_id(VideoFamily::MiniMaxH3, Some(16)), "minimax_h3_16gb");
        assert_eq!(recommended_video_profile_id(VideoFamily::MiniMaxH3, Some(12)), "minimax_h3_12gb");
        assert_eq!(recommended_video_profile_id(VideoFamily::MiniMaxH3, Some(10)), "minimax_h3_10gb");
        assert_eq!(recommended_video_profile_id(VideoFamily::LtxVideo, Some(24)), "ltx_2.5_24gb");
        assert_eq!(recommended_video_profile_id(VideoFamily::LtxVideo, Some(16)), "ltx_2.5_16gb");
        assert_eq!(recommended_video_profile_id(VideoFamily::LtxVideo, Some(12)), "ltx_2.5_12gb");
        assert_eq!(recommended_video_profile_id(VideoFamily::LtxVideo, Some(8)), "ltx_2.5_10gb");
    }

    #[test]
    fn test_allowed_step_up_variants() {
        // 12GB cards can pick Q5 variants without caveat
        assert!(is_allowed_step_up_variant("minimax_h3_q5_k_m", Some(12)));
        assert!(is_allowed_step_up_variant("minimax_h3_q5_k_s", Some(12)));
        assert!(is_allowed_step_up_variant("ltx_2.5_q5_k_xl", Some(12)));
        assert!(is_allowed_step_up_variant("ltx_2.5_q5_k_m", Some(12)));
        assert!(!is_allowed_step_up_variant("minimax_h3_16gb", Some(12)));

        // 10GB cards can pick Q4 variants without caveat
        assert!(is_allowed_step_up_variant("minimax_h3_12gb", Some(10)));
        assert!(is_allowed_step_up_variant("minimax_h3_q4_k_s", Some(10)));
        assert!(is_allowed_step_up_variant("ltx_2.5_12gb", Some(10)));
        assert!(is_allowed_step_up_variant("ltx_2.5_q4_k_m", Some(10)));
        assert!(!is_allowed_step_up_variant("minimax_h3_q5_k_m", Some(10)));
    }
}
