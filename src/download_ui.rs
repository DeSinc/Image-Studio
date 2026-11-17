use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, Rect, RichText, Sense, Stroke, StrokeKind, vec2};

use crate::engine::{self, EngState, Profile, Shared};
use crate::updater::{self, UpdaterState};

const CARD: Color32 = Color32::from_rgb(25, 29, 40);
const CARD_HI: Color32 = Color32::from_rgb(33, 38, 53);
const LINE: Color32 = Color32::from_rgb(52, 60, 82);
const ACCENT: Color32 = Color32::from_rgb(99, 102, 241);
const ACCENT_GLOW: Color32 = Color32::from_rgb(129, 140, 248);
const TEXT: Color32 = Color32::from_rgb(235, 238, 248);
const DIM: Color32 = Color32::from_rgb(168, 177, 202);
const GREEN: Color32 = Color32::from_rgb(52, 211, 153);
const AMBER: Color32 = Color32::from_rgb(251, 191, 36);
const RED: Color32 = Color32::from_rgb(248, 113, 113);

/// Crisp, zero-emoji pause icon (two vertical bars).
#[allow(dead_code)]
fn draw_pause_icon(p: &egui::Painter, center: egui::Pos2, color: Color32) {
    let stroke = Stroke::new(2.2, color);
    p.line_segment([center + vec2(-2.5, -4.5), center + vec2(-2.5, 4.5)], stroke);
    p.line_segment([center + vec2(2.5, -4.5), center + vec2(2.5, 4.5)], stroke);
}

/// Crisp, zero-emoji play / resume icon (right-pointing filled triangle).
#[allow(dead_code)]
fn draw_play_icon(p: &egui::Painter, center: egui::Pos2, color: Color32) {
    let pts = vec![
        center + vec2(-3.0, -5.0),
        center + vec2(5.0, 0.0),
        center + vec2(-3.0, 5.0),
    ];
    p.add(egui::Shape::convex_polygon(pts, color, Stroke::NONE));
}

/// Crisp minimize icon (horizontal bar).
fn draw_minimize_icon(p: &egui::Painter, center: egui::Pos2, color: Color32) {
    p.line_segment([center + vec2(-4.5, 2.5), center + vec2(4.5, 2.5)], Stroke::new(2.0, color));
}

#[derive(Clone, Debug)]
pub struct DownloadPaneState {
    pub is_expanded: bool,
    pub user_opened: bool,
    pub highlight_until: Option<Instant>,
    pub show_metered_dialog: bool,
}

impl Default for DownloadPaneState {
    fn default() -> Self {
        Self {
            is_expanded: false, // Starts minimized into the top-right update/download icon
            user_opened: false,
            highlight_until: None,
            show_metered_dialog: false,
        }
    }
}

/// Returns `Some((progress_0_to_1, is_paused, summary))` when an engine or model download is active.
pub fn active_download_progress(sh: &Shared) -> Option<(f32, bool, String)> {
    let st = sh.state.lock().unwrap().clone();
    let paused = matches!(st, EngState::ModelDownloadPaused { .. }) || sh.download_paused.load(Ordering::SeqCst);
    if let Some(ref mp) = *sh.model_dl_progress.lock().unwrap() {
        return Some((
            mp.overall_progress.clamp(0.0, 1.0),
            paused,
            format!("Downloading {} ({}/{}): {:.0}%", mp.current_file, mp.file_index, mp.total_files, mp.overall_progress * 100.0),
        ));
    }
    match st {
        EngState::DownloadingModels { current_file, file_index, total_files, overall_progress, .. }
        | EngState::ModelDownloadPaused { current_file, file_index, total_files, overall_progress, .. } => {
            Some((
                overall_progress.clamp(0.0, 1.0),
                paused,
                format!("Model download ({}/{}): {} ({:.0}%)", file_index, total_files, current_file, overall_progress * 100.0),
            ))
        }
        EngState::DownloadingEngine { step, progress, .. } => {
            Some((progress.clamp(0.0, 1.0), paused, format!("{step} ({:.0}%)", progress * 100.0)))
        }
        EngState::ExtractingEngine { step } => Some((0.0, false, step)),
        _ => {
            let pdls = crate::engine::all_profile_download_infos(sh);
            let loras: Vec<crate::lora::LoraDownloadStatus> = sh
                .lora_downloads
                .lock()
                .unwrap()
                .values()
                .filter(|d| !d.completed)
                .cloned()
                .collect();
            if pdls.is_empty() && loras.is_empty() {
                None
            } else if !pdls.is_empty() {
                let total_bytes: u64 = pdls.iter().map(|d| d.bytes_total).sum::<u64>()
                    + loras.iter().map(|l| l.total).sum::<u64>();
                let done_bytes: u64 = pdls.iter().map(|d| d.bytes_done).sum::<u64>()
                    + loras.iter().map(|l| l.downloaded).sum::<u64>();
                let prog = if total_bytes > 0 {
                    (done_bytes as f32 / total_bytes as f32).clamp(0.0, 1.0)
                } else {
                    pdls.iter().map(|d| d.progress).sum::<f32>() / pdls.len() as f32
                };
                let all_paused = pdls.iter().all(|d| d.status == crate::engine::ProfileDownloadStatus::Paused)
                    && loras.iter().all(|l| l.paused);
                let total_count = pdls.len() + loras.len();
                let summary = if total_count == 1 {
                    format!("Downloading {} ({:.0}%)", pdls[0].profile_name, prog * 100.0)
                } else {
                    format!("Downloading {} items ({:.0}%)", total_count, prog * 100.0)
                };
                Some((prog, all_paused, summary))
            } else {
                let total_bytes: u64 = loras.iter().map(|l| l.total).sum();
                let done_bytes: u64 = loras.iter().map(|l| l.downloaded).sum();
                let prog = if total_bytes > 0 {
                    (done_bytes as f32 / total_bytes as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let all_paused = loras.iter().all(|l| l.paused);
                let summary = if loras.len() == 1 {
                    format!("Downloading {} ({:.0}%)", loras[0].display_name, prog * 100.0)
                } else {
                    format!("Downloading {} LoRAs ({:.0}%)", loras.len(), prog * 100.0)
                };
                Some((prog, all_paused, summary))
            }
        }
    }
}

impl DownloadPaneState {
    pub fn toggle(&mut self, _sh: &Shared) {
        if self.is_expanded {
            self.is_expanded = false;
            self.user_opened = false;
        } else {
            self.is_expanded = true;
            self.user_opened = true;
        }
    }

    pub fn is_active(&self, sh: &Shared) -> bool {
        let st = sh.state.lock().unwrap();
        let has_model_dl = sh.model_dl_progress.lock().unwrap().is_some()
            || !sh.profile_downloads.lock().unwrap().is_empty()
            || !sh.lora_downloads.lock().unwrap().is_empty();
        let in_dl_state = matches!(
            *st,
            EngState::AwaitingDownloadConfirmation { .. }
                | EngState::DownloadingEngine { .. }
                | EngState::ExtractingEngine { .. }
                | EngState::DownloadingModels { .. }
                | EngState::ModelDownloadPaused { .. }
                | EngState::ModelDownloadFailed { .. }
                | EngState::RepairingEngine { .. }
        ) || sh.download_paused.load(Ordering::SeqCst) || has_model_dl;

        in_dl_state || (self.user_opened && self.is_expanded)
    }

    pub fn show(&mut self, ctx: &egui::Context, sh: &Arc<Shared>) {
        self.show_inner(ctx, sh, None);
    }

    pub fn show_combined(&mut self, ctx: &egui::Context, sh: &Arc<Shared>, updater: &Arc<Mutex<UpdaterState>>) {
        self.show_inner(ctx, sh, Some(updater));
    }

    fn show_inner(&mut self, ctx: &egui::Context, sh: &Arc<Shared>, updater: Option<&Arc<Mutex<UpdaterState>>>) {
        // 1. Check for metered connection switch notification
        if sh.metered_switch_detected.swap(false, Ordering::SeqCst) {
            self.show_metered_dialog = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::RequestUserAttention(egui::UserAttentionType::Critical));
            crate::engine::flash_taskbar();
        }

        // 2. Render metered confirmation modal dialog if triggered
        if self.show_metered_dialog {
            self.render_metered_modal(ctx, sh);
        }

        // 3. Only render the dropdown floater when expanded (minimized state lives in the top-right update icon)
        if !self.is_expanded {
            return;
        }

        let st = sh.state.lock().unwrap().clone();

        // Top-right dropdown floater anchored directly below the update/download icon
        egui::Area::new(egui::Id::new("floating_download_pane"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::RIGHT_TOP, vec2(-16.0, 48.0))
            .constrain(true)
            .show(ctx, |ui| {
                self.render_expanded_pane(ui, ctx, sh, &st, updater);
            });
    }

    #[allow(dead_code)]
    fn render_mini_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, sh: &Arc<Shared>, st: &EngState) {
        let w = 360.0f32;
        let h = 54.0f32;
        let (rect, resp) = ui.allocate_exact_size(vec2(w, h), Sense::click());

        let is_highlighted = self.highlight_until.map(|t| Instant::now() < t).unwrap_or(false);
        let border_stroke = if is_highlighted {
            let t = (Instant::now().elapsed().as_secs_f32() * 6.0).sin().abs();
            let r = (99.0 + (AMBER.r() as f32 - 99.0) * t) as u8;
            let g = (102.0 + (AMBER.g() as f32 - 102.0) * t) as u8;
            let b = (241.0 + (AMBER.b() as f32 - 241.0) * t) as u8;
            ctx.request_repaint();
            Stroke::new(2.0, Color32::from_rgb(r, g, b))
        } else if resp.hovered() {
            Stroke::new(1.0, ACCENT)
        } else {
            Stroke::new(1.0, LINE)
        };

        let p = ui.painter_at(rect);
        p.rect_filled(rect, 8, CARD);
        p.rect_stroke(rect, 8, border_stroke, StrokeKind::Inside);

        let paused = matches!(st, EngState::ModelDownloadPaused { .. }) || sh.download_paused.load(Ordering::SeqCst);
        let dl_prog = sh.model_dl_progress.lock().unwrap().clone();

        // Data extraction
        let (file_title, size_speed, progress) = match st {
            EngState::DownloadingModels { current_file, file_progress, bytes_done, bytes_total, speed_mb, .. } => {
                let size_str = if paused {
                    format!("{} / {} (Paused)", engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total))
                } else {
                    format!("{} / {} • {:.1} MB/s", engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total), speed_mb)
                };
                (current_file.clone(), size_str, *file_progress)
            }
            EngState::ModelDownloadPaused { current_file, file_progress, bytes_done, bytes_total, .. } => {
                let size_str = format!("{} / {} (Paused)", engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total));
                (current_file.clone(), size_str, *file_progress)
            }
            EngState::DownloadingEngine { step, progress, bytes_done, bytes_total, speed_mb } => {
                let size_str = if paused {
                    format!("{} / {} (Paused)", engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total))
                } else {
                    format!("{} / {} • {:.1} MB/s", engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total), speed_mb)
                };
                (step.clone(), size_str, *progress)
            }
            EngState::ExtractingEngine { step } => {
                if let Some(ref mp) = dl_prog {
                    let size_str = if paused {
                        format!("{} / {} (Paused)", engine::format_bytes(mp.bytes_done), engine::format_bytes(mp.bytes_total))
                    } else {
                        format!("{} / {} • {:.1} MB/s", engine::format_bytes(mp.bytes_done), engine::format_bytes(mp.bytes_total), mp.speed_mb)
                    };
                    (format!("Extracting • {}", mp.current_file), size_str, mp.file_progress)
                } else {
                    (step.clone(), "Extracting runtime".to_string(), 0.5)
                }
            }
            EngState::AwaitingDownloadConfirmation { metered } => {
                let label = if *metered { "Metered Connection" } else { "Download Confirmation" };
                ("Download Setup Required".to_string(), label.to_string(), 0.0)
            }
            EngState::ModelDownloadFailed { current_file, .. } => {
                (current_file.clone(), "Download Interrupted".to_string(), 0.0)
            }
            EngState::RepairingEngine { step } => {
                if let Some(ref mp) = dl_prog {
                    let size_str = if paused {
                        format!("{} / {} (Paused)", engine::format_bytes(mp.bytes_done), engine::format_bytes(mp.bytes_total))
                    } else {
                        format!("{} / {} • {:.1} MB/s", engine::format_bytes(mp.bytes_done), engine::format_bytes(mp.bytes_total), mp.speed_mb)
                    };
                    (format!("Repairing • {}", mp.current_file), size_str, mp.file_progress)
                } else {
                    (step.clone(), "Repairing".to_string(), 0.5)
                }
            }
            EngState::Ready { .. } | EngState::Starting { .. } | EngState::Detecting => {
                if let Some(ref mp) = dl_prog {
                    let size_str = if paused {
                        format!("{} / {} (Paused)", engine::format_bytes(mp.bytes_done), engine::format_bytes(mp.bytes_total))
                    } else {
                        format!("{} / {} • {:.1} MB/s", engine::format_bytes(mp.bytes_done), engine::format_bytes(mp.bytes_total), mp.speed_mb)
                    };
                    (mp.current_file.clone(), size_str, mp.file_progress)
                } else {
                    let loras: Vec<crate::lora::LoraDownloadStatus> = sh
                        .lora_downloads
                        .lock()
                        .unwrap()
                        .values()
                        .filter(|d| !d.completed)
                        .cloned()
                        .collect();
                    if !loras.is_empty() {
                        let done: u64 = loras.iter().map(|l| l.downloaded).sum();
                        let tot: u64 = loras.iter().map(|l| l.total).sum();
                        let speed: f32 = loras.iter().filter(|l| !l.paused && l.error.is_none()).map(|l| l.speed_mb).sum();
                        let prog = if tot > 0 { (done as f32 / tot as f32).clamp(0.0, 1.0) } else { 0.0 };
                        let title = if loras.len() == 1 {
                            format!("LoRA: {}", loras[0].display_name)
                        } else {
                            format!("{} LoRA downloads", loras.len())
                        };
                        let all_paused = loras.iter().all(|l| l.paused);
                        let size_str = if all_paused {
                            format!("{} / {} (Paused)", engine::format_bytes(done), engine::format_bytes(tot))
                        } else {
                            format!("{} / {} • {:.1} MB/s", engine::format_bytes(done), engine::format_bytes(tot), speed)
                        };
                        (title, size_str, prog)
                    } else if matches!(st, EngState::Ready { .. }) {
                        ("All models ready".to_string(), "Ready".to_string(), 1.0)
                    } else {
                        ("Engine Setup".to_string(), "Waiting".to_string(), 0.0)
                    }
                }
            }
            _ => ("Engine Setup".to_string(), "Waiting".to_string(), 0.0),
        };

        // Top line: File title on left, size/speed on right
        let title_area = Rect::from_min_max(rect.min + vec2(10.0, 7.0), rect.min + vec2(w - 10.0, 23.0));
        let title_font = egui::FontId::proportional(12.0);
        let right_font = egui::FontId::proportional(11.0);

        // Right text
        let right_galley = p.layout_no_wrap(size_speed, right_font, if paused { AMBER } else { DIM });
        let right_pos = egui::pos2(title_area.max.x - right_galley.size().x, title_area.min.y);
        p.galley(right_pos, right_galley, Color32::WHITE);

        // Truncate title to fit remaining width
        let max_title_w = (right_pos.x - title_area.min.x - 8.0).max(40.0);
        let fit_title = fit_text_ellipsis(&p, &file_title, title_font.clone(), max_title_w);
        p.text(title_area.min, egui::Align2::LEFT_TOP, fit_title, title_font, TEXT);

        // Bottom body: Progress bar on left, Pause/Resume button on right
        let btn_size = 22.0f32;
        let btn_rect = Rect::from_min_size(rect.min + vec2(w - 10.0 - btn_size, 26.0), vec2(btn_size, btn_size));

        let bar_rect = Rect::from_min_max(rect.min + vec2(10.0, 33.0), egui::pos2(btn_rect.min.x - 10.0, rect.min.y + 40.0));
        p.rect_filled(bar_rect, 3, CARD_HI);
        let fill_w = (bar_rect.width() * progress.clamp(0.0, 1.0)).max(0.0);
        if fill_w > 0.0 {
            let fill_rect = Rect::from_min_size(bar_rect.min, vec2(fill_w, bar_rect.height()));
            p.rect_filled(fill_rect, 3, if paused { AMBER } else { ACCENT });
        }

        // Pause/Resume button
        let btn_resp = ui.interact(btn_rect, ui.id().with("btn_mini_pause"), Sense::click());
        let btn_hovered = btn_resp.hovered();
        p.rect_filled(btn_rect, 5, if btn_hovered { CARD_HI } else { CARD });
        p.rect_stroke(btn_rect, 5, Stroke::new(1.0, if btn_hovered { ACCENT } else { LINE }), StrokeKind::Inside);

        let icon_col = if btn_hovered { Color32::WHITE } else { TEXT };
        let btn_resp = if paused {
            draw_play_icon(&p, btn_rect.center(), icon_col);
            btn_resp.on_hover_text("Resume download")
        } else {
            draw_pause_icon(&p, btn_rect.center(), icon_col);
            btn_resp.on_hover_text("Pause download")
        };

        if btn_resp.clicked() {
            if paused {
                sh.download_paused.store(false, Ordering::SeqCst);
                sh.retry.store(true, Ordering::SeqCst);
            } else {
                sh.download_paused.store(true, Ordering::SeqCst);
            }
        } else if resp.clicked() {
            // Click outside button expands floating pane
            self.is_expanded = true;
        }

        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text("Click to expand download manager");
    }

    fn render_expanded_pane(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        sh: &Arc<Shared>,
        st: &EngState,
        updater: Option<&Arc<Mutex<UpdaterState>>>,
    ) {
        let card_w = 460.0f32;
        let card_frame = egui::Frame::new()
            .fill(CARD)
            .stroke(Stroke::new(1.0, LINE))
            .corner_radius(10)
            .inner_margin(egui::Margin::symmetric(14, 12));

        card_frame.show(ui, |ui| {
            ui.set_width(card_w);

            // 1. Header with title, speed, and minimize button
            ui.horizontal(|ui| {
                ui.label(RichText::new("Updates & Downloads").size(14.0).strong().color(TEXT));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Minimize button
                    let (min_rect, min_resp) = ui.allocate_exact_size(vec2(22.0, 20.0), Sense::click());
                    let p = ui.painter_at(min_rect);
                    if min_resp.hovered() {
                        p.rect_filled(min_rect, 4, CARD_HI);
                    }
                    draw_minimize_icon(&p, min_rect.center(), if min_resp.hovered() { Color32::WHITE } else { DIM });
                    if min_resp.on_hover_text("Minimize to download bar").clicked() {
                        self.is_expanded = false;
                        self.user_opened = false;
                    }

                    // Speed badge
                    let is_paused = matches!(st, EngState::ModelDownloadPaused { .. }) || sh.download_paused.load(Ordering::SeqCst);
                    let dl_prog_hdr = sh.model_dl_progress.lock().unwrap().clone();
                    if !is_paused {
                        if let EngState::DownloadingModels { speed_mb, .. } | EngState::DownloadingEngine { speed_mb, .. } = st {
                            if *speed_mb > 0.05 {
                                ui.label(RichText::new(format!("{:.1} MB/s", speed_mb)).size(12.0).color(ACCENT).strong());
                            }
                        } else if let Some(ref mp) = dl_prog_hdr {
                            if mp.speed_mb > 0.05 {
                                ui.label(RichText::new(format!("{:.1} MB/s", mp.speed_mb)).size(12.0).color(ACCENT).strong());
                            }
                        }
                    }
                });
            });

            ui.add_space(6.0);
            ui.separator();
            ui.add_space(6.0);

            // 1b. App Update section (combined with Download Manager)
            if let Some(upd) = updater {
                let (up_avail, latest, is_dl, dl_prog, applied, checked, up_err) = {
                    let u = upd.lock().unwrap();
                    (
                        u.update_available,
                        u.latest_version.clone(),
                        u.downloading,
                        u.download_progress,
                        u.update_applied,
                        u.checked,
                        u.error.clone(),
                    )
                };
                let cur_ver = env!("CARGO_PKG_VERSION");
                ui.horizontal(|ui| {
                    ui.label(RichText::new("APP UPDATE").size(11.0).strong().color(DIM));
                    ui.label(RichText::new(format!("v{cur_ver}")).size(11.0).color(DIM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if applied {
                            if ui.button(RichText::new("Restart Now").size(11.5).strong().color(Color32::WHITE)).clicked() {
                                updater::restart_application();
                            }
                        } else if is_dl {
                            ui.label(RichText::new(format!("{:.0}%", dl_prog * 100.0)).size(11.5).color(GREEN).strong());
                        } else if up_avail {
                            if ui.button(RichText::new(format!("Update to v{latest}")).size(11.5).strong().color(GREEN)).clicked() {
                                updater::download_and_install_update(upd.clone(), ctx.clone());
                            }
                        } else if ui.small_button("Check for Updates").clicked() {
                            updater::check_for_updates(upd.clone(), ctx.clone());
                        }
                    });
                });
                if is_dl {
                    ui.add_space(3.0);
                    let (bar_r, _) = ui.allocate_exact_size(vec2(ui.available_width(), 5.0), Sense::hover());
                    let p = ui.painter();
                    p.rect_filled(bar_r, 2.5, CARD_HI);
                    let fw = (bar_r.width() * dl_prog.clamp(0.0, 1.0)).max(0.0);
                    if fw > 0.0 {
                        p.rect_filled(Rect::from_min_size(bar_r.min, vec2(fw, bar_r.height())), 2.5, GREEN);
                    }
                } else if applied {
                    ui.label(RichText::new("Update downloaded. Restart Image Studio to apply.").size(10.5).color(GREEN));
                } else if let Some(err) = up_err {
                    ui.label(RichText::new(format!("Update check error: {err}")).size(10.5).color(AMBER));
                } else if checked && !up_avail {
                    ui.label(RichText::new("Image Studio is up to date.").size(10.5).color(DIM));
                }
                ui.add_space(6.0);
                ui.separator();
                ui.add_space(6.0);
            }

            // 2. Active Download Status
            let is_paused = matches!(st, EngState::ModelDownloadPaused { .. }) || sh.download_paused.load(Ordering::SeqCst);
            let dl_prog = sh.model_dl_progress.lock().unwrap().clone();
            match st {
                EngState::DownloadingModels { current_file, file_index, total_files, file_progress, bytes_done, bytes_total, speed_mb, overall_progress } if !is_paused => {
                    Self::render_model_progress_block(ui, current_file, *file_index, *total_files, *file_progress, *bytes_done, *bytes_total, *speed_mb, *overall_progress);
                }
                EngState::DownloadingModels { current_file, file_index, total_files, file_progress, bytes_done, bytes_total, overall_progress, .. }
                | EngState::ModelDownloadPaused { current_file, file_index, total_files, file_progress, bytes_done, bytes_total, overall_progress } => {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("DOWNLOAD PAUSED").size(11.0).color(AMBER).strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(format!("{}/{}: {:.1}%", file_index, total_files, file_progress * 100.0)).size(11.0).color(DIM));
                        });
                    });
                    ui.label(RichText::new(current_file).size(12.5).color(TEXT).strong());
                    ui.add_space(4.0);

                    let bar = egui::ProgressBar::new(*file_progress).fill(AMBER);
                    ui.add_sized([ui.available_width(), 10.0], bar);
                    ui.add_space(4.0);

                    ui.label(RichText::new(format!("{} / {}", engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total))).size(11.0).color(DIM));
                    if *total_files > 1 {
                        ui.add_space(4.0);
                        let ov_bar = egui::ProgressBar::new(*overall_progress).fill(AMBER.linear_multiply(0.5));
                        ui.add_sized([ui.available_width(), 8.0], ov_bar);
                    }
                }
                EngState::DownloadingEngine { step, progress, bytes_done, bytes_total, speed_mb } => {
                    ui.label(RichText::new("Downloading AI Engine").size(12.0).color(ACCENT).strong());
                    ui.label(RichText::new(step).size(12.0).color(TEXT));
                    ui.add_space(4.0);
                    let bar = egui::ProgressBar::new(*progress).fill(ACCENT);
                    ui.add_sized([ui.available_width(), 10.0], bar);
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{} / {}", engine::format_bytes(*bytes_done), engine::format_bytes(*bytes_total))).size(11.0).color(DIM));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(format!("{:.1} MB/s", speed_mb)).size(11.0).color(DIM));
                        });
                    });
                }
                EngState::ExtractingEngine { step } => {
                    ui.label(RichText::new("Extracting Engine").size(12.0).color(ACCENT).strong());
                    ui.label(RichText::new(step).size(11.5).color(DIM));
                    ui.add_space(4.0);
                    let bar = egui::ProgressBar::new(0.5).fill(ACCENT).animate(true);
                    ui.add_sized([ui.available_width(), 10.0], bar);
                    if let Some(ref mp) = dl_prog {
                        if !is_paused {
                            ui.add_space(10.0);
                            Self::render_model_progress_block(ui, &mp.current_file, mp.file_index, mp.total_files, mp.file_progress, mp.bytes_done, mp.bytes_total, mp.speed_mb, mp.overall_progress);
                        }
                    }
                }
                EngState::AwaitingDownloadConfirmation { metered } => {
                    let title = if *metered { "Metered Connection Detected" } else { "Download Confirmation Required" };
                    ui.label(RichText::new(title).size(13.0).color(AMBER).strong());
                    let desc = if *metered {
                        "A metered network connection was detected. Automatic downloads are paused to prevent unexpected data usage. Confirm storage path and model selection below to begin."
                    } else {
                        "Confirm storage path and model selection below to begin downloading."
                    };
                    ui.label(RichText::new(desc).size(11.5).color(DIM));
                    ui.add_space(8.0);
                }
                EngState::ModelDownloadFailed { current_file, error } => {
                    ui.label(RichText::new("Download Interrupted").size(12.5).color(RED).strong());
                    ui.label(RichText::new(format!("File: {current_file}")).size(11.5).color(TEXT));
                    ui.label(RichText::new(error).size(11.0).color(DIM));
                }
                EngState::RepairingEngine { step } => {
                    ui.label(RichText::new("Repairing Environment").size(12.0).color(AMBER).strong());
                    ui.label(RichText::new(step).size(11.5).color(DIM));
                    if let Some(ref mp) = dl_prog {
                        if !is_paused {
                            ui.add_space(10.0);
                            Self::render_model_progress_block(ui, &mp.current_file, mp.file_index, mp.total_files, mp.file_progress, mp.bytes_done, mp.bytes_total, mp.speed_mb, mp.overall_progress);
                        }
                    }
                }
                EngState::Failed(msg) => {
                    ui.label(RichText::new("Engine Startup Failed").size(12.5).color(RED).strong());
                    ui.label(RichText::new(msg).size(11.0).color(DIM));
                }
                _ => {
                    let has_profile_dls = !sh.profile_downloads.lock().unwrap().is_empty();
                    if let Some(ref mp) = dl_prog {
                        if !is_paused {
                            Self::render_model_progress_block(ui, &mp.current_file, mp.file_index, mp.total_files, mp.file_progress, mp.bytes_done, mp.bytes_total, mp.speed_mb, mp.overall_progress);
                        } else {
                            ui.label(RichText::new("DOWNLOAD PAUSED").size(11.0).color(AMBER).strong());
                            ui.label(RichText::new(&mp.current_file).size(12.5).color(TEXT).strong());
                        }
                    } else if has_profile_dls {
                        // Rendered in the active model variant downloads section below
                    } else if !sh.dl_image_model.load(Ordering::SeqCst)
                        && !sh.dl_video_model.load(Ordering::SeqCst)
                        && !sh.dl_r2v_model.load(Ordering::SeqCst)
                    {
                        ui.label(RichText::new("No model downloads selected.").size(12.0).color(DIM));
                    } else {
                        ui.label(RichText::new("All models ready.").size(12.0).color(GREEN));
                    }
                }
            }

            // 2a. Active / Paused / Failed Model Quant Downloads (from Model Picker)
            let mut profile_dl_list: Vec<(String, String, crate::engine::ProfileDownloadStatus, f32, u64, u64, f32, String)> = sh
                .profile_downloads
                .lock()
                .unwrap()
                .iter()
                .map(|(id, t)| {
                    (
                        id.clone(),
                        t.profile.name.clone(),
                        t.status.clone(),
                        t.progress,
                        t.bytes_done,
                        t.bytes_total,
                        t.speed_mb,
                        t.current_file.clone(),
                    )
                })
                .collect();
            profile_dl_list.sort_by(|a, b| a.1.cmp(&b.1));
            if !profile_dl_list.is_empty() {
                ui.add_space(6.0);
                ui.label(RichText::new("MODEL QUANT DOWNLOADS").size(11.0).strong().color(DIM));
                ui.add_space(4.0);
                for (pid, pname, pstatus, pprog, pdone, ptotal, pspeed, pfile) in &profile_dl_list {
                    let is_err = matches!(pstatus, crate::engine::ProfileDownloadStatus::Failed(_));
                    let is_p_paused = *pstatus == crate::engine::ProfileDownloadStatus::Paused;
                    egui::Frame::NONE
                        .fill(Color32::from_rgb(20, 24, 34))
                        .stroke(Stroke::new(1.0, LINE))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::symmetric(8, 6))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(pname).size(11.5).strong().color(TEXT));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("Cancel").on_hover_text("Cancel model download").clicked() {
                                        crate::engine::cancel_profile_download(sh, pid);
                                    }
                                    if is_err || is_p_paused {
                                        let lbl = if is_err { "Retry" } else { "Resume" };
                                        if ui.small_button(lbl).clicked() {
                                            crate::engine::toggle_profile_download_pause(sh, pid, ctx);
                                        }
                                    } else if ui.small_button("Pause").clicked() {
                                        crate::engine::toggle_profile_download_pause(sh, pid, ctx);
                                    }
                                });
                            });
                            if !pfile.is_empty() {
                                ui.label(RichText::new(pfile).size(10.5).color(DIM));
                            }
                            let bar_col = if is_err {
                                RED
                            } else if is_p_paused {
                                AMBER
                            } else {
                                ACCENT
                            };
                            let bar = egui::ProgressBar::new(pprog.clamp(0.0, 1.0)).fill(bar_col);
                            ui.add_sized([ui.available_width(), 6.0], bar);
                            let status_txt = match pstatus {
                                crate::engine::ProfileDownloadStatus::Failed(err) => format!("Failed: {err}"),
                                crate::engine::ProfileDownloadStatus::Paused => format!(
                                    "Paused — {} / {} ({:.0}%)",
                                    crate::engine::format_bytes(*pdone),
                                    crate::engine::format_bytes(*ptotal),
                                    pprog * 100.0
                                ),
                                crate::engine::ProfileDownloadStatus::Downloading => format!(
                                    "{} / {} ({:.0}%) — {:.1} MB/s",
                                    crate::engine::format_bytes(*pdone),
                                    crate::engine::format_bytes(*ptotal),
                                    pprog * 100.0,
                                    pspeed
                                ),
                            };
                            ui.label(
                                RichText::new(status_txt)
                                    .size(10.5)
                                    .color(if is_err { RED } else { DIM }),
                            );
                        });
                    ui.add_space(4.0);
                }
            }

            // 2b. Active / Paused / Failed LoRA Downloads
            let mut lora_list: Vec<crate::lora::LoraDownloadStatus> = sh
                .lora_downloads
                .lock()
                .unwrap()
                .values()
                .filter(|d| !d.completed)
                .cloned()
                .collect();
            lora_list.sort_by(|a, b| a.display_name.cmp(&b.display_name));
            if !lora_list.is_empty() {
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(6.0);
                ui.label(RichText::new("LORA DOWNLOADS").size(11.0).strong().color(DIM));
                ui.add_space(4.0);
                for ld in &lora_list {
                    let prog = if ld.total > 0 {
                        (ld.downloaded as f32 / ld.total as f32).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    egui::Frame::NONE
                        .fill(Color32::from_rgb(20, 24, 34))
                        .stroke(Stroke::new(1.0, LINE))
                        .corner_radius(6)
                        .inner_margin(egui::Margin::symmetric(8, 6))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(&ld.display_name).size(11.5).strong().color(TEXT));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if ui.small_button("Cancel").on_hover_text("Cancel download & delete partial file").clicked() {
                                        let _ = crate::lora::cancel_and_delete_lora(sh, &ld.loras_dir, &ld.filename);
                                    }
                                    if ld.error.is_some() {
                                        if ui.small_button("Retry").clicked() {
                                            crate::lora::resume_lora_download(sh, &ld.filename, ctx);
                                        }
                                    } else if ld.paused {
                                        if ui.small_button("Resume").clicked() {
                                            crate::lora::resume_lora_download(sh, &ld.filename, ctx);
                                        }
                                    } else if ui.small_button("Pause").clicked() {
                                        crate::lora::pause_lora_download(sh, &ld.filename);
                                    }
                                });
                            });
                            let bar_col = if ld.error.is_some() {
                                RED
                            } else if ld.paused {
                                AMBER
                            } else {
                                ACCENT
                            };
                            ui.add_sized([ui.available_width(), 6.0], egui::ProgressBar::new(prog).fill(bar_col));
                            ui.horizontal(|ui| {
                                let status_txt = if let Some(ref err) = ld.error {
                                    format!("Failed: {err}")
                                } else if ld.paused {
                                    format!("Paused • {} / {} ({:.0}%)", engine::format_bytes(ld.downloaded), engine::format_bytes(ld.total), prog * 100.0)
                                } else {
                                    format!("{} / {} ({:.0}%) • {:.1} MB/s", engine::format_bytes(ld.downloaded), engine::format_bytes(ld.total), prog * 100.0, ld.speed_mb)
                                };
                                let col = if ld.error.is_some() { RED } else if ld.paused { AMBER } else { DIM };
                                ui.label(RichText::new(status_txt).size(10.5).color(col));
                            });
                        });
                    ui.add_space(4.0);
                }
            }

            ui.add_space(10.0);
            ui.separator();
            ui.add_space(8.0);

            // 3. Storage path selection row
            self.render_storage_location_row(ui, sh);
            ui.add_space(6.0);

            // 4. Model selection checkboxes
            self.render_model_checkboxes(ui, sh);
            ui.add_space(10.0);

            // 5. Action Controls (Pause / Resume / Retry / Begin Download)
            let is_paused = matches!(st, EngState::ModelDownloadPaused { .. }) || sh.download_paused.load(Ordering::SeqCst);
            let is_downloading = matches!(st, EngState::DownloadingModels { .. } | EngState::DownloadingEngine { .. })
                || dl_prog.is_some()
                || sh.model_dl_running.load(Ordering::SeqCst);
            let is_dl_failed = matches!(st, EngState::ModelDownloadFailed { .. });
            let is_eng_failed = matches!(st, EngState::Failed(_));
            let is_awaiting = matches!(st, EngState::AwaitingDownloadConfirmation { .. });

            ui.horizontal(|ui| {
                if is_awaiting {
                    let start_btn = egui::Button::new(RichText::new("Begin Download").size(12.5).strong().color(Color32::WHITE))
                        .fill(ACCENT)
                        .min_size(vec2(130.0, 28.0));
                    if ui.add(start_btn).clicked() {
                        sh.download_confirmed.store(true, Ordering::SeqCst);
                        sh.retry.store(true, Ordering::SeqCst);
                    }
                } else if is_paused {
                    let resume_btn = egui::Button::new(RichText::new("Resume Download").size(12.0).strong().color(Color32::WHITE))
                        .fill(ACCENT)
                        .min_size(vec2(130.0, 28.0));
                    if ui.add(resume_btn).clicked() {
                        sh.download_paused.store(false, Ordering::SeqCst);
                        crate::engine::spawn_model_downloads_if_needed(sh, ui.ctx());
                        if matches!(st, EngState::DownloadingEngine { .. } | EngState::AwaitingDownloadConfirmation { .. } | EngState::ModelDownloadPaused { .. }) {
                            sh.retry.store(true, Ordering::SeqCst);
                        }
                    }
                } else if is_downloading {
                    let pause_btn = egui::Button::new(RichText::new("Pause Download").size(12.0).strong().color(TEXT))
                        .fill(CARD_HI)
                        .stroke(Stroke::new(1.0, LINE))
                        .min_size(vec2(120.0, 28.0));
                    if ui.add(pause_btn).clicked() {
                        sh.download_paused.store(true, Ordering::SeqCst);
                    }
                } else if is_dl_failed {
                    let retry_btn = egui::Button::new(RichText::new("Retry Download").size(12.0).strong().color(Color32::WHITE))
                        .fill(ACCENT)
                        .min_size(vec2(130.0, 28.0));
                    if ui.add(retry_btn).clicked() {
                        sh.retry.store(true, Ordering::SeqCst);
                        crate::engine::spawn_model_downloads_if_needed(sh, ui.ctx());
                    }
                } else if is_eng_failed {
                    let repair_btn = egui::Button::new(RichText::new("Repair Dependencies & Retry").size(12.0).strong().color(Color32::WHITE))
                        .fill(ACCENT)
                        .min_size(vec2(180.0, 28.0));
                    if ui.add(repair_btn).clicked() {
                        crate::engine::repair_environment(sh.clone(), ui.ctx().clone());
                    }
                }
            });

            ui.add_space(8.0);

            // 6. Model Queue List
            let missing = Profile::missing_models(sh);
            if !missing.is_empty() {
                ui.separator();
                ui.add_space(6.0);
                ui.label(RichText::new("DOWNLOAD QUEUE").size(10.5).color(DIM).strong());
                ui.add_space(4.0);

                egui::ScrollArea::vertical().max_height(100.0).show(ui, |ui| {
                    let cur_active = dl_prog
                        .as_ref()
                        .map(|p| p.current_file.clone())
                        .unwrap_or_else(|| match st {
                            EngState::DownloadingModels { current_file, .. } => current_file.clone(),
                            EngState::ModelDownloadPaused { current_file, .. } => current_file.clone(),
                            _ => String::new(),
                        });

                    for m in &missing {
                        let is_active = m.filename == cur_active;
                        let (badge, col) = if is_active && is_paused {
                            ("PAUSED", AMBER)
                        } else if is_active {
                            ("DOWNLOADING", AMBER)
                        } else {
                            ("QUEUED", DIM)
                        };

                        ui.horizontal(|ui| {
                            let fname = fit_text_ellipsis(ui.painter(), &m.filename, egui::FontId::proportional(11.0), 280.0);
                            ui.label(RichText::new(fname).size(11.0).color(if is_active { TEXT } else { DIM }));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(badge).size(10.0).color(col).strong());
                                ui.label(RichText::new(engine::format_bytes(m.size_bytes)).size(10.0).color(DIM));
                            });
                        });
                    }
                });
            }
        });
    }

    fn render_storage_location_row(&mut self, ui: &mut egui::Ui, sh: &Arc<Shared>) {
        let cur_loc = sh
            .pending_models_dir
            .lock()
            .unwrap()
            .clone()
            .unwrap_or_else(|| sh.paths.lock().unwrap().models.clone());
        let display_str = cur_loc.display().to_string();

        ui.horizontal(|ui| {
            ui.label(RichText::new("Storage:").size(12.0).color(DIM));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let change_btn = ui.button(RichText::new("Change Location...").size(11.5))
                    .on_hover_text("Change model folder to another drive (e.g. D:\\ImageStudio).");
                if change_btn.clicked() {
                    if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                        let target_models = folder.join("models");
                        *sh.pending_models_dir.lock().unwrap() = Some(target_models.clone());
                        let mut cfg = sh.cfg.lock().unwrap();
                        cfg.models_dir = Some(target_models.clone());
                        cfg.save();
                        sh.paths.lock().unwrap().models = target_models;
                    }
                }
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    let avail = ui.available_width();
                    let fit = fit_text_ellipsis(ui.painter(), &display_str, egui::FontId::proportional(11.5), avail);
                    ui.label(RichText::new(fit).size(11.5).color(TEXT).strong())
                        .on_hover_text(&display_str);
                });
            });
        });
    }

    fn render_model_progress_block(
        ui: &mut egui::Ui,
        current_file: &str,
        file_index: usize,
        total_files: usize,
        file_progress: f32,
        bytes_done: u64,
        bytes_total: u64,
        speed_mb: f32,
        overall_progress: f32,
    ) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Downloading ({}/{})", file_index, total_files)).size(12.0).color(ACCENT).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(format!("{:.1}%", file_progress * 100.0)).size(12.0).color(AMBER).strong());
            });
        });
        ui.label(RichText::new(current_file).size(12.5).color(TEXT).strong());
        ui.add_space(4.0);

        let bar = egui::ProgressBar::new(file_progress).fill(ACCENT);
        ui.add_sized([ui.available_width(), 10.0], bar);
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{} / {}", engine::format_bytes(bytes_done), engine::format_bytes(bytes_total))).size(11.0).color(DIM));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(format!("{:.1} MB/s", speed_mb)).size(11.0).color(DIM));
            });
        });

        if total_files > 1 {
            ui.add_space(6.0);
            ui.label(RichText::new(format!("Overall Progress: {:.0}%", overall_progress * 100.0)).size(11.0).color(DIM));
            let ov_bar = egui::ProgressBar::new(overall_progress).fill(ACCENT.linear_multiply(0.6));
            ui.add_sized([ui.available_width(), 8.0], ov_bar);
        }
    }

    fn render_model_checkboxes(&mut self, ui: &mut egui::Ui, sh: &Arc<Shared>) {
        let mut dl_img = sh.dl_image_model.load(Ordering::SeqCst);
        let mut dl_vid = sh.dl_video_model.load(Ordering::SeqCst);
        let vid_prof = sh.active_video_profile.lock().unwrap().clone();
        let vid_gb = vid_prof.total_size_gb();
        let vid_short = if vid_prof.is_minimax_h3() {
            "MiniMax H3"
        } else if vid_prof.is_ltx25() {
            "LTX-2.5"
        } else {
            "LTX-2.3"
        };

        ui.horizontal(|ui| {
            let resp_img = ui.checkbox(&mut dl_img, RichText::new("Image Model (Qwen 2.1 - ~16 GB)").size(11.5));
            if resp_img.changed() {
                engine::toggle_model_download(sh, ui.ctx(), true, dl_img);
            }

            ui.add_space(12.0);

            let resp_vid = ui.checkbox(&mut dl_vid, RichText::new(format!("Video Model ({vid_short} - ~{vid_gb:.0} GB)")).size(11.5));
            if resp_vid.changed() {
                engine::toggle_model_download(sh, ui.ctx(), false, dl_vid);
            }
        });
    }

    fn render_metered_modal(&mut self, ctx: &egui::Context, sh: &Arc<Shared>) {
        // Modal backdrop overlay
        let screen = ctx.content_rect();
        let bg_p = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, egui::Id::new("metered_backdrop")));
        bg_p.rect_filled(screen, 0, Color32::from_rgba_unmultiplied(0, 0, 0, 160));

        egui::Window::new("Metered Connection Detected")
            .collapsible(false)
            .resizable(false)
            .fixed_size(vec2(420.0, 160.0))
            .anchor(egui::Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.label(RichText::new("A metered network connection was detected.").size(14.0).strong().color(TEXT));
                ui.add_space(6.0);
                ui.label(RichText::new("Your connection switched to a metered network. Downloads have been paused to prevent unexpected data charges.").size(12.0).color(DIM));
                ui.add_space(14.0);

                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let pause_btn = egui::Button::new(RichText::new("Pause").strong().color(TEXT))
                            .fill(CARD_HI)
                            .stroke(Stroke::new(1.0, LINE))
                            .min_size(vec2(90.0, 30.0));
                        if ui.add(pause_btn).clicked() {
                            sh.download_paused.store(true, Ordering::SeqCst);
                            self.show_metered_dialog = false;
                            self.highlight_until = Some(Instant::now() + Duration::from_secs(4));
                            ctx.request_repaint();
                        }

                        ui.add_space(8.0);

                        let cont_btn = egui::Button::new(RichText::new("Continue").strong().color(Color32::WHITE))
                            .fill(ACCENT)
                            .stroke(Stroke::new(1.0, ACCENT_GLOW))
                            .min_size(vec2(100.0, 30.0));
                        if ui.add(cont_btn).clicked() {
                            sh.metered_allowed.store(true, Ordering::SeqCst);
                            sh.download_paused.store(false, Ordering::SeqCst);
                            sh.retry.store(true, Ordering::SeqCst);
                            self.show_metered_dialog = false;
                            ctx.request_repaint();
                        }
                    });
                });
            });
    }
}

fn fit_text_ellipsis(p: &egui::Painter, text: &str, font: egui::FontId, max_w: f32) -> String {
    let g = p.layout_no_wrap(text.to_string(), font.clone(), Color32::WHITE);
    if g.size().x <= max_w {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    let mut low = 0;
    let mut high = chars.len();
    let mut best = text.to_string();
    while low <= high {
        let mid = (low + high) / 2;
        let candidate = format!("{}...", chars.iter().take(mid).collect::<String>());
        let layout = p.layout_no_wrap(candidate.clone(), font.clone(), Color32::WHITE);
        if layout.size().x <= max_w {
            best = candidate;
            low = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            high = mid - 1;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::engine::{EngState, Shared};

    fn make_test_shared(state: EngState) -> Arc<Shared> {
        let sh = Shared::new();
        *sh.state.lock().unwrap() = state;
        sh
    }

    #[test]
    fn test_download_pane_state_default() {
        let state = DownloadPaneState::default();
        assert!(!state.is_expanded);
        assert!(!state.user_opened);
        assert!(!state.show_metered_dialog);
    }

    #[test]
    fn test_download_pane_state_is_active_during_downloads() {
        let mut state = DownloadPaneState::default();
        state.is_expanded = false;

        let sh_downloading = make_test_shared(EngState::DownloadingModels {
            current_file: "model.safetensors".into(),
            file_index: 1,
            total_files: 2,
            file_progress: 0.5,
            bytes_done: 500,
            bytes_total: 1000,
            speed_mb: 25.0,
            overall_progress: 0.25,
        });
        assert!(state.is_active(&sh_downloading));

        let sh_paused = make_test_shared(EngState::ModelDownloadPaused {
            current_file: "model.safetensors".into(),
            file_index: 1,
            total_files: 2,
            file_progress: 0.5,
            bytes_done: 500,
            bytes_total: 1000,
            overall_progress: 0.25,
        });
        assert!(state.is_active(&sh_paused));

        let sh_awaiting = make_test_shared(EngState::AwaitingDownloadConfirmation { metered: false });
        assert!(state.is_active(&sh_awaiting));

        let sh_engine_dl = make_test_shared(EngState::DownloadingEngine {
            step: "downloading".into(),
            progress: 0.1,
            bytes_done: 10,
            bytes_total: 100,
            speed_mb: 5.0,
        });
        assert!(state.is_active(&sh_engine_dl));
    }

    #[test]
    fn test_download_pane_state_inactive_when_ready_and_not_manually_opened() {
        let mut state = DownloadPaneState::default();
        let sh_ready = make_test_shared(EngState::Ready {
            ours: true,
            gpu: "NVIDIA RTX 4070".into(),
        });

        // Even if is_expanded starts true, if not in dl state and user_opened is false, it's inactive
        assert!(!state.is_active(&sh_ready));

        // When user manually opens it, it becomes active
        state.user_opened = true;
        state.is_expanded = true;
        assert!(state.is_active(&sh_ready));

        // When user minimizes it, it becomes inactive again
        state.is_expanded = false;
        state.user_opened = false;
        assert!(!state.is_active(&sh_ready));
    }

    #[test]
    fn test_download_pane_toggle() {
        let mut state = DownloadPaneState::default();
        let sh_ready = make_test_shared(EngState::Ready {
            ours: true,
            gpu: "NVIDIA RTX 4070".into(),
        });

        // Initially inactive when Ready, even though is_expanded starts true
        assert!(!state.is_active(&sh_ready));

        // First toggle opens it immediately
        state.toggle(&sh_ready);
        assert!(state.is_active(&sh_ready));
        assert!(state.is_expanded);
        assert!(state.user_opened);

        // Second toggle closes it
        state.toggle(&sh_ready);
        assert!(!state.is_active(&sh_ready));
        assert!(!state.is_expanded);
        assert!(!state.user_opened);

        // During download, toggling opens and collapses the dropdown
        let sh_downloading = make_test_shared(EngState::DownloadingModels {
            current_file: "model.safetensors".into(),
            file_index: 1,
            total_files: 2,
            file_progress: 0.5,
            bytes_done: 500,
            bytes_total: 1000,
            speed_mb: 25.0,
            overall_progress: 0.25,
        });
        state.is_expanded = true;
        state.toggle(&sh_downloading);
        assert!(state.is_active(&sh_downloading));
        assert!(!state.is_expanded);

        state.toggle(&sh_downloading);
        assert!(state.is_active(&sh_downloading));
        assert!(state.is_expanded);
    }

    #[test]
    fn test_download_pane_active_with_concurrent_model_dl_progress() {
        let state = DownloadPaneState::default();
        let sh_ready = make_test_shared(EngState::Ready {
            ours: true,
            gpu: "NVIDIA RTX 4070".into(),
        });
        assert!(!state.is_active(&sh_ready));

        *sh_ready.model_dl_progress.lock().unwrap() = Some(crate::engine::ModelDownloadProgress {
            current_file: "ltx-2.3.safetensors".into(),
            file_index: 1,
            total_files: 3,
            file_progress: 0.25,
            bytes_done: 250,
            bytes_total: 1000,
            speed_mb: 50.0,
            overall_progress: 0.1,
        });
        assert!(state.is_active(&sh_ready));
    }
}
