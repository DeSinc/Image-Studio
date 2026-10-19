#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod app;
mod comfy;
mod engine;
mod updater;
pub mod video;

fn load_app_icon() -> Option<eframe::egui::IconData> {
    let bytes = include_bytes!("../assets/icon.png");
    let img = image::load_from_memory(bytes).ok()?.into_rgba8();
    let (width, height) = img.dimensions();
    Some(eframe::egui::IconData {
        rgba: img.into_raw(),
        width,
        height,
    })
}

fn main() -> eframe::Result {
    std::panic::set_hook(Box::new(|info| engine::log(&format!("PANIC: {info}"))));
    engine::log(&format!("=== start, build {} ===", env!("BUILD_STAMP")));
    engine::adopt_into_kill_job();

    let mut viewport = eframe::egui::ViewportBuilder::default()
        .with_title("Image Studio")
        .with_inner_size([1440.0, 880.0])
        .with_min_inner_size([880.0, 560.0])
        .with_drag_and_drop(true);

    if let Some(icon) = load_app_icon() {
        viewport = viewport.with_icon(icon);
    }

    let opts = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    engine::log("entering eframe::run_native");
    let res = eframe::run_native(
        "Image Studio",
        opts,
        Box::new(|cc| {
            engine::log("eframe App creator callback called");
            Ok(Box::new(app::App::new(cc)))
        }),
    );
    match &res {
        Ok(_) => engine::log("eframe exited Ok"),
        Err(e) => engine::log(&format!("eframe exited Err: {e}")),
    }
    res
}
