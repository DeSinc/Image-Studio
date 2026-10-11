#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod app;
mod comfy;
mod engine;

fn main() -> eframe::Result {
    std::panic::set_hook(Box::new(|info| engine::log(&format!("PANIC: {info}"))));
    engine::log(&format!("=== start, build {} ===", env!("BUILD_STAMP")));
    engine::adopt_into_kill_job();
    let opts = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Image Studio")
            .with_inner_size([1440.0, 880.0])
            .with_min_inner_size([880.0, 560.0])
            .with_drag_and_drop(true),
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
