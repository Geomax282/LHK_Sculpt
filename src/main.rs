#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

/// Window / taskbar icon (same logo as the .exe icon).
fn icon() -> Option<egui::IconData> {
    let im = image::load_from_memory(include_bytes!("../assets/icon_256.png")).ok()?.to_rgba8();
    Some(egui::IconData { width: im.width(), height: im.height(), rgba: im.into_raw() })
}

fn main() -> eframe::Result<()> {
    let mut viewport = egui::ViewportBuilder::default();
    if let Some(i) = icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(i));
    }
    let options = eframe::NativeOptions {
        viewport: viewport
            .with_inner_size([1200.0, 680.0])
            .with_min_inner_size([900.0, 560.0])
            .with_maximized(true)
            .with_title("LHK_Sculpt – Bild zu Relief")
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native("LHK_Sculpt", options, Box::new(|cc| Ok(Box::new(lhk_sculpt::app::App::new(cc)))))
}
