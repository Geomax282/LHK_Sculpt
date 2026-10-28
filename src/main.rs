#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
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
