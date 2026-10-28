//! LHK_Sculpt GUI: image list, analysis button, relief sliders and the
//! 3D viewer.

use crate::export;
use crate::mesh::{self, ReliefMesh, ReliefSettings};
use crate::pipeline::{self, Generated, ModelPaths, Models};
use crate::theme::{self, Icon};
use crate::viewer::{self, GpuVertex, Viewer};
use eframe::egui_wgpu::RenderState;
use egui::{Align, Frame, Layout, Margin, RichText};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Quality {
    Low,
    Medium,
    High,
    Ultra,
}

impl Quality {
    /// Triangles of the relief surface.
    pub fn triangles(self) -> usize {
        match self {
            Quality::Low => 750_000,
            Quality::Medium => 1_000_000,
            Quality::High => 1_500_000,
            Quality::Ultra => 2_500_000,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Quality::Low => "Niedrig (750k)",
            Quality::Medium => "Mittel (1M)",
            Quality::High => "Hoch (1,5M)",
            Quality::Ultra => "Ultra (2,5M)",
        }
    }
}

struct ImageEntry {
    path: PathBuf,
    thumb: Option<egui::TextureHandle>,
    result: Option<Arc<Generated>>,
    depth_tex: Option<egui::TextureHandle>,
    /// slider values (and side walls) last used for this image
    relief: Option<(ReliefSettings, bool)>,
}

enum JobMsg {
    Progress(String, f32),
    Done(usize, Result<Generated, String>),
}

enum ModelMsg {
    Loaded(Result<Arc<Models>, String>),
}

#[derive(Clone, Copy)]
struct MeshRequest {
    settings: ReliefSettings,
    walls: bool,
}

/// Export options: closed solid, width (0 = default size), total height
/// (0 = keep proportions).
#[derive(Clone, Copy)]
struct ExportOpts {
    solid: bool,
    width: f32,
    height: f32,
}

enum WorkerCmd {
    Update(MeshRequest),
    Export(PathBuf, MeshRequest, ExportOpts),
}

struct MeshOut {
    verts: Vec<GpuVertex>,
    faces: Option<Arc<Vec<[u32; 3]>>>,
    radius: f32,
    stats: String,
    ms: f64,
}

enum WorkerMsg {
    Mesh(MeshOut),
    Info(String),
}

struct MeshWorker {
    tx: Sender<WorkerCmd>,
    rx: Receiver<WorkerMsg>,
}

impl MeshWorker {
    fn spawn(gen: Arc<Generated>, triangles: usize, ctx: egui::Context) -> Self {
        let (tx, crx) = channel::<WorkerCmd>();
        let (wtx, rx) = channel::<WorkerMsg>();
        std::thread::spawn(move || {
            let rm = ReliefMesh::new(&gen.height, &gen.texture, Some(&gen.prob), triangles);
            let mut faces_cache: Option<(bool, Arc<Vec<[u32; 3]>>)> = None;
            let mut sent_walls: Option<bool> = None;
            while let Ok(mut cmd) = crx.recv() {
                // coalesce slider updates: keep only the newest
                if matches!(cmd, WorkerCmd::Update(_)) {
                    while let Ok(next) = crx.try_recv() {
                        let is_export = matches!(next, WorkerCmd::Export(..));
                        cmd = next;
                        if is_export {
                            break;
                        }
                    }
                }
                let t = std::time::Instant::now();
                match cmd {
                    WorkerCmd::Export(path, req, ex) => {
                        let mut s = req.settings;
                        s.fast = false;
                        let z = rm.surface(&s);
                        let (walls, bottom) = if ex.solid { (true, true) } else { (req.walls, false) };
                        let mut v = rm.vertices(&z, walls, bottom, &s);
                        let f = rm.faces(walls, bottom);
                        if ex.width > 0.0 {
                            mesh::scale_to_width(&mut v, ex.width);
                        }
                        if ex.height > 0.0 {
                            mesh::scale_z_to(&mut v, ex.height);
                        }
                        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("stl").to_ascii_lowercase();
                        let res = if ext == "obj" { export::write_obj(&path, &v, &f) } else { export::write_stl(&path, &v, &f) };
                        let _ = wtx.send(WorkerMsg::Info(match res {
                            Ok(_) => format!("Exportiert: {} ({} Dreiecke)", path.display(), f.len()),
                            Err(e) => format!("Export fehlgeschlagen: {e}"),
                        }));
                        ctx.request_repaint();
                    }
                    WorkerCmd::Update(req) => {
                        let z = rm.surface(&req.settings);
                        let faces = match &faces_cache {
                            Some((w, f)) if *w == req.walls => f.clone(),
                            _ => {
                                let f = Arc::new(rm.faces(req.walls, false));
                                faces_cache = Some((req.walls, f.clone()));
                                f
                            }
                        };
                        let pos = rm.vertices(&z, req.walls, false, &req.settings);
                        let normals = mesh::normals(&pos, &faces);
                        let send_faces = sent_walls != Some(req.walls);
                        sent_walls = Some(req.walls);
                        let r = pos.iter().fold(0f32, |r, p| r.max((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()));
                        let stats = format!("Gitter {}x{} · {} Vertices · {} Dreiecke", rm.width, rm.height, pos.len(), faces.len());
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        let out = MeshOut { verts: viewer::to_gpu(&pos, &normals), faces: if send_faces { Some(faces) } else { None }, radius: r, stats, ms };
                        if wtx.send(WorkerMsg::Mesh(out)).is_err() {
                            break;
                        }
                        ctx.request_repaint();
                    }
                }
            }
        });
        Self { tx, rx }
    }
}


#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Images,
    Relief,
    Export,
    Settings,
}

impl Page {
    fn title(self) -> (&'static str, &'static str) {
        match self {
            Page::Images => ("Bilder", "Bild wählen und Analyse starten"),
            Page::Relief => ("Relief", "Regler wirken sofort auf die 3D-Ansicht"),
            Page::Export => ("Export", "STL oder OBJ speichern"),
            Page::Settings => ("Einstellungen", "Modelle und Programm"),
        }
    }
}

/// Overall analysis progress from the pipeline's stage name and stage progress.
fn overall_progress(stage: &str, p: f32) -> f32 {
    let (a, b) = if stage.starts_with("Kleines") {
        (0.0, 0.1)
    } else if stage.starts_with("Tiefen") {
        (0.1, 0.3)
    } else if stage.starts_with("Detailfoto") {
        (0.3, 0.8)
    } else if stage.starts_with("Objekterkennung") {
        (0.8, 0.95)
    } else if stage.starts_with("Objekte") {
        (0.95, 1.0)
    } else if stage == "Fertig" {
        (1.0, 1.0)
    } else {
        (0.0, 0.02)
    };
    a + (b - a) * p.clamp(0.0, 1.0)
}

pub struct App {
    render_state: Option<RenderState>,
    viewer: Option<Viewer>,
    model_paths: Option<ModelPaths>,
    models: Option<Arc<Models>>,
    model_rx: Option<Receiver<ModelMsg>>,
    model_error: Option<String>,
    images: Vec<ImageEntry>,
    selected: Option<usize>,
    quality: Quality,
    depth_input: usize,
    export: ExportOpts,
    job_rx: Option<Receiver<JobMsg>>,
    job_status: (String, f32),
    worker: Option<MeshWorker>,
    worker_for: Option<usize>,
    settings: ReliefSettings,
    walls: bool,
    camera_fitted: bool,
    page: Page,
    status: String,
    log: Vec<String>,
    mesh_stats: String,
    pending_update: bool,
    /// Self-test (env LHK_AUTOTEST=<image>, LHK_SNAPSHOT=<png>): analyse,
    /// render, save a snapshot of the 3D view and quit.
    autotest: Option<(PathBuf, PathBuf, u32)>,
}

const MODEL_HINT: &str = "Modelldateien nicht gefunden. Bitte einen Ordner wählen, der depth_anything_v2_vits.pth, realesr-animevideov3-x2.bin und BiRefNet (model.safetensors) enthält.";

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        theme::apply(&cc.egui_ctx);
        let render_state = cc.wgpu_render_state.clone();
        let viewer = render_state.as_ref().map(Viewer::new);
        let mut app = Self {
            render_state,
            viewer,
            model_paths: ModelPaths::discover(),
            models: None,
            model_rx: None,
            model_error: None,
            images: Vec::new(),
            selected: None,
            quality: Quality::High,
            depth_input: 1024,
            export: ExportOpts { solid: true, width: 0.0, height: 0.0 },
            job_rx: None,
            job_status: (String::new(), 0.0),
            worker: None,
            worker_for: None,
            settings: ReliefSettings::default(),
            walls: true,
            camera_fitted: false,
            page: Page::Images,
            status: String::new(),
            log: Vec::new(),
            mesh_stats: String::new(),
            pending_update: false,
            autotest: None,
        };
        if let (Ok(img), Ok(snap)) = (std::env::var("LHK_AUTOTEST"), std::env::var("LHK_SNAPSHOT")) {
            app.add_images(&cc.egui_ctx, img.split(';').map(PathBuf::from).collect());
            app.autotest = Some((PathBuf::from(img), PathBuf::from(snap), 0));
        }
        app.start_model_load(cc.egui_ctx.clone());
        app
    }

    /// Status line: shown in the log and kept for the self-test report.
    fn note(&mut self, s: impl Into<String>) {
        let s = s.into();
        self.status = s.clone();
        self.log.push(s);
        if self.log.len() > 200 {
            self.log.remove(0);
        }
    }

    fn start_model_load(&mut self, ctx: egui::Context) {
        let Some(mp) = self.model_paths.clone() else {
            self.model_error = Some(MODEL_HINT.into());
            self.note(MODEL_HINT);
            return;
        };
        let (tx, rx) = channel();
        self.model_rx = Some(rx);
        self.model_error = None;
        self.note("Lade Modelle …");
        std::thread::spawn(move || {
            let r = Models::load(&mp).map(Arc::new).map_err(|e| format!("{e:#}"));
            let _ = tx.send(ModelMsg::Loaded(r));
            ctx.request_repaint();
        });
    }

    fn add_images(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        for p in paths {
            if self.images.iter().any(|e| e.path == p) {
                continue;
            }
            let thumb = image::ImageReader::open(&p).ok().and_then(|r| r.with_guessed_format().ok()).and_then(|r| r.decode().ok()).map(|im| {
                let t = im.thumbnail(256, 256).to_rgba8();
                let ci = egui::ColorImage::from_rgba_unmultiplied([t.width() as usize, t.height() as usize], t.as_raw());
                ctx.load_texture(p.display().to_string(), ci, Default::default())
            });
            self.images.push(ImageEntry { path: p, thumb, result: None, depth_tex: None, relief: None });
            if self.selected.is_none() {
                self.selected = Some(self.images.len() - 1);
            }
        }
    }

    fn start_analysis(&mut self, ctx: &egui::Context, idx: usize) {
        let Some(models) = self.models.clone() else { return };
        let path = self.images[idx].path.clone();
        let input = self.depth_input;
        let (tx, rx) = channel();
        self.job_rx = Some(rx);
        self.job_status = ("Starte Analyse …".into(), 0.0);
        self.note(format!("Analyse: {}", path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()));
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let txp = tx.clone();
            let c2 = ctx.clone();
            let r = pipeline::generate(&models, &path, input, &mut |s, p| {
                let _ = txp.send(JobMsg::Progress(s.to_string(), p));
                c2.request_repaint();
            });
            let _ = tx.send(JobMsg::Done(idx, r.map_err(|e| format!("{e:#}"))));
            ctx.request_repaint();
        });
    }

    /// Show the relief of image `idx` (analysis result kept in RAM); the
    /// slider values of the previously shown image are stored with it.
    fn start_worker(&mut self, ctx: &egui::Context, idx: usize) {
        let Some(gen) = self.images[idx].result.clone() else { return };
        if let Some(prev) = self.worker_for.and_then(|w| self.images.get_mut(w)) {
            prev.relief = Some((self.settings, self.walls));
        }
        if let Some((s, w)) = self.images[idx].relief {
            self.settings = s;
            self.walls = w;
        }
        self.worker = Some(MeshWorker::spawn(gen, self.quality.triangles(), ctx.clone()));
        self.worker_for = Some(idx);
        self.camera_fitted = false;
        self.send_update(false);
    }

    fn request(&self, fast: bool) -> MeshRequest {
        let mut s = self.settings;
        s.fast = fast;
        MeshRequest { settings: s, walls: self.walls }
    }

    fn send_update(&mut self, fast: bool) {
        if let Some(w) = &self.worker {
            let _ = w.tx.send(WorkerCmd::Update(self.request(fast)));
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.model_rx {
            if let Ok(ModelMsg::Loaded(r)) = rx.try_recv() {
                self.model_rx = None;
                match r {
                    Ok(m) => {
                        self.models = Some(m);
                        self.note("Modelle geladen.");
                    }
                    Err(e) => {
                        let e = format!("Modelle konnten nicht geladen werden: {e}");
                        self.note(e.clone());
                        self.model_error = Some(e);
                    }
                }
            }
        }
        let mut done = None;
        let mut stage = None;
        if let Some(rx) = &self.job_rx {
            while let Ok(m) = rx.try_recv() {
                match m {
                    JobMsg::Progress(s, p) => {
                        if s != self.job_status.0 {
                            stage = Some(s.clone());
                        }
                        self.job_status = (s, p);
                    }
                    JobMsg::Done(i, r) => done = Some((i, r)),
                }
            }
        }
        if let Some(s) = stage {
            self.note(s);
        }
        if let Some((i, r)) = done {
            self.job_rx = None;
            match r {
                Ok(g) => {
                    let d = &g.height;
                    let rgba: Vec<u8> = d.data.iter().flat_map(|&v| {
                        let b = (v.clamp(0.0, 1.0) * 255.0) as u8;
                        [b, b, b, 255]
                    }).collect();
                    let ci = egui::ColorImage::from_rgba_unmultiplied([d.w, d.h], &rgba);
                    self.images[i].depth_tex = Some(ctx.load_texture(format!("depth{i}"), ci, Default::default()));
                    self.note("Analyse fertig.");
                    self.images[i].result = Some(Arc::new(g));
                    self.selected = Some(i);
                    self.start_worker(ctx, i);
                    self.page = Page::Relief;
                }
                Err(e) => self.note(format!("Fehler bei der Analyse: {e}")),
            }
        }
        if let (Some(w), Some(rs)) = (&self.worker, &self.render_state) {
            let mut last = None;
            let mut infos = Vec::new();
            while let Ok(m) = w.rx.try_recv() {
                match m {
                    WorkerMsg::Mesh(o) => {
                        if let (Some(f), Some(v)) = (&o.faces, self.viewer.as_mut()) {
                            v.set_indices(rs, f);
                        }
                        last = Some(o);
                    }
                    WorkerMsg::Info(s) => infos.push(s),
                }
            }
            if let (Some(o), Some(v)) = (last, self.viewer.as_mut()) {
                v.set_vertices(rs, &o.verts);
                if !self.camera_fitted {
                    v.camera = viewer::Camera::fit(o.radius.max(1e-3));
                    self.camera_fitted = true;
                }
                self.mesh_stats = format!("{} · Update {:.0} ms", o.stats, o.ms);
            }
            for s in infos {
                self.note(s);
            }
        }
    }

    /// Label with an editable value on the right, full-width slider below.
    fn slider(&mut self, ui: &mut egui::Ui, label: &str, v: f32, range: std::ops::RangeInclusive<f32>, step: f64, decimals: usize, hint: &str) -> (f32, bool) {
        let mut x = v;
        let mut changed = false;
        let mut dragging = false;
        ui.horizontal(|ui| {
            theme::label(ui, label).on_hover_text(hint);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let r = ui.add(egui::DragValue::new(&mut x).range(range.clone()).speed(step).fixed_decimals(decimals));
                changed |= r.changed() || r.drag_stopped();
                dragging |= r.dragged() && !r.drag_stopped();
            });
        });
        ui.add_space(-4.0);
        ui.spacing_mut().slider_width = ui.available_width();
        let r = ui.add(egui::Slider::new(&mut x, range).step_by(step).show_value(false)).on_hover_text(hint);
        // quick preview while dragging, full quality once released
        changed |= r.changed() || r.drag_stopped();
        dragging |= r.dragged() && !r.drag_stopped();
        if changed {
            self.pending_update = true;
        }
        ui.add_space(4.0);
        (x, dragging)
    }

    fn page_images(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if theme::big_button(ui, true, "Bilder hinzufügen …").clicked() {
            if let Some(files) = rfd::FileDialog::new().add_filter("Bilder", &["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"]).pick_files() {
                self.add_images(ctx, files);
            }
        }
        ui.label(RichText::new("…oder Bilder ins Fenster ziehen").color(theme::TEXT_DIM));
        if self.model_error.is_some() {
            ui.colored_label(theme::ERROR, "Modelle fehlen – Ordner unter „Einstellungen“ wählen.");
            if ui.button("Zu den Einstellungen").clicked() {
                self.page = Page::Settings;
            }
        }

        let mut remove = None;
        for (i, e) in self.images.iter().enumerate() {
            let sel = self.selected == Some(i);
            let frame = theme::inset().fill(if sel { theme::FIELD_HOVER } else { theme::INSET }).inner_margin(8.0);
            frame.show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    if let Some(t) = &e.thumb {
                        let sz = t.size_vec2();
                        let s = 56.0 / sz.x.max(sz.y);
                        let img = egui::Image::from_texture(egui::load::SizedTexture::new(t.id(), sz * s)).rounding(6.0).sense(egui::Sense::click());
                        if ui.add(img).clicked() {
                            self.selected = Some(i);
                        }
                    }
                    ui.vertical(|ui| {
                        let name = e.path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                        let r = ui.add(egui::Label::new(RichText::new(name).size(15.0).color(if sel { theme::ACCENT } else { theme::TEXT })).truncate().sense(egui::Sense::click()));
                        if r.clicked() {
                            self.selected = Some(i);
                        }
                        let (txt, col) = if e.result.is_some() { ("analysiert", theme::TEXT) } else { ("nicht analysiert", theme::TEXT_DIM) };
                        ui.label(RichText::new(txt).small().color(col));
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.small_button("×").on_hover_text("entfernen").clicked() {
                            remove = Some(i);
                        }
                    });
                });
            });
        }
        if let Some(i) = remove {
            self.images.remove(i);
            match self.worker_for {
                Some(w) if w == i => {
                    self.worker = None;
                    self.worker_for = None;
                }
                Some(w) if w > i => self.worker_for = Some(w - 1),
                _ => {}
            }
            self.selected = if self.images.is_empty() { None } else { Some(i.min(self.images.len() - 1)) };
        }

        ui.add_space(6.0);
        theme::label(ui, "Qualität");
        egui::ComboBox::from_id_salt("quality").width(ui.available_width()).selected_text(self.quality.label()).show_ui(ui, |ui| {
            for q in [Quality::Low, Quality::Medium, Quality::High, Quality::Ultra] {
                ui.selectable_value(&mut self.quality, q, q.label());
            }
        });
        let dim_label = |d: usize| match d {
            518 => "Schnell (518)".to_string(),
            1024 => "Standard (1024)".to_string(),
            1200 => "Hoch (1200)".to_string(),
            d => d.to_string(),
        };
        theme::label(ui, "Tiefen-Auflösung");
        egui::ComboBox::from_id_salt("depth_input").width(ui.available_width()).selected_text(dim_label(self.depth_input)).show_ui(ui, |ui| {
            for d in [518usize, 1024, 1200] {
                ui.selectable_value(&mut self.depth_input, d, dim_label(d));
            }
        });

        if let Some(sel) = self.selected {
            ui.add_space(6.0);
            let e = &self.images[sel];
            let analysed = e.result.is_some();
            let shown = self.worker_for == Some(sel);
            let mut show = false;
            theme::inset().show(ui, |ui| {
                ui.set_width(ui.available_width());
                let name = e.path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                ui.label(RichText::new(name).color(theme::ACCENT));
                if let Some(g) = &e.result {
                    ui.label(format!("Höhenkarte: {} × {}", g.height.w, g.height.h));
                }
                ui.label(format!("Qualität: {}", self.quality.label()));
                if let Some(t) = &e.depth_tex {
                    let sz = t.size_vec2();
                    let s = (ui.available_width() / sz.x).min(180.0 / sz.y);
                    ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(t.id(), sz * s)).rounding(6.0));
                }
                if analysed {
                    let txt = if shown { "Mesh mit neuer Qualität neu aufbauen" } else { "Ergebnis anzeigen" };
                    show = ui.button(txt).clicked();
                } else {
                    ui.label(RichText::new("Zum Analysieren unten ▶ drücken.").color(theme::TEXT_DIM));
                }
            });
            if show {
                self.start_worker(ctx, sel);
            }
        }
    }

    fn page_relief(&mut self, ui: &mut egui::Ui) {
        if self.worker.is_none() {
            ui.label(RichText::new("Noch kein Relief. Unter „Bilder“ ein Bild wählen und Analyse starten.").color(theme::TEXT_DIM));
            return;
        }
        let mut dragging = false;
        macro_rules! sl {
            ($field:expr, $label:expr, $range:expr, $step:expr, $dec:expr, $hint:expr) => {{
                let (v, d) = self.slider(ui, $label, $field, $range, $step, $dec, $hint);
                $field = v;
                dragging |= d;
            }};
        }
        theme::section(ui, "Relief");
        sl!(self.settings.height, "Höhe", 0.0..=80.0, 0.1, 1, "Höhe des Reliefs in Modell-Einheiten (Standardgröße ca. 700 breit)");
        sl!(self.settings.detail, "Detail", 0.0..=20.0, 0.05, 2, "Stärke der Feindetails (Haare, Falten, Strukturen)");
        sl!(self.settings.smoothing, "Glättung", 0.0..=100.0, 1.0, 0, "Glättet flache Bereiche, Kanten und Umrisse bleiben scharf");
        theme::section(ui, "Objekte");
        sl!(self.settings.spike_guard, "Spike-Filter", 0.0..=100.0, 1.0, 0, "Drückt Ausreißer pro Objekt auf die passende Höhe herunter");
        sl!(self.settings.bg_sharpness, "Hintergrund-Schärfe %", 0.0..=300.0, 5.0, 0, "Nur der Hintergrund: <100 glatter, >100 schärfer. Objekte bleiben unverändert.");
        theme::section(ui, "Körper");
        ui.horizontal(|ui| {
            theme::label(ui, "Seitenwände");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.checkbox(&mut self.walls, "").changed() {
                    self.pending_update = true;
                }
            });
        });
        let (v, d) = self.slider(ui, "Sockel %", self.settings.base * 100.0, 0.0..=60.0, 1.0, 0, "Dicke unter dem tiefsten Punkt, in % der Reliefhöhe");
        self.settings.base = v / 100.0;
        dragging |= d;
        if self.pending_update {
            self.send_update(dragging);
            self.pending_update = false;
        }
        ui.add_space(8.0);
        if ui.button("Standardwerte").clicked() {
            self.settings = ReliefSettings::default();
            self.send_update(false);
        }
    }

    fn page_export(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            theme::label(ui, "Geschlossener Körper").on_hover_text("Seitenwände und Boden, druckfertig");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.checkbox(&mut self.export.solid, "");
            });
        });
        ui.horizontal(|ui| {
            theme::label(ui, "Breite").on_hover_text("0 = Standardgröße (Fläche ca. 748 000 Einheiten²)");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add(egui::DragValue::new(&mut self.export.width).range(0.0..=5000.0).min_decimals(0));
            });
        });
        ui.horizontal(|ui| {
            theme::label(ui, "Gesamthöhe").on_hover_text("0 = Höhe gleichmäßig mitskalieren; sonst feste Gesamthöhe inkl. Sockel");
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add(egui::DragValue::new(&mut self.export.height).range(0.0..=2000.0).speed(0.1));
            });
        });
        ui.label(RichText::new("0 = Standardgröße bzw. proportional").small().color(theme::TEXT_DIM));
        ui.add_space(8.0);
        let can = self.worker.is_some();
        if theme::big_button(ui, can, "STL / OBJ exportieren …").clicked() {
            if let Some(p) = rfd::FileDialog::new().add_filter("STL", &["stl"]).add_filter("OBJ", &["obj"]).set_file_name("relief.stl").save_file() {
                if let Some(w) = &self.worker {
                    let _ = w.tx.send(WorkerCmd::Export(p, self.request(false), self.export));
                    self.note("Exportiere …");
                }
            }
        }
        if !can {
            ui.label(RichText::new("Erst ein Bild analysieren.").color(theme::TEXT_DIM));
        }
    }

    fn page_settings(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        theme::label(ui, "Modellordner");
        let dir = self.model_paths.as_ref().and_then(|m| m.depth_pth.parent().map(|p| p.display().to_string())).unwrap_or_else(|| "–".into());
        theme::inset().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(dir).monospace());
            let (txt, col) = if self.models.is_some() {
                ("Modelle geladen", theme::TEXT)
            } else if self.model_rx.is_some() {
                ("Lade Modelle …", theme::TEXT_DIM)
            } else {
                ("Nicht geladen", theme::ERROR)
            };
            ui.label(RichText::new(txt).color(col));
        });
        if let Some(e) = &self.model_error {
            ui.colored_label(theme::ERROR, e);
        }
        if theme::big_button(ui, self.model_rx.is_none(), "Modellordner wählen …").clicked() {
            if let Some(d) = rfd::FileDialog::new().pick_folder() {
                let mp = ModelPaths::from_dir(&d);
                if mp.complete() {
                    ModelPaths::remember(&d);
                    self.model_paths = Some(mp);
                    self.models = None;
                    self.start_model_load(ctx.clone());
                } else {
                    self.model_error = Some(MODEL_HINT.into());
                    self.note(MODEL_HINT);
                }
            }
        }
        theme::section(ui, "Über");
        theme::inset().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(format!("LHK_Sculpt {}", env!("CARGO_PKG_VERSION")));
            for m in ["Depth Anything V2 Small – Apache-2.0", "BiRefNet – MIT", "Real-ESRGAN – BSD-3-Clause"] {
                ui.label(RichText::new(m).color(theme::TEXT_DIM));
            }
        });
    }

    fn run_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let busy = self.job_rx.is_some();
        let can = self.models.is_some() && self.selected.is_some() && !busy;
        let (text, col) = if busy {
            let (s, p) = &self.job_status;
            (format!("{s} · {:.0} %", overall_progress(s, *p) * 100.0), theme::TEXT)
        } else if self.model_error.is_some() {
            ("Modelle fehlen".into(), theme::ERROR)
        } else if self.models.is_none() {
            ("Lade Modelle …".into(), theme::TEXT_DIM)
        } else if self.selected.is_none() {
            ("Bild hinzufügen".into(), theme::TEXT_DIM)
        } else {
            ("Bereit – ▶ startet die Analyse".into(), theme::TEXT_DIM)
        };
        theme::inset().rounding(14.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                if theme::play_button(ui, can, busy).on_hover_text("Analyse starten").clicked() {
                    let i = self.selected.unwrap();
                    self.start_analysis(ctx, i);
                }
                ui.vertical(|ui| {
                    ui.add_space(4.0);
                    let frac = if busy { overall_progress(&self.job_status.0, self.job_status.1) } else { 0.0 };
                    theme::progress(ui, frac);
                    ui.label(RichText::new(text).small().color(col));
                });
            });
        });
    }

    fn viewer_area(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let avail = ui.available_size();
        let (rect, resp) = ui.allocate_exact_size(avail, egui::Sense::click_and_drag());
        let ppp = ctx.pixels_per_point();
        let Some((v, rs)) = self.viewer.as_mut().zip(self.render_state.as_ref()) else {
            ui.painter().rect_filled(rect, 10.0, theme::INSET);
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "wgpu nicht verfügbar", egui::FontId::proportional(20.0), theme::ERROR);
            return;
        };
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            v.camera.orbit(d.x, d.y);
        }
        if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
            let d = resp.drag_delta();
            v.camera.pan_by(d.x, d.y, rect.height());
        }
        if resp.hovered() {
            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                v.camera.zoom(scroll);
            }
        }
        if resp.double_clicked() {
            let r = v.camera.radius;
            v.camera = viewer::Camera::fit(r);
        }
        if v.has_mesh() {
            v.render(rs, (rect.width() * ppp) as u32, (rect.height() * ppp) as u32);
            if let Some(id) = v.texture_id {
                egui::Image::from_texture(egui::load::SizedTexture::new(id, rect.size())).rounding(10.0).paint_at(ui, rect);
            }
        } else {
            ui.painter().rect_filled(rect, 10.0, theme::INSET);
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "3D-Ansicht – noch kein Relief", egui::FontId::proportional(20.0), theme::TEXT_DIM);
        }
    }

    fn log_box(&mut self, ui: &mut egui::Ui) {
        theme::inset().show(ui, |ui| {
            ui.set_min_size(ui.available_size());
            egui::ScrollArea::vertical().stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
                for l in &self.log {
                    ui.label(RichText::new(l).size(14.0).color(theme::TEXT));
                }
                if self.job_rx.is_some() {
                    let (s, p) = &self.job_status;
                    ui.label(RichText::new(format!("{s} {:.0} %", p * 100.0)).size(14.0).color(theme::ACCENT));
                }
                if !self.mesh_stats.is_empty() {
                    ui.label(RichText::new(&self.mesh_stats).size(14.0).color(theme::TEXT_DIM));
                    ui.label(RichText::new("Linke Maus: drehen · Rechte Maus: verschieben · Rad: zoomen · Doppelklick: Ansicht zurücksetzen").size(13.0).color(theme::TEXT_DIM));
                }
            });
        });
    }
}

impl App {
    fn autotest_step(&mut self, ctx: &egui::Context) {
        let Some((_, snap, frames)) = self.autotest.clone() else { return };
        // analyse all images, then switch back to the first one
        if let Some(i) = self.images.iter().position(|e| e.result.is_none()) {
            if self.models.is_some() && self.job_rx.is_none() {
                self.start_analysis(ctx, i);
            }
            ctx.request_repaint();
            return;
        }
        if self.selected != Some(0) {
            self.selected = Some(0);
        }
        if self.camera_fitted && self.worker_for == Some(0) {
            if let Some(v) = self.viewer.as_mut() {
                if frames == 3 {
                    v.camera.yaw = 0.55;
                    v.camera.pitch = 0.35;
                }
            }
            if frames >= 8 {
                if let (Some(v), Some(rs)) = (&self.viewer, &self.render_state) {
                    if let Some((w, h, px)) = v.snapshot(rs) {
                        let _ = image::save_buffer(&snap, &px, w, h, image::ExtendedColorType::Rgba8);
                    }
                }
                std::fs::write(snap.with_extension("txt"), format!("{}\n{}\n", self.status, self.mesh_stats)).ok();
                if std::env::var("LHK_AUTOEXIT").is_ok() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                self.autotest = None;
                return;
            }
            self.autotest = Some((PathBuf::new(), snap, frames + 1));
        }
        ctx.request_repaint();
    }
}

impl eframe::App for App {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        theme::BG.to_normalized_gamma_f32()
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // the start-up maximize flag is not always honoured on Windows
        if ctx.cumulative_pass_nr() == 1 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
        }
        self.poll(ctx);
        self.autotest_step(ctx);
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if !dropped.is_empty() {
            self.add_images(ctx, dropped);
            self.page = Page::Images;
        }
        // selecting an analysed image shows its relief right away
        if let Some(s) = self.selected {
            if self.images.get(s).map_or(false, |e| e.result.is_some()) && self.worker_for != Some(s) {
                self.start_worker(ctx, s);
            }
        }

        // icon rail
        egui::SidePanel::left("rail")
            .exact_width(82.0)
            .resizable(false)
            .show_separator_line(false)
            .frame(Frame::none().fill(theme::BG).inner_margin(Margin { left: 14.0, right: 0.0, top: 14.0, bottom: 14.0 }))
            .show(ctx, |ui| {
                // rounded rail background drawn over the full panel content
                ui.painter().rect_filled(ui.max_rect(), 14.0, theme::PANEL);
                ui.add_space(9.0);
                ui.vertical_centered(|ui| {
                    for (p, icon, tip) in [(Page::Images, Icon::Images, "Bilder"), (Page::Relief, Icon::Relief, "Relief-Regler"), (Page::Export, Icon::Export, "Export")] {
                        if theme::rail_button(ui, icon, self.page == p).on_hover_text(tip).clicked() {
                            self.page = p;
                        }
                    }
                });
                ui.with_layout(Layout::bottom_up(Align::Center), |ui| {
                    ui.add_space(9.0);
                    if theme::rail_button(ui, Icon::Settings, self.page == Page::Settings).on_hover_text("Einstellungen").clicked() {
                        self.page = Page::Settings;
                    }
                });
            });

        // page content with the run bar at the bottom
        egui::SidePanel::left("content")
            .resizable(true)
            .default_width(430.0)
            .min_width(370.0)
            .show_separator_line(false)
            .frame(theme::card(Margin { left: 10.0, right: 0.0, top: 14.0, bottom: 14.0 }))
            .show(ctx, |ui| {
                let (title, sub) = self.page.title();
                ui.label(RichText::new(title).size(24.0).color(theme::ACCENT));
                ui.label(RichText::new(sub).color(theme::TEXT_DIM));
                ui.add_space(6.0);
                egui::TopBottomPanel::bottom("runbar")
                    .frame(Frame::none().inner_margin(Margin { left: 0.0, right: 0.0, top: 12.0, bottom: 0.0 }))
                    .show_separator_line(false)
                    .show_inside(ui, |ui| self.run_bar(ui, ctx));
                egui::CentralPanel::default().frame(Frame::none()).show_inside(ui, |ui| {
                    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                        ui.set_width(ui.available_width() - 6.0);
                        match self.page {
                            Page::Images => self.page_images(ui, ctx),
                            Page::Relief => self.page_relief(ui),
                            Page::Export => self.page_export(ui),
                            Page::Settings => self.page_settings(ui, ctx),
                        }
                    });
                });
            });

        // 3D view with the log below
        egui::CentralPanel::default()
            .frame(theme::card(Margin { left: 10.0, right: 14.0, top: 14.0, bottom: 14.0 }))
            .show(ctx, |ui| {
                egui::TopBottomPanel::bottom("log")
                    .exact_height(170.0)
                    .resizable(false)
                    .frame(Frame::none().inner_margin(Margin { left: 0.0, right: 0.0, top: 14.0, bottom: 0.0 }))
                    .show_separator_line(false)
                    .show_inside(ui, |ui| self.log_box(ui));
                egui::CentralPanel::default().frame(Frame::none()).show_inside(ui, |ui| self.viewer_area(ui, ctx));
            });
    }
}
