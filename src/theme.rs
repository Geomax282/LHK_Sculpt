//! Visual style of the GUI: dark, flat, rounded panels, an icon rail and
//! large labels. Icons are drawn with the painter (no icon font needed).

use egui::{pos2, vec2, Color32, FontFamily, FontId, Frame, Margin, Rect, Response, Rounding, Sense, Shape, Stroke, TextStyle, Ui};

pub const BG: Color32 = Color32::from_rgb(20, 22, 26);
pub const PANEL: Color32 = Color32::from_rgb(29, 32, 37);
pub const INSET: Color32 = Color32::from_rgb(37, 41, 47);
pub const FIELD: Color32 = Color32::from_rgb(45, 49, 57);
pub const FIELD_HOVER: Color32 = Color32::from_rgb(56, 61, 70);
pub const TEXT: Color32 = Color32::from_rgb(202, 206, 212);
pub const TEXT_DIM: Color32 = Color32::from_rgb(128, 134, 143);
pub const ACCENT: Color32 = Color32::from_rgb(238, 240, 243);
pub const ERROR: Color32 = Color32::from_rgb(236, 122, 112);

pub fn apply(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (TextStyle::Heading, FontId::new(22.0, FontFamily::Proportional)),
        (TextStyle::Body, FontId::new(15.0, FontFamily::Proportional)),
        (TextStyle::Button, FontId::new(15.0, FontFamily::Proportional)),
        (TextStyle::Small, FontId::new(12.5, FontFamily::Proportional)),
        (TextStyle::Monospace, FontId::new(13.0, FontFamily::Monospace)),
    ]
    .into();
    style.spacing.item_spacing = vec2(10.0, 9.0);
    style.spacing.button_padding = vec2(12.0, 6.0);
    style.spacing.interact_size.y = 28.0;
    style.spacing.combo_height = 300.0;

    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.window_fill = PANEL;
    v.extreme_bg_color = FIELD;
    v.faint_bg_color = INSET;
    v.window_rounding = Rounding::same(12.0);
    v.menu_rounding = Rounding::same(10.0);
    v.window_stroke = Stroke::new(1.0_f32, FIELD);
    v.selection.bg_fill = Color32::from_rgb(118, 126, 138);
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT);
    v.slider_trailing_fill = true;
    v.handle_shape = egui::style::HandleShape::Circle;
    v.error_fg_color = ERROR;
    for (w, fill) in [
        (&mut v.widgets.inactive, FIELD),
        (&mut v.widgets.hovered, FIELD_HOVER),
        (&mut v.widgets.active, FIELD_HOVER),
        (&mut v.widgets.open, FIELD),
    ] {
        w.bg_fill = fill;
        w.weak_bg_fill = fill;
        w.bg_stroke = Stroke::NONE;
        w.rounding = Rounding::same(10.0);
        w.fg_stroke = Stroke::new(1.0_f32, TEXT);
        w.expansion = 0.0;
    }
    v.widgets.hovered.fg_stroke = Stroke::new(1.5_f32, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.5_f32, ACCENT);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, INSET);
    v.widgets.noninteractive.rounding = Rounding::same(10.0);
    style.visuals = v;
    // always dark, also when the OS uses a light theme
    ctx.set_style_of(egui::Theme::Dark, style.clone());
    ctx.set_style_of(egui::Theme::Light, style);
    ctx.set_theme(egui::ThemePreference::Dark);
}

/// Large rounded panel; `m` is the gap to the window edge / neighbours.
pub fn card(m: Margin) -> Frame {
    Frame::none().fill(PANEL).rounding(14.0).inner_margin(18.0).outer_margin(m)
}

/// Darker box inside a card (info text, list entries, log).
pub fn inset() -> Frame {
    Frame::none().fill(INSET).rounding(12.0).inner_margin(12.0)
}

/// Field label as in the reference design: large and light.
pub fn label(ui: &mut Ui, text: &str) -> Response {
    ui.label(egui::RichText::new(text).size(17.0).color(TEXT))
}

pub fn section(ui: &mut Ui, text: &str) {
    ui.add_space(6.0);
    ui.label(egui::RichText::new(text.to_uppercase()).size(12.0).color(TEXT_DIM).strong());
}

/// Full-width button with large text.
pub fn big_button(ui: &mut Ui, enabled: bool, text: &str) -> Response {
    let w = ui.available_width();
    ui.add_enabled(enabled, egui::Button::new(egui::RichText::new(text).size(16.0)).min_size(vec2(w, 36.0)))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Icon {
    Images,
    Relief,
    Export,
    Settings,
}

/// Square button of the icon rail.
pub fn rail_button(ui: &mut Ui, icon: Icon, selected: bool) -> Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(48.0, 48.0), Sense::click());
    let fill = if selected {
        FIELD_HOVER
    } else if resp.hovered() {
        FIELD
    } else {
        Color32::TRANSPARENT
    };
    ui.painter().rect_filled(rect, 10.0, fill);
    let col = if selected || resp.hovered() { ACCENT } else { TEXT_DIM };
    paint_icon(ui.painter(), rect.shrink(12.0), icon, Stroke::new(2.0_f32, col));
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

fn paint_icon(p: &egui::Painter, r: Rect, icon: Icon, s: Stroke) {
    let at = |x: f32, y: f32| pos2(r.left() + x * r.width(), r.top() + y * r.height());
    match icon {
        Icon::Images => {
            p.rect_stroke(r.shrink2(vec2(0.0, 2.0)), 3.0, s);
            p.add(Shape::line(vec![at(0.08, 0.82), at(0.38, 0.5), at(0.58, 0.68), at(0.72, 0.56), at(0.92, 0.78)], s));
            p.circle_stroke(at(0.7, 0.3), r.width() * 0.08, s);
        }
        Icon::Relief => {
            for (y, k) in [(0.2, 0.3), (0.5, 0.7), (0.8, 0.45)] {
                p.line_segment([at(0.0, y), at(1.0, y)], s);
                p.circle_filled(at(k, y), r.width() * 0.11, s.color);
            }
        }
        Icon::Export => {
            p.line_segment([at(0.5, 0.0), at(0.5, 0.62)], s);
            p.add(Shape::line(vec![at(0.25, 0.4), at(0.5, 0.65), at(0.75, 0.4)], s));
            p.add(Shape::line(vec![at(0.05, 0.65), at(0.05, 0.95), at(0.95, 0.95), at(0.95, 0.65)], s));
        }
        Icon::Settings => {
            let c = r.center();
            let ro = r.width() * 0.5;
            let ri = r.width() * 0.34;
            for i in 0..8 {
                let a = i as f32 * std::f32::consts::TAU / 8.0;
                let d = vec2(a.cos(), a.sin());
                p.line_segment([c + d * ri, c + d * ro], Stroke::new(s.width * 1.8, s.color));
            }
            p.circle_stroke(c, ri, s);
            p.circle_stroke(c, r.width() * 0.13, s);
        }
    }
}

/// Start button of the run bar (play triangle, spinner while busy).
pub fn play_button(ui: &mut Ui, enabled: bool, busy: bool) -> Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(52.0, 52.0), if enabled { Sense::click() } else { Sense::hover() });
    let fill = if enabled && resp.hovered() { FIELD_HOVER } else { FIELD };
    ui.painter().rect_filled(rect, 10.0, fill);
    if busy {
        egui::Spinner::new().size(24.0).color(ACCENT).paint_at(ui, Rect::from_center_size(rect.center(), vec2(24.0, 24.0)));
    } else {
        let col = if enabled { ACCENT } else { TEXT_DIM };
        let c = rect.center();
        let tri = vec![c + vec2(-7.0, -11.0), c + vec2(11.0, 0.0), c + vec2(-7.0, 11.0)];
        ui.painter().add(Shape::closed_line(tri, Stroke::new(2.0_f32, col)));
    }
    if enabled {
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    } else {
        resp
    }
}

/// Flat progress bar: rounded track with a white fill.
pub fn progress(ui: &mut Ui, frac: f32) {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(w, 22.0), Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 11.0, FIELD);
    let f = frac.clamp(0.0, 1.0);
    if f > 0.0 {
        let fill = Rect::from_min_size(rect.min, vec2((rect.width() * f).max(22.0), rect.height()));
        p.rect_filled(fill, 11.0, ACCENT);
    }
}
