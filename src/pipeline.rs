//! Image → relief height map: Depth Anything V2 depth, BiRefNet objects,
//! Real-ESRGAN x2 for a cleaner detail photo, then LHK's object composition
//! (`objects.rs`).

use crate::cv::{self, Interp};
use crate::depth::{self, DepthAnythingV2};
use crate::esrgan::RealEsrganX2;
use crate::img::Mat;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Model files (searched in a user-chosen folder).
#[derive(Clone, Debug)]
pub struct ModelPaths {
    pub depth_pth: PathBuf,
    pub esrgan_bin: PathBuf,
    pub birefnet: Option<PathBuf>,
}

fn find_file(dir: &Path, name: &str, depth: usize) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    let mut dirs = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.is_file() && p.file_name().map_or(false, |n| n == name) {
            return Some(p);
        }
        if p.is_dir() {
            dirs.push(p);
        }
    }
    if depth == 0 {
        return None;
    }
    dirs.into_iter().find_map(|d| find_file(&d, name, depth - 1))
}

/// Depth Anything V2 Small checkpoint (Apache-2.0, commercial use allowed).
/// Base/Large are non-commercial; they load only via `LHK_DEPTH_MODEL`.
const DEPTH_NAMES: [&str; 1] = ["depth_anything_v2_vits.pth"];
const ESRGAN_NAME: &str = "realesr-animevideov3-x2.bin";

impl ModelPaths {
    pub fn from_dir(dir: &Path) -> Self {
        // LHK_DEPTH_MODEL: explicit depth checkpoint (e.g. another variant)
        let depth_pth = std::env::var_os("LHK_DEPTH_MODEL")
            .map(PathBuf::from)
            .or_else(|| DEPTH_NAMES.iter().find_map(|n| find_file(dir, n, 2)))
            .unwrap_or_else(|| dir.join(DEPTH_NAMES[0]));
        let esrgan_bin = find_file(dir, ESRGAN_NAME, 2).unwrap_or_else(|| dir.join(ESRGAN_NAME));
        // BiRefNet: prefer a "birefnet" folder (Hugging Face snapshot layout)
        let birefnet = [dir.join("birefnet"), dir.join("models").join("birefnet")]
            .iter()
            .find_map(|d| find_file(d, "model.safetensors", 6))
            .or_else(|| find_file(dir, "model.safetensors", 1));
        Self { depth_pth, esrgan_bin, birefnet }
    }

    pub fn complete(&self) -> bool {
        self.depth_pth.exists() && self.esrgan_bin.exists() && self.birefnet.is_some()
    }

    fn config_file() -> Option<PathBuf> {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
        Some(PathBuf::from(home).join(".lhk_sculpt_models"))
    }

    /// Remember a user-chosen model folder for the next start.
    pub fn remember(dir: &Path) {
        if let Some(f) = Self::config_file() {
            let _ = std::fs::write(f, dir.to_string_lossy().as_bytes());
        }
    }

    /// Env var `LHK_MODELS`, the remembered folder, `models` next to the
    /// program (or in the .app bundle's Resources).
    pub fn discover() -> Option<Self> {
        let mut cands: Vec<PathBuf> = Vec::new();
        if let Ok(d) = std::env::var("LHK_MODELS") {
            cands.push(PathBuf::from(d));
        }
        if let Some(s) = Self::config_file().and_then(|f| std::fs::read_to_string(f).ok()) {
            cands.push(PathBuf::from(s.trim()));
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(d) = exe.parent() {
                cands.push(d.join("models"));
                if let Some(p) = d.parent() {
                    cands.push(p.join("models"));
                    cands.push(p.join("Resources").join("models"));
                }
            }
        }
        cands.into_iter().map(|d| Self::from_dir(&d)).find(|m| m.complete())
    }
}

pub struct Models {
    pub depth: DepthAnythingV2,
    pub esrgan: RealEsrganX2,
    pub birefnet: crate::birefnet::BiRefNet,
}

impl Models {
    /// Load on the best device (Metal on Apple Silicon with `--features metal`);
    /// if a smoke test fails there, fall back to the CPU.
    pub fn load(p: &ModelPaths) -> Result<Self> {
        let dev = depth::best_device();
        if !dev.is_cpu() {
            match Self::load_on(p, &dev).and_then(|m| m.smoke_test().map(|_| m)) {
                Ok(m) => return Ok(m),
                Err(e) => eprintln!("GPU-Backend nicht nutzbar ({e:#}), verwende CPU"),
            }
        }
        Self::load_on(p, &candle_core::Device::Cpu)
    }

    fn load_on(p: &ModelPaths, dev: &candle_core::Device) -> Result<Self> {
        let Some(b) = &p.birefnet else { bail!("BiRefNet (model.safetensors) nicht gefunden") };
        Ok(Self {
            depth: DepthAnythingV2::load(&p.depth_pth, dev)?,
            esrgan: RealEsrganX2::load(&p.esrgan_bin, dev)?,
            birefnet: crate::birefnet::BiRefNet::load(b, dev)?,
        })
    }

    fn smoke_test(&self) -> Result<()> {
        let rgb = Mat::<u8>::filled(28, 28, 3, 128);
        run_depth_anything(&self.depth, &rgb, 28)?;
        self.esrgan.upscale_rgb(&rgb, |_| {})?;
        Ok(())
    }
}

/// Result of the analysis, input of `mesh::ReliefMesh`.
pub struct Generated {
    /// relief height in [0, 1] (working resolution)
    pub height: Mat<f32>,
    /// detail texture (gray), same size
    pub texture: Mat<u8>,
    /// object probability from BiRefNet (source resolution)
    pub prob: Mat<f32>,
    /// network depth in [0, 1] (source resolution)
    pub depth: Mat<f32>,
    /// gray detail photo (upscaled)
    pub photo: Mat<u8>,
}

/// Load an image as RGB (8 bit); transparency is composited on white.
pub fn read_image_rgb(path: &Path) -> Result<Mat<u8>> {
    let bytes = std::fs::read(path).with_context(|| format!("Bild nicht lesbar: {}", path.display()))?;
    if bytes.len() > 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
        if let Ok(m) = decode_jpeg_rgb(&bytes) {
            return Ok(m);
        }
    }
    let img = image::load_from_memory(&bytes).with_context(|| format!("Bildformat nicht unterstützt: {}", path.display()))?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let rgba = img.to_rgba8().into_raw();
    let mut out = Mat::<u8>::new(h, w, 3);
    for i in 0..w * h {
        let a = rgba[i * 4 + 3] as u32;
        for c in 0..3 {
            let v = rgba[i * 4 + c] as u32;
            out.data[i * 3 + c] = ((v * a + 255 * (255 - a) + 127) / 255) as u8;
        }
    }
    Ok(out)
}

/// JPEG via libjpeg-turbo (mozjpeg crate), gray JPEGs expanded to RGB.
fn decode_jpeg_rgb(bytes: &[u8]) -> Result<Mat<u8>> {
    let d = mozjpeg::Decompress::new_mem(bytes)?;
    let mut s = d.rgb()?;
    let (w, h) = (s.width(), s.height());
    let px: Vec<[u8; 3]> = s.read_scanlines()?;
    s.finish()?;
    Ok(Mat::from_vec(h, w, 3, px.into_iter().flatten().collect()))
}

/// Depth Anything V2 inference as in its reference `infer_image`: resize
/// (keep aspect, sides multiples of 14, short side >= `input_size`,
/// bicubic), ImageNet normalisation, bilinear back to the image size.
/// Returns the depth min-max normalised to [0, 1] (near = 1).
pub fn run_depth_anything(model: &DepthAnythingV2, rgb: &Mat<u8>, input_size: usize) -> Result<Vec<f32>> {
    let (h, w) = (rgb.h, rgb.w);
    let f = Mat::from_vec(h, w, 3, rgb.data.iter().map(|&v| v as f32 / 255.0f32).collect());
    let nh = depth::lower_bound_multiple_of_14(input_size, h);
    let nw = depth::lower_bound_multiple_of_14(input_size, w);
    let r = cv::resize::resize_f32(&f, nw, nh, Interp::Cubic);
    let chw = depth::normalize_to_chw(&r.data, nh, nw);
    let x = depth::tensor_from_chw(chw, nh, nw, model.device())?;
    let (d, dh, dw) = model.forward(&x)?;
    let up = depth::bilinear_chw(&d, 1, dh, dw, h, w, false);
    let mn = up.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = up.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let den = (mx - mn) + 1e-8f32;
    Ok(up.iter().map(|&v| (v - mn) / den).collect())
}

/// Short side below which the source is upscaled before depth estimation.
const SMALL_SOURCE: usize = 700;

pub fn generate(models: &Models, path: &Path, input_size: usize, progress: &mut dyn FnMut(&str, f32)) -> Result<Generated> {
    let mut rgb = read_image_rgb(path)?;
    if rgb.h < 2 || rgb.w < 2 {
        bail!("Bild ist zu klein");
    }
    if rgb.h.min(rgb.w) < SMALL_SOURCE {
        progress("Kleines Bild hochskalieren (Real-ESRGAN x2)", 0.0);
        // the upscaler returns BGR
        let bgr = models.esrgan.upscale_rgb(&rgb, |p| progress("Kleines Bild hochskalieren (Real-ESRGAN x2)", p))?;
        rgb = cv::color::swap_rb(&bgr);
    }
    progress("Tiefenschätzung (Depth Anything V2)", 0.0);
    let dn = run_depth_anything(&models.depth, &rgb, input_size)?;
    let depth = Mat::from_vec(rgb.h, rgb.w, 1, dn);
    progress("Detailfoto aufbereiten (Real-ESRGAN x2)", 0.0);
    let up = models.esrgan.upscale_rgb(&rgb, |p| progress("Detailfoto aufbereiten (Real-ESRGAN x2)", p))?;
    let photo = cv::color::bgr2gray(&up);
    drop(up);
    progress("Objekterkennung (BiRefNet)", 0.0);
    let prob = models.birefnet.predict(&rgb)?;
    progress("Objekte zusammensetzen", 0.0);
    let (height, texture) = crate::objects::compose(&depth, &photo, &prob, &crate::objects::ObjectParams::default());
    progress("Fertig", 1.0);
    Ok(Generated { height, texture, prob, depth, photo })
}
