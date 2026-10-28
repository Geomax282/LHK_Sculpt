//! Real-ESRGAN `realesr-animevideov3-x2` (SRVGGNetCompact) as run by the
//! realesrgan-ncnn-vulkan tool (Real-ESRGAN, BSD-3-Clause) with `-s 2`.
//!
//! Network (from the shipped .param): conv3x3(3->64)+PReLU, 16x
//! [conv3x3(64->64)+PReLU], conv3x3(64->48), PixelShuffle(4), + nearest x4
//! of the input, then bicubic x0.5. Weights are read from the ncnn .bin.
//! Tiling (tile 200, prepadding 10, reflect-101 at image borders) follows
//! realesrgan-ncnn-vulkan so tile seams match.

use crate::img::Mat;
use anyhow::{bail, Context, Result};
use candle_core::{Device, Module, Tensor};
use candle_nn::{Conv2d, Conv2dConfig};
use half::f16;
use std::path::Path;

pub const TILE: usize = 200;
pub const PREPADDING: usize = 10;

pub struct RealEsrganX2 {
    device: Device,
    convs: Vec<Conv2d>,
    prelu: Vec<Tensor>, // (1, 64, 1, 1)
}

struct BinReader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> BinReader<'a> {
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        let end = self.p + n * 4;
        if end > self.b.len() {
            bail!("ncnn bin zu kurz");
        }
        let v = self.b[self.p..end].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        self.p = end;
        Ok(v)
    }
    /// Weight blob with ncnn storage tag (fp32 = 0, fp16 = 0x01306B47).
    fn tagged(&mut self, n: usize) -> Result<Vec<f32>> {
        let tag = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into()?);
        self.p += 4;
        match tag {
            0x0130_6B47 => {
                let end = self.p + n * 2;
                let v = self.b[self.p..end].chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect();
                self.p = end + ((4 - (n * 2) % 4) % 4);
                Ok(v)
            }
            0 => self.f32s(n),
            t => bail!("unbekanntes ncnn Gewichtsformat {t:#x}"),
        }
    }
}

impl RealEsrganX2 {
    pub fn load(bin_path: &Path, device: &Device) -> Result<Self> {
        let bytes = std::fs::read(bin_path).with_context(|| format!("RealESRGAN Modell fehlt: {}", bin_path.display()))?;
        let mut r = BinReader { b: &bytes, p: 0 };
        let mut convs = Vec::new();
        let mut prelu = Vec::new();
        let chans: Vec<(usize, usize)> = std::iter::once((3, 64)).chain(std::iter::repeat((64, 64)).take(16)).chain(std::iter::once((64, 48))).collect();
        for (i, &(ci, co)) in chans.iter().enumerate() {
            let w = r.tagged(co * ci * 9)?;
            let b = r.f32s(co)?;
            let wt = Tensor::from_vec(w, (co, ci, 3, 3), device)?;
            let bt = Tensor::from_vec(b, co, device)?;
            convs.push(Conv2d::new(wt, Some(bt), Conv2dConfig { padding: 1, stride: 1, dilation: 1, groups: 1 }));
            if i < 17 {
                let s = r.f32s(64)?;
                prelu.push(Tensor::from_vec(s, (1, 64, 1, 1), device)?);
            }
        }
        if r.p != bytes.len() {
            bail!("ncnn bin: {} Restbytes", bytes.len() - r.p);
        }
        Ok(Self { device: device.clone(), convs, prelu })
    }

    /// Network on one tile: input CHW in [0,1], returns 2x CHW.
    fn run_tile(&self, chw: Vec<f32>, h: usize, w: usize) -> Result<Vec<f32>> {
        let x = Tensor::from_vec(chw.clone(), (1, 3, h, w), &self.device)?;
        let mut t = x;
        for i in 0..18 {
            t = self.convs[i].forward(&t)?;
            if i < 17 {
                // PReLU: max(0,x) + a*min(0,x)
                let pos = t.relu()?;
                let neg = (t - &pos)?;
                t = (pos + neg.broadcast_mul(&self.prelu[i])?)?;
            }
        }
        let v = t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?; // (48, h, w)
        let (h4, w4) = (h * 4, w * 4);
        let mut up = vec![0f32; 3 * h4 * w4];
        for c in 0..3 {
            for y in 0..h4 {
                for xx in 0..w4 {
                    let (i, j) = (y % 4, xx % 4);
                    let src_c = c * 16 + i * 4 + j;
                    let sv = v[(src_c * h + y / 4) * w + xx / 4];
                    up[(c * h4 + y) * w4 + xx] = sv + chw[(c * h + y / 4) * w + xx / 4];
                }
            }
        }
        Ok(ncnn_bicubic_half(&up, 3, h4, w4))
    }

    /// Upscale an RGB image 2x; returns BGR like `cv2.imread` of the PNG the
    /// executable writes.
    pub fn upscale_rgb(&self, rgb: &Mat<u8>, mut progress: impl FnMut(f32)) -> Result<Mat<u8>> {
        assert_eq!(rgb.c, 3);
        let (h, w) = (rgb.h, rgb.w);
        let xt = w.div_ceil(TILE);
        let yt = h.div_ceil(TILE);
        let mut out = Mat::<u8>::new(h * 2, w * 2, 3);
        let total = (xt * yt) as f32;
        let mut done = 0f32;
        for yi in 0..yt {
            for xi in 0..xt {
                let tx0 = (xi * TILE) as i64 - PREPADDING as i64;
                let tx1 = ((xi + 1) * TILE).min(w) as i64 + PREPADDING as i64;
                let ty0 = (yi * TILE) as i64 - PREPADDING as i64;
                let ty1 = ((yi + 1) * TILE).min(h) as i64 + PREPADDING as i64;
                let (tw, th) = ((tx1 - tx0) as usize, (ty1 - ty0) as usize);
                let mut chw = vec![0f32; 3 * tw * th];
                let refl = |p: i64, n: i64| -> usize { ((n - 1) - ((p.abs()) - (n - 1)).abs()) as usize };
                for y in 0..th {
                    let sy = refl(ty0 + y as i64, h as i64);
                    for x in 0..tw {
                        let sx = refl(tx0 + x as i64, w as i64);
                        for c in 0..3 {
                            chw[(c * th + y) * tw + x] = rgb.at(sy, sx, c) as f32 * (1.0 / 255.0);
                        }
                    }
                }
                let o = self.run_tile(chw, th, tw)?;
                let (ow, _oh) = (tw * 2, th * 2);
                let cw = ((xi + 1) * TILE).min(w) - xi * TILE;
                let chh = ((yi + 1) * TILE).min(h) - yi * TILE;
                for y in 0..chh * 2 {
                    for x in 0..cw * 2 {
                        let (sy, sx) = (y + PREPADDING * 2, x + PREPADDING * 2);
                        let dy = yi * TILE * 2 + y;
                        let dx = xi * TILE * 2 + x;
                        for c in 0..3 {
                            let v = o[(c * th * 2 + sy) * ow + sx] * 255.0;
                            let q = (v + 0.5).clamp(0.0, 255.0) as u8;
                            // store BGR
                            out.set(dy, dx, 2 - c, q);
                        }
                    }
                }
                done += 1.0;
                progress(done / total);
            }
        }
        Ok(out)
    }
}

/// ncnn Interp (bicubic, scale 0.5, align_corner 0) incl. its border fix-ups.
fn ncnn_bicubic_half(src: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let (oh, ow) = (h / 2, w / 2);
    let xs = cubic_coeffs(w, ow);
    let ys = cubic_coeffs(h, oh);
    let mut out = vec![0f32; c * oh * ow];
    for ch in 0..c {
        let s = &src[ch * h * w..(ch + 1) * h * w];
        // horizontal into rows buffer
        let mut rows = vec![0f32; h * ow];
        for y in 0..h {
            for (dx, (sx, a)) in xs.iter().enumerate() {
                let r = &s[y * w..y * w + w];
                let i = *sx as usize;
                rows[y * ow + dx] = r[i - 1] * a[0] + r[i] * a[1] + r[i + 1] * a[2] + r[i + 2] * a[3];
            }
        }
        for (dy, (sy, b)) in ys.iter().enumerate() {
            let i = *sy as usize;
            for dx in 0..ow {
                out[(ch * oh + dy) * ow + dx] =
                    rows[(i - 1) * ow + dx] * b[0] + rows[i * ow + dx] * b[1] + rows[(i + 1) * ow + dx] * b[2] + rows[(i + 2) * ow + dx] * b[3];
            }
        }
    }
    out
}

fn interpolate_cubic(fx: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let fx0 = fx + 1.0;
    let fx1 = fx;
    let fx2 = 1.0 - fx;
    let c0 = A * fx0 * fx0 * fx0 - 5.0 * A * fx0 * fx0 + 8.0 * A * fx0 - 4.0 * A;
    let c1 = (A + 2.0) * fx1 * fx1 * fx1 - (A + 3.0) * fx1 * fx1 + 1.0;
    let c2 = (A + 2.0) * fx2 * fx2 * fx2 - (A + 3.0) * fx2 * fx2 + 1.0;
    [c0, c1, c2, 1.0 - c0 - c1 - c2]
}

fn cubic_coeffs(w: usize, outw: usize) -> Vec<(i64, [f32; 4])> {
    let scale = w as f64 / outw as f64;
    let w = w as i64;
    (0..outw)
        .map(|dx| {
            let mut fx = ((dx as f64 + 0.5) * scale - 0.5) as f32;
            let mut sx = fx.floor() as i64;
            fx -= sx as f32;
            let mut a = interpolate_cubic(fx);
            if sx <= -1 {
                sx = 1;
                a = [1.0 - a[3], a[3], 0.0, 0.0];
            }
            if sx == 0 {
                sx = 1;
                a = [a[0] + a[1], a[2], a[3], 0.0];
            }
            if sx == w - 2 {
                sx = w - 3;
                a = [0.0, a[0], a[1], a[2] + a[3]];
            }
            if sx >= w - 1 {
                sx = w - 3;
                a = [0.0, 0.0, a[0], 1.0 - a[0]];
            }
            (sx, a)
        })
        .collect()
}
