//! Object-aware layered relief (LHK extension).
//!
//! BiRefNet finds the salient objects (people incl. hair, glasses, ...).
//! Each connected object becomes one coherent, rounded body built from the
//! network depth inside it; the background is inpainted behind the objects
//! and laid out as a flatter layer underneath. Fine detail (hair strands,
//! folds) is added per object as small, zero-mean height variations, so no
//! brightness-driven craters and no isolated spikes appear.

use crate::cv::filter::{gaussian_blur_f32, Border};
use crate::cv::{self, Interp};
use crate::img::Mat;
use crate::nd;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct ObjectParams {
    /// height of the object layer above the background (0..1 of total)
    pub obj_lift: f32,
    /// relief range inside an object from the network depth
    pub obj_depth: f32,
    /// rounded "inflation" of the silhouette (body volume)
    pub obj_inflate: f32,
    /// inflation radius as a fraction of the object size
    pub inflate_radius: f32,
    /// background relief range
    pub bg_depth: f32,
    /// photo structure in the background (rocks, landscape)
    pub bg_struct: f32,
    /// fine photo structure on objects (hair, folds)
    pub obj_struct: f32,
    /// smoothing sigma of the object depth in px (removes spikes)
    pub obj_smooth: f32,
    /// soft-mask feather in px
    pub feather: f32,
    /// normalisation of the background structure (1 local, 0 global)
    pub bg_norm: f32,
    /// size multiplier of the background structure band
    pub bg_scale: f32,
    /// depth of the outline groove carved around objects
    pub groove: f32,
    /// width of the outline groove in px
    pub groove_w: f32,
    /// internal: pixel-size factor of the working resolution (set by apply)
    pub px_scale: f32,
}

impl Default for ObjectParams {
    fn default() -> Self {
        // tuned jointly against four reference reliefs (tools/ref/clean_tune.py,
        // Depth Anything V2 Small)
        Self {
            obj_lift: 0.0567,
            obj_depth: 0.8501,
            obj_inflate: 0.30,
            inflate_radius: 0.03,
            bg_depth: 1.1548,
            bg_struct: 0.2913,
            obj_struct: 0.0149,
            obj_smooth: 0.5227,
            feather: 1.9573,
            bg_norm: 0.4913,
            bg_scale: 1.4243,
            groove: 0.1863,
            groove_w: 2.8709,
            px_scale: 1.0,
        }
    }
}
fn minmax(v: &[f32]) -> (f32, f32) {
    v.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)))
}

/// Fill `known == false` pixels by harmonic (membrane) interpolation of the
/// known ones, solved on a coarse grid and upsampled.
fn inpaint(values: &Mat<f32>, known: &[bool]) -> Mat<f32> {
    let (h, w) = (values.h, values.w);
    let scale = (256.0 / h.max(w) as f32).min(1.0);
    let (sh, sw) = (((h as f32 * scale) as usize).max(4), ((w as f32 * scale) as usize).max(4));
    let kv = Mat::from_vec(h, w, 1, values.data.iter().zip(known).map(|(&v, &k)| if k { v } else { 0.0 }).collect());
    let kw = Mat::from_vec(h, w, 1, known.iter().map(|&k| k as u8 as f32).collect());
    let sv = cv::resize::resize_f32(&kv, sw, sh, Interp::Area);
    let swt = cv::resize::resize_f32(&kw, sw, sh, Interp::Area);
    let fixed: Vec<bool> = swt.data.iter().map(|&a| a > 0.5).collect();
    let mut cur: Vec<f32> = sv.data.iter().zip(swt.data.iter()).map(|(&v, &a)| if a > 1e-3 { v / a } else { 0.0 }).collect();
    let mean_known = {
        let (s, n) = cur.iter().zip(&fixed).filter(|(_, &f)| f).fold((0f64, 0usize), |(s, n), (&v, _)| (s + v as f64, n + 1));
        if n > 0 {
            (s / n as f64) as f32
        } else {
            0.0
        }
    };
    for (v, &f) in cur.iter_mut().zip(&fixed) {
        if !f {
            *v = mean_known;
        }
    }
    let mut next = cur.clone();
    for _ in 0..600 {
        next.par_chunks_mut(sw).enumerate().for_each(|(y, o)| {
            for x in 0..sw {
                let i = y * sw + x;
                if fixed[i] {
                    o[x] = cur[i];
                    continue;
                }
                let mut s = 0f32;
                let mut n = 0f32;
                if y > 0 {
                    s += cur[i - sw];
                    n += 1.0;
                }
                if y + 1 < sh {
                    s += cur[i + sw];
                    n += 1.0;
                }
                if x > 0 {
                    s += cur[i - 1];
                    n += 1.0;
                }
                if x + 1 < sw {
                    s += cur[i + 1];
                    n += 1.0;
                }
                o[x] = s / n;
            }
        });
        std::mem::swap(&mut cur, &mut next);
    }
    let up = cv::resize::resize_f32(&Mat::from_vec(sh, sw, 1, cur), w, h, Interp::Linear);
    Mat::from_vec(h, w, 1, (0..h * w).map(|i| if known[i] { values.data[i] } else { up.data[i] }).collect())
}

/// 4-connected components of a boolean mask; returns labels (0 = none) and count.
pub fn components(mask: &[bool], h: usize, w: usize) -> (Vec<u32>, u32) {
    let mut lab = vec![0u32; h * w];
    let mut n = 0u32;
    let mut stack = Vec::new();
    for s in 0..h * w {
        if !mask[s] || lab[s] != 0 {
            continue;
        }
        n += 1;
        lab[s] = n;
        stack.push(s);
        while let Some(p) = stack.pop() {
            let (y, x) = (p / w, p % w);
            let mut push = |q: usize| {
                if mask[q] && lab[q] == 0 {
                    lab[q] = n;
                    stack.push(q);
                }
            };
            if y > 0 {
                push(p - w);
            }
            if y + 1 < h {
                push(p + w);
            }
            if x > 0 {
                push(p - 1);
            }
            if x + 1 < w {
                push(p + 1);
            }
        }
    }
    (lab, n)
}

/// Zero-mean, contrast-normalised band-pass of the photo in [-1, 1].
fn structure(gray: &Mat<f32>, s0: f64, s1: f64, sn: f64) -> Vec<f32> {
    structure_norm(gray, s0, s1, sn, 1.0)
}

/// Band-pass with blendable normalisation: alpha = 1 normalises by the local
/// contrast (every texture equally strong), alpha = 0 by the global contrast
/// (smooth areas stay smooth, busy areas stay busy).
fn structure_norm(gray: &Mat<f32>, s0: f64, s1: f64, sn: f64, alpha: f32) -> Vec<f32> {
    let a = gaussian_blur_f32(gray, 0, s0, Border::Reflect101);
    let b = gaussian_blur_f32(gray, 0, s1, Border::Reflect101);
    let bp: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(a, b)| a - b).collect();
    let sq = Mat::from_vec(gray.h, gray.w, 1, bp.iter().map(|v| v * v).collect());
    let var = gaussian_blur_f32(&sq, 0, sn, Border::Reflect101);
    let g = (sq.data.iter().map(|&v| v as f64).sum::<f64>() / sq.len() as f64).sqrt() as f32 + 0.04;
    let alpha = alpha.clamp(0.0, 1.0);
    bp.iter()
        .zip(var.data.iter())
        .map(|(&b, &v)| {
            let local = v.sqrt() + 0.04;
            (b / (local.powf(alpha) * g.powf(1.0 - alpha)) * 0.6).tanh()
        })
        .collect()
}

/// Build the layered relief. Returns (height in [0,1], object mask in [0,1]).
pub fn layered_height(depth: &Mat<f32>, photo: &Mat<u8>, prob: &Mat<f32>, p: &ObjectParams) -> (Mat<f32>, Mat<f32>) {
    let (h, w) = (depth.h, depth.w);
    let n = h * w;
    let prob = if prob.h == h && prob.w == w { prob.clone() } else { cv::resize::resize_f32(prob, w, h, Interp::Linear) };
    let photo = if photo.h == h && photo.w == w { photo.clone() } else { cv::resize::resize_u8(photo, w, h, Interp::Area) };
    let gray = Mat::from_vec(h, w, 1, photo.data.iter().map(|&v| v as f32 / 255.0).collect());
    let hard: Vec<bool> = prob.data.iter().map(|&v| v > 0.5).collect();

    // --- background layer: depth outside objects, inpainted behind them
    let bg_known: Vec<bool> = prob.data.iter().map(|&v| v < 0.2).collect();
    let bg = inpaint(depth, &bg_known);
    let r = p.px_scale.max(0.05) as f64;
    let bg = gaussian_blur_f32(&bg, 0, (1.5 * r).max(0.3), Border::Reflect101);
    let (bmn, bmx) = minmax(&bg.data);
    let bsc = p.bg_scale.max(0.2) as f64;
    let bstruct = structure_norm(&gray, (1.5 * bsc * r).max(0.3), 10.0 * bsc * r, 25.0 * bsc * r, p.bg_norm);
    let bg_h: Vec<f32> = (0..n).map(|i| (bg.data[i] - bmn) / (bmx - bmn + 1e-8) * p.bg_depth + bstruct[i] * p.bg_struct).collect();

    // --- objects
    let (lab, nobj) = components(&hard, h, w);
    let mut obj_h = vec![0f32; n];
    if nobj > 0 {
        // smoothed network depth inside objects (spike removal)
        let ds = gaussian_blur_f32(depth, 0, (p.obj_smooth as f64 * r).max(0.3), Border::Reflect101);
        // per-pixel distance to the silhouette for the rounded volume; the
        // image border is NOT a silhouette (objects cut by the frame
        // continue), so the mask is replicated into a wide margin first
        let m = (h.max(w) / 4).max(16);
        let (ph, pw) = (h + 2 * m, w + 2 * m);
        let mut wide = vec![false; ph * pw];
        for y in 0..ph {
            for x in 0..pw {
                let sy = (y as i64 - m as i64).clamp(0, h as i64 - 1) as usize;
                let sx = (x as i64 - m as i64).clamp(0, w as i64 - 1) as usize;
                wide[y * pw + x] = hard[sy * w + sx];
            }
        }
        let wd = nd::distance_transform_edt(&wide, ph, pw);
        let dist: Vec<f64> = (0..h * w).map(|i| wd[(i / w + m) * pw + i % w + m]).collect();
        let ostruct = structure(&gray, (0.8 * r).max(0.3), (4.0 * r).max(0.6), (10.0 * r).max(1.0));
        for k in 1..=nobj {
            let idx: Vec<usize> = (0..n).filter(|&i| lab[i] == k).collect();
            if idx.len() < 50 {
                continue;
            }
            let mut v: Vec<f64> = idx.iter().map(|&i| ds.data[i] as f64).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            let lo = nd::percentile_sorted(&v, 3.0) as f32;
            let hi = nd::percentile_sorted(&v, 99.0) as f32;
            let (mut y0, mut y1, mut x0, mut x1) = (h, 0, w, 0);
            for &i in &idx {
                y0 = y0.min(i / w);
                y1 = y1.max(i / w);
                x0 = x0.min(i % w);
                x1 = x1.max(i % w);
            }
            let size = ((y1 - y0 + 1).min(x1 - x0 + 1)) as f32;
            let r = (size * p.inflate_radius).max(3.0);
            // the object sits on the background directly behind it
            let base: f32 = idx.iter().map(|&i| bg_h[i]).sum::<f32>() / idx.len() as f32;
            for &i in &idx {
                let dd = dist[i] as f32;
                let inflate = (dd / r).min(1.0);
                let inflate = (inflate * (2.0 - inflate)).sqrt(); // circular profile
                let rel = ((ds.data[i] - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.05);
                obj_h[i] = base + p.obj_lift * inflate + p.obj_depth * rel * inflate + p.obj_struct * ostruct[i] * inflate;
            }
        }
    }

    // --- compose with a feathered mask
    let m = Mat::from_vec(h, w, 1, hard.iter().map(|&b| b as u8 as f32).collect());
    let m = gaussian_blur_f32(&m, 0, (p.feather as f64 * r).max(0.3), Border::Reflect101);
    let mut out: Vec<f32> = (0..n).map(|i| bg_h[i] * (1.0 - m.data[i]) + obj_h[i].max(bg_h[i]) * m.data[i]).collect();
    // outline groove just outside each object
    if p.groove > 0.0 && nobj > 0 {
        let outside: Vec<bool> = hard.iter().map(|&b| !b).collect();
        let od = nd::distance_transform_edt(&outside, h, w);
        let gw = (p.groove_w * p.px_scale).max(0.3);
        for i in 0..n {
            if outside[i] {
                let d = od[i] as f32 / gw;
                if d < 3.0 {
                    out[i] -= p.groove * (-d * d).exp();
                }
            }
        }
    }
    // final spike guard: tiny median-like clamp against 3x3 neighbourhood
    let src = out.clone();
    out.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
        for x in 0..w {
            let mut nb = [0f32; 8];
            let mut c = 0;
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    if yy != y || xx != x {
                        nb[c] = src[yy * w + xx];
                        c += 1;
                    }
                }
            }
            let (lo, hi) = nb[..c].iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
            o[x] = src[y * w + x].clamp(lo, hi);
        }
    });
    let (mn, mx) = minmax(&out);
    let hm = Mat::from_vec(h, w, 1, out.iter().map(|&v| (v - mn) / (mx - mn + 1e-8)).collect());
    (hm, m)
}

/// Clay render of the height plus fine per-object detail -> detail texture.
pub fn detail_texture(height: &Mat<f32>, photo: &Mat<u8>, mask: &Mat<f32>, p: &ObjectParams) -> Mat<u8> {
    let (h, w) = (height.h, height.w);
    let photo = if photo.h == h && photo.w == w { photo.clone() } else { cv::resize::resize_u8(photo, w, h, Interp::Area) };
    let gray = Mat::from_vec(h, w, 1, photo.data.iter().map(|&v| v as f32 / 255.0).collect());
    let r = p.px_scale.max(0.05) as f64;
    let fine = structure(&gray, (0.8 * r).max(0.3), (4.0 * r).max(0.6), (10.0 * r).max(1.0));
    let bsc = p.bg_scale.max(0.2) as f64;
    let mid = structure_norm(&gray, (1.5 * bsc * r).max(0.3), 10.0 * bsc * r, 25.0 * bsc * r, p.bg_norm);
    let d = gaussian_blur_f32(height, 0, r.max(0.3), Border::Reflect101);
    let zs = 0.06 * w as f32;
    let l = {
        let v = [-0.45f32, 0.55, 0.70];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    };
    let mut out = Mat::<u8>::new(h, w, 1);
    let ks = (p.obj_struct * 12.0).min(0.5);
    let kb = (p.bg_struct * 3.0).min(0.5);
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
        let (ym, yp) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for x in 0..w {
            let (xm, xp) = (x.saturating_sub(1), (x + 1).min(w - 1));
            let gx = (d.data[y * w + xp] - d.data[y * w + xm]) * 0.5 * zs;
            let gy = (d.data[yp * w + x] - d.data[ym * w + x]) * 0.5 * zs;
            let nl = (gx * gx + gy * gy + 1.0).sqrt();
            let lam = ((-gx * l[0] + gy * l[1] + l[2]) / nl).max(0.0);
            let i = y * w + x;
            let m = mask.data[i];
            let s = 0.3 + 0.7 * lam + (fine[i] * ks * m + mid[i] * kb * (1.0 - m)) * 0.5;
            o[x] = (s.clamp(0.0, 1.0) * 255.0) as u8;
        }
    });
    out
}

/// Full object mode: returns (height in [0, 1], detail texture) at the
/// working resolution (<= 1400 px, the relief grid is ~1000 px). Pixel-sized
/// parameters refer to the (upscaled) photo resolution.
pub fn compose(depth_float: &Mat<f32>, photo: &Mat<u8>, prob: &Mat<f32>, p: &ObjectParams) -> (Mat<f32>, Mat<u8>) {
    let (out_w, out_h) = (photo.w, photo.h);
    let r = (1400.0 / out_w.max(out_h) as f32).min(1.0);
    let (ww, wh) = (((out_w as f32 * r).round() as usize).max(8), ((out_h as f32 * r).round() as usize).max(8));
    let mut q = *p;
    q.px_scale = r;
    let d = cv::resize::resize_f32(depth_float, ww, wh, Interp::Cubic);
    let ph = cv::resize::resize_u8(photo, ww, wh, Interp::Area);
    let (hm, m) = layered_height(&d, &ph, prob, &q);
    let tex = detail_texture(&hm, &ph, &m, &q);
    (hm, tex)
}