//! Relief mesh from the object-mode height map: a regular grid surface,
//! edge-aware smoothing, fine detail, per-object spike guard and
//! background sharpness, plus a closed solid (side walls, flat bottom).
//!
//! All lengths are model units. By default the model covers an XY area of
//! `MODEL_AREA` (about 700 x 1070 for a 2:3 portrait), so heights are
//! independent of the image and grid resolution.

use crate::cv::filter::{gaussian_blur_f32, Border};
use crate::cv::{self, Interp};
use crate::img::Mat;
use rayon::prelude::*;

/// Default XY area of the model in square units.
pub const MODEL_AREA: f32 = 748_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReliefSettings {
    /// height of the base relief (model units)
    pub height: f32,
    /// amplitude of the fine detail texture (model units)
    pub detail: f32,
    /// edge-aware smoothing of the base relief, 0..100
    pub smoothing: f32,
    /// per-object spike guard, 0 (off) ..100
    pub spike_guard: f32,
    /// background-only sharpness in % (100 = unchanged)
    pub bg_sharpness: f32,
    /// base thickness below the lowest point, fraction of the relief range
    pub base: f32,
    /// quicker approximations while a slider is dragged
    pub fast: bool,
}

impl Default for ReliefSettings {
    fn default() -> Self {
        // tuned against reference reliefs (tools/ref/autotune.py)
        Self { height: 21.93, detail: 3.81, smoothing: 20.87, spike_guard: 75.0, bg_sharpness: 42.95, base: 0.15, fast: false }
    }
}

/// Objects on the grid: label per vertex (0 = background) and the weight
/// of background-only filters (0 on and right next to objects).
struct ObjectLayers {
    labels: Vec<u32>,
    count: u32,
    bg_weight: Vec<f32>,
}

pub struct ReliefMesh {
    pub width: usize,
    pub height: usize,
    /// base relief in [0, 1]
    base: Vec<f32>,
    /// detail texture, zero-centred in [-0.5, 0.5]
    tex: Vec<f32>,
    /// grid spacing in model units
    cell: f32,
    objects: Option<ObjectLayers>,
}

/// Grid size with about `vertices` vertices and the aspect of `w` x `h`.
fn grid_size(w: usize, h: usize, vertices: usize) -> (usize, usize) {
    let s = (vertices as f64 / (w * h) as f64).sqrt();
    (((w as f64 * s).round() as usize).max(2), ((h as f64 * s).round() as usize).max(2))
}

fn resize_to(m: &Mat<f32>, w: usize, h: usize) -> Mat<f32> {
    if m.w == w && m.h == h {
        return m.clone();
    }
    let interp = if w * h < m.w * m.h { Interp::Area } else { Interp::Cubic };
    cv::resize::resize_f32(m, w, h, interp)
}

impl ReliefMesh {
    /// `height01`: base relief in [0, 1]; `tex`: detail texture (gray);
    /// `prob`: object probability (any resolution); `triangles`: target
    /// triangle count of the top surface.
    pub fn new(height01: &Mat<f32>, tex: &Mat<u8>, prob: Option<&Mat<f32>>, triangles: usize) -> Self {
        let (w, h) = grid_size(height01.w, height01.h, (triangles / 2).max(16));
        let base = resize_to(height01, w, h).data;
        let t = Mat::from_vec(tex.h, tex.w, 1, tex.data.iter().map(|&v| v as f32 / 255.0 - 0.5).collect());
        let tex = resize_to(&t, w, h).data;
        let cell = (MODEL_AREA / ((w - 1) * (h - 1)) as f32).sqrt();
        let objects = prob.map(|p| ObjectLayers::new(p, w, h));
        Self { width: w, height: h, base, tex, cell, objects }
    }

    /// Height of every grid vertex (model units).
    pub fn surface(&self, s: &ReliefSettings) -> Vec<f32> {
        let (w, h) = (self.width, self.height);
        let b = if s.smoothing > 0.0 { edge_aware_smooth(&self.base, h, w, s.smoothing, s.fast) } else { self.base.clone() };
        let mut z: Vec<f32> = b.par_iter().zip(self.tex.par_iter()).map(|(&b, &t)| b * s.height + t * s.detail).collect();
        if s.spike_guard > 0.0 || (s.bg_sharpness - 100.0).abs() > 0.05 {
            object_postfilter(&mut z, h, w, self.objects.as_ref(), s);
        }
        z
    }

    /// Face list (top surface first; then walls; then bottom).
    pub fn faces(&self, walls: bool, bottom: bool) -> Vec<[u32; 3]> {
        let (w, h) = (self.width, self.height);
        let mut f = Vec::with_capacity(2 * (w - 1) * (h - 1) + 8 * (w + h));
        for y in 0..h - 1 {
            for x in 0..w - 1 {
                let i00 = (y * w + x) as u32;
                let i01 = i00 + 1;
                let i10 = i00 + w as u32;
                let i11 = i10 + 1;
                f.push([i00, i10, i01]);
                f.push([i01, i10, i11]);
            }
        }
        if walls || bottom {
            let n = (w * h) as u32;
            let p = self.perimeter().len() as u32;
            // wall copies: top ring n..n+p, bottom ring n+p..n+2p, centre n+2p
            for k in 0..p {
                let k1 = (k + 1) % p;
                let (at, bt, ab, bb) = (n + k, n + k1, n + p + k, n + p + k1);
                f.push([at, ab, bb]);
                f.push([at, bb, bt]);
            }
            if bottom {
                let c = n + 2 * p;
                for k in 0..p {
                    f.push([c, n + p + (k + 1) % p, n + p + k]);
                }
            }
        }
        f
    }

    /// Vertices matching `faces(walls, bottom)`; centred at the origin.
    pub fn vertices(&self, z: &[f32], walls: bool, bottom: bool, s: &ReliefSettings) -> Vec<[f32; 3]> {
        let (w, h) = (self.width, self.height);
        let (cx, cy) = ((w - 1) as f32 * 0.5, (h - 1) as f32 * 0.5);
        let c = self.cell;
        let mut v: Vec<[f32; 3]> = (0..w * h).into_par_iter().map(|i| [((i % w) as f32 - cx) * c, (cy - (i / w) as f32) * c, z[i]]).collect();
        if walls || bottom {
            let (zmn, zmx) = minmax(z);
            let zb = zmn - ((zmx - zmn) * s.base).max(0.5);
            let ring = self.perimeter();
            for &i in &ring {
                v.push(v[i]);
            }
            for &i in &ring {
                let p = v[i];
                v.push([p[0], p[1], zb]);
            }
            if bottom {
                v.push([0.0, 0.0, zb]);
            }
        }
        let (zmn, zmx) = v.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), p| (a.min(p[2]), b.max(p[2])));
        let zc = (zmn + zmx) * 0.5;
        v.par_iter_mut().for_each(|p| p[2] -= zc);
        v
    }

    /// Border vertices counter-clockwise seen from above (+z).
    fn perimeter(&self) -> Vec<usize> {
        let (w, h) = (self.width, self.height);
        let mut r = Vec::with_capacity(2 * (w + h));
        for x in 0..w - 1 {
            r.push((h - 1) * w + x); // bottom row, left to right
        }
        for y in (1..h).rev() {
            r.push(y * w + w - 1); // right column, upwards
        }
        for x in (1..w).rev() {
            r.push(x); // top row, right to left
        }
        for y in 0..h - 1 {
            r.push(y * w); // left column, downwards
        }
        r
    }
}

fn minmax(v: &[f32]) -> (f32, f32) {
    v.par_iter().fold(|| (f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x))).reduce(|| (f32::INFINITY, f32::NEG_INFINITY), |(a, b), (c, d)| (a.min(c), b.max(d)))
}

/// k-th smallest value (approximate on large inputs: strided sample).
fn quantile(v: &[f32], q: f32) -> f32 {
    let step = (v.len() / 200_000).max(1);
    let mut s: Vec<f32> = v.iter().step_by(step).cloned().collect();
    let k = ((s.len() - 1) as f32 * q) as usize;
    *s.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
}

/// Gaussian smoothing that fades out on steep slopes (object outlines and
/// terrain steps stay crisp). `amount` 0..100.
fn edge_aware_smooth(b: &[f32], h: usize, w: usize, amount: f32, fast: bool) -> Vec<f32> {
    let sigma = (amount / 100.0 * 2.6).max(0.05) as f64;
    let sigma = if fast { sigma.min(1.5) } else { sigma };
    let mut g = vec![0f32; h * w];
    g.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
        let (ym, yp) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for x in 0..w {
            let (xm, xp) = (x.saturating_sub(1), (x + 1).min(w - 1));
            let gx = b[y * w + xp] - b[y * w + xm];
            let gy = b[yp * w + x] - b[ym * w + x];
            o[x] = (gx * gx + gy * gy).sqrt();
        }
    });
    let g0 = (0.16 * quantile(&g, 0.999)).max(1e-6);
    let bl = gaussian_blur_f32(&Mat::from_vec(h, w, 1, b.to_vec()), 0, sigma, Border::Reflect101);
    (0..h * w)
        .into_par_iter()
        .map(|i| {
            let r = g[i] / g0;
            let wt = 1.0 / (1.0 + r * r * r * r);
            b[i] + (bl.data[i] - b[i]) * wt
        })
        .collect()
}

impl ObjectLayers {
    fn new(prob: &Mat<f32>, w: usize, h: usize) -> Self {
        let pr = cv::resize::resize_f32(prob, w, h, Interp::Linear);
        let hard: Vec<bool> = pr.data.iter().map(|&v| v > 0.5).collect();
        let (mut lab, n) = crate::objects::components(&hard, h, w);
        // tiny blobs belong to the background
        let mut sizes = vec![0usize; n as usize + 1];
        for &l in &lab {
            sizes[l as usize] += 1;
        }
        let min = (h * w / 4000).max(64);
        let mut remap = vec![0u32; n as usize + 1];
        let mut count = 0;
        for l in 1..=n as usize {
            if sizes[l] >= min {
                count += 1;
                remap[l] = count;
            }
        }
        for l in lab.iter_mut() {
            *l = remap[*l as usize];
        }
        let ind = Mat::from_vec(h, w, 1, lab.iter().map(|&l| (l > 0) as u8 as f32).collect());
        let soft = gaussian_blur_f32(&ind, 0, 1.5, Border::Reflect101);
        let bg_weight = soft.data.iter().map(|&m| ((0.5 - m) * 2.0).clamp(0.0, 1.0)).collect();
        Self { labels: lab, count, bg_weight }
    }
}

/// Identity below 0.6 t, then compresses smoothly; |result| < t.
fn soft_clip(x: f32, t: f32) -> f32 {
    let a = 0.6 * t;
    let ax = x.abs();
    if ax <= a {
        return x;
    }
    let b = t - a;
    x.signum() * (a + b * ((ax - a) / b).tanh())
}

/// Gaussian mean of `z` over the pixels of label `l` only (normalised
/// convolution), evaluated inside the label's bounding box.
fn region_mean(z: &[f32], lab: &[u32], l: u32, h: usize, w: usize, bbox: (usize, usize, usize, usize), sigma: f64, out: &mut [f32]) {
    let pad = (3.0 * sigma).ceil() as usize + 1;
    let (y0, y1, x0, x1) = (bbox.0.saturating_sub(pad), (bbox.1 + pad + 1).min(h), bbox.2.saturating_sub(pad), (bbox.3 + pad + 1).min(w));
    let (ch, cw) = (y1 - y0, x1 - x0);
    let mut zm = vec![0f32; ch * cw];
    let mut m = vec![0f32; ch * cw];
    for y in 0..ch {
        for x in 0..cw {
            let i = (y + y0) * w + x + x0;
            if lab[i] == l {
                zm[y * cw + x] = z[i];
                m[y * cw + x] = 1.0;
            }
        }
    }
    let bz = gaussian_blur_f32(&Mat::from_vec(ch, cw, 1, zm), 0, sigma, Border::Reflect101);
    let bm = gaussian_blur_f32(&Mat::from_vec(ch, cw, 1, m), 0, sigma, Border::Reflect101);
    for y in 0..ch {
        for x in 0..cw {
            let i = (y + y0) * w + x + x0;
            if lab[i] == l {
                let k = y * cw + x;
                out[i] = if bm.data[k] > 1e-4 { bz.data[k] / bm.data[k] } else { z[i] };
            }
        }
    }
}

/// Background sharpness (background only) and a per-object spike guard:
/// every object (and the background) is compared with its own smooth
/// envelope; deviations far beyond the object's typical detail are
/// compressed and nothing may rise above its highest plausible point.
fn object_postfilter(z: &mut [f32], h: usize, w: usize, layers: Option<&ObjectLayers>, p: &ReliefSettings) {
    let single;
    let ly = match layers {
        Some(l) => l,
        None => {
            single = ObjectLayers { labels: vec![0; h * w], count: 0, bg_weight: vec![1.0; h * w] };
            &single
        }
    };
    let n = h * w;
    let mut bbox = vec![(usize::MAX, 0usize, usize::MAX, 0usize); ly.count as usize + 1];
    for i in 0..n {
        let b = &mut bbox[ly.labels[i] as usize];
        let (y, x) = (i / w, i % w);
        b.0 = b.0.min(y);
        b.1 = b.1.max(y);
        b.2 = b.2.min(x);
        b.3 = b.3.max(x);
    }
    let size = h.max(w) as f64;
    let mut env = vec![0f32; n];

    // background sharpness: scale the fine background detail only
    let gain = (p.bg_sharpness / 100.0).clamp(0.0, 3.0);
    if layers.is_some() && (gain - 1.0).abs() > 1e-3 && bbox[0].0 != usize::MAX {
        region_mean(z, &ly.labels, 0, h, w, bbox[0], (0.0025 * size).max(1.0), &mut env);
        for i in 0..n {
            if ly.labels[i] == 0 {
                z[i] += (gain - 1.0) * (z[i] - env[i]) * ly.bg_weight[i];
            }
        }
    }

    if p.spike_guard <= 0.0 {
        return;
    }
    let s = (p.spike_guard / 100.0).clamp(0.0, 1.0);
    let k = 2.5 + (1.0 - s) * 10.0;
    let (zmn, zmx) = minmax(z);
    let floor = (zmx - zmn) * 1e-4 + 1e-6;
    let sigma = (0.006 * size).max(2.0);
    // corrections must move a point towards its 5x5 neighbourhood (never
    // create new steps where the object mask and the depth disagree)
    let orig = z.to_vec();
    let med = median5(&orig, h, w);
    for l in 0..=ly.count {
        if bbox[l as usize].0 == usize::MAX {
            continue;
        }
        region_mean(z, &ly.labels, l, h, w, bbox[l as usize], sigma, &mut env);
        let idx: Vec<usize> = (0..n).filter(|&i| ly.labels[i] == l).collect();
        let mut ar: Vec<f32> = idx.iter().map(|&i| (z[i] - env[i]).abs()).collect();
        let mid = ar.len() / 2;
        let mad = *ar.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
        let t = k * 1.4826 * mad + floor;
        // highest plausible point of this object
        let mut ev: Vec<f32> = idx.iter().map(|&i| env[i]).collect();
        let q = ((ev.len() - 1) as f32 * 0.995) as usize;
        let ceil = *ev.select_nth_unstable_by(q, |a, b| a.total_cmp(b)).1 + t;
        for &i in &idx {
            let mut v = env[i] + soft_clip(z[i] - env[i], t);
            if v > ceil {
                v = ceil + soft_clip(v - ceil, t);
            }
            if (v - med[i]).abs() <= (orig[i] - med[i]).abs() {
                z[i] = v;
            } else if (v - orig[i]) * (med[i] - orig[i]) > 0.0 {
                z[i] = med[i];
            }
        }
    }
    // local pass: thin needles (hair tips, mask edges) against their own
    // neighbourhood, relative to the local detail strength
    let med = median5(z, h, w);
    let r: Vec<f32> = z.iter().zip(&med).map(|(a, b)| a - b).collect();
    let r2 = Mat::from_vec(h, w, 1, r.iter().map(|v| v * v).collect());
    let lr = gaussian_blur_f32(&r2, 0, (0.01 * size).max(3.0), Border::Reflect101);
    z.par_iter_mut().enumerate().for_each(|(i, v)| {
        let t = k * lr.data[i].sqrt() + floor;
        *v = med[i] + soft_clip(r[i], t);
    });
}

/// 5x5 median (border: clamped window).
fn median5(z: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
        let mut nb = [0f32; 25];
        for x in 0..w {
            let mut c = 0;
            for yy in y.saturating_sub(2)..(y + 3).min(h) {
                for xx in x.saturating_sub(2)..(x + 3).min(w) {
                    nb[c] = z[yy * w + xx];
                    c += 1;
                }
            }
            let s = &mut nb[..c];
            let m = c / 2;
            o[x] = *s.select_nth_unstable_by(m, |a, b| a.total_cmp(b)).1;
        }
    });
    out
}

/// Area-weighted vertex normals.
pub fn normals(verts: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut acc = vec![[0f32; 3]; verts.len()];
    for f in faces {
        let (a, b, c) = (verts[f[0] as usize], verts[f[1] as usize], verts[f[2] as usize]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        for &i in f {
            let t = &mut acc[i as usize];
            t[0] += n[0];
            t[1] += n[1];
            t[2] += n[2];
        }
    }
    acc.par_iter()
        .map(|v| {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if l < 1e-12 {
                [0.0, 0.0, 1.0]
            } else {
                [v[0] / l, v[1] / l, v[2] / l]
            }
        })
        .collect()
}

/// Uniform scale so the X extent equals `width` (vertices stay centred).
pub fn scale_to_width(verts: &mut [[f32; 3]], width: f32) {
    let (mn, mx) = verts.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), p| (a.min(p[0]), b.max(p[0])));
    if mx > mn && width > 0.0 {
        let s = width / (mx - mn);
        verts.iter_mut().for_each(|p| *p = [p[0] * s, p[1] * s, p[2] * s]);
    }
}

/// Scale Z only so the total height (incl. base) equals `height`.
pub fn scale_z_to(verts: &mut [[f32; 3]], height: f32) {
    let (mn, mx) = verts.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), p| (a.min(p[2]), b.max(p[2])));
    if mx > mn && height > 0.0 {
        let s = height / (mx - mn);
        let c = (mn + mx) * 0.5;
        verts.iter_mut().for_each(|p| p[2] = (p[2] - c) * s);
    }
}
