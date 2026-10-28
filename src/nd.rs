//! numpy / scipy.ndimage semantics needed by the relief pipeline.

use rayon::prelude::*;

/// np.percentile(a, q) with the default 'linear' method.
pub fn percentile_linear(a: &[f64], q: f64) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(|x, y| x.total_cmp(y));
    percentile_sorted(&v, q)
}

pub fn percentile_sorted(v: &[f64], q: f64) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let idx = q / 100.0 * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let t = idx - lo as f64;
    let (a, b) = (v[lo], v[hi]);
    let d = b - a;
    // numpy _lerp: a + d*t, or b - d*(1-t) when t >= 0.5
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    }
}

pub fn mean(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NAN;
    }
    a.par_iter().sum::<f64>() / a.len() as f64
}

/// Population standard deviation (np.std, ddof=0).
pub fn std(a: &[f64]) -> f64 {
    let m = mean(a);
    let v = a.par_iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / a.len() as f64;
    v.sqrt()
}

pub fn std_f32(a: &[f32]) -> f64 {
    let n = a.len() as f64;
    let m = a.par_iter().map(|&x| x as f64).sum::<f64>() / n;
    (a.par_iter().map(|&x| (x as f64 - m) * (x as f64 - m)).sum::<f64>() / n).sqrt()
}

/// np.gradient for a 2-D float32 array: (gy, gx), central differences in the
/// interior and first-order one-sided differences at the edges.
pub fn gradient2d_f32(a: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0f32; h * w];
    let mut gy = vec![0f32; h * w];
    gx.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
        let r = &a[y * w..y * w + w];
        if w == 1 {
            o[0] = 0.0;
            return;
        }
        o[0] = r[1] - r[0];
        for x in 1..w - 1 {
            o[x] = (r[x + 1] - r[x - 1]) / 2.0;
        }
        o[w - 1] = r[w - 1] - r[w - 2];
    });
    gy.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
        for x in 0..w {
            o[x] = if h == 1 {
                0.0
            } else if y == 0 {
                a[w + x] - a[x]
            } else if y == h - 1 {
                a[y * w + x] - a[(y - 1) * w + x]
            } else {
                (a[(y + 1) * w + x] - a[(y - 1) * w + x]) / 2.0
            };
        }
    });
    (gy, gx)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NdMode {
    /// scipy 'nearest': a a a | a b c d | d d d
    Nearest,
    /// scipy 'reflect': c b a | a b c d | d c b
    Reflect,
}

fn ext_index(i: i64, n: i64, mode: NdMode) -> usize {
    match mode {
        NdMode::Nearest => i.clamp(0, n - 1) as usize,
        NdMode::Reflect => {
            let mut i = i;
            let period = 2 * n;
            i = i.rem_euclid(period);
            if i >= n {
                i = period - 1 - i;
            }
            i as usize
        }
    }
}

/// scipy.ndimage.uniform_filter1d on one line (double running sum), result
/// written back as f32.
fn uniform_line(line: &[f32], size: usize, mode: NdMode, out: &mut [f32]) {
    let n = line.len() as i64;
    let size1 = (size / 2) as i64;
    let size2 = size as i64 - size1 - 1;
    let padded: Vec<f64> = (-size1..n + size2).map(|i| line[ext_index(i, n, mode)] as f64).collect();
    let fs = size as f64;
    let mut tmp = 0f64;
    for v in padded.iter().take(size) {
        tmp += *v;
    }
    tmp /= fs;
    out[0] = tmp as f32;
    for ll in 1..n as usize {
        tmp += (padded[ll + size - 1] - padded[ll - 1]) / fs;
        out[ll] = tmp as f32;
    }
}

/// scipy.ndimage.uniform_filter (2-D, float32 in/out): axis 0 then axis 1.
pub fn uniform_filter2d_f32(a: &[f32], h: usize, w: usize, size: usize, mode: NdMode) -> Vec<f32> {
    // axis 0 (columns)
    let mut t = vec![0f32; h * w];
    let cols: Vec<Vec<f32>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let col: Vec<f32> = (0..h).map(|y| a[y * w + x]).collect();
            let mut o = vec![0f32; h];
            uniform_line(&col, size, mode, &mut o);
            o
        })
        .collect();
    for (x, col) in cols.iter().enumerate() {
        for y in 0..h {
            t[y * w + x] = col[y];
        }
    }
    let mut out = vec![0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, o)| uniform_line(&t[y * w..y * w + w], size, mode, o));
    out
}

const EDT_BIG: f64 = 1e30;

/// Exact 1-D squared distance transform (Felzenszwalb & Huttenlocher).
/// Values >= EDT_BIG mean "no feature on this line".
fn dt1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let fq = f[q] + (q * q) as f64;
        loop {
            let p = v[k];
            let s = (fq - (f[p] + (p * p) as f64)) / (2.0 * q as f64 - 2.0 * p as f64);
            if s <= z[k] && k > 0 {
                k -= 1;
                continue;
            }
            if s <= z[k] {
                v[0] = q;
                z[0] = f64::NEG_INFINITY;
                z[1] = f64::INFINITY;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f64::INFINITY;
            }
            break;
        }
    }
    let mut k = 0usize;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        d[q] = if f[p] >= EDT_BIG { EDT_BIG } else { (q as f64 - p as f64).powi(2) + f[p] };
    }
}

/// scipy.ndimage.distance_transform_edt(mask): Euclidean distance of every
/// `true` pixel to the nearest `false` pixel (0 for false pixels).
pub fn distance_transform_edt(mask: &[bool], h: usize, w: usize) -> Vec<f64> {
    let inf = EDT_BIG;
    // columns
    let mut g = vec![0f64; h * w];
    let cols: Vec<Vec<f64>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let f: Vec<f64> = (0..h).map(|y| if mask[y * w + x] { inf } else { 0.0 }).collect();
            let mut d = vec![0f64; h];
            let mut v = vec![0usize; h];
            let mut z = vec![0f64; h + 1];
            dt1d(&f, &mut d, &mut v, &mut z);
            d
        })
        .collect();
    for (x, c) in cols.iter().enumerate() {
        for y in 0..h {
            g[y * w + x] = c[y];
        }
    }
    let mut out = vec![0f64; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
        let f = &g[y * w..y * w + w];
        let mut d = vec![0f64; w];
        let mut v = vec![0usize; w];
        let mut z = vec![0f64; w + 1];
        dt1d(f, &mut d, &mut v, &mut z);
        for x in 0..w {
            o[x] = d[x].sqrt();
        }
    });
    out
}
