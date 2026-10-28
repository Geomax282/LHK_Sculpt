//! Separable float filtering as in OpenCV's FilterEngine (GaussianBlur on
//! CV_32F), plus Laplacian/Sobel on 8U with CV_64F output.

use crate::img::Mat;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Border {
    /// BORDER_REFLECT_101 (OpenCV default): gfedcb|abcdefgh|gfedcba
    Reflect101,
    /// BORDER_REPLICATE: aaaaaa|abcdefgh|hhhhhhh
    Replicate,
}

#[inline]
pub fn border_idx(p: i64, len: i64, b: Border) -> usize {
    if len == 1 {
        return 0;
    }
    let mut p = p;
    match b {
        Border::Replicate => p.clamp(0, len - 1) as usize,
        Border::Reflect101 => {
            loop {
                if p < 0 {
                    p = -p;
                } else if p >= len {
                    p = 2 * len - 2 - p;
                } else {
                    break;
                }
            }
            p as usize
        }
    }
}

/// `getGaussianKernelBitExact` result cast to float (CV_32F kernel).
pub fn gaussian_kernel_f32(n: usize, sigma: f64) -> Vec<f32> {
    if sigma <= 0.0 {
        match n {
            1 => return vec![1.0],
            3 => return vec![0.25, 0.5, 0.25],
            5 => return vec![0.0625, 0.25, 0.375, 0.25, 0.0625],
            7 => return vec![0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
            _ => {}
        }
    }
    let sigma_x = if sigma > 0.0 { sigma } else { (n as f64).mul_add(0.15, 0.35) };
    let scale2x = -0.125 / (sigma_x * sigma_x);
    let n2 = (n - 1) / 2;
    let mut values = vec![0f64; n2 + 1];
    let mut sum = 0f64;
    let mut x = 1i64 - n as i64;
    for v in values.iter_mut().take(n2) {
        let t = (((x * x) as f64) * scale2x).exp();
        *v = t;
        sum += t;
        x += 2;
    }
    sum *= 2.0;
    sum += 1.0;
    if n % 2 == 0 {
        sum += 1.0;
    }
    let mul1 = 1.0 / sum;
    let mut res = vec![0f64; n];
    if n % 2 == 1 {
        for i in 0..n2 {
            let t = values[i] * mul1;
            res[i] = t;
            res[n - 1 - i] = t;
        }
        res[n2] = mul1;
    } else {
        for i in 0..n2 {
            let t = values[i] * mul1;
            res[i] = t;
            res[n - 1 - i] = t;
        }
        res[n2] = mul1;
        res[n2 + 1] = mul1;
    }
    res.into_iter().map(|v| v as f32).collect()
}

/// Ksize as computed by GaussianBlur when ksize is (0,0):
/// `cvRound(sigma*(depth == CV_8U ? 3 : 4)*2 + 1) | 1`.
pub fn auto_ksize_f32(sigma: f64) -> usize {
    (((sigma * 4.0 * 2.0 + 1.0).round_ties_even() as i64) | 1) as usize
}

/// Separable symmetric filter on a single-channel float image:
/// row pass (taps in order) then symmetric column pass (centre + pairs).
pub fn sep_filter_f32(src: &Mat<f32>, kx: &[f32], ky: &[f32], border: Border) -> Mat<f32> {
    assert_eq!(src.c, 1);
    let (h, w) = (src.h, src.w);
    let ax = (kx.len() / 2) as i64;
    let ay = (ky.len() / 2) as i64;
    // Row pass
    let mut tmp = vec![0f32; h * w];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = src.row(y);
        let n = kx.len();
        // padded row
        let padded: Vec<f32> = (-(ax)..(w as i64 + ax)).map(|i| row[border_idx(i, w as i64, border)]).collect();
        if n == 3 {
            let (k0, k1) = (kx[1], kx[2]);
            for x in 0..w {
                let s = &padded[x..x + 3];
                out[x] = s[1] * k0 + (s[0] + s[2]) * k1;
            }
        } else if n == 5 {
            let (k0, k1, k2) = (kx[2], kx[3], kx[4]);
            for x in 0..w {
                let s = &padded[x..x + 5];
                out[x] = s[2] * k0 + (s[1] + s[3]) * k1 + (s[0] + s[4]) * k2;
            }
        } else {
            for x in 0..w {
                let s = &padded[x..x + n];
                let mut acc = 0f32;
                for k in 0..n {
                    acc += kx[k] * s[k];
                }
                out[x] = acc;
            }
        }
    });
    // Column pass (symmetric): s = f0*S0 + sum_k f_k*(S_k + S_-k)
    let mut out = Mat::<f32>::new(h, w, 1);
    let c = ay as usize;
    out.data.par_chunks_mut(w).enumerate().for_each(|(y, orow)| {
        let rows: Vec<&[f32]> = (-(ay)..=ay)
            .map(|d| {
                let sy = border_idx(y as i64 + d, h as i64, border);
                &tmp[sy * w..sy * w + w]
            })
            .collect();
        for x in 0..w {
            let mut s = ky[c] * rows[c][x];
            for k in 1..=c {
                s += ky[c + k] * (rows[c + k][x] + rows[c - k][x]);
            }
            orow[x] = s;
        }
    });
    out
}

/// cv2.GaussianBlur(src_f32, (k,k), sigma, borderType)
pub fn gaussian_blur_f32(src: &Mat<f32>, ksize: usize, sigma: f64, border: Border) -> Mat<f32> {
    let k = if ksize == 0 { auto_ksize_f32(sigma) } else { ksize };
    let kern = gaussian_kernel_f32(k, sigma);
    sep_filter_f32(src, &kern, &kern, border)
}

/// cv2.Laplacian(u8, CV_64F) with ksize=1 (kernel 0 1 0 / 1 -4 1 / 0 1 0).
pub fn laplacian_u8_f64(src: &Mat<u8>) -> Vec<f64> {
    let (h, w) = (src.h as i64, src.w as i64);
    let b = Border::Reflect101;
    let mut out = vec![0f64; (h * w) as usize];
    out.par_chunks_mut(w as usize).enumerate().for_each(|(y, o)| {
        let y = y as i64;
        for x in 0..w {
            let g = |yy: i64, xx: i64| src.data[border_idx(yy, h, b) * w as usize + border_idx(xx, w, b)] as f64;
            o[x as usize] = g(y - 1, x) + g(y + 1, x) + g(y, x - 1) + g(y, x + 1) - 4.0 * g(y, x);
        }
    });
    out
}

/// cv2.Sobel(u8, CV_64F, dx, dy, ksize=3). Returns (gx, gy).
pub fn sobel3_u8_f64(src: &Mat<u8>) -> (Vec<f64>, Vec<f64>) {
    let (h, w) = (src.h as i64, src.w as i64);
    let b = Border::Reflect101;
    let n = (h * w) as usize;
    let mut gx = vec![0f64; n];
    let mut gy = vec![0f64; n];
    gx.par_chunks_mut(w as usize).zip(gy.par_chunks_mut(w as usize)).enumerate().for_each(|(y, (ox, oy))| {
        let y = y as i64;
        let g = |yy: i64, xx: i64| src.data[border_idx(yy, h, b) * w as usize + border_idx(xx, w, b)] as f64;
        for x in 0..w {
            let a00 = g(y - 1, x - 1);
            let a01 = g(y - 1, x);
            let a02 = g(y - 1, x + 1);
            let a10 = g(y, x - 1);
            let a12 = g(y, x + 1);
            let a20 = g(y + 1, x - 1);
            let a21 = g(y + 1, x);
            let a22 = g(y + 1, x + 1);
            ox[x as usize] = (a02 - a00) + 2.0 * (a12 - a10) + (a22 - a20);
            oy[x as usize] = (a20 - a00) + 2.0 * (a21 - a01) + (a22 - a02);
        }
    });
    (gx, gy)
}
