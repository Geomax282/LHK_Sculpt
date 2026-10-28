//! Port of OpenCV 4.12 `cv::resize` (imgproc/src/resize.cpp) for the pixel
//! types and interpolation modes LHK_Sculpt uses.
//!
//! The coefficient tables, border clamping, area tables and fixed-point
//! arithmetic follow the C++ implementation so results match `cv2.resize`.

use crate::img::Mat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Linear,
    Cubic,
    Area,
    Lanczos4,
    Nearest,
}

const INTER_RESIZE_COEF_BITS: i32 = 11;
const INTER_RESIZE_COEF_SCALE: f32 = (1 << INTER_RESIZE_COEF_BITS) as f32;

/// Pixel types supported by the float (non fixed-point) code paths.
pub trait FloatPix: Copy + Default + Send + Sync {
    fn to_f(self) -> f32;
    fn from_f(v: f32) -> Self;
}

impl FloatPix for f32 {
    #[inline]
    fn to_f(self) -> f32 {
        self
    }
    #[inline]
    fn from_f(v: f32) -> Self {
        v
    }
}

impl FloatPix for u16 {
    #[inline]
    fn to_f(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_f(v: f32) -> Self {
        sat_u16_f(v)
    }
}

impl FloatPix for u8 {
    #[inline]
    fn to_f(self) -> f32 {
        self as f32
    }
    #[inline]
    fn from_f(v: f32) -> Self {
        sat_u8_f(v)
    }
}

/// `saturate_cast<uchar>(float)`: cvRound (ties to even) then clamp.
#[inline]
pub fn sat_u8_f(v: f32) -> u8 {
    let r = v.round_ties_even();
    if r.is_nan() {
        return 0;
    }
    r.clamp(0.0, 255.0) as u8
}

/// `saturate_cast<ushort>(float)`.
#[inline]
pub fn sat_u16_f(v: f32) -> u16 {
    let r = v.round_ties_even();
    if r.is_nan() {
        return 0;
    }
    r.clamp(0.0, 65535.0) as u16
}

#[inline]
fn sat_u8_i(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline]
fn cv_floor(v: f64) -> i32 {
    v.floor() as i32
}

#[inline]
fn cv_round(v: f64) -> i32 {
    v.round_ties_even() as i32
}

fn interpolate_cubic(x: f32, c: &mut [f32]) {
    const A: f32 = -0.75;
    c[0] = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
    c[1] = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    c[2] = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
    c[3] = 1.0 - c[0] - c[1] - c[2];
}

fn interpolate_lanczos4(x: f32, c: &mut [f32]) {
    const S45: f64 = 0.707_106_781_186_547_524_400_844_362_104_85;
    const CS: [[f64; 2]; 8] = [
        [1.0, 0.0],
        [-S45, -S45],
        [0.0, 1.0],
        [S45, -S45],
        [-1.0, 0.0],
        [S45, S45],
        [0.0, -1.0],
        [-S45, S45],
    ];
    let mut sum = 0f32;
    let y0 = -(x as f64 + 3.0) * std::f64::consts::PI * 0.25;
    let s0 = y0.sin();
    let c0 = y0.cos();
    for i in 0..8 {
        let y0_ = x + 3.0 - i as f32;
        if y0_.abs() >= 1e-6 {
            let y = -(y0_ as f64) * std::f64::consts::PI * 0.25;
            c[i] = ((CS[i][0] * s0 + CS[i][1] * c0) / (y * y)) as f32;
        } else {
            c[i] = 1e30;
        }
        sum += c[i];
    }
    let sum = 1.0 / sum;
    for v in c.iter_mut().take(8) {
        *v *= sum;
    }
}

/// Coefficient table for one axis of the generic (separable) resize.
struct AxisTab {
    ofs: Vec<i32>,     // source index per destination index (pixel units)
    coef: Vec<f32>,    // ksize coefficients per destination index
    xmin: usize,
    xmax: usize,
}

fn build_axis(ssize: usize, dsize: usize, scale: f64, inv_scale: f64, interp: Interp, ksize: usize) -> AxisTab {
    let ksize2 = (ksize / 2) as i32;
    let area_mode = interp == Interp::Area;
    let mut ofs = vec![0i32; dsize];
    let mut coef = vec![0f32; dsize * ksize];
    let mut xmin = 0usize;
    let mut xmax = dsize;
    let mut cbuf = [0f32; 8];
    for dx in 0..dsize {
        let mut fx: f32;
        let mut sx: i32;
        if !area_mode {
            fx = ((dx as f64 + 0.5) * scale - 0.5) as f32;
            sx = (fx as f64).floor() as i32;
            fx -= sx as f32;
        } else {
            sx = cv_floor(dx as f64 * scale);
            fx = ((dx as f64 + 1.0) - (sx as f64 + 1.0) * inv_scale) as f32;
            fx = if fx <= 0.0 { 0.0 } else { fx - (fx as f64).floor() as f32 };
        }
        let not_wide = interp != Interp::Cubic && interp != Interp::Lanczos4;
        if sx < ksize2 - 1 {
            xmin = dx + 1;
            if sx < 0 && not_wide {
                fx = 0.0;
                sx = 0;
            }
        }
        if sx + ksize2 >= ssize as i32 {
            xmax = xmax.min(dx);
            if sx >= ssize as i32 - 1 && not_wide {
                fx = 0.0;
                sx = ssize as i32 - 1;
            }
        }
        ofs[dx] = sx;
        match interp {
            Interp::Cubic => interpolate_cubic(fx, &mut cbuf),
            Interp::Lanczos4 => interpolate_lanczos4(fx, &mut cbuf),
            _ => {
                cbuf[0] = 1.0 - fx;
                cbuf[1] = fx;
            }
        }
        coef[dx * ksize..dx * ksize + ksize].copy_from_slice(&cbuf[..ksize]);
    }
    AxisTab { ofs, coef, xmin, xmax }
}

fn ksize_of(interp: Interp) -> usize {
    match interp {
        Interp::Cubic => 4,
        Interp::Lanczos4 => 8,
        _ => 2,
    }
}

/// Generic separable resize for float-path pixel types (16U / 32F in OpenCV,
/// where the work type is float).
fn resize_generic_float<T: FloatPix>(src: &Mat<T>, dw: usize, dh: usize, interp: Interp, sx_: f64, sy_: f64, isx: f64, isy: f64) -> Mat<T> {
    let cn = src.c;
    let ksize = ksize_of(interp);
    let ksize2 = (ksize / 2) as i32;
    let xt = build_axis(src.w, dw, sx_, isx, interp, ksize);
    let yt = build_axis(src.h, dh, sy_, isy, interp, ksize);
    let swidth = (src.w * cn) as i32;
    let dwidth = dw * cn;
    let mut out = Mat::<T>::new(dh, dw, cn);

    // Horizontal pass over one source row -> float buffer of dwidth.
    let hresize = |row: &[T], buf: &mut [f32]| {
        for dx in 0..dw {
            let sx0 = xt.ofs[dx] * cn as i32;
            let a = &xt.coef[dx * ksize..dx * ksize + ksize];
            for k in 0..cn {
                let sx = sx0 + k as i32;
                let v: f32;
                if ksize == 2 {
                    // HResizeLinear: interior uses both taps; after xmax only S[sx]*1
                    if dx < xt.xmax {
                        v = row[sx as usize].to_f() * a[0] + row[(sx + cn as i32) as usize].to_f() * a[1];
                    } else {
                        v = row[sx as usize].to_f();
                    }
                } else if dx >= xt.xmin && dx < xt.xmax {
                    let base = sx - (ksize2 - 1) * cn as i32;
                    let mut acc = row[base as usize].to_f() * a[0];
                    for j in 1..ksize {
                        acc += row[(base + j as i32 * cn as i32) as usize].to_f() * a[j];
                    }
                    v = acc;
                } else {
                    let base = sx - (ksize2 - 1) * cn as i32;
                    let mut acc = 0f32;
                    for j in 0..ksize {
                        let mut sxj = base + j as i32 * cn as i32;
                        if sxj < 0 || sxj >= swidth {
                            while sxj < 0 {
                                sxj += cn as i32;
                            }
                            while sxj >= swidth {
                                sxj -= cn as i32;
                            }
                        }
                        acc += row[sxj as usize].to_f() * a[j];
                    }
                    v = acc;
                }
                buf[dx * cn + k] = v;
            }
        }
    };

    let rows: Vec<Vec<f32>> = {
        use rayon::prelude::*;
        // Precompute horizontally resized source rows that are referenced.
        let mut needed = vec![false; src.h];
        for dy in 0..dh {
            let sy0 = yt.ofs[dy];
            for k in 0..ksize as i32 {
                let sy = (sy0 - ksize2 + 1 + k).clamp(0, src.h as i32 - 1);
                needed[sy as usize] = true;
            }
        }
        (0..src.h)
            .into_par_iter()
            .map(|sy| {
                if needed[sy] {
                    let mut buf = vec![0f32; dwidth];
                    hresize(src.row(sy), &mut buf);
                    buf
                } else {
                    Vec::new()
                }
            })
            .collect()
    };

    {
        use rayon::prelude::*;
        out.data
            .par_chunks_mut(dwidth)
            .enumerate()
            .for_each(|(dy, drow)| {
                let sy0 = yt.ofs[dy];
                let b = &yt.coef[dy * ksize..dy * ksize + ksize];
                let srows: Vec<&Vec<f32>> = (0..ksize as i32)
                    .map(|k| &rows[(sy0 - ksize2 + 1 + k).clamp(0, src.h as i32 - 1) as usize])
                    .collect();
                for x in 0..dwidth {
                    let mut acc = srows[0][x] * b[0];
                    for k in 1..ksize {
                        acc += srows[k][x] * b[k];
                    }
                    drow[x] = T::from_f(acc);
                }
            });
    }
    out
}

/// Generic separable resize for 8U with OpenCV's fixed-point coefficients.
fn resize_generic_u8(src: &Mat<u8>, dw: usize, dh: usize, interp: Interp, sx_: f64, sy_: f64, isx: f64, isy: f64) -> Mat<u8> {
    let cn = src.c;
    let ksize = ksize_of(interp);
    let ksize2 = (ksize / 2) as i32;
    let xt = build_axis(src.w, dw, sx_, isx, interp, ksize);
    let yt = build_axis(src.h, dh, sy_, isy, interp, ksize);
    let ia: Vec<i32> = xt.coef.iter().map(|&c| cv_round((c * INTER_RESIZE_COEF_SCALE) as f64).clamp(-32768, 32767)).collect();
    let ib: Vec<i32> = yt.coef.iter().map(|&c| cv_round((c * INTER_RESIZE_COEF_SCALE) as f64).clamp(-32768, 32767)).collect();
    let swidth = (src.w * cn) as i32;
    let dwidth = dw * cn;
    let one = 1i32 << INTER_RESIZE_COEF_BITS;

    let hresize = |row: &[u8], buf: &mut [i32]| {
        for dx in 0..dw {
            let sx0 = xt.ofs[dx] * cn as i32;
            let a = &ia[dx * ksize..dx * ksize + ksize];
            for k in 0..cn {
                let sx = sx0 + k as i32;
                let v: i32 = if ksize == 2 {
                    if dx < xt.xmax {
                        row[sx as usize] as i32 * a[0] + row[(sx + cn as i32) as usize] as i32 * a[1]
                    } else {
                        row[sx as usize] as i32 * one
                    }
                } else {
                    let base = sx - (ksize2 - 1) * cn as i32;
                    let mut acc = 0i32;
                    for j in 0..ksize {
                        let mut sxj = base + j as i32 * cn as i32;
                        while sxj < 0 {
                            sxj += cn as i32;
                        }
                        while sxj >= swidth {
                            sxj -= cn as i32;
                        }
                        acc += row[sxj as usize] as i32 * a[j];
                    }
                    acc
                };
                buf[dx * cn + k] = v;
            }
        }
    };

    use rayon::prelude::*;
    let mut needed = vec![false; src.h];
    for dy in 0..dh {
        for k in 0..ksize as i32 {
            let sy = (yt.ofs[dy] - ksize2 + 1 + k).clamp(0, src.h as i32 - 1);
            needed[sy as usize] = true;
        }
    }
    let rows: Vec<Vec<i32>> = (0..src.h)
        .into_par_iter()
        .map(|sy| {
            if needed[sy] {
                let mut buf = vec![0i32; dwidth];
                hresize(src.row(sy), &mut buf);
                buf
            } else {
                Vec::new()
            }
        })
        .collect();

    let mut out = Mat::<u8>::new(dh, dw, cn);
    out.data.par_chunks_mut(dwidth).enumerate().for_each(|(dy, drow)| {
        let sy0 = yt.ofs[dy];
        let b = &ib[dy * ksize..dy * ksize + ksize];
        let srows: Vec<&Vec<i32>> = (0..ksize as i32)
            .map(|k| &rows[(sy0 - ksize2 + 1 + k).clamp(0, src.h as i32 - 1) as usize])
            .collect();
        for x in 0..dwidth {
            if ksize == 2 {
                // VResizeLinearVec_32s8u: ((b0*(S0>>4))>>16 + (b1*(S1>>4))>>16 + 2) >> 2
                let s0 = srows[0][x] >> 4;
                let s1 = srows[1][x] >> 4;
                let v = (((b[0] * s0) >> 16) + ((b[1] * s1) >> 16) + 2) >> 2;
                drow[x] = sat_u8_i(v);
            } else {
                let mut acc = 0i32;
                for k in 0..ksize {
                    acc = acc.wrapping_add(srows[k][x].wrapping_mul(b[k]));
                }
                drow[x] = sat_u8_i((acc + (1 << 21)) >> 22);
            }
        }
    });
    out
}

struct AreaTab {
    di: usize,
    si: usize,
    alpha: f32,
}

fn compute_area_tab(ssize: usize, dsize: usize, cn: usize, scale: f64) -> Vec<AreaTab> {
    let mut tab = Vec::with_capacity(ssize * 2);
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell_width = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let mut sx2 = fsx2.floor() as i64;
        sx2 = sx2.min(ssize as i64 - 1);
        sx1 = sx1.min(sx2);
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push(AreaTab { di: dx * cn, si: (sx1 as usize - 1) * cn, alpha: ((sx1 as f64 - fsx1) / cell_width) as f32 });
        }
        for sx in sx1..sx2 {
            tab.push(AreaTab { di: dx * cn, si: sx as usize * cn, alpha: (1.0 / cell_width) as f32 });
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            tab.push(AreaTab {
                di: dx * cn,
                si: sx2 as usize * cn,
                alpha: ((fsx2 - sx2 as f64).min(1.0).min(cell_width) / cell_width) as f32,
            });
        }
    }
    tab
}

/// `resizeArea_` (true area averaging, non-integer scale, WT = float).
fn resize_area<T: FloatPix>(src: &Mat<T>, dw: usize, dh: usize, scale_x: f64, scale_y: f64) -> Mat<T> {
    let cn = src.c;
    let xtab = compute_area_tab(src.w, dw, cn, scale_x);
    let ytab = compute_area_tab(src.h, dh, 1, scale_y);
    let dwidth = dw * cn;
    // Group ytab entries by destination row (they are emitted in order).
    let mut row_start = vec![0usize; dh + 1];
    {
        let mut k = 0;
        for dy in 0..dh {
            row_start[dy] = k;
            while k < ytab.len() && ytab[k].di == dy {
                k += 1;
            }
        }
        row_start[dh] = ytab.len();
    }
    let mut out = Mat::<T>::new(dh, dw, cn);
    use rayon::prelude::*;
    out.data.par_chunks_mut(dwidth).enumerate().for_each(|(dy, drow)| {
        let mut buf = vec![0f32; dwidth];
        let mut sum = vec![0f32; dwidth];
        let mut first = true;
        for j in row_start[dy]..row_start[dy + 1] {
            let beta = ytab[j].alpha;
            let s = src.row(ytab[j].si);
            buf.iter_mut().for_each(|v| *v = 0.0);
            for t in &xtab {
                for k in 0..cn {
                    buf[t.di + k] += s[t.si + k].to_f() * t.alpha;
                }
            }
            if first {
                for dx in 0..dwidth {
                    sum[dx] = beta * buf[dx];
                }
                first = false;
            } else {
                for dx in 0..dwidth {
                    sum[dx] += beta * buf[dx];
                }
            }
        }
        for dx in 0..dwidth {
            drow[dx] = T::from_f(sum[dx]);
        }
    });
    out
}

/// `resizeAreaFast_` (integer scale factors).
fn resize_area_fast<T: FloatPix>(src: &Mat<T>, dw: usize, dh: usize, ix: usize, iy: usize, int_sum: bool) -> Mat<T> {
    let cn = src.c;
    let area = ix * iy;
    let scale = 1.0f32 / area as f32;
    let mut out = Mat::<T>::new(dh, dw, cn);
    let swidth = src.w * cn;
    use rayon::prelude::*;
    out.data.par_chunks_mut(dw * cn).enumerate().for_each(|(dy, drow)| {
        let sy0 = dy * iy;
        for dx in 0..dw {
            for k in 0..cn {
                let sx0 = dx * ix * cn + k;
                if int_sum {
                    let mut sum = 0i64;
                    for sy in 0..iy {
                        for sx in 0..ix {
                            let idx = sx0 + sx * cn;
                            if idx < swidth && sy0 + sy < src.h {
                                sum += src.data[(sy0 + sy) * swidth + idx].to_f() as i64;
                            }
                        }
                    }
                    if ix == 2 && iy == 2 {
                        // SIMD path: rounding shift (sum + 2) >> 2
                        drow[dx * cn + k] = T::from_f(((sum + 2) >> 2) as f32);
                    } else {
                        drow[dx * cn + k] = T::from_f(sum as f32 * scale);
                    }
                } else {
                    let mut sum = 0f32;
                    for sy in 0..iy {
                        for sx in 0..ix {
                            sum += src.data[(sy0 + sy) * swidth + sx0 + sx * cn].to_f();
                        }
                    }
                    drow[dx * cn + k] = T::from_f(sum * scale);
                }
            }
        }
    });
    out
}

fn resize_nearest<T: Copy + Default>(src: &Mat<T>, dw: usize, dh: usize, ifx: f64, ify: f64) -> Mat<T> {
    let cn = src.c;
    let mut out = Mat::<T>::new(dh, dw, cn);
    let xofs: Vec<usize> = (0..dw).map(|x| ((x as f64 * ifx).floor() as usize).min(src.w - 1)).collect();
    for y in 0..dh {
        let sy = ((y as f64 * ify).floor() as usize).min(src.h - 1);
        for x in 0..dw {
            for k in 0..cn {
                out.data[(y * dw + x) * cn + k] = src.data[(sy * src.w + xofs[x]) * cn + k];
            }
        }
    }
    out
}

struct Plan {
    scale_x: f64,
    scale_y: f64,
    inv_x: f64,
    inv_y: f64,
    interp: Interp,
    area_fast: Option<(usize, usize)>,
    area_true: bool,
}

fn plan(sw: usize, sh: usize, dw: usize, dh: usize, interp: Interp) -> Plan {
    let inv_x = dw as f64 / sw as f64;
    let inv_y = dh as f64 / sh as f64;
    let scale_x = 1.0 / inv_x;
    let scale_y = 1.0 / inv_y;
    let iscale_x = cv_round(scale_x);
    let iscale_y = cv_round(scale_y);
    let is_area_fast = (scale_x - iscale_x as f64).abs() < f64::EPSILON && (scale_y - iscale_y as f64).abs() < f64::EPSILON;
    let mut interp = interp;
    if interp == Interp::Linear && is_area_fast && iscale_x == 2 && iscale_y == 2 {
        interp = Interp::Area;
    }
    let mut area_fast = None;
    let mut area_true = false;
    if interp == Interp::Area && scale_x >= 1.0 && scale_y >= 1.0 {
        if is_area_fast {
            area_fast = Some((iscale_x as usize, iscale_y as usize));
        } else {
            area_true = true;
        }
    }
    Plan { scale_x, scale_y, inv_x, inv_y, interp, area_fast, area_true }
}

pub fn resize_f32(src: &Mat<f32>, dw: usize, dh: usize, interp: Interp) -> Mat<f32> {
    if dw == src.w && dh == src.h {
        return src.clone();
    }
    let p = plan(src.w, src.h, dw, dh, interp);
    if p.interp == Interp::Nearest {
        return resize_nearest(src, dw, dh, p.scale_x, p.scale_y);
    }
    if let Some((ix, iy)) = p.area_fast {
        return resize_area_fast(src, dw, dh, ix, iy, false);
    }
    if p.area_true {
        return resize_area(src, dw, dh, p.scale_x, p.scale_y);
    }
    resize_generic_float(src, dw, dh, p.interp, p.scale_x, p.scale_y, p.inv_x, p.inv_y)
}

pub fn resize_u16(src: &Mat<u16>, dw: usize, dh: usize, interp: Interp) -> Mat<u16> {
    if dw == src.w && dh == src.h {
        return src.clone();
    }
    let p = plan(src.w, src.h, dw, dh, interp);
    if p.interp == Interp::Nearest {
        return resize_nearest(src, dw, dh, p.scale_x, p.scale_y);
    }
    if let Some((ix, iy)) = p.area_fast {
        return resize_area_fast(src, dw, dh, ix, iy, true);
    }
    if p.area_true {
        return resize_area(src, dw, dh, p.scale_x, p.scale_y);
    }
    resize_generic_float(src, dw, dh, p.interp, p.scale_x, p.scale_y, p.inv_x, p.inv_y)
}

pub fn resize_u8(src: &Mat<u8>, dw: usize, dh: usize, interp: Interp) -> Mat<u8> {
    if dw == src.w && dh == src.h {
        return src.clone();
    }
    let p = plan(src.w, src.h, dw, dh, interp);
    if p.interp == Interp::Nearest {
        return resize_nearest(src, dw, dh, p.scale_x, p.scale_y);
    }
    if let Some((ix, iy)) = p.area_fast {
        return resize_area_fast(src, dw, dh, ix, iy, true);
    }
    if p.area_true {
        return resize_area(src, dw, dh, p.scale_x, p.scale_y);
    }
    resize_generic_u8(src, dw, dh, p.interp, p.scale_x, p.scale_y, p.inv_x, p.inv_y)
}
