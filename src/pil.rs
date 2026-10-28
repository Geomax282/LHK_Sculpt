//! Pillow (PIL) operations with Pillow's integer math (Pillow, MIT-CMU).

use crate::img::Mat;

/// Image.convert("L") from RGB: (R*19595 + G*38470 + B*7471 + 0x8000) >> 16
pub fn rgb_to_l(rgb: &Mat<u8>) -> Mat<u8> {
    let data = rgb
        .data
        .chunks_exact(rgb.c)
        .map(|p| ((p[0] as u32 * 19595 + p[1] as u32 * 38470 + p[2] as u32 * 7471 + 0x8000) >> 16) as u8)
        .collect();
    Mat::from_vec(rgb.h, rgb.w, 1, data)
}

/// white = Image.new("RGB", size, white); white.paste(rgba, mask=alpha)
/// Pillow BLEND: DIV255(in*a + out*(255-a)) with DIV255(x) = ((x+128)>>8 + x+128)>>8
pub fn composite_rgba_on_white(rgba: &Mat<u8>) -> Mat<u8> {
    let mut out = Mat::<u8>::new(rgba.h, rgba.w, 3);
    for (o, p) in out.data.chunks_exact_mut(3).zip(rgba.data.chunks_exact(4)) {
        let a = p[3] as u32;
        for c in 0..3 {
            let t = p[c] as u32 * a + 255 * (255 - a) + 128;
            o[c] = (((t >> 8) + t) >> 8) as u8;
        }
    }
    out
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let x = x * std::f64::consts::PI;
        x.sin() / x
    }
}

fn lanczos(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Pillow precompute_coeffs + normalize_coeffs_8bpc for one axis.
fn coeffs(in_size: usize, out_size: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    coeffs_with(in_size, out_size, 3.0, lanczos)
}

fn bilinear_filter(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

fn coeffs_with(in_size: usize, out_size: usize, support0: f64, filter: fn(f64) -> f64) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = support0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut kk = vec![0f64; out_size * ksize];
    let mut bounds = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        xmax -= xmin;
        let k = &mut kk[xx * ksize..(xx + 1) * ksize];
        let mut ww = 0.0;
        for x in 0..xmax as usize {
            let w = filter((x as f64 + xmin as f64 - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for v in k.iter_mut().take(xmax as usize) {
            if ww != 0.0 {
                *v /= ww;
            }
        }
        bounds.push((xmin as usize, xmax as usize));
    }
    let ik = kk
        .iter()
        .map(|&v| {
            let s = v * (1u64 << PRECISION_BITS) as f64;
            if v < 0.0 {
                (-0.5 + s).trunc() as i32
            } else {
                (0.5 + s).trunc() as i32
            }
        })
        .collect();
    (ksize, bounds, ik)
}

#[inline]
fn clip8(v: i32) -> u8 {
    let v = v >> PRECISION_BITS;
    v.clamp(0, 255) as u8
}

/// Image.resize((w, h), Image.BILINEAR) for an RGB image (8 bit), as done by
/// torchvision's T.Resize on PIL images.
pub fn resize_bilinear_rgb(src: &Mat<u8>, ow: usize, oh: usize) -> Mat<u8> {
    assert_eq!(src.c, 3);
    let mut cur = src.clone();
    if ow != src.w {
        let (ks, b, k) = coeffs_with(src.w, ow, 1.0, bilinear_filter);
        let mut out = Mat::<u8>::new(cur.h, ow, 3);
        for y in 0..cur.h {
            let row = cur.row(y);
            for xx in 0..ow {
                let (xmin, xmax) = b[xx];
                let kk = &k[xx * ks..];
                for c in 0..3 {
                    let mut ss: i32 = 1 << (PRECISION_BITS - 1);
                    for x in 0..xmax {
                        ss = ss.wrapping_add(row[(x + xmin) * 3 + c] as i32 * kk[x]);
                    }
                    out.data[(y * ow + xx) * 3 + c] = clip8(ss);
                }
            }
        }
        cur = out;
    }
    if oh != src.h {
        let (ks, b, k) = coeffs_with(src.h, oh, 1.0, bilinear_filter);
        let mut out = Mat::<u8>::new(oh, cur.w, 3);
        for yy in 0..oh {
            let (ymin, ymax) = b[yy];
            let kk = &k[yy * ks..];
            for x in 0..cur.w * 3 {
                let mut ss: i32 = 1 << (PRECISION_BITS - 1);
                for y in 0..ymax {
                    ss = ss.wrapping_add(cur.data[(y + ymin) * cur.w * 3 + x] as i32 * kk[y]);
                }
                out.data[yy * cur.w * 3 + x] = clip8(ss);
            }
        }
        cur = out;
    }
    cur
}

/// Image.resize((w, h), Image.LANCZOS) for an "L" image (8 bit).
pub fn resize_lanczos_l(src: &Mat<u8>, ow: usize, oh: usize) -> Mat<u8> {
    assert_eq!(src.c, 1);
    let mut cur = src.clone();
    if ow != src.w {
        let (ks, b, k) = coeffs(src.w, ow);
        let mut out = Mat::<u8>::new(cur.h, ow, 1);
        for y in 0..cur.h {
            let row = cur.row(y);
            for xx in 0..ow {
                let (xmin, xmax) = b[xx];
                let kk = &k[xx * ks..];
                let mut ss: i32 = 1 << (PRECISION_BITS - 1);
                for x in 0..xmax {
                    ss = ss.wrapping_add(row[x + xmin] as i32 * kk[x]);
                }
                out.data[y * ow + xx] = clip8(ss);
            }
        }
        cur = out;
    }
    if oh != src.h {
        let (ks, b, k) = coeffs(src.h, oh);
        let mut out = Mat::<u8>::new(oh, cur.w, 1);
        for yy in 0..oh {
            let (ymin, ymax) = b[yy];
            let kk = &k[yy * ks..];
            for x in 0..cur.w {
                let mut ss: i32 = 1 << (PRECISION_BITS - 1);
                for y in 0..ymax {
                    ss = ss.wrapping_add(cur.data[(y + ymin) * cur.w + x] as i32 * kk[y]);
                }
                out.data[yy * cur.w + x] = clip8(ss);
            }
        }
        cur = out;
    }
    cur
}
