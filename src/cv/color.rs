//! `cv::cvtColor` gray conversions (fixed-point 8U path).

use crate::img::Mat;

// RGB2Gray<uchar> in OpenCV 4.x uses 15-bit coefficients (verified vs cv2).
const R2Y: u32 = 9798;
const G2Y: u32 = 19235;
const B2Y: u32 = 3735;
const SHIFT: u32 = 15;

#[inline]
fn gray(r: u8, g: u8, b: u8) -> u8 {
    ((b as u32 * B2Y + g as u32 * G2Y + r as u32 * R2Y + (1 << (SHIFT - 1))) >> SHIFT) as u8
}

/// COLOR_BGR2GRAY / COLOR_BGRA2GRAY (alpha ignored). 1-channel input is copied.
pub fn bgr2gray(src: &Mat<u8>) -> Mat<u8> {
    to_gray(src, 2, 0)
}

/// COLOR_RGB2GRAY / COLOR_RGBA2GRAY.
pub fn rgb2gray(src: &Mat<u8>) -> Mat<u8> {
    to_gray(src, 0, 2)
}

fn to_gray(src: &Mat<u8>, ri: usize, bi: usize) -> Mat<u8> {
    if src.c == 1 {
        return src.clone();
    }
    let c = src.c;
    let data = src.data.chunks_exact(c).map(|p| gray(p[ri], p[1], p[bi])).collect();
    Mat::from_vec(src.h, src.w, 1, data)
}

/// COLOR_GRAY2RGB / GRAY2BGR.
pub fn gray2rgb(src: &Mat<u8>) -> Mat<u8> {
    let data = src.data.iter().flat_map(|&v| [v, v, v]).collect();
    Mat::from_vec(src.h, src.w, 3, data)
}

/// Swap channel 0 and 2 (BGR <-> RGB), keeping alpha if present.
pub fn swap_rb(src: &Mat<u8>) -> Mat<u8> {
    let mut out = src.clone();
    if src.c >= 3 {
        for p in out.data.chunks_exact_mut(src.c) {
            p.swap(0, 2);
        }
    }
    out
}
