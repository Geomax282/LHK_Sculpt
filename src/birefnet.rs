//! BiRefNet (Swin-L backbone, multi-scale input, ASPP-Deformable decoder)
//! – port of the official BiRefNet model code (ZhengPeng7/BiRefNet, MIT), for
//! salient object segmentation (people incl. hair, glasses, accessories).

use crate::depth::bilinear_chw;
use crate::img::Mat;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};
use rayon::prelude::*;
use std::path::Path;

const WS: usize = 12;
const DIMS: [usize; 4] = [192, 384, 768, 1536];
const DEPTHS: [usize; 4] = [2, 2, 18, 2];
const HEADS: [usize; 4] = [6, 12, 24, 48];

fn lnorm(vb: &VarBuilder, name: &str, c: usize) -> Result<LayerNorm> {
    Ok(LayerNorm::new(vb.get(c, &format!("{name}.weight"))?, vb.get(c, &format!("{name}.bias"))?, 1e-5))
}

fn linear(vb: &VarBuilder, name: &str, i: usize, o: usize, bias: bool) -> Result<Linear> {
    let w = vb.get((o, i), &format!("{name}.weight"))?;
    let b = if bias { Some(vb.get(o, &format!("{name}.bias"))?) } else { None };
    Ok(Linear::new(w, b))
}

fn conv(vb: &VarBuilder, name: &str, i: usize, o: usize, k: usize, stride: usize, pad: usize, bias: bool) -> Result<Conv2d> {
    let w = vb.get((o, i, k, k), &format!("{name}.weight"))?;
    let b = if bias { Some(vb.get(o, &format!("{name}.bias"))?) } else { None };
    Ok(Conv2d::new(w, b, Conv2dConfig { padding: pad, stride, dilation: 1, groups: 1 }))
}

/// Eval-mode BatchNorm2d folded into scale/shift (eps 1e-5).
struct Bn {
    scale: Tensor,
    shift: Tensor,
}

impl Bn {
    fn load(vb: &VarBuilder, name: &str, c: usize) -> Result<Self> {
        let w = vb.get(c, &format!("{name}.weight"))?;
        let b = vb.get(c, &format!("{name}.bias"))?;
        let m = vb.get(c, &format!("{name}.running_mean"))?;
        let v = vb.get(c, &format!("{name}.running_var"))?;
        let scale = (w / (v + 1e-5)?.sqrt()?)?;
        let shift = (b - (m * &scale)?)?;
        Ok(Self { scale: scale.reshape((1, c, 1, 1))?, shift: shift.reshape((1, c, 1, 1))? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)?)
    }
}

// ---------------------------------------------------------------- Swin-L

struct SwinBlock {
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    rel_bias: Tensor, // (heads, N, N)
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    heads: usize,
    shift: usize,
}

fn rel_index() -> Vec<usize> {
    let n = WS * WS;
    let mut idx = vec![0usize; n * n];
    for a in 0..n {
        let (ah, aw) = (a / WS, a % WS);
        for b in 0..n {
            let (bh, bw) = (b / WS, b % WS);
            let dh = ah as i64 - bh as i64 + WS as i64 - 1;
            let dw = aw as i64 - bw as i64 + WS as i64 - 1;
            idx[a * n + b] = (dh * (2 * WS as i64 - 1) + dw) as usize;
        }
    }
    idx
}

impl SwinBlock {
    fn load(vb: &VarBuilder, dim: usize, heads: usize, shift: usize, dev: &Device) -> Result<Self> {
        let table = vb.get(((2 * WS - 1) * (2 * WS - 1), heads), "attn.relative_position_bias_table")?.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let idx = rel_index();
        let n = WS * WS;
        let mut bias = vec![0f32; heads * n * n];
        for h in 0..heads {
            for i in 0..n * n {
                bias[h * n * n + i] = table[idx[i]][h];
            }
        }
        Ok(Self {
            norm1: lnorm(vb, "norm1", dim)?,
            qkv: linear(vb, "attn.qkv", dim, dim * 3, true)?,
            proj: linear(vb, "attn.proj", dim, dim, true)?,
            rel_bias: Tensor::from_vec(bias, (1, heads, n, n), dev)?,
            norm2: lnorm(vb, "norm2", dim)?,
            fc1: linear(vb, "mlp.fc1", dim, dim * 4, true)?,
            fc2: linear(vb, "mlp.fc2", dim * 4, dim, true)?,
            heads,
            shift,
        })
    }

    fn forward(&self, x: &Tensor, h: usize, w: usize, mask: Option<&Tensor>) -> Result<Tensor> {
        let (_, _, c) = x.dims3()?;
        let shortcut = x.clone();
        let mut t = self.norm1.forward(x)?.reshape((h, w, c))?;
        let pr = (WS - w % WS) % WS;
        let pb = (WS - h % WS) % WS;
        if pr > 0 {
            t = t.pad_with_zeros(1, 0, pr)?;
        }
        if pb > 0 {
            t = t.pad_with_zeros(0, 0, pb)?;
        }
        let (hp, wp) = (h + pb, w + pr);
        let s = self.shift;
        if s > 0 {
            t = Tensor::cat(&[t.narrow(0, s, hp - s)?, t.narrow(0, 0, s)?], 0)?;
            t = Tensor::cat(&[t.narrow(1, s, wp - s)?, t.narrow(1, 0, s)?], 1)?;
        }
        let (nh, nw) = (hp / WS, wp / WS);
        let n = WS * WS;
        let win = t.reshape((nh, WS, nw, WS, c))?.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((nh * nw, n, c))?;
        let hd = c / self.heads;
        let qkv = self.qkv.forward(&win)?.reshape((nh * nw, n, 3, self.heads, hd))?.permute((2, 0, 3, 1, 4))?;
        let q = (qkv.get(0)?.contiguous()? * (hd as f64).powf(-0.5))?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let mut attn = q.matmul(&k.transpose(2, 3)?.contiguous()?)?.broadcast_add(&self.rel_bias)?;
        if let (true, Some(m)) = (s > 0, mask) {
            attn = attn.broadcast_add(m)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let o = attn.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((nh * nw, n, c))?;
        let o = self.proj.forward(&o)?;
        let mut t = o.reshape((nh, nw, WS, WS, c))?.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape((hp, wp, c))?;
        if s > 0 {
            t = Tensor::cat(&[t.narrow(0, hp - s, s)?, t.narrow(0, 0, hp - s)?], 0)?;
            t = Tensor::cat(&[t.narrow(1, wp - s, s)?, t.narrow(1, 0, wp - s)?], 1)?;
        }
        if pr > 0 || pb > 0 {
            t = t.narrow(0, 0, h)?.narrow(1, 0, w)?;
        }
        let t = t.contiguous()?.reshape((1, h * w, c))?;
        let x = (shortcut + t)?;
        let m = self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu_erf()?)?;
        Ok((x + m)?)
    }
}

/// Shifted-window attention mask (nW, 1, N, N) with -100 between regions.
fn shift_mask(h: usize, w: usize, dev: &Device) -> Result<Tensor> {
    let hp = h.div_ceil(WS) * WS;
    let wp = w.div_ceil(WS) * WS;
    let s = WS / 2;
    let region = |p: usize, len: usize| -> usize {
        if p < len - WS {
            0
        } else if p < len - s {
            1
        } else {
            2
        }
    };
    let mut img = vec![0usize; hp * wp];
    for y in 0..hp {
        for x in 0..wp {
            img[y * wp + x] = region(y, hp) * 3 + region(x, wp);
        }
    }
    let (nh, nw) = (hp / WS, wp / WS);
    let n = WS * WS;
    let mut out = vec![0f32; nh * nw * n * n];
    for wy in 0..nh {
        for wx in 0..nw {
            let wi = wy * nw + wx;
            let ids: Vec<usize> = (0..n).map(|k| img[(wy * WS + k / WS) * wp + wx * WS + k % WS]).collect();
            for i in 0..n {
                for j in 0..n {
                    out[(wi * n + i) * n + j] = if ids[j] != ids[i] { -100.0 } else { 0.0 };
                }
            }
        }
    }
    Ok(Tensor::from_vec(out, (nh * nw, 1, n, n), dev)?)
}

struct Stage {
    blocks: Vec<SwinBlock>,
    down: Option<(LayerNorm, Linear)>,
    norm_out: LayerNorm,
    dim: usize,
}

struct Swin {
    patch: Conv2d,
    patch_norm: LayerNorm,
    stages: Vec<Stage>,
}

impl Swin {
    fn load(vb: &VarBuilder, dev: &Device) -> Result<Self> {
        let patch = conv(vb, "patch_embed.proj", 3, DIMS[0], 4, 4, 0, true)?;
        let patch_norm = lnorm(vb, "patch_embed.norm", DIMS[0])?;
        let mut stages = Vec::new();
        for i in 0..4 {
            let l = vb.pp(format!("layers.{i}"));
            let blocks = (0..DEPTHS[i])
                .map(|b| SwinBlock::load(&l.pp(format!("blocks.{b}")), DIMS[i], HEADS[i], if b % 2 == 0 { 0 } else { WS / 2 }, dev))
                .collect::<Result<Vec<_>>>()?;
            let down = if i < 3 { Some((lnorm(&l, "downsample.norm", DIMS[i] * 4)?, linear(&l, "downsample.reduction", DIMS[i] * 4, DIMS[i] * 2, false)?)) } else { None };
            stages.push(Stage { blocks, down, norm_out: lnorm(vb, &format!("norm{i}"), DIMS[i])?, dim: DIMS[i] });
        }
        Ok(Self { patch, patch_norm, stages })
    }

    /// Returns the four stage outputs as (1, C, H, W).
    fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let t = self.patch.forward(x)?;
        let (_, c, mut h, mut w) = t.dims4()?;
        let mut t = self.patch_norm.forward(&t.flatten_from(2)?.transpose(1, 2)?.contiguous()?)?;
        let _ = c;
        let dev = x.device().clone();
        let mut outs = Vec::new();
        for st in &self.stages {
            let mask = shift_mask(h, w, &dev)?;
            for b in &st.blocks {
                t = b.forward(&t, h, w, Some(&mask))?;
            }
            let o = st.norm_out.forward(&t)?.reshape((1, h, w, st.dim))?.permute((0, 3, 1, 2))?.contiguous()?;
            outs.push(o);
            if let Some((n, r)) = &st.down {
                let g = t.reshape((h / 2, 2, w / 2, 2, st.dim))?.permute((0, 2, 3, 1, 4))?.contiguous()?.reshape((1, (h / 2) * (w / 2), 4 * st.dim))?;
                t = r.forward(&n.forward(&g)?)?;
                h /= 2;
                w /= 2;
            }
        }
        Ok(outs)
    }
}

// ---------------------------------------------------------------- decoder

/// torchvision deform_conv2d bilinear sample (zero outside).
#[inline]
fn bilinear_sample(img: &[f32], h: usize, w: usize, y: f32, x: f32) -> f32 {
    if y <= -1.0 || y >= h as f32 || x <= -1.0 || x >= w as f32 {
        return 0.0;
    }
    let yl = y.floor() as i64;
    let xl = x.floor() as i64;
    let (yh, xh) = (yl + 1, xl + 1);
    let ly = y - yl as f32;
    let lx = x - xl as f32;
    let (hy, hx) = (1.0 - ly, 1.0 - lx);
    let g = |yy: i64, xx: i64| -> f32 {
        if yy >= 0 && xx >= 0 && yy < h as i64 && xx < w as i64 {
            img[yy as usize * w + xx as usize]
        } else {
            0.0
        }
    };
    hy * hx * g(yl, xl) + hy * lx * g(yl, xh) + ly * hx * g(yh, xl) + ly * lx * g(yh, xh)
}

struct DeformConv {
    offset: Conv2d,
    modulator: Conv2d,
    weight: Tensor, // (out, in*k*k)
    k: usize,
    cin: usize,
    cout: usize,
}

impl DeformConv {
    fn load(vb: &VarBuilder, cin: usize, cout: usize, k: usize) -> Result<Self> {
        let p = k / 2;
        let w = vb.get((cout, cin, k, k), "regular_conv.weight")?;
        Ok(Self {
            offset: conv(vb, "offset_conv", cin, 2 * k * k, k, 1, p, true)?,
            modulator: conv(vb, "modulator_conv", cin, k * k, k, 1, p, true)?,
            weight: w.reshape((cout, cin * k * k))?,
            k,
            cin,
            cout,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, c, h, w) = x.dims4()?;
        let dev = x.device().clone();
        let k = self.k;
        let kk = k * k;
        let p = (k / 2) as f32;
        let off = self.offset.forward(x)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let modl = (candle_nn::ops::sigmoid(&self.modulator.forward(x)?)? * 2.0)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let xin = x.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let npx = h * w;
        let chunk = (1usize << 24) / (c * kk).max(1); // ~64 MB of columns
        let chunk = chunk.clamp(256, npx);
        let mut outs: Vec<Tensor> = Vec::new();
        let mut start = 0;
        while start < npx {
            let len = chunk.min(npx - start);
            // cols: (c*kk, len)
            let mut cols = vec![0f32; c * kk * len];
            cols.par_chunks_mut(len).enumerate().for_each(|(row, dst)| {
                let ci = row / kk;
                let t = row % kk;
                let (i, j) = (t / k, t % k);
                let img = &xin[ci * npx..(ci + 1) * npx];
                for (q, d) in dst.iter_mut().enumerate() {
                    let pix = start + q;
                    let (oy, ox) = (pix / w, pix % w);
                    let dy = off[(2 * t) * npx + pix];
                    let dx = off[(2 * t + 1) * npx + pix];
                    let m = modl[t * npx + pix];
                    let y = oy as f32 - p + i as f32 + dy;
                    let xx = ox as f32 - p + j as f32 + dx;
                    *d = bilinear_sample(img, h, w, y, xx) * m;
                }
            });
            let ct = Tensor::from_vec(cols, (c * kk, len), &dev)?;
            outs.push(self.weight.matmul(&ct)?);
            start += len;
        }
        let _ = (self.cin, self.cout);
        Ok(Tensor::cat(&outs, 1)?.reshape((1, self.cout, h, w))?)
    }
}

struct AsppMod {
    conv: DeformConv,
    bn: Bn,
}

impl AsppMod {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.bn.forward(&self.conv.forward(x)?)?.relu()?)
    }
}

struct AsppDeformable {
    aspp1: AsppMod,
    deforms: Vec<AsppMod>,
    gap_conv: Conv2d,
    gap_bn: Bn,
    conv1: Conv2d,
    bn1: Bn,
}

impl AsppDeformable {
    fn load(vb: &VarBuilder, cin: usize) -> Result<Self> {
        let m = |name: String, k: usize| -> Result<AsppMod> {
            Ok(AsppMod { conv: DeformConv::load(&vb.pp(format!("{name}.atrous_conv")), cin, 256, k)?, bn: Bn::load(vb, &format!("{name}.bn"), 256)? })
        };
        Ok(Self {
            aspp1: m("aspp1".into(), 1)?,
            deforms: [1usize, 3, 7].iter().enumerate().map(|(i, &k)| m(format!("aspp_deforms.{i}"), k)).collect::<Result<Vec<_>>>()?,
            gap_conv: conv(vb, "global_avg_pool.1", cin, 256, 1, 1, 0, false)?,
            gap_bn: Bn::load(vb, "global_avg_pool.2", 256)?,
            conv1: conv(vb, "conv1", 256 * 5, cin, 1, 1, 0, false)?,
            bn1: Bn::load(vb, "bn1", cin)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        let mut parts = vec![self.aspp1.forward(x)?];
        for d in &self.deforms {
            parts.push(d.forward(x)?);
        }
        let g = x.mean_keepdim(2)?.mean_keepdim(3)?;
        let g = self.gap_bn.forward(&self.gap_conv.forward(&g)?)?.relu()?;
        parts.push(g.broadcast_as((1, 256, h, w))?.contiguous()?);
        let cat = Tensor::cat(&parts, 1)?;
        Ok(self.bn1.forward(&self.conv1.forward(&cat)?)?.relu()?)
    }
}

struct DecBlk {
    conv_in: Conv2d,
    bn_in: Bn,
    att: AsppDeformable,
    conv_out: Conv2d,
    bn_out: Bn,
}

impl DecBlk {
    fn load(vb: &VarBuilder, cin: usize, cout: usize) -> Result<Self> {
        Ok(Self {
            conv_in: conv(vb, "conv_in", cin, 64, 3, 1, 1, true)?,
            bn_in: Bn::load(vb, "bn_in", 64)?,
            att: AsppDeformable::load(&vb.pp("dec_att"), 64)?,
            conv_out: conv(vb, "conv_out", 64, cout, 3, 1, 1, true)?,
            bn_out: Bn::load(vb, "bn_out", cout)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = self.bn_in.forward(&self.conv_in.forward(x)?)?.relu()?;
        let t = self.att.forward(&t)?;
        self.bn_out.forward(&self.conv_out.forward(&t)?)
    }
}

struct SimpleConvs {
    c1: Conv2d,
    c2: Conv2d,
}

impl SimpleConvs {
    fn load(vb: &VarBuilder, cin: usize, cout: usize) -> Result<Self> {
        Ok(Self { c1: conv(vb, "conv1", cin, 64, 3, 1, 1, true)?, c2: conv(vb, "conv_out", 64, cout, 3, 1, 1, true)? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.c2.forward(&self.c1.forward(x)?)?)
    }
}

struct Gdt {
    conv: Conv2d,
    bn: Bn,
    attn: Conv2d,
}

impl Gdt {
    fn load(vb: &VarBuilder, i: usize, cin: usize) -> Result<Self> {
        Ok(Self {
            conv: conv(vb, &format!("gdt_convs_{i}.0"), cin, 16, 3, 1, 1, true)?,
            bn: Bn::load(vb, &format!("gdt_convs_{i}.1"), 16)?,
            attn: conv(vb, &format!("gdt_convs_attn_{i}.0"), 16, 1, 1, 1, 0, true)?,
        })
    }
    fn forward(&self, p: &Tensor) -> Result<Tensor> {
        let g = self.bn.forward(&self.conv.forward(p)?)?.relu()?;
        let a = candle_nn::ops::sigmoid(&self.attn.forward(&g)?)?;
        Ok(p.broadcast_mul(&a)?)
    }
}

pub struct BiRefNet {
    device: Device,
    bb: Swin,
    squeeze: DecBlk,
    ipt: Vec<SimpleConvs>, // ipt_blk5..ipt_blk2 (ipt_blk1 handled fused)
    ipt1_c1: Conv2d,
    ipt1_fused: Conv2d, // conv_out of ipt_blk1 folded with conv_out1's 48 input weights
    dec: Vec<DecBlk>,   // block4..block1
    lat: Vec<Conv2d>,   // lateral4..2
    gdt: Vec<Gdt>,      // 4..2
    out_w_main: Tensor, // conv_out1 weights for the 192 decoder channels, (1,192,1,1)
    out_bias: f32,
}

/// Interpolate a (1,C,H,W) tensor bilinearly with align_corners=True.
fn interp(t: &Tensor, oh: usize, ow: usize) -> Result<Tensor> {
    let (_, c, h, w) = t.dims4()?;
    if h == oh && w == ow {
        return Ok(t.clone());
    }
    let dev = t.device().clone();
    let v = t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(Tensor::from_vec(bilinear_chw(&v, c, h, w, oh, ow, true), (1, c, oh, ow), &dev)?)
}

/// einops 'b c (hg h) (wg w) -> b (c hg wg) h w' for a (1,3,S,S) input.
fn image2patches(x: &[f32], s: usize, r: usize, dev: &Device) -> Result<Tensor> {
    let g = s / r;
    let mut out = vec![0f32; 3 * g * g * r * r];
    for c in 0..3 {
        for hg in 0..g {
            for wg in 0..g {
                let oc = (c * g + hg) * g + wg;
                for hh in 0..r {
                    for ww in 0..r {
                        out[(oc * r + hh) * r + ww] = x[(c * s + hg * r + hh) * s + wg * r + ww];
                    }
                }
            }
        }
    }
    Ok(Tensor::from_vec(out, (1, 3 * g * g, r, r), dev)?)
}

impl BiRefNet {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }.with_context(|| format!("BiRefNet nicht lesbar: {}", path.display()))?;
        let bb = Swin::load(&vb.pp("bb"), device)?;
        let ch = [3072usize, 1536, 768, 384];
        let squeeze = DecBlk::load(&vb.pp("squeeze_module.0"), ch[0] + 384 + 768 + 1536, ch[0])?;
        let d = vb.pp("decoder");
        let ipt = vec![
            SimpleConvs::load(&d.pp("ipt_blk5"), 1024 * 3, ch[0] / 8)?,
            SimpleConvs::load(&d.pp("ipt_blk4"), 256 * 3, ch[0] / 8)?,
            SimpleConvs::load(&d.pp("ipt_blk3"), 64 * 3, ch[1] / 8)?,
            SimpleConvs::load(&d.pp("ipt_blk2"), 16 * 3, ch[2] / 8)?,
        ];
        let dec = vec![
            DecBlk::load(&d.pp("decoder_block4"), ch[0] + ch[0] / 8, ch[1])?,
            DecBlk::load(&d.pp("decoder_block3"), ch[1] + ch[0] / 8, ch[2])?,
            DecBlk::load(&d.pp("decoder_block2"), ch[2] + ch[1] / 8, ch[3])?,
            DecBlk::load(&d.pp("decoder_block1"), ch[3] + ch[2] / 8, ch[3] / 2)?,
        ];
        let lat = vec![
            conv(&d, "lateral_block4.conv", ch[1], ch[1], 1, 1, 0, true)?,
            conv(&d, "lateral_block3.conv", ch[2], ch[2], 1, 1, 0, true)?,
            conv(&d, "lateral_block2.conv", ch[3], ch[3], 1, 1, 0, true)?,
        ];
        let gdt = vec![Gdt::load(&d, 4, ch[1])?, Gdt::load(&d, 3, ch[2])?, Gdt::load(&d, 2, ch[3])?];
        // conv_out1: 1x1 over [192 decoder ch | 48 ipt_blk1 ch]
        let w_out = d.get((1, 240, 1, 1), "conv_out1.0.weight")?.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let b_out = d.get(1, "conv_out1.0.bias")?.to_device(&Device::Cpu)?.to_vec1::<f32>()?[0];
        let out_w_main = Tensor::from_vec(w_out[..192].to_vec(), (1, 192, 1, 1), device)?;
        let ipt1 = d.pp("ipt_blk1");
        let ipt1_c1 = conv(&ipt1, "conv1", 3, 64, 3, 1, 1, true)?;
        let w2 = ipt1.get((48, 64, 3, 3), "conv_out.weight")?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let b2 = ipt1.get(48, "conv_out.bias")?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let mut fw = vec![0f32; 64 * 9];
        let mut fb = 0f32;
        for o in 0..48 {
            let a = w_out[192 + o];
            fb += a * b2[o];
            for i in 0..64 * 9 {
                fw[i] += a * w2[o * 64 * 9 + i];
            }
        }
        let ipt1_fused = Conv2d::new(Tensor::from_vec(fw, (1, 64, 3, 3), device)?, None, Conv2dConfig { padding: 1, stride: 1, dilation: 1, groups: 1 });
        Ok(Self { device: device.clone(), bb, squeeze, ipt, ipt1_c1, ipt1_fused, dec, lat, gdt, out_w_main, out_bias: b_out + fb })
    }

    /// Logits (1024x1024) for a normalised (3,1024,1024) CHW input.
    pub fn forward_logits(&self, chw: &[f32]) -> Result<Vec<f32>> {
        const S: usize = 1024;
        let dev = &self.device;
        let x = Tensor::from_vec(chw.to_vec(), (1, 3, S, S), dev)?;
        let f = self.bb.forward(&x)?;
        let xh = Tensor::from_vec(bilinear_chw(chw, 3, S, S, S / 2, S / 2, true), (1, 3, S / 2, S / 2), dev)?;
        let fh = self.bb.forward(&xh)?;
        let mut feats = Vec::new();
        for (a, b) in f.iter().zip(fh.iter()) {
            let (_, _, h, w) = a.dims4()?;
            feats.push(Tensor::cat(&[a.clone(), interp(b, h, w)?], 1)?);
        }
        let (x1, x2, x3) = (&feats[0], &feats[1], &feats[2]);
        let (_, _, h4, w4) = feats[3].dims4()?;
        let x4 = Tensor::cat(&[interp(x1, h4, w4)?, interp(x2, h4, w4)?, interp(x3, h4, w4)?, feats[3].clone()], 1)?;
        let x4 = self.squeeze.forward(&x4)?;
        drop(f);
        drop(fh);

        let size = |t: &Tensor| -> Result<(usize, usize)> {
            let (_, _, h, w) = t.dims4()?;
            Ok((h, w))
        };
        // level 4
        let (h, w) = size(&x4)?;
        let x4 = Tensor::cat(&[x4.clone(), self.ipt[0].forward(&image2patches(chw, S, h, dev)?)?], 1)?;
        let p4 = self.gdt[0].forward(&self.dec[0].forward(&x4)?)?;
        let (h3, w3) = size(x3)?;
        let p3i = (interp(&p4, h3, w3)? + self.lat[0].forward(x3)?)?;
        let p3i = Tensor::cat(&[p3i, self.ipt[1].forward(&image2patches(chw, S, h3, dev)?)?], 1)?;
        let p3 = self.gdt[1].forward(&self.dec[1].forward(&p3i)?)?;
        let (h2, w2) = size(x2)?;
        let p2i = (interp(&p3, h2, w2)? + self.lat[1].forward(x2)?)?;
        let p2i = Tensor::cat(&[p2i, self.ipt[2].forward(&image2patches(chw, S, h2, dev)?)?], 1)?;
        let p2 = self.gdt[2].forward(&self.dec[2].forward(&p2i)?)?;
        let (h1, w1) = size(x1)?;
        let p1i = (interp(&p2, h1, w1)? + self.lat[2].forward(x1)?)?;
        let p1i = Tensor::cat(&[p1i, self.ipt[3].forward(&image2patches(chw, S, h1, dev)?)?], 1)?;
        let p1 = self.dec[3].forward(&p1i)?; // (1,192,256,256)
        let _ = (h, w);
        // conv_out1 is 1x1 and commutes with the (linear) upsampling
        let main = p1.broadcast_mul(&self.out_w_main)?.sum_keepdim(1)?;
        let main = interp(&main, S, S)?;
        let side = self.ipt1_fused.forward(&self.ipt1_c1.forward(&x)?)?;
        let logits = ((main + side)? + self.out_bias as f64)?;
        Ok(logits.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Foreground probability at the image size (like _predict_mask, float).
    pub fn predict(&self, rgb: &Mat<u8>) -> Result<Mat<f32>> {
        let r = crate::pil::resize_bilinear_rgb(rgb, 1024, 1024);
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        let n = 1024 * 1024;
        let mut chw = vec![0f32; 3 * n];
        for i in 0..n {
            for c in 0..3 {
                chw[c * n + i] = (r.data[i * 3 + c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
        let logits = self.forward_logits(&chw)?;
        let prob: Vec<f32> = logits.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
        let up = bilinear_chw(&prob, 1, 1024, 1024, rgb.h, rgb.w, false);
        Ok(Mat::from_vec(rgb.h, rgb.w, 1, up))
    }
}

#[allow(dead_code)]
fn _d(t: &Tensor) -> Result<usize> {
    Ok(t.dim(D::Minus1)?)
}
