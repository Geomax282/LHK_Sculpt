//! Depth Anything V2 (ViT/14 + DPT head) – a port of the original PyTorch
//! implementation (Depth-Anything-V2, Apache-2.0); the Small/Base/Large
//! variant is detected from the checkpoint (LHK_Sculpt ships Small).
//!
//! Dense layers run on candle (CPU, or Metal on Apple Silicon); the small
//! interpolation steps (positional-embedding bicubic, DPT bilinear) are done
//! on the CPU with PyTorch's exact index/weight formulas.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, LayerNorm, Linear, VarBuilder};
use std::collections::HashMap;
use std::path::Path;

const PATCH: usize = 14;

/// Encoder/decoder sizes of the Depth Anything V2 variants (by embed dim).
#[derive(Clone, Copy)]
struct Cfg {
    embed: usize,
    heads: usize,
    depth: usize,
    features: usize,
    out_ch: [usize; 4],
    layer_idx: [usize; 4],
}

impl Cfg {
    fn for_embed(embed: usize) -> Result<Self> {
        Ok(match embed {
            384 => Self { embed, heads: 6, depth: 12, features: 64, out_ch: [48, 96, 192, 384], layer_idx: [2, 5, 8, 11] },
            768 => Self { embed, heads: 12, depth: 12, features: 128, out_ch: [96, 192, 384, 768], layer_idx: [2, 5, 8, 11] },
            1024 => Self { embed, heads: 16, depth: 24, features: 256, out_ch: [256, 512, 1024, 1024], layer_idx: [4, 11, 17, 23] },
            e => anyhow::bail!("unbekannte Depth-Anything-Variante (embed {e})"),
        })
    }
}

pub fn best_device() -> Device {
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
    }
    Device::Cpu
}

struct Block {
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    ls1: Tensor,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    ls2: Tensor,
}

fn ln(vb: &VarBuilder, name: &str, embed: usize) -> Result<LayerNorm> {
    let w = vb.get(embed, &format!("{name}.weight"))?;
    let b = vb.get(embed, &format!("{name}.bias"))?;
    Ok(LayerNorm::new(w, b, 1e-6))
}

fn lin(vb: &VarBuilder, name: &str, i: usize, o: usize) -> Result<Linear> {
    let w = vb.get((o, i), &format!("{name}.weight"))?;
    let b = vb.get(o, &format!("{name}.bias"))?;
    Ok(Linear::new(w, Some(b)))
}

fn conv(vb: &VarBuilder, name: &str, i: usize, o: usize, k: usize, stride: usize, pad: usize, bias: bool) -> Result<Conv2d> {
    let w = vb.get((o, i, k, k), &format!("{name}.weight"))?;
    let b = if bias { Some(vb.get(o, &format!("{name}.bias"))?) } else { None };
    Ok(Conv2d::new(w, b, Conv2dConfig { padding: pad, stride, dilation: 1, groups: 1 }))
}

fn convt(vb: &VarBuilder, name: &str, c: usize, k: usize) -> Result<ConvTranspose2d> {
    let w = vb.get((c, c, k, k), &format!("{name}.weight"))?;
    let b = vb.get(c, &format!("{name}.bias"))?;
    Ok(ConvTranspose2d::new(w, Some(b), ConvTranspose2dConfig { padding: 0, output_padding: 0, stride: k, dilation: 1 }))
}

struct Rcu {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl Rcu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.relu()?;
        let out = self.conv1.forward(&out)?;
        let out = out.relu()?;
        let out = self.conv2.forward(&out)?;
        Ok((out + x)?)
    }
}

struct Fusion {
    rcu1: Rcu,
    rcu2: Rcu,
    out_conv: Conv2d,
}

pub struct DepthAnythingV2 {
    device: Device,
    cfg: Cfg,
    cls_token: Tensor,
    pos_embed: Vec<f32>, // (1370, 768)
    patch_embed: Conv2d,
    blocks: Vec<Block>,
    norm: LayerNorm,
    projects: Vec<Conv2d>,
    resize0: ConvTranspose2d,
    resize1: ConvTranspose2d,
    resize3: Conv2d,
    layer_rn: Vec<Conv2d>,
    refine: Vec<Fusion>, // refinenet1..4
    output_conv1: Conv2d,
    output_conv2a: Conv2d,
    output_conv2b: Conv2d,
}

impl DepthAnythingV2 {
    /// Load the PyTorch checkpoint (a plain state_dict `.pth`).
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let tensors = candle_core::pickle::read_all(path).with_context(|| format!("Konnte Modell nicht lesen: {}", path.display()))?;
        let map: HashMap<String, Tensor> = tensors.into_iter().collect();
        let embed = map.get("pretrained.cls_token").context("cls_token fehlt")?.dims().last().copied().unwrap_or(0);
        let cfg = Cfg::for_embed(embed)?;
        let (ed, n_blocks, feat, out_ch) = (cfg.embed, cfg.depth, cfg.features, cfg.out_ch);
        let vb = VarBuilder::from_tensors(map, DType::F32, device);
        let p = vb.pp("pretrained");
        let cls_token = p.get((1, 1, ed), "cls_token")?;
        let pos_embed = p.get((1, 1370, ed), "pos_embed")?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let patch_embed = conv(&p, "patch_embed.proj", 3, ed, PATCH, PATCH, 0, true)?;
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            let b = p.pp(format!("blocks.{i}"));
            blocks.push(Block {
                norm1: ln(&b, "norm1", ed)?,
                qkv: lin(&b, "attn.qkv", ed, ed * 3)?,
                proj: lin(&b, "attn.proj", ed, ed)?,
                ls1: b.get(ed, "ls1.gamma")?,
                norm2: ln(&b, "norm2", ed)?,
                fc1: lin(&b, "mlp.fc1", ed, ed * 4)?,
                fc2: lin(&b, "mlp.fc2", ed * 4, ed)?,
                ls2: b.get(ed, "ls2.gamma")?,
            });
        }
        let norm = ln(&p, "norm", ed)?;
        let h = vb.pp("depth_head");
        let projects = (0..4).map(|i| conv(&h, &format!("projects.{i}"), ed, out_ch[i], 1, 1, 0, true)).collect::<Result<Vec<_>>>()?;
        let resize0 = convt(&h, "resize_layers.0", out_ch[0], 4)?;
        let resize1 = convt(&h, "resize_layers.1", out_ch[1], 2)?;
        let resize3 = conv(&h, "resize_layers.3", out_ch[3], out_ch[3], 3, 2, 1, true)?;
        let s = h.pp("scratch");
        let layer_rn = (0..4)
            .map(|i| conv(&s, &format!("layer{}_rn", i + 1), out_ch[i], feat, 3, 1, 1, false))
            .collect::<Result<Vec<_>>>()?;
        let mut refine = Vec::new();
        for i in 1..=4 {
            let r = s.pp(format!("refinenet{i}"));
            let rcu = |n: &str| -> Result<Rcu> {
                Ok(Rcu {
                    conv1: conv(&r, &format!("{n}.conv1"), feat, feat, 3, 1, 1, true)?,
                    conv2: conv(&r, &format!("{n}.conv2"), feat, feat, 3, 1, 1, true)?,
                })
            };
            refine.push(Fusion { rcu1: rcu("resConfUnit1")?, rcu2: rcu("resConfUnit2")?, out_conv: conv(&r, "out_conv", feat, feat, 1, 1, 0, true)? });
        }
        let output_conv1 = conv(&s, "output_conv1", feat, feat / 2, 3, 1, 1, true)?;
        let output_conv2a = conv(&s, "output_conv2.0", feat / 2, 32, 3, 1, 1, true)?;
        let output_conv2b = conv(&s, "output_conv2.2", 32, 1, 1, 1, 0, true)?;
        Ok(Self {
            device: device.clone(),
            cfg,
            cls_token,
            pos_embed,
            patch_embed,
            blocks,
            norm,
            projects,
            resize0,
            resize1,
            resize3,
            layer_rn,
            refine,
            output_conv1,
            output_conv2a,
            output_conv2b,
        })
    }

    /// DINOv2 `interpolate_pos_encoding` for a (gh x gw) patch grid:
    /// bicubic (A=-0.75) with scale_factor ((gh+0.1)/37, (gw+0.1)/37).
    fn pos_encoding(&self, gh: usize, gw: usize) -> Result<Tensor> {
        let ed = self.cfg.embed;
        let m = 37usize;
        if gh == m && gw == m {
            return Ok(Tensor::from_vec(self.pos_embed.clone(), (1, 1370, ed), &self.device)?);
        }
        let sqrt_n = (1369f64).sqrt();
        let sx = (gh as f64 + 0.1) / sqrt_n;
        let sy = (gw as f64 + 0.1) / sqrt_n;
        let oh = (m as f64 * sx).floor() as usize;
        let ow = (m as f64 * sy).floor() as usize;
        assert_eq!(oh, gh);
        assert_eq!(ow, gw);
        // grid laid out (row=i, col=j, ch) from tokens 1..
        let src = &self.pos_embed[ed..];
        let hs = (1.0 / sx) as f32;
        let ws = (1.0 / sy) as f32;
        let wy: Vec<(i64, [f32; 4])> = (0..oh).map(|o| cubic_index_weights(o, hs)).collect();
        let wx: Vec<(i64, [f32; 4])> = (0..ow).map(|o| cubic_index_weights(o, ws)).collect();
        let mut out = vec![0f32; ed * (1 + oh * ow)];
        out[..ed].copy_from_slice(&self.pos_embed[..ed]);
        let get = |y: i64, x: i64, c: usize| -> f32 {
            let y = y.clamp(0, m as i64 - 1) as usize;
            let x = x.clamp(0, m as i64 - 1) as usize;
            src[(y * m + x) * ed + c]
        };
        for oy in 0..oh {
            let (iy, cy) = wy[oy];
            for ox in 0..ow {
                let (ix, cx) = wx[ox];
                let dst = &mut out[ed * (1 + oy * ow + ox)..ed * (2 + oy * ow + ox)];
                for (c, d) in dst.iter_mut().enumerate() {
                    let mut rows = [0f32; 4];
                    for (k, r) in rows.iter_mut().enumerate() {
                        let yy = iy - 1 + k as i64;
                        *r = get(yy, ix - 1, c) * cx[0] + get(yy, ix, c) * cx[1] + get(yy, ix + 1, c) * cx[2] + get(yy, ix + 2, c) * cx[3];
                    }
                    *d = rows[0] * cy[0] + rows[1] * cy[1] + rows[2] * cy[2] + rows[3] * cy[3];
                }
            }
        }
        Ok(Tensor::from_vec(out, (1, 1 + oh * ow, ed), &self.device)?)
    }

    fn attention(&self, blk: &Block, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        let heads = self.cfg.heads;
        let hd = c / heads;
        let qkv = blk.qkv.forward(x)?.reshape((b, n, 3, heads, hd))?.permute((2, 0, 3, 1, 4))?;
        let scale = (hd as f64).powf(-0.5);
        let q = (qkv.get(0)?.contiguous()? * scale)?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let kt = k.transpose(2, 3)?.contiguous()?;
        // Chunk over queries to bound memory (exact: softmax is per row).
        let chunk = 1024usize;
        let mut outs = Vec::new();
        let mut start = 0;
        while start < n {
            let len = chunk.min(n - start);
            let qc = q.narrow(2, start, len)?;
            let attn = qc.matmul(&kt)?;
            let attn = candle_nn::ops::softmax_last_dim(&attn)?;
            outs.push(attn.matmul(&v)?);
            start += len;
        }
        let o = Tensor::cat(&outs, 2)?; // (b, heads, n, hd)
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, n, c))?;
        Ok(blk.proj.forward(&o)?)
    }

    /// Forward pass on a normalised (1,3,H,W) tensor with H,W multiples of 14.
    /// Returns the relu'd depth (H, W) as a flat Vec plus its size.
    pub fn forward(&self, x: &Tensor) -> Result<(Vec<f32>, usize, usize)> {
        let ed = self.cfg.embed;
        let (_, _, hh, ww) = x.dims4()?;
        let gh = hh / PATCH;
        let gw = ww / PATCH;
        let t = self.patch_embed.forward(x)?; // (1, 768, gh, gw)
        let t = t.flatten_from(2)?.transpose(1, 2)?; // (1, N, 768)
        let t = Tensor::cat(&[&self.cls_token, &t], 1)?;
        let mut t = (t + self.pos_encoding(gh, gw)?)?;
        let mut feats = Vec::new();
        for (i, blk) in self.blocks.iter().enumerate() {
            let a = self.attention(blk, &blk.norm1.forward(&t)?)?;
            t = (&t + a.broadcast_mul(&blk.ls1)?)?;
            let m = blk.fc2.forward(&blk.fc1.forward(&blk.norm2.forward(&t)?)?.gelu_erf()?)?;
            t = (&t + m.broadcast_mul(&blk.ls2)?)?;
            if self.cfg.layer_idx.contains(&i) {
                feats.push(self.norm.forward(&t)?);
            }
        }
        // DPT head
        let mut layers = Vec::new();
        for (i, f) in feats.iter().enumerate() {
            let n = f.dim(1)?;
            let pt = f.narrow(1, 1, n - 1)?; // drop cls
            let pt = pt.transpose(1, 2)?.contiguous()?.reshape((1, ed, gh, gw))?;
            let y = self.projects[i].forward(&pt)?;
            let y = match i {
                0 => self.resize0.forward(&y)?,
                1 => self.resize1.forward(&y)?,
                2 => y,
                _ => self.resize3.forward(&y)?,
            };
            layers.push(self.layer_rn[i].forward(&y)?);
        }
        let size_of = |t: &Tensor| -> Result<(usize, usize)> {
            let (_, _, h, w) = t.dims4()?;
            Ok((h, w))
        };
        // refinenet4(layer4) -> size of layer3
        let r4 = &self.refine[3];
        let p4 = r4.rcu2.forward(&layers[3])?;
        let p4 = r4.out_conv.forward(&bilinear_ac(&p4, size_of(&layers[2])?)?)?;
        let r3 = &self.refine[2];
        let p3 = r3.rcu2.forward(&(&p4 + r3.rcu1.forward(&layers[2])?)?)?;
        let p3 = r3.out_conv.forward(&bilinear_ac(&p3, size_of(&layers[1])?)?)?;
        let r2 = &self.refine[1];
        let p2 = r2.rcu2.forward(&(&p3 + r2.rcu1.forward(&layers[1])?)?)?;
        let p2 = r2.out_conv.forward(&bilinear_ac(&p2, size_of(&layers[0])?)?)?;
        let r1 = &self.refine[0];
        let p1 = r1.rcu2.forward(&(&p2 + r1.rcu1.forward(&layers[0])?)?)?;
        let (h1, w1) = size_of(&p1)?;
        let p1 = r1.out_conv.forward(&bilinear_ac(&p1, (h1 * 2, w1 * 2))?)?;
        let out = self.output_conv1.forward(&p1)?;
        let out = bilinear_ac(&out, (gh * PATCH, gw * PATCH))?;
        let out = self.output_conv2a.forward(&out)?.relu()?;
        let out = self.output_conv2b.forward(&out)?.relu()?;
        let out = out.relu()?; // DepthAnythingV2.forward F.relu
        let v = out.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        Ok((v, gh * PATCH, gw * PATCH))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// PyTorch bicubic source index & weights (align_corners=False, A=-0.75).
fn cubic_index_weights(dst: usize, scale: f32) -> (i64, [f32; 4]) {
    let real = scale * (dst as f32 + 0.5) - 0.5;
    let idx = real.floor();
    let t = real - idx;
    const A: f32 = -0.75;
    let cc1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let cc2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    let x2 = 1.0 - t;
    (idx as i64, [cc2(t + 1.0), cc1(t), cc1(x2), cc2(x2 + 1.0)])
}

/// Linear source indices/weights as in PyTorch upsample_bilinear2d.
fn linear_index_weights(in_size: usize, out_size: usize, align_corners: bool, dst: usize) -> (usize, usize, f32, f32) {
    let scale: f32 = if align_corners {
        if out_size > 1 {
            (in_size as f32 - 1.0) / (out_size as f32 - 1.0)
        } else {
            0.0
        }
    } else {
        in_size as f32 / out_size as f32
    };
    let real = if align_corners {
        scale * dst as f32
    } else {
        let r = scale * (dst as f32 + 0.5) - 0.5;
        if r < 0.0 {
            0.0
        } else {
            r
        }
    };
    let i0 = (real as i64).max(0) as usize;
    let i0 = i0.min(in_size - 1);
    let l1 = (real - i0 as f32).clamp(0.0, 1.0);
    let l0 = 1.0 - l1;
    let i1 = i0 + if i0 < in_size - 1 { 1 } else { 0 };
    (i0, i1, l0, l1)
}

/// Bilinear resize of a CHW float plane set (PyTorch semantics).
pub fn bilinear_chw(src: &[f32], c: usize, h: usize, w: usize, oh: usize, ow: usize, align_corners: bool) -> Vec<f32> {
    use rayon::prelude::*;
    let ys: Vec<_> = (0..oh).map(|y| linear_index_weights(h, oh, align_corners, y)).collect();
    let xs: Vec<_> = (0..ow).map(|x| linear_index_weights(w, ow, align_corners, x)).collect();
    let mut out = vec![0f32; c * oh * ow];
    out.par_chunks_mut(oh * ow).enumerate().for_each(|(ch, o)| {
        let s = &src[ch * h * w..(ch + 1) * h * w];
        for (y, &(y0, y1, hy0, hy1)) in ys.iter().enumerate() {
            for (x, &(x0, x1, wx0, wx1)) in xs.iter().enumerate() {
                let v00 = s[y0 * w + x0];
                let v01 = s[y0 * w + x1];
                let v10 = s[y1 * w + x0];
                let v11 = s[y1 * w + x1];
                o[y * ow + x] = hy0 * (wx0 * v00 + wx1 * v01) + hy1 * (wx0 * v10 + wx1 * v11);
            }
        }
    });
    out
}

fn bilinear_ac(t: &Tensor, size: (usize, usize)) -> Result<Tensor> {
    let (_, c, h, w) = t.dims4()?;
    let dev = t.device().clone();
    let v = t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let o = bilinear_chw(&v, c, h, w, size.0, size.1, true);
    Ok(Tensor::from_vec(o, (1, c, size.0, size.1), &dev)?)
}

/// Mean/std normalisation exactly like NormalizeImage + PrepareForNet:
/// (float32 image - mean) / std evaluated in float64, then cast to float32.
pub fn normalize_to_chw(rgb01: &[f32], h: usize, w: usize) -> Vec<f32> {
    const MEAN: [f64; 3] = [0.485, 0.456, 0.406];
    const STD: [f64; 3] = [0.229, 0.224, 0.225];
    let mut out = vec![0f32; 3 * h * w];
    for i in 0..h * w {
        for c in 0..3 {
            out[c * h * w + i] = ((rgb01[i * 3 + c] as f64 - MEAN[c]) / STD[c]) as f32;
        }
    }
    out
}

/// DepthAnything `Resize(..., keep_aspect_ratio=False, ensure_multiple_of=14,
/// resize_method='lower_bound')` target size for one axis.
pub fn lower_bound_multiple_of_14(target: usize, size: usize) -> usize {
    let scale = target as f64 / size as f64;
    let x = scale * size as f64;
    let mut y = ((x / 14.0).round_ties_even() * 14.0) as usize;
    if y < target {
        y = ((x / 14.0).ceil() * 14.0) as usize;
    }
    y
}

pub fn tensor_from_chw(v: Vec<f32>, h: usize, w: usize, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(v, (1, 3, h, w), device)?)
}

#[allow(dead_code)]
fn _assert_dims(t: &Tensor) -> Result<usize> {
    Ok(t.dim(D::Minus1)?)
}
