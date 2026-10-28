//! Command line: runs the pipeline headless (analysis, object composition,
//! mesh/STL export) for batch use and parameter tuning.

use anyhow::{bail, Context, Result};
use lhk_sculpt::img::Mat;
use lhk_sculpt::mesh::{self, ReliefMesh, ReliefSettings};
use lhk_sculpt::npy;
use lhk_sculpt::objects::{self, ObjectParams};
use lhk_sculpt::pipeline::{self, ModelPaths, Models};
use lhk_sculpt::export;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn load_f32(p: &Path) -> Result<Mat<f32>> {
    let n = npy::read(p)?;
    Ok(Mat::from_vec(n.shape[0], n.shape[1], 1, n.to_f32()))
}

fn load_u8(p: &Path) -> Result<Mat<u8>> {
    let n = npy::read(p)?;
    Ok(Mat::from_vec(n.shape[0], n.shape[1], 1, n.to_u8()))
}

fn save_f32(p: &Path, m: &Mat<f32>) -> Result<()> {
    npy::write_f32(p, &[m.h, m.w], &m.data)
}

fn save_u8(p: &Path, m: &Mat<u8>) -> Result<()> {
    npy::write_u8(p, &[m.h, m.w], &m.data)
}

/// `generate <image> <outdir> [input_size]` – full analysis; writes
/// height, texture, prob, depth_float and photo (.npy).
fn cmd_generate(args: &[String]) -> Result<()> {
    let img = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out)?;
    let input: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1024);
    let mp = ModelPaths::discover().context("Modelle nicht gefunden – LHK_MODELS auf den Modellordner setzen")?;
    let models = Models::load(&mp)?;
    let t = Instant::now();
    let g = pipeline::generate(&models, &img, input, &mut |s, p| eprintln!("[{:>5.1}%] {s}", p * 100.0))?;
    eprintln!("generate {:?}", t.elapsed());
    save_f32(&out.join("height.npy"), &g.height)?;
    save_u8(&out.join("texture.npy"), &g.texture)?;
    save_f32(&out.join("prob.npy"), &g.prob)?;
    save_f32(&out.join("depth_float.npy"), &g.depth)?;
    save_u8(&out.join("photo.npy"), &g.photo)?;
    Ok(())
}

/// `objretune <gendir> <outdir> key=val..` – object composition from saved
/// network depth, photo and object probability (no AI re-run).
fn cmd_objretune(args: &[String]) -> Result<()> {
    let dir = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out)?;
    let mut p = ObjectParams::default();
    for a in &args[2..] {
        let (k, v) = a.split_once('=').context("key=value")?;
        let v: f32 = v.parse()?;
        match k {
            "lift" => p.obj_lift = v,
            "od" => p.obj_depth = v,
            "inf" => p.obj_inflate = v,
            "ir" => p.inflate_radius = v,
            "bd" => p.bg_depth = v,
            "bs" => p.bg_struct = v,
            "os" => p.obj_struct = v,
            "sm" => p.obj_smooth = v,
            "fe" => p.feather = v,
            "gr" => p.groove = v,
            "bn" => p.bg_norm = v,
            "bsc" => p.bg_scale = v,
            "gw" => p.groove_w = v,
            _ => bail!("unknown {k}"),
        }
    }
    let df = load_f32(&dir.join("depth_float.npy"))?;
    let photo = load_u8(&dir.join("photo.npy"))?;
    let prob = load_f32(&dir.join("prob.npy"))?;
    let t = Instant::now();
    let (h, tex) = objects::compose(&df, &photo, &prob, &p);
    eprintln!("compose {:?}", t.elapsed());
    save_f32(&out.join("height.npy"), &h)?;
    save_u8(&out.join("texture.npy"), &tex)?;
    std::fs::copy(dir.join("prob.npy"), out.join("prob.npy"))?;
    Ok(())
}

/// `mesh <dir> <out.stl> key=val..` – relief solid (walls + bottom) in the
/// default model size; also writes the top grid heights as <out>.z.npy.
/// keys: height detail smooth spike bgsharp base tri
fn cmd_mesh(args: &[String]) -> Result<()> {
    let dir = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    let mut s = ReliefSettings::default();
    let mut tri = 1_500_000usize;
    for a in &args[2..] {
        let (k, v) = a.split_once('=').context("key=value")?;
        match k {
            "height" => s.height = v.parse()?,
            "detail" => s.detail = v.parse()?,
            "smooth" => s.smoothing = v.parse()?,
            "spike" => s.spike_guard = v.parse()?,
            "bgsharp" => s.bg_sharpness = v.parse()?,
            "base" => s.base = v.parse()?,
            "tri" => tri = v.parse()?,
            _ => bail!("unknown param {k}"),
        }
    }
    let h = load_f32(&dir.join("height.npy"))?;
    let tex = load_u8(&dir.join("texture.npy"))?;
    let prob = load_f32(&dir.join("prob.npy"))?;
    let t = Instant::now();
    let rm = ReliefMesh::new(&h, &tex, Some(&prob), tri);
    let z = rm.surface(&s);
    let v = rm.vertices(&z, true, true, &s);
    let f = rm.faces(true, true);
    eprintln!("mesh {:?} grid {}x{}", t.elapsed(), rm.width, rm.height);
    export::write_stl(&out, &v, &f)?;
    npy::write_f32(&out.with_extension("z.npy"), &[rm.height, rm.width], &z)?;
    let _ = mesh::MODEL_AREA;
    println!("grid={}x{} verts={} faces={}", rm.width, rm.height, v.len(), f.len());
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: lhk_cli <generate|objretune|mesh> ...");
        return Ok(());
    }
    match args[1].as_str() {
        "generate" => cmd_generate(&args[2..]),
        "objretune" => cmd_objretune(&args[2..]),
        "mesh" => cmd_mesh(&args[2..]),
        _ => bail!("unknown command"),
    }
}
