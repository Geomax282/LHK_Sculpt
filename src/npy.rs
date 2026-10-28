//! Tiny .npy reader/writer used by the reference-comparison tooling.

use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
use std::path::Path;

#[derive(Debug, Clone)]
pub enum NpyData {
    U8(Vec<u8>),
    U16(Vec<u16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
}

#[derive(Debug, Clone)]
pub struct Npy {
    pub shape: Vec<usize>,
    pub data: NpyData,
}

impl Npy {
    pub fn to_f64(&self) -> Vec<f64> {
        match &self.data {
            NpyData::U8(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::U16(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::I32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::I64(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::F32(v) => v.iter().map(|&x| x as f64).collect(),
            NpyData::F64(v) => v.clone(),
            NpyData::Bool(v) => v.iter().map(|&x| x as u8 as f64).collect(),
        }
    }
    pub fn to_f32(&self) -> Vec<f32> {
        self.to_f64().into_iter().map(|v| v as f32).collect()
    }
    pub fn to_u8(&self) -> Vec<u8> {
        match &self.data {
            NpyData::U8(v) => v.clone(),
            NpyData::Bool(v) => v.iter().map(|&b| b as u8).collect(),
            _ => self.to_f64().into_iter().map(|v| v as u8).collect(),
        }
    }
    pub fn to_u16(&self) -> Vec<u16> {
        match &self.data {
            NpyData::U16(v) => v.clone(),
            _ => self.to_f64().into_iter().map(|v| v as u16).collect(),
        }
    }
    pub fn to_i64(&self) -> Vec<i64> {
        self.to_f64().into_iter().map(|v| v as i64).collect()
    }
}

pub fn read(path: &Path) -> Result<Npy> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    if &buf[..6] != b"\x93NUMPY" {
        bail!("not an npy file");
    }
    let major = buf[6];
    let (hlen, off) = if major == 1 {
        (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&buf[off..off + hlen])?;
    let body = &buf[off + hlen..];
    let descr = header.split("'descr':").nth(1).context("descr")?.trim_start();
    let descr = descr.split('\'').nth(1).context("descr q")?.to_string();
    if header.contains("'fortran_order': True") {
        bail!("fortran order unsupported");
    }
    let shp = header.split("'shape':").nth(1).context("shape")?;
    let shp = &shp[shp.find('(').context("(")? + 1..shp.find(')').context(")")?];
    let shape: Vec<usize> = shp.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect();
    let n: usize = shape.iter().product();
    let data = match descr.as_str() {
        "|u1" | "u1" => NpyData::U8(body[..n].to_vec()),
        "|b1" | "b1" => NpyData::Bool(body[..n].iter().map(|&b| b != 0).collect()),
        "<u2" => NpyData::U16(body[..n * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()),
        "<i4" => NpyData::I32(body[..n * 4].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()),
        "<i8" => NpyData::I64(body[..n * 8].chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()),
        "<f4" => NpyData::F32(body[..n * 4].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
        "<f8" => NpyData::F64(body[..n * 8].chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()),
        d => bail!("unsupported dtype {d}"),
    };
    Ok(Npy { shape, data })
}

fn write_raw(path: &Path, descr: &str, shape: &[usize], bytes: &[u8]) -> Result<()> {
    let shp = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shp}, }}");
    let total = 10 + header.len() + 1;
    let pad = (64 - total % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;
    f.write_all(header.as_bytes())?;
    f.write_all(bytes)?;
    Ok(())
}

pub fn write_f32(path: &Path, shape: &[usize], v: &[f32]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_raw(path, "<f4", shape, &b)
}
pub fn write_f64(path: &Path, shape: &[usize], v: &[f64]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_raw(path, "<f8", shape, &b)
}
pub fn write_u8(path: &Path, shape: &[usize], v: &[u8]) -> Result<()> {
    write_raw(path, "|u1", shape, v)
}
pub fn write_u16(path: &Path, shape: &[usize], v: &[u16]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_raw(path, "<u2", shape, &b)
}
pub fn write_i32(path: &Path, shape: &[usize], v: &[i32]) -> Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_raw(path, "<i4", shape, &b)
}
