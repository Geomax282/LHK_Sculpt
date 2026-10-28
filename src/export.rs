//! Mesh export (binary STL, OBJ).

use anyhow::Result;
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn write_stl(path: &Path, verts: &[[f32; 3]], faces: &[[u32; 3]]) -> Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    let mut header = [0u8; 80];
    let tag = b"LHK_Sculpt relief";
    header[..tag.len()].copy_from_slice(tag);
    f.write_all(&header)?;
    f.write_all(&(faces.len() as u32).to_le_bytes())?;
    for t in faces {
        let a = verts[t[0] as usize];
        let b = verts[t[1] as usize];
        let c = verts[t[2] as usize];
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let mut n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if l > 0.0 {
            n = [n[0] / l, n[1] / l, n[2] / l];
        }
        for x in n.iter().chain(a.iter()).chain(b.iter()).chain(c.iter()) {
            f.write_all(&x.to_le_bytes())?;
        }
        f.write_all(&[0u8, 0u8])?;
    }
    Ok(())
}

pub fn write_obj(path: &Path, verts: &[[f32; 3]], faces: &[[u32; 3]]) -> Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "# LHK_Sculpt relief")?;
    for v in verts {
        writeln!(f, "v {} {} {}", v[0], v[1], v[2])?;
    }
    for t in faces {
        writeln!(f, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1)?;
    }
    Ok(())
}
