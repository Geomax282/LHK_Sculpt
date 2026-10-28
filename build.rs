//! Windows: embed the app icon and version info (company, product) into
//! the executables via windres (MinGW). Skipped elsewhere or without windres.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ico = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("assets").join("icon.ico");
    let v: Vec<u32> = env!("CARGO_PKG_VERSION").split('.').map(|s| s.parse().unwrap_or(0)).collect();
    let (ma, mi, pa) = (v[0], v.get(1).copied().unwrap_or(0), v.get(2).copied().unwrap_or(0));
    let ver = env!("CARGO_PKG_VERSION");
    let rc = format!(
        r#"1 ICON "{ico}"
1 VERSIONINFO
FILEVERSION {ma},{mi},{pa},0
PRODUCTVERSION {ma},{mi},{pa},0
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040704B0"
    BEGIN
      VALUE "CompanyName", "Land & Holzkommando JOL GbR"
      VALUE "FileDescription", "LHK_Sculpt - Bild zu Relief"
      VALUE "FileVersion", "{ver}"
      VALUE "ProductName", "LHK_Sculpt"
      VALUE "ProductVersion", "{ver}"
      VALUE "LegalCopyright", "Land & Holzkommando JOL GbR"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x0407, 1200
  END
END
"#,
        ico = ico.display().to_string().replace('\\', "/")
    );
    let rc_path = out.join("app.rc");
    std::fs::write(&rc_path, rc).unwrap();
    let res = out.join("app_res.o");
    match Command::new("windres").arg(&rc_path).arg("-O").arg("coff").arg("-o").arg(&res).status() {
        Ok(s) if s.success() => {
            println!("cargo:rustc-link-arg-bins={}", res.display());
        }
        _ => println!("cargo:warning=windres nicht gefunden - Exe ohne Icon"),
    }
}
