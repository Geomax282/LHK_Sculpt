#!/usr/bin/env bash
# Build LHK_Sculpt for Apple Silicon (arm64) with Metal acceleration and
# package it as LHK_Sculpt.app.
#
# Voraussetzungen: Xcode Command Line Tools (xcode-select --install) und Rust
# (https://rustup.rs).
#
# Modelle: vorher ./fetch_models.sh ausfuehren (Ordner "models"). Er wird
# ins App-Bundle kopiert. Alternativ kann der Ordner beim ersten Start
# gewaehlt werden.
set -euo pipefail
cd "$(dirname "$0")"

rustup target add aarch64-apple-darwin >/dev/null
cargo build --release --target aarch64-apple-darwin --features metal

APP=dist/LHK_Sculpt.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/aarch64-apple-darwin/release/LHK_Sculpt "$APP/Contents/MacOS/LHK_Sculpt"
cp target/aarch64-apple-darwin/release/lhk_cli "$APP/Contents/MacOS/lhk_cli"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>LHK_Sculpt</string>
  <key>CFBundleDisplayName</key><string>LHK_Sculpt</string>
  <key>CFBundleIdentifier</key><string>de.lhk.sculpt</string>
  <key>CFBundleVersion</key><string>0.1.1</string>
  <key>CFBundleShortVersionString</key><string>0.1.1</string>
  <key>CFBundleExecutable</key><string>LHK_Sculpt</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
# app icon (assets/icon.png -> AppIcon.icns)
ICONSET=$(mktemp -d)/AppIcon.iconset
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s assets/icon.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) assets/icon.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
if [ -d models ]; then
  cp -R models "$APP/Contents/Resources/models"
fi
# ad-hoc signature so Gatekeeper allows local start on Apple Silicon
codesign --force --deep -s - "$APP" || true
echo "Fertig: $APP"
