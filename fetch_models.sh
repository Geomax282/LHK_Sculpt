#!/usr/bin/env bash
# Download the model weights (official sources, commercial use allowed)
# into ./models.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models/birefnet models/licenses

get() { [ -f "$2" ] || curl -L --fail -o "$2" "$1"; }

get https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth \
    models/depth_anything_v2_vits.pth
get https://huggingface.co/ZhengPeng7/BiRefNet/resolve/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4/model.safetensors \
    models/birefnet/model.safetensors
if [ ! -f models/realesr-animevideov3-x2.bin ]; then
  tmp=$(mktemp -d)
  curl -L --fail -o "$tmp/esr.zip" \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip
  unzip -j -o "$tmp/esr.zip" "*realesr-animevideov3-x2.bin" -d models
  rm -rf "$tmp"
fi
get https://raw.githubusercontent.com/DepthAnything/Depth-Anything-V2/main/LICENSE models/licenses/Depth-Anything-V2-Small-LICENSE.txt
get https://raw.githubusercontent.com/ZhengPeng7/BiRefNet/main/LICENSE models/licenses/BiRefNet-LICENSE.txt
get https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/LICENSE models/licenses/Real-ESRGAN-LICENSE.txt
echo "Modelle in $(pwd)/models"
