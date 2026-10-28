# Third-party notices

LHK_Sculpt contains Rust re-implementations of algorithms from the projects
below. No source files, binaries or model weights of these projects are
included in this repository.

| Project | Used for | License |
|---|---|---|
| Depth Anything V2 (code) | `src/depth.rs` network architecture | Apache-2.0 |
| BiRefNet (ZhengPeng7) | `src/birefnet.rs` network architecture | MIT |
| Real-ESRGAN / realesrgan-ncnn-vulkan | `src/esrgan.rs` (SRVGG network, tiling) | BSD-3-Clause |
| OpenCV | `src/cv/` (resize, color, Gaussian blur, findContours) | Apache-2.0 |
| Pillow | `src/pil.rs` (resampling, conversions) | MIT-CMU (HPND) |
| NumPy / SciPy | `src/nd.rs` (percentile, gradient, EDT, uniform filter) | BSD-3-Clause |

Model weights are not part of the source repository. The release package
contains these unmodified weights (license texts in `models/licenses/`):

| Weights | Source | License |
|---|---|---|
| Depth-Anything-V2-Small | huggingface.co/depth-anything/Depth-Anything-V2-Small | Apache-2.0 |
| BiRefNet | huggingface.co/ZhengPeng7/BiRefNet | MIT |
| realesr-animevideov3-x2 | github.com/xinntao/Real-ESRGAN (realesrgan-ncnn-vulkan release) | BSD-3-Clause |

The Depth Anything V2 Base/Large weights are licensed CC-BY-NC-4.0
(non-commercial only) and are neither shipped nor loaded automatically.

Rust crate dependencies are listed in `Cargo.toml` / `Cargo.lock`; they are
published under MIT and/or Apache-2.0 (mozjpeg: IJG/BSD-style, zlib).
