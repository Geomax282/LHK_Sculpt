//! 3D relief viewer: renders the mesh offscreen with wgpu (Vulkan/DX12 on
//! Windows, Metal on macOS) and shows the result as an egui image.

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::RenderState;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const MSAA: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    eye: [f32; 4],
    color: [f32; 4],
}

const SHADER: &str = r#"
struct U { mvp: mat4x4<f32>, model: mat4x4<f32>, eye: vec4<f32>, color: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) n: vec3<f32>, @location(1) w: vec3<f32> };
@vertex fn vs_main(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>) -> VOut {
  var o: VOut;
  o.pos = u.mvp * vec4<f32>(p, 1.0);
  o.n = (u.model * vec4<f32>(n, 0.0)).xyz;
  o.w = (u.model * vec4<f32>(p, 1.0)).xyz;
  return o;
}
@fragment fn fs_main(i: VOut, @builtin(front_facing) ff: bool) -> @location(0) vec4<f32> {
  var n = normalize(i.n);
  if (!ff) { n = -n; }
  let v = normalize(u.eye.xyz - i.w);
  let l1 = normalize(v + vec3<f32>(0.35, 0.55, 0.25));
  let l2 = normalize(vec3<f32>(-0.6, -0.2, 0.5));
  let d = max(dot(n, l1), 0.0) * 0.78 + max(dot(n, l2), 0.0) * 0.22;
  let h = normalize(l1 + v);
  let s = pow(max(dot(n, h), 0.0), 48.0) * 0.18;
  let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.12;
  let c = u.color.rgb * (0.16 + d) + vec3<f32>(s + rim);
  return vec4<f32>(c, 1.0);
}
"#;

pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub pan: Vec3,
    pub radius: f32,
}

impl Camera {
    pub fn fit(radius: f32) -> Self {
        Self { yaw: 0.0, pitch: 0.0, dist: radius * 2.6, pan: Vec3::ZERO, radius }
    }
    fn eye(&self) -> Vec3 {
        let dir = Vec3::new(self.yaw.sin() * self.pitch.cos(), self.pitch.sin(), self.yaw.cos() * self.pitch.cos());
        self.pan + dir * self.dist
    }
    pub fn matrices(&self, aspect: f32) -> (Mat4, Vec3) {
        let eye = self.eye();
        let up = Vec3::Y;
        let view = Mat4::look_at_rh(eye, self.pan, up);
        let near = (self.dist - self.radius * 2.0).max(self.radius * 0.01);
        let proj = Mat4::perspective_rh(35f32.to_radians(), aspect.max(0.01), near, self.dist + self.radius * 3.0);
        (proj * view, eye)
    }
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.55, 1.55);
    }
    pub fn zoom(&mut self, s: f32) {
        self.dist = (self.dist * (1.0 - s * 0.0015)).clamp(self.radius * 0.2, self.radius * 20.0);
    }
    pub fn pan_by(&mut self, dx: f32, dy: f32, view_h: f32) {
        let eye = self.eye();
        let fwd = (self.pan - eye).normalize();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);
        let k = self.dist * 0.62 / view_h.max(1.0);
        self.pan += (-right * dx + up * dy) * k;
    }
}

pub struct Viewer {
    pipeline: wgpu::RenderPipeline,
    ubuf: wgpu::Buffer,
    bind: wgpu::BindGroup,
    vbuf: Option<wgpu::Buffer>,
    vcap: usize,
    ibuf: Option<wgpu::Buffer>,
    icount: u32,
    targets: Option<(u32, u32, wgpu::Texture, wgpu::TextureView, wgpu::TextureView, wgpu::TextureView)>,
    pub texture_id: Option<egui::TextureId>,
    pub camera: Camera,
    pub color: [f32; 3],
}

impl Viewer {
    pub fn new(rs: &RenderState) -> Self {
        let device = &rs.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("relief"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("u"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }] });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("relief"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: COLOR_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: MSAA, ..Default::default() },
            multiview: None,
            cache: None,
        });
        Self {
            pipeline,
            ubuf,
            bind,
            vbuf: None,
            vcap: 0,
            ibuf: None,
            icount: 0,
            targets: None,
            texture_id: None,
            camera: Camera::fit(1.0),
            color: [0.80, 0.76, 0.70],
        }
    }

    pub fn has_mesh(&self) -> bool {
        self.icount > 0
    }

    pub fn set_indices(&mut self, rs: &RenderState, faces: &[[u32; 3]]) {
        self.ibuf = Some(rs.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("idx"),
            contents: bytemuck::cast_slice(faces),
            usage: wgpu::BufferUsages::INDEX,
        }));
        self.icount = (faces.len() * 3) as u32;
    }

    pub fn set_vertices(&mut self, rs: &RenderState, verts: &[GpuVertex]) {
        if self.vbuf.is_none() || self.vcap < verts.len() {
            self.vbuf = Some(rs.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vtx"),
                size: (verts.len().max(1) * std::mem::size_of::<GpuVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.vcap = verts.len();
        }
        rs.queue.write_buffer(self.vbuf.as_ref().unwrap(), 0, bytemuck::cast_slice(verts));
    }

    fn ensure_targets(&mut self, rs: &RenderState, w: u32, h: u32) {
        if let Some((tw, th, ..)) = &self.targets {
            if *tw == w && *th == h {
                return;
            }
        }
        let dev = &rs.device;
        let mk = |format, samples, usage| {
            dev.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let resolve = mk(COLOR_FORMAT, 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let msaa = mk(COLOR_FORMAT, MSAA, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = mk(DEPTH_FORMAT, MSAA, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let rv = resolve.create_view(&Default::default());
        let mv = msaa.create_view(&Default::default());
        let dv = depth.create_view(&Default::default());
        let mut renderer = rs.renderer.write();
        match self.texture_id {
            Some(id) => renderer.update_egui_texture_from_wgpu_texture(dev, &rv, wgpu::FilterMode::Linear, id),
            None => self.texture_id = Some(renderer.register_native_texture(dev, &rv, wgpu::FilterMode::Linear)),
        }
        self.targets = Some((w, h, resolve, rv, mv, dv));
    }

    /// Render into the offscreen texture (size in physical pixels).
    pub fn render(&mut self, rs: &RenderState, w: u32, h: u32) {
        let (w, h) = (w.max(16), h.max(16));
        self.ensure_targets(rs, w, h);
        let (mvp, eye) = self.camera.matrices(w as f32 / h as f32);
        let u = Uniforms {
            mvp: mvp.to_cols_array_2d(),
            model: Mat4::IDENTITY.to_cols_array_2d(),
            eye: [eye.x, eye.y, eye.z, 1.0],
            color: [self.color[0], self.color[1], self.color[2], 1.0],
        };
        rs.queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
        let (_, _, _, rv, mv, dv) = self.targets.as_ref().unwrap();
        let mut enc = rs.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("relief"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: mv,
                    resolve_target: Some(rv),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.018, g: 0.022, b: 0.028, a: 1.0 }), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: dv,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let (Some(vb), Some(ib)) = (&self.vbuf, &self.ibuf) {
                if self.icount > 0 {
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &self.bind, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..self.icount, 0, 0..1);
                }
            }
        }
        rs.queue.submit(Some(enc.finish()));
    }
}

impl Viewer {
    /// Read back the last rendered frame as RGBA8 (for self-tests).
    pub fn snapshot(&self, rs: &RenderState) -> Option<(u32, u32, Vec<u8>)> {
        let (w, h, tex, ..) = self.targets.as_ref()?;
        let (w, h) = (*w, *h);
        let bpr = (w * 4).div_ceil(256) * 256;
        let buf = rs.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (bpr * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = rs.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture { texture: tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyBuffer { buffer: &buf, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        rs.queue.submit(Some(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        rs.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            out.extend_from_slice(&data[(y * bpr) as usize..(y * bpr + w * 4) as usize]);
        }
        Some((w, h, out))
    }
}

/// Interleave positions and normals; the relief is shown with Z towards the
/// viewer (mesh space: X right, Y up, Z out of the image).
pub fn to_gpu(verts: &[[f32; 3]], normals: &[[f32; 3]]) -> Vec<GpuVertex> {
    verts.iter().zip(normals.iter()).map(|(p, n)| GpuVertex { pos: *p, nrm: *n }).collect()
}
