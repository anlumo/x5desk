//! Player preview: renders straight from the two NV12 / P010 lens pictures with the template seam (or a stitcher's RAFT
//! flow, `Preview::seam`), or from an equirect (a paused frame's RAFT stitch, the X5 Webcam Mode), at window size, in
//! a render pass, so it can draw into a texture egui shows (`Rgba8Unorm`, `register_native_texture`) or into egui's
//! own pass from a paint callback.

use crate::{Calib, Ctx, Frame, Pano, Pose, Src, State, Stitcher, bind, calib::PLEN, init};
use glam::{Mat3, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Rectilinear, or Panini with `View::panini` > 0.
    Perspective,
    /// The whole sphere, centred on the view direction.
    Equirect,
    /// General perspective from eye distance `View::planet` (ρ = (d + 1)·sin θ / (d + cos θ) for a ray θ off the
    /// axis): stereographic at 1 ("tiny planet" when looking down, pitch −90°), rectilinear at 0, so a tiny planet
    /// unfolds into a perspective view continuously.
    Planet,
}

#[derive(Clone, Copy, Debug)]
pub struct View {
    pub mode: Mode,
    /// Degrees: yaw turns right, pitch looks up, roll turns the image about the view axis.
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    /// Horizontal field of view in degrees, up to (less half a degree) 180 + 180·`panini` (perspective) or
    /// 2·acos(−`planet`) (planet: 180 at 0, 360 at 1).
    pub fov: f32,
    /// Panini distance d, 0 (rectilinear) ..= 1.
    pub panini: f32,
    /// Planet eye distance d (`Mode::Planet`), 0 (rectilinear) ..= 1 (stereographic).
    pub planet: f32,
}

impl Default for View {
    fn default() -> Self {
        Self {
            mode: Mode::Perspective,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            fov: 90.0,
            panini: 0.0,
            planet: 1.0,
        }
    }
}

/// What a preview draws: both lenses (lens 0, lens 1) or an equirect. `[&Src; 2]` and `&Pano` convert into it.
#[derive(Clone, Copy)]
pub enum Input<'a> {
    Lenses([&'a Src; 2]),
    Pano(&'a Pano),
}

impl<'a> From<[&'a Src; 2]> for Input<'a> {
    fn from(s: [&'a Src; 2]) -> Self {
        Input::Lenses(s)
    }
}

impl<'a> From<&'a Pano> for Input<'a> {
    fn from(p: &'a Pano) -> Self {
        Input::Pano(p)
    }
}

pub struct Preview {
    cx: Ctx,
    st: State,
    /// Fragment entries `fs` (lenses), `fe` (equirect).
    pipe: [wgpu::RenderPipeline; 2],
    view: wgpu::Buffer,
    g3: wgpu::BindGroup,
    srgb: bool,
}

impl Preview {
    /// For lens pictures of `fr`'s size, drawing into `format` targets (an sRGB format gets linear values).
    pub fn new(
        cx: &Ctx,
        cal: &Calib,
        fr: Frame,
        blend: f64,
        colour: Option<f32>,
        format: wgpu::TextureFormat,
    ) -> Self {
        let st = State::new(cx, &cal.pack(fr, blend.clamp(0.0, 16.0), None), colour);
        Self::with(cx, st, format, None)
    }

    /// Without lens calibration (the Webcam Mode): draws `Input::Pano` only.
    pub fn pano(cx: &Ctx, format: wgpu::TextureFormat) -> Self {
        Self::with(cx, State::new(cx, &[0.0; PLEN], None), format, None)
    }

    /// Draws the lenses through `st`'s seam flow (RAFT), e.g. a reframed export: no intermediate equirect, so a narrow
    /// view keeps the lenses' detail. Shares `st`'s calibration, pose, colour fit and flows: per frame `pose`, then
    /// `st.belt` and `st.flow`, then `colour` and `render` with the lenses; `st.encode` is not needed (`st.out` can be
    /// minimal, 8 × 2).
    pub fn seam(st: &Stitcher, format: wgpu::TextureFormat) -> Self {
        Self::with(&st.cx, st.st.clone(), format, Some(&st.flows))
    }

    fn with(
        cx: &Ctx,
        st: State,
        format: wgpu::TextureFormat,
        flows: Option<&wgpu::Buffer>,
    ) -> Self {
        let d = &cx.dev;
        let view = init(
            d,
            &[0; 80],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let v = (5, view.as_entire_binding());
        let (l3, g3, fs) = match flows {
            Some(f) => (
                &cx.g3[5],
                bind(d, &cx.g3[5], &[(2, f.as_entire_binding()), v]),
                "ff",
            ),
            None => (&cx.g3[3], bind(d, &cx.g3[3], &[v]), "fs"),
        };
        let pipe = |fs, g0, g1| {
            let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(g0), g1, Some(&cx.state), Some(l3)],
                immediate_size: 0,
            });
            d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(fs),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &cx.module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &cx.module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipe = [
            pipe(fs, &cx.lens, Some(&cx.lens)),
            pipe("fe", &cx.pano, None),
        ];
        Self {
            cx: cx.clone(),
            st,
            pipe,
            view,
            g3,
            srgb: format.is_srgb(),
        }
    }

    /// The shown frame's orientation and rolling shutter (an equirect takes only the orientation).
    pub fn pose(&self, p: &Pose) {
        self.cx
            .q
            .write_buffer(&self.st.pose, 0, bytemuck::bytes_of(&p.bytes()));
    }

    /// Records the passes of a newly shown video frame: the colour fit (when on) and the template seam's low band
    /// (until the first, the seam draws with the single ramp).
    pub fn colour(&self, enc: &mut wgpu::CommandEncoder, src: [&Src; 2]) {
        if self.st.colour.is_some() || self.st.low.is_some() {
            let mut c = enc.begin_compute_pass(&Default::default());
            self.st.colour(&self.cx, &mut c, src);
            self.st.low(&self.cx, &mut c, src);
        }
    }

    /// Sets the view and the viewport it fills (target pixels: x, y, w, h).
    pub fn set(&self, v: &View, rect: [f32; 4]) {
        let r = f32::to_radians;
        // view camera x right, y up, z forward; at rest forward = −X (lens 0, panorama centre), up = +Z, right = +Y
        let m = Mat3::from_cols(Vec3::Y, Vec3::Z, Vec3::NEG_X)
            * Mat3::from_rotation_y(r(v.yaw))
            * Mat3::from_rotation_x(-r(v.pitch))
            * Mat3::from_rotation_z(r(v.roll));
        let (mode, f, d) = match v.mode {
            Mode::Perspective => {
                let d = v.panini.clamp(0.0, 1.0);
                // rectilinear stops short of 180°, Panini (d = 1) of 360°
                let h = r(v.fov / 2.0).min(r(89.5) + d * r(90.0));
                (0u32, (d + 1.0) / (d + h.cos()) * h.sin(), d)
            }
            Mode::Equirect => (1, 1.0, 0.0),
            Mode::Planet => {
                // the eye sees up to θ = acos(−d) off the axis; ρ(h) with d + cos h = (d − 1) + 2·cos²(h/2), which
                // keeps the stereographic edge (h → 180°, f = 2·tan(h/2)) exact
                let d = v.planet.clamp(0.0, 1.0) as f64;
                let h = (v.fov as f64 / 2.0)
                    .to_radians()
                    .min((-d).acos() - 0.5f64.to_radians())
                    / 2.0;
                let f = (d + 1.0) * 2.0 * h.sin() * h.cos() / (d - 1.0 + 2.0 * h.cos() * h.cos());
                (2, f as f32, d as f32)
            }
        };
        let mut b = [0f32; 20];
        for (c, col) in m.to_cols_array().chunks(3).enumerate() {
            b[4 * c..4 * c + 3].copy_from_slice(col);
        }
        b[12..16].copy_from_slice(&rect);
        b[16] = f32::from_bits(mode);
        b[17] = f;
        b[18] = d;
        b[19] = f32::from_bits(self.srgb as u32);
        self.cx
            .q
            .write_buffer(&self.view, 0, bytemuck::bytes_of(&b));
    }

    /// Draws into the caller's pass; the caller's viewport should match `set`'s rect.
    pub fn draw<'a>(&self, pass: &mut wgpu::RenderPass<'_>, src: impl Into<Input<'a>>) {
        match src.into() {
            Input::Lenses(s) => {
                pass.set_pipeline(&self.pipe[0]);
                pass.set_bind_group(0, &s[0].0, &[]);
                pass.set_bind_group(1, &s[1].0, &[]);
            }
            Input::Pano(p) => {
                pass.set_pipeline(&self.pipe[1]);
                pass.set_bind_group(0, &p.0, &[]);
            }
        }
        pass.set_bind_group(2, &self.st.bg, &[]);
        pass.set_bind_group(3, &self.g3, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Sets `v` for the whole of `target` and records a pass that draws it.
    pub fn render<'a>(
        &self,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        size: [u32; 2],
        v: &View,
        src: impl Into<Input<'a>>,
    ) {
        self.set(v, [0.0, 0.0, size[0] as f32, size[1] as f32]);
        let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("preview"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        self.draw(&mut p, src);
    }
}
