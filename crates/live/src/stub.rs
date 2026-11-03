//! Outside Linux: the DMA-BUF export, OBS frame protocol and PipeWire camera node have nothing to talk to (they are
//! Linux kernel and desktop interfaces), so these stand-ins keep the API the app builds against and refuse to start:
//! `obs::Server::bind`, `pw::Cam::new` and `dmabuf::Image::new` fail with [`unsupported`], and the app shows that.
use std::io;

/// Why an output does not start here.
pub fn unsupported() -> io::Error {
    io::Error::new(
        io::ErrorKind::Unsupported,
        "only on Linux (DMA-BUF, OBS plugin, PipeWire)",
    )
}

pub mod dmabuf {
    use std::os::fd::OwnedFd;
    use video::{Error, Gpu, Result};

    pub const LINEAR: u64 = 0;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Fmt {
        /// R, G, B, A bytes: DRM ABGR8888, PipeWire RGBx / RGBA
        Rgba,
        /// B, G, R, A bytes: DRM ARGB8888, PipeWire BGRx / BGRA
        Bgra,
        /// render through the plane views `R8Unorm` / `Rg8Unorm`
        Nv12,
    }

    impl Fmt {
        pub fn wgpu(self) -> wgpu::TextureFormat {
            match self {
                Fmt::Rgba => wgpu::TextureFormat::Rgba8Unorm,
                Fmt::Bgra => wgpu::TextureFormat::Bgra8Unorm,
                Fmt::Nv12 => wgpu::TextureFormat::NV12,
            }
        }

        /// DRM fourcc
        pub fn drm(self) -> u32 {
            u32::from_le_bytes(*match self {
                Fmt::Rgba => b"AB24",
                Fmt::Bgra => b"AR24",
                Fmt::Nv12 => b"NV12",
            })
        }
    }

    /// A memory plane in the DMA-BUF: byte offset, row pitch and size.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Plane {
        pub off: u32,
        pub stride: u32,
        pub len: u32,
    }

    /// None here.
    pub fn modifiers(_: &Gpu, _: Fmt, _: wgpu::TextureUsages, _: u32, _: u32) -> Vec<(u64, u32)> {
        vec![]
    }

    pub struct Image {
        pub tex: wgpu::Texture,
        pub fmt: Fmt,
        pub w: u32,
        pub h: u32,
        pub modifier: u64,
        pub fd: OwnedFd,
        pub planes: Vec<Plane>,
        pub size: u64,
    }

    fn no() -> Error {
        Error::Gpu(super::unsupported().to_string())
    }

    impl Image {
        pub fn new(
            _: &Gpu,
            _: Fmt,
            _: u32,
            _: u32,
            _: wgpu::TextureUsages,
            _: &[u64],
        ) -> Result<Image> {
            Err(no())
        }

        pub fn acquire(&self) -> Result<()> {
            Err(no())
        }

        pub fn release(&self) -> Result<OwnedFd> {
            Err(no())
        }
    }

    pub struct Ring {
        pub imgs: Vec<Image>,
    }

    impl Ring {
        pub fn new(
            _: &Gpu,
            _: Fmt,
            _: u32,
            _: u32,
            _: wgpu::TextureUsages,
            _: &[u64],
            _: usize,
        ) -> Result<Ring> {
            Err(no())
        }

        pub fn next(&mut self, _: impl Fn(usize) -> bool) -> Option<usize> {
            None
        }
    }
}

pub mod obs {
    use std::{
        io,
        os::fd::BorrowedFd,
        path::{Path, PathBuf},
    };

    pub const HOLD: usize = 4;
    pub const BT601: u32 = 1;
    pub const BT709: u32 = 2;
    pub const FULL: u32 = 0x100;
    pub const MOD_LINEAR: u64 = 0;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Format {
        pub modifier: u64,
        pub w: u32,
        pub h: u32,
        pub fourcc: u32,
        pub fps: [u32; 2],
        pub color: u32,
        pub rate: u32,
        pub ch: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Frame {
        pub ts: u64,
        pub modifier: u64,
        pub w: u32,
        pub h: u32,
        pub fourcc: u32,
        pub buf: u32,
        pub planes: u32,
        pub sync: u32,
        pub off: [u32; 4],
        pub stride: [u32; 4],
    }

    pub fn path() -> io::Result<PathBuf> {
        Err(super::unsupported())
    }

    /// Never bound here.
    pub enum Server {}

    impl Server {
        pub fn bind(_: &Path) -> io::Result<Self> {
            Err(super::unsupported())
        }

        pub fn poll(&mut self) -> bool {
            match *self {}
        }

        pub fn add(&mut self, _: u32, _: &str) {
            match *self {}
        }

        pub fn remove(&mut self, _: u32) {
            match *self {}
        }

        pub fn format(&mut self, _: u32, _: Format) {
            match *self {}
        }

        pub fn watched(&self, _: u32) -> bool {
            match *self {}
        }

        pub fn clients(&self, _: u32) -> usize {
            match *self {}
        }

        pub fn takes(&self, _: u32, _: u32, _: u64) -> bool {
            match *self {}
        }

        pub fn modifiers(&self, _: u32, _: u32) -> Vec<u64> {
            match *self {}
        }

        pub fn busy(&self, _: u32, _: u32) -> bool {
            match *self {}
        }

        pub fn frame(
            &mut self,
            _: u32,
            _: &Frame,
            _: &[BorrowedFd],
            _: Option<BorrowedFd>,
        ) -> usize {
            match *self {}
        }

        pub fn audio(&mut self, _: u32, _: u64, _: u32, _: u32, _: &[f32]) -> usize {
            match *self {}
        }
    }
}

pub mod pw {
    use super::dmabuf::{Fmt, Image};
    use std::{io, sync::Arc};
    use video::Gpu;

    pub const USAGE: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT
        .union(wgpu::TextureUsages::TEXTURE_BINDING)
        .union(wgpu::TextureUsages::COPY_SRC)
        .union(wgpu::TextureUsages::COPY_DST)
        .union(wgpu::TextureUsages::STORAGE_BINDING);
    /// Flat 1080p and the 2:1 equirect.
    pub const SIZES: [[u32; 2]; 2] = [[1920, 1080], [3840, 1920]];

    /// The negotiated stream.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Mode {
        pub fmt: Fmt,
        pub w: u32,
        pub h: u32,
        pub modifier: Option<u64>,
    }

    /// Never published here.
    pub enum Cam {}

    impl Cam {
        pub fn new(_: Arc<Gpu>, _: &str, _: &[[u32; 2]], _: u32) -> io::Result<Cam> {
            Err(super::unsupported())
        }

        pub fn node(&self) -> Option<u32> {
            match *self {}
        }

        pub fn mode(&self) -> Option<Mode> {
            match *self {}
        }

        pub fn error(&self) -> Option<String> {
            match *self {}
        }

        pub fn dequeue(&self) -> video::Result<Option<Frame<'_>>> {
            match *self {}
        }
    }

    pub struct Frame<'a> {
        pub img: Arc<Image>,
        pub mode: Mode,
        cam: &'a Cam,
    }

    impl Frame<'_> {
        pub fn queue(self, _: u64) -> video::Result<()> {
            match *self.cam {}
        }
    }
}
