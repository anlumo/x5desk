//! The shared Vulkan device (`Gpu`), H.265 / H.264 parsers, Vulkan Video decode into NV12 wgpu textures and encode
//! from NV12 / P010 wgpu textures.
mod bits;
mod bitw;
pub mod decode;
pub mod encode;
pub mod gpu;
pub mod h264;
pub mod h265;

pub use decode::{Codec, Decoder, Ring};
pub use encode::{Cfg, Encoder, Input, Packet, Rate};
pub use gpu::{Gpu, VideoQueue};

#[derive(Debug)]
pub enum Error {
    Vk(ash::vk::Result),
    /// Malformed or unsupported stream, or a missing parameter set / reference.
    Bad(&'static str),
    /// Device setup failed.
    Gpu(String),
    /// `Decoder::decode`: the output slot is still held by its consumer; nothing was decoded, retry.
    /// `Input::acquire`: the encoder has not finished the slot's last frame; `Encoder::encode`: `encode::SLOTS`
    /// frames are in flight, collect a packet first.
    Full,
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<ash::vk::Result> for Error {
    fn from(r: ash::vk::Result) -> Self {
        Error::Vk(r)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Vk(r) => write!(f, "Vulkan: {r}"),
            Error::Bad(s) => write!(f, "stream: {s}"),
            Error::Gpu(s) => write!(f, "GPU setup: {s}"),
            Error::Full => write!(f, "output ring full"),
        }
    }
}

impl std::error::Error for Error {}
