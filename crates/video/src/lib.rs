//! The shared device (`Gpu`), H.265 / H.264 parsers, hardware decode into NV12 / P010 pictures that wgpu samples and
//! encode from NV12 / P010 pictures that wgpu fills: Vulkan Video on Linux, VideoToolbox on macOS (`mac`), with the
//! same API. A decoded frame's planes come from `Ring::planes` on both.
mod bits;
// the Vulkan encoder writes its parameter sets; the parser tests write test streams
#[cfg(any(target_os = "linux", test))]
mod bitw;
#[cfg(target_os = "linux")]
pub mod decode;
#[cfg(target_os = "linux")]
pub mod encode;
#[cfg(target_os = "linux")]
pub mod gpu;
pub mod h264;
pub mod h265;
#[cfg(target_os = "macos")]
mod mac;
mod pic;
mod stream;

#[cfg(target_os = "macos")]
pub use mac::{decode, encode, gpu};

pub use decode::{Codec, Decoder, Ring};
pub use encode::{Cfg, Encoder, Input, Packet, Rate};
pub use gpu::Gpu;
#[cfg(target_os = "linux")]
pub use gpu::VideoQueue;

#[derive(Debug)]
pub enum Error {
    Vk(ash::vk::Result),
    /// Malformed or unsupported stream, or a missing parameter set / reference.
    Bad(&'static str),
    /// Device setup failed.
    Gpu(String),
    /// A macOS framework call failed with this OSStatus / CVReturn.
    Os(&'static str, i32),
    /// `Decoder::decode`: the output slot is still held by its consumer; nothing was decoded, retry.
    /// `Input::acquire`: the encoder has not finished the slot's last frame; `Encoder::encode`: `encode::SLOTS`
    /// frames are in flight, collect a packet first.
    Full,
    /// `Decoder::decode`: an H.265 RASL picture of a BLA or of a CRA where decoding started or restarted
    /// (`Decoder::restart`), or a picture shown before the H.264 non-IDR I picture there: it may reference pictures
    /// before that point, so it is neither decoded nor output, and no frame index is used. Nothing changed but the
    /// parameter sets it carried and, H.264, its reference marking (as a non-existing frame) and POC / frame_num state.
    Skip,
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
            Error::Os(c, s) => write!(f, "{c}: OSStatus {s}"),
            Error::Full => write!(f, "output ring full"),
            Error::Skip => write!(f, "leading picture skipped"),
        }
    }
}

impl std::error::Error for Error {}
