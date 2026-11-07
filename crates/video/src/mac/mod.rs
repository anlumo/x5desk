//! The macOS backend: Metal under wgpu, VideoToolbox for decode and encode.
pub mod decode;
pub mod encode;
pub mod gpu;
pub(crate) mod sys;
