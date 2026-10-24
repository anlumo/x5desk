//! Live output: FLV and an RTMP publisher, the frame protocol for our OBS plugin, DMA-BUF export of render
//! targets and the PipeWire camera node.

mod amf;
#[cfg(target_os = "linux")]
pub mod dmabuf;
pub mod flv;
#[cfg(target_os = "linux")]
pub mod obs;
#[cfg(target_os = "linux")]
pub mod pw;
pub mod rtmp;
#[cfg(all(test, target_os = "linux"))]
mod tests;
