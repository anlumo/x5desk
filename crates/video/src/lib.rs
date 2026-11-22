//! The shared device (`Gpu`), H.265 / H.264 parsers, hardware decode into NV12 / P010 pictures that wgpu samples and
//! encode from NV12 / P010 pictures that wgpu fills: Vulkan Video on Linux, VideoToolbox on macOS (`mac`), with the
//! same API. A decoded frame's planes come from `Ring::planes` on both.
mod bits;
// the Vulkan encoder writes its parameter sets; `check` and the parser tests write test streams
#[cfg(any(target_os = "linux", target_os = "macos", test))]
mod bitw;
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub mod check;
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

/// The adapter `--gpu` chose for every `Gpu` of the process: an index in [`gpus`]' list or a part of an adapter's
/// name ([`pick`]); unset, wgpu's high-performance adapter, as without the flag.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub static PICK: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Which of the adapters named `n` `w` means: index `w`, else the one whose name has `w` in it, in any case.
pub fn pick(n: &[String], w: &str) -> std::result::Result<usize, String> {
    let l = w.to_lowercase();
    let m: Vec<usize> = match w.parse() {
        Ok(i) if i < n.len() => vec![i],
        _ => (0..n.len())
            .filter(|&i| n[i].to_lowercase().contains(&l))
            .collect(),
    };
    match m[..] {
        [i] => Ok(i),
        [] => Err(format!("no GPU is {w:?} or has it in its name")),
        _ => Err(format!(
            "{w:?} is in {} GPUs' names: give the index",
            m.len()
        )),
    }
}

/// wgpu's high-performance adapter: the one taken without `--gpu`.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn best(i: &wgpu::Instance) -> Result<wgpu::Adapter> {
    pollster::block_on(i.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|e| Error::Gpu(e.to_string()))
}

/// The adapter of `Gpu::new`: [`PICK`]'s, else [`best`]. Err: none, or `PICK` names none or several (with the list).
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn adapter(i: &wgpu::Instance) -> Result<wgpu::Adapter> {
    let Some(w) = PICK.get() else { return best(i) };
    let mut a = pollster::block_on(i.enumerate_adapters(wgpu::Backends::all()));
    let n: Vec<_> = a.iter().map(|a| a.get_info().name).collect();
    let k = pick(&n, w).map_err(|e| Error::Gpu(format!("--gpu: {e}\n{}", table(i, &a))))?;
    Ok(a.swap_remove(k))
}

/// `a` as `--gpu list` shows them: index, name, driver, the H.265 and H.264 it decodes and encodes, the default
/// marked.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn table(i: &wgpu::Instance, a: &[wgpu::Adapter]) -> String {
    let def = best(i).ok().map(|d| d.get_info());
    let mut s = String::from("GPUs, for --gpu <index or a part of the name>:");
    for (k, a) in a.iter().enumerate() {
        let (x, [d, e]) = (a.get_info(), gpu::codecs(a));
        let v = format!("{} {}", x.driver, x.driver_info);
        let drv = match v.trim() {
            "" => String::new(),
            v => format!(", driver {v}"),
        };
        let m = if def.as_ref() == Some(&x) {
            " (the default)"
        } else {
            ""
        };
        s += &format!("\n  {k}: {}{drv}; decodes {d}; encodes {e}{m}", x.name);
    }
    if a.is_empty() {
        s += "\n  none found";
    }
    s
}

/// `x5desk --gpu list`: every adapter, as [`table`] shows them.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub fn gpus() -> String {
    let i = gpu::instance();
    table(
        &i,
        &pollster::block_on(i.enumerate_adapters(wgpu::Backends::all())),
    )
}

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

#[cfg(test)]
mod tests {
    /// `--gpu`'s value: an adapter's index, else a part of one adapter's name in any case (an index past the last one
    /// too); one in no name or in several is an error.
    #[test]
    fn pick() {
        let n = [
            "NVIDIA GeForce RTX 4090",
            "AMD Radeon 8060S Graphics (RADV GFX1151)",
            "AMD Radeon RX 7900 XTX (RADV NAVI31)",
        ]
        .map(String::from);
        let p = |w| super::pick(&n, w);
        assert_eq!(
            [
                p("0"),
                p("2"),
                p("4090"),
                p("rtx"),
                p("8060S"),
                p("xtx"),
                p("31")
            ],
            [Ok(0), Ok(2), Ok(0), Ok(0), Ok(1), Ok(2), Ok(2)]
        );
        assert!(p("radeon").unwrap_err().contains("in 2 GPUs' names"));
        assert!(p("intel").unwrap_err().contains("no GPU"));
        assert!(p("12").unwrap_err().contains("no GPU"));
    }
}
