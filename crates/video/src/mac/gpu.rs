//! The one Metal device: wgpu's queue, which the decoders' copies out of VideoToolbox's pictures and the encoders'
//! copies into theirs also go through, so wgpu, egui, the stitcher and the video code share it.
use super::sys::{self, Cf};
use crate::{Error, Result};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;
use wgpu_hal::api::Metal as Mtl;

pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// the device under wgpu's
    pub mtl: Retained<ProtocolObject<dyn MTLDevice>>,
    /// IOSurface-backed pictures as Metal textures (CVMetalTextureCache; `flush` after making some)
    pub(crate) cache: Cf,
}

/// wgpu's Metal instance.
pub(crate) fn instance() -> wgpu::Instance {
    let mut d = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    d.backends = wgpu::Backends::METAL;
    wgpu::Instance::new(d)
}

/// VideoToolbox decodes and encodes both, whichever the adapter, as `--gpu list` says it.
pub(crate) fn codecs(_: &wgpu::Adapter) -> [String; 2] {
    ["H.265, H.264 (VideoToolbox)"; 2].map(String::from)
}

impl Gpu {
    /// Opens the high-performance Metal adapter (`--gpu`'s: `crate::PICK`) with every adapter feature and the
    /// experimental features. VideoToolbox decodes and encodes on the media engine and hands over IOSurfaces, so there
    /// are no video queues to open.
    pub fn new() -> Result<Gpu> {
        let err = |e: &dyn std::fmt::Display| Error::Gpu(e.to_string());
        let instance = instance();
        let adapter = crate::adapter(&instance)?;
        let (features, limits) = (adapter.features(), adapter.limits());
        if !features.contains(wgpu::Features::TEXTURE_FORMAT_16BIT_NORM) {
            return Err(err(&"no 16-bit normalised textures (P010 planes)"));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("x5desk"),
            required_features: features,
            required_limits: limits,
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .map_err(|e| err(&e))?;
        let mtl = unsafe { device.as_hal::<Mtl>() }
            .ok_or_else(|| err(&"not a Metal device"))?
            .raw_device()
            .clone();
        let mut cache = std::ptr::null();
        let r = unsafe {
            sys::CVMetalTextureCacheCreate(
                std::ptr::null(),
                std::ptr::null(),
                Retained::as_ptr(&mtl) as *mut _,
                std::ptr::null(),
                &mut cache,
            )
        };
        let cache = Cf::own(cache)
            .filter(|_| r == 0)
            .ok_or_else(|| err(&format!("CVMetalTextureCacheCreate: {r}")))?;
        Ok(Gpu {
            instance,
            adapter,
            device,
            queue,
            mtl,
            cache,
        })
    }

    /// The texture cache's housekeeping, which CoreVideo leaves to its user (CVMetalTextureCacheFlush): textures
    /// nothing holds any more and older than its age-out (1 s) go, with the IOSurfaces they keep (old sessions'
    /// pictures). Cheap: each decoded picture calls it.
    pub(crate) fn flush(&self) {
        unsafe { sys::CVMetalTextureCacheFlush(self.cache.0, 0) };
    }

    /// What an idle host gives back: the texture cache's old pictures (`flush`; VideoToolbox keeps no decoder memory
    /// of ours parked, as Linux's bin).
    pub fn trim(&self) {
        self.flush()
    }
}
