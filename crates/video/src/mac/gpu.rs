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
    /// IOSurface-backed pictures as Metal textures (CVMetalTextureCache)
    pub(crate) cache: Cf,
}

impl Gpu {
    /// Opens the Metal adapter with every adapter feature and the experimental features. VideoToolbox decodes and
    /// encodes on the media engine and hands over IOSurfaces, so there are no video queues to open.
    pub fn new() -> Result<Gpu> {
        let err = |e: &dyn std::fmt::Display| Error::Gpu(e.to_string());
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        d.backends = wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(d);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| err(&e))?;
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
}
