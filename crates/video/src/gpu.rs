//! The one Vulkan device: wgpu's graphics / compute queue plus the video decode and encode queues, opened through
//! wgpu-hal so wgpu, egui, the stitcher and the video code share it.
use crate::{Error, Result};
use ash::{ext, khr, vk};
use std::{
    ffi::CStr,
    sync::{Arc, Mutex},
};
use wgpu_hal::api::Vulkan as Vk;

/// A video queue family and its queue 0. `vkQueueSubmit` needs the queue externally synchronised: hold the lock
/// (decode and encode share one lock when they share the family).
pub struct VideoQueue {
    pub family: u32,
    pub ops: vk::VideoCodecOperationFlagsKHR,
    pub queue: Arc<Mutex<vk::Queue>>,
}

pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub entry: ash::Entry,
    pub inst: ash::Instance,
    pub dev: ash::Device,
    pub pd: vk::PhysicalDevice,
    /// wgpu's queue family
    pub gfx: u32,
    pub dec: VideoQueue,
    /// None when the driver encodes on no queue family but wgpu's
    pub enc: Option<VideoQueue>,
    pub mem: vk::PhysicalDeviceMemoryProperties,
    /// every enabled device extension, wgpu's and ours
    pub ext: Vec<&'static CStr>,
    // ash 0.38 has only raw function tables for the video extensions
    pub vqi: khr::video_queue::Instance,
    pub vq: khr::video_queue::Device,
    pub vdq: khr::video_decode_queue::Device,
    pub veqi: khr::video_encode_queue::Instance,
    pub veq: khr::video_encode_queue::Device,
    pub mem_fd: Option<khr::external_memory_fd::Device>,
    pub sem_fd: Option<khr::external_semaphore_fd::Device>,
}

const EXT: [&CStr; 13] = [
    khr::video_queue::NAME,
    khr::video_decode_queue::NAME,
    khr::video_decode_h264::NAME,
    khr::video_decode_h265::NAME,
    khr::video_encode_queue::NAME,
    khr::video_encode_h264::NAME,
    khr::video_encode_h265::NAME,
    khr::video_maintenance1::NAME,
    khr::external_memory_fd::NAME,
    ext::external_memory_dma_buf::NAME,
    ext::image_drm_format_modifier::NAME,
    khr::external_semaphore_fd::NAME,
    // DMA-BUF images that a foreign consumer (OBS, PipeWire) reads: crate live hands them to QUEUE_FAMILY_FOREIGN
    ext::queue_family_foreign::NAME,
];

impl Gpu {
    /// Opens the high-performance Vulkan adapter with every adapter feature, the experimental features, the decode
    /// and encode queues, and whichever of the video and external memory / semaphore extensions the driver has.
    pub fn new() -> Result<Gpu> {
        let err = |e: &dyn std::fmt::Display| Error::Gpu(e.to_string());
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        d.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(d);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| err(&e))?;
        let (features, limits) = (adapter.features(), adapter.limits());
        if !features.contains(wgpu::Features::TEXTURE_FORMAT_NV12) {
            return Err(err(&"no NV12 textures"));
        }
        let mut ext = vec![];
        let (open, inst, entry, pd, dec, enc) = unsafe {
            let hal = adapter
                .as_hal::<Vk>()
                .ok_or_else(|| err(&"not a Vulkan adapter"))?;
            let si = hal.shared_instance();
            let (inst, entry, pd) = (
                si.raw_instance().clone(),
                si.entry().clone(),
                hal.raw_physical_device(),
            );
            let n = inst.get_physical_device_queue_family_properties2_len(pd);
            let mut vp = vec![vk::QueueFamilyVideoPropertiesKHR::default(); n];
            let mut qp: Vec<_> = vp
                .iter_mut()
                .map(|v| vk::QueueFamilyProperties2::default().push_next(v))
                .collect();
            inst.get_physical_device_queue_family_properties2(pd, &mut qp);
            let fl: Vec<_> = qp
                .iter()
                .map(|q| q.queue_family_properties.queue_flags)
                .collect();
            drop(qp);
            let fam = |f| {
                fl.iter()
                    .position(|q: &vk::QueueFlags| q.contains(f))
                    .map(|i| (i as u32, vp[i].video_codec_operations))
            };
            let dec = fam(vk::QueueFlags::VIDEO_DECODE_KHR)
                .ok_or_else(|| err(&"no video decode queue"))?;
            let enc = fam(vk::QueueFlags::VIDEO_ENCODE_KHR);
            let have = inst.enumerate_device_extension_properties(pd)?;
            let add: Vec<_> = EXT
                .into_iter()
                .filter(|e| have.iter().any(|h| h.extension_name_as_c_str() == Ok(e)))
                .collect();
            let mut m1 =
                vk::PhysicalDeviceVideoMaintenance1FeaturesKHR::default().video_maintenance1(true);
            let m1 = add
                .contains(&khr::video_maintenance1::NAME)
                .then_some(&mut m1);
            static PRIO: [f32; 1] = [1.0];
            let out = &mut ext;
            let cb: Box<wgpu_hal::vulkan::CreateDeviceCallback> = Box::new(move |a| {
                for e in add {
                    if !a.extensions.contains(&e) {
                        a.extensions.push(e);
                    }
                }
                out.clone_from(a.extensions);
                for f in [Some(dec.0), enc.map(|e| e.0)].into_iter().flatten() {
                    // one create info per family (wgpu's included)
                    if !a
                        .queue_create_infos
                        .iter()
                        .any(|q| q.queue_family_index == f)
                    {
                        a.queue_create_infos.push(
                            vk::DeviceQueueCreateInfo::default()
                                .queue_family_index(f)
                                .queue_priorities(&PRIO),
                        );
                    }
                }
                if let Some(m1) = m1 {
                    *a.create_info = std::mem::take(a.create_info).push_next(m1);
                }
            });
            let open = hal
                .open_with_callback(features, &limits, &wgpu::MemoryHints::Performance, Some(cb))
                .map_err(|e| err(&e))?;
            (open, inst, entry, pd, dec, enc)
        };
        let (device, queue) = unsafe {
            adapter.create_device_from_hal(
                open,
                &wgpu::DeviceDescriptor {
                    label: Some("x5desk"),
                    required_features: features,
                    required_limits: limits,
                    experimental_features: wgpu::ExperimentalFeatures::enabled(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    ..Default::default()
                },
            )
        }
        .map_err(|e| err(&e))?;
        let (dev, gfx) = unsafe {
            let h = device
                .as_hal::<Vk>()
                .ok_or_else(|| err(&"not a Vulkan device"))?;
            (h.raw_device().clone(), h.queue_family_index())
        };
        // queue 0 of wgpu's family is wgpu's own queue, submitted to under wgpu's lock, not ours
        if dec.0 == gfx {
            return Err(err(&"video decode only on wgpu's queue family"));
        }
        let q = |(family, ops)| VideoQueue {
            family,
            ops,
            queue: Arc::new(Mutex::new(unsafe { dev.get_device_queue(family, 0) })),
        };
        let dec = q(dec);
        let enc = enc.filter(|e| e.0 != gfx).map(|e| match e.0 == dec.family {
            true => VideoQueue {
                family: e.0,
                ops: e.1,
                queue: dec.queue.clone(),
            },
            false => q(e),
        });
        let has = |e| ext.contains(&e);
        Ok(Gpu {
            vqi: khr::video_queue::Instance::new(&entry, &inst),
            vq: khr::video_queue::Device::new(&inst, &dev),
            vdq: khr::video_decode_queue::Device::new(&inst, &dev),
            veqi: khr::video_encode_queue::Instance::new(&entry, &inst),
            veq: khr::video_encode_queue::Device::new(&inst, &dev),
            mem_fd: has(khr::external_memory_fd::NAME)
                .then(|| khr::external_memory_fd::Device::new(&inst, &dev)),
            sem_fd: has(khr::external_semaphore_fd::NAME)
                .then(|| khr::external_semaphore_fd::Device::new(&inst, &dev)),
            mem: unsafe { inst.get_physical_device_memory_properties(pd) },
            dec,
            enc,
            ext,
            instance,
            adapter,
            device,
            queue,
            entry,
            inst,
            dev,
            pd,
            gfx,
        })
    }

    /// Device memory for `req` from the first allowed type that has `want`.
    pub fn alloc(
        &self,
        req: vk::MemoryRequirements,
        want: vk::MemoryPropertyFlags,
    ) -> Result<vk::DeviceMemory> {
        let m = &self.mem;
        let t = (0..m.memory_type_count)
            .find(|&i| {
                req.memory_type_bits >> i & 1 == 1
                    && m.memory_types[i as usize].property_flags.contains(want)
            })
            .ok_or(Error::Bad("no matching memory type"))?;
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(t);
        Ok(unsafe { self.dev.allocate_memory(&info, None) }?)
    }
}
