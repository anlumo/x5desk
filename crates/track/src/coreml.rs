//! macOS: YuNet through Core ML on the Neural Engine (`models/yunet_b8.mlpackage`, from the ONNX by
//! `models/coreml.py`), which leaves the GPU to the stitcher. On the M4 Core ML predicts a run of 8 tiles of 640² in
//! 9 ms (a Swift benchmark of the package); the face pass takes 12.5 ms per run of up to 8 with the copies, read-back
//! and decoding, where burn's YuNet takes 24 on the GPU (footage medians, `tests/footage.rs` `privacy`). The crops come
//! from the crop pass's buffer through a mapped copy (shared memory on Apple silicon), the scores and boxes go back in
//! burn's layout. burn's YuNet stays the fallback, and the other platforms'.
use crate::privacy::FS;
use objc2::{
    AnyThread,
    rc::{Retained, autoreleasepool},
    runtime::{AnyObject, ProtocolObject},
};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};
use std::{
    os::unix::fs::{DirBuilderExt, MetadataExt},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

/// Tiles per run: the package's fixed batch (`privacy::MAX`).
pub(crate) const B: usize = 8;
const FILES: [(&str, &[u8]); 3] = [
    (
        "Manifest.json",
        include_bytes!("../models/yunet_b8.mlpackage/Manifest.json"),
    ),
    (
        "Data/com.apple.CoreML/model.mlmodel",
        include_bytes!("../models/yunet_b8.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
    ),
    (
        "Data/com.apple.CoreML/weights/weight.bin",
        include_bytes!("../models/yunet_b8.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
    ),
];
/// Per stride: the score's two factors and the box, as the ONNX names them.
const OUT: [(usize, [&str; 3]); 3] = [
    (8, ["cls_8", "obj_8", "bbox_8"]),
    (16, ["cls_16", "obj_16", "bbox_16"]),
    (32, ["cls_32", "obj_32", "bbox_32"]),
];

pub(crate) struct Yunet {
    model: Retained<MLModel>,
    /// `B` tiles of f32 NCHW, mapped while Core ML reads them
    buf: wgpu::Buffer,
    /// a dir of this process's own the model was compiled into (`new` with nowhere to keep it), removed with it
    own: Option<PathBuf>,
}

// Core ML's models take predictions from any thread; `Ai` runs one at a time (BURN)
unsafe impl Send for Yunet {}

fn ns(e: Retained<objc2_foundation::NSError>) -> String {
    e.localizedDescription().to_string()
}

fn url(p: &Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&p.to_string_lossy()))
}

static N: AtomicUsize = AtomicUsize::new(0);

/// A name beside `p` no other process or call uses.
fn aside(p: &Path, what: &str) -> PathBuf {
    p.with_extension(format!(
        "{what}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Relaxed)
    ))
}

/// Where compiled models are kept between runs: `x5desk-coreml` in the temp dir, if it is a dir of this user's that
/// no other can write to (made 0700 when missing; in a shared tmp another user may have made it first, or put a
/// symlink there: then the model would be someone else's, so none is kept).
pub(crate) fn root() -> Option<PathBuf> {
    let d = std::env::temp_dir().join("x5desk-coreml");
    let _ = std::fs::DirBuilder::new().mode(0o700).create(&d);
    let m = d.symlink_metadata().ok()?;
    // SAFETY: geteuid has no preconditions
    let me = unsafe { libc::geteuid() };
    (m.is_dir() && m.uid() == me && m.mode() & 0o022 == 0).then_some(d)
}

/// The compiled model in `root`: one per package (its FNV-1a hash), shared by this user's processes.
pub(crate) fn cached(root: &Path) -> PathBuf {
    let h = FILES
        .iter()
        .flat_map(|(n, b)| n.bytes().chain(b.iter().copied()))
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
        });
    root.join(format!("yunet_b8-{h:016x}.mlmodelc"))
}

/// Writes the package beside `at`, compiles it and renames the compiled model to `at`, unless another process did
/// first: then that one stands.
fn compile(at: &Path) -> Result<(), String> {
    let dir = aside(at, "pkg");
    let r = (|| {
        let pkg = dir.join("yunet_b8.mlpackage");
        for (f, b) in FILES {
            let p = pkg.join(f);
            std::fs::create_dir_all(p.parent().unwrap())
                .map_err(|e| format!("{}: {e}", p.display()))?;
            std::fs::write(&p, b).map_err(|e| format!("{}: {e}", p.display()))?;
        }
        #[allow(deprecated)]
        let c = unsafe { MLModel::compileModelAtURL_error(&url(&pkg)) }.map_err(ns)?;
        let c = PathBuf::from(c.path().ok_or("compiled model has no path")?.to_string());
        let moved = std::fs::rename(&c, at);
        let _ = std::fs::remove_dir_all(&c);
        moved.or_else(|e| match at.is_dir() {
            true => Ok(()),
            false => Err(format!("{}: {e}", at.display())),
        })
    })();
    let _ = std::fs::remove_dir_all(&dir);
    r
}

fn load(at: &Path) -> Result<Retained<MLModel>, String> {
    unsafe {
        let cfg = MLModelConfiguration::new();
        cfg.setComputeUnits(MLComputeUnits::CPUAndNeuralEngine);
        MLModel::modelWithContentsOfURL_configuration_error(&url(at), &cfg).map_err(ns)
    }
}

impl Yunet {
    /// The model compiled once (`root`, `cached`) and loaded for the Neural Engine and the CPU: hundreds of ms with
    /// the Neural Engine's own compile, which Core ML caches too (under `~/Library/Caches/<process name>`).
    pub fn new(wd: &wgpu::Device) -> Result<Yunet, String> {
        match root() {
            Some(r) => Yunet::at(wd, &r),
            // nowhere to keep it: a dir of this process's own, removed with the model
            None => {
                let d = aside(&std::env::temp_dir().join("x5desk-coreml"), "own");
                std::fs::DirBuilder::new()
                    .mode(0o700)
                    .create(&d)
                    .map_err(|e| format!("{}: {e}", d.display()))?;
                let y = Yunet::at(wd, &d).map(|mut y| {
                    y.own = Some(d.clone());
                    y
                });
                if y.is_err() {
                    let _ = std::fs::remove_dir_all(&d);
                }
                y
            }
        }
    }

    /// [`Yunet::new`] with its compiled models kept in `root`.
    pub fn at(wd: &wgpu::Device, root: &Path) -> Result<Yunet, String> {
        let at = cached(root);
        // a thread of ours has no pool for what Core ML autoreleases
        let model = autoreleasepool(|_| match at.is_dir().then(|| load(&at)) {
            Some(Ok(m)) => Ok(m),
            stale => {
                // missing, or one that won't load (cut short, a macOS that compiles differently): made again. A stale
                // one is moved aside before it goes, so a good one another process puts there meanwhile stays.
                if stale.is_some() {
                    let t = aside(&at, "stale");
                    if std::fs::rename(&at, &t).is_ok() {
                        let _ = std::fs::remove_dir_all(&t);
                    }
                }
                compile(&at)?;
                load(&at)
            }
        })?;
        let buf = wd.create_buffer(&wgpu::BufferDescriptor {
            label: Some("YuNet input (Core ML)"),
            size: (B * 3 * (FS * FS) as usize * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Yunet {
            model,
            buf,
            own: None,
        })
    }

    /// Scores and boxes of the first `n` tiles in `src` (the crop pass's buffer: f32 NCHW, `FS`², at its start) as
    /// burn's path gives them: per tile, strides 8, 16, 32, per anchor √(cls · obj) (each clamped to 0..1) and the box.
    pub fn run(
        &self,
        wd: &wgpu::Device,
        q: &wgpu::Queue,
        src: &wgpu::Buffer,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        let s = (FS * FS) as usize;
        let mut e = wd.create_command_encoder(&Default::default());
        e.copy_buffer_to_buffer(src, 0, &self.buf, 0, (n * 3 * s * 4) as u64);
        let i = q.submit([e.finish()]);
        let sl = self.buf.slice(..);
        sl.map_async(wgpu::MapMode::Read, |_| {});
        let r = (|| {
            wd.poll(wgpu::PollType::Wait {
                submission_index: Some(i),
                timeout: None,
            })
            .map_err(|e| format!("map: {e}"))?;
            let m = sl.get_mapped_range().map_err(|e| format!("map: {e:?}"))?;
            // the tiles past `n` hold what an earlier run left there: their outputs are not read
            let p = NonNull::new(m.as_ptr() as *mut std::ffi::c_void).ok_or("no mapping")?;
            // what Core ML autoreleases (an array per output read, 12 MB a run) goes before the mapping does
            autoreleasepool(|_| unsafe { self.predict(p, n) })
        })();
        // also a map still pending after an error: it is cancelled
        self.buf.unmap();
        r
    }

    unsafe fn predict(&self, p: NonNull<std::ffi::c_void>, n: usize) -> Result<Vec<f32>, String> {
        let f = FS as usize;
        let num = |v: &[usize]| {
            NSArray::from_retained_slice(
                &v.iter()
                    .map(|&x| NSNumber::new_usize(x))
                    .collect::<Vec<_>>(),
            )
        };
        let input = unsafe {
            MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                MLMultiArray::alloc(),
                p,
                &num(&[B, 3, f, f]),
                MLMultiArrayDataType::Float32,
                &num(&[3 * f * f, f * f, f, 1]),
                None,
            )
        }
        .map_err(ns)?;
        let v = unsafe { MLFeatureValue::featureValueWithMultiArray(&input) };
        let d = NSDictionary::<NSString, AnyObject>::from_retained_objects(
            &[&*NSString::from_str("input")],
            &[Retained::into_super(Retained::into_super(v))],
        );
        let fp = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &d,
            )
        }
        .map_err(ns)?;
        let out = unsafe {
            self.model
                .predictionFromFeatures_error(ProtocolObject::from_ref(&*fp))
        }
        .map_err(ns)?;
        let get = |name: &str| -> Result<Out, String> {
            let a = unsafe { out.featureValueForName(&NSString::from_str(name)) }
                .and_then(|v| unsafe { v.multiArrayValue() })
                .ok_or_else(|| format!("no output {name}"))?;
            Out::new(a)
        };
        let arrs = OUT
            .iter()
            .map(|(st, [c, o, b])| Ok(((f / st).pow(2), [get(c)?, get(o)?, get(b)?])))
            .collect::<Result<Vec<_>, String>>()?;
        let m: usize = arrs.iter().map(|a| a.0).sum();
        let mut v = vec![0f32; n * m * 5];
        let mut rows = v.chunks_exact_mut(5);
        for t in 0..n {
            for (g, [c, o, b]) in &arrs {
                let (c, o, b) = (c.row(t, *g, 1), o.row(t, *g, 1), b.row(t, *g, 4));
                for k in 0..*g {
                    let r = rows.next().unwrap();
                    r[0] = (c[k].clamp(0.0, 1.0) * o[k].clamp(0.0, 1.0)).sqrt();
                    r[1..].copy_from_slice(&b[4 * k..][..4]);
                }
            }
        }
        Ok(v)
    }
}

/// An output array: its data, element type and strides (Core ML may pad rows for the Neural Engine).
struct Out {
    _a: Retained<MLMultiArray>,
    p: *const u8,
    f16: bool,
    st: [usize; 3],
}

impl Out {
    fn new(a: Retained<MLMultiArray>) -> Result<Out, String> {
        let t = unsafe { a.dataType() };
        let f16 = match t {
            MLMultiArrayDataType::Float32 => false,
            MLMultiArrayDataType::Float16 => true,
            t => return Err(format!("output type {t:?}")),
        };
        let st: Vec<usize> = unsafe { a.strides() }
            .iter()
            .map(|n| n.as_usize())
            .collect();
        let st = st.try_into().map_err(|s| format!("output strides {s:?}"))?;
        #[allow(deprecated)]
        let p = unsafe { a.dataPointer() }.as_ptr() as *const u8;
        Ok(Out { _a: a, p, f16, st })
    }

    /// Tile `t`'s `g` anchors × `c` values as f32, in order: borrowed when the array holds them so (f32, unpadded),
    /// else gathered.
    fn row(&self, t: usize, g: usize, c: usize) -> std::borrow::Cow<'_, [f32]> {
        let at = |k: usize, j: usize| {
            let i = t * self.st[0] + k * self.st[1] + j * self.st[2];
            unsafe {
                match self.f16 {
                    true => half((self.p as *const u16).add(i).read_unaligned()),
                    false => (self.p as *const f32).add(i).read_unaligned(),
                }
            }
        };
        if !self.f16 && self.st[1] == c && self.st[2] == 1 && (self.p as usize).is_multiple_of(4) {
            let p = unsafe { (self.p as *const f32).add(t * self.st[0]) };
            return std::borrow::Cow::Borrowed(unsafe { std::slice::from_raw_parts(p, g * c) });
        }
        (0..g * c).map(|i| at(i / c, i % c)).collect()
    }
}

/// IEEE 754 half to single precision.
pub(crate) fn half(h: u16) -> f32 {
    let (s, e, m) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1f) as u32,
        (h & 0x3ff) as u32,
    );
    let bits = match (e, m) {
        (0, 0) => s << 31,
        // subnormal: normalise
        (0, _) => {
            let z = m.leading_zeros() - 22;
            (s << 31) | ((112 - z) << 23) | ((m << (z + 1)) & 0x3ff) << 13
        }
        (31, _) => (s << 31) | 0x7f80_0000 | (m << 13),
        _ => (s << 31) | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

impl Drop for Yunet {
    fn drop(&mut self) {
        if let Some(d) = &self.own {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}
