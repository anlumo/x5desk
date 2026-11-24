//! Seam flow with an openly licensed network: torchvision RAFT-small (weights C_T_V2, BSD-3-Clause, `models/`),
//! compiled in by burn-onnx (`build.rs`) and run by burn on the shared wgpu device. The network sees the flow region
//! of the belt (RAFT needs at least 128 px per side); `post.wgsl` extends its flow over the rest. Nothing goes
//! through the CPU: the belt pass writes straight into the input tensors' buffers, and the flows are copied out of
//! the output tensor's buffer.

use burn::backend::Cube;
use burn::cubecl::wgpu::{AutoCompiler, WgpuResource, WgpuServer};
use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::module::{avg_pool2d, max_pool2d};
use burn::tensor::ops::{
    AvgPoolOptions, GridSampleOptions, GridSamplePaddingMode, InterpolateMode, MaxPoolOptions,
    PadMode,
};
use burn::tensor::{DType, Device, FloatDType, Tensor, TensorData, s, wgpu::WgpuSetup};
use burn_cubecl::CubeBackend;
use std::fs::{DirBuilder, File, OpenOptions, TryLockError};
use std::io::ErrorKind;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[allow(clippy::all, unused)]
mod model {
    include!(concat!(env!("OUT_DIR"), "/model/raft_small_128x2176.rs"));
    use burn::tensor::DType;

    /// Every generated conv call (`build.rs`): one with f16 weights runs on an f16 copy of its input.
    fn hc(c: &Conv2d, x: Tensor<4>) -> Tensor<4> {
        if c.weight.val().dtype() == DType::F16 {
            c.forward(x.cast(DType::F16)).cast(DType::F32)
        } else {
            c.forward(x)
        }
    }
}

/// Generated model shape (`build.rs` input `models/raft_small_128x2176.onnx`): flow region columns, rows.
pub const SIZE: (u32, u32) = (128, 2176);

/// Convs (in, out channels) run in f16: the GRU (242 → 96, ×3 per iteration) and the flow head's hidden layer
/// (96 → 128), ≈ 84 % of the update block's FLOPs; on Vulkan they then run on the cooperative-matrix units. The rest
/// stays f32: the convs on the flow itself (2 → 64 in, 128 → 2 out) drift by pixels in f16.
const HALF: [(usize, usize); 2] = [(242, 96), (96, 128)];

/// Casts the `HALF` conv weights and their biases to f16.
struct Half(bool);

impl ModuleMapper for Half {
    fn map_float<const D: usize>(&mut self, p: Param<Tensor<D>>) -> Param<Tensor<D>> {
        // a conv weight [out, in, k, k] decides; its bias comes next
        let half = match D {
            4 => {
                let d = p.val().dims();
                self.0 = HALF.contains(&(d[1], d[0]));
                self.0
            }
            1 => std::mem::take(&mut self.0),
            _ => false,
        };
        if half {
            p.map(|t| t.cast(FloatDType::F16))
        } else {
            p
        }
    }
}

type Res = cubecl_runtime::storage::ManagedResource<WgpuResource>;

/// A tensor's storage on the wgpu device; holding it keeps the allocation in place.
fn storage(t: &Tensor<4>) -> Result<(Res, [u32; 4]), String> {
    let f = t
        .clone()
        .try_into_primitive::<Cube>()
        .map_err(|e| format!("RAFT tensor not on the cubecl backend: {e:?}"))?;
    let c = f.client.clone().resolve_tensor_float::<CubeBackend>(f);
    if c.dtype != DType::F32 {
        return Err(format!("RAFT tensor is {:?}, not f32", c.dtype));
    }
    let st = c.meta.strides();
    let r = c
        .client
        .get_resource::<WgpuServer<AutoCompiler>>(c.handle.clone())
        .map_err(|e| format!("RAFT tensor storage: {e:?}"))?;
    c.client.flush().map_err(|e| format!("RAFT flush: {e:?}"))?;
    Ok((r, std::array::from_fn(|i| st[i] as u32)))
}

/// Keeps what burn learns about the device (its measured throughput, the autotune results) under `base`, so RAFT's
/// first job on a later start only compiles its kernels (≈ 2 s instead of ≈ 10 s). Kept apart per GPU model (`ad`'s
/// PCI vendor and device, `<base>/<vendor>-<device>`): cubecl 0.11 files its autotune picks under the device's index
/// in the process, the same on every GPU, so it would run another GPU's picks here, and one that doesn't fit this GPU
/// panics instead of being tuned away (the Radeon 8060S's pick for RAFT's f16 convolutions, `simple_sync_cmma`, asks
/// the RTX 4090 for 90112 bytes of workgroup memory, which has 49152). One process at a time per dir:
/// cubecl's database (cubecl-environment 0.11 on turso 0.8's experimental multi-process WAL) loses acknowledged
/// writes, reads corrupt pages and panics when processes share it, so this takes the first slot `<n>` there it can
/// lock (the kernel unlocks it when the process ends, however it ends). With no `base`, or no slot locked there (no
/// write access, no locks on that file system, a symlink as a lock), it keeps nothing: cubecl keeps what it learns
/// in memory. `base` is a dir no other user can write to. No build records (a session per run). The first call wins
/// (burn reads its configuration once), `attach`'s default (cubecl's own root) included; cubecl.toml files are not
/// read, the `CUBECL_*` variables are.
pub fn cache(base: Option<&Path>, ad: &wgpu::Adapter) {
    use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
    if CubeClRuntimeConfig::storage().lock().is_some() {
        return;
    }
    let i = ad.get_info();
    let gpu = format!("{:04x}-{:04x}", i.vendor, i.device);
    let d = match base.and_then(|b| slot(&b.join(gpu), File::try_lock)) {
        Some((d, l)) => {
            // held until the process ends
            std::mem::forget(l);
            d
        }
        // no dir can be made there, so cubecl's database can't be opened: it falls back to memory
        None => "/dev/null".into(),
    };
    let mut c = CubeClRuntimeConfig::default().override_from_env();
    c.environment.path = CacheConfig::Directory(d);
    c.environment.records.level = burn::cubecl::records::RecordLevel::Off;
    CubeClRuntimeConfig::try_set(c);
}

/// The first dir `<base>/<n>` (made owner-only) whose `lock` this process locks with `lock`, and the lock (held while
/// the file lives). None once anything fails but a lock held elsewhere: a slot used unlocked could be shared.
fn slot(base: &Path, lock: fn(&File) -> Result<(), TryLockError>) -> Option<(PathBuf, File)> {
    let mut b = DirBuilder::new();
    b.recursive(true).mode(0o700);
    for n in 0u32.. {
        let d = base.join(n.to_string());
        b.create(&d).ok()?;
        // never through a symlink: made with O_EXCL, or a plain file there already
        let (p, mut o) = (d.join("lock"), OpenOptions::new());
        o.write(true).mode(0o600);
        let f = match o.clone().create_new(true).open(&p) {
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                p.symlink_metadata().ok()?.is_file().then(|| o.open(&p))?
            }
            r => r,
        }
        .ok()?;
        match lock(&f) {
            Ok(()) => return Some((d, f)),
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Error(_)) => return None,
        }
    }
    None
}

/// burn's device per wgpu device. cubecl registers a new runtime for every setup it is handed, and every registration
/// installs cubecl's own uncaptured-error handler (it panics on any error) and device-lost callback on the device
/// (cubecl-wgpu 0.11 `PoisonWatch::watch`, called while registering), replacing the caller's. So burn attaches once
/// per device, and everything after that reuses the attachment. Keyed by instance and device: a `wgpu::Device`
/// compares by its id alone, and ids restart in every instance.
static BURN: Mutex<Vec<(wgpu::Instance, wgpu::Device, Device)>> = Mutex::new(Vec::new());

/// burn on the shared device, attached on the first call for `dev` and reused after that (by `Raft::new` too); the
/// device for other burn models on it (crate track). cubecl replaces the device's uncaptured-error handler and
/// device-lost callback while attaching: call this right after opening the device, before installing your own, then
/// nothing replaces them (wgpu has no way to read a handler back, so it can't be restored, and a re-install after
/// `Raft::new` leaves its ≈ 150 ms model load with cubecl's handler, which panics on any error). Attaching alone takes
/// ≈ 0.2 ms (`tests/stitch.rs` `errors`). `ad`, `inst`: what `dev` was opened from; burn shares the
/// queue, so its submits interleave with the caller's (safe around a decoder's `acquire` / `release`: staged ops stay
/// ordered), at most 4 kernels each: with cubecl's 32 a RAFT run held the queue up to 30 ms on the Radeon 8060S, and
/// a played frame's draw waited behind it (9 ms with 4; a run 80 instead of 75 ms, a 5.7K RAFT export 12.3 instead of
/// 12.7 fps, its output the same).
pub fn attach(
    inst: &wgpu::Instance,
    ad: &wgpu::Adapter,
    dev: &wgpu::Device,
    q: &wgpu::Queue,
) -> Result<Device, String> {
    let mut b = BURN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((.., d)) = b.iter().find(|(i, w, _)| i == inst && w == dev) {
        return Ok(d.clone());
    }
    // for a user that chose no cache dir (tests): cubecl's own root
    let root = burn::cubecl::config::cache::CacheConfig::Target.root();
    cache(Some(&root), ad);
    let setup = WgpuSetup {
        instance: inst.clone(),
        adapter: ad.clone(),
        device: dev.clone(),
        queue: q.clone(),
        backend: ad.get_info().backend,
    };
    let d = Device::wgpu_options()
        .setup(setup)
        .tasks_max(4)
        .init()
        .map_err(|e| format!("burn on the shared device: {e:?}"))?;
    b.push((inst.clone(), dev.clone(), d.clone()));
    Ok(d)
}

pub struct Raft {
    m: model::Model,
    dev: Device,
    /// Sampling grid of the flow region, [1, H, W, (x, y)] in [-1, 1] (align corners).
    base: Tensor<4>,
    /// Inputs [1, 3, H, W]: lens 0's belt, lens 1's belt; the belt pass writes their storage.
    inp: [Tensor<4>; 2],
    pub(crate) io: [Res; 2],
}

/// Region flows of one run: L2R and R2L, [2, 2 (x, y), H, W] f32 with element strides `st`.
pub(crate) struct Out {
    pub(crate) r: Res,
    pub(crate) st: [u32; 4],
}

impl Raft {
    /// RAFT on the shared device, through `attach` (which replaces the device's uncaptured-error handler on the
    /// first attach to `dev`: attach first, then install yours). Several `Raft`s on one device share burn's runtime
    /// (memory pools, compiled and autotuned kernels); each has its own inputs, so each can run on its own thread.
    /// The first run on a device compiles and autotunes the kernels (seconds).
    pub fn new(
        inst: &wgpu::Instance,
        ad: &wgpu::Adapter,
        dev: &wgpu::Device,
        q: &wgpu::Queue,
    ) -> Result<Self, String> {
        let dev = attach(inst, ad, dev, q)?;
        let m = model::Model::from_embedded(&dev);
        let m = if dev.supports_dtype(FloatDType::F16) {
            m.map(&mut Half(false))
        } else {
            m
        };
        let (w, h) = (SIZE.0 as usize, SIZE.1 as usize);
        let g: Vec<f32> = (0..w * h)
            .flat_map(|i| {
                let (y, x) = (i / w, i % w);
                [x as f32 / (w - 1) as f32, y as f32 / (h - 1) as f32].map(|v| 2.0 * v - 1.0)
            })
            .collect();
        let base = Tensor::from_data(TensorData::new(g, [1, h, w, 2]), &dev);
        let inp = [0, 1].map(|_| Tensor::<4>::zeros([1, 3, h, w], &dev));
        let [(a, sa), (b, sb)] = [storage(&inp[0])?, storage(&inp[1])?];
        // the belt pass writes contiguous NCHW
        let c = [3 * h * w, h * w, w, 1].map(|v| v as u32);
        if sa != c || sb != c {
            return Err(format!(
                "RAFT input strides {sa:?} / {sb:?}, belt writes {c:?}"
            ));
        }
        let io = [a, b];
        Ok(Self {
            m,
            dev,
            base,
            inp,
            io,
        })
    }

    /// Both directions, then the consistency fill and nearer-wins; submitted, not waited for.
    pub(crate) fn run(&self) -> Result<Out, String> {
        let [i0, i1] = &self.inp;
        let fl = Tensor::cat(
            vec![
                self.m.forward(i0.clone(), i1.clone()),
                self.m.forward(i1.clone(), i0.clone()),
            ],
            0,
        );
        let (r, st) = storage(&self.post(fl))?;
        Ok(Out { r, st })
    }

    /// Forward-backward check per direction: weight m = 1 while |f + b(x + f)|² ≤ t² = 0.01(|f|² + |b|²) + 0.5 px²,
    /// falling linearly to 0 at 4t². Doubtful pixels (no reliable match, e.g. dark floor or an object only one lens
    /// sees) blend toward the m-weighted normalized-convolution average of their neighbours (three box passes ≈
    /// Gaussian σ 8 px) instead of flipping from frame to frame. Then nearer parallax wins at depth edges: L2R x
    /// (nearer = more negative) takes the minimum, R2L x the maximum over 9 × 9, so a close object's edge moves with
    /// the object. In and out: [2 (L2R, R2L), 2 (x, y), H, W].
    fn post(&self, fl: Tensor<4>) -> Tensor<4> {
        let [_, _, h, w] = fl.dims();
        let other = Tensor::cat(
            vec![fl.clone().slice(s![1..2]), fl.clone().slice(s![0..1])],
            0,
        );
        // b = other(x + f), border-clamped bilinear; the grid is assembled with `cat`, not `permute`: burn 0.22's
        // grid_sample misreads a fused, permuted (strided) grid
        let px = [2.0 / (w - 1) as f32, 2.0 / (h - 1) as f32];
        let d = |c: usize| {
            fl.clone()
                .slice(s![.., c..c + 1])
                .mul_scalar(px[c])
                .reshape([2, h, w, 1])
        };
        let grid = self.base.clone() + Tensor::cat(vec![d(0), d(1)], 3);
        let b = other.grid_sample_2d(
            grid,
            GridSampleOptions::new(InterpolateMode::Bilinear)
                .with_padding_mode(GridSamplePaddingMode::Border)
                .with_align_corners(true),
        );
        let sq = |t: Tensor<4>| (t.clone() * t).sum_dim(1);
        let t2 = (sq(fl.clone()) + sq(b.clone()))
            .mul_scalar(0.01)
            .add_scalar(0.5);
        let m = (sq(fl.clone() + b) / t2)
            .neg()
            .add_scalar(4.0)
            .div_scalar(3.0)
            .clamp(0.0, 1.0);
        // normalized convolution of (x·m, y·m, m): three edge-clamped box passes of radius 8, rows then columns
        let mut c = Tensor::cat(vec![fl.clone() * m.clone(), m.clone()], 1);
        for _ in 0..3 {
            c = avg_pool2d(
                c.pad([(0, 0), (8, 8)], PadMode::Edge),
                AvgPoolOptions::new([1, 17]).with_stride([1, 1]),
            );
            c = avg_pool2d(
                c.pad([(8, 8), (0, 0)], PadMode::Edge),
                AvgPoolOptions::new([17, 1]).with_stride([1, 1]),
            );
        }
        let d = c.clone().slice(s![.., 2..3]);
        let fill = (c.slice(s![.., 0..2]) / d.clone().clamp_min(1e-6))
            .mask_fill(d.lower_equal_elem(1e-6), 0.0);
        let fl = fl * m.clone() + fill * m.neg().add_scalar(1.0);
        let sign = Tensor::<1>::from_floats([-1.0, 1.0], &self.dev).reshape([2, 1, 1, 1]);
        let x = max_pool2d(
            fl.clone().slice(s![.., 0..1]) * sign.clone(),
            MaxPoolOptions::new([9, 9])
                .with_stride([1, 1])
                .with_padding([4, 4]),
        ) * sign;
        Tensor::cat(vec![x, fl.slice(s![.., 1..2])], 1)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File, TryLockError};
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::path::PathBuf;

    struct Rm(PathBuf);

    impl Drop for Rm {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn base(n: &str) -> Rm {
        let run = std::env::var_os("XDG_RUNTIME_DIR").map_or_else(std::env::temp_dir, Into::into);
        Rm(run.join(format!("x5desk-{n}-{}", std::process::id())))
    }

    /// A dir another lock holds is skipped (in one process too: `flock` locks belong to the open file), a released one
    /// is taken again, so dirs are reused and never outnumber the processes that held them at once. Slots are
    /// owner-only.
    #[test]
    fn slots() {
        let b = base("slots");
        let (d0, l0) = super::slot(&b.0, File::try_lock).unwrap();
        let (d1, _l1) = super::slot(&b.0, File::try_lock).unwrap();
        assert_eq!((&d0, d1), (&b.0.join("0"), b.0.join("1")));
        assert_eq!(
            fs::metadata(&d0).unwrap().permissions().mode() & 0o777,
            0o700
        );
        drop(l0);
        assert_eq!(super::slot(&b.0, File::try_lock).unwrap().0, d0);
    }

    /// No slot where locking fails but on a lock held elsewhere (`ENOLCK`: NFS without lockd, some FUSE file
    /// systems), and none through a symlink as a lock: what it points to stays as it was, or missing.
    #[test]
    fn no_slot() {
        let b = base("noslot");
        let nolck = |_: &File| Err(TryLockError::Error(std::io::Error::other("ENOLCK")));
        assert!(super::slot(&b.0, nolck).is_none());
        assert!(b.0.join("0/lock").is_file() && !b.0.join("1").exists());
        let (v, gone) = (b.0.join("victim"), b.0.join("gone"));
        fs::write(&v, "kept").unwrap();
        for t in [&v, &gone] {
            fs::remove_file(b.0.join("0/lock")).unwrap();
            symlink(t, b.0.join("0/lock")).unwrap();
            assert!(super::slot(&b.0, File::try_lock).is_none(), "{t:?}");
        }
        assert_eq!(fs::read_to_string(&v).unwrap(), "kept");
        assert!(!gone.exists() && !b.0.join("1").exists());
    }
}
