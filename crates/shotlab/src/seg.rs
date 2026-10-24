//! Person segmentation: PP-HumanSeg (PaddleSeg HumanSeg mobile, HRNet-W18-Small v1 at 192², from the OpenCV zoo,
//! Apache-2.0; `models/README`), compiled in by burn-onnx (`build.rs`) and run by burn on the shared device as crate
//! track runs its networks. Input RGB as (c − ½) / ½, NCHW; output the softmax over (background, person) per pixel.
//! Nothing goes through the CPU: our pass writes the input tensor's buffer, the output is copied out of its own.

use burn::backend::Cube;
use burn::cubecl::wgpu::{AutoCompiler, WgpuResource, WgpuServer};
use burn::tensor::{DType, Device, Tensor};
use burn_cubecl::CubeBackend;

#[allow(clippy::all, unused)]
pub(crate) mod model {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/human_segmentation_pphumanseg_2023mar.rs"
    ));
}

/// Network input and output side, pixels.
pub const SIDE: u32 = 192;

pub(crate) type Res = cubecl_runtime::storage::ManagedResource<WgpuResource>;

/// A tensor's storage on the wgpu device and its element strides; holding it keeps the allocation in place.
fn storage(t: &Tensor<4>) -> Result<(Res, [u32; 4]), String> {
    let f = t
        .clone()
        .try_into_primitive::<Cube>()
        .map_err(|e| format!("segmentation tensor not on the cubecl backend: {e:?}"))?;
    let c = f.client.clone().resolve_tensor_float::<CubeBackend>(f);
    if c.dtype != DType::F32 {
        return Err(format!("segmentation tensor is {:?}, not f32", c.dtype));
    }
    let st = c.meta.strides();
    let r = c
        .client
        .get_resource::<WgpuServer<AutoCompiler>>(c.handle.clone())
        .map_err(|e| format!("segmentation tensor storage: {e:?}"))?;
    c.client.flush().map_err(|e| format!("flush: {e:?}"))?;
    Ok((r, std::array::from_fn(|i| st[i] as u32)))
}

pub(crate) struct Net {
    m: model::Model,
    inp: Tensor<4>,
    /// The input's storage (contiguous NCHW, one picture).
    pub(crate) io: Res,
    /// The last output and its storage, kept until the next run so the copy out of it stays valid.
    out: Option<(Tensor<4>, Res)>,
}

impl Net {
    pub(crate) fn new(dev: &Device) -> Result<Self, String> {
        let s = SIDE as usize;
        let inp = Tensor::<4>::zeros([1, 3, s, s], dev);
        let (io, st) = storage(&inp)?;
        if st != [3 * s * s, s * s, s, 1].map(|v| v as u32) {
            return Err(format!(
                "segmentation input strides {st:?}: our pass writes contiguous NCHW"
            ));
        }
        Ok(Self {
            m: model::Model::from_embedded(dev),
            inp,
            io,
            out: None,
        })
    }

    /// Runs on the input as last submitted; submitted, not waited for. The output's buffer (element 0 at its offset)
    /// and strides: the person probability of pixel (x, y) is element st[1] + st[2]·y + st[3]·x.
    pub(crate) fn run(&mut self) -> Result<(&WgpuResource, [u32; 4]), String> {
        let o = self.m.forward(self.inp.clone());
        let (r, st) = storage(&o)?;
        let (_, r) = self.out.insert((o, r));
        Ok((r.resource(), st))
    }
}
