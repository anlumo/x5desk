//! Segmentation networks, compiled in by burn-onnx (`build.rs`) and run by burn on the shared device as crate track
//! runs its networks (`models/README`): people with PP-HumanSeg (PaddleSeg HumanSeg mobile, HRNet-W18-Small v1 at 192²,
//! from the OpenCV zoo, Apache-2.0; input RGB as (c − ½) / ½, output the softmax over (background, person) per pixel),
//! the sky with EfficientViT-seg B1 on ADE20K (MIT HAN Lab, Apache-2.0; input RGB normalised by ImageNet's mean and
//! deviation, output the sky's share of the softmax over ADE20K's 150 classes at an eighth of the input's side). NCHW.
//! Nothing goes through the CPU: our pass writes the input tensor's buffer, the output is copied out of its own.

use burn::backend::Cube;
use burn::cubecl::wgpu::{AutoCompiler, WgpuResource, WgpuServer};
use burn::tensor::{DType, Device, Tensor, activation::softmax};
use burn_cubecl::CubeBackend;

#[allow(clippy::all, unused)]
pub(crate) mod model {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/human_segmentation_pphumanseg_2023mar.rs"
    ));
}

#[allow(clippy::all, unused)]
pub(crate) mod sky {
    include!(concat!(
        env!("OUT_DIR"),
        "/model/efficientvit_seg_b1_ade20k_6x384.rs"
    ));
}

/// Person network input and output side, pixels.
pub const SIDE: u32 = 192;
/// Sky network: tiles per run, their input and output side, pixels.
pub const TILES: u32 = 6;
pub const SKY: u32 = 384;
pub const SKY_OUT: u32 = SKY / 8;

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

enum M {
    Person(Box<model::Model>),
    Sky(Box<sky::Model>),
}

pub(crate) struct Net {
    m: M,
    inp: Tensor<4>,
    /// The input's storage (contiguous NCHW).
    pub(crate) io: Res,
    /// The last output and its storage, kept until the next run so the copy out of it stays valid.
    out: Option<(Tensor<4>, Res)>,
}

impl Net {
    /// The sky network with `sky`, else the person network.
    pub(crate) fn new(dev: &Device, sky: bool) -> Result<Self, String> {
        let (n, s, m) = match sky {
            false => (
                1,
                SIDE as usize,
                M::Person(Box::new(model::Model::from_embedded(dev))),
            ),
            true => (
                TILES as usize,
                SKY as usize,
                M::Sky(Box::new(sky::Model::from_embedded(dev))),
            ),
        };
        let inp = Tensor::<4>::zeros([n, 3, s, s], dev);
        let (io, st) = storage(&inp)?;
        if st != [3 * s * s, s * s, s, 1].map(|v| v as u32) {
            return Err(format!(
                "segmentation input strides {st:?}: our pass writes contiguous NCHW"
            ));
        }
        Ok(Self {
            m,
            inp,
            io,
            out: None,
        })
    }

    /// Runs on the input as last submitted; submitted, not waited for. The output's buffer (element 0 at its offset)
    /// and strides: the person probability of pixel (x, y) is element st[1] + st[2]·y + st[3]·x, the sky's of pixel
    /// (x, y) of tile t element st[0]·t + st[2]·y + st[3]·x.
    pub(crate) fn run(&mut self) -> Result<(&WgpuResource, [u32; 4]), String> {
        let o = match &self.m {
            M::Person(m) => m.forward(self.inp.clone()),
            // ADE20K's class 2
            M::Sky(m) => softmax(m.forward(self.inp.clone()), 1).narrow(1, 2, 1),
        };
        let (r, st) = storage(&o)?;
        let (_, r) = self.out.insert((o, r));
        Ok((r.resource(), st))
    }
}
