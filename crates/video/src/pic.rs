//! What both decode backends share: the codecs, ring and DPB sizes, and the picture management the parameter set
//! parsers do (`h264` / `h265`): DPB slots, POC, and which leading pictures to skip.
use crate::{Result, h264, h265};

/// Output ring size (the least; `Decoder::with_out` makes it larger): how far decoding runs ahead of the consumer.
/// Also the bitstream and command buffer ring.
pub const OUT: usize = 4;
/// Most DPB slots: 16 references and the current picture.
pub const MAXS: usize = 17;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    H264,
    H265,
}

/// The picture in a DPB slot. `num`: H.264 FrameNum, or LongTermFrameIdx when `long`. `none`: an H.264
/// "non-existing" frame (8.2.5.2) with no picture in its slot, never a reference slot of a decode.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Ref {
    pub live: bool,
    pub long: bool,
    pub num: u32,
    pub poc: [i32; 2],
    pub none: bool,
}

/// Picture management's answer: the current picture's slot, how many reference slots it listed, its coded size,
/// display window and bit depth (10 when `deep`, else 8). VideoToolbox keeps its own DPB: slots unused there.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) struct Pic {
    pub setup: usize,
    pub n: usize,
    pub w: u32,
    pub h: u32,
    pub view: [u32; 4],
    pub deep: bool,
}

pub(crate) enum St {
    H264(h264::State),
    H265(h265::State),
}

impl St {
    /// 1: slice, 2: parameter set, 0: anything else
    pub fn kind(&self, n: &[u8]) -> u8 {
        match self {
            St::H264(_) => match h264::nal_type(n) {
                1..=5 => 1,
                7 | 8 => 2,
                _ => 0,
            },
            St::H265(_) => match h265::nal_type(n) {
                0..=31 => 1,
                32..=34 => 2,
                _ => 0,
            },
        }
    }

    pub fn put(&mut self, n: &[u8]) -> Result<bool> {
        match self {
            St::H264(s) => s.put(n),
            St::H265(s) => s.put(n),
        }
    }
}
