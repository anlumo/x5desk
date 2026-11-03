//! What both encode backends share: the configuration and the packets.
use crate::decode::Codec;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rate {
    /// Constant bitrate, bit/s. RADV 26.2 starts at QP 25 whatever the target, so the first IDR can outgrow the
    /// buffer (`Cfg::vbv`): at 1080p 8 Mbit/s on the X5's live preview it is 697 kB with a 250, 500 or 1000 ms buffer
    /// (90 % full: 225, 450, 900 kB) and 116 kB with 100 ms (90 kB). The following frames pay it back.
    Cbr(u64),
    /// Variable bitrate: average and peak bit/s.
    Vbr(u64, u64),
    /// Constant QP, no rate control.
    Qp(u8),
}

#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub codec: Codec,
    /// 10-bit: P010 input, H.265 Main 10.
    pub deep: bool,
    /// Picture size, even.
    pub w: u32,
    pub h: u32,
    /// Frame rate numerator, denominator.
    pub fps: [u32; 2],
    /// IDR period in frames.
    pub gop: u32,
    pub rate: Rate,
    /// CBR / VBR: the rate control buffer (the decoder's buffering) in ms of the average bitrate, starting 90 % full
    /// (RADV ignores the initial occupancy). 1000 for files; 100 for low latency (RTMP), which also cuts the start-up
    /// burst (`Rate::Cbr`). RADV's CBR holds the target from a buffer of about 500 ms on; with less the P-frames run
    /// over their share and the excess is never paid back: +4 to +10 % at 100 ms (each IDR capped at about the
    /// buffer), up to +2.4 % at 250 ms (Mesa 26.2.5, the `cbr` test).
    pub vbv: u32,
    /// Driver quality level (higher: better and slower, per the spec), clamped to the highest usable
    /// (`Encoder::levels`). RADV (Mesa 26.2.5): 0 and 1 write identical H.265 streams (both the firmware's balanced
    /// preset), 2 measures 0.2 dB lower PSNR(Y) on the X5's footage.
    pub quality: u32,
    /// Full range (0–255) instead of limited (16–235); BT.709 primaries, transfer and matrix either way.
    pub full: bool,
    /// Chroma sample location: 0 left (the X5, most video), 1 centre (the 2 × 2 mean of crate stitch), 2 top-left.
    pub chroma: u8,
}

/// One encoded access unit. Reuse it across `Encoder::packet` calls: the vectors keep their allocations.
#[derive(Clone, Debug, Default)]
pub struct Packet {
    /// Annex B as the encoder wrote it: the picture's NAL units with start codes, no parameter sets.
    pub data: Vec<u8>,
    /// Each NAL unit in `data`, without its start code.
    pub nals: Vec<Range<usize>>,
    pub frame: u64,
    pub pts: i64,
    /// Equal to `pts`: there is no reordering.
    pub dts: i64,
    /// IDR
    pub key: bool,
}

impl Packet {
    /// The NAL units without start codes (crate live's FLV writer).
    pub fn nal(&self) -> impl Iterator<Item = &[u8]> {
        self.nals.iter().map(|r| &self.data[r.clone()])
    }

    /// Appends the NAL units with 4-byte big-endian lengths: an MP4 sample (crate media's `Mux::write`).
    pub fn prefixed(&self, o: &mut Vec<u8>) {
        for n in self.nal() {
            o.extend_from_slice(&(n.len() as u32).to_be_bytes());
            o.extend_from_slice(n);
        }
    }
}
