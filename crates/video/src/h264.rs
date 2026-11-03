//! H.264 parameter sets and slice headers as Vulkan Video decode needs them (progressive 4:2:0 8-bit up to High,
//! POC types 0 and 2, no scaling matrices), plus POC (8.2.1) and reference marking (8.2.5: gaps in frame_num,
//! sliding window and MMCO 1-6).
use crate::{
    Error::{Bad, Skip},
    Result,
    bits::Bits,
    decode::{MAXS, Pic, Ref},
};
use ash::vk::native::*;

pub fn nal_type(n: &[u8]) -> u8 {
    n[0] & 0x1f
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Sps {
    pub profile_idc: u8,
    pub constraints: u8,
    pub level_idc: u8,
    pub level: StdVideoH264LevelIdc,
    pub id: u8,
    pub qpprime_bypass: bool,
    pub log2_max_frame_num: u32,
    pub poc_type: u8,
    pub log2_max_poc_lsb: u32,
    pub max_num_ref_frames: u8,
    pub gaps: bool,
    pub width_mbs: u32,
    pub height_mbs: u32,
    pub direct_8x8: bool,
    pub crop: Option<[u32; 4]>,
}

/// `n` scaling lists (4:2:0: 6 of 4×4, then the 8×8 ones): VideoToolbox (macOS) applies them itself, so they are
/// stepped over (7.3.2.1.1.1); the Vulkan decoder takes none.
fn matrices(b: &mut Bits, n: usize) -> Result<()> {
    if cfg!(not(target_os = "macos")) {
        return Err(Bad("H.264 scaling matrices unsupported"));
    }
    for i in 0..n {
        if b.f() {
            let (mut last, mut next) = (8i32, 8i32);
            for _ in 0..if i < 6 { 16 } else { 64 } {
                if next != 0 {
                    // delta_scale is −128..=127; past the end (a truncated set) se() reads i32::MIN
                    let d = b.se();
                    if !(-128..=127).contains(&d) {
                        return Err(Bad("H.264 scaling list"));
                    }
                    next = (last + d + 256) % 256;
                }
                last = if next == 0 { last } else { next };
            }
        }
    }
    Ok(())
}

pub fn sps(n: &[u8]) -> Result<Sps> {
    let mut b = Bits::new(&n[1..]);
    let mut s = Sps {
        profile_idc: b.u(8) as u8,
        constraints: b.u(8) as u8,
        level_idc: b.u(8) as u8,
        ..Default::default()
    };
    // StdVideoH264LevelIdc 1.0 ..= 6.2; 9 is level 1b
    let l = [
        10, 11, 12, 13, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62,
    ]
    .iter()
    .position(|&l| l == s.level_idc.max(10));
    s.level = l.ok_or(Bad("H.264 level"))? as StdVideoH264LevelIdc;
    s.id = b.ue().min(32) as u8;
    if matches!(
        s.profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        if b.ue() != 1 {
            return Err(Bad("H.264 chroma format other than 4:2:0"));
        }
        if b.ue() != 0 || b.ue() != 0 {
            return Err(Bad("H.264 bit depth other than 8"));
        }
        s.qpprime_bypass = b.f();
        if b.f() {
            matrices(&mut b, 8)?;
        }
    }
    s.log2_max_frame_num = b.ue().min(12) + 4;
    s.poc_type = b.ue().min(3) as u8;
    match s.poc_type {
        0 => s.log2_max_poc_lsb = b.ue().min(12) + 4,
        2 => {}
        _ => return Err(Bad("H.264 POC type 1 unsupported")),
    }
    s.max_num_ref_frames = b.ue().min(99) as u8;
    if s.max_num_ref_frames as usize >= MAXS {
        return Err(Bad("too many reference frames"));
    }
    s.gaps = b.f();
    (s.width_mbs, s.height_mbs) = (b.ue().min(1023) + 1, b.ue().min(1023) + 1);
    if !b.f() {
        return Err(Bad("interlaced H.264 unsupported"));
    }
    s.direct_8x8 = b.f();
    if b.f() {
        let c = [b.ue(), b.ue(), b.ue(), b.ue()].map(u64::from);
        if 2 * (c[0] + c[1]) >= 16 * s.width_mbs as u64
            || 2 * (c[2] + c[3]) >= 16 * s.height_mbs as u64
        {
            return Err(Bad("H.264 cropping"));
        }
        s.crop = Some(c.map(|x| x as u32));
    }
    // the VUI follows; nothing in it changes decoding
    if b.end() {
        return Err(Bad("truncated SPS"));
    }
    Ok(s)
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Pps {
    pub id: u8,
    pub sps_id: u8,
    pub cabac: bool,
    pub bottom_poc: bool,
    pub ref_idx_default: [u8; 2],
    pub weighted_pred: bool,
    pub weighted_bipred: u8,
    pub qp_minus26: i8,
    pub qs_minus26: i8,
    pub chroma_qp: i8,
    pub deblock_ctl: bool,
    pub constrained_intra: bool,
    pub redundant: bool,
    pub t8x8: bool,
    pub chroma_qp2: i8,
}

pub fn pps(n: &[u8]) -> Result<Pps> {
    let mut b = Bits::new(&n[1..]);
    let id = b.ue();
    if id > 255 {
        return Err(Bad("PPS id"));
    }
    let mut p = Pps {
        id: id as u8,
        sps_id: b.ue().min(32) as u8,
        cabac: b.f(),
        bottom_poc: b.f(),
        ..Default::default()
    };
    if b.ue() != 0 {
        return Err(Bad("H.264 slice groups unsupported"));
    }
    p.ref_idx_default = [b.ue().min(31) as u8, b.ue().min(31) as u8];
    (p.weighted_pred, p.weighted_bipred) = (b.f(), b.u(2) as u8);
    (p.qp_minus26, p.qs_minus26, p.chroma_qp) = (b.se() as i8, b.se() as i8, b.se() as i8);
    (p.deblock_ctl, p.constrained_intra, p.redundant) = (b.f(), b.f(), b.f());
    p.chroma_qp2 = p.chroma_qp;
    if b.more() {
        p.t8x8 = b.f();
        if b.f() {
            matrices(&mut b, 6 + 2 * p.t8x8 as usize)?;
        }
        p.chroma_qp2 = b.se() as i8;
    }
    if b.end() {
        return Err(Bad("truncated PPS"));
    }
    Ok(p)
}

/// What picture management needs from a slice header.
#[derive(Clone, Copy, Default, Debug)]
pub struct Slice {
    pub idr: bool,
    pub nri: u8,
    pub intra: bool,
    pub pps_id: u8,
    pub frame_num: u32,
    pub idr_id: u16,
    pub poc_lsb: i32,
    pub poc_bottom: i32,
    pub long_ref: bool,
    pub adaptive: bool,
    /// memory_management_control_operation and its one or two values
    pub mmco: [(u8, u32, u32); 16],
    pub nm: usize,
}

/// Parameter sets by id, the POC / frame_num state of the previous pictures, whether a picture came since the start
/// or a restart, and the POC of the non-IDR I picture decoding started at.
pub struct State {
    pub sps: [Option<Sps>; 32],
    pub pps: Vec<Option<Pps>>,
    raw: Vec<Vec<u8>>, // the NAL unit stored under each id: SPS 0..32, PPS 32..288
    prev_msb: i32,
    prev_lsb: i32,
    prev_fn: u32,
    prev_fno: u32,
    prev_ref: u32,       // PrevRefFrameNum
    max_lt: Option<u32>, // MaxLongTermFrameIdx, None = "no long-term frame indices"
    started: bool,
    rp: Option<i32>,
}

impl Default for State {
    fn default() -> Self {
        let (sps, pps) = (Default::default(), vec![None; 256]);
        State {
            sps,
            pps,
            raw: vec![vec![]; 288],
            prev_msb: 0,
            prev_lsb: 0,
            prev_fn: 0,
            prev_fno: 0,
            prev_ref: 0,
            max_lt: None,
            started: false,
            rp: None,
        }
    }
}

impl State {
    /// Decoding restarts at the next picture (a seek): everything but the parameter sets starts over as in a new
    /// decoder; the caller empties the DPB.
    pub fn restart(&mut self) {
        (self.prev_msb, self.prev_lsb, self.prev_fn, self.prev_fno) = (0, 0, 0, 0);
        (self.prev_ref, self.max_lt, self.started, self.rp) = (0, None, false, None);
    }

    /// Stores an SPS / PPS under its id, replacing the one before (the last received is active); true when that
    /// changed what the id holds.
    pub fn put(&mut self, n: &[u8]) -> Result<bool> {
        let k = match nal_type(n) {
            7 => {
                let s = sps(n)?;
                let i = s.id as usize;
                *self.sps.get_mut(i).ok_or(Bad("SPS id"))? = Some(s);
                i
            }
            8 => {
                let p = pps(n)?;
                let i = p.id as usize;
                self.pps[i] = Some(p);
                32 + i
            }
            _ => return Ok(false),
        };
        if self.raw[k] == n {
            return Ok(false);
        }
        self.raw[k].clear();
        self.raw[k].extend_from_slice(n);
        Ok(true)
    }

    fn sets(&self, pps_id: u32) -> Result<(&Pps, &Sps)> {
        let p = self
            .pps
            .get(pps_id as usize)
            .and_then(Option::as_ref)
            .ok_or(Bad("missing PPS"))?;
        Ok((
            p,
            self.sps
                .get(p.sps_id as usize)
                .and_then(Option::as_ref)
                .ok_or(Bad("missing SPS"))?,
        ))
    }

    /// Every stored parameter set as received (SPS, then PPS, each by id): what VideoToolbox's format description is
    /// made from.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub(crate) fn raw(&self) -> impl Iterator<Item = &[u8]> {
        self.raw.iter().filter(|n| !n.is_empty()).map(Vec::as_slice)
    }

    pub fn slice(&self, n: &[u8]) -> Result<Slice> {
        let mut b = Bits::new(&n[1..]);
        let mut s = Slice {
            idr: nal_type(n) == 5,
            nri: n[0] >> 5 & 3,
            ..Default::default()
        };
        b.ue(); // first_mb_in_slice
        let st = b.ue() % 5; // P B I SP SI
        s.intra = st == 2 || st == 4;
        let pps_id = b.ue();
        let (pps, sps) = self.sets(pps_id)?;
        s.pps_id = pps_id as u8;
        s.frame_num = b.u(sps.log2_max_frame_num);
        if s.idr {
            s.idr_id = b.ue() as u16;
        }
        if sps.poc_type == 0 {
            s.poc_lsb = b.u(sps.log2_max_poc_lsb) as i32;
            if pps.bottom_poc {
                s.poc_bottom = b.se();
            }
        }
        if pps.redundant {
            b.ue();
        }
        if st == 1 {
            b.f(); // direct_spatial_mv_pred_flag
        }
        let mut nref = [pps.ref_idx_default[0] as u32, pps.ref_idx_default[1] as u32];
        let lists = [1, 2, 0, 1, 0][st as usize];
        if lists > 0 {
            if b.f() {
                nref[0] = b.ue();
                if st == 1 {
                    nref[1] = b.ue();
                }
            }
            if nref[0] > 31 || nref[1] > 31 {
                return Err(Bad("too many active references"));
            }
            // ref_pic_list_modification: the decoder parses it again from the bitstream
            for _ in 0..lists {
                if b.f() {
                    loop {
                        match b.ue() {
                            0..=2 => _ = b.ue(),
                            3 => break,
                            _ => return Err(Bad("ref_pic_list_modification")),
                        }
                    }
                }
            }
        }
        if pps.weighted_pred && (st == 0 || st == 3) || pps.weighted_bipred == 1 && st == 1 {
            b.ue(); // luma_log2_weight_denom
            b.ue(); // chroma_log2_weight_denom (4:2:0)
            for &n in &nref[..lists] {
                for _ in 0..=n {
                    if b.f() {
                        (0..2).for_each(|_| _ = b.se());
                    }
                    if b.f() {
                        (0..4).for_each(|_| _ = b.se());
                    }
                }
            }
        }
        if s.nri != 0 {
            if s.idr {
                b.f(); // no_output_of_prior_pics_flag
                s.long_ref = b.f();
            } else if b.f() {
                s.adaptive = true;
                loop {
                    let op = b.ue();
                    let m = match op {
                        0 => break,
                        1 | 2 | 4 | 6 => (op as u8, b.ue(), 0),
                        3 => (3, b.ue(), b.ue()),
                        5 => (5, 0, 0),
                        _ => return Err(Bad("memory_management_control_operation")),
                    };
                    *s.mmco.get_mut(s.nm).ok_or(Bad("too many MMCOs"))? = m;
                    s.nm += 1;
                }
            }
        }
        if b.end() {
            return Err(Bad("truncated slice header"));
        }
        Ok(s)
    }

    /// POC and reference marking for the picture in `au`: infers the frames of a frame_num gap, fills `p`, lists
    /// the reference slots active for its decode in `act`, then marks (sliding window or MMCO) and puts the picture
    /// in a free slot (`Pic::setup`). On an error nothing has changed, except for `Skip`: a picture shown before the
    /// non-IDR I picture decoding started or restarted at, marked but not decoded.
    pub(crate) fn picture(
        &mut self,
        au: &[&[u8]],
        refs: &mut [Ref],
        p: &mut StdVideoDecodeH264PictureInfo,
        act: &mut [u8; MAXS],
    ) -> Result<Pic> {
        let (mut first, mut intra) = (None, true);
        for n in au
            .iter()
            .filter(|n| n.len() > 1 && matches!(nal_type(n), 1 | 5))
        {
            let s = self.slice(n)?;
            intra &= s.intra;
            first.get_or_insert(s);
        }
        let s = first.ok_or(Bad("no slice"))?;
        let (pps, sps) = self.sets(s.pps_id as u32).map(|(p, s)| (*p, *s))?;
        let (fnum, refpic) = (s.frame_num, s.nri != 0);
        let setup = match refs.iter().position(|r| !r.live) {
            Some(i) => i,
            // an IDR references nothing and empties the DPB
            None if refpic && s.idr => 0,
            None => return Err(Bad("DPB full")),
        };
        let (max_fn, mx, w, h) = (
            1u32 << sps.log2_max_frame_num,
            (sps.max_num_ref_frames as usize).max(1),
            sps.width_mbs * 16,
            sps.height_mbs * 16,
        );
        // FrameNumWrap of frame_num `n` seen from the frame with frame_num `f`
        let wrap = |n: u32, f: u32| n as i64 - if n > f { max_fn as i64 } else { 0 };
        // 8.2.5.3 for the frame `f`: with `mx` references, the oldest short-term one goes; returns its slot
        let slide = |refs: &mut [Ref], f: u32| {
            if refs.iter().filter(|r| r.live).count() < mx {
                return None;
            }
            let i = (refs.iter().enumerate())
                .filter(|r| r.1.live && !r.1.long)
                .min_by_key(|r| wrap(r.1.num, f))?
                .0;
            refs[i].live = false;
            Some(i)
        };
        // 8.2.5.2: frame_num values skipped since PrevRefFrameNum. Without gaps allowed they are pictures lost on the
        // way (the spec has decoders infer that), handled the same: each becomes a "non-existing" short-term reference
        // through the sliding window, so the MMCO 1 meant for a lost picture still unmarks one (the camera drops its
        // one reference only that way: without this the reference before a loss stays for good). Only the last `mx`
        // can survive. One that pushes a picture out takes its slot, whose picture stands in for the missing one
        // (FFmpeg's concealment too); one in a free slot has none.
        if !s.idr && fnum != self.prev_ref {
            let n = fnum.wrapping_sub(self.prev_ref + 1) % max_fn;
            for u in (1..=n.min(mx as u32))
                .rev()
                .map(|d| fnum.wrapping_sub(d) % max_fn)
            {
                let i = match slide(refs, u) {
                    Some(i) => i,
                    None => {
                        let Some(j) = (0..refs.len()).find(|&j| !refs[j].live && j != setup) else {
                            continue;
                        };
                        refs[j].none = true;
                        j
                    }
                };
                (refs[i].live, refs[i].long, refs[i].num) = (true, false, u);
            }
            self.prev_ref = fnum.wrapping_sub(1) % max_fn;
        }
        // 8.2.1
        let (mut top, mut bot) = if sps.poc_type == 0 {
            if s.idr {
                (self.prev_msb, self.prev_lsb) = (0, 0);
            }
            let (max, lsb, pl, pm) = (
                1i32 << sps.log2_max_poc_lsb,
                s.poc_lsb,
                self.prev_lsb,
                self.prev_msb,
            );
            let msb = match () {
                _ if lsb < pl && pl - lsb >= max / 2 => pm + max,
                _ if lsb > pl && lsb - pl > max / 2 => pm - max,
                _ => pm,
            };
            if refpic {
                (self.prev_msb, self.prev_lsb) = (msb, lsb);
            }
            (msb + lsb, msb + lsb + s.poc_bottom)
        } else {
            let off = match () {
                _ if s.idr => 0,
                _ if self.prev_fn > fnum => self.prev_fno + max_fn,
                _ => self.prev_fno,
            };
            let t = if s.idr {
                0
            } else {
                2 * (off + fnum) as i32 - !refpic as i32
            };
            self.prev_fno = off;
            (t, t)
        };
        self.prev_fn = fnum;
        // decoding started at a non-IDR I picture, a recovery point (D.2.8): the pictures shown before it may
        // reference pictures before it, never decoded. They are skipped, but marked like any other (later frame_num
        // values and MMCOs count on them), a reference as a non-existing frame.
        if s.idr {
            self.rp = None;
        } else if !self.started && intra {
            self.rp = Some(top.min(bot));
        }
        self.started = true;
        let skip = self.rp.is_some_and(|r| top.min(bot) < r);
        p.flags.set_is_intra(intra as u32);
        p.flags.set_IdrPicFlag(s.idr as u32);
        p.flags.set_is_reference(refpic as u32);
        (p.seq_parameter_set_id, p.pic_parameter_set_id) = (sps.id, pps.id);
        (p.frame_num, p.idr_pic_id, p.PicOrderCnt) = (fnum as u16, s.idr_id, [top, bot]);
        let mut na = 0;
        for (i, _) in (refs.iter().enumerate()).filter(|r| r.1.live && !r.1.none && !s.idr) {
            (act[na], na) = (i as u8, na + 1);
        }
        // 8.2.5, for frames: PicNum = FrameNumWrap, LongTermPicNum = LongTermFrameIdx
        let (mut long, mut num, mut mmco5) = (false, fnum, false);
        let pic = |r: &Ref| wrap(r.num, fnum);
        if refpic && s.idr {
            refs.iter_mut().for_each(|r| r.live = false);
            (long, num, self.max_lt) = (s.long_ref, 0, s.long_ref.then_some(0));
        } else if refpic && s.adaptive {
            for &(op, a, c) in &s.mmco[..s.nm] {
                let x = fnum as i64 - (a as i64 + 1);
                match op {
                    1 => refs
                        .iter_mut()
                        .filter(|r| r.live && !r.long && pic(r) == x)
                        .for_each(|r| r.live = false),
                    2 => refs
                        .iter_mut()
                        .filter(|r| r.live && r.long && r.num == a)
                        .for_each(|r| r.live = false),
                    3 => {
                        let t = refs.iter().position(|r| r.live && !r.long && pic(r) == x);
                        refs.iter_mut()
                            .filter(|r| r.live && r.long && r.num == c)
                            .for_each(|r| r.live = false);
                        if let Some(t) = t {
                            (refs[t].long, refs[t].num) = (true, c);
                        }
                    }
                    4 => {
                        self.max_lt = a.checked_sub(1);
                        refs.iter_mut()
                            .filter(|r| r.live && r.long && self.max_lt.is_none_or(|m| r.num > m))
                            .for_each(|r| r.live = false);
                    }
                    5 => {
                        refs.iter_mut().for_each(|r| r.live = false);
                        (self.max_lt, mmco5) = (None, true);
                    }
                    _ => {
                        refs.iter_mut()
                            .filter(|r| r.live && r.long && r.num == a)
                            .for_each(|r| r.live = false);
                        (long, num) = (true, a);
                    }
                }
            }
        } else if refpic {
            // at most max_num_ref_frames references once the current one is added
            slide(refs, fnum);
        }
        if refpic {
            self.prev_ref = fnum;
        }
        if mmco5 {
            // after decoding, the picture counts as frame_num 0 with its POC rebased to 0
            let m = top.min(bot);
            (top, bot, num, self.prev_fn, self.prev_fno) = (top - m, bot - m, 0, 0, 0);
            (self.prev_msb, self.prev_lsb, self.prev_ref, self.rp) = (0, top, 0, None);
        }
        refs[setup] = Ref {
            live: refpic,
            long,
            num,
            poc: [top, bot],
            none: skip,
        };
        if skip {
            return Err(Skip);
        }
        Ok(Pic {
            setup,
            n: na,
            w,
            h,
            view: sps.view(),
            deep: false,
        })
    }
}

impl Sps {
    /// Display window (x, y, w, h) in the decoded picture: the cropping rectangle (4:2:0 frames: units of 2).
    pub fn view(&self) -> [u32; 4] {
        let [l, r, t, b] = self.crop.unwrap_or_default();
        [
            2 * l,
            2 * t,
            16 * self.width_mbs - 2 * (l + r),
            16 * self.height_mbs - 2 * (t + b),
        ]
    }

    pub fn std(&self) -> StdVideoH264SequenceParameterSet {
        let mut s: StdVideoH264SequenceParameterSet = unsafe { std::mem::zeroed() };
        let (f, c) = (&mut s.flags, self.constraints as u32);
        f.set_constraint_set0_flag(c >> 7 & 1);
        f.set_constraint_set1_flag(c >> 6 & 1);
        f.set_constraint_set2_flag(c >> 5 & 1);
        f.set_constraint_set3_flag(c >> 4 & 1);
        f.set_constraint_set4_flag(c >> 3 & 1);
        f.set_constraint_set5_flag(c >> 2 & 1);
        f.set_direct_8x8_inference_flag(self.direct_8x8 as u32);
        f.set_frame_mbs_only_flag(1);
        f.set_gaps_in_frame_num_value_allowed_flag(self.gaps as u32);
        f.set_qpprime_y_zero_transform_bypass_flag(self.qpprime_bypass as u32);
        f.set_frame_cropping_flag(self.crop.is_some() as u32);
        (s.profile_idc, s.level_idc) = (self.profile_idc as u32, self.level);
        s.chroma_format_idc = StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420;
        s.seq_parameter_set_id = self.id;
        s.log2_max_frame_num_minus4 = (self.log2_max_frame_num - 4) as u8;
        s.pic_order_cnt_type = self.poc_type as u32;
        s.log2_max_pic_order_cnt_lsb_minus4 = self.log2_max_poc_lsb.saturating_sub(4) as u8;
        s.max_num_ref_frames = self.max_num_ref_frames;
        (s.pic_width_in_mbs_minus1, s.pic_height_in_map_units_minus1) =
            (self.width_mbs - 1, self.height_mbs - 1);
        if let Some(c) = self.crop {
            [
                s.frame_crop_left_offset,
                s.frame_crop_right_offset,
                s.frame_crop_top_offset,
                s.frame_crop_bottom_offset,
            ] = c;
        }
        s
    }
}

impl Pps {
    pub fn std(&self) -> StdVideoH264PictureParameterSet {
        let mut p: StdVideoH264PictureParameterSet = unsafe { std::mem::zeroed() };
        let f = &mut p.flags;
        f.set_transform_8x8_mode_flag(self.t8x8 as u32);
        f.set_redundant_pic_cnt_present_flag(self.redundant as u32);
        f.set_constrained_intra_pred_flag(self.constrained_intra as u32);
        f.set_deblocking_filter_control_present_flag(self.deblock_ctl as u32);
        f.set_weighted_pred_flag(self.weighted_pred as u32);
        f.set_bottom_field_pic_order_in_frame_present_flag(self.bottom_poc as u32);
        f.set_entropy_coding_mode_flag(self.cabac as u32);
        (p.seq_parameter_set_id, p.pic_parameter_set_id) = (self.sps_id, self.id);
        [
            p.num_ref_idx_l0_default_active_minus1,
            p.num_ref_idx_l1_default_active_minus1,
        ] = self.ref_idx_default;
        p.weighted_bipred_idc = self.weighted_bipred as u32;
        (p.pic_init_qp_minus26, p.pic_init_qs_minus26) = (self.qp_minus26, self.qs_minus26);
        (p.chroma_qp_index_offset, p.second_chroma_qp_index_offset) =
            (self.chroma_qp, self.chroma_qp2);
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitw::W;

    /// SPS and PPS shaped like the X5's live preview: High 2656 × 1328, 16-bit frame_num and POC LSB, one reference
    /// frame, no gaps allowed.
    fn sets() -> [Vec<u8>; 2] {
        let mut s = W::default();
        s.u(8, 0x67).u(8, 100).u(8, 0).u(8, 50).ue(0);
        s.ue(1).ue(0).ue(0).f(false).f(false);
        s.ue(12).ue(0).ue(12).ue(1).f(false);
        s.ue(165).ue(82).f(true).f(true).f(false).f(false);
        let mut p = W::default();
        p.u(8, 0x68).ue(0).ue(0).f(true).f(false).ue(0);
        p.ue(0).ue(0).f(false).u(2, 0);
        p.ue(0).ue(0).ue(0).f(false).f(false).f(false);
        [s.nal(), p.nal()]
    }

    /// The camera's slice header: 0 an IDR, 1 a non-reference P picture, 2 a reference P picture whose MMCO 1
    /// unmarks the previous frame_num (its only way to drop the reference before it).
    fn slice(kind: u8, fnum: u32, poc: u32) -> Vec<u8> {
        let mut w = W::default();
        w.u(8, [0x65, 0x01, 0x61][kind as usize]).ue(0);
        w.ue(if kind == 0 { 7 } else { 5 }).ue(0).u(16, fnum);
        if kind == 0 {
            w.ue(0);
        }
        w.u(16, poc);
        if kind != 0 {
            w.f(false).f(false);
        }
        match kind {
            0 => _ = w.f(false).f(false),
            2 => _ = w.f(true).ue(1).ue(0).ue(0),
            _ => {}
        }
        w.nal()
    }

    /// The live preview as the link drops it when the decode thread falls behind: whole access units, two
    /// reference pictures among them in one GOP. The reference after each loss unmarks (MMCO 1) the lost
    /// picture's frame_num; without the frame_num gap's "non-existing" frame (8.2.5.2) standing in for it, the
    /// picture before the loss stays a reference for good and the second loss fills the 3 slots.
    #[test]
    fn loss() {
        let mut st = State::default();
        for n in sets() {
            st.put(&n).unwrap();
        }
        // max_num_ref_frames + 2, as Decoder::new sizes it
        let mut refs = vec![Ref::default(); 3];
        let (mut p, mut act) = (unsafe { std::mem::zeroed() }, [0; MAXS]);
        // GOPs of 30 as the camera sends them; lost: references 4 and 10, non-references 5 and 33
        for k in (0..90).filter(|k| ![4, 5, 10, 33].contains(k)) {
            let i = k % 30;
            let kind = [0, 1, 2][(i > 0) as usize + (i > 0 && i % 2 == 0) as usize];
            let n = slice(kind, (i as u32 + 1) / 2, 2 * i as u32);
            let pic = st.picture(&[&n], &mut refs, &mut p, &mut act);
            let pic = pic.unwrap_or_else(|e| panic!("access unit {k}: {e}"));
            let live = refs.iter().filter(|r| r.live).count();
            assert!(live <= 1, "access unit {k}: {live} references");
            // a P picture predicts from a slot with a picture in it: after a loss, the one before stands in
            assert_eq!(pic.n, (i > 0) as usize, "access unit {k}");
        }
    }

    /// An IDR decodes into a DPB whose slots are all references (a stream that broke the limits before): it
    /// references nothing and empties the DPB.
    #[test]
    fn idr_empties() {
        let mut st = State::default();
        for n in sets() {
            st.put(&n).unwrap();
        }
        let mut refs = vec![
            Ref {
                live: true,
                ..Default::default()
            };
            3
        ];
        let (mut p, mut act) = (unsafe { std::mem::zeroed() }, [0; MAXS]);
        let pic = st.picture(&[&slice(0, 0, 0)], &mut refs, &mut p, &mut act);
        let pic = pic.unwrap();
        assert_eq!((pic.n, refs.iter().filter(|r| r.live).count()), (0, 1));
    }
}
