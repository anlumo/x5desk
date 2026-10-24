//! H.264 parameter sets and slice headers as Vulkan Video decode needs them (progressive 4:2:0 8-bit up to High,
//! POC types 0 and 2, no scaling matrices), plus POC (8.2.1) and reference marking (8.2.5: sliding window and
//! MMCO 1-6).
use crate::{
    Error::Bad,
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
            return Err(Bad("H.264 scaling matrices unsupported"));
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
            return Err(Bad("H.264 scaling matrices unsupported"));
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

/// Parameter sets by id and the POC / frame_num state of the previous pictures.
pub struct State {
    pub sps: [Option<Sps>; 32],
    pub pps: Vec<Option<Pps>>,
    raw: Vec<Vec<u8>>, // the NAL unit stored under each id: SPS 0..32, PPS 32..288
    prev_msb: i32,
    prev_lsb: i32,
    prev_fn: u32,
    prev_fno: u32,
    max_lt: Option<u32>, // MaxLongTermFrameIdx, None = "no long-term frame indices"
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
            max_lt: None,
        }
    }
}

impl State {
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

    /// POC and reference marking for the picture in `au`: fills `p`, lists the reference slots active for its
    /// decode in `act`, then marks (sliding window or MMCO) and puts the picture in a free slot (`Pic::setup`). On
    /// an error nothing has changed.
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
        let setup = refs.iter().position(|r| !r.live).ok_or(Bad("DPB full"))?;
        let (max_fn, w, h) = (
            1u32 << sps.log2_max_frame_num,
            sps.width_mbs * 16,
            sps.height_mbs * 16,
        );
        let (fnum, refpic) = (s.frame_num, s.nri != 0);
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
        p.flags.set_is_intra(intra as u32);
        p.flags.set_IdrPicFlag(s.idr as u32);
        p.flags.set_is_reference(refpic as u32);
        (p.seq_parameter_set_id, p.pic_parameter_set_id) = (sps.id, pps.id);
        (p.frame_num, p.idr_pic_id, p.PicOrderCnt) = (fnum as u16, s.idr_id, [top, bot]);
        let mut na = 0;
        for (i, _) in refs.iter().enumerate().filter(|r| r.1.live) {
            (act[na], na) = (i as u8, na + 1);
        }
        // 8.2.5, for frames: PicNum = FrameNumWrap, LongTermPicNum = LongTermFrameIdx
        let (mut long, mut num, mut mmco5) = (false, fnum, false);
        let pic = |r: &Ref| {
            if r.num > fnum {
                r.num as i64 - max_fn as i64
            } else {
                r.num as i64
            }
        };
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
            // sliding window: at most max_num_ref_frames references once the current one is added
            let full =
                refs.iter().filter(|r| r.live).count() >= (sps.max_num_ref_frames as usize).max(1);
            let old = refs
                .iter()
                .enumerate()
                .filter(|r| r.1.live && !r.1.long)
                .min_by_key(|r| pic(r.1))
                .map(|r| r.0);
            if let (true, Some(i)) = (full, old) {
                refs[i].live = false;
            }
        }
        if mmco5 {
            // after decoding, the picture counts as frame_num 0 with its POC rebased to 0
            let m = top.min(bot);
            (top, bot, num, self.prev_fn, self.prev_fno) = (top - m, bot - m, 0, 0, 0);
            (self.prev_msb, self.prev_lsb) = (0, top);
        }
        refs[setup] = Ref {
            live: refpic,
            long,
            num,
            poc: [top, bot],
        };
        Ok(Pic {
            setup,
            n: na,
            w,
            h,
            view: sps.view(),
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
