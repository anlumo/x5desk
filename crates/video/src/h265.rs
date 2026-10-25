//! H.265 parameter sets and slice headers as Vulkan Video decode needs them (Main, 4:2:0 8-bit; PCM, tiles and
//! inter RPS prediction; no long-term references, scaling lists or range extensions), plus POC (8.3.1) and RPS
//! (8.3.2) picture management.
use crate::{
    Error::Bad,
    Result,
    bits::Bits,
    decode::{MAXS, Pic, Ref},
};
use ash::vk::native::*;

pub fn nal_type(n: &[u8]) -> u8 {
    n[0] >> 1 & 0x3f
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Ptl {
    pub tier: bool,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub level: StdVideoH265LevelIdc,
    pub progressive: bool,
    pub interlaced: bool,
    pub non_packed: bool,
    pub frame_only: bool,
}

fn ptl(b: &mut Bits, max_sub: u32) -> Result<Ptl> {
    b.u(2);
    let tier = b.f();
    let profile_idc = b.u(5) as u8;
    b.u(32);
    let (progressive, interlaced, non_packed, frame_only) = (b.f(), b.f(), b.f(), b.f());
    b.skip(44);
    let level_idc = b.u(8) as u8;
    let mut sub = [(false, false); 7];
    for s in &mut sub[..max_sub as usize] {
        *s = (b.f(), b.f());
    }
    if max_sub > 0 {
        b.skip(2 * (8 - max_sub));
    }
    for &(p, l) in &sub[..max_sub as usize] {
        b.skip(if p { 88 } else { 0 } + if l { 8 } else { 0 });
    }
    // StdVideoH265LevelIdc 1.0 ..= 6.2
    let level = [30, 60, 63, 90, 93, 120, 123, 150, 153, 156, 180, 183, 186]
        .iter()
        .position(|&l| l == level_idc);
    let level = level.ok_or(Bad("H.265 level"))? as StdVideoH265LevelIdc;
    Ok(Ptl {
        tier,
        profile_idc,
        level_idc,
        level,
        progressive,
        interlaced,
        non_packed,
        frame_only,
    })
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Dpbm {
    pub sub_layer_ordering: bool,
    pub max_dec_minus1: [u8; 7],
    pub reorder: [u8; 7],
    pub latency_plus1: [u32; 7],
}

fn dpbm(b: &mut Bits, max_sub: u32) -> Dpbm {
    let mut d = Dpbm {
        sub_layer_ordering: b.f(),
        ..Default::default()
    };
    let first = if d.sub_layer_ordering {
        0
    } else {
        max_sub as usize
    };
    for i in first..=max_sub as usize {
        (d.max_dec_minus1[i], d.reorder[i], d.latency_plus1[i]) =
            (b.ue() as u8, b.ue() as u8, b.ue());
    }
    for i in 0..first {
        (d.max_dec_minus1[i], d.reorder[i], d.latency_plus1[i]) = (
            d.max_dec_minus1[first],
            d.reorder[first],
            d.latency_plus1[first],
        );
    }
    d
}

fn hrd(b: &mut Bits, common: bool, max_sub: u32) {
    let (mut nal, mut vcl, mut sub_pic) = (false, false, false);
    if common {
        (nal, vcl) = (b.f(), b.f());
        if nal || vcl {
            sub_pic = b.f();
            b.skip(if sub_pic { 19 } else { 0 } + 8 + if sub_pic { 4 } else { 0 } + 15);
        }
    }
    for _ in 0..=max_sub {
        let general = b.f();
        let within = general || b.f();
        let low_delay = if within {
            b.ue();
            false
        } else {
            b.f()
        };
        let cpb = if low_delay { 0 } else { b.ue().min(31) };
        for _ in 0..nal as u32 + vcl as u32 {
            for _ in 0..=cpb {
                for _ in 0..2 + 2 * sub_pic as u32 {
                    b.ue();
                }
                b.f();
            }
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct Vps {
    pub id: u8,
    pub max_sub: u8,
    pub temporal_id_nesting: bool,
    pub ptl: Ptl,
    pub dpbm: Dpbm,
    pub timing: Option<(u32, u32, Option<u32>)>,
}

pub fn vps(n: &[u8]) -> Result<Vps> {
    let mut b = Bits::new(&n[2..]);
    let id = b.u(4) as u8;
    b.u(8);
    let max_sub = b.u(3).min(6);
    let temporal_id_nesting = b.f();
    b.u(16);
    let ptl = ptl(&mut b, max_sub)?;
    let dpbm = dpbm(&mut b, max_sub);
    let max_layer_id = b.u(6);
    let sets = b.ue().min(1024);
    b.skip(sets * (max_layer_id + 1));
    let timing = b.f().then(|| {
        let (n, s) = (b.u(32), b.u(32));
        (n, s, b.f().then(|| b.ue()))
    });
    // vps HRD parameters and extensions follow; Vulkan does not need them
    if b.end() {
        return Err(Bad("truncated VPS"));
    }
    Ok(Vps {
        id,
        max_sub: max_sub as u8,
        temporal_id_nesting,
        ptl,
        dpbm,
        timing,
    })
}

/// DeltaPocS0 or DeltaPocS1 with UsedByCurrPicS* as a bit mask.
#[derive(Clone, Copy, Default, Debug)]
pub struct L {
    pub n: usize,
    pub d: [i32; 16],
    pub used: u16,
}

impl L {
    fn push(&mut self, d: i32, u: bool) -> Result<()> {
        if self.n == 16 {
            return Err(Bad("RPS too large"));
        }
        (self.d[self.n], self.used) = (d, self.used | (u as u16) << self.n);
        self.n += 1;
        Ok(())
    }
    pub fn has(&self, d: i32) -> bool {
        self.d[..self.n].contains(&d)
    }
}

/// Short-term RPS: the syntax (for the Std struct) and the derived lists (for decoding).
#[derive(Clone, Copy, Default, Debug)]
pub struct StRps {
    pub inter: bool,
    pub idx_minus1: u32,
    pub sign: bool,
    pub abs_minus1: u32,
    pub use_delta: u32,
    pub used: u32,
    pub s: [L; 2],
    pub ref_n: u8, // NumDeltaPocs[RefRpsIdx] when inter
}

fn st_rps(b: &mut Bits, idx: usize, sets: &[StRps], num: usize) -> Result<StRps> {
    let mut r = StRps {
        inter: idx != 0 && b.f(),
        ..Default::default()
    };
    if !r.inter {
        let (neg, pos) = (b.ue(), b.ue());
        if neg > 16 || pos > 16 {
            return Err(Bad("RPS too large"));
        }
        for (l, n, sign) in [(0, neg, -1), (1, pos, 1)] {
            let mut p = 0;
            for _ in 0..n {
                p += sign * (b.ue() as i32 + 1);
                r.s[l].push(p, b.f())?;
            }
        }
        return Ok(r);
    }
    if idx == num {
        r.idx_minus1 = b.ue();
    }
    (r.sign, r.abs_minus1) = (b.f(), b.ue().min(1 << 15));
    let rf = sets
        .get(idx.wrapping_sub(r.idx_minus1 as usize + 1))
        .ok_or(Bad("RPS reference"))?;
    let d = (1 - 2 * r.sign as i32) * (r.abs_minus1 as i32 + 1);
    let (n0, n) = (rf.s[0].n, rf.s[0].n + rf.s[1].n);
    r.ref_n = n as u8;
    let (mut used, mut usd) = ([false; 33], [true; 33]);
    for j in 0..=n {
        used[j] = b.f();
        if used[j] {
            r.used |= 1 << j;
        } else {
            usd[j] = b.f();
        }
        r.use_delta |= (usd[j] as u32) << j;
    }
    // (7-61), (7-62)
    let (s0, s1) = (&rf.s[0], &rf.s[1]);
    for j in (0..s1.n).rev() {
        let p = s1.d[j] + d;
        if p < 0 && usd[n0 + j] {
            r.s[0].push(p, used[n0 + j])?;
        }
    }
    if d < 0 && usd[n] {
        r.s[0].push(d, used[n])?;
    }
    for j in 0..n0 {
        let p = s0.d[j] + d;
        if p < 0 && usd[j] {
            r.s[0].push(p, used[j])?;
        }
    }
    for j in (0..n0).rev() {
        let p = s0.d[j] + d;
        if p > 0 && usd[j] {
            r.s[1].push(p, used[j])?;
        }
    }
    if d > 0 && usd[n] {
        r.s[1].push(d, used[n])?;
    }
    for j in 0..s1.n {
        let p = s1.d[j] + d;
        if p > 0 && usd[n0 + j] {
            r.s[1].push(p, used[n0 + j])?;
        }
    }
    Ok(r)
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Vui {
    pub full_range: bool,
    pub colour: Option<(u8, u8, u8)>,
    pub timing: Option<(u32, u32)>,
    pub aspect: (u8, u16, u16),
    pub video_format: u8,
}

fn vui(b: &mut Bits, max_sub: u32) -> Vui {
    let mut v = Vui::default();
    if b.f() {
        v.aspect.0 = b.u(8) as u8;
        if v.aspect.0 == 255 {
            (v.aspect.1, v.aspect.2) = (b.u(16) as u16, b.u(16) as u16);
        }
    }
    if b.f() {
        b.f();
    }
    if b.f() {
        (v.video_format, v.full_range) = (b.u(3) as u8, b.f());
        if b.f() {
            v.colour = Some((b.u(8) as u8, b.u(8) as u8, b.u(8) as u8));
        }
    }
    if b.f() {
        b.ue();
        b.ue();
    }
    b.u(3); // neutral_chroma, field_seq, frame_field_info
    if b.f() {
        (0..4).for_each(|_| _ = b.ue());
    }
    if b.f() {
        v.timing = Some((b.u(32), b.u(32)));
        if b.f() {
            b.ue();
        }
        if b.f() {
            hrd(b, true, max_sub);
        }
    }
    if b.f() {
        b.u(3);
        (0..5).for_each(|_| _ = b.ue());
    }
    v
}

#[derive(Clone, Default, Debug)]
pub struct Sps {
    pub vps_id: u8,
    pub max_sub: u8,
    pub temporal_id_nesting: bool,
    pub ptl: Ptl,
    pub id: u8,
    pub width: u32,
    pub height: u32,
    pub conf_win: Option<[u32; 4]>,
    pub log2_max_poc_lsb: u32,
    pub dpbm: Dpbm,
    pub log2_min_cb_minus3: u8,
    pub log2_diff_max_min_cb: u8,
    pub log2_min_tb_minus2: u8,
    pub log2_diff_max_min_tb: u8,
    pub max_th_inter: u8,
    pub max_th_intra: u8,
    pub amp: bool,
    pub sao: bool,
    /// PCM sample bit depth luma / chroma minus 1, log2 min size minus 3, log2 size difference, loop filter off
    pub pcm: Option<(u8, u8, u8, u8, bool)>,
    pub st: Vec<StRps>,
    pub temporal_mvp: bool,
    pub strong_intra: bool,
    pub vui: Option<Vui>,
}

pub fn sps(n: &[u8]) -> Result<Sps> {
    let mut b = Bits::new(&n[2..]);
    let mut s = Sps {
        vps_id: b.u(4) as u8,
        max_sub: b.u(3).min(6) as u8,
        temporal_id_nesting: b.f(),
        ..Default::default()
    };
    s.ptl = ptl(&mut b, s.max_sub as u32)?;
    s.id = b.ue().min(16) as u8;
    if b.ue() != 1 {
        return Err(Bad("H.265 chroma format other than 4:2:0"));
    }
    (s.width, s.height) = (b.ue(), b.ue());
    if b.f() {
        let c = [b.ue(), b.ue(), b.ue(), b.ue()].map(u64::from);
        if 2 * (c[0] + c[1]) >= s.width as u64 || 2 * (c[2] + c[3]) >= s.height as u64 {
            return Err(Bad("H.265 conformance window"));
        }
        s.conf_win = Some(c.map(|x| x as u32));
    }
    if b.ue() != 0 || b.ue() != 0 {
        return Err(Bad("H.265 bit depth other than 8"));
    }
    s.log2_max_poc_lsb = b.ue().min(12) + 4;
    s.dpbm = dpbm(&mut b, s.max_sub as u32);
    s.log2_min_cb_minus3 = b.ue() as u8;
    s.log2_diff_max_min_cb = b.ue() as u8;
    s.log2_min_tb_minus2 = b.ue() as u8;
    s.log2_diff_max_min_tb = b.ue() as u8;
    s.max_th_inter = b.ue() as u8;
    s.max_th_intra = b.ue() as u8;
    if b.f() {
        return Err(Bad("H.265 scaling lists unsupported"));
    }
    (s.amp, s.sao) = (b.f(), b.f());
    if b.f() {
        s.pcm = Some((
            b.u(4) as u8,
            b.u(4) as u8,
            b.ue() as u8,
            b.ue() as u8,
            b.f(),
        ));
    }
    let n = b.ue() as usize;
    if n > 64 {
        return Err(Bad("too many short-term RPS"));
    }
    for i in 0..n {
        let r = st_rps(&mut b, i, &s.st, n)?;
        s.st.push(r);
    }
    if b.f() {
        return Err(Bad("H.265 long-term references unsupported"));
    }
    (s.temporal_mvp, s.strong_intra) = (b.f(), b.f());
    if b.f() {
        s.vui = Some(vui(&mut b, s.max_sub as u32));
    }
    if b.f() {
        return Err(Bad("H.265 SPS extensions unsupported"));
    }
    if b.end() {
        return Err(Bad("truncated SPS"));
    }
    Ok(s)
}

#[derive(Clone, Default, Debug)]
pub struct Pps {
    pub id: u8,
    pub sps_id: u8,
    pub dependent_slices: bool,
    pub output_flag_present: bool,
    pub extra_bits: u8,
    pub sign_data_hiding: bool,
    pub cabac_init_present: bool,
    pub ref_idx_default: [u8; 2],
    pub init_qp_minus26: i8,
    pub constrained_intra: bool,
    pub transform_skip: bool,
    pub cu_qp_delta: Option<u8>,
    pub cb_qp: i8,
    pub cr_qp: i8,
    pub slice_chroma_qp: bool,
    pub weighted_pred: bool,
    pub weighted_bipred: bool,
    pub transquant_bypass: bool,
    /// columns minus 1, rows minus 1, uniform, column widths minus 1, row heights minus 1, loop filter across tiles
    pub tiles: Option<(u8, u8, bool, [u16; 19], [u16; 21], bool)>,
    pub entropy_sync: bool,
    pub lf_across_slices: bool,
    /// override enabled, disabled, beta / 2, tc / 2
    pub deblock: Option<(bool, bool, i8, i8)>,
    pub lists_mod: bool,
    pub log2_parallel_merge_minus2: u8,
    pub slice_ext: bool,
}

pub fn pps(n: &[u8]) -> Result<Pps> {
    let mut b = Bits::new(&n[2..]);
    let mut p = Pps {
        id: b.ue().min(64) as u8,
        sps_id: b.ue().min(16) as u8,
        ..Default::default()
    };
    (p.dependent_slices, p.output_flag_present, p.extra_bits) = (b.f(), b.f(), b.u(3) as u8);
    (p.sign_data_hiding, p.cabac_init_present) = (b.f(), b.f());
    p.ref_idx_default = [b.ue().min(14) as u8, b.ue().min(14) as u8];
    p.init_qp_minus26 = b.se() as i8;
    (p.constrained_intra, p.transform_skip) = (b.f(), b.f());
    if b.f() {
        p.cu_qp_delta = Some(b.ue() as u8);
    }
    (p.cb_qp, p.cr_qp) = (b.se() as i8, b.se() as i8);
    (
        p.slice_chroma_qp,
        p.weighted_pred,
        p.weighted_bipred,
        p.transquant_bypass,
    ) = (b.f(), b.f(), b.f(), b.f());
    let tiles = b.f();
    p.entropy_sync = b.f();
    if tiles {
        let (c, r) = (b.ue(), b.ue());
        if c > 18 || r > 20 {
            return Err(Bad("too many tiles"));
        }
        let (uniform, mut cw, mut rh) = (b.f(), [0; 19], [0; 21]);
        if !uniform {
            cw[..c as usize].iter_mut().for_each(|w| *w = b.ue() as u16);
            rh[..r as usize].iter_mut().for_each(|h| *h = b.ue() as u16);
        }
        p.tiles = Some((c as u8, r as u8, uniform, cw, rh, b.f()));
    }
    p.lf_across_slices = b.f();
    if b.f() {
        let (ov, dis) = (b.f(), b.f());
        let (beta, tc) = if dis {
            (0, 0)
        } else {
            (b.se() as i8, b.se() as i8)
        };
        p.deblock = Some((ov, dis, beta, tc));
    }
    if b.f() {
        return Err(Bad("H.265 scaling lists unsupported"));
    }
    (p.lists_mod, p.log2_parallel_merge_minus2, p.slice_ext) = (b.f(), b.ue() as u8, b.f());
    if b.f() {
        return Err(Bad("H.265 PPS extensions unsupported"));
    }
    if b.end() {
        return Err(Bad("truncated PPS"));
    }
    Ok(p)
}

/// The first slice segment header of a picture, up to the short-term RPS.
#[derive(Debug, Default)]
pub struct Slice {
    pub nal_type: u8,
    pub tid: u8,
    pub pps_id: u8,
    pub slice_type: u32,
    pub poc_lsb: i32,
    pub rps: StRps, // empty for IDR
    pub rps_sps: bool,
    pub rps_bits: u16,
}

/// Parameter sets by id, the POC of prevTid0Pic, and whether a picture came since the start or an end of sequence.
pub struct State {
    pub vps: [Option<Vps>; 16],
    pub sps: [Option<Sps>; 16],
    pub pps: Vec<Option<Pps>>,
    raw: Vec<Vec<u8>>, // the NAL unit stored under each id: VPS 0..16, SPS 16..32, PPS 32..96
    prev: i32,
    started: bool,
}

impl Default for State {
    fn default() -> Self {
        State {
            vps: Default::default(),
            sps: Default::default(),
            pps: vec![None; 64],
            raw: vec![vec![]; 96],
            prev: 0,
            started: false,
        }
    }
}

impl State {
    /// Stores a VPS / SPS / PPS under its id, replacing the one before (the last received is active); true when
    /// that changed what the id holds.
    pub fn put(&mut self, n: &[u8]) -> Result<bool> {
        let k = match nal_type(n) {
            32 => {
                let v = vps(n)?;
                let i = v.id as usize;
                self.vps[i] = Some(v);
                i
            }
            33 => {
                let s = sps(n)?;
                let i = s.id as usize;
                *self.sps.get_mut(i).ok_or(Bad("SPS id"))? = Some(s);
                16 + i
            }
            34 => {
                let p = pps(n)?;
                let i = p.id as usize;
                *self.pps.get_mut(i).ok_or(Bad("PPS id"))? = Some(p);
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

    pub fn slice(&self, n: &[u8]) -> Result<Slice> {
        let t = nal_type(n);
        let mut b = Bits::new(&n[2..]);
        if !b.f() {
            return Err(Bad("not the first slice segment"));
        }
        if (16..=23).contains(&t) {
            b.f();
        }
        let pps_id = b.ue();
        let p = self
            .pps
            .get(pps_id as usize)
            .and_then(Option::as_ref)
            .ok_or(Bad("missing PPS"))?;
        let s = self
            .sps
            .get(p.sps_id as usize)
            .and_then(Option::as_ref)
            .ok_or(Bad("missing SPS"))?;
        b.skip(p.extra_bits as u32);
        let mut sl = Slice {
            nal_type: t,
            tid: (n[1] & 7).saturating_sub(1),
            pps_id: pps_id as u8,
            slice_type: b.ue(),
            ..Default::default()
        };
        if p.output_flag_present {
            b.f();
        }
        if t != 19 && t != 20 {
            sl.poc_lsb = b.u(s.log2_max_poc_lsb) as i32;
            sl.rps_sps = b.f();
            let n = s.st.len();
            if !sl.rps_sps {
                let p0 = b.p;
                sl.rps = st_rps(&mut b, n, &s.st, n)?;
                sl.rps_bits = (b.p - p0) as u16;
            } else {
                let k = if n > 1 {
                    32 - (n as u32 - 1).leading_zeros()
                } else {
                    0
                };
                sl.rps = *s.st.get(b.u(k) as usize).ok_or(Bad("RPS index"))?;
            }
        }
        if b.end() {
            return Err(Bad("truncated slice header"));
        }
        Ok(sl)
    }

    /// POC and RPS for the picture in `au`: drops the DPB pictures its RPS does not list, fills `p`, lists the
    /// active reference slots in `act`, and puts the current picture in a free slot (`Pic::setup`). On an error
    /// nothing has changed.
    pub(crate) fn picture(
        &mut self,
        au: &[&[u8]],
        refs: &mut [Ref],
        p: &mut StdVideoDecodeH265PictureInfo,
        act: &mut [u8; MAXS],
    ) -> Result<Pic> {
        let n = au
            .iter()
            .find(|n| n.len() > 2 && nal_type(n) < 32 && n[2] & 0x80 != 0)
            .ok_or(Bad("no first slice segment"))?;
        let sl = self.slice(n)?;
        let t = sl.nal_type;
        let pps = self.pps[sl.pps_id as usize]
            .as_ref()
            .ok_or(Bad("missing PPS"))?;
        let sps = self
            .sps
            .get(pps.sps_id as usize)
            .and_then(Option::as_ref)
            .ok_or(Bad("missing SPS"))?;
        // NoRaslOutputFlag: IDR, BLA, and a CRA first in the stream or after an end of sequence start a new POC
        // sequence and empty the DPB (RASL pictures after such a CRA fail with "missing reference")
        let restart = (16..=20).contains(&t) || t == 21 && !self.started;
        let max = 1 << sps.log2_max_poc_lsb;
        let (lsb, pl, pm) = (sl.poc_lsb, self.prev & (max - 1), self.prev & !(max - 1));
        let poc = lsb
            + match () {
                _ if restart => 0,
                _ if lsb < pl && pl - lsb >= max / 2 => pm + max,
                _ if lsb > pl && lsb - pl > max / 2 => pm - max,
                _ => pm,
            };
        let rps = &sl.rps;
        let keep = |r: &Ref| {
            r.live && !restart && (rps.s[0].has(r.poc[0] - poc) || rps.s[1].has(r.poc[0] - poc))
        };
        p.RefPicSetStCurrBefore = [0xff; 8];
        p.RefPicSetStCurrAfter = [0xff; 8];
        p.RefPicSetLtCurr = [0xff; 8];
        for (l, out) in [
            (&rps.s[0], &mut p.RefPicSetStCurrBefore),
            (&rps.s[1], &mut p.RefPicSetStCurrAfter),
        ] {
            for (j, i) in (0..l.n).filter(|i| l.used >> i & 1 == 1).enumerate() {
                let slot = refs
                    .iter()
                    .position(|r| keep(r) && r.poc[0] == poc + l.d[i])
                    .ok_or(Bad("missing reference"))?;
                *out.get_mut(j).ok_or(Bad("too many references"))? = slot as u8;
            }
        }
        let setup = refs.iter().position(|r| !keep(r)).ok_or(Bad("DPB full"))?;
        // prevTid0Pic: TemporalId 0 and not RADL, RASL or a sub-layer non-reference picture
        if sl.tid == 0 && !(6..=9).contains(&t) && !(t <= 14 && t % 2 == 0) {
            self.prev = poc;
        }
        self.started = !au.iter().any(|n| n.len() > 1 && nal_type(n) == 36);
        refs.iter_mut().for_each(|r| r.live = keep(r));
        p.flags.set_IrapPicFlag((16..=23).contains(&t) as u32);
        p.flags.set_IdrPicFlag((t == 19 || t == 20) as u32);
        p.flags
            .set_IsReference((t <= 14 && t % 2 == 1 || (16..=23).contains(&t)) as u32);
        p.flags
            .set_short_term_ref_pic_set_sps_flag(sl.rps_sps as u32);
        p.sps_video_parameter_set_id = sps.vps_id;
        p.pps_seq_parameter_set_id = sps.id;
        p.pps_pic_parameter_set_id = pps.id;
        p.NumDeltaPocsOfRefRpsIdx = if rps.inter && !sl.rps_sps {
            rps.ref_n
        } else {
            0
        };
        p.PicOrderCntVal = poc;
        p.NumBitsForSTRefPicSetInSlice = sl.rps_bits;
        let mut na = 0;
        for (i, _) in refs.iter().enumerate().filter(|r| r.1.live) {
            (act[na], na) = (i as u8, na + 1);
        }
        // every decoded picture is a short-term reference until an RPS leaves it out (8.3.2): a sub-layer
        // non-reference picture can still be referenced from a higher sub-layer
        refs[setup] = Ref {
            live: true,
            long: false,
            num: 0,
            poc: [poc; 2],
            none: false,
        };
        Ok(Pic {
            setup,
            n: na,
            w: sps.width,
            h: sps.height,
            view: sps.view(),
        })
    }
}

// ---- Std structs; the boxes keep the pointers inside valid while the holder moves ----

fn std_ptl(p: &Ptl) -> Box<StdVideoH265ProfileTierLevel> {
    let mut f: StdVideoH265ProfileTierLevelFlags = unsafe { std::mem::zeroed() };
    f.set_general_tier_flag(p.tier as u32);
    f.set_general_progressive_source_flag(p.progressive as u32);
    f.set_general_interlaced_source_flag(p.interlaced as u32);
    f.set_general_non_packed_constraint_flag(p.non_packed as u32);
    f.set_general_frame_only_constraint_flag(p.frame_only as u32);
    Box::new(StdVideoH265ProfileTierLevel {
        flags: f,
        general_profile_idc: p.profile_idc as u32,
        general_level_idc: p.level,
    })
}

fn std_dpbm(d: &Dpbm) -> Box<StdVideoH265DecPicBufMgr> {
    Box::new(StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1: d.latency_plus1,
        max_dec_pic_buffering_minus1: d.max_dec_minus1,
        max_num_reorder_pics: d.reorder,
    })
}

pub struct StdVps {
    pub raw: StdVideoH265VideoParameterSet,
    _ptl: Box<StdVideoH265ProfileTierLevel>,
    _dpbm: Box<StdVideoH265DecPicBufMgr>,
}

impl Vps {
    pub fn std(&self) -> StdVps {
        let (ptl, dpbm) = (std_ptl(&self.ptl), std_dpbm(&self.dpbm));
        let mut v: StdVideoH265VideoParameterSet = unsafe { std::mem::zeroed() };
        v.flags
            .set_vps_temporal_id_nesting_flag(self.temporal_id_nesting as u32);
        v.flags
            .set_vps_sub_layer_ordering_info_present_flag(self.dpbm.sub_layer_ordering as u32);
        v.vps_video_parameter_set_id = self.id;
        v.vps_max_sub_layers_minus1 = self.max_sub;
        if let Some((n, s, poc)) = self.timing {
            v.flags.set_vps_timing_info_present_flag(1);
            (v.vps_num_units_in_tick, v.vps_time_scale) = (n, s);
            if let Some(p) = poc {
                v.flags.set_vps_poc_proportional_to_timing_flag(1);
                v.vps_num_ticks_poc_diff_one_minus1 = p;
            }
        }
        (v.pDecPicBufMgr, v.pProfileTierLevel) = (&*dpbm, &*ptl);
        StdVps {
            raw: v,
            _ptl: ptl,
            _dpbm: dpbm,
        }
    }
}

fn std_st_rps(r: &StRps) -> StdVideoH265ShortTermRefPicSet {
    let mut s: StdVideoH265ShortTermRefPicSet = unsafe { std::mem::zeroed() };
    s.flags
        .set_inter_ref_pic_set_prediction_flag(r.inter as u32);
    s.flags.set_delta_rps_sign(r.sign as u32);
    s.delta_idx_minus1 = r.idx_minus1;
    s.use_delta_flag = r.use_delta as u16;
    s.abs_delta_rps_minus1 = r.abs_minus1 as u16;
    s.used_by_curr_pic_flag = r.used as u16;
    (s.num_negative_pics, s.num_positive_pics) = (r.s[0].n as u8, r.s[1].n as u8);
    (s.used_by_curr_pic_s0_flag, s.used_by_curr_pic_s1_flag) = (r.s[0].used, r.s[1].used);
    for (l, out) in [
        (&r.s[0], &mut s.delta_poc_s0_minus1),
        (&r.s[1], &mut s.delta_poc_s1_minus1),
    ] {
        let mut prev = 0;
        for (o, &d) in out.iter_mut().zip(&l.d[..l.n]) {
            (*o, prev) = (((d - prev).abs() - 1) as u16, d);
        }
    }
    s
}

pub struct StdSps {
    pub raw: StdVideoH265SequenceParameterSet,
    _ptl: Box<StdVideoH265ProfileTierLevel>,
    _dpbm: Box<StdVideoH265DecPicBufMgr>,
    _st: Vec<StdVideoH265ShortTermRefPicSet>,
    _vui: Option<Box<StdVideoH265SequenceParameterSetVui>>,
}

impl Sps {
    /// Display window (x, y, w, h) in the decoded picture: the conformance window (4:2:0: units of 2).
    pub fn view(&self) -> [u32; 4] {
        let [l, r, t, b] = self.conf_win.unwrap_or_default();
        [
            2 * l,
            2 * t,
            self.width - 2 * (l + r),
            self.height - 2 * (t + b),
        ]
    }

    pub fn std(&self) -> StdSps {
        let (ptl, dpbm) = (std_ptl(&self.ptl), std_dpbm(&self.dpbm));
        let st: Vec<_> = self.st.iter().map(std_st_rps).collect();
        let mut s: StdVideoH265SequenceParameterSet = unsafe { std::mem::zeroed() };
        let f = &mut s.flags;
        f.set_sps_temporal_id_nesting_flag(self.temporal_id_nesting as u32);
        f.set_conformance_window_flag(self.conf_win.is_some() as u32);
        f.set_sps_sub_layer_ordering_info_present_flag(self.dpbm.sub_layer_ordering as u32);
        f.set_amp_enabled_flag(self.amp as u32);
        f.set_sample_adaptive_offset_enabled_flag(self.sao as u32);
        f.set_sps_temporal_mvp_enabled_flag(self.temporal_mvp as u32);
        f.set_strong_intra_smoothing_enabled_flag(self.strong_intra as u32);
        if let Some((l, c, mn, d, lf)) = self.pcm {
            f.set_pcm_enabled_flag(1);
            f.set_pcm_loop_filter_disabled_flag(lf as u32);
            (
                s.pcm_sample_bit_depth_luma_minus1,
                s.pcm_sample_bit_depth_chroma_minus1,
            ) = (l, c);
            (
                s.log2_min_pcm_luma_coding_block_size_minus3,
                s.log2_diff_max_min_pcm_luma_coding_block_size,
            ) = (mn, d);
        }
        s.chroma_format_idc = StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420;
        (s.pic_width_in_luma_samples, s.pic_height_in_luma_samples) = (self.width, self.height);
        s.sps_video_parameter_set_id = self.vps_id;
        s.sps_max_sub_layers_minus1 = self.max_sub;
        s.sps_seq_parameter_set_id = self.id;
        s.log2_max_pic_order_cnt_lsb_minus4 = (self.log2_max_poc_lsb - 4) as u8;
        s.log2_min_luma_coding_block_size_minus3 = self.log2_min_cb_minus3;
        s.log2_diff_max_min_luma_coding_block_size = self.log2_diff_max_min_cb;
        s.log2_min_luma_transform_block_size_minus2 = self.log2_min_tb_minus2;
        s.log2_diff_max_min_luma_transform_block_size = self.log2_diff_max_min_tb;
        s.max_transform_hierarchy_depth_inter = self.max_th_inter;
        s.max_transform_hierarchy_depth_intra = self.max_th_intra;
        s.num_short_term_ref_pic_sets = st.len() as u8;
        if let Some(c) = self.conf_win {
            [
                s.conf_win_left_offset,
                s.conf_win_right_offset,
                s.conf_win_top_offset,
                s.conf_win_bottom_offset,
            ] = c;
        }
        (s.pProfileTierLevel, s.pDecPicBufMgr) = (&*ptl, &*dpbm);
        s.pShortTermRefPicSet = if st.is_empty() {
            std::ptr::null()
        } else {
            st.as_ptr()
        };
        // video signal, colour, aspect and timing only: no HRD
        let vui = self.vui.map(|v| {
            let mut u: StdVideoH265SequenceParameterSetVui = unsafe { std::mem::zeroed() };
            u.flags.set_video_signal_type_present_flag(1);
            u.flags.set_video_full_range_flag(v.full_range as u32);
            u.video_format = v.video_format;
            if let Some((p, t, m)) = v.colour {
                u.flags.set_colour_description_present_flag(1);
                (
                    u.colour_primaries,
                    u.transfer_characteristics,
                    u.matrix_coeffs,
                ) = (p, t, m);
            }
            if v.aspect.0 != 0 {
                u.flags.set_aspect_ratio_info_present_flag(1);
                (u.aspect_ratio_idc, u.sar_width, u.sar_height) =
                    (v.aspect.0 as u32, v.aspect.1, v.aspect.2);
            }
            if let Some((n, t)) = v.timing {
                u.flags.set_vui_timing_info_present_flag(1);
                (u.vui_num_units_in_tick, u.vui_time_scale) = (n, t);
            }
            Box::new(u)
        });
        if let Some(v) = &vui {
            s.flags.set_vui_parameters_present_flag(1);
            s.pSequenceParameterSetVui = &**v;
        }
        StdSps {
            raw: s,
            _ptl: ptl,
            _dpbm: dpbm,
            _st: st,
            _vui: vui,
        }
    }
}

impl Pps {
    pub fn std(&self, vps_id: u8) -> StdVideoH265PictureParameterSet {
        let mut p: StdVideoH265PictureParameterSet = unsafe { std::mem::zeroed() };
        let f = &mut p.flags;
        f.set_dependent_slice_segments_enabled_flag(self.dependent_slices as u32);
        f.set_output_flag_present_flag(self.output_flag_present as u32);
        f.set_sign_data_hiding_enabled_flag(self.sign_data_hiding as u32);
        f.set_cabac_init_present_flag(self.cabac_init_present as u32);
        f.set_constrained_intra_pred_flag(self.constrained_intra as u32);
        f.set_transform_skip_enabled_flag(self.transform_skip as u32);
        f.set_cu_qp_delta_enabled_flag(self.cu_qp_delta.is_some() as u32);
        f.set_pps_slice_chroma_qp_offsets_present_flag(self.slice_chroma_qp as u32);
        f.set_weighted_pred_flag(self.weighted_pred as u32);
        f.set_weighted_bipred_flag(self.weighted_bipred as u32);
        f.set_transquant_bypass_enabled_flag(self.transquant_bypass as u32);
        f.set_tiles_enabled_flag(self.tiles.is_some() as u32);
        f.set_entropy_coding_sync_enabled_flag(self.entropy_sync as u32);
        f.set_pps_loop_filter_across_slices_enabled_flag(self.lf_across_slices as u32);
        f.set_lists_modification_present_flag(self.lists_mod as u32);
        f.set_slice_segment_header_extension_present_flag(self.slice_ext as u32);
        if let Some((ov, dis, beta, tc)) = self.deblock {
            f.set_deblocking_filter_control_present_flag(1);
            f.set_deblocking_filter_override_enabled_flag(ov as u32);
            f.set_pps_deblocking_filter_disabled_flag(dis as u32);
            (p.pps_beta_offset_div2, p.pps_tc_offset_div2) = (beta, tc);
        }
        if let Some((c, r, uni, cw, rh, lf)) = self.tiles {
            f.set_uniform_spacing_flag(uni as u32);
            f.set_loop_filter_across_tiles_enabled_flag(lf as u32);
            (
                p.num_tile_columns_minus1,
                p.num_tile_rows_minus1,
                p.column_width_minus1,
                p.row_height_minus1,
            ) = (c, r, cw, rh);
        }
        p.pps_pic_parameter_set_id = self.id;
        p.pps_seq_parameter_set_id = self.sps_id;
        p.sps_video_parameter_set_id = vps_id;
        p.num_extra_slice_header_bits = self.extra_bits;
        [
            p.num_ref_idx_l0_default_active_minus1,
            p.num_ref_idx_l1_default_active_minus1,
        ] = self.ref_idx_default;
        p.init_qp_minus26 = self.init_qp_minus26;
        p.diff_cu_qp_delta_depth = self.cu_qp_delta.unwrap_or(0);
        (p.pps_cb_qp_offset, p.pps_cr_qp_offset) = (self.cb_qp, self.cr_qp);
        p.log2_parallel_merge_level_minus2 = self.log2_parallel_merge_minus2;
        p
    }
}
