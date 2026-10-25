//! The camera's capture-parameter JSON (`/CONF`, PROTOCOL.md §8.10, `analysis/app_features.md` §3–4): per lens block
//! (`Sphere06C` 360, `Fisheyes06C` single lens) its modes, per mode its settings, each with the values allowed given
//! the values of the settings it depends on (`depend_on` joined by `|`, else `default`). And how a setting reaches
//! the camera: `SET_PHOTOGRAPHY_OPTIONS` with the mode's function mode, or `SET_OPTIONS`. The app maps the JSON's
//! strings to protobuf values through an asset of its own (`proto.json`); here they come from the enum names in
//! `proto/`, plus the few the names do not give. The writes in `capture::CHECKED` ran on an X5 and read back; the
//! others go out the same way.

use link::{Mode as Sub, pb};
use pb::PhotographyOptionType as T;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(b: &[u8]) -> Option<Json> {
        let mut p = 0;
        let v = node(b, &mut p)?;
        ws(b, &mut p);
        (p == b.len()).then_some(v)
    }

    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|e| e.0 == k).map(|e| &e.1),
            _ => None,
        }
    }

    fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(v) => v,
            _ => &[],
        }
    }

    /// A list entry as the JSON spells it.
    fn text(&self) -> String {
        match self {
            Json::Str(s) => s.clone(),
            Json::Num(n) => format!("{n}"),
            Json::Bool(b) => b.to_string(),
            _ => String::new(),
        }
    }
}

fn ws(b: &[u8], p: &mut usize) {
    while b.get(*p).is_some_and(u8::is_ascii_whitespace) {
        *p += 1
    }
}

fn node(b: &[u8], p: &mut usize) -> Option<Json> {
    ws(b, p);
    let lit = |p: &mut usize, s: &[u8], v: Json| {
        b[*p..].starts_with(s).then(|| {
            *p += s.len();
            v
        })
    };
    match *b.get(*p)? {
        c @ (b'{' | b'[') => {
            *p += 1;
            let (mut o, mut a) = (vec![], vec![]);
            ws(b, p);
            if b.get(*p) == Some(&(c + 2)) {
                *p += 1;
            } else {
                loop {
                    ws(b, p);
                    if c == b'{' {
                        let k = text(b, p)?;
                        ws(b, p);
                        (b.get(*p) == Some(&b':')).then_some(())?;
                        *p += 1;
                        o.push((k, node(b, p)?));
                    } else {
                        a.push(node(b, p)?);
                    }
                    ws(b, p);
                    match *b.get(*p)? {
                        b',' => *p += 1,
                        x if x == c + 2 => {
                            *p += 1;
                            break;
                        }
                        _ => return None,
                    }
                }
            }
            Some(if c == b'{' {
                Json::Obj(o)
            } else {
                Json::Arr(a)
            })
        }
        b'"' => text(b, p).map(Json::Str),
        b't' => lit(p, b"true", Json::Bool(true)),
        b'f' => lit(p, b"false", Json::Bool(false)),
        b'n' => lit(p, b"null", Json::Null),
        _ => {
            let n = b[*p..]
                .iter()
                .take_while(|c| b"+-.eE0123456789".contains(c))
                .count();
            let x = std::str::from_utf8(&b[*p..*p + n]).ok()?.parse().ok()?;
            *p += n;
            Some(Json::Num(x))
        }
    }
}

fn text(b: &[u8], p: &mut usize) -> Option<String> {
    (b.get(*p) == Some(&b'"')).then_some(())?;
    *p += 1;
    let mut s = vec![];
    loop {
        let c = *b.get(*p)?;
        *p += 1;
        match c {
            b'"' => return String::from_utf8(s).ok(),
            b'\\' => {
                let e = *b.get(*p)?;
                *p += 1;
                let ch = match e {
                    b'b' => '\u{8}',
                    b'f' => '\u{c}',
                    b'n' => '\n',
                    b'r' => '\r',
                    b't' => '\t',
                    b'u' => {
                        let hex = |p: &mut usize| {
                            let h = std::str::from_utf8(b.get(*p..*p + 4)?).ok()?;
                            *p += 4;
                            u32::from_str_radix(h, 16).ok()
                        };
                        let mut u = hex(p)?;
                        // a surrogate pair
                        if (0xd800..0xdc00).contains(&u) && b[*p..].starts_with(b"\\u") {
                            *p += 2;
                            u = 0x10000 + ((u - 0xd800) << 10) + (hex(p)? - 0xdc00);
                        }
                        char::from_u32(u)?
                    }
                    e => e as char,
                };
                s.extend(ch.encode_utf8(&mut [0; 4]).as_bytes());
            }
            c => s.push(c),
        }
    }
}

/// A setting of a mode: its name, the settings it depends on, its value lists by their values (`a|b`, `default`).
#[derive(Clone, Debug)]
pub struct Key {
    pub name: String,
    pub dep: Vec<String>,
    pub list: Vec<(String, Vec<String>)>,
}

impl Key {
    /// The values allowed when `cur` gives the current value of each setting (None: unknown).
    pub fn allowed(&self, cur: &dyn Fn(&str) -> Option<String>) -> &[String] {
        let k = self
            .dep
            .iter()
            .map(|d| cur(d).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("|");
        let l = self.list.iter();
        l.clone()
            .find(|l| l.0 == k)
            .or_else(|| l.clone().find(|l| l.0 == "default"))
            .map_or(&[], |l| &l.1)
    }

    /// Every value any list names, in first-seen order.
    pub fn all(&self) -> Vec<&str> {
        let mut v: Vec<&str> = vec![];
        for x in self.list.iter().flat_map(|l| &l.1) {
            if !v.contains(&x.as_str()) {
                v.push(x)
            }
        }
        v
    }
}

#[derive(Clone, Debug)]
pub struct Mode {
    pub name: String,
    pub keys: Vec<Key>,
}

/// A lens block: its photo, video and live mode names in the camera's order, and the modes.
#[derive(Clone, Debug, Default)]
pub struct Block {
    pub name: String,
    pub photo: Vec<String>,
    pub video: Vec<String>,
    pub live: Vec<String>,
    pub modes: Vec<Mode>,
}

impl Block {
    pub fn mode(&self, n: &str) -> Option<&Mode> {
        self.modes.iter().find(|m| m.name == n)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Conf {
    pub version: i32,
    pub blocks: Vec<Block>,
}

impl Conf {
    pub fn parse(version: i32, b: &[u8]) -> Result<Conf, String> {
        let Json::Obj(top) = Json::parse(b).ok_or("/CONF: not JSON")? else {
            return Err("/CONF: not an object".into());
        };
        let names = |j: &Json, k| {
            j.get(k)
                .map_or(vec![], |a| a.arr().iter().map(Json::text).collect())
        };
        let blocks = top
            .iter()
            .map(|(name, j)| {
                let Json::Obj(m) = j else {
                    return Block::default();
                };
                let modes = m
                    .iter()
                    .filter_map(|(mn, mj)| {
                        let Json::Obj(ks) = mj else { return None };
                        let keys = ks
                            .iter()
                            .map(|(k, kj)| Key {
                                name: k.clone(),
                                dep: names(kj, "depend_on"),
                                list: match kj.get("list") {
                                    Some(Json::Obj(l)) => l
                                        .iter()
                                        .map(|(a, v)| {
                                            (a.clone(), v.arr().iter().map(Json::text).collect())
                                        })
                                        .collect(),
                                    _ => vec![],
                                },
                            })
                            .collect();
                        Some(Mode {
                            name: mn.clone(),
                            keys,
                        })
                    })
                    .collect();
                Block {
                    name: name.clone(),
                    photo: names(j, "photo_sub_mode"),
                    video: names(j, "video_sub_mode"),
                    live: names(j, "live_sub_mode"),
                    modes,
                }
            })
            .collect();
        Ok(Conf { version, blocks })
    }

    /// `Sphere06C` (both lenses) or `Fisheyes06C` (one).
    pub fn block(&self, both: bool) -> Option<&Block> {
        let n = if both { "Sphere06C" } else { "Fisheyes06C" };
        self.blocks.iter().find(|b| b.name == n)
    }
}

/// Where a setting's value lives on the wire. `Dur` is read only: its encoding is not known for sure. `P3` (AE_SWITCH
/// in the app) is 1 / 0, as the app's `DefaultAESwitchCameraImpl` writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Prog,
    Iso,
    Shutter,
    Ev,
    Wb,
    IsoTop,
    Gamma,
    PanoExp,
    ILog,
    Hdr,
    Pure,
    Res,
    PhotoRes,
    Raw,
    PhotoHdr,
    Aeb,
    Timer,
    Accel,
    Burst,
    Fov,
    Horizon,
    Size,
    Splice,
    P3,
    Dur,
}

/// The slot and label of the settings this app shows as controls, in the order it shows them.
pub const KEYS: [(&str, Slot, &str); 25] = [
    ("record_resolution", Slot::Res, "Resolution"),
    ("photo_resolution", Slot::PhotoRes, "Resolution"),
    ("exposure_program", Slot::Prog, "Exposure"),
    ("exposure_iso", Slot::Iso, "ISO"),
    ("exposure_shutter_speed", Slot::Shutter, "Shutter"),
    ("exposure_bias", Slot::Ev, "EV"),
    ("video_iso_top_limit", Slot::IsoTop, "Max ISO"),
    ("white_balance", Slot::Wb, "White balance"),
    ("exposure_individual", Slot::PanoExp, "Lens exposure"),
    ("color_mode", Slot::Gamma, "Colour"),
    ("hdr_switch", Slot::Hdr, "Active HDR"),
    ("i_log_switch", Slot::ILog, "I-Log"),
    ("p3_switch", Slot::P3, "AdaptiveTone"),
    ("pure_video_enhance_switch", Slot::Pure, "Low light boost"),
    ("hdr_photo_mode", Slot::PhotoHdr, "HDR"),
    ("aeb_capture_num", Slot::Aeb, "Brackets"),
    ("raw_capture_type", Slot::Raw, "Format"),
    ("splicing_base_enable", Slot::Splice, "In-camera stitching"),
    ("photography_self_timer", Slot::Timer, "Self-timer"),
    ("burst_capture_params", Slot::Burst, "Burst"),
    ("accelerate_frequency", Slot::Accel, "Speed"),
    ("record_duration", Slot::Dur, "Duration"),
    ("fov_type", Slot::Fov, "Field of view"),
    ("horizon_lock", Slot::Horizon, "Tilt fix"),
    ("photo_size_id", Slot::Size, "Aspect ratio"),
];

pub fn slot(key: &str) -> Option<Slot> {
    KEYS.iter().find(|k| k.0 == key).map(|k| k.1)
}

pub fn wired(s: Slot) -> bool {
    s != Slot::Dur
}

/// `1/8000`, `1.3`, `0` (auto) → seconds.
fn secs(v: &str) -> Option<f64> {
    match v.split_once('/') {
        Some((a, b)) => Some(a.parse::<f64>().ok()? / b.parse::<f64>().ok()?),
        None => v.parse().ok(),
    }
}

/// `5760_2880_30[_plus]` → `VideoResolution`: the enum's name, else for a 2:1 picture the per-lens square
/// (`RES_PANO_2880_2880P48`, then `RES_2880_2880P30`), as the camera numbers its 360 modes.
fn res(v: &str) -> Option<pb::VideoResolution> {
    let p: Vec<&str> = v.split('_').collect();
    let (w, h, f) = (p.first()?, p.get(1)?, p.get(2)?);
    let plus = match p.get(3) {
        Some(&"plus") => "_PLUS",
        None => "",
        _ => return None,
    };
    let mut n = vec![format!("RES_{w}_{h}P{f}{plus}")];
    if w.parse::<u32>().ok()? == 2 * h.parse::<u32>().ok()? {
        n.push(format!("RES_PANO_{h}_{h}P{f}{plus}"));
        n.push(format!("RES_{h}_{h}P{f}{plus}"));
    }
    n.iter().find_map(|n| pb::VideoResolution::from_str_name(n))
}

/// The wire value of `/CONF` value `v` of a setting in `s` (`Burst`: count × 100 + seconds).
pub fn val(s: Slot, v: &str) -> Option<f64> {
    use pb::photography_options as po;
    let b = |v: &str| match v {
        "true" => Some(1.0),
        "false" => Some(0.0),
        _ => None,
    };
    let e = |x: Option<i32>| x.map(f64::from);
    match s {
        Slot::Prog => e(po::exposure_options::Program::from_str_name(v).map(|x| x as i32)),
        Slot::Iso | Slot::Wb | Slot::IsoTop | Slot::Aeb | Slot::Timer | Slot::Accel | Slot::Dur => {
            v.parse::<u32>().ok().map(f64::from)
        }
        Slot::Shutter => secs(v),
        Slot::Ev => Some((v.parse::<f64>().ok()? * 10.0).round()),
        Slot::Gamma => e(pb::GammaMode::from_str_name(v).map(|x| x as i32)),
        Slot::PanoExp => match v {
            "off" => Some(0.0),
            "light_individual" => Some(1.0),
            "individual" => Some(2.0),
            _ => None,
        },
        Slot::ILog | Slot::Hdr | Slot::Pure | Slot::Horizon | Slot::Splice | Slot::P3 => b(v),
        Slot::Res => e(res(v).map(|x| x as i32)),
        Slot::PhotoRes => e(pb::PhotoSize::from_str_name(&format!("Size_{v}")).map(|x| x as i32)),
        Slot::Raw => e(
            pb::RawCaptureType::from_str_name(&format!("RAW_CAPTURE_TYPE_{v}")).map(|x| x as i32),
        ),
        Slot::PhotoHdr => e(match v {
            "OFF" => Some(po::PhotoHdrType::HdrOff as i32),
            "ON" => Some(po::PhotoHdrType::HdrAuto as i32),
            "AEB" => Some(po::PhotoHdrType::HdrAeb as i32),
            _ => None,
        }),
        Slot::Burst => {
            let (n, t) = v.split_once('/')?;
            Some(n.parse::<u32>().ok()? as f64 * 100.0 + t.parse::<u32>().ok()? as f64)
        }
        // the names the FovType enum spells otherwise
        Slot::Fov => e(match v {
            "ACTION_VIEW" => Some(pb::FovType::FovPov as i32),
            "MAX_VIEW" => Some(pb::FovType::Infinitywide as i32),
            "MEGA_VIEW" => Some(pb::FovType::Mega as i32),
            v => pb::FovType::from_str_name(&format!("FOV_{v}")).map(|x| x as i32),
        }),
        Slot::Size => e(pb::PhotoSizeId::from_str_name(&format!(
            "PHOTO_SIZE_{}",
            v.replace("235:100", "2P35_1").replace(':', "_")
        ))
        .map(|x| x as i32)),
    }
}

/// The exposure options a mode reads and writes: still (and video, as the app writes both) for photo modes, video
/// for the others.
fn expo(photo: bool) -> &'static [T] {
    if photo {
        &[T::VideoExposureOptions, T::StillExposureOptions]
    } else {
        &[T::VideoExposureOptions]
    }
}

/// The photography option types slot `s` reads and writes (none for `Splice`, an `Options` field).
pub fn types(s: Slot, photo: bool) -> &'static [T] {
    match s {
        Slot::Prog | Slot::Iso | Slot::Shutter => expo(photo),
        Slot::Ev => &[T::ExposureBias],
        Slot::Wb => &[T::WhiteBalanceValue],
        Slot::IsoTop => &[T::VideoIsoTopLimit],
        Slot::Gamma => &[T::VideoGammaMode],
        Slot::PanoExp => &[T::PanoExposureMode],
        Slot::ILog => &[T::ILogSwitch],
        Slot::Hdr => &[T::HdrSwitchStatus],
        Slot::Pure => &[T::PureVideoEnhancType],
        Slot::Res => &[T::RecordResolution],
        Slot::PhotoRes => &[T::PhotoResolution],
        Slot::Raw => &[T::RawCaptureType],
        Slot::PhotoHdr => &[T::PhotoHdrType],
        Slot::Aeb => &[T::AebCaptureNum],
        Slot::Timer => &[T::PhotographySelfTimer],
        Slot::Accel => &[T::AccelerateFrequency],
        Slot::Burst => &[T::BurstCaptureNum, T::BurstCaptureTime],
        Slot::Fov => &[T::FovType],
        Slot::Horizon => &[T::FlowstateHorizonEnable],
        Slot::Size => &[T::PhotoSizeId],
        Slot::P3 => &[T::P3Switch],
        Slot::Dur => &[T::RecordDuraion],
        Slot::Splice => &[],
    }
}

/// The camera's value in slot `s` (`splice`: `Options::enable_internal_splicing`).
pub fn get(s: Slot, o: &pb::PhotographyOptions, splice: bool, photo: bool) -> f64 {
    let x = if photo {
        o.still_exposure.as_ref()
    } else {
        o.video_exposure.as_ref()
    };
    let x = x.cloned().unwrap_or_default();
    match s {
        Slot::Prog => x.program as f64,
        Slot::Iso => x.iso as f64,
        Slot::Shutter => x.shutter_speed,
        Slot::Ev => o.exposure_bias as f64,
        Slot::Wb => o.white_balance_value as f64,
        Slot::IsoTop => o.video_iso_top_limit as f64,
        Slot::Gamma => o.gamma_mode as f64,
        Slot::PanoExp => o.pano_exposure_mode as f64,
        Slot::ILog => o.i_log_switch as u8 as f64,
        Slot::Hdr => o.hdr_switch_status as u8 as f64,
        Slot::Pure => o.purevideo_enhanc_switch as u8 as f64,
        Slot::Res => o.record_resolution as f64,
        Slot::PhotoRes => o.photo_resolution as f64,
        Slot::Raw => o.raw_capture_type as f64,
        Slot::PhotoHdr => o.hdr_mode as f64,
        Slot::Aeb => o.aeb_capture_num as f64,
        Slot::Timer => o.photography_self_timer as f64,
        Slot::Accel => o.accelerate_frequency as f64,
        Slot::Burst => o.burst_capture_num as f64 * 100.0 + o.burst_capture_time as f64,
        Slot::Fov => o.fov_type as f64,
        Slot::Horizon => (o.flowstate_horizon_enable != 0) as u8 as f64,
        Slot::Size => o.photo_size_id as f64,
        Slot::Splice => splice as u8 as f64,
        Slot::P3 => (o.p3_switch != 0) as u8 as f64,
        Slot::Dur => o.record_duration as f64,
    }
}

/// Equal on the wire: shutter speeds are doubles the camera may round.
pub fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(1.0)
}

/// A setting change as it goes out.
#[derive(Clone, Debug, PartialEq)]
pub enum Write {
    Photo(pb::SetPhotographyOptions),
    Opts(pb::SetOptions),
}

/// The message that sets slot `s` to `/CONF` value `v` in function mode `fm`, from `cur` (the camera's values of the
/// mode: the exposure options go whole, with the other two values as they are). None for a read-only slot or an
/// unknown value.
pub fn write(
    s: Slot,
    v: &str,
    cur: &pb::PhotographyOptions,
    fm: pb::FunctionMode,
    photo: bool,
) -> Option<Write> {
    let x = val(s, v).filter(|_| wired(s))?;
    if s == Slot::Splice {
        let o = pb::Options {
            enable_internal_splicing: x != 0.0,
            ..Default::default()
        };
        return Some(Write::Opts(pb::SetOptions {
            option_types: vec![pb::OptionType::InternalSplicing as i32],
            value: Some(o),
        }));
    }
    let mut o = pb::PhotographyOptions::default();
    let (i, u, b) = (x as i32, x as u32, x != 0.0);
    match s {
        Slot::Prog | Slot::Iso | Slot::Shutter => {
            let src = if photo {
                &cur.still_exposure
            } else {
                &cur.video_exposure
            };
            let mut e = src.unwrap_or_default();
            match s {
                Slot::Prog => e.program = i,
                Slot::Iso => e.iso = u,
                _ => e.shutter_speed = x,
            }
            if photo {
                o.still_exposure = Some(e);
            }
            o.video_exposure = Some(e);
        }
        Slot::Ev => o.exposure_bias = i,
        Slot::Wb => o.white_balance_value = u,
        Slot::IsoTop => o.video_iso_top_limit = u,
        Slot::Gamma => o.gamma_mode = i,
        Slot::PanoExp => o.pano_exposure_mode = i,
        Slot::ILog => o.i_log_switch = b,
        Slot::Hdr => o.hdr_switch_status = b,
        Slot::Pure => o.purevideo_enhanc_switch = b,
        Slot::Res => o.record_resolution = i,
        Slot::PhotoRes => o.photo_resolution = i,
        Slot::Raw => o.raw_capture_type = i,
        Slot::PhotoHdr => o.hdr_mode = i,
        Slot::Aeb => o.aeb_capture_num = u,
        Slot::Timer => o.photography_self_timer = u,
        Slot::Accel => o.accelerate_frequency = u,
        Slot::Burst => (o.burst_capture_num, o.burst_capture_time) = (u / 100, u % 100),
        Slot::Fov => o.fov_type = i,
        Slot::Horizon => o.flowstate_horizon_enable = b as i32,
        Slot::Size => o.photo_size_id = i,
        Slot::P3 => o.p3_switch = u,
        Slot::Splice | Slot::Dur => unreachable!("filtered above"),
    }
    Some(Write::Photo(pb::SetPhotographyOptions {
        option_types: types(s, photo).iter().map(|&t| t as i32).collect(),
        value: Some(o),
        function_mode: fm as i32,
    }))
}

/// A photo mode (else a video or live one).
pub fn photo(mode: &str) -> bool {
    mode.starts_with("PHOTO_")
}

/// The function mode a `/CONF` mode's photography options live in.
pub fn func(mode: &str) -> Option<pb::FunctionMode> {
    use pb::FunctionMode as F;
    Some(match mode {
        "PHOTO_SINGLE" => F::NormalImage,
        "PHOTO_HDR" => F::HdrImage,
        "PHOTO_BURST" => F::Burst,
        "PHOTO_INTERVAL" => F::IntervalShooting,
        "PHOTO_AEB_NIGHT" => F::AebNightImage,
        "PHOTO_STARLAPSE" => F::StarlapseImage,
        "VIDEO_NORMAL" => F::NormalVideo,
        "VIDEO_BULLETTIME" => F::HighFrameRate,
        "VIDEO_TIMELAPSE" => F::MobileTimelapse,
        "VIDEO_HDR" => F::HdrVideo,
        "VIDEO_TIMESHIFT" => F::Timeshift,
        "VIDEO_SUPER" => F::SuperVideo,
        "VIDEO_LOOPRECORDING" => F::LoopRecordingVideo,
        "VIDEO_SELFIE" => F::SelfieVideo,
        "VIDEO_PURE" => F::PureVideo,
        "VIDEO_LIVING_MODE_PANO" | "VIDEO_LIVING_MODE_WIDE" => F::LiveStream,
        "VIDEO_DASHCAM" => F::DashCam,
        "VIDEO_GIMBAL" => F::Ptz,
        _ => return None,
    })
}

/// The sub-mode option value of a `/CONF` mode (`SET_OPTIONS` 40 / 41).
pub fn sub(mode: &str) -> Option<Sub> {
    use pb::{PhotoSubMode as P, VideoSubMode as V};
    Some(match mode {
        "VIDEO_DASHCAM" => Sub::Video(V::VideoDashCam),
        "VIDEO_GIMBAL" => Sub::Video(V::VideoPtz),
        "VIDEO_LIVING_MODE_PANO" | "VIDEO_LIVING_MODE_WIDE" => Sub::Video(V::VideoLiveview),
        m if photo(m) => Sub::Photo(P::from_str_name(m)?),
        m => Sub::Video(V::from_str_name(m)?),
    })
}

/// The `/CONF` name of the camera's sub-mode (`VIDEO_NONE`: a photo mode). The live modes share one sub-mode; it reads
/// as the 360 one.
pub fn current(o: &pb::Options) -> Option<String> {
    use pb::VideoSubMode as V;
    Some(match o.video_sub_mode() {
        V::VideoNone => o.photo_sub_mode().as_str_name().into(),
        V::VideoDashCam => "VIDEO_DASHCAM".into(),
        V::VideoPtz => "VIDEO_GIMBAL".into(),
        V::VideoLiveview => "VIDEO_LIVING_MODE_PANO".into(),
        v => v.as_str_name().into(),
    })
}

/// What the shutter does in a mode (app_features.md §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shot {
    Photo(pb::take_picture::Mode),
    Rec(pb::CaptureMode),
    Lapse(pb::TimelapseMode),
    /// The live modes: the preview, streamed by this app.
    None,
}

pub fn shot(mode: &str) -> Shot {
    use pb::{CaptureMode as C, TimelapseMode as L, take_picture::Mode as P};
    match mode {
        "PHOTO_SINGLE" | "PHOTO_HDR" => Shot::Photo(P::Normal),
        "PHOTO_BURST" => Shot::Photo(P::Burst),
        "PHOTO_AEB_NIGHT" => Shot::Photo(P::AebNight),
        "PHOTO_INTERVAL" => Shot::Lapse(L::TimelapseIntervalShooting),
        "PHOTO_STARLAPSE" => Shot::Lapse(L::TimelapseStarlapseShooting),
        "VIDEO_TIMELAPSE" => Shot::Lapse(L::MobileTimelapseVideo),
        "VIDEO_BULLETTIME" => Shot::Rec(C::Bullettime),
        "VIDEO_HDR" => Shot::Rec(C::Hdr),
        "VIDEO_TIMESHIFT" => Shot::Rec(C::Timeshift),
        "VIDEO_SUPER" => Shot::Rec(C::SuperVideo),
        "VIDEO_LOOPRECORDING" => Shot::Rec(C::Looprecording),
        "VIDEO_SELFIE" => Shot::Rec(C::Selfie),
        "VIDEO_PURE" => Shot::Rec(C::PureVideo),
        "VIDEO_DASHCAM" => Shot::Rec(C::DashCam),
        "VIDEO_GIMBAL" => Shot::Rec(C::Ptz),
        m if m.starts_with("VIDEO_LIVING") => Shot::None,
        _ => Shot::Rec(C::Normal),
    }
}

/// Highlight marks: while recording in the normal, loop and HDR modes, FreeFrame (single lens) and TimeShift
/// (app: `OneX5PanoramaSupportConfig`; TimeShift only at 1×, which this app does not check).
pub fn marks(mode: &str) -> bool {
    matches!(
        mode,
        "VIDEO_NORMAL" | "VIDEO_LOOPRECORDING" | "VIDEO_HDR" | "VIDEO_SUPER" | "VIDEO_TIMESHIFT"
    )
}

/// A mode's name for the UI.
pub fn label(mode: &str) -> &str {
    match mode {
        "PHOTO_SINGLE" => "Photo",
        "PHOTO_BURST" => "Burst",
        "PHOTO_INTERVAL" => "Interval",
        "PHOTO_STARLAPSE" => "Starlapse",
        "VIDEO_NORMAL" => "Video",
        "VIDEO_PURE" => "PureVideo",
        "VIDEO_GIMBAL" => "InstaFrame",
        "VIDEO_TIMELAPSE" => "Timelapse",
        "VIDEO_TIMESHIFT" => "TimeShift",
        "VIDEO_BULLETTIME" => "Bullet Time",
        "VIDEO_DASHCAM" => "Road Mode",
        "VIDEO_LOOPRECORDING" => "Loop Recording",
        "VIDEO_SUPER" => "FreeFrame",
        "VIDEO_SELFIE" => "Me Mode",
        "VIDEO_LIVING_MODE_PANO" => "360° Live",
        "VIDEO_LIVING_MODE_WIDE" => "Reframe Live",
        m => m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn hex(s: &str) -> Vec<u8> {
        s.split_whitespace()
            .map(|x| u8::from_str_radix(x, 16).unwrap())
            .collect()
    }

    #[test]
    fn json() {
        let j =
            Json::parse(br#" {"a": [1, -2.5e1, "x\u00e9\"", true, null, {}], "b": {"c": []}} "#)
                .unwrap();
        assert_eq!(
            j.get("a"),
            Some(&Json::Arr(vec![
                Json::Num(1.0),
                Json::Num(-25.0),
                Json::Str("x\u{e9}\"".into()),
                Json::Bool(true),
                Json::Null,
                Json::Obj(vec![])
            ]))
        );
        assert_eq!(
            j.get("b").and_then(|b| b.get("c")),
            Some(&Json::Arr(vec![]))
        );
        assert_eq!(
            Json::parse(br#""\ud83d\ude00""#),
            Some(Json::Str("\u{1f600}".into()))
        );
        for bad in [&b"{"[..], b"[1,]", b"{\"a\" 1}", b"tru", b"[1] 2"] {
            assert_eq!(Json::parse(bad), None, "{}", String::from_utf8_lossy(bad));
        }
    }

    /// The wire values against `proto/` (enum numbers) and the app's encodings: EV in tenths (the Insta360 app's
    /// `capture/model/setting/EV.java` has EV_MINUS_4 = −40 … EV_4 = 40, which `setExposureEV` writes into
    /// exposure_bias as it is), white balance in K, the shutter in seconds.
    #[test]
    fn values() {
        let r = |v| val(Slot::Res, v);
        // a 2:1 360 picture is numbered by its per-lens square
        assert_eq!(r("3840_1920_30"), Some(0.0));
        assert_eq!(r("5760_2880_30"), Some(10.0));
        assert_eq!(r("5760_2880_30_plus"), Some(317.0));
        assert_eq!(r("5760_2880_48"), Some(338.0));
        assert_eq!(r("7680_3840_30"), Some(167.0));
        assert_eq!(r("11264_5632_30"), Some(306.0));
        assert_eq!(r("7680_3840"), None);
        assert_eq!(val(Slot::PhotoRes, "11968_5984"), Some(11.0));
        assert_eq!(val(Slot::Ev, "-0.3"), Some(-3.0));
        assert_eq!(val(Slot::Ev, "3.7"), Some(37.0));
        assert_eq!(val(Slot::Shutter, "1/8000"), Some(1.0 / 8000.0));
        assert_eq!(val(Slot::Shutter, "1.3"), Some(1.3));
        assert_eq!(val(Slot::Prog, "FULL_AUTO"), Some(5.0));
        assert_eq!(val(Slot::Prog, "SHUTTER_PRIORITY"), Some(2.0));
        assert_eq!(val(Slot::Gamma, "VIVID"), Some(2.0));
        assert_eq!(val(Slot::Gamma, "FLAT"), Some(3.0));
        assert_eq!(val(Slot::Raw, "PURESHOT_RAW"), Some(4.0));
        assert_eq!(val(Slot::PhotoHdr, "AEB"), Some(2.0));
        assert_eq!(val(Slot::PanoExp, "individual"), Some(2.0));
        assert_eq!(val(Slot::Burst, "15/3"), Some(1503.0));
        assert_eq!(val(Slot::Fov, "MEGA_VIEW"), Some(14.0));
        assert_eq!(val(Slot::Fov, "DEWARP"), Some(13.0));
        assert_eq!(val(Slot::Fov, "LINEAR"), Some(1.0));
        assert_eq!(val(Slot::Size, "235:100"), Some(5.0));
        assert_eq!(val(Slot::Size, "9:16"), Some(2.0));
        assert_eq!(val(Slot::Hdr, "true"), Some(1.0));
    }

    /// The messages, byte for byte from the field numbers of `set_photography_options.proto` /
    /// `photography_options.proto` / `set_options.proto`.
    #[test]
    fn messages() {
        let cur = pb::PhotographyOptions {
            video_exposure: Some(pb::photography_options::ExposureOptions {
                program: 3,
                iso: 400,
                shutter_speed: 0.01,
            }),
            ..Default::default()
        };
        let w = |s, v, photo| {
            let fm = if photo {
                pb::FunctionMode::NormalImage
            } else {
                pb::FunctionMode::NormalVideo
            };
            match write(s, v, &cur, fm, photo).unwrap() {
                Write::Photo(m) => m.encode_to_vec(),
                Write::Opts(m) => m.encode_to_vec(),
            }
        };
        // types [7], value {exposure_bias (sint32) −3 → zigzag 5}, function mode 7
        assert_eq!(w(Slot::Ev, "-0.3", false), hex("08 07 12 02 38 05 18 07"));
        // types [39], value {white_balance_value 5000}, function mode 7
        assert_eq!(
            w(Slot::Wb, "5000", false),
            hex("08 27 12 04 b8 02 88 27 18 07")
        );
        // types [31], value {record_resolution 10}
        assert_eq!(
            w(Slot::Res, "5760_2880_30", false),
            hex("08 1f 12 03 f8 01 0a 18 07")
        );
        // ISO in a video mode: the video exposure whole {program 3, iso 800, shutter 0.01}, type [21]
        let mut e = hex("08 15 12 11 aa 01 0e 08 03 10 a0 06 19");
        e.extend(0.01f64.to_le_bytes());
        e.extend(hex("18 07"));
        assert_eq!(w(Slot::Iso, "800", false), e);
        // a photo mode writes the still exposure too, starting from the still values (none here), types [21, 20]
        assert_eq!(
            w(Slot::Iso, "200", true),
            hex("08 15 08 14 12 0c a2 01 03 10 c8 01 aa 01 03 10 c8 01 18 06")
        );
        // types [77], value {hdr_mode 2}, function mode 6
        assert_eq!(
            w(Slot::PhotoHdr, "AEB", true),
            hex("08 4d 12 03 e8 04 02 18 06")
        );
        // SET_OPTIONS types [135], value {enable_internal_splicing (field 135) true}
        assert_eq!(w(Slot::Splice, "false", true), hex("08 87 01 12 00"));
        assert_eq!(
            w(Slot::Splice, "true", true),
            hex("08 87 01 12 03 b8 08 01")
        );
        // burst 15 in 3 s: types [44, 51]
        assert_eq!(
            w(Slot::Burst, "15/3", true),
            hex("08 2c 08 33 12 06 e0 02 0f 98 03 03 18 06")
        );
        // types [86], value {p3_switch 1}
        assert_eq!(
            w(Slot::P3, "true", false),
            hex("08 56 12 03 b0 05 01 18 07")
        );
        assert!(write(Slot::Dur, "60", &cur, pb::FunctionMode::NormalVideo, false).is_none());
        assert!(
            write(
                Slot::Res,
                "1_2_3",
                &cur,
                pb::FunctionMode::NormalVideo,
                false
            )
            .is_none()
        );
    }

    #[test]
    fn modes() {
        assert_eq!(func("VIDEO_NORMAL"), Some(pb::FunctionMode::NormalVideo));
        assert_eq!(
            func("PHOTO_STARLAPSE"),
            Some(pb::FunctionMode::StarlapseImage)
        );
        assert!(matches!(
            sub("VIDEO_GIMBAL"),
            Some(Sub::Video(pb::VideoSubMode::VideoPtz))
        ));
        assert!(matches!(
            sub("PHOTO_BURST"),
            Some(Sub::Photo(pb::PhotoSubMode::PhotoBurst))
        ));
        let mut o = pb::Options {
            video_sub_mode: pb::VideoSubMode::VideoNone as i32,
            ..Default::default()
        };
        assert_eq!(current(&o).as_deref(), Some("PHOTO_SINGLE"));
        o.video_sub_mode = pb::VideoSubMode::VideoDashCam as i32;
        assert_eq!(current(&o).as_deref(), Some("VIDEO_DASHCAM"));
        o.video_sub_mode = pb::VideoSubMode::VideoPure as i32;
        assert_eq!(current(&o).as_deref(), Some("VIDEO_PURE"));
        assert_eq!(shot("VIDEO_PURE"), Shot::Rec(pb::CaptureMode::PureVideo));
        assert_eq!(
            shot("PHOTO_INTERVAL"),
            Shot::Lapse(pb::TimelapseMode::TimelapseIntervalShooting)
        );
        assert_eq!(shot("VIDEO_LIVING_MODE_PANO"), Shot::None);
        assert_eq!(
            shot("PHOTO_SINGLE"),
            Shot::Photo(pb::take_picture::Mode::Normal)
        );
    }

    /// The camera's own `/CONF` (`X5_CONF`, e.g. `re/hw/x5-conf.bin`): every mode and setting parses, every value
    /// the controls offer has a wire value, and dependencies pick their lists.
    #[test]
    fn camera_conf() {
        let Some(p) = std::env::var_os("X5_CONF") else {
            eprintln!("X5_CONF not set: skipping");
            return;
        };
        let c = Conf::parse(71, &std::fs::read(p).unwrap()).unwrap();
        let s = c.block(true).unwrap();
        assert!(s.video.contains(&"VIDEO_NORMAL".to_string()));
        assert_eq!(s.photo.len() + s.video.len() + s.live.len(), s.modes.len());
        let (mut n, mut bad) = (0, vec![]);
        for b in &c.blocks {
            for m in &b.modes {
                assert!(func(&m.name).is_some(), "{}", m.name);
                assert!(sub(&m.name).is_some(), "{}", m.name);
                for k in &m.keys {
                    let Some(sl) = slot(&k.name) else { continue };
                    for v in k.all() {
                        n += 1;
                        if val(sl, v).is_none() {
                            bad.push(format!("{}.{}.{} = {v}", b.name, m.name, k.name));
                        }
                    }
                }
            }
        }
        println!(
            "{n} values of the shown settings, {} without a wire value: {bad:?}",
            bad.len()
        );
        assert!(bad.is_empty());
        let v = s.mode("VIDEO_NORMAL").unwrap();
        let k = |n: &str| v.keys.iter().find(|k| k.name == n).unwrap();
        // I-Log needs Active HDR at 5.7K, AdaptiveTone off
        let cur = |hdr: &'static str| {
            move |d: &str| {
                Some(match d {
                    "hdr_switch" => hdr.into(),
                    "record_resolution" => "5760_2880_30".into(),
                    _ => "false".into(),
                })
            }
        };
        assert_eq!(k("i_log_switch").allowed(&cur("true")), ["true", "false"]);
        assert_eq!(k("i_log_switch").allowed(&cur("false")), ["false"]);
        assert_eq!(k("exposure_program").allowed(&cur("false")).len(), 6);
    }
}
