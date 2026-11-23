//! The camera's capture-parameter JSON (`/CONF`, PROTOCOL.md §8.10, `analysis/app_features.md` §3–4): per lens block
//! (`Sphere06C` 360, `Fisheyes06C` single lens) its modes, per mode its settings, each with the values allowed given
//! the values of the settings it depends on (`depend_on` joined by `|`, else `default`). And how a setting reaches
//! the camera: `SET_PHOTO_OPTIONS` with the mode's profile, or `SET_OPTIONS`. The app maps the JSON's strings to
//! protobuf values through an asset of its own (`proto.json`); here the enum values in `proto/` spell them behind the
//! enum's prefix (resolutions by the per-lens picture), plus the few they do not give. The writes in [`CHECKED`] ran on an X5 and read back, so did the
//! switches to the modes in [`SWITCHED`] and the shutter in those of [`SHOT`] (360, fw v1.13.21: the app's
//! `capture::tests::real_writes` / `real_ilog`); the others go out the same way. The capture worker sends only values
//! a mode's lists allow (`Key::allowed`).

use link::{Mode as Sub, pb};
use pb::Param as T;

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

/// The settings (mode, key) whose writes ran on an X5 (360) and read back.
pub const CHECKED: [(&str, &str); 8] = [
    ("PHOTO_SINGLE", "white_balance"),
    ("VIDEO_NORMAL", "record_resolution"),
    ("VIDEO_NORMAL", "p3_switch"),
    ("VIDEO_NORMAL", "exposure_program"),
    ("VIDEO_NORMAL", "exposure_iso"),
    ("VIDEO_NORMAL", "exposure_shutter_speed"),
    ("VIDEO_NORMAL", "white_balance"),
    ("VIDEO_NORMAL", "i_log_switch"),
];

/// The modes a switch to ran on an X5 (360) and read back.
pub const SWITCHED: [&str; 2] = ["PHOTO_SINGLE", "VIDEO_NORMAL"];

/// The modes whose shutter ran on an X5 (360): the photo, the normal recording (with highlight marks). By mode name:
/// [`shot`] gives unknown modes the normal recording too.
pub const SHOT: [&str; 2] = ["PHOTO_SINGLE", "VIDEO_NORMAL"];

/// Whether the camera's sub-mode values are ones the protocol knows, so [`current`]'s name is the camera's mode: a
/// value it does not know reads as the enum's first (VIDEO_NORMAL, PHOTO_SINGLE), a mode it may well not be in.
pub fn known(o: &pb::Settings) -> bool {
    use pb::{PhotoMode as P, VideoMode as V};
    match V::try_from(o.video_mode) {
        Ok(V::VideoNone) => P::try_from(o.photo_mode).is_ok(),
        Ok(_) => true,
        Err(_) => false,
    }
}

/// A `/CONF` value as people read it: `0` as Auto or Off where it means that, units, a resolution's parts.
pub fn show(s: Slot, v: &str) -> String {
    let p: Vec<&str> = v.split('_').collect();
    match (s, v) {
        (Slot::Iso | Slot::Shutter | Slot::Wb | Slot::IsoTop, "0") => "Auto".into(),
        (Slot::Timer | Slot::Accel, "0") => "Off".into(),
        (Slot::Wb, v) => format!("{v} K"),
        (Slot::Timer, v) => format!("{v} s"),
        (Slot::Accel, v) => format!("{v}×"),
        (Slot::Shutter, v) => format!("{v} s"),
        (Slot::Ev, v) if !v.starts_with('-') && v != "0" => format!("+{v}"),
        (Slot::Res, _) if p.len() >= 3 => format!(
            "{} × {}, {} fps{}",
            p[0],
            p[1],
            p[2],
            if p.len() > 3 { " +" } else { "" }
        ),
        (Slot::PhotoRes, _) if p.len() == 2 => format!("{} × {}", p[0], p[1]),
        (Slot::Dur, v) => v.parse::<u32>().map_or(v.into(), |x| match x {
            0 => "Unlimited".into(),
            x if x % 3600 == 0 => format!("{} h", x / 3600),
            x => format!("{} min", x / 60),
        }),
        (Slot::Burst, v) => v
            .split_once('/')
            .map_or(v.into(), |(n, t)| format!("{n} in {t} s")),
        _ => v.replace('_', " ").to_lowercase(),
    }
}

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

/// `5760_2880_30[_plus]` → `Resolution`: the value of that name, else for a 2:1 picture the per-lens square
/// (`RESOLUTION_360_2880_2880_48`, then `RESOLUTION_2880_2880_30`), as the camera numbers its 360 modes.
fn res(v: &str) -> Option<pb::Resolution> {
    let p: Vec<&str> = v.split('_').collect();
    let (w, h, f) = (p.first()?, p.get(1)?, p.get(2)?);
    let plus = match p.get(3) {
        Some(&"plus") => "_PLUS",
        None => "",
        _ => return None,
    };
    let mut n = vec![format!("RESOLUTION_{w}_{h}_{f}{plus}")];
    if w.parse::<u32>().ok()? == 2 * h.parse::<u32>().ok()? {
        n.push(format!("RESOLUTION_360_{h}_{h}_{f}{plus}"));
        n.push(format!("RESOLUTION_{h}_{h}_{f}{plus}"));
    }
    n.iter().find_map(|n| pb::Resolution::from_str_name(n))
}

/// The wire value of `/CONF` value `v` of a setting in `s` (`Burst`: count × 100 + seconds).
pub fn val(s: Slot, v: &str) -> Option<f64> {
    use pb::params as po;
    let b = |v: &str| match v {
        "true" => Some(1.0),
        "false" => Some(0.0),
        _ => None,
    };
    let e = |x: Option<i32>| x.map(f64::from);
    match s {
        Slot::Prog => {
            e(po::exposure::Program::from_str_name(&format!("PROGRAM_{v}")).map(|x| x as i32))
        }
        Slot::Iso | Slot::Wb | Slot::IsoTop | Slot::Aeb | Slot::Timer | Slot::Accel | Slot::Dur => {
            v.parse::<u32>().ok().map(f64::from)
        }
        Slot::Shutter => secs(v),
        Slot::Ev => Some((v.parse::<f64>().ok()? * 10.0).round()),
        Slot::Gamma => e(pb::Color::from_str_name(&format!("COLOR_{v}")).map(|x| x as i32)),
        Slot::PanoExp => match v {
            "off" => Some(0.0),
            "light_individual" => Some(1.0),
            "individual" => Some(2.0),
            _ => None,
        },
        Slot::ILog | Slot::Hdr | Slot::Pure | Slot::Horizon | Slot::Splice | Slot::P3 => b(v),
        Slot::Res => e(res(v).map(|x| x as i32)),
        Slot::PhotoRes => e(
            pb::PhotoResolution::from_str_name(&format!("PHOTO_RESOLUTION_{v}")).map(|x| x as i32),
        ),
        Slot::Raw => e(pb::Format::from_str_name(&format!("FORMAT_{v}")).map(|x| x as i32)),
        Slot::PhotoHdr => {
            e(po::PhotoHdr::from_str_name(&format!("PHOTO_HDR_{v}")).map(|x| x as i32))
        }
        Slot::Burst => {
            let (n, t) = v.split_once('/')?;
            Some(n.parse::<u32>().ok()? as f64 * 100.0 + t.parse::<u32>().ok()? as f64)
        }
        Slot::Fov => e(pb::Fov::from_str_name(&format!("FOV_{v}")).map(|x| x as i32)),
        Slot::Size => e(
            pb::Aspect::from_str_name(&format!("ASPECT_{}", v.replace(':', "_"))).map(|x| x as i32),
        ),
    }
}

/// The exposures a mode reads and writes: photo (and video, as the app writes both) for photo modes, video for the
/// others.
fn expo(photo: bool) -> &'static [T] {
    if photo {
        &[T::VideoExposure, T::PhotoExposure]
    } else {
        &[T::VideoExposure]
    }
}

/// The params slot `s` reads and writes (none for `Splice`, a `Settings` field).
pub fn types(s: Slot, photo: bool) -> &'static [T] {
    match s {
        Slot::Prog | Slot::Iso | Slot::Shutter => expo(photo),
        Slot::Ev => &[T::Ev],
        Slot::Wb => &[T::WhiteBalance],
        Slot::IsoTop => &[T::IsoMax],
        Slot::Gamma => &[T::Color],
        Slot::PanoExp => &[T::LensExposure],
        Slot::ILog => &[T::Log],
        Slot::Hdr => &[T::Hdr],
        Slot::Pure => &[T::LowLight],
        Slot::Res => &[T::Resolution],
        Slot::PhotoRes => &[T::PhotoResolution],
        Slot::Raw => &[T::Format],
        Slot::PhotoHdr => &[T::PhotoHdr],
        Slot::Aeb => &[T::Brackets],
        Slot::Timer => &[T::Timer],
        Slot::Accel => &[T::Speed],
        Slot::Burst => &[T::BurstCount, T::BurstSecs],
        Slot::Fov => &[T::Fov],
        Slot::Horizon => &[T::Horizon],
        Slot::Size => &[T::Aspect],
        Slot::P3 => &[T::P3],
        Slot::Dur => &[T::Duration],
        Slot::Splice => &[],
    }
}

/// The camera's value in slot `s` (`splice`: `Settings::stitching`).
pub fn get(s: Slot, o: &pb::Params, splice: bool, photo: bool) -> f64 {
    let x = if photo {
        o.photo_exposure.as_ref()
    } else {
        o.video_exposure.as_ref()
    };
    let x = x.cloned().unwrap_or_default();
    match s {
        Slot::Prog => x.program as f64,
        Slot::Iso => x.iso as f64,
        Slot::Shutter => x.shutter,
        Slot::Ev => o.ev as f64,
        Slot::Wb => o.white_balance as f64,
        Slot::IsoTop => o.iso_max as f64,
        Slot::Gamma => o.color as f64,
        Slot::PanoExp => o.lens_exposure as f64,
        Slot::ILog => o.log as u8 as f64,
        Slot::Hdr => o.hdr as u8 as f64,
        Slot::Pure => o.low_light as u8 as f64,
        Slot::Res => o.resolution as f64,
        Slot::PhotoRes => o.photo_resolution as f64,
        Slot::Raw => o.format as f64,
        Slot::PhotoHdr => o.photo_hdr as f64,
        Slot::Aeb => o.brackets as f64,
        Slot::Timer => o.timer as f64,
        Slot::Accel => o.speed as f64,
        Slot::Burst => o.burst_count as f64 * 100.0 + o.burst_secs as f64,
        Slot::Fov => o.fov as f64,
        Slot::Horizon => (o.horizon != 0) as u8 as f64,
        Slot::Size => o.aspect as f64,
        Slot::Splice => splice as u8 as f64,
        Slot::P3 => (o.p3 != 0) as u8 as f64,
        Slot::Dur => o.duration as f64,
    }
}

/// Equal on the wire: shutter speeds are doubles the camera may round.
pub fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(1.0)
}

/// A setting change as it goes out.
#[derive(Clone, Debug, PartialEq)]
pub enum Write {
    Photo(pb::WriteParams),
    Opts(pb::WriteSettings),
}

/// The message that sets slot `s` to `/CONF` value `v` in profile `fm`, from `cur` (the camera's values of the
/// mode: the exposure options go whole, with the other two values as they are). None for a read-only slot, an
/// unknown value, or an exposure value without the camera's exposure options (a default one would reset the
/// program and shutter).
pub fn write(s: Slot, v: &str, cur: &pb::Params, fm: pb::Profile, photo: bool) -> Option<Write> {
    let x = val(s, v).filter(|_| wired(s))?;
    if s == Slot::Splice {
        let o = pb::Settings {
            stitching: x != 0.0,
            ..Default::default()
        };
        return Some(Write::Opts(pb::WriteSettings {
            keys: vec![pb::Setting::Stitching as i32],
            settings: Some(o),
        }));
    }
    let mut o = pb::Params::default();
    let (i, u, b) = (x as i32, x as u32, x != 0.0);
    match s {
        Slot::Prog | Slot::Iso | Slot::Shutter => {
            let src = if photo {
                &cur.photo_exposure
            } else {
                &cur.video_exposure
            };
            let mut e = (*src)?;
            match s {
                Slot::Prog => e.program = i,
                Slot::Iso => e.iso = u,
                _ => e.shutter = x,
            }
            if photo {
                o.photo_exposure = Some(e);
            }
            o.video_exposure = Some(e);
        }
        Slot::Ev => o.ev = i,
        Slot::Wb => o.white_balance = u,
        Slot::IsoTop => o.iso_max = u,
        Slot::Gamma => o.color = i,
        Slot::PanoExp => o.lens_exposure = i,
        Slot::ILog => o.log = b,
        Slot::Hdr => o.hdr = b,
        Slot::Pure => o.low_light = b,
        Slot::Res => o.resolution = i,
        Slot::PhotoRes => o.photo_resolution = i,
        Slot::Raw => o.format = i,
        Slot::PhotoHdr => o.photo_hdr = i,
        Slot::Aeb => o.brackets = u,
        Slot::Timer => o.timer = u,
        Slot::Accel => o.speed = u,
        Slot::Burst => (o.burst_count, o.burst_secs) = (u / 100, u % 100),
        Slot::Fov => o.fov = i,
        Slot::Horizon => o.horizon = b as i32,
        Slot::Size => o.aspect = i,
        Slot::P3 => o.p3 = u,
        Slot::Splice | Slot::Dur => unreachable!("filtered above"),
    }
    Some(Write::Photo(pb::WriteParams {
        keys: types(s, photo).iter().map(|&t| t as i32).collect(),
        params: Some(o),
        profile: fm as i32,
    }))
}

/// A photo mode (else a video or live one).
pub fn photo(mode: &str) -> bool {
    mode.starts_with("PHOTO_")
}

/// The profile a `/CONF` mode's params live in.
pub fn func(mode: &str) -> Option<pb::Profile> {
    use pb::Profile as F;
    Some(match mode {
        "PHOTO_SINGLE" => F::PhotoSingle,
        "PHOTO_HDR" => F::PhotoHdr,
        "PHOTO_BURST" => F::PhotoBurst,
        "PHOTO_INTERVAL" => F::PhotoInterval,
        "PHOTO_AEB_NIGHT" => F::PhotoAebNight,
        "PHOTO_STARLAPSE" => F::PhotoStarlapse,
        "VIDEO_NORMAL" => F::VideoNormal,
        "VIDEO_BULLETTIME" => F::VideoBullettime,
        "VIDEO_TIMELAPSE" => F::VideoTimelapse,
        "VIDEO_HDR" => F::VideoHdr,
        "VIDEO_TIMESHIFT" => F::VideoTimeshift,
        "VIDEO_SUPER" => F::VideoSuper,
        "VIDEO_LOOPRECORDING" => F::VideoLooprecording,
        "VIDEO_SELFIE" => F::VideoSelfie,
        "VIDEO_PURE" => F::VideoPure,
        "VIDEO_LIVING_MODE_PANO" | "VIDEO_LIVING_MODE_WIDE" => F::VideoLive,
        "VIDEO_DASHCAM" => F::VideoDashcam,
        "VIDEO_GIMBAL" => F::VideoGimbal,
        _ => return None,
    })
}

/// The photo or video mode setting of a `/CONF` mode (`SET_OPTIONS` 40 / 41).
pub fn sub(mode: &str) -> Option<Sub> {
    use pb::{PhotoMode as P, VideoMode as V};
    Some(match mode {
        "VIDEO_LIVING_MODE_PANO" | "VIDEO_LIVING_MODE_WIDE" => Sub::Video(V::VideoLive),
        m if photo(m) => Sub::Photo(P::from_str_name(m)?),
        m => Sub::Video(V::from_str_name(m)?),
    })
}

/// The `/CONF` name of the camera's mode (video mode `VIDEO_NONE`: a photo mode). The live modes share one video mode;
/// it reads as the 360 one.
pub fn current(o: &pb::Settings) -> Option<String> {
    use pb::VideoMode as V;
    Some(match o.video_mode() {
        V::VideoNone => o.photo_mode().as_str_name().into(),
        V::VideoLive => "VIDEO_LIVING_MODE_PANO".into(),
        v => v.as_str_name().into(),
    })
}

/// What the shutter does in a mode (app_features.md §3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shot {
    Photo(pb::take_photo::Kind),
    Rec(pb::RecordMode),
    Lapse(pb::LapseMode),
    /// The live modes: the preview, streamed by this app.
    None,
}

pub fn shot(mode: &str) -> Shot {
    use pb::{LapseMode as L, RecordMode as C, take_photo::Kind as P};
    match mode {
        "PHOTO_SINGLE" | "PHOTO_HDR" => Shot::Photo(P::Single),
        "PHOTO_BURST" => Shot::Photo(P::Burst),
        "PHOTO_AEB_NIGHT" => Shot::Photo(P::AebNight),
        "PHOTO_INTERVAL" => Shot::Lapse(L::Interval),
        "PHOTO_STARLAPSE" => Shot::Lapse(L::Starlapse),
        "VIDEO_TIMELAPSE" => Shot::Lapse(L::Video),
        "VIDEO_BULLETTIME" => Shot::Rec(C::Bullettime),
        "VIDEO_HDR" => Shot::Rec(C::Hdr),
        "VIDEO_TIMESHIFT" => Shot::Rec(C::Timeshift),
        "VIDEO_SUPER" => Shot::Rec(C::Super),
        "VIDEO_LOOPRECORDING" => Shot::Rec(C::Looprecording),
        "VIDEO_SELFIE" => Shot::Rec(C::Selfie),
        "VIDEO_PURE" => Shot::Rec(C::Pure),
        "VIDEO_DASHCAM" => Shot::Rec(C::Dashcam),
        "VIDEO_GIMBAL" => Shot::Rec(C::Gimbal),
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
    /// `ev` as it is), white balance in K, the shutter in seconds.
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

    /// The messages, byte for byte from the field numbers of `params.proto` / `settings.proto`.
    #[test]
    fn messages() {
        let cur = pb::Params {
            video_exposure: Some(pb::params::Exposure {
                program: 3,
                iso: 400,
                shutter: 0.01,
            }),
            ..Default::default()
        };
        let w = |s, v, photo| {
            let fm = if photo {
                pb::Profile::PhotoSingle
            } else {
                pb::Profile::VideoNormal
            };
            match write(s, v, &cur, fm, photo).unwrap() {
                Write::Photo(m) => m.encode_to_vec(),
                Write::Opts(m) => m.encode_to_vec(),
            }
        };
        // keys [7], value {ev (sint32) −3 → zigzag 5}, profile 7
        assert_eq!(w(Slot::Ev, "-0.3", false), hex("08 07 12 02 38 05 18 07"));
        // keys [39], value {white_balance 5000}, profile 7
        assert_eq!(
            w(Slot::Wb, "5000", false),
            hex("08 27 12 04 b8 02 88 27 18 07")
        );
        // keys [31], value {resolution 10}
        assert_eq!(
            w(Slot::Res, "5760_2880_30", false),
            hex("08 1f 12 03 f8 01 0a 18 07")
        );
        // ISO in a video mode: the video exposure whole {program 3, iso 800, shutter 0.01}, key [21]
        let mut e = hex("08 15 12 11 aa 01 0e 08 03 10 a0 06 19");
        e.extend(0.01f64.to_le_bytes());
        e.extend(hex("18 07"));
        assert_eq!(w(Slot::Iso, "800", false), e);
        // a photo mode writes the photo exposure too, starting from the photo values, keys [21, 20]; without the
        // camera's photo values nothing (a default would reset the program and shutter)
        assert!(write(Slot::Iso, "200", &cur, pb::Profile::PhotoSingle, true).is_none());
        let still = pb::Params {
            photo_exposure: Some(Default::default()),
            ..cur
        };
        match write(Slot::Iso, "200", &still, pb::Profile::PhotoSingle, true) {
            Some(Write::Photo(m)) => assert_eq!(
                m.encode_to_vec(),
                hex("08 15 08 14 12 0c a2 01 03 10 c8 01 aa 01 03 10 c8 01 18 06")
            ),
            x => panic!("{x:?}"),
        }
        // keys [77], value {photo_hdr 2}, profile 6
        assert_eq!(
            w(Slot::PhotoHdr, "AEB", true),
            hex("08 4d 12 03 e8 04 02 18 06")
        );
        // SET_OPTIONS keys [135], value {stitching (field 135) true}
        assert_eq!(w(Slot::Splice, "false", true), hex("08 87 01 12 00"));
        assert_eq!(
            w(Slot::Splice, "true", true),
            hex("08 87 01 12 03 b8 08 01")
        );
        // burst 15 in 3 s: keys [44, 51]
        assert_eq!(
            w(Slot::Burst, "15/3", true),
            hex("08 2c 08 33 12 06 e0 02 0f 98 03 03 18 06")
        );
        // keys [86], value {p3 1}
        assert_eq!(
            w(Slot::P3, "true", false),
            hex("08 56 12 03 b0 05 01 18 07")
        );
        assert!(write(Slot::Dur, "60", &cur, pb::Profile::VideoNormal, false).is_none());
        assert!(write(Slot::Res, "1_2_3", &cur, pb::Profile::VideoNormal, false).is_none());
    }

    #[test]
    fn modes() {
        assert_eq!(func("VIDEO_NORMAL"), Some(pb::Profile::VideoNormal));
        assert_eq!(func("PHOTO_STARLAPSE"), Some(pb::Profile::PhotoStarlapse));
        assert!(matches!(
            sub("VIDEO_GIMBAL"),
            Some(Sub::Video(pb::VideoMode::VideoGimbal))
        ));
        assert!(matches!(
            sub("PHOTO_BURST"),
            Some(Sub::Photo(pb::PhotoMode::PhotoBurst))
        ));
        let mut o = pb::Settings {
            video_mode: pb::VideoMode::VideoNone as i32,
            ..Default::default()
        };
        assert_eq!(current(&o).as_deref(), Some("PHOTO_SINGLE"));
        o.video_mode = pb::VideoMode::VideoDashcam as i32;
        assert_eq!(current(&o).as_deref(), Some("VIDEO_DASHCAM"));
        o.video_mode = pb::VideoMode::VideoPure as i32;
        assert_eq!(current(&o).as_deref(), Some("VIDEO_PURE"));
        assert_eq!(shot("VIDEO_PURE"), Shot::Rec(pb::RecordMode::Pure));
        assert_eq!(shot("PHOTO_INTERVAL"), Shot::Lapse(pb::LapseMode::Interval));
        assert_eq!(shot("VIDEO_LIVING_MODE_PANO"), Shot::None);
        assert_eq!(
            shot("PHOTO_SINGLE"),
            Shot::Photo(pb::take_photo::Kind::Single)
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
