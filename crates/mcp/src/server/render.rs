//! The GPU tools: `catalog`, `render_still` and `export`, and the render jobs in the job tools. A call is checked in
//! this order before the GPU starts: its arguments (a feature agents can't use yet, or the other tool's field, said by
//! name), every id and range against the backend's [`Catalog`], the host (its jobs file its own, not ending, no timed
//! recording of its own: a crash mid-render would lose that recording's stop), the source (a readable X5 recording,
//! its main files), the backend's check of the source, and once the GPU is up its limits. Then the job is reserved:
//! the record saved before its `.part` exists, the `.part` made new at the first free name and its identity saved, the
//! job kept in `runs`; it goes to its lane at once when the GPU is up, else it is parked for the host's start task.
//! Every reader reads the record ([`crate::renders`]): its end, or the poller's snapshot of the run, never the
//! backend. A done still answers with a preview image made from its file ([`preview`]).

use super::*;
use crate::{
    host::{Live, LiveSt, Parked, Preview},
    job::{Jobs, Which},
    render::{
        Cap, Catalog, GpuInfo, Kind as RKind, Lane, Named, Output, Plan, Preset, Range, Render,
        Req, RunSt, Start, View as RView, Want,
    },
    renders::{ENDED, End, Prog as RProg, RJob, Snap},
};
use media::part::{Id, Space};
use std::{
    collections::BTreeMap,
    fmt::Display,
    io::{self, Read},
    sync::atomic::Ordering::Relaxed,
    time::SystemTime,
};

/// Unfinished render jobs a host takes at most.
const UNFINISHED: usize = 16;

/// Free space an export leaves beside its estimate, and a still.
const SPARE: u64 = 2 << 30;
const SPARE_STILL: u64 = 256 << 20;

/// A still's long edge at most (zune decodes up to 16384), a 360 still's least width, a flat still's least side.
const STILL_MAX: u32 = 8192;
const STILL_360: u32 = 512;
const STILL_FLAT: u32 = 16;

/// A name's bytes at most.
const NAME: usize = 120;

/// How long render_still waits by default and at most, s.
const WAIT: f64 = 50.0;
const WAIT_MAX: f64 = 300.0;

/// The preview: its long edge at most, its bytes at most, the qualities tried in turn, the previews kept.
const PREVIEW: u32 = 1568;
const PREVIEW_BYTES: usize = 1_000_000;
const QUALITY: [u8; 3] = [85, 70, 55];
const PREVIEWS: usize = 8;

/// What a call answers once the host ends.
const SHUT: &str = "the host is shutting down";

/// What a host whose jobs are in memory only answers.
const MEMORY: &str =
    "another x5desk host holds the jobs file: GPU tools need it to clean up after a crash";

/// The fields only one of the tools takes.
const STILL_ONLY: [&str; 3] = ["time_s", "jpeg_quality", "wait_s"];
const EXPORT_ONLY: [&str; 9] = [
    "preset",
    "start_s",
    "end_s",
    "codec",
    "ten_bit",
    "mbps",
    "qp",
    "fps",
    "allow_frozen",
];

/// "360" (the number 360 too) or "flat"
#[derive(Deserialize, JsonSchema)]
#[serde(untagged)]
pub(super) enum OutArg {
    N(u32),
    S(String),
}

/// a number (30, 29.97) or a catalog fps label ("source", "29.97")
#[derive(Deserialize, JsonSchema)]
#[serde(untagged)]
pub(super) enum FpsArg {
    N(f64),
    S(String),
}

/// A flat output's projection and direction, degrees.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ViewArgs {
    /// a catalog projection id or alias (default: the first, perspective)
    projection: Option<String>,
    /// right, -180 to 180 (0: the centre of the 360 output)
    yaw: Option<f32>,
    /// up, -90 to 90
    pitch: Option<f32>,
    /// -180 to 180
    roll: Option<f32>,
    /// horizontal, within the projection's range (catalog projections)
    fov: Option<f32>,
}

/// Both tools' arguments, the other tool's unset.
#[derive(Default)]
struct Ask {
    path: String,
    output: Option<OutArg>,
    width: Option<u32>,
    height: Option<u32>,
    view: Option<ViewArgs>,
    look: Option<String>,
    look_strength: Option<f32>,
    adjust: Option<BTreeMap<String, f32>>,
    stabilisation: Option<String>,
    smoothing_s: Option<f64>,
    raft: Option<bool>,
    colour_match: Option<bool>,
    subdir: Option<String>,
    name: Option<String>,
    time_s: Option<f64>,
    jpeg_quality: Option<u8>,
    wait_s: Option<f64>,
    preset: Option<String>,
    start_s: Option<f64>,
    end_s: Option<f64>,
    codec: Option<String>,
    ten_bit: Option<bool>,
    mbps: Option<f32>,
    qp: Option<u8>,
    fps: Option<FpsArg>,
    allow_frozen: Option<bool>,
}

/// A render tool's arguments: the fields both take, written into each (serde ignores `deny_unknown_fields` on a
/// struct with a `flatten`), then its own; and its [`Ask`].
macro_rules! render_args {
    ($(#[$m:meta])* $n:ident { $($(#[$fm:meta])* $f:ident: $t:ty,)* }) => {
        $(#[$m])*
        #[derive(Deserialize, JsonSchema)]
        #[serde(deny_unknown_fields)]
        pub(super) struct $n {
            /// absolute path of an X5 recording (.insv; an .lrv or _10_ file finds the .insv beside it) under a read
            /// root
            path: String,
            /// "360" (the number 360 too) or "flat"; default: a still 360, an export its preset's
            output: Option<OutArg>,
            /// a still: 360 a multiple of 8, 512-8192 (height width/2), flat with height, 16-8192 a side; an export:
            /// catalog outputs' lists
            width: Option<u32>,
            height: Option<u32>,
            /// flat only: projection and direction (default: the projection's, catalog projections)
            view: Option<ViewArgs>,
            /// a catalog look id, alias or name
            look: Option<String>,
            /// 0-1 (default 1)
            look_strength: Option<f32>,
            /// catalog adjust ids to values, e.g. {"exposure": 0.5}
            adjust: Option<BTreeMap<String, f32>>,
            /// off | free | horizon | direction (default: horizon with IMU data, else off)
            #[serde(alias = "stabilization")]
            stabilisation: Option<String>,
            /// 0-2 (default 0.5)
            smoothing_s: Option<f64>,
            /// optical-flow seams (default true, unless the GPU's RAFT is unavailable)
            raft: Option<bool>,
            /// the lenses' colours matched (default true)
            #[serde(alias = "color_match")]
            colour_match: Option<bool>,
            /// a folder in the agent folder (relative, no `..`; made if missing)
            subdir: Option<String>,
            /// the file name without extension (a safe name, not a camera file's; numbered when taken)
            name: Option<String>,
            $($(#[$fm])* $f: $t,)*
        }

        impl From<$n> for Ask {
            fn from(a: $n) -> Ask {
                Ask {
                    path: a.path,
                    output: a.output,
                    width: a.width,
                    height: a.height,
                    view: a.view,
                    look: a.look,
                    look_strength: a.look_strength,
                    adjust: a.adjust,
                    stabilisation: a.stabilisation,
                    smoothing_s: a.smoothing_s,
                    raft: a.raft,
                    colour_match: a.colour_match,
                    subdir: a.subdir,
                    name: a.name,
                    $($f: a.$f,)*
                    ..Ask::default()
                }
            }
        }
    };
}

render_args! {
    /// `render_still` arguments.
    StillArgs {
        /// the frame shown at this time, s on the clip (default 0); refused outside [0, duration)
        time_s: Option<f64>,
        /// 50-100 (default 92)
        jpeg_quality: Option<u8>,
        /// how long this call waits, s (default 50, at most 300; 0: answer the job at once)
        wait_s: Option<f64>,
    }
}

render_args! {
    /// `export` arguments.
    ExportArgs {
        /// a catalog preset id (default: catalog's default_preset); the fields below override it
        preset: Option<String>,
        /// in, s on the clip (default its start); at least one frame before end_s; refused outside the clip
        start_s: Option<f64>,
        /// out, s on the clip (default its end)
        end_s: Option<f64>,
        /// "h265" | "h264"
        codec: Option<String>,
        /// 10-bit (H.265 only)
        ten_bit: Option<bool>,
        /// constant bit rate, Mbit/s (1-400), or a constant QP (10-45): not both
        mbps: Option<f32>,
        qp: Option<u8>,
        /// "source" (default), or 24, 25, 29.97, 30, 60 (a number or its label)
        fps: Option<FpsArg>,
        /// export even when pictures can't be decoded (they repeat the one before); default false
        allow_frozen: Option<bool>,
    }
}

fn lane(k: RKind) -> Lane {
    match k {
        RKind::Still => Lane::Still,
        RKind::Export => Lane::Export,
    }
}

/// A list for a message.
fn ids<'a>(v: impl IntoIterator<Item = &'a String>) -> String {
    v.into_iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ")
}

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1e9)
}

fn bits(ten: bool) -> &'static str {
    if ten { " 10-bit" } else { "" }
}

/// The backend's call `f`, its panic an error.
pub(crate) fn caught<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f))
        .map_err(|p| format!("the GPU backend panicked: {}", panicked(&*p)))
}

/// A blocking backend call on tokio's blocking threads, its panic an error.
pub(crate) async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> T + Send + 'static,
) -> Result<T, String> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| match e.try_into_panic() {
            Ok(p) => format!("the GPU backend panicked: {}", panicked(&*p)),
            Err(e) => e.to_string(),
        })
}

/// The backend's GPU state, asked inline (a panic reads failed for good).
pub(crate) fn gpu_st(r: &dyn Render) -> GpuSt {
    caught(|| r.gpu()).unwrap_or_else(|why| GpuSt::Failed {
        why,
        step: None,
        retry_in: None,
    })
}

/// What the GPU tools answer while the GPU can't be used; none while it is up, off, starting, or its retry is due.
fn unusable(st: &GpuSt) -> Option<String> {
    match st {
        GpuSt::Failed {
            retry_in: Some(d), ..
        } if d.is_zero() => None,
        GpuSt::Failed {
            why,
            step,
            retry_in,
        } => {
            let step = step.map_or(String::new(), |s| format!(" (self-test step {s})"));
            let again = retry_in.map_or(String::new(), |d| {
                format!("; tried again in {} s", d.as_secs().max(1))
            });
            Some(format!(
                "GPU tools are unavailable: {why}{step}{again}; the other tools work."
            ))
        }
        GpuSt::Dead(why) => Some(format!(
            "the GPU device was lost ({why}); GPU tools are off until this MCP server restarts (Claude Code: /mcp, \
             reconnect)"
        )),
        _ => None,
    }
}

/// `x` within range `r`.
fn within<T: Into<f64> + Display + Copy>(what: &str, x: T, r: &Range) -> Result<(), String> {
    let v: f64 = x.into();
    match v.is_finite() && v >= r.min && v <= r.max {
        true => Ok(()),
        false => Err(format!("{what} {x}: {} to {}", r.min, r.max)),
    }
}

/// The choice of `v` that `s` names by its id, display name or an alias, any case.
fn named(v: &[Named], s: &str) -> Option<String> {
    let s = s.trim();
    let is = |x: &str| x.eq_ignore_ascii_case(s);
    (v.iter())
        .find(|n| is(&n.id) || n.name.as_deref().is_some_and(is) || n.aliases.iter().any(|a| is(a)))
        .map(|n| n.id.clone())
}

/// The arguments of `kind`'s tool. A field of a feature agents can't use yet, or of the other tool, is said so before
/// serde's own message, which then comes only for real typos.
fn parse(kind: RKind, a: JsonObject, c: &Catalog) -> Result<Ask, String> {
    let (other, tool) = match kind {
        RKind::Still => (&EXPORT_ONLY[..], "export"),
        RKind::Export => (&STILL_ONLY[..], "render_still"),
    };
    for k in a.keys() {
        let is = |x: &String| x.eq_ignore_ascii_case(k);
        let later = (c.later.iter())
            .find(|l| is(&l.id) || l.aliases.iter().any(is) || l.effects.iter().any(is));
        if let Some(l) = later {
            return Err(format!(
                "`{k}` is not available to agents yet (catalog: later: {})",
                l.id
            ));
        }
        if other.contains(&k.as_str()) {
            return Err(format!("`{k}` is for {tool} only"));
        }
    }
    Ok(match kind {
        RKind::Still => args::<StillArgs>(a)?.into(),
        RKind::Export => args::<ExportArgs>(a)?.into(),
    })
}

fn output(o: &OutArg) -> Result<Output, String> {
    match o {
        OutArg::N(360) => Ok(Output::Pano),
        OutArg::S(s) if s.trim() == "360" => Ok(Output::Pano),
        OutArg::S(s) if s.trim().eq_ignore_ascii_case("flat") => Ok(Output::Flat),
        OutArg::N(n) => Err(format!("output {n}: \"360\" or \"flat\"")),
        OutArg::S(s) => Err(format!("output {s:?}: \"360\" or \"flat\"")),
    }
}

/// A flat view checked against the catalog's projections (the first when none is named).
fn view(v: &ViewArgs, c: &Catalog) -> Result<RView, String> {
    let p = match &v.projection {
        Some(s) => {
            let is = |x: &String| x.eq_ignore_ascii_case(s.trim());
            (c.projections.iter())
                .find(|p| is(&p.id) || p.aliases.iter().any(is))
                .ok_or_else(|| {
                    let all = c.projections.iter().map(|p| &p.id);
                    format!("projection {s:?}: one of {}", ids(all))
                })?
        }
        None => c
            .projections
            .first()
            .ok_or("this x5desk has no flat views")?,
    };
    let r = |a: f32, b: f32| Range {
        min: a as f64,
        max: b as f64,
        default: None,
    };
    if let Some(f) = v.fov {
        within(&format!("fov ({})", p.id), f, &r(p.fov[0], p.fov[1]))?
    }
    for (n, x, m) in [
        ("yaw", v.yaw, 180.0),
        ("pitch", v.pitch, 90.0),
        ("roll", v.roll, 180.0),
    ] {
        if let Some(x) = x {
            within(n, x, &r(-m, m))?
        }
    }
    Ok(RView {
        projection: v.projection.as_ref().map(|_| p.id.clone()),
        yaw: v.yaw,
        pitch: v.pitch,
        roll: v.roll,
        fov: v.fov,
    })
}

/// A still's size as asked: 360 a multiple of 8, 512-8192 wide, its height half that; flat both sides, 16-8192.
fn still_size(a: &Ask, out: Output, c: &Catalog) -> Result<Option<[u32; 2]>, String> {
    match out {
        Output::Pano => {
            let Some(w) = a.width.or(a.height.and_then(|h| h.checked_mul(2))) else {
                return Ok(None);
            };
            if w % 8 != 0 || !(STILL_360..=STILL_MAX).contains(&w) {
                return Err(format!(
                    "width {w}: a 360 still is a multiple of 8, {STILL_360} to {STILL_MAX} wide"
                ));
            }
            if let Some(h) = a.height
                && h.checked_mul(2) != Some(w)
            {
                return Err(format!(
                    "height {h}: a 360 still is half as high as wide ({})",
                    w / 2
                ));
            }
            Ok(Some([w, w / 2]))
        }
        Output::Flat => match (a.width, a.height) {
            (Some(w), Some(h)) => {
                for (n, x) in [("width", w), ("height", h)] {
                    if !(STILL_FLAT..=STILL_MAX).contains(&x) {
                        return Err(format!(
                            "{n} {x}: a flat still is {STILL_FLAT} to {STILL_MAX} a side"
                        ));
                    }
                }
                Ok(Some([w, h]))
            }
            (None, None) => Ok(None),
            _ => Err(format!(
                "a flat still takes width and height together (default {} × {})",
                c.still.flat[0], c.still.flat[1]
            )),
        },
    }
}

/// An export's size as asked, from the catalog's lists, and the size it renders at (its preset's when not asked).
type Sizes = (Option<[u32; 2]>, Option<[u32; 2]>);
fn video_size(a: &Ask, out: Output, p: Option<&Preset>, c: &Catalog) -> Result<Sizes, String> {
    let of = p.filter(|p| p.output == out).map(|p| [p.width, p.height]);
    let asked = match out {
        Output::Pano => {
            let w = a.width.or(a.height.and_then(|h| h.checked_mul(2)));
            if let Some(w) = w
                && !c.pano_widths.contains(&w)
            {
                let all: Vec<String> = c.pano_widths.iter().map(u32::to_string).collect();
                return Err(format!(
                    "width {w}: a 360 export is {} wide (catalog outputs)",
                    ids(&all)
                ));
            }
            if let (Some(h), Some(w)) = (a.height, w)
                && h.checked_mul(2) != Some(w)
            {
                return Err(format!(
                    "height {h}: a 360 export is half as high as wide ({})",
                    w / 2
                ));
            }
            w.map(|w| [w, w / 2])
        }
        Output::Flat => match (a.width, a.height) {
            (Some(w), Some(h)) if c.flat_sizes.contains(&[w, h]) => Some([w, h]),
            (Some(w), Some(h)) => {
                let all: Vec<String> = (c.flat_sizes.iter())
                    .map(|[a, b]| format!("{a} × {b}"))
                    .collect();
                return Err(format!(
                    "{w} × {h}: a flat export is {} (catalog outputs)",
                    ids(&all)
                ));
            }
            (None, None) => None,
            _ => {
                return Err(
                    "a flat export takes width and height together (catalog outputs)".into(),
                );
            }
        },
    };
    Ok((asked, asked.or(of)))
}

/// `a` checked against catalog `c` (an export starts from preset `def` unless it names one): every id and range, the
/// sizes, the rules between fields; the ids as the catalog spells them.
fn want(kind: RKind, a: &Ask, c: &Catalog, def: &str) -> Result<Want, String> {
    let still = kind == RKind::Still;
    let preset = match &a.preset {
        Some(p) => Some(
            (c.presets.iter())
                .find(|x| x.id.eq_ignore_ascii_case(p.trim()))
                .ok_or_else(|| {
                    let all = c.presets.iter().map(|x| &x.id);
                    format!("preset {p:?}: not a catalog preset ({})", ids(all))
                })?,
        ),
        None if !still => c.presets.iter().find(|x| x.id == def),
        None => None,
    };
    let output = a.output.as_ref().map(output).transpose()?;
    let out = output.or(preset.map(|p| p.output)).unwrap_or(Output::Pano);
    let mut w = Want {
        preset: preset.map(|p| p.id.clone()),
        output,
        ..Want::default()
    };
    if let Some(v) = &a.view {
        if out != Output::Flat {
            return Err("view: for output flat only (360 is the whole sphere)".into());
        }
        w.view = Some(view(v, c)?);
    }
    if let Some(l) = &a.look {
        w.look = Some(named(&c.looks, l).ok_or_else(|| {
            let all = c.looks.iter().map(|x| &x.id);
            format!("look {l:?}: not a catalog look ({})", ids(all))
        })?);
    }
    if let Some(x) = a.look_strength {
        within("look_strength", x, &c.look_strength)?
    }
    for (k, &x) in a.adjust.iter().flatten() {
        let j = c.adjust.iter().find(|j| j.id == *k).ok_or_else(|| {
            let all = c.adjust.iter().map(|x| &x.id);
            format!("adjust {k:?}: not a catalog adjust id ({})", ids(all))
        })?;
        let r = Range {
            min: j.min as f64,
            max: j.max as f64,
            default: None,
        };
        within(&format!("adjust {k}"), x, &r)?;
        w.adjust.insert(j.id.clone(), x);
    }
    if let Some(s) = &a.stabilisation {
        let m = c
            .stabilisation
            .iter()
            .find(|x| x.eq_ignore_ascii_case(s.trim()));
        w.stabilisation = Some(
            m.cloned()
                .ok_or_else(|| format!("stabilisation {s:?}: one of {}", ids(&c.stabilisation)))?,
        );
    }
    if let Some(x) = a.smoothing_s {
        within("smoothing_s", x, &c.smoothing_s)?
    }
    (w.look_strength, w.smoothing_s) = (a.look_strength, a.smoothing_s);
    (w.raft, w.colour_match) = (a.raft, a.colour_match);
    if still {
        [w.width, w.height] = match still_size(a, out, c)? {
            Some([x, y]) => [Some(x), Some(y)],
            None => [None, None],
        };
        if let Some(t) = a.time_s
            && !(t.is_finite() && t >= 0.0)
        {
            return Err(format!("time_s {t}: a time on the clip, from 0 s"));
        }
        if let Some(q) = a.jpeg_quality {
            within("jpeg_quality", q, &c.still.jpeg_quality)?
        }
        (w.time_s, w.jpeg_quality) = (a.time_s, a.jpeg_quality);
        return Ok(w);
    }
    for (n, t) in [("start_s", a.start_s), ("end_s", a.end_s)] {
        if let Some(t) = t
            && !(t.is_finite() && t >= 0.0)
        {
            return Err(format!("{n} {t}: a time on the clip, from 0 s"));
        }
    }
    if let (Some(s), Some(e)) = (a.start_s, a.end_s)
        && s >= e
    {
        return Err(format!("start_s {s} is not before end_s {e}"));
    }
    if a.mbps.is_some() && a.qp.is_some() {
        return Err("mbps or qp, not both".into());
    }
    if let Some(x) = a.mbps {
        within("mbps", x, &c.mbps)?
    }
    if let Some(x) = a.qp {
        within("qp", x, &c.qp)?
    }
    let given = match &a.codec {
        Some(x) => Some(
            (c.codecs.iter())
                .find(|k| k.id.eq_ignore_ascii_case(x.trim()))
                .ok_or_else(|| {
                    let all = c.codecs.iter().map(|k| &k.id);
                    format!("codec {x:?}: one of {}", ids(all))
                })?,
        ),
        None => None,
    };
    let codec = given.or_else(|| preset.and_then(|p| c.codecs.iter().find(|k| k.id == p.codec)));
    if let Some(k) = codec.filter(|k| a.ten_bit == Some(true) && !k.ten_bit) {
        let all = c.codecs.iter().filter(|k| k.ten_bit).map(|k| &k.id);
        return Err(format!(
            "ten_bit: {} has no 10-bit ({} has)",
            k.id,
            ids(all)
        ));
    }
    let (asked, size) = video_size(a, out, preset, c)?;
    if let (Some([x, y]), Some(k)) = (size, codec)
        && x.max(y) > k.max_px
    {
        return Err(format!(
            "{x} × {y}: {} goes up to {} a side",
            k.id, k.max_px
        ));
    }
    [w.width, w.height] = match asked {
        Some([x, y]) => [Some(x), Some(y)],
        None => [None, None],
    };
    w.fps = match &a.fps {
        None => None,
        Some(f) => {
            let hit = match f {
                FpsArg::N(x) => {
                    (c.fps.iter()).find(|l| l.parse::<f64>().is_ok_and(|v| (v - x).abs() <= 0.01))
                }
                FpsArg::S(s) => c.fps.iter().find(|l| l.eq_ignore_ascii_case(s.trim())),
            };
            let shown = match f {
                FpsArg::N(x) => x.to_string(),
                FpsArg::S(s) => format!("{s:?}"),
            };
            Some(
                hit.cloned()
                    .ok_or_else(|| format!("fps {shown}: one of {}", ids(&c.fps)))?,
            )
        }
    };
    (w.start_s, w.end_s, w.codec) = (a.start_s, a.end_s, given.map(|k| k.id.clone()));
    (w.ten_bit, w.mbps, w.qp) = (a.ten_bit, a.mbps, a.qp);
    w.allow_frozen = a.allow_frozen.unwrap_or(false);
    Ok(w)
}

/// Whether `s` is a name in the camera's pattern (`VID_`, `IMG_`, `LRV_`, `PRO_` + 8 digits `_` 6 digits `_` 2
/// digits `_` digits, any case): a render under it would be taken for a camera file (`local`, `camera_delete`).
fn camera(s: &str) -> bool {
    let u = s.to_ascii_uppercase();
    let pro = u.strip_prefix("PRO_");
    let r = (["VID_", "IMG_", "LRV_"].iter())
        .find_map(|p| pro.unwrap_or(&u).strip_prefix(p))
        .or(pro);
    let n = |x: &str, k: usize| {
        !x.is_empty() && (k == 0 || x.len() == k) && x.bytes().all(|b| b.is_ascii_digit())
    };
    r.is_some_and(|r| {
        matches!(r.split('_').collect::<Vec<_>>()[..],
            [a, b, c, d] if n(a, 8) && n(b, 6) && n(c, 2) && n(d, 0))
    })
}

/// The stem `s` names for a `kind` file: its extension stripped when it is the kind's, another refused; a safe file
/// name of at most [`NAME`] bytes, without a leading dot, not a `.part`, not a camera file's.
fn name(s: &str, kind: RKind) -> Result<String, String> {
    let (ext, what) = match kind {
        RKind::Still => ("jpg", "a still"),
        RKind::Export => ("mp4", "an export"),
    };
    let stem = match Path::new(s).extension().and_then(|e| e.to_str()) {
        Some(e)
            if e.eq_ignore_ascii_case(ext)
                || kind == RKind::Still && e.eq_ignore_ascii_case("jpeg") =>
        {
            &s[..s.len() - e.len() - 1]
        }
        Some(e) if e.bytes().any(|b| b.is_ascii_alphabetic()) => {
            return Err(format!("name {s:?}: {what} is a .{ext}, not a .{e}"));
        }
        _ => s,
    };
    if stem.is_empty() || stem.starts_with('.') {
        return Err(format!(
            "name {s:?}: give a name that does not begin with a dot"
        ));
    }
    if stem.len() > NAME {
        return Err(format!("name {s:?}: {NAME} bytes at most"));
    }
    let l = stem.to_ascii_lowercase();
    if !album::safe(stem) || !album::safe(&format!("{stem}.{ext}")) || l.ends_with(".part") {
        return Err(format!("name {s:?}: not a safe file name"));
    }
    if camera(stem) {
        return Err(format!("name {s:?}: a camera file's name: pick another"));
    }
    Ok(stem.into())
}

/// Why a timed recording of this host keeps renders off now, if one does (step 3): its stop is this host's timer,
/// which a crash mid-render would lose.
fn timed(sh: &Shared, s: &State) -> Option<String> {
    if let Some(u) = s.rec.as_ref().and_then(|r| r.until) {
        let t = SystemTime::now() + u.saturating_duration_since(Instant::now());
        return Some(format!(
            "the camera records until {} (capture_record, seconds): render after it ends",
            utc(t)
        ));
    }
    (sh.timed.load(Relaxed)).then(|| {
        "a timed capture_record start is under way: render after its recording ends".into()
    })
}

/// A timed `capture_record` start under way ([`Shared::timed`]) until dropped.
pub(super) struct Timed<'a>(&'a Shared);

impl Drop for Timed<'_> {
    fn drop(&mut self) {
        self.0.timed.store(false, Relaxed)
    }
}

/// Marks a timed recording's start under way, unless this host renders or a run of it may still be in a driver call
/// (one settled stuck): renders and timed recordings exclude each other, both marked under `st`.
pub(super) fn timed_start(sh: &Shared) -> Result<Timed<'_>, String> {
    let _s = sh.st.lock().unwrap();
    if sh.wedged.load(Relaxed) {
        return Err("a render of this host is stuck in a GPU driver call: a recording with seconds waits until this \
                    MCP server restarts, since its stop is this host's timer, which a crash there would lose; \
                    capture_record start without seconds records meanwhile"
            .into());
    }
    match sh.rendering.load(Relaxed) {
        0 => {
            sh.timed.store(true, Relaxed);
            Ok(Timed(sh))
        }
        n => Err(format!(
            "this host renders ({n} unfinished render job{}, job_list): a recording with seconds waits until they \
             end, since its stop is this host's timer, which a crash mid-render would lose; capture_record start \
             without seconds records meanwhile",
            if n == 1 { "" } else { "s" }
        )),
    }
}

/// One more render of this host until dropped, unless kept ([`Raised::keep`]): raised under `st`, refused while a
/// timed recording of this host runs or starts.
struct Raised<'a>(&'a Shared);

impl Raised<'_> {
    /// The job is in `runs`: its end lowers the count.
    fn keep(self) {
        std::mem::forget(self)
    }
}

impl Drop for Raised<'_> {
    fn drop(&mut self) {
        self.0.rendering.fetch_sub(1, Relaxed);
    }
}

fn raise(sh: &Shared) -> Result<Raised<'_>, String> {
    let s = sh.st.lock().unwrap();
    if let Some(w) = timed(sh, &s) {
        return Err(w);
    }
    sh.rendering.fetch_add(1, Relaxed);
    Ok(Raised(sh))
}

/// The checks of the host (step 3): its jobs file its own (a crash's `.part` is cleaned up through it), not ending,
/// no timed recording of its own.
fn host_ok(sh: &Shared) -> Result<(), String> {
    let (mem, closing) = {
        let j = sh.jobs.lock().unwrap();
        (j.r.file().is_none(), j.r.closing || sh.ct.is_cancelled())
    };
    match (mem, closing) {
        (true, false) => return Err(MEMORY.into()),
        (_, true) => return Err(SHUT.into()),
        _ => {}
    }
    match timed(sh, &sh.st.lock().unwrap()) {
        Some(w) => Err(w),
        None => Ok(()),
    }
}

/// The source's main files in lens order (step 4): `path` readable under the read roots and UTF-8, an X5 recording
/// (no photo, no `.mp4`), its `.insv`s paired by name, each readable too (a sibling linked out of a root is not).
fn source(sh: &Shared, path: &str) -> Result<Vec<PathBuf>, String> {
    let roots = sh.read_roots();
    let p = paths::readable(path, &roots)?;
    let utf8 = |p: &Path| {
        p.to_str()
            .map(String::from)
            .ok_or_else(|| format!("{}: the path is no UTF-8", p.display()))
    };
    utf8(&p)?;
    match cam::work::ext(&p).as_str() {
        "insp" | "jpg" | "dng" => {
            return Err(format!(
                "{path}: a photo; render_still and export take X5 recordings (.insv): photos later"
            ));
        }
        "mp4" => {
            return Err(format!(
                "{path}: an .mp4 has no Insta360 trailer (x5desk's own exports neither): give the camera's .insv"
            ));
        }
        _ => {}
    }
    let main: Vec<PathBuf> = (cam::work::pair(&p).main.into_iter())
        .filter(|f| cam::work::ext(f) == "insv")
        .collect();
    if main.is_empty() {
        return Err(format!(
            "{path}: an .lrv is the camera's proxy: album_download without proxy fetches the .insv"
        ));
    }
    main.iter()
        .map(|f| paths::readable(&utf8(f)?, &roots))
        .collect()
}

/// The device's limits for plan `p` on `lane` (step 6): the lane usable, RAFT there if it renders with it, the size
/// within the texture limit, the source's tracks among what it decodes, the codec, size and rate among what it
/// encodes.
fn fit(i: &GpuInfo, p: &Plan, lane: Lane) -> Result<(), String> {
    if i.wedged.contains(&lane) {
        return Err(format!(
            "the GPU's {} lane is stuck: this MCP server must restart to use it",
            lane.name()
        ));
    }
    if p.raft
        && let Some(w) = &i.raft
    {
        return Err(format!(
            "RAFT (optical-flow seams) is unavailable on this GPU: {w}; pass raft: false"
        ));
    }
    let [w, h] = p.size;
    if w.max(h) > i.max_texture {
        return Err(format!(
            "{w} × {h} is over this GPU's largest texture, {} a side",
            i.max_texture
        ));
    }
    let fits = |c: &Cap, w: u32, h: u32| {
        (c.min[0]..=c.max[0]).contains(&w) && (c.min[1]..=c.max[1]).contains(&h)
    };
    if !i.decode.is_empty() {
        for t in &p.tracks {
            let [x, y] = t.size;
            if !(i.decode.iter())
                .any(|c| c.codec == t.codec && c.ten_bit == t.ten_bit && fits(c, x, y))
            {
                return Err(format!(
                    "this GPU cannot decode the source's {}{} at {x} × {y}",
                    t.codec,
                    bits(t.ten_bit)
                ));
            }
        }
    }
    let Some(k) = &p.codec else { return Ok(()) };
    let caps: Vec<&Cap> = (i.encode.iter())
        .filter(|c| c.codec == *k && c.ten_bit == p.ten_bit)
        .collect();
    let Some(c) = caps.iter().find(|c| fits(c, w, h)) else {
        return Err(match caps.first() {
            None if i.encode.is_empty() => "this GPU encodes no video: stills only".into(),
            None => {
                let all: Vec<String> = (i.encode.iter())
                    .map(|c| format!("{}{}", c.codec, bits(c.ten_bit)))
                    .collect();
                format!(
                    "this GPU does not encode {k}{}: it encodes {}",
                    bits(p.ten_bit),
                    ids(&all)
                )
            }
            Some(c) => format!(
                "{w} × {h} is outside what this GPU encodes in {k}{}: {} × {} to {} × {}",
                bits(p.ten_bit),
                c.min[0],
                c.min[1],
                c.max[0],
                c.max[1]
            ),
        });
    };
    let has = |m: &str| c.rate.is_empty() || c.rate.iter().any(|r| r == m);
    if let Some(m) = p.mbps {
        if !has("cbr") {
            return Err(format!(
                "this GPU has no constant bit rate for {k}: give qp"
            ));
        }
        if let Some(x) = c.max_mbps
            && m > x
        {
            return Err(format!(
                "mbps {m}: this GPU encodes {k} at up to {x} Mbit/s"
            ));
        }
    }
    if let Some(q) = p.qp {
        if !has("cqp") {
            return Err(format!("this GPU has no constant QP for {k}: give mbps"));
        }
        if let Some([a, b]) = c.qp
            && !(a..=b).contains(&q)
        {
            return Err(format!("qp {q}: this GPU takes {a} to {b} for {k}"));
        }
    }
    Ok(())
}

/// Step 6: the device when it is up, its limits checked; none while it is off or starting (the job parks, and is
/// checked when it is submitted); why GPU tools can't be used else.
fn device(st: &GpuSt, p: &Plan, lane: Lane) -> Result<Option<GpuInfo>, String> {
    if let Some(e) = unusable(st) {
        return Err(e);
    }
    match st {
        GpuSt::Ready(i) => fit(i, p, lane).map(|_| Some(i.clone())),
        _ => Ok(None),
    }
}

/// Exports' default preset: the user's (the app's `preset=`) when it is a catalog id, not read under `--sim` or
/// `dir` (no file of the user's is); else the catalog's.
fn default_preset(sh: &Shared, c: &Catalog) -> String {
    let user = sh
        .downloads
        .then(|| crate::host::app_settings().2)
        .flatten();
    user.filter(|p| c.presets.iter().any(|x| x.id == *p))
        .unwrap_or_else(|| c.default_preset.clone())
}

/// What the device accepts of catalog `c`: its codecs (10-bit and the long edge as it encodes them), the video sizes
/// one of them encodes within the texture limit, the rates within its limits.
fn accepted(c: &Catalog, i: &GpuInfo) -> Catalog {
    let caps = |k: &String| -> Vec<&Cap> { i.encode.iter().filter(|x| x.codec == *k).collect() };
    let codecs: Vec<_> = (c.codecs.iter())
        .filter(|k| !caps(&k.id).is_empty())
        .map(|k| crate::render::Codec {
            id: k.id.clone(),
            ten_bit: k.ten_bit && caps(&k.id).iter().any(|x| x.ten_bit),
            max_px: (caps(&k.id).iter().map(|x| x.max[0].max(x.max[1])).max())
                .map_or(k.max_px, |m| m.min(k.max_px)),
        })
        .collect();
    let ok = |w: u32, h: u32| {
        w.max(h) <= i.max_texture
            && (codecs.iter()).any(|k| {
                w.max(h) <= k.max_px
                    && caps(&k.id).iter().any(|x| {
                        (x.min[0]..=x.max[0]).contains(&w) && (x.min[1]..=x.max[1]).contains(&h)
                    })
            })
    };
    let mut a = c.clone();
    a.pano_widths.retain(|&w| ok(w, w / 2));
    a.flat_sizes.retain(|&[w, h]| ok(w, h));
    if let Some(m) = (i.encode.iter())
        .filter_map(|x| x.max_mbps)
        .reduce(f32::max)
    {
        a.mbps.max = a.mbps.max.min(m as f64)
    }
    let qp: Vec<[u8; 2]> = i.encode.iter().filter_map(|x| x.qp).collect();
    if !qp.is_empty() {
        a.qp.min = a.qp.min.max(qp.iter().map(|q| q[0]).min().unwrap() as f64);
        a.qp.max = a.qp.max.min(qp.iter().map(|q| q[1]).max().unwrap() as f64);
    }
    a.codecs = codecs;
    a
}

/// The catalog as the tool shows it: `c` in the layout agents read, with mcp's own rules, the sources, the preview,
/// the limits and the GPU (`raft`: why the GPU has none).
fn shown_catalog(c: &Catalog, def: &str, gpu: Value, raft: Option<&str>) -> Value {
    let r = |r: &Range| json!({ "min": r.min, "max": r.max });
    let projections: Vec<Value> = (c.projections.iter())
        .map(|p| {
            json!({
                "id": p.id,
                "aliases": p.aliases,
                "fov": p.fov,
                "default": { "yaw": p.yaw, "pitch": p.pitch, "roll": p.roll, "fov": p.default_fov },
            })
        })
        .collect();
    let mut rf = json!({
        "default": c.raft && raft.is_none(),
        "note": "optical-flow seams; the server's first RAFT job compiles kernels (10-46 s)",
    });
    if let Some(w) = raft {
        rf["available"] = false.into();
        rf["why"] = w.into();
    }
    json!({
        "version": 1,
        "presets": c.presets,
        "default_preset": def,
        "outputs": {
            "360": {
                "video_widths": c.pano_widths,
                "still": format!("width a multiple of 8, {STILL_360}-{STILL_MAX}; height width/2"),
            },
            "flat": {
                "video_sizes": c.flat_sizes,
                "still": format!("{STILL_FLAT}-{STILL_MAX} a side"),
            },
        },
        "codecs": c.codecs,
        "rate": { "mbps": r(&c.mbps), "qp": r(&c.qp), "rule": "mbps or qp, not both" },
        "fps": { "labels": c.fps, "note": "a number works too: 30, 29.97" },
        "raft": rf,
        "colour_match": { "default": c.colour_match },
        "stabilisation": {
            "modes": c.stabilisation,
            "default": "horizon when the recording has IMU data, else off",
            "smoothing_s": c.smoothing_s,
        },
        "looks": c.looks,
        "look_strength": c.look_strength,
        "adjust": c.adjust,
        "projections": projections,
        "view_angles": {
            "yaw": [-180, 180], "pitch": [-90, 90], "roll": [-180, 180],
            "unit": "degrees: yaw right, pitch up; yaw 0 faces the centre of the 360 output (x = width/2)",
        },
        "still": {
            "default_output": "360",
            "default_size": { "360": c.still.pano_width, "flat": c.still.flat },
            "max_long_edge": STILL_MAX,
            "jpeg_quality": c.still.jpeg_quality,
            "note": "preset-free",
        },
        "export": {
            "default": "the whole clip at default_preset",
            "audio": "the source's AAC, copied",
            "allow_frozen": "false: a picture that can't be decoded fails the export",
        },
        "sources": {
            "accepted": "an X5 recording: .insv with its Insta360 trailer and both lenses (one file, or its _00_/_10_ \
                         pair), under a read root; an .lrv or a _10_ file renders from the .insv beside it",
            "refused": {
                ".lrv alone": "the camera's proxy: album_download without proxy fetches the .insv",
                ".mp4": "no Insta360 trailer (x5desk's own exports too)",
                ".insp/.jpg/.dng": "photos: later",
                "camera items": "album_download them first",
            },
        },
        "preview": { "long_edge": PREVIEW, "max_bytes": PREVIEW_BYTES, "per_result": 1 },
        "limits": {
            "unfinished_jobs": UNFINISHED,
            "free_space": "an export needs its estimated size plus 2 GB free, after this server's other unfinished \
                           jobs",
        },
        "later": c.later,
        "gpu": gpu,
    })
}

/// A done still's preview: `dst`, while it is the job's file (`id`; opened without following a link), decoded,
/// shrunk to a long edge of at most [`PREVIEW`] and encoded at the first of [`QUALITY`] under [`PREVIEW_BYTES`].
pub(crate) fn preview(dst: &Path, id: &Id) -> Result<Preview, String> {
    let gone = || format!("{}: not the still's file any more", paths::shown(dst));
    if !media::part::is(dst, id) {
        return Err(gone());
    }
    let mut o = std::fs::File::options();
    o.read(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut o, libc::O_NOFOLLOW);
    let mut b = vec![];
    (o.open(dst).and_then(|mut f| f.read_to_end(&mut b)))
        .map_err(|e| format!("{}: {e}", paths::shown(dst)))?;
    let mut px = vec![];
    let (w, h) = (media::mjpeg::Mjpeg::default().rgba(&b, &mut px))
        .map_err(|e| format!("{}: {e}", paths::shown(dst)))?;
    let (w, h) = (w as u32, h as u32);
    let (nw, nh) = media::scale::fit(w, h, PREVIEW);
    let small = match (nw, nh) == (w, h) {
        true => px,
        false => media::scale::area(&px, w, h, nw, nh),
    };
    let mut last = None;
    for q in QUALITY {
        let j = media::jpeg::encode(&small, nw, nh, q, None)?;
        let fits = j.len() <= PREVIEW_BYTES;
        last = Some(Preview {
            jpg: j.into(),
            w: nw,
            h: nh,
            q,
            src: [w, h],
        });
        if fits {
            break;
        }
    }
    last.ok_or_else(gone)
}

/// Keeps a done still's preview, the newest [`PREVIEWS`].
pub(crate) fn keep_preview(sh: &Shared, h: &str, p: Preview) {
    let mut v = sh.previews.lock().unwrap();
    v.retain(|x| x.0 != h);
    if v.len() >= PREVIEWS {
        v.pop_front();
    }
    v.push_back((h.to_string(), p))
}

/// burn's cubecl base for the GPU's start: the user's `<cache>/x5desk/cubecl-mcp` (apart from the GUI's slots,
/// warm across sessions); under `--sim` and `dir` (the test hook) a dir of the test runs', as the app's.
fn cubecl(sh: &Shared) -> Option<PathBuf> {
    match sh.downloads {
        true => {
            cam::dirs::xdg("XDG_CACHE_HOME", ".cache").map(|d| d.join("x5desk").join("cubecl-mcp"))
        }
        false => cam::dirs::mine(
            std::env::var_os("XDG_RUNTIME_DIR")
                .map_or_else(std::env::temp_dir, PathBuf::from)
                .join("x5desk-cubecl"),
        ),
    }
}

/// The GPU's log lines into the host's log (the backend keeps the newest start's sink: no host kept alive by it).
fn sink(sh: &Arc<Shared>) -> Arc<dyn Fn(String) + Send + Sync> {
    let w = Arc::downgrade(sh);
    Arc::new(move |l: String| {
        if let Some(sh) = w.upgrade() {
            sh.log.line(&format!("gpu: {l}"))
        }
    })
}

/// What comes after a reservation: the submit (the GPU is up), the host's start task, or nothing (one runs).
enum Next {
    Submit(Box<Parked>),
    Start,
    Wait,
}

/// Reserves a job of `kind` in `dir` (steps 1-4 of the reservation, under `jobs` → `q` → `runs`): the host not
/// ending, at most [`UNFINISHED`] unfinished jobs, the free space after what this host's other jobs on the volume
/// still need (their estimates less what they wrote); the record saved before its `.part` exists; the `.part` made
/// new at the first free name of `stem` (numbered past a file, a `.part`, a dangling link, an unfinished download or
/// render there); its identity saved; the job kept in `runs`, which the end's sweep reads under the same lock. Its
/// handle, and what comes next.
fn reserve(
    sh: &Shared,
    dir: &Path,
    stem: &str,
    sp: Space,
    req: Req,
    plan: Plan,
    ready: bool,
) -> Result<(String, Next), String> {
    let kind = req.kind;
    let mut jobs = sh.jobs.lock().unwrap();
    let q = sh.q.lock().unwrap();
    let mut runs = sh.runs.lock().unwrap();
    if jobs.r.closing || sh.ct.is_cancelled() {
        return Err(SHUT.into());
    }
    if runs.len() >= UNFINISHED {
        return Err(format!(
            "{UNFINISHED} render jobs of this server are unfinished: job_wait or job_cancel one first"
        ));
    }
    let spare = match kind {
        RKind::Export => SPARE,
        RKind::Still => SPARE_STILL,
    };
    // what the others still need: what they wrote has left `free` already (a `.part`, or the file once published)
    let written = |l: &Live| {
        let part = media::part::of(&l.dst);
        (media::part::len_if(&part, &l.id)).or_else(|| media::part::len_if(&l.dst, &l.id))
    };
    let others: u64 = (runs.values())
        .filter(|l| l.vol == sp.vol)
        .map(|l| l.bytes.saturating_sub(written(l).unwrap_or(0)))
        .sum();
    let (left, need) = (
        sp.free.saturating_sub(others),
        plan.bytes.saturating_add(spare),
    );
    if left < need {
        return Err(format!(
            "not enough space in {}: needs about {}, {} free after this server's other jobs",
            paths::shown(dir),
            gb(need),
            gb(left)
        ));
    }
    let file = format!("{stem}.{}", plan.ext);
    let at = |n: u32| dir.join(paths::numbered(&file, n));
    let taken = |p: &Path| {
        !paths::free(p)
            || q.v.iter().any(|d| d.dst == p && d.st != dl::St::Done)
            || runs.values().any(|l| l.dst == p)
    };
    let free = |from: u32| (from..).find(|&n| !taken(&at(n))).unwrap();
    let mut n = free(1);
    let h = jobs.handle();
    jobs.r.add(
        h.clone(),
        RJob {
            h: String::new(),
            kind,
            at: SystemTime::now(),
            src: req.files.clone(),
            dst: at(n),
            part: None,
            lane: lane(kind),
            settings: plan.settings.clone(),
            frames: plan.frames,
            bytes: plan.bytes,
            cancelled: false,
            end: None,
            live: None,
        },
    );
    let i = jobs.r.v.len() - 1;
    let unkept = |jobs: &mut Jobs, e: io::Error| {
        jobs.r.v.pop();
        let _ = jobs.r.save();
        format!("can't keep the job record: {e}")
    };
    // no `.part` exists without its record on disk
    if let Err(e) = jobs.r.save() {
        return Err(unkept(&mut jobs, e));
    }
    let part = loop {
        match media::part::create(&jobs.r.v[i].dst) {
            Ok(p) => break p,
            // another writer took the name since
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                n = free(n + 1);
                jobs.r.v[i].dst = at(n);
                if let Err(e) = jobs.r.save() {
                    return Err(unkept(&mut jobs, e));
                }
            }
            Err(e) => {
                let p = paths::shown(&media::part::of(&jobs.r.v[i].dst));
                jobs.r.v.pop();
                let _ = jobs.r.save();
                return Err(format!("{p}: {e}"));
            }
        }
    };
    let dst = jobs.r.v[i].dst.clone();
    jobs.r.v[i].part = Some(part.id);
    if let Err(e) = jobs.r.save() {
        media::part::remove_if(&media::part::of(&dst), &part.id);
        return Err(unkept(&mut jobs, e));
    }
    let (id, bytes) = (part.id, plan.bytes);
    let p = Box::new(Parked { part, req, plan });
    let (st, next) = match ready {
        true => (LiveSt::Submitting, Next::Submit(p)),
        false => match sh.starting.swap(true, Relaxed) {
            false => (LiveSt::Parked(p), Next::Start),
            true => (LiveSt::Parked(p), Next::Wait),
        },
    };
    let l = Live {
        st,
        dst,
        id,
        vol: sp.vol,
        bytes,
        cancel: false,
        stuck: None,
    };
    runs.insert(h.clone(), l);
    Ok((h, next))
}

/// Submits job `h` to its lane, the device's limits checked first (it may have parked before the GPU was up; one
/// that took RAFT by default on a GPU without it is checked again without). One cancelled before (job_cancel, the
/// host's end) ends cancelled and never reaches its lane; else its run is kept in `runs` (read as queued until the
/// poller reads it), cancelled at once when a cancel came meanwhile; a refusal or a panic fails the job, its `.part`
/// settled.
async fn submit(sh: &Arc<Shared>, h: String, p: Box<Parked>, i: &GpuInfo) -> Result<(), String> {
    let fail = |why: String| {
        sh.end_render(&h, End::failed(&why));
        Err(why)
    };
    let Some(r) = sh.render.clone() else {
        return fail(NO_GPU.into());
    };
    let Parked {
        part,
        mut req,
        mut plan,
    } = *p;
    let lane = lane(req.kind);
    if plan.raft && i.raft.is_some() && req.want.raft.is_none() {
        req.want.raft = Some(false);
        let r2 = r.clone();
        let (q, again) = match blocking(move || (r2.check(&req), req)).await {
            Ok((p, q)) => (q, p),
            Err(e) => {
                drop(part);
                return fail(e);
            }
        };
        (req, plan) = match again {
            Ok(p) => (q, p),
            Err(e) => {
                drop(part);
                return fail(e);
            }
        };
        let mut jobs = sh.jobs.lock().unwrap();
        if let Some(x) = jobs.r.v.iter_mut().find(|x| x.h == h) {
            (x.settings, x.frames) = (plan.settings.clone(), plan.frames)
        }
        if let Err(e) = jobs.r.save() {
            sh.log.line(&format!("the renders file: {e}"))
        }
    }
    if let Err(e) = fit(i, &plan, lane) {
        drop(part);
        return fail(e);
    }
    let (dst, cancel, closing) = {
        let jobs = sh.jobs.lock().unwrap();
        match sh.runs.lock().unwrap().get(&h) {
            Some(l) => (l.dst.clone(), l.cancel, jobs.r.closing),
            None => return Err(SHUT.into()),
        }
    };
    // cancelled while it waited (job_cancel, the host's end): never handed to its lane
    if cancel {
        drop(part);
        let why = if closing { ENDED } else { "" };
        sh.end_render(&h, End::cancelled(why));
        return Ok(());
    }
    let got = blocking(move || r.submit(req, &plan, &dst, part, lane)).await;
    let run = match got.and_then(|x| x) {
        Ok(run) => run,
        Err(e) => return fail(e),
    };
    let cancel = {
        let mut jobs = sh.jobs.lock().unwrap();
        let mut runs = sh.runs.lock().unwrap();
        // queued in its lane from here on, until the poller reads it
        if let Some(x) = jobs.r.v.iter_mut().find(|x| x.h == h)
            && x.live.is_none()
        {
            x.live = Some(Snap::after(None, RunSt::Queued, Instant::now(), x.at))
        }
        match runs.get_mut(&h) {
            Some(l) => {
                l.st = LiveSt::Run(run.clone());
                l.cancel || jobs.r.closing || sh.ct.is_cancelled()
            }
            None => true,
        }
    };
    if cancel {
        let _ = caught(|| run.cancel());
    }
    Ok(())
}

/// The parked jobs taken out to submit (`take`), or whether any is parked; with none, [`Shared::starting`] cleared
/// under the same lock (a job parked after it starts a task of its own).
fn waiting(sh: &Shared, take: bool) -> Vec<(String, Option<Box<Parked>>)> {
    let mut runs = sh.runs.lock().unwrap();
    let v: Vec<_> = (runs.iter_mut())
        .filter(|(_, l)| matches!(l.st, LiveSt::Parked(_)))
        .map(|(h, l)| match take {
            true => match std::mem::replace(&mut l.st, LiveSt::Submitting) {
                LiveSt::Parked(p) => (h.clone(), Some(p)),
                _ => unreachable!("parked"),
            },
            false => (h.clone(), None),
        })
        .collect();
    if v.is_empty() {
        sh.starting.store(false, Relaxed)
    }
    v
}

/// The host's start task (one at a time, [`Shared::starting`]): the GPU started on a blocking thread (a start under
/// way is waited for), then the jobs parked meanwhile submitted, or failed with why it did not start; again while
/// more were parked. A halted start (the host ends) leaves them to the end.
async fn starter(sh: Arc<Shared>) {
    let Some(r) = sh.render.clone() else { return };
    loop {
        let s = Start {
            cubecl: cubecl(&sh),
            child: sh.gpu_child.clone(),
            log: sink(&sh),
            halt: sh.halt.clone(),
        };
        let r2 = r.clone();
        let got = blocking(move || r2.start(&s)).await.and_then(|x| x);
        if sh.halt.load(Relaxed) {
            return;
        }
        let parked: Vec<(String, Box<Parked>)> = (waiting(&sh, true).into_iter())
            .filter_map(|(h, p)| Some((h, p?)))
            .collect();
        if parked.is_empty() {
            return;
        }
        match &got {
            Ok(i) => {
                let _ = sh.gpu_at.set(SystemTime::now());
                for (h, p) in parked {
                    let _ = submit(&sh, h, p, i).await;
                }
            }
            Err(e) => {
                sh.log.line(&format!("the GPU did not start: {e}"));
                let why = unusable(&gpu_st(&*r)).unwrap_or_else(|| {
                    format!("GPU tools are unavailable: {e}; the other tools work.")
                });
                for (h, p) in parked {
                    drop(p);
                    sh.end_render(&h, End::failed(&why));
                }
            }
        }
        // another round only for jobs parked meanwhile
        if waiting(&sh, false).is_empty() {
            return;
        }
    }
}

/// What an export's first answer notes: its length, frames and size, a time guess from this server's last export of
/// that size and RAFT setting, and how to take a part when no range was given.
fn note(jobs: &Jobs, h: &str, a: &Ask, p: &Plan) -> String {
    let [w, hh] = p.size;
    let mut s = format!("{:.2} s, {} frames at {w} × {hh}", p.secs, p.frames);
    if a.start_s.is_none() && a.end_s.is_none() {
        s = format!("the whole clip: {s} (start_s/end_s: a part)")
    }
    let keys = ["width", "height", "raft"];
    let same = |r: &&RJob| {
        r.kind == RKind::Export
            && r.h != h
            && (keys.iter()).all(|k| !p.settings[k].is_null() && r.settings[k] == p.settings[k])
    };
    let fps = (jobs.r.v.iter().rev().filter(same)).find_map(|r| {
        (r.end.as_ref())
            .filter(|e| e.state == "done" && e.secs > 0.0 && e.frames > 0)
            .map(|e| e.frames as f64 / e.secs)
    });
    if let Some(f) = fps {
        s += &format!(
            "; about {:.0} s at this server's last {f:.1} frames/s for that size",
            p.frames as f64 / f
        )
    }
    s + "; job_wait follows it"
}

/// What job_cancel answers when the run has not stopped within 3 s.
const STOPPING: &str = "stopping at its next frame (the first RAFT kernel compile can take up to a minute); \
                        job_wait follows it";

/// The render record handle `h` names.
fn at(jobs: &Jobs, h: &str) -> Result<usize, String> {
    match jobs.which(h)? {
        Which::Render(i) => Ok(i),
        Which::Dl(_) => Err(format!("{} is a download, no render job", h.trim())),
    }
}

/// One decimal.
fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// A render job as job_status and job_wait show it: its size, a still's time and an export's length from its settings
/// at the top too, and once finished the run's time (`secs`; `elapsed_s` says it too, and the age before).
pub(super) fn render_dto(r: &RJob, p: &RProg) -> Value {
    let note = match p.state {
        "done" if r.cancelled => Some("finished before the cancel"),
        "cancelled" if !p.err.is_empty() => Some(p.err.as_str()),
        _ => None,
    };
    let mut v = json!({
        "job": r.h,
        "kind": r.kind.name(),
        "state": p.state,
        "finished": p.finished(),
        "phase": p.phase,
        "path": paths::shown(&r.dst),
        "source": r.src.iter().map(|s| paths::shown(s)).collect::<Vec<_>>(),
        "frames_done": p.done,
        "frames_total": p.total,
        "fps": p.fps.map(r1),
        "eta_s": p.eta.map(r1),
        "elapsed_s": p.secs.map(r1),
        "secs": p.secs.filter(|_| p.finished()).map(r1),
        "width": r.settings["width"],
        "height": r.settings["height"],
        "bytes": p.bytes,
        "estimated_bytes": r.bytes,
        "settings": r.settings,
        "created": utc(r.at),
        "error": (p.state == "failed").then(|| p.err.clone()),
        "cancel_requested": r.cancelled,
        "note": note,
    });
    let at = |k: &str| r.settings[k].as_f64();
    match r.kind {
        RKind::Export => {
            v["concealed_frames"] = p.concealed.into();
            let d = (at("start_s").zip(at("end_s"))).map(|(a, b)| ((b - a) * 1e3).round() / 1e3);
            v["duration_s"] = d.into()
        }
        RKind::Still => {
            v["concealed"] = (p.concealed > 0).into();
            v["time_s"] = at("time_s").into()
        }
    }
    v
}

/// A render job's row in job_list: no settings (job_status has them).
pub(super) fn row(r: &RJob, p: &RProg) -> Value {
    json!({
        "job": r.h,
        "kind": r.kind.name(),
        "state": p.state,
        "path": paths::shown(&r.dst),
        "frames_done": p.done,
        "frames_total": p.total,
        "created": utc(r.at),
    })
}

/// How long no frame has come: since the last one, or since the job was made while it has none.
fn quiet(r: &RJob) -> Duration {
    match &r.live {
        Some(s) => s.last_frame_at.elapsed(),
        None => SystemTime::now().duration_since(r.at).unwrap_or_default(),
    }
}

/// What job_wait adds to the frames done while none comes: a quarter of what is left to 1 every 5 s, so a client that
/// resets its timeout on progress keeps waiting through the GPU's start, the setup and a first RAFT compile.
fn creep(quiet: Duration) -> f64 {
    (1.0 - 0.75f64.powi((quiet.as_secs() / 5).min(1000) as i32)).min(0.99)
}

/// A progress notification's message: the frames, or before the first one what the job does.
fn told(k: RKind, p: &RProg) -> String {
    let k = k.name();
    let what = match p.phase {
        Some("frames" | "finishing") if p.done > 0 => {
            let mut m = format!("{} of {} frames", p.done, p.total);
            if let Some(f) = p.fps.filter(|f| *f > 0.0) {
                m += &format!(", {f:.1} frames/s")
            }
            match p.eta.filter(|e| e.is_finite() && *e > 0.0) {
                Some(e) if p.phase == Some("frames") => m += &format!(", {e:.0} s left"),
                _ if p.phase == Some("finishing") => m += ", finishing the file",
                _ => {}
            }
            m
        }
        Some("gpu_starting") => "starting the GPU".into(),
        Some("waiting") => "waiting for the renders before it".into(),
        Some("setting_up") => "setting up".into(),
        Some("compiling") => {
            "compiling the optical-flow kernels (the first RAFT render: up to a minute)".into()
        }
        Some("frames") => "the first frame".into(),
        Some(_) => "finishing the file".into(),
        None => p.state.into(),
    };
    format!("{k}: {what}")
}

/// A job reserved and on its way: its handle, what the call asked and what the backend checked.
struct Made {
    h: String,
    ask: Ask,
    plan: Plan,
}

impl Conn {
    /// `catalog`: the backend's catalog, narrowed to what the device takes once it is up, with mcp's rules and the
    /// GPU's state; never starts the GPU.
    pub(super) fn catalog(&self) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let r = sh.render.as_ref().ok_or(NO_GPU)?;
        let c = caught(|| r.catalog())?;
        let def = default_preset(sh, &c);
        let st = gpu_st(&**r);
        let (c, raft) = match &st {
            GpuSt::Ready(i) => (accepted(&c, i), i.raft.clone()),
            _ => (c, None),
        };
        let gpu = gpu_value(st, sh.gpu_at.get().copied());
        ok(shown_catalog(&c, &def, gpu, raft.as_deref()))
    }

    /// Checks a render call in order (nothing of it starts the GPU), then reserves its job and submits it when the GPU
    /// is up, or parks it for the host's start task (started here when none runs).
    async fn make(&self, kind: RKind, a: JsonObject) -> Result<Made, String> {
        let sh = self.host.sh.clone();
        let r = sh.render.clone().ok_or(NO_GPU)?;
        // 1. the arguments
        let c = caught(|| r.catalog())?;
        let ask = parse(kind, a, &c)?;
        // 2. against the catalog
        let mut want = want(kind, &ask, &c, &default_preset(&sh, &c))?;
        let stem = ask.name.as_deref().map(|n| name(n, kind)).transpose()?;
        let dir = paths::sub(&sh.agent, ask.subdir.as_deref().unwrap_or(""))?;
        // 3. the host
        host_ok(&sh)?;
        // 4. the source
        let files = source(&sh, &ask.path)?;
        // 5. the backend's check; a GPU known to have no RAFT renders without it unless asked to
        let st = gpu_st(&*r);
        if let GpuSt::Ready(i) = &st
            && i.raft.is_some()
            && want.raft.is_none()
        {
            want.raft = Some(false)
        }
        let req = Req { kind, files, want };
        let r2 = r.clone();
        let (plan, req) = blocking(move || (r2.check(&req), req)).await?;
        let plan = plan?;
        // 6. the device, once it is up
        let info = device(&st, &plan, lane(kind))?;
        // 7. the destination, then the reservation
        let dir = paths::make(&sh.agent, &dir)?;
        if dir.to_str().is_none() {
            return Err(format!("{}: the path is no UTF-8", dir.display()));
        }
        // a record knows its file by the identity `create` saw, which FAT and exFAT change (macOS: as it is written)
        if media::part::fat(&dir) {
            return Err(format!(
                "{}: on FAT or exFAT, whose file numbers can change while a file is written: x5desk can't \
                 keep track of renders there",
                paths::shown(&dir)
            ));
        }
        let sp = media::part::avail(&dir).map_err(|e| format!("{}: {e}", paths::shown(&dir)))?;
        let raised = raise(&sh)?;
        let stem = stem.unwrap_or_else(|| plan.stem.clone());
        let (h, next) = reserve(&sh, &dir, &stem, sp, req, plan.clone(), info.is_some())?;
        raised.keep();
        crate::host::poll(&sh);
        sh.log
            .line(&format!("{} reserved {} {h}", self.client, kind.name()));
        match (next, info) {
            (Next::Submit(p), Some(i)) => {
                // its own task: a call dropped meanwhile leaves no job half submitted
                let (s2, h2) = (sh.clone(), h.clone());
                let t = tokio::spawn(async move { submit(&s2, h2, p, &i).await });
                let r = t.await.map_err(|e| e.to_string()).and_then(|r| r);
                r.map_err(|e| format!("{} {h} failed: {e}", kind.name()))?
            }
            (Next::Start, _) => drop(tokio::spawn(starter(sh.clone()))),
            _ => {}
        }
        Ok(Made { h, ask, plan })
    }

    /// `render_still`: the job made, then followed up to `wait_s`; done, the file and its preview image.
    pub(super) async fn render_still(
        &self,
        a: JsonObject,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<CallToolResult, String> {
        let m = self.make(RKind::Still, a).await?;
        let wait = m.ask.wait_s.unwrap_or(WAIT);
        let wait = Duration::from_secs_f64(match wait.is_finite() {
            true => wait.clamp(0.0, WAIT_MAX),
            false => WAIT,
        });
        // from here on the call's cancel leaves the still running (job_cancel stops it)
        let mut v = self.follow(&m.h, wait, ctx).await?;
        match v["state"].as_str() {
            Some("done") => self.answer(&m.h, v).await,
            Some("failed") => Err(format!(
                "still {} failed: {}",
                m.h,
                v["error"].as_str().unwrap_or("")
            )),
            _ if v["finished"] == true => ok(v),
            _ => {
                v["note"] = "still rendering: job_wait answers with the image".into();
                ok(v)
            }
        }
    }

    /// `export`: the job made, answered at once with its length, frames and size.
    pub(super) async fn export(&self, a: JsonObject) -> Result<CallToolResult, String> {
        let m = self.make(RKind::Export, a).await?;
        let mut v = self.job(&m.h)?;
        if v["finished"] != true {
            v["note"] = note(&self.sh().jobs.lock().unwrap(), &m.h, &m.ask, &m.plan).into();
        }
        ok(v)
    }

    /// A render job's DTO once it is finished or after `wait`. With a progress token, at most one notification a
    /// second: frames done plus [`creep`], out of the frames + 1.
    async fn follow(
        &self,
        h: &str,
        wait: Duration,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<Value, String> {
        let sh = self.sh();
        let (t0, tok) = (Instant::now(), ctx.meta.get_progress_token());
        let (mut sent, mut last) = (t0, -1.0);
        loop {
            let (p, k, q, closing) = {
                let jobs = sh.jobs.lock().unwrap();
                let i = at(&jobs, h)?;
                let r = &jobs.r.v[i];
                (jobs.r.prog(i), r.kind, quiet(r), jobs.r.closing)
            };
            // the hosting client went: its renders stop, and the front's drain is not held up
            if closing && !p.finished() {
                return Err(SHUT.into());
            }
            if p.finished() || t0.elapsed() >= wait {
                let mut v = self.job(h)?;
                v["waited_s"] = json!(r1(t0.elapsed().as_secs_f64()));
                return Ok(v);
            }
            let x = p.done as f64 + creep(q);
            if let Some(t) = &tok
                && sent.elapsed() >= Duration::from_secs(1)
                && x > last
            {
                (sent, last) = (Instant::now(), x);
                let n = ProgressNotificationParam::new(t.clone(), x)
                    .with_total(p.total as f64 + 1.0)
                    .with_message(told(k, &p));
                let _ = ctx.peer.notify_progress(n).await;
            }
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(200)) => {}
                _ = ctx.ct.cancelled() => return Err("cancelled".into()),
                _ = sh.ct.cancelled() => return Err(SHUT.into()),
            }
        }
    }

    /// job_wait on a render job: its state once finished or after `wait`; a done still with its image.
    pub(super) async fn render_wait(
        &self,
        h: &str,
        wait: Duration,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<CallToolResult, String> {
        let v = self.follow(h, wait, ctx).await?;
        self.answer(h, v).await
    }

    /// A render job's answer: a done still's JSON in the first text block and its preview image after it (as
    /// album_thumbnail answers, no structured content a client might prefer over the image), else structured.
    async fn answer(&self, h: &str, mut v: Value) -> Result<CallToolResult, String> {
        if v["kind"] != "still" || v["state"] != "done" {
            return ok(v);
        }
        let (h, dst, id) = {
            let jobs = self.sh().jobs.lock().unwrap();
            let r = &jobs.r.v[at(&jobs, h)?];
            (r.h.clone(), r.dst.clone(), r.part)
        };
        let p = match id {
            Some(id) => self.preview(&h, dst, id).await,
            None => Err("no identity recorded".into()),
        };
        let mut c = vec![];
        match p {
            Ok(p) => {
                v["width"] = p.src[0].into();
                v["height"] = p.src[1].into();
                v["preview"] =
                    json!({ "width": p.w, "height": p.h, "bytes": p.jpg.len(), "quality": p.q });
                c.push(ContentBlock::image(b64(&p.jpg), "image/jpeg"))
            }
            Err(e) => v["preview_error"] = e.into(),
        }
        c.insert(0, ContentBlock::text(v.to_string()));
        Ok(CallToolResult::success(c))
    }

    /// Still `h`'s preview: kept, or made from its file while that is the job's (`id`).
    async fn preview(&self, h: &str, dst: PathBuf, id: Id) -> Result<Preview, String> {
        let sh = self.sh();
        if !media::part::is(&dst, &id) {
            return Err(format!(
                "{}: not the still's file any more",
                paths::shown(&dst)
            ));
        }
        let kept = (sh.previews.lock().unwrap().iter())
            .find(|x| x.0 == h)
            .map(|x| x.1.clone());
        if let Some(p) = kept {
            return Ok(p);
        }
        let p = blocking(move || preview(&dst, &id)).await??;
        keep_preview(sh, h, p.clone());
        Ok(p)
    }

    /// job_cancel on a render job: the cancel kept with the record (a finished one stays as it ended); a parked job
    /// ended at once, its `.part` settled; a run cancelled on its lane (no lock held); then up to 3 s for it to end.
    /// `cancelled` means no file and no `.part`; one that finished first reads done.
    pub(super) async fn render_cancel(&self, h: &str) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let run = {
            let mut jobs = sh.jobs.lock().unwrap();
            let i = at(&jobs, h)?;
            let h = jobs.r.v[i].h.clone();
            let mut runs = sh.runs.lock().unwrap();
            let (run, parked) = match runs.get_mut(&h) {
                Some(l) => {
                    l.cancel = true;
                    match &l.st {
                        LiveSt::Run(r) => (Some(r.clone()), false),
                        st => (None, matches!(st, LiveSt::Parked(_))),
                    }
                }
                None => (None, false),
            };
            let parked = parked.then(|| runs.remove(&h)).flatten();
            drop(runs);
            if !jobs.r.prog(i).finished() && !jobs.r.v[i].cancelled {
                jobs.r.v[i].cancelled = true;
                if let Err(e) = jobs.r.save() {
                    sh.log.line(&format!("the renders file: {e}"))
                }
            }
            if let Some(l) = parked {
                sh.settle_render(&mut jobs, &h, l, End::cancelled(""))
            }
            run
        };
        if let Some(r) = run {
            let _ = caught(|| r.cancel());
        }
        let end = Instant::now() + Duration::from_secs(3);
        let mut v = self.job(h)?;
        while v["finished"] != true && Instant::now() < end {
            tokio::time::sleep(Duration::from_millis(50)).await;
            v = self.job(h)?
        }
        if v["finished"] != true {
            v["note"] = STOPPING.into()
        }
        ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Opts,
        config::Source,
        host::Rec,
        render::{Adjust, Backend, Codec, Later, Phase, Projection, Run, RunSt, Still},
        renders::{Renders, Snap, tests::rjob},
    };
    use rmcp::{RoleClient, ServiceExt, service::RunningService};
    use std::{
        io::Write,
        sync::atomic::{AtomicBool, AtomicU64, AtomicUsize},
    };

    /// The DTO and the row as the job tools show them, per kind and state; the notifications' text and value.
    #[test]
    fn dto() {
        let mut r = Renders::default();
        r.add(
            "1x3k.9fz2-4".into(),
            rjob("/agent/VID_20260117_141148_00_635_pano57.mp4"),
        );
        for (k, x) in [
            ("width", 5760.0),
            ("height", 2880.0),
            ("start_s", 1.5),
            ("end_s", 7.34),
        ] {
            r.v[0].settings[k] = x.into()
        }
        let v = render_dto(&r.v[0], &r.prog(0));
        // the size and the length at the top, as the export's first answer shows them; no run time yet
        assert_eq!(
            (&v["width"], &v["height"], &v["duration_s"], &v["secs"]),
            (&json!(5760.0), &json!(2880.0), &json!(5.84), &Value::Null)
        );
        assert!(v.get("time_s").is_none());
        assert_eq!(
            (
                &v["kind"],
                &v["state"],
                &v["phase"],
                &v["finished"],
                &v["frames_total"]
            ),
            (
                &json!("export"),
                &json!("queued"),
                &json!("gpu_starting"),
                &json!(false),
                &json!(175)
            )
        );
        assert_eq!(
            (
                &v["bytes"],
                &v["error"],
                &v["note"],
                &v["concealed_frames"],
                &v["cancel_requested"]
            ),
            (
                &Value::Null,
                &Value::Null,
                &Value::Null,
                &json!(0),
                &json!(false)
            )
        );
        assert_eq!(v["settings"]["preset"], "pano57");
        assert_eq!(v["source"][0], "/v/VID_20260117_141148_00_635.insv");
        assert_eq!(told(RKind::Export, &r.prog(0)), "export: starting the GPU");
        let t = Instant::now();
        r.v[0].live = Some(Snap {
            st: RunSt::Running {
                phase: Phase::Frames,
                done: 120,
                total: 175,
                fps: 12.34,
                eta: 4.46,
                concealed: 0,
            },
            at: t,
            last_frame_at: t,
            moved: t,
        });
        r.v[0].cancelled = true;
        let v = render_dto(&r.v[0], &r.prog(0));
        assert_eq!(
            (
                &v["state"],
                &v["phase"],
                &v["frames_done"],
                &v["fps"],
                &v["eta_s"],
                &v["cancel_requested"]
            ),
            (
                &json!("running"),
                &json!("frames"),
                &json!(120),
                &json!(12.3),
                &json!(4.5),
                &json!(true)
            )
        );
        assert_eq!(
            told(RKind::Export, &r.prog(0)),
            "export: 120 of 175 frames, 12.3 frames/s, 4 s left"
        );
        // a done export the cancel came too late for
        r.v[0].end = Some(End::done(175, 14.0, 9_000_000, 0));
        let v = render_dto(&r.v[0], &r.prog(0));
        assert_eq!(
            (
                &v["state"],
                &v["phase"],
                &v["bytes"],
                &v["fps"],
                &v["elapsed_s"],
                &v["note"]
            ),
            (
                &json!("done"),
                &Value::Null,
                &json!(9_000_000),
                &json!(12.5),
                &json!(14.0),
                &json!("finished before the cancel")
            )
        );
        assert_eq!(v["secs"], 14.0);
        let w = row(&r.v[0], &r.prog(0));
        let keys: Vec<&String> = w.as_object().unwrap().keys().collect();
        assert_eq!(
            keys,
            [
                "created",
                "frames_done",
                "frames_total",
                "job",
                "kind",
                "path",
                "state"
            ]
        );
        // a still its host left: cancelled, why in the note; a failed one says why
        let mut s = rjob("/agent/x_3.00s_360_7680.jpg");
        (s.kind, s.frames) = (RKind::Still, 1);
        s.settings = json!({ "width": 7680, "height": 3840, "time_s": 3.0 });
        s.end = Some(End::cancelled(crate::renders::GONE));
        r.add("zz.0000-2".into(), s);
        let v = render_dto(&r.v[1], &r.prog(1));
        assert_eq!(
            (
                &v["kind"],
                &v["state"],
                &v["note"],
                &v["concealed"],
                &v["elapsed_s"]
            ),
            (
                &json!("still"),
                &json!("cancelled"),
                &json!(crate::renders::GONE),
                &json!(false),
                &Value::Null
            )
        );
        assert!(v.get("concealed_frames").is_none() && v.get("duration_s").is_none());
        assert_eq!(
            (&v["width"], &v["height"], &v["time_s"], &v["secs"]),
            (&json!(7680), &json!(3840), &json!(3.0), &Value::Null)
        );
        r.v[1].end = Some(End {
            state: "failed".into(),
            err: "no IMU".into(),
            ..End::cancelled("")
        });
        let v = render_dto(&r.v[1], &r.prog(1));
        assert_eq!((&v["error"], &v["note"]), (&json!("no IMU"), &Value::Null));
    }

    /// While no frame comes the value creeps towards the next frame, never reaching it.
    #[test]
    fn creeps() {
        let s = |n| creep(Duration::from_secs(n));
        assert_eq!((s(0), s(4)), (0.0, 0.0));
        assert_eq!(s(5), 0.25);
        assert!(s(10) > s(5) && s(60) > s(10) && s(600) < 1.0 && s(u64::MAX) < 1.0);
    }

    /// The recording the fake's tests render.
    const REC: &str = "VID_20260117_141148_00_635.insv";

    /// A run of the fake: in the state the test sets (queued at first) until cancelled, unless `stuck` (it ignores its
    /// stop); it holds the `.part` it was handed.
    struct FRun {
        stop: AtomicBool,
        stuck: AtomicBool,
        st: Mutex<RunSt>,
        part: media::part::Part,
        lane: Lane,
    }

    impl FRun {
        fn set(&self, st: RunSt) {
            *self.st.lock().unwrap() = st
        }
    }

    impl Run for FRun {
        fn state(&self) -> RunSt {
            match self.stop.load(Relaxed) && !self.stuck.load(Relaxed) {
                true => RunSt::Cancelled,
                false => self.st.lock().unwrap().clone(),
            }
        }
        fn cancel(&self) {
            self.stop.store(true, Relaxed)
        }
    }

    /// A backend for the checks: the catalog of [`cat`]; a check that answers a plan of `bytes` (or `fail`) and keeps
    /// what it was asked; a GPU state the test sets; a start that counts, waits `slow` ms and answers `start` (its
    /// GPU then up, or failed for 60 s); a submit that keeps its runs; probes that count and answer `probe`; wedges
    /// and trims recorded.
    struct F {
        fail: Mutex<Option<String>>,
        bytes: AtomicU64,
        gpu: Mutex<GpuSt>,
        start: Mutex<Result<GpuInfo, String>>,
        slow: AtomicU64,
        checks: AtomicUsize,
        starts: AtomicUsize,
        asked: Mutex<Option<Want>>,
        runs: Mutex<Vec<Arc<FRun>>>,
        probe: Mutex<Result<(), String>>,
        probes: AtomicUsize,
        wedged: Mutex<Vec<(Lane, String)>>,
        trims: AtomicUsize,
    }

    fn fake() -> Arc<F> {
        Arc::new(F {
            fail: Mutex::default(),
            bytes: AtomicU64::new(1000),
            gpu: Mutex::new(GpuSt::Off),
            start: Mutex::new(Ok(info())),
            slow: AtomicU64::new(0),
            checks: AtomicUsize::new(0),
            starts: AtomicUsize::new(0),
            asked: Mutex::default(),
            runs: Mutex::default(),
            probe: Mutex::new(Ok(())),
            probes: AtomicUsize::new(0),
            wedged: Mutex::default(),
            trims: AtomicUsize::new(0),
        })
    }

    impl F {
        /// Its `n`th run.
        fn run(&self, n: usize) -> Arc<FRun> {
            self.runs.lock().unwrap()[n].clone()
        }
    }

    /// A GPU that encodes H.265 up to 8192 and H.264 up to 4096, RAFT there.
    fn info() -> GpuInfo {
        let cap = |c: &str, m| Cap {
            codec: c.into(),
            min: [16, 16],
            max: [m, m],
            rate: vec!["cbr".into(), "cqp".into()],
            max_mbps: Some(400.0),
            qp: Some([10, 45]),
            ..Default::default()
        };
        GpuInfo {
            adapter: "Fake GPU".into(),
            encode: vec![cap("h265", 8192), cap("h264", 4096)],
            max_texture: 16384,
            ..Default::default()
        }
    }

    /// Two presets, two codecs, two looks, two projections, one adjustment, Shot Lab and face blur later.
    fn cat() -> Catalog {
        let r = |min, max, d| Range {
            min,
            max,
            default: Some(d),
        };
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let preset = |id: &str, output, width, height, codec: &str| Preset {
            id: id.into(),
            name: id.into(),
            output,
            width,
            height,
            codec: codec.into(),
            ten_bit: codec == "h265",
            mbps: 60.0,
        };
        let proj = |id: &str, aliases, fov, pitch| Projection {
            id: id.into(),
            aliases,
            fov,
            yaw: 0.0,
            pitch,
            roll: 0.0,
            default_fov: fov[1] - 10.0,
        };
        Catalog {
            presets: vec![
                preset("p360", Output::Pano, 3840, 1920, "h265"),
                preset("flat", Output::Flat, 1280, 720, "h264"),
            ],
            default_preset: "p360".into(),
            pano_widths: vec![3840, 5760, 7680],
            flat_sizes: vec![[1280, 720], [1920, 1080]],
            codecs: vec![
                Codec {
                    id: "h265".into(),
                    ten_bit: true,
                    max_px: 8192,
                },
                Codec {
                    id: "h264".into(),
                    ten_bit: false,
                    max_px: 4096,
                },
            ],
            mbps: r(1.0, 400.0, 60.0),
            qp: r(10.0, 45.0, 24.0),
            fps: s(&["source", "24", "25", "29.97", "30", "60"]),
            raft: true,
            colour_match: true,
            stabilisation: s(&["off", "free", "horizon", "direction"]),
            smoothing_s: r(0.0, 2.0, 0.5),
            looks: vec![
                Named {
                    id: "mono".into(),
                    name: Some("Black & White".into()),
                    aliases: s(&["black"]),
                },
                Named {
                    id: "vivid".into(),
                    name: Some("Vivid".into()),
                    aliases: vec![],
                },
            ],
            look_strength: r(0.0, 1.0, 1.0),
            adjust: vec![Adjust {
                id: "exposure".into(),
                label: "Exposure (EV)".into(),
                min: -3.0,
                max: 3.0,
                default: 0.0,
            }],
            projections: vec![
                proj("perspective", s(&["persp"]), [20.0, 160.0], 0.0),
                proj("planet", vec![], [90.0, 350.0], -90.0),
            ],
            still: Still {
                pano_width: 7680,
                flat: [1920, 1080],
                jpeg_quality: r(50.0, 100.0, 92.0),
            },
            later: vec![
                Later {
                    id: "face_blur".into(),
                    aliases: s(&["faces", "blur"]),
                    ..Default::default()
                },
                Later {
                    id: "shot_lab".into(),
                    aliases: s(&["effect"]),
                    effects: s(&["trails", "stars"]),
                    ..Default::default()
                },
            ],
        }
    }

    impl Render for F {
        fn catalog(&self) -> Catalog {
            cat()
        }
        fn check(&self, r: &Req) -> Result<Plan, String> {
            self.checks.fetch_add(1, Relaxed);
            *self.asked.lock().unwrap() = Some(r.want.clone());
            if let Some(e) = self.fail.lock().unwrap().clone() {
                return Err(e);
            }
            let still = r.kind == RKind::Still;
            let stem = r.files[0].file_stem().unwrap().to_str().unwrap();
            let raft = r.want.raft.unwrap_or(true);
            Ok(Plan {
                frames: if still { 1 } else { 10 },
                secs: 0.4,
                size: [64, 32],
                codec: (!still).then(|| "h265".into()),
                ten_bit: false,
                raft,
                mbps: None,
                qp: None,
                tracks: vec![],
                stem: format!("{stem}_{}", if still { "still" } else { "p360" }),
                ext: if still { "jpg" } else { "mp4" },
                bytes: self.bytes.load(Relaxed),
                settings: json!({ "width": 64, "height": 32, "raft": raft }),
            })
        }
        fn start(&self, _: &Start) -> Result<GpuInfo, String> {
            self.starts.fetch_add(1, Relaxed);
            std::thread::sleep(Duration::from_millis(self.slow.load(Relaxed)));
            let r = self.start.lock().unwrap().clone();
            *self.gpu.lock().unwrap() = match &r {
                Ok(i) => GpuSt::Ready(i.clone()),
                Err(e) => GpuSt::Failed {
                    why: e.clone(),
                    step: Some(1),
                    retry_in: Some(Duration::from_secs(60)),
                },
            };
            r
        }
        fn gpu(&self) -> GpuSt {
            self.gpu.lock().unwrap().clone()
        }
        fn selftest(&self) -> Result<GpuInfo, (u8, String)> {
            Err((1, "no self-test here".into()))
        }
        fn probe(&self) -> Result<(), String> {
            self.probes.fetch_add(1, Relaxed);
            self.probe.lock().unwrap().clone()
        }
        fn submit(
            &self,
            _: Req,
            _: &Plan,
            _: &Path,
            part: media::part::Part,
            lane: Lane,
        ) -> Result<Arc<dyn Run>, String> {
            let r = Arc::new(FRun {
                stop: AtomicBool::new(false),
                stuck: AtomicBool::new(false),
                st: Mutex::new(RunSt::Queued),
                part,
                lane,
            });
            self.runs.lock().unwrap().push(r.clone());
            Ok(r)
        }
        fn wedge(&self, lane: Lane, why: &str) {
            self.wedged.lock().unwrap().push((lane, why.into()))
        }
        fn trim(&self) {
            self.trims.fetch_add(1, Relaxed);
        }
    }

    type Cli = RunningService<RoleClient, ()>;

    /// A `--sim` host on fake `f` (tiers read, local, camera; its agent folder made) whose `--allow-dir` holds a
    /// recording, a proxy of another, an `.mp4` and a photo; a client on it.
    async fn setup(f: &Arc<F>, n: &str) -> (Arc<Host>, Cli, PathBuf) {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-rs-{n}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for x in [
            REC,
            "LRV_20260117_141148_01_636.lrv",
            "x.mp4",
            "IMG_20260117_141148_00_637.insp",
        ] {
            std::fs::write(d.join(x), b"x").unwrap()
        }
        let h = Host::new(Opts {
            sim: Some(Default::default()),
            tiers: Some(Source::Fixed(Tiers::parse("read,local,camera").unwrap())),
            allow: vec![d.clone()],
            backend: Backend {
                render: Some(f.clone()),
            },
            ..Default::default()
        })
        .await
        .unwrap();
        assert!(h.agent().starts_with(std::env::temp_dir()));
        std::fs::create_dir_all(h.agent()).unwrap();
        let (up_c, up_s) = tokio::io::duplex(1 << 16);
        let (down_s, down_c) = tokio::io::duplex(1 << 16);
        let hh = h.clone();
        tokio::spawn(async move { hh.serve(up_s, down_s, "test", None).await });
        let c = ().serve((down_c, up_c)).await.unwrap();
        (h, c, d)
    }

    /// A call's JSON (a still's from its first text block), or its error.
    async fn call(c: &Cli, n: &str, a: Value) -> Result<Value, String> {
        let p = CallToolRequestParams::new(n.to_string())
            .with_arguments(a.as_object().unwrap().clone());
        let r = c.call_tool(p).await.unwrap();
        let t = r
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.clone());
        match r.is_error {
            Some(true) => Err(t.unwrap_or_default()),
            _ => Ok((r.structured_content)
                .unwrap_or_else(|| serde_json::from_str(&t.unwrap()).unwrap())),
        }
    }

    /// Waits up to 5 s for `f`.
    async fn until(mut f: impl FnMut() -> bool) {
        let t = Instant::now();
        while !f() && t.elapsed() < Duration::from_secs(5) {
            tokio::time::sleep(Duration::from_millis(20)).await
        }
        assert!(f(), "not within 5 s")
    }

    /// The checks before the GPU starts, in their order: each stage's refusal comes while a later stage would refuse
    /// too, and none reaches the backend's check before the source passed, nor its start.
    #[tokio::test(flavor = "multi_thread")]
    async fn checks() {
        let f = fake();
        let (h, c, d) = setup(&f, "checks").await;
        let p = d.join(REC).display().to_string();
        let ex = |a: Value| {
            let mut v = json!({ "path": p });
            v.as_object_mut()
                .unwrap()
                .extend(a.as_object().unwrap().clone());
            v
        };
        // the catalog: never the GPU
        let v = call(&c, "catalog", json!({})).await.unwrap();
        assert_eq!(
            (v["version"].as_i64(), v["default_preset"].as_str()),
            (Some(1), Some("p360"))
        );
        assert_eq!(
            (&v["gpu"]["state"], &v["outputs"]["360"]["video_widths"]),
            (&json!("off"), &json!([3840, 5760, 7680]))
        );
        assert_eq!(v["limits"]["unfinished_jobs"], 16);
        // 1. the arguments: a later feature by name before a bad look; the other tool's fields; a typo
        let e = |r: Result<Value, String>| r.unwrap_err();
        let r = call(&c, "export", ex(json!({ "faces": true, "look": "sepia" }))).await;
        assert_eq!(
            e(r),
            "`faces` is not available to agents yet (catalog: later: face_blur)"
        );
        let r = call(&c, "export", ex(json!({ "trails": 1 }))).await;
        assert_eq!(
            e(r),
            "`trails` is not available to agents yet (catalog: later: shot_lab)"
        );
        let r = call(&c, "render_still", ex(json!({ "allow_frozen": true }))).await;
        assert_eq!(e(r), "`allow_frozen` is for export only");
        let r = call(&c, "export", ex(json!({ "time_s": 1 }))).await;
        assert_eq!(e(r), "`time_s` is for render_still only");
        let r = call(&c, "export", ex(json!({ "lok": "mono" }))).await;
        assert!(e(r).contains("unknown field `lok`"));
        // 2. the catalog, before the host: a memory-only host says the bad argument first
        h.sh.jobs.lock().unwrap().r.close().unwrap();
        for (k, a, w) in [
            (
                "export",
                json!({ "look": "sepia" }),
                "look \"sepia\": not a catalog look (mono, vivid)",
            ),
            (
                "export",
                json!({ "preset": "pano99" }),
                "preset \"pano99\": not a catalog preset (p360, flat)",
            ),
            (
                "export",
                json!({ "output": 180 }),
                "output 180: \"360\" or \"flat\"",
            ),
            (
                "export",
                json!({ "width": 4000 }),
                "width 4000: a 360 export is 3840, 5760, 7680 wide (catalog outputs)",
            ),
            (
                "export",
                json!({ "preset": "flat", "width": 1280, "height": 721 }),
                "1280 × 721: a flat export is 1280 × 720, 1920 × 1080 (catalog outputs)",
            ),
            (
                "export",
                json!({ "codec": "h264", "width": 5760 }),
                "5760 × 2880: h264 goes up to 4096 a side",
            ),
            (
                "export",
                json!({ "codec": "h264", "ten_bit": true }),
                "ten_bit: h264 has no 10-bit (h265 has)",
            ),
            (
                "export",
                json!({ "mbps": 50, "qp": 20 }),
                "mbps or qp, not both",
            ),
            ("export", json!({ "qp": 50 }), "qp 50: 10 to 45"),
            (
                "export",
                json!({ "fps": 31 }),
                "fps 31: one of source, 24, 25, 29.97, 30, 60",
            ),
            (
                "export",
                json!({ "view": { "fov": 90 } }),
                "view: for output flat only (360 is the whole sphere)",
            ),
            (
                "export",
                json!({ "output": "flat", "view": { "fov": 170 } }),
                "fov (perspective) 170: 20 to 160",
            ),
            (
                "export",
                json!({ "output": "flat", "view": { "projection": "planet", "pitch": -91 } }),
                "pitch -91: -90 to 90",
            ),
            (
                "export",
                json!({ "adjust": { "exposure": 4 } }),
                "adjust exposure 4: -3 to 3",
            ),
            (
                "export",
                json!({ "adjust": { "gamma": 1 } }),
                "adjust \"gamma\": not a catalog adjust id (exposure)",
            ),
            (
                "export",
                json!({ "stabilization": "wobbly" }),
                "stabilisation \"wobbly\": one of off, free, horizon, direction",
            ),
            (
                "export",
                json!({ "start_s": 2, "end_s": 1 }),
                "start_s 2 is not before end_s 1",
            ),
            (
                "export",
                json!({ "name": "IMG_20260201_111111_00_701" }),
                "name \"IMG_20260201_111111_00_701\": a camera file's name: pick another",
            ),
            (
                "export",
                json!({ "name": "clip.mov" }),
                "name \"clip.mov\": an export is a .mp4, not a .mov",
            ),
            (
                "export",
                json!({ "subdir": "../out" }),
                "../out: `..` is refused",
            ),
            (
                "render_still",
                json!({ "width": 1001 }),
                "width 1001: a 360 still is a multiple of 8, 512 to 8192 wide",
            ),
            (
                "render_still",
                json!({ "output": "flat", "width": 100 }),
                "a flat still takes width and height together (default 1920 × 1080)",
            ),
            (
                "render_still",
                json!({ "jpeg_quality": 40 }),
                "jpeg_quality 40: 50 to 100",
            ),
            (
                "render_still",
                json!({ "name": "x.mp4" }),
                "name \"x.mp4\": a still is a .jpg, not a .mp4",
            ),
        ] {
            assert_eq!(e(call(&c, k, ex(a.clone())).await), w, "{k} {a}");
        }
        // 3. the host before the source (the path is bad too): its jobs in memory only, a timed recording of its own
        let nope = json!({ "path": d.join("nope.insv").display().to_string() });
        assert_eq!(e(call(&c, "export", nope.clone()).await), MEMORY);
        let rf = h.sim().unwrap().dir.join("mcp-renders");
        h.sh.jobs.lock().unwrap().r = Renders::load(rf).unwrap().0;
        let rec = |until| Rec {
            serial: None,
            n: 1,
            sent: true,
            seen: true,
            mode: "VIDEO_NORMAL".into(),
            at: Instant::now(),
            until,
            due: false,
            stop: None,
            sent_at: None,
        };
        h.sh.st.lock().unwrap().rec = Some(rec(Some(Instant::now() + Duration::from_secs(60))));
        let r = e(call(&c, "render_still", nope.clone()).await);
        assert!(
            r.starts_with("the camera records until ") && r.ends_with("render after it ends"),
            "{r}"
        );
        h.sh.st.lock().unwrap().rec = Some(rec(None));
        h.sh.timed.store(true, Relaxed);
        let r = e(call(&c, "export", nope.clone()).await);
        assert!(
            r.starts_with("a timed capture_record start is under way"),
            "{r}"
        );
        h.sh.timed.store(false, Relaxed);
        // 4. the source: an untimed recording coexists; missing, a proxy alone, an .mp4, a photo
        assert!(e(call(&c, "export", nope).await).contains("nope.insv: "));
        for (x, w) in [
            (
                "LRV_20260117_141148_01_636.lrv",
                "an .lrv is the camera's proxy",
            ),
            ("x.mp4", "an .mp4 has no Insta360 trailer"),
            (
                "IMG_20260117_141148_00_637.insp",
                "a photo; render_still and export take X5 recordings",
            ),
        ] {
            let a = json!({ "path": d.join(x).display().to_string() });
            let r = e(call(&c, "export", a).await);
            assert!(r.contains(w), "{x}: {r}");
        }
        assert_eq!(
            f.checks.load(Relaxed),
            0,
            "the backend never asked before the source passed"
        );
        // 5. the backend's check, every id as the catalog spells it, numbers mapped onto labels
        *f.fail.lock().unwrap() = Some("no Insta360 trailer".into());
        let a = ex(json!({
            "output": 360, "fps": 30, "look": "BLACK & white", "stabilization": "Horizon", "color_match": false,
            "adjust": { "exposure": 0.5 }, "codec": "H265", "ten_bit": true,
        }));
        assert_eq!(e(call(&c, "export", a).await), "no Insta360 trailer");
        let w = f.asked.lock().unwrap().clone().unwrap();
        assert_eq!(
            (
                w.preset.as_deref(),
                w.output,
                w.fps.as_deref(),
                w.look.as_deref(),
                w.stabilisation.as_deref()
            ),
            (
                Some("p360"),
                Some(Output::Pano),
                Some("30"),
                Some("mono"),
                Some("horizon")
            )
        );
        assert_eq!(
            (w.colour_match, w.codec.as_deref(), w.adjust["exposure"]),
            (Some(false), Some("h265"), 0.5)
        );
        let a = ex(
            json!({ "fps": "29.97", "output": "flat", "preset": "flat", "view": { "projection": "persp" } }),
        );
        assert_eq!(e(call(&c, "export", a).await), "no Insta360 trailer");
        let w = f.asked.lock().unwrap().clone().unwrap();
        assert_eq!(
            (w.fps.as_deref(), w.view.unwrap().projection.as_deref()),
            (Some("29.97"), Some("perspective"))
        );
        // a 360 still's height follows its width; no preset
        assert_eq!(
            e(call(&c, "render_still", ex(json!({ "width": 3840 }))).await),
            "no Insta360 trailer"
        );
        let w = f.asked.lock().unwrap().clone().unwrap();
        assert_eq!(
            (w.width, w.height, w.preset),
            (Some(3840), Some(1920), None)
        );
        assert_eq!(f.checks.load(Relaxed), 3);
        *f.fail.lock().unwrap() = None;
        // 6. the device: lost, failed for good, a lane stuck, RAFT asked of a GPU without, a codec it lacks
        let set = |g| *f.gpu.lock().unwrap() = g;
        set(GpuSt::Dead("lost".into()));
        let r = e(call(&c, "export", ex(json!({}))).await);
        assert!(
            r.starts_with("the GPU device was lost (lost); GPU tools are off"),
            "{r}"
        );
        set(GpuSt::Failed {
            why: "no Vulkan device".into(),
            step: Some(1),
            retry_in: None,
        });
        let r = e(call(&c, "export", ex(json!({}))).await);
        assert_eq!(
            r,
            "GPU tools are unavailable: no Vulkan device (self-test step 1); the other tools work."
        );
        let mut i = info();
        i.wedged = vec![Lane::Export];
        set(GpuSt::Ready(i.clone()));
        let r = e(call(&c, "export", ex(json!({}))).await);
        assert_eq!(
            r,
            "the GPU's export lane is stuck: this MCP server must restart to use it"
        );
        i.wedged.clear();
        i.raft = Some("burn's attach failed".into());
        set(GpuSt::Ready(i.clone()));
        let r = e(call(&c, "render_still", ex(json!({ "raft": true }))).await);
        assert_eq!(
            r,
            "RAFT (optical-flow seams) is unavailable on this GPU: burn's attach failed; pass raft: false"
        );
        i.encode.remove(0);
        set(GpuSt::Ready(i.clone()));
        let r = e(call(&c, "export", ex(json!({}))).await);
        assert_eq!(r, "this GPU does not encode h265: it encodes h264");
        // the catalog shows what this GPU takes
        let v = call(&c, "catalog", json!({})).await.unwrap();
        assert_eq!(
            (&v["outputs"]["360"]["video_widths"], &v["codecs"][0]["id"]),
            (&json!([3840]), &json!("h264"))
        );
        assert_eq!(
            (
                &v["raft"]["available"],
                &v["raft"]["default"],
                &v["gpu"]["state"]
            ),
            (&json!(false), &json!(false), &json!("ready"))
        );
        // 7. the destination: too little space; nothing reserved
        set(GpuSt::Ready(info()));
        f.bytes.store(u64::MAX / 4, Relaxed);
        let r = e(call(&c, "export", ex(json!({}))).await);
        assert!(
            r.starts_with("not enough space in ")
                && r.ends_with("free after this server's other jobs"),
            "{r}"
        );
        f.bytes.store(1000, Relaxed);
        assert!(
            std::fs::read_dir(h.agent()).unwrap().next().is_none(),
            "no .part"
        );
        assert!(h.sh.jobs.lock().unwrap().r.v.is_empty() && h.sh.runs.lock().unwrap().is_empty());
        assert_eq!(
            (f.starts.load(Relaxed), h.sh.rendering.load(Relaxed)),
            (0, 0)
        );
        drop(c);
        h.shutdown().await;
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// With the GPU up a job is reserved and submitted at once: its record saved with its `.part`'s identity, the
    /// name numbered past a file, a `.part` and a dangling link; at most 16 unfinished, the space counting the others;
    /// a timed recording refused meanwhile; job_cancel cancels the run; a done still answers with its image.
    #[tokio::test(flavor = "multi_thread")]
    async fn reserves() {
        let f = fake();
        *f.gpu.lock().unwrap() = GpuSt::Ready(info());
        let (h, c, d) = setup(&f, "reserves").await;
        let p = d.join(REC).display().to_string();
        let a = h.agent().to_path_buf();
        let stem = "VID_20260117_141148_00_635_p360";
        std::fs::write(a.join(format!("{stem}.mp4")), b"x").unwrap();
        std::fs::write(a.join(format!("{stem}-2.mp4.part")), b"x").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(a.join("gone"), a.join(format!("{stem}-3.mp4"))).unwrap();
        let n = if cfg!(unix) { 4 } else { 3 };
        let v = call(&c, "export", json!({ "path": p })).await.unwrap();
        let dst = a.join(format!("{stem}-{n}.mp4"));
        assert_eq!(
            (&v["state"], &v["phase"], &v["path"]),
            (
                &json!("queued"),
                &json!("waiting"),
                &json!(paths::shown(&dst))
            )
        );
        let note = v["note"].as_str().unwrap();
        assert_eq!(
            note,
            "the whole clip: 0.40 s, 10 frames at 64 × 32 (start_s/end_s: a part); job_wait follows it"
        );
        let (k, _) = Renders::load(h.sim().unwrap().dir.join("mcp-renders")).unwrap();
        let id = k.v[0].part.unwrap();
        assert!(media::part::is(&media::part::of(&dst), &id) && k.v[0].end.is_none());
        assert_eq!(
            (f.runs.lock().unwrap().len(), h.sh.rendering.load(Relaxed)),
            (1, 1)
        );
        assert!(matches!(
            h.sh.runs.lock().unwrap()[v["job"].as_str().unwrap()].st,
            LiveSt::Run(_)
        ));
        // named, into a subfolder; a still, answered at once with wait_s 0
        let w = call(
            &c,
            "export",
            json!({ "path": p, "name": "clip.MP4", "subdir": "a/b" }),
        )
        .await
        .unwrap();
        assert_eq!(w["path"], paths::shown(&a.join("a/b/clip.mp4")));
        let s = call(&c, "render_still", json!({ "path": p, "wait_s": 0 }))
            .await
            .unwrap();
        assert_eq!(
            (&s["state"], &s["note"]),
            (
                &json!("queued"),
                &json!("still rendering: job_wait answers with the image")
            )
        );
        // a timed recording waits for them
        let r = call(
            &c,
            "capture_record",
            json!({ "action": "start", "seconds": 5 }),
        )
        .await;
        assert!(
            r.unwrap_err()
                .starts_with("this host renders (3 unfinished render jobs")
        );
        // at most 16 unfinished; the free space counts the others' estimates on the volume
        let sp = media::part::avail(&a).unwrap();
        let fill = |k: usize, bytes| {
            let mut runs = h.sh.runs.lock().unwrap();
            for i in 0..k {
                let l = Live {
                    st: LiveSt::Submitting,
                    dst: a.join(format!("other-{i}.mp4")),
                    id,
                    vol: sp.vol,
                    bytes,
                    cancel: false,
                    stuck: None,
                };
                runs.insert(format!("other-{i}"), l);
            }
        };
        let unfill = || {
            h.sh.runs
                .lock()
                .unwrap()
                .retain(|k, _| !k.starts_with("other-"))
        };
        fill(13, 0);
        let r = call(&c, "export", json!({ "path": p })).await;
        assert_eq!(
            r.unwrap_err(),
            "16 render jobs of this server are unfinished: job_wait or job_cancel one first"
        );
        unfill();
        fill(1, u64::MAX / 2);
        let r = call(&c, "export", json!({ "path": p })).await.unwrap_err();
        assert!(
            r.contains(", 0.0 GB free after this server's other jobs"),
            "{r}"
        );
        unfill();
        // what another job wrote has left the free space already: it counts only what it still needs (a sparse
        // `.part` here, 4 GiB long, of an estimate 1 GiB over the free space)
        let o = a.join("other-w.mp4");
        let wp = media::part::create(&o).unwrap();
        wp.f.set_len(4 << 30).unwrap();
        let free = media::part::avail(&a).unwrap().free;
        h.sh.runs.lock().unwrap().insert(
            "other-w".into(),
            Live {
                st: LiveSt::Submitting,
                dst: o.clone(),
                id: wp.id,
                vol: sp.vol,
                bytes: free + (1 << 30),
                cancel: false,
                stuck: None,
            },
        );
        let r = call(&c, "export", json!({ "path": p, "name": "beside" })).await;
        h.sh.runs.lock().unwrap().remove("other-w");
        std::fs::remove_file(media::part::of(&o)).unwrap();
        let r = r.unwrap();
        assert!(h.sh.end_render(r["job"].as_str().unwrap(), End::cancelled("")));
        // job_cancel: the run cancelled on its lane, settled by the poller (no `.part` left)
        let x = call(&c, "job_cancel", json!({ "job": v["job"] }))
            .await
            .unwrap();
        assert!(f.runs.lock().unwrap()[0].stop.load(Relaxed));
        assert_eq!(
            (&x["state"], &x["cancel_requested"], &x["note"]),
            (&json!("cancelled"), &json!(true), &Value::Null)
        );
        assert!(!media::part::of(&dst).exists());
        // the still done: its file written into the handed `.part` and published, then job_wait answers the image
        let px: Vec<u8> = (0..3200 * 1600)
            .flat_map(|i| [(i % 3200 / 13) as u8, (i / 3200 / 7) as u8, 99, 255])
            .collect();
        let jpg = media::jpeg::encode(&px, 3200, 1600, 92, None).unwrap();
        let sd = a.join("VID_20260117_141148_00_635_still.jpg");
        let run = f.runs.lock().unwrap()[2].clone();
        (&run.part.f).write_all(&jpg).unwrap();
        media::part::publish(&media::part::of(&sd), &sd).unwrap();
        let sj = s["job"].as_str().unwrap();
        assert!(h.sh.end_render(sj, End::done(1, 0.5, 0, 0)));
        let r = (c.call_tool(
            CallToolRequestParams::new("job_wait")
                .with_arguments(json!({ "job": sj }).as_object().unwrap().clone()),
        ))
        .await
        .unwrap();
        assert!(r.structured_content.is_none() && r.content.len() == 2);
        let j: Value = serde_json::from_str(&r.content[0].as_text().unwrap().text).unwrap();
        assert_eq!(
            (&j["state"], &j["width"], &j["height"], &j["bytes"]),
            (
                &json!("done"),
                &json!(3200),
                &json!(1600),
                &json!(jpg.len())
            )
        );
        assert_eq!(
            (&j["preview"]["width"], &j["preview"]["height"]),
            (&json!(1568), &json!(784))
        );
        let img = r.content[1].as_image().unwrap();
        assert_eq!(img.mime_type, "image/jpeg");
        // every job ended: no `.part` left, the count back to 0, a timed recording no longer refused for them
        assert!(
            !h.sh
                .end_render(v["job"].as_str().unwrap(), End::cancelled(""))
        );
        assert!(h.sh.end_render(w["job"].as_str().unwrap(), End::cancelled("")));
        assert!(
            !media::part::of(&dst).exists() && !media::part::of(&a.join("a/b/clip.mp4")).exists()
        );
        assert_eq!(h.sh.rendering.load(Relaxed), 0);
        let r = call(
            &c,
            "capture_record",
            json!({ "action": "start", "seconds": 5 }),
        )
        .await;
        assert!(!r.unwrap_err().contains("this host renders"));
        drop(c);
        h.shutdown().await;
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// With the GPU off a job parks and the host's start task starts it: a failed start fails the job (no `.part`) and
    /// a retry not due yet refuses the next call; a parked job cancelled before the start ends is cancelled at once,
    /// nothing submitted for it; one parked when the start succeeds is submitted.
    #[tokio::test(flavor = "multi_thread")]
    async fn parks() {
        let f = fake();
        *f.start.lock().unwrap() = Err("no Vulkan device".into());
        let (h, c, d) = setup(&f, "parks").await;
        let p = d.join(REC).display().to_string();
        let v = call(&c, "export", json!({ "path": p })).await.unwrap();
        assert_eq!(
            (&v["state"], &v["phase"]),
            (&json!("queued"), &json!("gpu_starting"))
        );
        let job = json!({ "job": v["job"] });
        let t = Instant::now();
        let mut x = call(&c, "job_status", job.clone()).await.unwrap();
        while x["state"] == "queued" && t.elapsed() < Duration::from_secs(5) {
            tokio::time::sleep(Duration::from_millis(20)).await;
            x = call(&c, "job_status", job.clone()).await.unwrap()
        }
        assert_eq!(x["state"], "failed");
        assert!(
            x["error"]
                .as_str()
                .unwrap()
                .starts_with("GPU tools are unavailable: no Vulkan device (self-test step 1)")
        );
        assert!(
            std::fs::read_dir(h.agent()).unwrap().next().is_none(),
            "no .part"
        );
        assert_eq!(
            (f.starts.load(Relaxed), h.sh.rendering.load(Relaxed)),
            (1, 0)
        );
        let r = call(&c, "export", json!({ "path": p })).await.unwrap_err();
        assert!(
            r.ends_with("; tried again in 60 s; the other tools work."),
            "{r}"
        );
        // parked, then cancelled while the start runs
        *f.gpu.lock().unwrap() = GpuSt::Off;
        *f.start.lock().unwrap() = Ok(info());
        f.slow.store(500, Relaxed);
        let v = call(&c, "export", json!({ "path": p })).await.unwrap();
        let x = call(&c, "job_cancel", json!({ "job": v["job"] }))
            .await
            .unwrap();
        assert_eq!(
            (&x["state"], &x["cancel_requested"]),
            (&json!("cancelled"), &json!(true))
        );
        assert!(
            std::fs::read_dir(h.agent()).unwrap().next().is_none(),
            "no .part"
        );
        until(|| !h.sh.starting.load(Relaxed)).await;
        assert_eq!(
            (f.starts.load(Relaxed), f.runs.lock().unwrap().len()),
            (2, 0)
        );
        // parked while the GPU starts, submitted once it is up
        *f.gpu.lock().unwrap() = GpuSt::Starting;
        let v = call(&c, "export", json!({ "path": p })).await.unwrap();
        until(|| f.runs.lock().unwrap().len() == 1).await;
        until(|| {
            matches!(
                h.sh.runs
                    .lock()
                    .unwrap()
                    .get(v["job"].as_str().unwrap())
                    .map(|l| &l.st),
                Some(LiveSt::Run(_))
            )
        })
        .await;
        assert_eq!(f.starts.load(Relaxed), 3);
        drop(c);
        h.shutdown().await;
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A record that can't be saved refuses the call, and no `.part` is made for it.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread")]
    async fn save_fails() {
        use std::os::unix::fs::PermissionsExt;
        if unsafe { libc::geteuid() } == 0 {
            return;
        }
        let f = fake();
        *f.gpu.lock().unwrap() = GpuSt::Ready(info());
        let (h, c, d) = setup(&f, "save").await;
        let s = h.sim().unwrap().dir.clone();
        let mode = |m| std::fs::set_permissions(&s, std::fs::Permissions::from_mode(m)).unwrap();
        mode(0o500);
        let r = call(
            &c,
            "export",
            json!({ "path": d.join(REC).display().to_string() }),
        )
        .await;
        mode(0o700);
        assert!(r.unwrap_err().starts_with("can't keep the job record: "));
        assert!(
            std::fs::read_dir(h.agent()).unwrap().next().is_none(),
            "no .part"
        );
        assert!(h.sh.jobs.lock().unwrap().r.v.is_empty() && h.sh.runs.lock().unwrap().is_empty());
        assert_eq!(
            (h.sh.rendering.load(Relaxed), f.runs.lock().unwrap().len()),
            (0, 0)
        );
        drop(c);
        h.shutdown().await;
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Waits up to 5 s for job `job` to read `state`; its DTO then.
    async fn reads(c: &Cli, job: &Value, state: &str) -> Value {
        let t = Instant::now();
        loop {
            let v = call(c, "job_status", json!({ "job": job })).await.unwrap();
            if v["state"] == state || t.elapsed() > Duration::from_secs(5) {
                assert_eq!(v["state"], state, "{v}");
                return v;
            }
            tokio::time::sleep(Duration::from_millis(20)).await
        }
    }

    fn running(phase: Phase, done: u64) -> RunSt {
        RunSt::Running {
            phase,
            done,
            total: 10,
            fps: 5.0,
            eta: 1.4,
            concealed: 0,
        }
    }

    /// Run `r`'s file written and published at `dst`, as a job's end does.
    fn publish(r: &FRun, dst: &Path, b: &[u8]) {
        (&r.part.f).write_all(b).unwrap();
        media::part::publish(&media::part::of(dst), dst).unwrap();
    }

    /// A 3200 × 1600 JPEG.
    fn jpeg() -> Vec<u8> {
        let px: Vec<u8> = (0..3200 * 1600)
            .flat_map(|i| [(i % 3200 / 13) as u8, (i / 3200 / 7) as u8, 99, 255])
            .collect();
        media::jpeg::encode(&px, 3200, 1600, 92, None).unwrap()
    }

    /// The poller writes the runs' states into their records and settles each that ends by identity: a done export
    /// with its file's bytes, a done still once its preview is kept (the call waiting for it answers with the image),
    /// a failed one without its `.part`. Idle sleep is kept off meanwhile; a failure that says the device was lost
    /// probes the GPU; once idle, the GPU is trimmed once. The log says how each ended.
    #[tokio::test(flavor = "multi_thread")]
    async fn polls() {
        let f = fake();
        *f.gpu.lock().unwrap() = GpuSt::Ready(info());
        let (h, c, d) = setup(&f, "polls").await;
        let p = d.join(REC).display().to_string();
        let a = h.agent().to_path_buf();
        let v = call(&c, "export", json!({ "path": p })).await.unwrap();
        let job = v["job"].clone();
        until(|| h.sh.keep.lock().unwrap().is_some()).await;
        f.run(0).set(running(Phase::Frames, 3));
        let x = reads(&c, &job, "running").await;
        assert_eq!(
            (&x["phase"], &x["frames_done"], &x["fps"], &x["eta_s"]),
            (&json!("frames"), &json!(3), &json!(5.0), &json!(1.4))
        );
        let dst = a.join("VID_20260117_141148_00_635_p360.mp4");
        publish(&f.run(0), &dst, b"12345");
        f.run(0).set(RunSt::Done {
            frames: 10,
            secs: 2.0,
            bytes: 5,
            concealed: 0,
        });
        let x = reads(&c, &job, "done").await;
        assert_eq!(
            (
                &x["bytes"],
                &x["frames_done"],
                &x["fps"],
                &x["concealed_frames"]
            ),
            (&json!(5), &json!(10), &json!(5.0), &json!(0))
        );
        assert!(dst.exists() && !media::part::of(&dst).exists());
        // a still: the call waiting for it answers with its image, made before the job reads done
        let jpg = jpeg();
        let sd = a.join("VID_20260117_141148_00_635_still.jpg");
        let ask = CallToolRequestParams::new("render_still").with_arguments(
            json!({ "path": p, "wait_s": 10 })
                .as_object()
                .unwrap()
                .clone(),
        );
        let drive = async {
            until(|| f.runs.lock().unwrap().len() == 2).await;
            let r = f.run(1);
            assert_eq!(r.lane, Lane::Still);
            publish(&r, &sd, &jpg);
            r.set(RunSt::Done {
                frames: 1,
                secs: 0.5,
                bytes: jpg.len() as u64,
                concealed: 1,
            });
        };
        let (r, ()) = tokio::join!(c.call_tool(ask), drive);
        let r = r.unwrap();
        assert!(r.is_error != Some(true) && r.content.len() == 2, "{r:?}");
        let s: Value = serde_json::from_str(&r.content[0].as_text().unwrap().text).unwrap();
        assert_eq!(
            (&s["state"], &s["concealed"], &s["preview"]["width"]),
            (&json!("done"), &json!(true), &json!(1568))
        );
        let sj = s["job"].as_str().unwrap();
        assert!(h.sh.previews.lock().unwrap().iter().any(|x| x.0 == sj));
        // a failure that says the device was lost: failed after its frames so far, no `.part`, the GPU probed
        let v = call(&c, "export", json!({ "path": p, "name": "lost" }))
            .await
            .unwrap();
        f.run(2).set(running(Phase::Frames, 4));
        reads(&c, &v["job"], "running").await;
        f.run(2)
            .set(RunSt::Failed("wgpu: the device was lost".into()));
        let x = reads(&c, &v["job"], "failed").await;
        assert_eq!(
            (&x["error"], &x["frames_done"]),
            (&json!("wgpu: the device was lost"), &json!(4))
        );
        assert!(!media::part::of(&a.join("lost.mp4")).exists());
        until(|| f.probes.load(Relaxed) == 1).await;
        // a picture that could not be decoded: the export's error names the way out; a cancel keeps the frames made
        let fz = call(&c, "export", json!({ "path": p, "name": "frozen" }))
            .await
            .unwrap();
        let why = format!(
            "frame 4: 2 source pictures {} (did the computer sleep?)",
            crate::render::CONCEALED
        );
        f.run(3).set(RunSt::Failed(why.clone()));
        let x = reads(&c, &fz["job"], "failed").await;
        assert_eq!(
            x["error"],
            format!("{why}; pass allow_frozen: true to export it with repeated frames")
        );
        let cv = call(&c, "export", json!({ "path": p, "name": "cancel" }))
            .await
            .unwrap();
        f.run(4).set(running(Phase::Frames, 6));
        reads(&c, &cv["job"], "running").await;
        let x = call(&c, "job_cancel", json!({ "job": cv["job"] }))
            .await
            .unwrap();
        assert_eq!(
            (&x["state"], &x["frames_done"]),
            (&json!("cancelled"), &json!(6))
        );
        // idle: sleep allowed again, the GPU trimmed once
        until(|| h.sh.keep.lock().unwrap().is_none()).await;
        until(|| f.trims.load(Relaxed) == 1).await;
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert_eq!(
            (f.trims.load(Relaxed), h.sh.rendering.load(Relaxed)),
            (1, 0)
        );
        let log = std::fs::read_to_string(&h.sim().unwrap().log).unwrap();
        for l in [
            format!(
                "export {} done: 10 frames in 2.0 s (5.0 frames/s) → VID_20260117_141148_00_635_p360.mp4",
                job.as_str().unwrap()
            ),
            format!("still {sj} done in 0.5 s → VID_20260117_141148_00_635_still.jpg"),
            format!(
                "export {} failed: wgpu: the device was lost",
                v["job"].as_str().unwrap()
            ),
            "the GPU's probe passed".into(),
        ] {
            assert!(log.contains(&l), "{l}\n{log}");
        }
        drop(c);
        h.shutdown().await;
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The watchdog: a run that does not move is cancelled (whatever the probe says: it passes here); one that ends
    /// then fails saying why; one that stays stuck is settled failed after the grace, its `.part` gone and its lane
    /// wedged. A run finishing its file is left alone, and the other lane goes on.
    #[tokio::test(flavor = "multi_thread")]
    async fn watchdog() {
        let f = fake();
        *f.gpu.lock().unwrap() = GpuSt::Ready(info());
        let (h, c, d) = setup(&f, "watch").await;
        let p = d.join(REC).display().to_string();
        let a = h.agent().to_path_buf();
        // stuck in frames, its stop ignored
        let x = call(&c, "export", json!({ "path": p, "name": "x" }))
            .await
            .unwrap();
        f.run(0).stuck.store(true, Relaxed);
        f.run(0).set(running(Phase::Frames, 3));
        // a still that does not move in its setup, stopped by a cancel
        let y = call(&c, "render_still", json!({ "path": p, "wait_s": 0 }))
            .await
            .unwrap();
        f.run(1).set(running(Phase::Setup, 0));
        // finishing its file
        let z = call(&c, "export", json!({ "path": p, "name": "z" }))
            .await
            .unwrap();
        f.run(2).set(running(Phase::Finishing, 10));
        let s = reads(&c, &y["job"], "failed").await;
        assert_eq!(s["error"], "stuck in setup: nothing for 1 s");
        let s = reads(&c, &x["job"], "failed").await;
        assert_eq!(
            (&s["error"], &s["frames_done"]),
            (
                &json!(
                    "stuck in frames: nothing for 1 s; the GPU's export lane is off until this MCP server restarts"
                ),
                &json!(3)
            )
        );
        assert!(f.run(0).stop.load(Relaxed) && !media::part::of(&a.join("x.mp4")).exists());
        until(|| f.wedged.lock().unwrap().len() == 1).await;
        assert_eq!(
            f.wedged.lock().unwrap()[0],
            (
                Lane::Export,
                "the GPU's export lane is stuck (stuck in frames: nothing for 1 s): this MCP server must restart \
                 to use it"
                    .into()
            )
        );
        assert!(f.probes.load(Relaxed) >= 1);
        let s = reads(&c, &z["job"], "running").await;
        assert_eq!(
            (&s["phase"], f.run(2).stop.load(Relaxed)),
            (&json!("finishing"), false)
        );
        // the still lane goes on
        let w = call(&c, "render_still", json!({ "path": p, "wait_s": 0 }))
            .await
            .unwrap();
        let sd = a.join("VID_20260117_141148_00_635_still.jpg");
        publish(&f.run(3), &sd, &jpeg());
        f.run(3).set(RunSt::Done {
            frames: 1,
            secs: 0.1,
            bytes: 1,
            concealed: 0,
        });
        reads(&c, &w["job"], "done").await;
        assert_eq!(f.wedged.lock().unwrap().len(), 1);
        // no render left, but the stuck one's thread may be in the driver yet: no timed recording on this host's timer
        f.run(2).set(RunSt::Cancelled);
        reads(&c, &z["job"], "cancelled").await;
        assert_eq!(h.sh.rendering.load(Relaxed), 0);
        let r = call(
            &c,
            "capture_record",
            json!({ "action": "start", "seconds": 5 }),
        )
        .await;
        assert!(
            r.as_ref()
                .unwrap_err()
                .starts_with("a render of this host is stuck in a GPU driver call"),
            "{r:?}"
        );
        drop(c);
        h.shutdown().await;
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A host prunes its own records as it ends them: every unfinished one and the [`crate::renders::KEEP`] that
    /// ended last stay, so an export made before many stills and ending after them is still there to read.
    #[tokio::test(flavor = "multi_thread")]
    async fn prunes_own() {
        let f = fake();
        *f.gpu.lock().unwrap() = GpuSt::Ready(info());
        let (h, c, d) = setup(&f, "prunes").await;
        let p = d.join(REC).display().to_string();
        let x = call(&c, "export", json!({ "path": p })).await.unwrap();
        let keep = crate::renders::KEEP;
        for i in 0..keep + 5 {
            let mut j = rjob(&format!("/agent/s{i}.jpg"));
            j.end = Some(End::cancelled(""));
            h.sh.jobs.lock().unwrap().r.add(format!("old-{i}"), j)
        }
        f.run(0).set(RunSt::Cancelled);
        reads(&c, &x["job"], "cancelled").await;
        let v: Vec<String> = (h.sh.jobs.lock().unwrap().r.v.iter())
            .map(|r| r.h.clone())
            .collect();
        assert_eq!(
            (v.len(), v[0].as_str(), v[v.len() - 1].as_str()),
            (keep, "old-6", x["job"].as_str().unwrap())
        );
        drop(c);
        h.shutdown().await;
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A suspect GPU is probed, once while the probe passes; a job failing as a lost device fails probes it again, and
    /// a failed probe stops every run of this host, each failing as the device was lost. A dead GPU refuses new jobs.
    #[tokio::test(flavor = "multi_thread")]
    async fn dead() {
        let f = fake();
        *f.gpu.lock().unwrap() = GpuSt::Ready(GpuInfo {
            suspect: true,
            ..info()
        });
        let (h, c, d) = setup(&f, "dead").await;
        let p = d.join(REC).display().to_string();
        let x = call(&c, "export", json!({ "path": p, "name": "x" }))
            .await
            .unwrap();
        until(|| f.probes.load(Relaxed) == 1).await;
        tokio::time::sleep(Duration::from_millis(600)).await;
        assert_eq!(f.probes.load(Relaxed), 1);
        *f.probe.lock().unwrap() = Err("the readback timed out".into());
        let y = call(&c, "export", json!({ "path": p, "name": "y" }))
            .await
            .unwrap();
        f.run(0).set(RunSt::Failed("VK_ERROR_DEVICE_LOST".into()));
        let s = reads(&c, &x["job"], "failed").await;
        assert_eq!(s["error"], "VK_ERROR_DEVICE_LOST");
        let s = reads(&c, &y["job"], "failed").await;
        assert_eq!(
            s["error"],
            "the GPU device was lost (the readback timed out)"
        );
        assert!(f.run(1).stop.load(Relaxed) && f.probes.load(Relaxed) == 2);
        assert!(std::fs::read_dir(h.agent()).unwrap().next().is_none());
        *f.gpu.lock().unwrap() = GpuSt::Dead("the readback timed out".into());
        let e = call(&c, "export", json!({ "path": p })).await.unwrap_err();
        assert!(
            e.contains("the GPU device was lost (the readback timed out); GPU tools are off"),
            "{e}"
        );
        drop(c);
        h.shutdown().await;
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The host's end cancels its renders and settles each by identity: one that stops at once, one stuck in the
    /// driver at the deadline, its `.part` removed (its writer would go on into an orphan inode). The records, saved,
    /// say why; idle sleep is allowed again; nothing is left in the agent folder.
    #[tokio::test(flavor = "multi_thread")]
    async fn ends() {
        let f = fake();
        *f.gpu.lock().unwrap() = GpuSt::Ready(info());
        let (h, c, d) = setup(&f, "ends").await;
        let p = d.join(REC).display().to_string();
        let x = call(&c, "export", json!({ "path": p, "name": "x" }))
            .await
            .unwrap();
        let y = call(&c, "export", json!({ "path": p, "name": "y" }))
            .await
            .unwrap();
        f.run(1).stuck.store(true, Relaxed);
        f.run(1).set(running(Phase::Frames, 2));
        until(|| h.sh.keep.lock().unwrap().is_some()).await;
        reads(&c, &y["job"], "running").await;
        drop(c);
        let t = Instant::now();
        h.shutdown().await;
        let e = t.elapsed();
        assert!(
            e >= crate::host::RENDERS && e < crate::host::RENDERS + Duration::from_secs(2),
            "{e:?}"
        );
        assert!(f.run(0).stop.load(Relaxed) && f.run(1).stop.load(Relaxed));
        let (k, _) = Renders::load(h.sim().unwrap().dir.join("mcp-renders")).unwrap();
        let st: Vec<(&str, &str, &str)> = (k.v.iter())
            .map(|r| (r.h.as_str(), r.end.as_ref().unwrap()))
            .map(|(h, e)| (h, e.state.as_str(), e.err.as_str()))
            .collect();
        let (xj, yj) = (x["job"].as_str().unwrap(), y["job"].as_str().unwrap());
        assert_eq!(st, [(xj, "cancelled", ENDED), (yj, "cancelled", ENDED)]);
        assert!(std::fs::read_dir(h.agent()).unwrap().next().is_none());
        assert!(h.sh.keep.lock().unwrap().is_none() && h.sh.rendering.load(Relaxed) == 0);
        let log = std::fs::read_to_string(&h.sim().unwrap().log).unwrap();
        assert!(
            log.contains(&format!("export {yj} still in frames at the host's end")),
            "{log}"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// At the hosting client's end the host stops its renders before the drain: a call waiting for a still answers at
    /// once that the host is shutting down, a parked job is cancelled at once (no `.part`) and the GPU's start halted,
    /// new renders are refused; the other tools serve on until the end, which finds nothing more to do.
    #[tokio::test(flavor = "multi_thread")]
    async fn closing() {
        let f = fake();
        *f.gpu.lock().unwrap() = GpuSt::Ready(info());
        let (h, c, d) = setup(&f, "closing").await;
        let p = d.join(REC).display().to_string();
        let ask = CallToolRequestParams::new("render_still").with_arguments(
            json!({ "path": p, "wait_s": 30 })
                .as_object()
                .unwrap()
                .clone(),
        );
        let mut parked = Value::Null;
        let mut at = Instant::now();
        let drive = async {
            until(|| f.runs.lock().unwrap().len() == 1).await;
            f.run(0).stuck.store(true, Relaxed);
            *f.gpu.lock().unwrap() = GpuSt::Off;
            f.slow.store(1000, Relaxed);
            parked = call(&c, "export", json!({ "path": p })).await.unwrap();
            assert_eq!(parked["phase"], "gpu_starting");
            at = Instant::now();
            h.closing();
        };
        let (r, ()) = tokio::join!(c.call_tool(ask), drive);
        let r = r.unwrap();
        let t = r.content[0].as_text().unwrap().text.clone();
        assert!(r.is_error == Some(true) && t.contains(SHUT), "{t}");
        assert!(
            at.elapsed() < Duration::from_millis(600),
            "{:?}",
            at.elapsed()
        );
        let s = reads(&c, &parked["job"], "cancelled").await;
        assert_eq!(s["note"], ENDED);
        let pd = h.agent().join("VID_20260117_141148_00_635_p360.mp4");
        assert!(!media::part::of(&pd).exists() && h.sh.halt.load(Relaxed));
        assert!(f.run(0).stop.load(Relaxed));
        let e = call(&c, "export", json!({ "path": p })).await.unwrap_err();
        assert!(e.contains(SHUT), "{e}");
        assert!(call(&c, "status", json!({})).await.is_ok());
        // the still, stopped at last, cancelled as its host's
        f.run(0).stuck.store(false, Relaxed);
        let s = reads(&c, &json!(nth(&h, 0)), "cancelled").await;
        assert_eq!(s["note"], ENDED);
        until(|| h.sh.keep.lock().unwrap().is_none()).await;
        drop(c);
        let t = Instant::now();
        h.shutdown().await;
        assert!(t.elapsed() < crate::host::RENDERS, "{:?}", t.elapsed());
        assert_eq!(f.starts.load(Relaxed), 1);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The handle of host `h`'s `n`th render job.
    fn nth(h: &Host, n: usize) -> String {
        h.sh.jobs.lock().unwrap().r.v[n].h.clone()
    }

    /// Names: a matching extension stripped, another refused, the camera's pattern refused (not our tagged stems).
    #[test]
    fn names() {
        for s in [
            "IMG_20260201_111111_00_701",
            "vid_20260117_141148_00_635",
            "LRV_20260117_141148_01_635",
            "PRO_VID_20250101_120000_00_001",
            "PRO_20250101_120000_00_001",
        ] {
            assert!(camera(s), "{s}");
        }
        for s in [
            "VID_20260117_141148_00_635_pano57",
            "VID_2026011_141148_00_635",
            "VID_20260117_141148_00_",
            "clip",
        ] {
            assert!(!camera(s), "{s}");
        }
        let n = name;
        assert_eq!(n("clip.MP4", RKind::Export), Ok("clip".into()));
        assert_eq!(n("take 1.5", RKind::Export), Ok("take 1.5".into()));
        assert_eq!(n("a.jpeg", RKind::Still), Ok("a".into()));
        for (s, w) in [
            (".mp4", "does not begin with a dot"),
            (".x", "does not begin with a dot"),
            ("a.part.mp4", "not a safe file name"),
            ("a/b", "not a safe file name"),
            ("CON", "not a safe file name"),
            ("IMG_20260201_111111_00_701.mp4", "a camera file's name"),
        ] {
            assert!(n(s, RKind::Export).unwrap_err().contains(w), "{s}");
        }
        assert!(
            n(&"x".repeat(121), RKind::Export)
                .unwrap_err()
                .contains("120 bytes at most")
        );
    }

    /// A still's preview: the whole picture at a long edge of 1568, under 1 MB; refused once the name is another file.
    #[test]
    fn previews() {
        let d = std::env::temp_dir().join(format!("x5desk-mcp-rs-pv-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let dst = d.join("s.jpg");
        let px: Vec<u8> = (0..3200 * 1600)
            .flat_map(|i| [(i % 251) as u8, (i / 3200 % 256) as u8, 7, 255])
            .collect();
        let jpg = media::jpeg::encode(&px, 3200, 1600, 92, None).unwrap();
        let p = media::part::create(&dst).unwrap();
        (&p.f).write_all(&jpg).unwrap();
        media::part::publish(&media::part::of(&dst), &dst).unwrap();
        let v = preview(&dst, &p.id).unwrap();
        assert_eq!((v.w, v.h, v.src, v.q), (1568, 784, [3200, 1600], 85));
        assert!(v.jpg.len() < PREVIEW_BYTES);
        let mut out = vec![];
        assert_eq!(
            media::mjpeg::Mjpeg::default()
                .rgba(&v.jpg, &mut out)
                .unwrap(),
            (1568, 784)
        );
        std::fs::remove_file(&dst).unwrap();
        std::fs::write(&dst, &jpg).unwrap();
        assert!(
            preview(&dst, &p.id)
                .unwrap_err()
                .contains("not the still's file any more")
        );
        std::fs::remove_dir_all(&d).unwrap();
    }
}
