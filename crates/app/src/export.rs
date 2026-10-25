//! Export screen (PLAN.md milestone 5, groups 7, 8, 15): jobs made from the player or the library, run one after
//! another on the export thread. A video job is a one-clip editor timeline (crate edit) that crate export renders:
//! 360 or the player's view, H.265 8 / 10-bit or H.264 at a bitrate or a constant QP, the source's or another frame
//! rate, RAFT or template seam, colour matching, stabilisation, a look and adjustments (crate fx). A frame grab is the
//! same job into a JPEG (GPano for 360). Privacy blur: a pass first finds the faces on the clip stitched as the export
//! stitches it (crate track: YuNet on `face_tiles`, linked over frames by `Hide`), then the job's effect hook blurs
//! their regions: `track::Blur` in a 360 frame, [`Blur`] in a flat view (an editor timeline: a pass per clip over its
//! source span, each output frame blurred with its clip's regions in its view). Photos (the camera's equirect JPEGs)
//! render here too: the view through the preview, the effects, the blur, a JPEG. A Deep Track follow view
//! (`Spec::follow`) turns a flat output onto the subject, the track's directions turned into the frame the export
//! stitches in; a Shot Lab job ([`Lab`]) brings its effect's hook maker, run on the export thread for the output's
//! frame size and format.

use crate::{clip, gyro, photo, player::Look};
use eframe::egui;
use glam::{Mat3, Quat, Vec2, Vec3};
use media::mp4::{self, Mp4};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering::Relaxed},
        mpsc,
    },
    thread,
    time::Instant,
};
use stitch::{Ctx, Mode, Preview, View};
use video::{Codec, Gpu, Rate};

const CANCELLED: &str = "cancelled";
/// Face detection: YuNet score (privacy takes the faces it is less sure of too), every few frames (`Hide` bridges
/// them), on an equirect this wide (the face tiles' resolution, crate track).
const MIN: f32 = 0.5;
const STEP: usize = 3;
const FACES_W: u32 = 3072;

/// What a job is made from: the player's or the library's file and how it shows it.
#[derive(Clone, Debug)]
pub struct Spec {
    /// main files in lens order (one with both lenses on the X5), or the photo
    pub files: Vec<PathBuf>,
    pub photo: bool,
    /// clip length, the frame shown (s)
    pub dur: f64,
    pub t: f64,
    pub view: View,
    pub look: Look,
    pub stab: gyro::Mode,
    pub smooth: f64,
    /// a flat output follows this Deep Track subject instead of `view`'s direction
    pub follow: Option<Arc<crate::ai::Track>>,
    /// the player previews the face blur: the export starts with it on
    pub faces: bool,
}

impl Spec {
    /// A library entry not open in the player: the recording's main files, the default view, horizon lock.
    pub fn of(p: &Path) -> Result<Spec, String> {
        let (files, photo, dur) = if photo::is(p) {
            (vec![p.to_path_buf()], true, 0.0)
        } else {
            let w = crate::library::pair(p);
            if w.main.is_empty() {
                return Err(format!(
                    "{}: no full-resolution file to export",
                    p.display()
                ));
            }
            let d = probe(&w.main[0])?.1;
            (w.main, false, d)
        };
        Ok(Spec {
            files,
            photo,
            dur,
            t: 0.0,
            view: Look::Perspective.view(),
            look: Look::Perspective,
            stab: Some(stab::Lock::Horizon),
            smooth: gyro::SMOOTH,
            follow: None,
            faces: false,
        })
    }

    fn name(&self) -> String {
        let n = |p: &PathBuf| {
            p.file_name()
                .map_or(String::new(), |n| n.to_string_lossy().into())
        };
        self.files.iter().map(n).collect::<Vec<_>>().join(" + ")
    }
}

/// `x` with `<preset key>[:<look>][:faces][:noraft][:qp=<n>]` (a look by its name's first word, any case: vivid,
/// warm, cool, film, black, teal).
pub(crate) fn parse(mut x: Set, tok: &str) -> Result<Set, String> {
    let mut w = tok.split(':');
    let k = w.next().unwrap_or("");
    x.apply((PRESETS.iter().position(|p| p.0 == k)).ok_or_else(|| format!("no preset {k}"))?);
    for o in w {
        match o {
            "faces" => x.faces = true,
            "noraft" => x.raft = false,
            _ if o.starts_with("qp=") => {
                x.qp = Some(o[3..].parse().map_err(|_| format!("bad {o}"))?)
            }
            _ => {
                x.look = Some(
                    (fx::Look::ALL.into_iter())
                        .find(|l| l.name().split(' ').next().unwrap().eq_ignore_ascii_case(o))
                        .ok_or_else(|| format!("no look {o}"))?,
                )
            }
        }
    }
    Ok(x)
}

/// The source's frame rate (first video track: timescale over the first frame's duration) and length (s).
pub(crate) fn probe(p: &Path) -> Result<(edit::Rate, f64), String> {
    let m = Mp4::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let t = (m.tracks.iter())
        .find(|t| matches!(t.codec, mp4::Codec::Hevc { .. } | mp4::Codec::Avc { .. }))
        .ok_or_else(|| format!("{}: no video track", p.display()))?;
    match (t.samples.first(), t.samples.last()) {
        (Some(a), Some(b)) if a.dur > 0 => Ok((
            edit::Rate {
                num: t.timescale,
                den: a.dur,
            },
            (b.pts + b.dur as i64 + t.shift(m.timescale)) as f64 / t.timescale as f64,
        )),
        _ => Err(format!("{}: empty video track", p.display())),
    }
}

/// Presets: key (test hook, file names), name, 360, size, codec, 10-bit, Mbit/s.
#[rustfmt::skip]
pub const PRESETS: [(&str, &str, bool, [u32; 2], Codec, bool, f32); 7] = [
    ("pano57",    "360 5.7K HEVC 10-bit",       true,  [5760, 2880], Codec::H265, true,  120.0),
    ("pano4k",    "360 4K HEVC",                true,  [3840, 1920], Codec::H265, false, 60.0),
    ("pano8k",    "360 8K HEVC 10-bit",         true,  [7680, 3840], Codec::H265, true,  200.0),
    ("pano4k264", "360 4K H.264",               true,  [3840, 1920], Codec::H264, false, 60.0),
    ("flat1080",  "Flat 1080p H.264",           false, [1920, 1080], Codec::H264, false, 20.0),
    ("flat4k",    "Flat 4K HEVC",               false, [3840, 2160], Codec::H265, false, 50.0),
    ("vert1080",  "Vertical 1080 × 1920 H.264", false, [1080, 1920], Codec::H264, false, 20.0),
];
/// 360 widths (photos also at the camera's 11904) and flat sizes.
const PANO: [u32; 4] = [3840, 5760, 7680, 11904];
const FLAT: [[u32; 2]; 6] = [
    [1280, 720],
    [1920, 1080],
    [2560, 1440],
    [3840, 2160],
    [1080, 1920],
    [1080, 1080],
];
/// Output frame rates; None: the source's.
const FPS: [(Option<(u32, u32)>, &str); 6] = [
    (None, "Source"),
    (Some((24, 1)), "24"),
    (Some((25, 1)), "25"),
    (Some((30000, 1001)), "29.97"),
    (Some((30, 1)), "30"),
    (Some((60, 1)), "60"),
];
/// H.264 and H.265 encode up to these sizes (RADV).
const H264_MAX: u32 = 4096;
const H265_MAX: u32 = 8192;

/// Export settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Set {
    /// the preset these came from, while unchanged
    pub preset: Option<usize>,
    /// 360 equirect `w` × `w / 2`, else a flat view `w` × `h`
    pub pano: bool,
    pub w: u32,
    pub h: u32,
    pub codec: Codec,
    pub deep: bool,
    /// CBR Mbit/s, or a constant QP
    pub mbps: f32,
    pub qp: Option<u8>,
    pub fps: Option<(u32, u32)>,
    pub raft: bool,
    pub colour: bool,
    pub stab: gyro::Mode,
    pub smooth: f64,
    pub look: Option<fx::Look>,
    pub amt: f32,
    pub adj: fx::Params,
    pub faces: bool,
    /// JPEG quality of grabs and photos
    pub jq: u8,
}

impl Set {
    pub fn apply(&mut self, i: usize) {
        let (_, _, pano, [w, h], codec, deep, mbps) = PRESETS[i];
        (
            self.pano, self.w, self.h, self.codec, self.deep, self.mbps, self.qp,
        ) = (pano, w, h, codec, deep, mbps, None);
        self.preset = Some(i);
    }

    fn size(&self) -> [u32; 2] {
        [self.w, if self.pano { self.w / 2 } else { self.h }]
    }

    /// Why these cannot export `s` (a grab: one frame), if they cannot.
    fn check(&self, s: &Spec, grab: bool, range: [f64; 2]) -> Option<String> {
        let [w, h] = self.size();
        let video = !grab && !s.photo;
        if video && self.codec == Codec::H264 && w.max(h) > H264_MAX {
            return Some(format!(
                "H.264 encodes up to {H264_MAX} px: H.265 for {w} × {h}"
            ));
        }
        if video && w.max(h) > H265_MAX {
            return Some(format!("Video encodes up to {H265_MAX} px wide"));
        }
        if video && range[1] - range[0] < 0.01 {
            return Some("The range is empty.".into());
        }
        None
    }
}

impl Default for Set {
    fn default() -> Set {
        let mut s = Set {
            preset: None,
            pano: true,
            w: 0,
            h: 0,
            codec: Codec::H265,
            deep: false,
            mbps: 0.0,
            qp: None,
            fps: None,
            raft: true,
            colour: true,
            stab: Some(stab::Lock::Horizon),
            smooth: gyro::SMOOTH,
            look: None,
            amt: 1.0,
            adj: fx::Params::default(),
            faces: false,
            jq: 92,
        };
        s.apply(0);
        s
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum St {
    Queued,
    /// face detection: frames done of all
    Faces(usize, usize),
    Run(export::Progress),
    /// the final progress, wall time (s)
    Done(export::Progress, f64),
    Failed(String),
    Cancelled,
}

/// A queued job: what it makes, its state (the export thread writes it), its cancel flag, a note (faces found).
pub struct Item {
    pub name: String,
    pub path: PathBuf,
    st: Mutex<St>,
    note: Mutex<String>,
    stop: AtomicBool,
}

impl Item {
    pub fn st(&self) -> St {
        self.st.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn set(&self, s: St) {
        *self.st.lock().unwrap_or_else(|e| e.into_inner()) = s;
    }

    fn note(&self) -> String {
        self.note.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn over(&self) -> bool {
        matches!(self.st(), St::Done(..) | St::Failed(_) | St::Cancelled)
    }
}

struct Task {
    it: Arc<Item>,
    spec: Spec,
    set: Set,
    /// in and out (s); a grab takes the frame at `spec.t`
    range: [f64; 2],
    grab: bool,
    /// an editor timeline instead of the source's range
    ed: Option<Box<Tl>>,
    /// a Shot Lab effect on the frames
    lab: Option<Lab>,
}

/// Makes an effect hook for frames of this size and format (`export::Fx`), on the export thread.
pub type Maker =
    Box<dyn FnOnce(&Gpu, u32, u32, wgpu::TextureFormat) -> Result<export::Hook, String> + Send>;

/// A Shot Lab effect for a job: its hook's maker (taken by the job); `flat`: what it makes is no sphere
/// (`export::Job::flat`); `key` names the output file.
pub struct Lab {
    pub key: &'static str,
    pub name: String,
    pub hook: Mutex<Option<Maker>>,
    pub flat: bool,
}

/// An editor timeline to export (`export::Cut::Edit`) with the editor's timeline-wide effects and stabilisation.
#[derive(Clone)]
pub struct Tl {
    /// the project's name (output file names)
    pub name: String,
    pub tl: edit::Timeline,
    /// per `tl.files` entry: its files (a recording's main files; the music file)
    pub files: Vec<Vec<PathBuf>>,
    pub looks: std::collections::HashMap<String, fx::Cube>,
    pub adjust: fx::Params,
    pub over: Vec<(fx::Image, fx::Place, f32)>,
    pub stab: gyro::Mode,
    pub smooth: f64,
}

pub struct Exp {
    ctx: egui::Context,
    pub src: Option<Spec>,
    pub set: Set,
    range: [f64; 2],
    dest: PathBuf,
    items: Vec<Arc<Item>>,
    tx: mpsc::Sender<Task>,
    th: Option<thread::JoinHandle<()>>,
    dlg: Option<mpsc::Receiver<Option<PathBuf>>>,
    pub err: Option<String>,
    /// test hook: print each job's end (how many printed)
    log: Option<usize>,
}

impl Exp {
    /// `dest`: the output directory (default `~/Videos`); `log`: print each job's end (test hook).
    pub fn new(
        g: &Arc<Gpu>,
        cx: &Ctx,
        ctx: &egui::Context,
        dest: Option<PathBuf>,
        log: bool,
    ) -> Exp {
        let (tx, rx) = mpsc::channel();
        let (g, cx, c) = (g.clone(), cx.clone(), ctx.clone());
        let th = thread::Builder::new()
            .name("export".into())
            .spawn(move || worker(g, cx, c, rx))
            .expect("export thread");
        Exp {
            ctx: ctx.clone(),
            src: None,
            set: Set::default(),
            range: [0.0; 2],
            dest: dest
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Videos")))
                .unwrap_or_else(|| ".".into()),
            items: vec![],
            tx,
            th: Some(th),
            dlg: None,
            err: None,
            log: log.then_some(0),
        }
    }

    /// Makes `s` the source: the whole clip, its stabilisation.
    pub fn source(&mut self, s: Spec) {
        self.range = [0.0, s.dur];
        if !s.photo {
            (self.set.stab, self.set.smooth) = (s.stab, s.smooth);
        }
        self.set.faces |= s.faces;
        self.src = Some(s);
        self.err = None;
    }

    /// Every job done, failed or cancelled.
    pub fn idle(&self) -> bool {
        self.items.iter().all(|i| i.over())
    }

    /// A free output path: `<dest>/<source stem>_<pre><tag>.<ext>`, numbered when taken (on disk or by a queued job).
    fn path(&self, s: &Spec, x: &Set, grab: bool, pre: &str) -> PathBuf {
        let stem = s.files[0]
            .file_stem()
            .map_or("x5".into(), |n| n.to_string_lossy());
        let [w, h] = x.size();
        let kind = match x.pano {
            true => format!("360_{w}"),
            false => format!("flat_{w}x{h}"),
        };
        let tag = match (grab && !s.photo, x.preset) {
            (true, _) => format!("{:.2}s_{kind}", s.t),
            (false, Some(i)) if !s.photo => PRESETS[i].0.to_string(),
            _ => kind,
        };
        let ext = if grab || s.photo { "jpg" } else { "mp4" };
        (1..)
            .map(|i| match i {
                1 => self.dest.join(format!("{stem}_{pre}{tag}.{ext}")),
                _ => self.dest.join(format!("{stem}_{pre}{tag}-{i}.{ext}")),
            })
            .find(|p| !p.exists() && self.items.iter().all(|i| &i.path != p))
            .unwrap()
    }

    /// Queues the source with the settings `x`: an export, or (`grab`) the frame shown as a JPEG.
    pub fn queue(&mut self, x: Set, grab: bool) -> Result<(), String> {
        let s = self
            .src
            .clone()
            .ok_or("Open a file in the library first.")?;
        self.push(s, x, self.range, grab, None)
    }

    /// Queues Shot Lab effect `l` on source span `range` (s) of `s` with the settings `x`.
    pub fn lab(&mut self, s: Spec, x: Set, range: [f64; 2], l: Lab) -> Result<(), String> {
        if s.photo || s.files.len() > 1 {
            return Err("Shot Lab works on X5 videos with both lenses in one file".into());
        }
        if self.src.is_none() {
            self.source(s.clone());
        }
        self.push(s, x, range, false, Some(l))
    }

    fn push(
        &mut self,
        s: Spec,
        x: Set,
        range: [f64; 2],
        grab: bool,
        lab: Option<Lab>,
    ) -> Result<(), String> {
        if let Some(e) = x.check(&s, grab, range) {
            return Err(e);
        }
        std::fs::create_dir_all(&self.dest).map_err(|e| format!("{}: {e}", self.dest.display()))?;
        let pre = match (&lab, &s.follow) {
            (Some(l), _) => format!("{}_", l.key),
            (_, Some(_)) if !x.pano => "follow_".into(),
            _ => String::new(),
        };
        let path = self.path(&s, &x, grab, &pre);
        let what = match (grab, s.photo, x.preset) {
            (true, false, _) => format!(
                "frame at {:.2} s, {}",
                s.t,
                if x.pano { "360" } else { "view" }
            ),
            (_, false, Some(i)) => PRESETS[i].1.into(),
            _ => format!(
                "{} {} × {}",
                if x.pano { "360" } else { "flat" },
                x.size()[0],
                x.size()[1]
            ),
        };
        let what = match (&lab, &s.follow) {
            (Some(l), _) => format!("{} · {what}", l.name),
            (_, Some(_)) if !x.pano => format!("{what}, following the tracked subject"),
            _ => what,
        };
        let it = Arc::new(Item {
            name: format!("{}: {what}", s.name()),
            path,
            st: Mutex::new(St::Queued),
            note: Mutex::new(String::new()),
            stop: AtomicBool::new(false),
        });
        self.items.push(it.clone());
        self.tx
            .send(Task {
                it,
                spec: s,
                set: x,
                range,
                grab,
                ed: None,
                lab,
            })
            .map_err(|_| "the export thread is gone".to_string())
    }

    /// Queues the editor timeline `e` with the settings `x` (output, codec, seam, colour matching; the timeline brings
    /// its own looks, adjustments and stabilisation): `<dest>/<project>_<tag>.mp4`.
    pub fn timeline(&mut self, e: Tl, x: Set) -> Result<(), String> {
        let (dur, f) = (
            e.tl.dur(),
            e.tl.files.first().ok_or("The timeline is empty.")?,
        );
        if e.tl.is_empty() {
            return Err("The timeline is empty.".into());
        }
        let s = Spec {
            files: vec![PathBuf::from(&f.path)],
            photo: false,
            dur,
            t: 0.0,
            view: Look::Perspective.view(),
            look: Look::Perspective,
            stab: e.stab,
            smooth: e.smooth,
            follow: None,
            faces: false,
        };
        if let Some(m) = x.check(&s, false, [0.0, dur]) {
            return Err(m);
        }
        std::fs::create_dir_all(&self.dest).map_err(|e| format!("{}: {e}", self.dest.display()))?;
        let [w, h] = x.size();
        let tag = match (x.preset, x.pano) {
            (Some(i), _) => PRESETS[i].0.to_string(),
            (None, true) => format!("360_{w}"),
            (None, false) => format!("flat_{w}x{h}"),
        };
        let path = (1..)
            .map(|i| match i {
                1 => self.dest.join(format!("{}_{tag}.mp4", e.name)),
                _ => self.dest.join(format!("{}_{tag}-{i}.mp4", e.name)),
            })
            .find(|p| !p.exists() && self.items.iter().all(|i| &i.path != p))
            .unwrap();
        let what = match x.preset {
            Some(i) => PRESETS[i].1.to_string(),
            None => format!("{} {w} × {h}", if x.pano { "360" } else { "flat" }),
        };
        let it = Arc::new(Item {
            name: format!(
                "{} (timeline, {} clips, {dur:.2} s): {what}",
                e.name,
                e.tl.clips.len()
            ),
            path,
            st: Mutex::new(St::Queued),
            note: Mutex::new(String::new()),
            stop: AtomicBool::new(false),
        });
        self.items.push(it.clone());
        self.tx
            .send(Task {
                it,
                spec: s,
                set: x,
                range: [0.0, dur],
                grab: false,
                ed: Some(Box::new(e)),
                lab: None,
            })
            .map_err(|_| "the export thread is gone".to_string())
    }

    /// A grab of the frame shown: 360 (`pano`) or the view, at the settings' size when they are of that kind (else
    /// 7680 or 1920 × 1080).
    pub fn grab(&mut self, pano: bool) -> Result<(), String> {
        let mut x = self.set;
        if x.pano != pano {
            (x.pano, x.w, x.h) = (pano, if pano { 7680 } else { 1920 }, 1080);
        }
        x.preset = None;
        self.queue(x, true)
    }

    /// Test hook: queues an export with `parse(tok)` over the settings.
    pub fn auto(&mut self, tok: &str) -> Result<(), String> {
        let x = parse(self.set, tok)?;
        self.queue(x, false)
    }

    /// Every frame: the hook's log, a finished folder dialog.
    pub fn tick(&mut self) {
        if let Some(r) = &self.dlg
            && let Ok(p) = r.try_recv()
        {
            self.dlg = None;
            if let Some(p) = p {
                self.dest = p;
            }
        }
        let Some(n) = &mut self.log else { return };
        while let Some(i) = self.items.get(*n).filter(|i| i.over()) {
            match i.st() {
                St::Done(p, s) => println!(
                    "x5desk export: {}: done, {} frames in {s:.1} s ({:.2} frames/s steady) → {}{}",
                    i.name,
                    p.done,
                    p.fps,
                    i.path.display(),
                    match i.note() {
                        n if n.is_empty() => n,
                        n => format!(" ({n})"),
                    }
                ),
                s => println!("x5desk export: {}: {s:?}", i.name),
            }
            *n += 1;
        }
    }

    /// The screen; returns a finished output to open in the library.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<PathBuf> {
        let mut open = None;
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| {
                ui.heading("Export");
                self.settings(ui);
                ui.separator();
                open = self.queue_ui(ui);
            });
        open
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        let Some(s) = &self.src else {
            ui.label("Open a video or a photo in the library and press Export…, or pick Export… from a file's menu.");
            return;
        };
        let (photo, dur) = (s.photo, s.dur);
        ui.horizontal_wrapped(|ui| {
            ui.label("Source");
            ui.monospace(s.name());
            if !photo {
                ui.label(format!("{dur:.2} s"));
            }
            if !self.set.pano {
                let v = &s.view;
                ui.label(format!(
                    "· view: {} yaw {:.0}° pitch {:.0}° fov {:.0}°",
                    Look::ALL.iter().find(|l| l.0 == s.look).map_or("", |l| l.1),
                    v.yaw,
                    v.pitch,
                    v.fov
                ));
                if s.follow.is_some() {
                    ui.label("· following the tracked subject");
                }
            }
        });
        let x0 = self.set;
        let x = &mut self.set;
        egui::Grid::new("export set")
            .num_columns(2)
            .spacing([16.0, 6.0])
            .show(ui, |ui| {
                if !photo {
                    ui.label("Range");
                    ui.horizontal(|ui| {
                        let [a, b] = &mut self.range;
                        let dv = |v| egui::DragValue::new(v).range(0.0..=dur).speed(0.05).max_decimals(2).suffix(" s");
                        ui.add(dv(a));
                        ui.label("to");
                        ui.add(dv(b));
                        *a = a.min(*b);
                        if ui.button("Whole clip").clicked() {
                            self.range = [0.0, dur];
                        }
                    });
                    ui.end_row();
                    ui.label("Preset");
                    egui::ComboBox::from_id_salt("preset")
                        .selected_text(x.preset.map_or("Custom", |i| PRESETS[i].1))
                        .show_ui(ui, |ui| {
                            for (i, p) in PRESETS.iter().enumerate() {
                                if ui.selectable_label(x.preset == Some(i), p.1).clicked() {
                                    x.apply(i);
                                }
                            }
                        });
                    ui.end_row();
                }
                ui.label("Output");
                ui.horizontal_wrapped(|ui| {
                    if ui.selectable_label(x.pano, "360").clicked() && !x.pano {
                        (x.pano, x.w) = (true, 5760);
                    }
                    if ui.selectable_label(!x.pano, "Flat view").clicked() && x.pano {
                        (x.pano, x.w, x.h) = (false, 1920, 1080);
                    }
                    let [w, h] = x.size();
                    egui::ComboBox::from_id_salt("size")
                        .selected_text(format!("{w} × {h}"))
                        .show_ui(ui, |ui| match x.pano {
                            true => {
                                for w in PANO.into_iter().filter(|&w| photo || w < 11904) {
                                    ui.selectable_value(&mut x.w, w, format!("{w} × {}", w / 2));
                                }
                            }
                            false => {
                                for [w, h] in FLAT {
                                    if ui.selectable_label([x.w, x.h] == [w, h], format!("{w} × {h}")).clicked() {
                                        (x.w, x.h) = (w, h);
                                    }
                                }
                            }
                        });
                });
                ui.end_row();
                if photo {
                    ui.label("JPEG quality");
                    ui.add(egui::Slider::new(&mut x.jq, 50..=100));
                    ui.end_row();
                } else {
                    ui.label("Codec");
                    ui.horizontal_wrapped(|ui| {
                        ui.selectable_value(&mut x.codec, Codec::H265, "H.265");
                        ui.selectable_value(&mut x.codec, Codec::H264, "H.264");
                        let h265 = x.codec == Codec::H265;
                        x.deep &= h265;
                        ui.add_enabled(h265, egui::Checkbox::new(&mut x.deep, "10-bit"))
                            .on_hover_text("H.265 Main 10 (H.264 is 8-bit)");
                        ui.separator();
                        let mut q = x.qp.is_some();
                        ui.selectable_value(&mut q, false, "Bitrate");
                        ui.selectable_value(&mut q, true, "Constant quality");
                        match (q, &mut x.qp) {
                            (false, qp) => {
                                *qp = None;
                                ui.add(egui::DragValue::new(&mut x.mbps).range(1.0..=400.0).suffix(" Mbit/s"));
                            }
                            (true, qp) => {
                                let v = qp.get_or_insert(24);
                                ui.add(egui::DragValue::new(v).range(10..=45).prefix("QP "))
                                    .on_hover_text("Lower is better and bigger");
                            }
                        }
                        ui.separator();
                        egui::ComboBox::from_id_salt("fps")
                            .selected_text(FPS.iter().find(|f| f.0 == x.fps).map_or("", |f| f.1))
                            .show_ui(ui, |ui| {
                                for (f, n) in FPS {
                                    ui.selectable_value(&mut x.fps, f, n);
                                }
                            });
                        ui.label("fps");
                    });
                    ui.end_row();
                    ui.label("Stitch");
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut x.raft, "RAFT seam")
                            .on_hover_text("Optical flow at the seam on every frame (slower); off: the template seam");
                        ui.checkbox(&mut x.colour, "Colour matching")
                            .on_hover_text("Match the two lenses' colour at the seam");
                    });
                    ui.end_row();
                    ui.label("Stabilisation");
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_id_salt("xstab")
                            .selected_text(gyro::MODES.iter().find(|m| m.0 == x.stab).map_or("", |m| m.1))
                            .show_ui(ui, |ui| {
                                for (m, n) in gyro::MODES {
                                    ui.selectable_value(&mut x.stab, m, n);
                                }
                            });
                        ui.add_enabled(
                            x.stab.is_some(),
                            egui::Slider::new(&mut x.smooth, 0.0..=gyro::SMOOTH_MAX)
                                .step_by(0.05)
                                .suffix(" s")
                                .text("Smoothing"),
                        );
                    });
                    ui.end_row();
                }
                ui.label("Look");
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("look")
                        .selected_text(x.look.map_or("None", |l| l.name()))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut x.look, None, "None");
                            for l in fx::Look::ALL {
                                ui.selectable_value(&mut x.look, Some(l), l.name());
                            }
                        });
                    ui.add_enabled(
                        x.look.is_some(),
                        egui::Slider::new(&mut x.amt, 0.0..=1.0).text("Intensity"),
                    );
                });
                ui.end_row();
                ui.label("Adjust");
                ui.collapsing("Colour adjustments", |ui| {
                    let a = &mut x.adj;
                    for (v, n, r) in [
                        (&mut a.exposure, "Exposure (EV)", 3.0),
                        (&mut a.temperature, "Temperature", 1.0),
                        (&mut a.tint, "Tint", 1.0),
                        (&mut a.contrast, "Contrast", 1.0),
                        (&mut a.highlights, "Highlights", 1.0),
                        (&mut a.shadows, "Shadows", 1.0),
                        (&mut a.brightness, "Brightness", 1.0),
                        (&mut a.saturation, "Saturation", 1.0),
                        (&mut a.vibrance, "Vibrance", 1.0),
                        (&mut a.clarity, "Clarity", 1.0),
                        (&mut a.brilliance, "Brilliance", 1.0),
                    ] {
                        ui.add(egui::Slider::new(v, -r..=r).text(n));
                    }
                    for (v, n) in [
                        (&mut a.sharpen, "Sharpen"),
                        (&mut a.purple, "Purple fringe"),
                        (&mut a.auto, "Auto colour"),
                        (&mut a.water, "Underwater"),
                    ] {
                        ui.add(egui::Slider::new(v, 0.0..=1.0).text(n));
                    }
                    if ui.button("Reset").clicked() {
                        *a = fx::Params::default();
                    }
                });
                ui.end_row();
                ui.label("Privacy");
                ui.checkbox(&mut x.faces, "Blur faces")
                    .on_hover_text("Find faces first (YuNet), then blur them in every frame");
                ui.end_row();
                ui.label("Destination");
                ui.horizontal_wrapped(|ui| {
                    ui.monospace(self.dest.display().to_string());
                    if ui.button("Change…").clicked() && self.dlg.is_none() {
                        let (tx, rx) = mpsc::channel();
                        let (ctx, d) = (self.ctx.clone(), self.dest.clone());
                        // the portal dialog blocks: off the UI thread
                        thread::spawn(move || {
                            let _ = tx.send(rfd::FileDialog::new().set_directory(d).pick_folder());
                            ctx.request_repaint();
                        });
                        self.dlg = Some(rx);
                    }
                });
                ui.end_row();
            });
        // an output setting changed by hand: no preset any more
        let x = &mut self.set;
        if x.preset.is_some()
            && (x.pano, x.w, x.h, x.codec, x.deep, x.mbps, x.qp)
                != (x0.pano, x0.w, x0.h, x0.codec, x0.deep, x0.mbps, x0.qp)
            && x.preset == x0.preset
        {
            x.preset = None;
        }
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            let s = self.src.as_ref().unwrap();
            let (why, t) = (self.set.check(s, false, self.range), s.t);
            let r = ui
                .add_enabled(
                    why.is_none(),
                    egui::Button::new(if photo { "Export photo" } else { "Export" }),
                )
                .on_disabled_hover_text(why.unwrap_or_default());
            let mut r = if r.clicked() {
                Some(self.queue(self.set, false))
            } else {
                None
            };
            if !photo {
                if ui
                    .button("Grab 360")
                    .on_hover_text(format!("The frame at {t:.2} s as a 360 JPEG"))
                    .clicked()
                {
                    r = Some(self.grab(true));
                }
                if ui
                    .button("Grab view")
                    .on_hover_text(format!("The view of the frame at {t:.2} s as a JPEG"))
                    .clicked()
                {
                    r = Some(self.grab(false));
                }
            }
            if let Some(r) = r {
                self.err = r.err();
            }
        });
        if let Some(e) = &self.err {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
    }

    fn queue_ui(&mut self, ui: &mut egui::Ui) -> Option<PathBuf> {
        let mut open = None;
        ui.horizontal(|ui| {
            ui.heading("Queue");
            if ui.button("Clear finished").clicked() {
                self.items.retain(|i| !i.over());
                if let Some(n) = &mut self.log {
                    *n = 0;
                }
            }
        });
        if self.items.is_empty() {
            ui.label("Nothing queued.");
            return None;
        }
        let mmss = |s: f64| format!("{}:{:02}", (s / 60.0) as u32, (s % 60.0) as u32);
        // the bar comes last: only the last grid column gets the width that is left
        egui::Grid::new("export queue")
            .num_columns(3)
            .striped(true)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                for i in &self.items {
                    ui.label(&i.name)
                        .on_hover_text(i.path.display().to_string());
                    let st = i.st();
                    ui.horizontal(|ui| match &st {
                        St::Queued | St::Faces(..) | St::Run(_) => {
                            if ui.button("Cancel").clicked() {
                                i.stop.store(true, Relaxed);
                            }
                        }
                        St::Done(..) => {
                            if ui
                                .button("Open")
                                .on_hover_text("Open it in the library")
                                .clicked()
                            {
                                open = Some(i.path.clone());
                            }
                        }
                        _ => {}
                    });
                    let (f, s) = match &st {
                        St::Queued => (0.0, "queued".into()),
                        St::Faces(k, n) => (
                            *k as f32 / (*n).max(1) as f32,
                            format!("finding faces: {k} of {n} frames"),
                        ),
                        St::Run(p) if p.done == 0 => (0.0, "starting…".into()),
                        St::Run(p) => (
                            p.done as f32 / p.total.max(1) as f32,
                            format!(
                                "{} of {} frames · {:.1} frames/s · {} left",
                                p.done,
                                p.total,
                                p.fps,
                                mmss(p.eta)
                            ),
                        ),
                        St::Done(p, s) => (
                            1.0,
                            match (p.total, i.note()) {
                                (1, n) if n.is_empty() => format!("done in {}", mmss(*s)),
                                (1, n) => format!("done in {} · {n}", mmss(*s)),
                                (_, n) if n.is_empty() => {
                                    format!("done: {} frames in {}", p.done, mmss(*s))
                                }
                                (_, n) => format!("done: {} frames in {} · {n}", p.done, mmss(*s)),
                            },
                        ),
                        St::Failed(e) => (0.0, format!("failed: {e}")),
                        St::Cancelled => (0.0, "cancelled".into()),
                    };
                    let bar = egui::ProgressBar::new(f).text(s);
                    ui.add(match st {
                        St::Failed(_) => bar.fill(ui.visuals().error_fg_color),
                        _ => bar,
                    });
                    ui.end_row();
                }
            });
        open
    }
}

impl Drop for Exp {
    /// Cancels everything and waits for the running job to stop (it removes its partial file).
    fn drop(&mut self) {
        for i in &self.items {
            i.stop.store(true, Relaxed);
        }
        self.tx = mpsc::channel().0;
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
    }
}

/// Runs `f` with this thread's wgpu errors and panics captured (they fail the job instead of reaching the device's
/// handler, which panics), as crate export does on its threads.
pub(crate) fn scoped<T>(
    dev: &wgpu::Device,
    f: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    use wgpu::ErrorFilter as E;
    let s = [E::Validation, E::OutOfMemory, E::Internal].map(|f| dev.push_error_scope(f));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|x| {
        let m = (x.downcast_ref::<String>().cloned())
            .or_else(|| x.downcast_ref::<&str>().map(|s| s.to_string()));
        Err(format!("panic: {}", m.unwrap_or_default()))
    });
    let mut e = None;
    for g in s.into_iter().rev() {
        if let Some(x) = pollster::block_on(g.pop()) {
            e.get_or_insert(x);
        }
    }
    match e {
        Some(e) => Err(format!("wgpu: {e}")),
        None => r,
    }
}

/// What the export thread keeps between jobs: the RAFT model, the effects' pipelines (the face detector is the app's
/// `ai::MODELS`).
#[derive(Default)]
struct Keep {
    raft: Option<stitch::Raft>,
    fx: Option<fx::Ctx>,
}

fn worker(g: Arc<Gpu>, cx: Ctx, ctx: egui::Context, rx: mpsc::Receiver<Task>) {
    let mut k = Keep::default();
    for t in rx {
        if t.it.stop.load(Relaxed) {
            t.it.set(St::Cancelled);
            continue;
        }
        let t0 = Instant::now();
        let r = scoped(&g.device, || match (&t.ed, t.spec.photo) {
            (Some(e), _) => k.timeline(&g, &cx, &t, e, &ctx),
            (_, true) => k.photo(&g, &cx, &t),
            _ => k.video(&g, &cx, &t, &ctx),
        });
        t.it.set(match r {
            Ok(p) => St::Done(p, t0.elapsed().as_secs_f64()),
            Err(_) if t.it.stop.load(Relaxed) => St::Cancelled,
            Err(e) => St::Failed(e),
        });
        ctx.request_repaint();
    }
}

/// Face regions per frame.
type Regs = Vec<Vec<track::Region>>;

impl Keep {
    fn video(
        &mut self,
        g: &Arc<Gpu>,
        cx: &Ctx,
        t: &Task,
        ctx: &egui::Context,
    ) -> Result<export::Progress, String> {
        let (s, x) = (&t.spec, &t.set);
        let stab = x.stab.map(|l| (l, x.smooth));
        let (src, _) = probe(&s.files[0])?;
        let fps = x.fps.map_or(src, |(num, den)| edit::Rate { num, den });
        // a grab: one output frame from the frame shown
        let [a, b] = match t.grab {
            true => [s.t, (s.t + fps.t(1)).min(s.dur)],
            false => t.range,
        };
        let a = a.min(b - fps.t(1)).max(0.0);
        let faces = match x.faces {
            true => Some(self.faces(g, cx, t, &s.files, stab, [a, b], src, &mut [0; 5], ctx)?),
            false => None,
        };
        let mut tl = edit::Timeline::new(fps);
        tl.files.push(edit::File {
            path: s.files[0].to_string_lossy().into(),
            dur: s.dur,
        });
        let id = x.look.map_or("", |l| l.name());
        tl.clips.push(edit::Clip {
            look: id.into(),
            amt: x.amt,
            ..edit::Clip::new(0, a, b)
        });
        tl.fix();
        let e = export::Edit {
            tl,
            files: vec![s.files.clone()],
            looks: x
                .look
                .map(|l| (id.to_string(), l.cube()))
                .into_iter()
                .collect(),
            adjust: x.adj,
            over: vec![],
            abr: 192_000,
        };
        let v = s.view;
        // a followed subject: its directions turned into the frame this export stitches in
        let fol = match (&s.follow, x.pano) {
            (Some(f), false) => {
                let q = crate::ai::rots(&shotlab::Clip::open(&s.files[0])?, (x.stab, x.smooth))?;
                Some((f.clone(), Arc::<[Quat]>::from(q)))
            }
            _ => None,
        };
        // output time t shows source a + t (one clip at 1×)
        let at = move |t: f64| {
            let d = fol.as_ref().and_then(|(f, q)| {
                let k = f.frame(a + t);
                Some(f.turn(k, *q.get(k)?)? * f.dir(k)?)
            });
            d.map_or(v, |d| {
                let [yaw, pitch] = track::yaw_pitch(d);
                View { yaw, pitch, ..v }
            })
        };
        let deep = x.deep && x.codec == Codec::H265 && !t.grab;
        let [w, h] = x.size();
        let fmt = match deep {
            true => wgpu::TextureFormat::Rgba16Float,
            false => wgpu::TextureFormat::Rgba8Unorm,
        };
        let mut fxs: Vec<export::Hook> = vec![];
        if let Some(f) = faces {
            // the blur in the view of each output frame (the follow view turns)
            let at = at.clone();
            fxs.push(hook(g, !x.pano, vec![f], move |_, t| {
                Some((0, a + t, at(t)))
            }));
        }
        let flat = t.lab.as_ref().is_some_and(|l| l.flat);
        if let Some(l) = &t.lab {
            // the hook is made here, on the export thread, from the maker the job carries
            let mk = l.hook.lock().unwrap_or_else(|e| e.into_inner()).take();
            fxs.push((mk.ok_or("the effect ran already")?)(g, w, h, fmt)?);
        }
        let job = export::Job {
            cut: export::Cut::Edit(Box::new(e)),
            path: t.it.path.clone(),
            out: match x.pano {
                true => export::Out::Pano(x.w),
                false => export::Out::Flat {
                    w: x.w,
                    h: x.h,
                    view: Some(Box::new(at)),
                },
            },
            enc: match t.grab {
                true => export::Enc::Jpeg(x.jq),
                false => export::Enc::Video {
                    codec: x.codec,
                    deep: x.deep && x.codec == Codec::H265,
                    rate: x
                        .qp
                        .map_or(Rate::Cbr((x.mbps as f64 * 1e6) as u64), Rate::Qp),
                    gop: ((fps.num + fps.den / 2) / fps.den).max(1),
                },
            },
            raft: x.raft,
            colour: x.colour.then_some(0.1),
            stab,
            fx: match fxs.len() {
                0 => None,
                1 => fxs.pop(),
                _ => Some(Box::new(move |f: export::Fx| {
                    let export::Fx { enc, tex, k, t } = f;
                    for h in &mut fxs {
                        h(export::Fx {
                            enc: &mut *enc,
                            tex,
                            k,
                            t,
                        });
                    }
                })),
            },
            flat,
        };
        let (it, mut last) = (&t.it, Instant::now());
        export::run(g, cx, &mut self.raft, job, &it.stop, &mut |p| {
            it.set(St::Run(p));
            if last.elapsed().as_millis() >= 100 {
                last = Instant::now();
                ctx.request_repaint();
            }
        })
    }

    /// An editor timeline: its cuts, speeds, freeze frames, views (flat output), looks and soundtrack through
    /// `export::Cut::Edit`, with the job's output, codec, seam and face blur settings.
    fn timeline(
        &mut self,
        g: &Arc<Gpu>,
        cx: &Ctx,
        t: &Task,
        e: &Tl,
        ctx: &egui::Context,
    ) -> Result<export::Progress, String> {
        let (x, fps) = (&t.set, e.tl.out.fps);
        let stab = e.stab.map(|l| (l, e.smooth));
        let fx = match x.faces {
            true => {
                // a face pass per clip over its source span; an output frame takes its clip's regions and view
                let (mut sets, mut sum) = (vec![], [0; 5]);
                for c in &e.tl.clips {
                    let f = &e.files[c.file as usize];
                    let (r, _) = probe(&f[0])?;
                    sets.push(self.faces(g, cx, t, f, stab, [c.a, c.b], r, &mut sum, ctx)?);
                }
                let tl = e.tl.clone();
                Some(hook(g, !x.pano, sets, move |n, _| {
                    let v = tl.frame(n)?;
                    Some((v.k, v.src, export::view(&v.view)))
                }))
            }
            false => None,
        };
        let job = export::Job {
            cut: export::Cut::Edit(Box::new(export::Edit {
                tl: e.tl.clone(),
                files: e.files.clone(),
                looks: e.looks.clone(),
                adjust: e.adjust,
                over: e.over.clone(),
                abr: 192_000,
            })),
            path: t.it.path.clone(),
            out: match x.pano {
                true => export::Out::Pano(x.w),
                false => export::Out::Flat {
                    w: x.w,
                    h: x.h,
                    view: None,
                },
            },
            enc: export::Enc::Video {
                codec: x.codec,
                deep: x.deep && x.codec == Codec::H265,
                rate: x
                    .qp
                    .map_or(Rate::Cbr((x.mbps as f64 * 1e6) as u64), Rate::Qp),
                gop: ((fps.num + fps.den / 2) / fps.den).max(1),
            },
            raft: x.raft,
            colour: x.colour.then_some(0.1),
            stab,
            fx,
            flat: false,
        };
        let (it, mut last) = (&t.it, Instant::now());
        export::run(g, cx, &mut self.raft, job, &it.stop, &mut |p| {
            it.set(St::Run(p));
            if last.elapsed().as_millis() >= 100 {
                last = Instant::now();
                ctx.request_repaint();
            }
        })
    }

    /// The faces of source span `[a, b]` (s) of clip `f` every `STEP` frames, found on the clip stitched with the
    /// poses the export stitches with, linked into regions for every frame; the frame times. Adds its counts to `sum`
    /// (detections, frames searched, tracks, frames blurred, frames) and notes the totals.
    #[allow(clippy::too_many_arguments)]
    fn faces(
        &mut self,
        g: &Arc<Gpu>,
        cx: &Ctx,
        t: &Task,
        f: &[PathBuf],
        stab: Option<(stab::Lock, f64)>,
        [a, b]: [f64; 2],
        fps: edit::Rate,
        sum: &mut [usize; 5],
        ctx: &egui::Context,
    ) -> Result<(Vec<f64>, Regs), String> {
        if f.len() > 1 {
            return Err("face blur: recordings in a file per lens are not supported yet".into());
        }
        let c = shotlab::Clip::open(&f[0])?;
        let poses = match stab {
            Some((l, s)) => Some(
                c.poses(l, s)
                    .ok_or("face blur: no IMU record to stabilise with")?,
            ),
            None => None,
        };
        let k0 = clip::frame_at(&c.times, a);
        let k1 = c.times.partition_point(|&x| x < b - 1e-3).max(k0 + 1) - 1;
        let ks: Vec<usize> = (k0..=k1).step_by(STEP).collect();
        let (it, n, tiles) = (&t.it, ks.len(), track::face_tiles());
        let (mut hide, mut pano, mut i, mut found) = (track::Hide::new(), None, 0, 0);
        it.set(St::Faces(0, n));
        crate::ai::MODELS.with(g, false, Some(ctx), |ai| {
            // shotlab decodes and submits each stitch on this thread between the calls: a surface configure the UI
            // retries waits for that (no wait there depends on another thread); the detection runs under BURN instead,
            // never inside GATE (the UI takes BURN, then GATE)
            let mut gate = Some(crate::GATE.read());
            c.frames(
                g,
                cx,
                FACES_W,
                false,
                &ks,
                |k| c.pose(k, poses.as_deref()),
                |tex, k| {
                    if it.stop.load(Relaxed) {
                        return Err(CANCELLED.into());
                    }
                    let p =
                        pano.get_or_insert_with(|| ai.pano(&tex.create_view(&Default::default())));
                    gate.take();
                    // burn submits throughout
                    let r = {
                        let _b = crate::BURN.lock();
                        ai.faces(p, &tiles, MIN)
                    };
                    gate = Some(crate::GATE.read());
                    let r = r?;
                    found += r.len();
                    hide.push(k, r);
                    i += 1;
                    it.set(St::Faces(i, n));
                    ctx.request_repaint();
                    Ok(())
                },
            )
        })?;
        let r = hide.regions(c.times.len(), fps.num as f32 / fps.den as f32);
        let on = r[k0..=k1].iter().filter(|r| !r.is_empty()).count();
        let tr = hide.tracks(fps.num as f32 / fps.den as f32).len();
        for (s, x) in sum.iter_mut().zip([found, n, tr, on, k1 + 1 - k0]) {
            *s += x;
        }
        let [found, n, tr, on, all] = *sum;
        *it.note.lock().unwrap_or_else(|e| e.into_inner()) = format!(
            "{found} face detections on {n} frames, {tr} tracks, blurred on {on} of {all} frames"
        );
        Ok((c.times, r))
    }

    /// A photo: its 360 or its view through the preview, the look and adjustments, the faces blurred; a JPEG (GPano
    /// for 360).
    fn photo(&mut self, g: &Arc<Gpu>, cx: &Ctx, t: &Task) -> Result<export::Progress, String> {
        let (s, x) = (&t.spec, &t.set);
        let (px, w0, h0) = photo::load(&s.files[0])?;
        if let Some(e) = photo::flat(&s.files[0], w0, h0) {
            return Err(e);
        }
        let src = photo::upload(g, &px, w0, h0)?;
        drop(px);
        let pano = cx.pano(&src.create_view(&Default::default()));
        let [w, h] = x.size();
        let f = wgpu::TextureFormat::Rgba8Unorm;
        use wgpu::TextureUsages as U;
        let tex = |w, h, usage| {
            g.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("photo export"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: f,
                usage,
                view_formats: &[],
            })
        };
        let pv = Preview::pano(cx, f);
        let eq = View {
            mode: Mode::Equirect,
            ..View::default()
        };
        // faces on an equirect of the photo at the face tiles' resolution, in the photo's frame
        let regs = match x.faces {
            true => {
                let d = tex(
                    FACES_W,
                    FACES_W / 2,
                    U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
                );
                let dv = d.create_view(&Default::default());
                let mut e = g.device.create_command_encoder(&Default::default());
                pv.render(&mut e, &dv, [FACES_W, FACES_W / 2], &eq, &pano);
                {
                    let _g = crate::GATE.read();
                    g.queue.submit([e.finish()]);
                }
                crate::ai::MODELS.with(g, false, None, |ai| {
                    let _b = crate::BURN.lock();
                    ai.faces(&ai.pano(&dv), &track::face_tiles(), MIN)
                })?
            }
            false => vec![],
        };
        let out = tex(
            w,
            h,
            U::RENDER_ATTACHMENT
                | U::TEXTURE_BINDING
                | U::STORAGE_BINDING
                | U::COPY_SRC
                | U::COPY_DST,
        );
        let ov = out.create_view(&Default::default());
        let v = if x.pano { eq } else { s.view };
        let mut e = g.device.create_command_encoder(&Default::default());
        pv.render(&mut e, &ov, [w, h], &v, &pano);
        if x.look.is_some() || x.adj != fx::Params::default() {
            let c = self
                .fx
                .get_or_insert_with(|| fx::Ctx::new(&g.device, &g.queue));
            let mut fx = fx::Fx::new(c, w, h, x.pano);
            fx.set(&fx::Params {
                mix: x.amt,
                ..x.adj
            });
            fx.look(x.look.map(|l| l.cube()).as_ref());
            let io = fx.io(&out, &out);
            fx.run(&mut e, &io, &[]);
        }
        if !regs.is_empty() {
            let tmp = tex(w, h, U::TEXTURE_BINDING | U::COPY_DST);
            e.copy_texture_to_texture(out.as_image_copy(), tmp.as_image_copy(), out.size());
            let tv = tmp.create_view(&Default::default());
            match x.pano {
                true => {
                    track::Blur::new(&g.device, f).run(&g.device, &g.queue, &mut e, &tv, &ov, &regs)
                }
                false => {
                    Blur::new(&g.device, f).run(&g.device, &g.queue, &mut e, &tv, &ov, &v, &regs)
                }
            }
        }
        let px = {
            let _g = crate::GATE.read();
            g.queue.submit([e.finish()]);
            stitch::read_texture(&g.device, &g.queue, &out)
        };
        *t.it.note.lock().unwrap_or_else(|e| e.into_inner()) = match x.faces {
            true => format!("{} faces blurred", regs.len()),
            false => String::new(),
        };
        let b = export::jpeg::encode(
            &px,
            w,
            h,
            x.jq,
            x.pano.then(|| export::jpeg::gpano(w, h)).as_deref(),
        )?;
        let mut part = t.it.path.clone().into_os_string();
        part.push(".part");
        let io = |e: std::io::Error| format!("{}: {e}", t.it.path.display());
        std::fs::write(&part, b).map_err(io)?;
        std::fs::rename(&part, &t.it.path).map_err(io)?;
        Ok(export::Progress {
            done: 1,
            total: 1,
            ..Default::default()
        })
    }
}

/// The export's effect hook that blurs the face regions of each output frame: `pick(k, t)` gives output frame k at
/// time t (s) its face pass (an index into `sets`: frame times, regions per frame), source time and view (that of a
/// `flat` output; unused on a 360 one); the regions of the pass's frame with the latest time ≤ that source time blur.
fn hook(
    g: &Gpu,
    flat: bool,
    sets: Vec<(Vec<f64>, Regs)>,
    mut pick: impl FnMut(u64, f64) -> Option<(usize, f64, View)> + Send + 'static,
) -> export::Hook {
    let (dev, q) = (g.device.clone(), g.queue.clone());
    enum Pass {
        Pano(track::Blur),
        Flat(Blur),
    }
    // per frame texture (a new one per decode pass): it, its copy, views of both, the pass made for its format
    let mut st: Option<(
        wgpu::Texture,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::TextureView,
        Pass,
    )> = None;
    Box::new(move |x: export::Fx| {
        let Some((i, s, v)) = pick(x.k, x.t) else {
            return;
        };
        let (times, regs) = &sets[i];
        let k = times.partition_point(|&t| t <= s + 1e-6).saturating_sub(1);
        let r = &regs[k.min(regs.len() - 1)];
        if r.is_empty() {
            return;
        }
        if st.as_ref().is_none_or(|s| &s.0 != x.tex) {
            let t = dev.create_texture(&wgpu::TextureDescriptor {
                label: Some("blur input"),
                size: x.tex.size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: x.tex.format(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let f = x.tex.format();
            let p = match flat {
                true => Pass::Flat(Blur::new(&dev, f)),
                false => Pass::Pano(track::Blur::new(&dev, f)),
            };
            let (tv, dv) = (
                t.create_view(&Default::default()),
                x.tex.create_view(&Default::default()),
            );
            st = Some((x.tex.clone(), t, tv, dv, p));
        }
        let (_, t, tv, dv, p) = st.as_mut().unwrap();
        x.enc
            .copy_texture_to_texture(x.tex.as_image_copy(), t.as_image_copy(), x.tex.size());
        match p {
            Pass::Pano(b) => b.run(&dev, &q, x.enc, tv, dv, r),
            Pass::Flat(b) => b.run(&dev, &q, x.enc, tv, dv, &v, r),
        }
    })
}

/// The preview's projection of `v` (crate stitch, `Preview::set`): view camera → world, mode (0 perspective / Panini,
/// 1 equirect, 2 planet), plane half-width, Panini or planet distance.
fn proj(v: &View) -> (Mat3, u32, f32, f32) {
    let c = track::Cam::view(v.yaw, v.pitch, v.roll, 90.0, 1.0);
    let m = Mat3::from_cols(c.r, c.u, c.f);
    let r = f32::to_radians;
    match v.mode {
        Mode::Perspective => {
            let d = v.panini.clamp(0.0, 1.0);
            let h = r(v.fov / 2.0).min(r(89.5) + d * r(90.0));
            (m, 0, (d + 1.0) / (d + h.cos()) * h.sin(), d)
        }
        Mode::Equirect => (m, 1, 1.0, 0.0),
        Mode::Planet => {
            let d = v.planet.clamp(0.0, 1.0) as f64;
            let h = (v.fov as f64 / 2.0)
                .to_radians()
                .min((-d).acos() - 0.5f64.to_radians())
                / 2.0;
            let f = (d + 1.0) * 2.0 * h.sin() * h.cos() / (d - 1.0 + 2.0 * h.cos() * h.cos());
            (m, 2, f as f32, d as f32)
        }
    }
}

/// Target pixel of camera-frame direction `c` (x right, y up, z forward) on a `s` target under `proj`'s mode, plane
/// half-width and distance; None behind the projection. `blur.wgsl` `uv` times the size.
fn px(c: Vec3, mode: u32, f: f32, d: f32, s: Vec2) -> Option<Vec2> {
    let x = match mode {
        1 => {
            let u = Vec2::new(
                c.x.atan2(c.z) / std::f32::consts::TAU + 0.5,
                0.5 - c.y.clamp(-1.0, 1.0).asin() / std::f32::consts::PI,
            );
            return Some(u * s);
        }
        2 => {
            let t = c.z.clamp(-1.0, 1.0).acos();
            if d + t.cos() < 1e-3 {
                return None;
            }
            (d + 1.0) * t.sin() / (d + t.cos()) * Vec2::new(c.x, c.y).normalize_or_zero()
        }
        _ => {
            let (lon, l) = (c.x.atan2(c.z), Vec2::new(c.x, c.z).length());
            if d + lon.cos() < 1e-3 || l < 1e-6 {
                return None;
            }
            let k = (d + 1.0) / (d + lon.cos());
            Vec2::new(k * lon.sin(), k * c.y / l)
        }
    };
    Some(Vec2::new(x.x / f + 1.0, 1.0 - x.y * s.x / (f * s.y)) * 0.5 * s)
}

/// `blur.wgsl` `Reg`: forward, right, up, ellipse half extents and blur radius, quad.
type Reg = [[f32; 4]; 5];

/// Privacy blur in a flat view: crate track's `Blur` (an equirect) through the preview's projection, so the faces
/// found on the sphere blur where the view shows them. A render pass from `src` into `dst` (the same size, the format
/// given); regions in the world frame of the view (the output frame the faces were found in). One `run` per submit.
pub struct Blur {
    pipe: [wgpu::RenderPipeline; 2],
    bgl: wgpu::BindGroupLayout,
    smp: wgpu::Sampler,
    ub: wgpu::Buffer,
    buf: wgpu::Buffer,
    regs: Vec<Reg>,
}

impl Blur {
    pub fn new(dev: &wgpu::Device, format: wgpu::TextureFormat) -> Blur {
        let e = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty,
            count: None,
        };
        let bgl = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                e(
                    0,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                e(
                    1,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                e(
                    2,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                e(
                    3,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
            ],
        });
        let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let m = dev.create_shader_module(wgpu::include_wgsl!("export/blur.wgsl"));
        let pipe = |vs, fs, topology, blend| {
            dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("flat blur"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &m,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &m,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        use wgpu::{BlendComponent as C, BlendFactor as F, BlendOperation as O};
        // premultiplied over, the destination's alpha kept
        let over = wgpu::BlendState {
            color: C {
                src_factor: F::One,
                dst_factor: F::OneMinusSrcAlpha,
                operation: O::Add,
            },
            alpha: C {
                src_factor: F::Zero,
                dst_factor: F::One,
                operation: O::Add,
            },
        };
        let buf = |size, usage| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flat blur"),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Blur {
            pipe: [
                pipe("full", "copy", wgpu::PrimitiveTopology::TriangleList, None),
                pipe(
                    "quad",
                    "blur",
                    wgpu::PrimitiveTopology::TriangleStrip,
                    Some(over),
                ),
            ],
            bgl,
            smp: dev.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            ub: buf(80, wgpu::BufferUsages::UNIFORM),
            buf: buf(64 * size_of::<Reg>() as u64, wgpu::BufferUsages::STORAGE),
            regs: vec![],
        }
    }

    /// Records `dst` = `src` with regions `r` blurred where view `v` shows them.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        dev: &wgpu::Device,
        q: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        v: &View,
        r: &[track::Region],
    ) {
        let t = src.texture().size();
        let s = Vec2::new(t.width as f32, t.height as f32);
        let (m, mode, f, d) = proj(v);
        let mut u = [0f32; 20];
        for (c, col) in m.to_cols_array().chunks(3).enumerate() {
            u[4 * c..4 * c + 3].copy_from_slice(col);
        }
        u[12..14].copy_from_slice(&s.to_array());
        u[14] = f32::from_bits(mode);
        u[15] = f;
        u[16] = d;
        q.write_buffer(&self.ub, 0, bytemuck::cast_slice(&u));
        self.regs.clear();
        for x in r {
            let ext = Vec2::new(
                (x.size.x / 2.0).to_radians().tan(),
                (x.size.y / 2.0).to_radians().tan(),
            );
            let c = track::Cam::new(x.dir, Vec3::Z, ext);
            // the quad: the feathered ellipse's outline (1.25 of it) in the view, or the whole frame where some of
            // it does not map (an equirect view, a region reaching behind the projection)
            let mut b = [s.x, s.y, 0.0, 0.0];
            let pts = (0..16).map(|i| {
                let a = Vec2::from_angle(i as f32 * std::f32::consts::TAU / 16.0) * 1.25 * ext;
                px(
                    m.transpose() * (c.f + a.x * c.r + a.y * c.u).normalize(),
                    mode,
                    f,
                    d,
                    s,
                )
            });
            for p in pts.chain([px(m.transpose() * c.f, mode, f, d, s)]) {
                match p.filter(|_| mode != 1) {
                    Some(p) => b = [b[0].min(p.x), b[1].min(p.y), b[2].max(p.x), b[3].max(p.y)],
                    None => {
                        b = [0.0, 0.0, s.x, s.y];
                        break;
                    }
                }
            }
            let b = [
                (b[0] - 2.0).max(0.0),
                (b[1] - 2.0).max(0.0),
                (b[2] + 2.0).min(s.x),
                (b[3] + 2.0).min(s.y),
            ];
            if b[0] >= b[2] || b[1] >= b[3] {
                continue;
            }
            let w = |v: Vec3| [v.x, v.y, v.z, 0.0];
            self.regs.push([
                w(c.f),
                w(c.r),
                w(c.u),
                [ext.x, ext.y, ext.min_element() / 3.0, 0.0],
                b,
            ]);
        }
        if (self.regs.len() * size_of::<Reg>()) as u64 > self.buf.size() {
            self.buf = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flat blur"),
                size: (self.regs.len().next_power_of_two() * size_of::<Reg>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        q.write_buffer(&self.buf, 0, bytemuck::cast_slice(&self.regs));
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.smp),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ub.as_entire_binding(),
                },
            ],
        });
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("flat blur"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        rp.set_bind_group(0, &bg, &[]);
        rp.set_pipeline(&self.pipe[0]);
        rp.draw(0..3, 0..1);
        if !self.regs.is_empty() {
            rp.set_pipeline(&self.pipe[1]);
            rp.draw(0..4, 0..self.regs.len() as u32);
        }
    }
}

#[cfg(test)]
mod tests;
