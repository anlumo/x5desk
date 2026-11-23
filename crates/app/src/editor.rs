//! Editor screen (PLAN.md milestone 7, groups 10, 12, 13) on crate edit's model: a timeline of clips from the
//! library (drag in, trim, split, delete, reorder, speed with ramp keys, freeze frames), reframing keys from the
//! viewer's view with easing and camera-movement presets (edit) or Effects cuts (shotlab), a look (ours or an
//! imported `.cube`) and colour adjustments per clip, overlays (fx), clip volume envelopes, mute, music with ducking
//! and denoise (media), undo / redo (edit's log for the timeline, steps of the extras between its steps), project
//! files and "Export timeline" (`export::Cut::Edit` on the Export screen's queue). The viewer frames what a flat export
//! shows and plays the timeline through decks (`deck`) on the preview sound's clock (`sound`); paused, it shows the
//! frame's RAFT stitch from the app's worker (`player::still`). Thumbnails come from a worker (`thumbs`).
//!
//! AI (`ai`): subject tracking from a box drawn in the viewer follows the subject from the playhead to the clip's end
//! and becomes the clip's reframing keys (its states on the clip in the timeline); auto framing's path keys the whole
//! clip the same way; auto edit puts a clip's highlights, framed automatically, on the timeline.

mod deck;
pub mod doc;
mod sound;
#[cfg(test)]
mod tests;
mod thumbs;

use crate::{
    ai, clip,
    export::Tl,
    gyro,
    player::{self, Look},
};
use deck::{DECKS, Deck, Info, Viewer};
use doc::{Extra, Over, Pic};
use edit::{Clip, Ease, Key, On, Proj, Project, Rate};
use eframe::{egui, egui_wgpu::RenderState};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant},
};
use stitch::{Mode, Pose, View};
use video::Gpu;

/// A new timeline's rate until its first clip gives its own.
const FPS: Rate = Rate {
    num: 30000,
    den: 1001,
};
const EASES: [(Ease, &str); 6] = [
    (Ease::Smooth, "Smooth"),
    (Ease::Linear, "Linear"),
    (Ease::In, "Ease in"),
    (Ease::Out, "Ease out"),
    (Ease::InOut, "Ease in-out"),
    (Ease::Hold, "Hold"),
];
/// Music files the import takes (crate media's `music`).
const MUSIC: [&str; 7] = ["mp3", "flac", "wav", "ogg", "m4a", "aac", "mp4"];
/// Timeline rows: ruler, view keys, video, audio, music (heights, px).
const ROWS: [f32; 5] = [18.0, 16.0, 64.0, 28.0, 18.0];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tab {
    Clip,
    View,
    Colour,
    Audio,
    Over,
}

const TABS: [(Tab, &str); 5] = [
    (Tab::Clip, "Clip"),
    (Tab::View, "Reframe"),
    (Tab::Colour, "Colour"),
    (Tab::Audio, "Audio"),
    (Tab::Over, "Overlays"),
];

/// Test hook (`X5DESK_TEST`): what the editor does at start.
#[derive(Clone, Debug, Default)]
pub struct Hook {
    /// start on the editor
    pub on: bool,
    pub project: Option<PathBuf>,
    /// recordings appended in order
    pub add: Vec<PathBuf>,
    /// the playhead (s), the selected clip, the inspector tab; play (once the editor is idle)
    pub at: Option<f64>,
    pub sel: Option<usize>,
    pub tab: Option<Tab>,
    /// undo / redo steps after loading
    pub undo: usize,
    pub redo: usize,
    /// save the project here once set up
    pub save: Option<PathBuf>,
    /// export the timeline with these settings (`export::parse` tokens)
    pub export: Option<String>,
    /// the export preset the viewer frames for (a key of `export::PRESETS`)
    pub preset: Option<String>,
    pub play: bool,
    /// preview volume (`vol=`)
    pub vol: Option<f32>,
    /// preview the 360 frame
    pub pano: bool,
    /// tracking: the box drawn once the editor is set up, 0..1 of the export preset's flat frame (16:9 for a 360
    /// preset) whatever the window
    pub tbox: Option<[f32; 4]>,
}

/// A tracking or auto framing run for clip `k` (of file `path`, source `a..b`), keyed as view `v` turned onto the
/// subject.
struct Trk {
    t: Arc<ai::Track>,
    k: usize,
    path: String,
    a: f64,
    b: f64,
    v: edit::View,
    /// its keys are on the clip
    keyed: bool,
}

impl Tab {
    pub fn parse(s: &str) -> Option<Tab> {
        TABS.iter()
            .find(|t| t.1.eq_ignore_ascii_case(s) || format!("{:?}", t.0).eq_ignore_ascii_case(s))
            .map(|t| t.0)
    }
}

type PoseKey = deck::Key;
type Poses = (String, PoseKey, Result<Vec<stab::Pose>, String>);
/// a music file and its length as counted
type Count = (PathBuf, f64);

enum Drag {
    /// a clip's body
    Move(usize),
    /// a trim handle: clip, right edge, the clip at the start, pointer x at the start
    Trim(usize, bool, Clip, f32),
    /// a view key: track, its index now, its index at the start (its widget's id through the drag)
    Key(On, usize, usize),
    /// a speed key: clip, index, the clip at the start, pointer x at the start
    Speed(usize, usize, Clip, f32),
    /// a volume point: clip, index
    Vol(usize, usize),
    /// an inspector slider (or the Colour tab's adjustments, as one): its id; it edits the selected clip
    Slider(egui::Id),
}

/// What the timeline's widgets ask for, applied after drawing.
enum Act {
    Seek(f64),
    /// a drag of the ruler starts (pausing playback) or ends (resuming it)
    Scrub(bool),
    Sel(usize),
    Mv(usize, usize),
    Trim(usize, f64, f64),
    Hold(usize, f64),
    Begin,
    End,
    MvKey(On, usize, f64),
    KeySel(On, usize),
    Add(PathBuf, usize),
    Speed(usize, Vec<(f64, f64)>),
    Vol(usize, Vec<(f64, f32)>),
}

/// An undo step: one of edit's (its log holds it), or the extras before (undone: after) a change.
enum Step {
    Ed,
    Ex(Extra),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dlg {
    Add,
    Open,
    Save,
    Music,
    /// auto edit's music
    Beat,
    Png,
    Nadir,
    Cube,
}

pub struct Editor {
    g: Arc<Gpu>,
    cx: stitch::Ctx,
    ctx: egui::Context,
    pub pj: Project,
    pub x: Extra,
    pub path: Option<PathBuf>,
    /// the project as last saved or loaded
    saved: String,
    /// undo history: steps done, steps undone; edit's undo and redo depths as last seen, the extras as of its last
    /// step. Edit's log moves only through `flip` (both depths follow) or a change, which grows its undo and drops its
    /// redo: a drag that ends where it started leaves both as they were.
    hist: Vec<Step>,
    fut: Vec<Step>,
    eh: (usize, usize),
    xs: Extra,
    info: HashMap<String, Result<Info, String>>,
    decks: Vec<Deck>,
    vw: Viewer,
    thumbs: thumbs::Thumbs,
    snd: Option<sound::Sound>,
    /// RAFT stitch of the paused frame: the app's worker, its replies, lens picture copies, the stitch and its frame,
    /// a job out, RAFT failed, since when the frame shows
    rw: mpsc::Sender<player::Job>,
    rr: (mpsc::Sender<player::Reply>, mpsc::Receiver<player::Reply>),
    snap: Option<player::Snap>,
    seam: Option<(player::Key, stitch::Pano)>,
    busy: bool,
    raft_off: bool,
    since: Instant,
    pq: (mpsc::Sender<Poses>, mpsc::Receiver<Poses>),
    /// a music file whose length is being counted (its container does not say), and the count
    mq: (Option<PathBuf>, mpsc::Sender<Count>, mpsc::Receiver<Count>),
    /// files whose poses are being made, and for which setting: one thread per file, the latest setting next
    asked: HashMap<String, PoseKey>,
    /// playhead, output s
    pub t: f64,
    pub playing: bool,
    pub sel: Option<usize>,
    /// keys go on the output timeline's track (else the clip under the playhead)
    pub whole: bool,
    key: Option<(On, usize)>,
    /// the viewer's view as the user turned it, until a key or playback takes over
    free: Option<View>,
    pano: bool,
    zoom: f32,
    /// the playhead the timeline last showed and its view's width; a ruler drag paused playback
    seen: (f64, f32),
    resume: bool,
    pub vol: f32,
    ease: Ease,
    pub tab: Tab,
    /// edit generation (deck assignments), a new frame or view to draw, the viewer's size
    ver: u64,
    fresh: bool,
    dirty: bool,
    size: [u32; 2],
    drag: Option<Drag>,
    /// an undo group is open
    grp: bool,
    /// freeze length, preset length, effect parameters
    hold: f64,
    pdur: f64,
    turns: f32,
    lapse: f64,
    every: f64,
    /// export: preset key, RAFT seam
    pub preset: usize,
    raft: bool,
    pub err: Option<String>,
    dlg: Option<(Dlg, mpsc::Receiver<Option<PathBuf>>)>,
    /// a timeline export for the Export screen, with its `export::parse` tokens
    pub ask: Option<(Tl, String)>,
    hook: Option<Hook>,
    /// test hook: playback statistics (frames shown, start: wall time and playhead); play once idle
    log: Option<(u64, Option<(std::time::Instant, f64)>)>,
    autoplay: bool,
    /// tracking: the run, armed (a box drawn in the viewer starts one), the box being drawn
    trk: Option<Trk>,
    arm: bool,
    from: Option<egui::Pos2>,
    tbox: Option<[f32; 4]>,
    /// an auto edit being made; export it with these `export::parse` tokens once it is on the timeline (test hook)
    auto: Option<Arc<ai::Auto>>,
    pub autox: Option<String>,
    /// the music auto edits lay under their highlights, the cuts on its beats
    beat: Option<PathBuf>,
}

fn name(p: &str) -> &str {
    Path::new(p)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(p)
}

fn mmss(t: f64) -> String {
    format!("{}:{:05.2}", (t / 60.0) as u32, t % 60.0)
}

/// The editor view of a viewer view.
pub fn to_edit(v: &View) -> edit::View {
    let p = match v.mode {
        Mode::Perspective if v.panini > 0.0 => Proj::Panini(v.panini.min(1.0)),
        Mode::Perspective => Proj::Persp,
        Mode::Planet => Proj::Planet(v.planet.clamp(0.0, 1.0)),
        Mode::Equirect => Proj::Equirect,
    };
    edit::View::look(v.yaw, v.pitch, v.roll, v.fov.clamp(1.0, 360.0), p)
}

fn look_of(v: &View) -> Look {
    match v.mode {
        Mode::Equirect => Look::Equirect,
        Mode::Planet => Look::Planet,
        _ if v.panini > 0.0 => Look::Panini,
        _ => Look::Perspective,
    }
}

/// The source time clip `k` starts on (a freeze frame's picture).
fn src0(tl: &edit::Timeline, k: usize) -> f64 {
    let c = &tl.clips[k];
    if c.hold > 0.0 {
        c.a
    } else {
        tl.content(k, 0.0)
    }
}

impl Editor {
    /// `still`: the app's RAFT worker (`player::still`).
    pub fn new(
        g: &Arc<Gpu>,
        cx: &stitch::Ctx,
        ctx: &egui::Context,
        still: mpsc::Sender<player::Job>,
        hook: Option<Hook>,
    ) -> Editor {
        let pj = Project::new(FPS).expect("a valid frame rate");
        let x = Extra::default();
        Editor {
            saved: doc::save(&pj, &x),
            hist: vec![],
            fut: vec![],
            eh: (0, 0),
            xs: x.clone(),
            g: g.clone(),
            cx: cx.clone(),
            ctx: ctx.clone(),
            pj,
            x,
            path: None,
            info: HashMap::new(),
            decks: vec![],
            vw: Viewer::new(g),
            thumbs: thumbs::Thumbs::new(g, cx, ctx),
            snd: None,
            rw: still,
            rr: mpsc::channel(),
            snap: None,
            seam: None,
            busy: false,
            raft_off: false,
            since: Instant::now(),
            pq: mpsc::channel(),
            mq: {
                let (tx, rx) = mpsc::channel();
                (None, tx, rx)
            },
            asked: HashMap::new(),
            t: 0.0,
            playing: false,
            sel: None,
            whole: false,
            key: None,
            free: None,
            pano: false,
            zoom: 60.0,
            seen: (0.0, 0.0),
            resume: false,
            vol: hook
                .as_ref()
                .map_or(1.0, |h| h.vol.unwrap_or(0.1).clamp(0.0, 1.0)),
            ease: Ease::Smooth,
            tab: Tab::Clip,
            ver: 0,
            fresh: false,
            dirty: true,
            size: [0; 2],
            drag: None,
            grp: false,
            hold: 1.0,
            pdur: 2.0,
            turns: 1.0,
            lapse: 4.0,
            every: 0.5,
            preset: 4,
            raft: true,
            err: None,
            dlg: None,
            ask: None,
            hook,
            log: None,
            autoplay: false,
            trk: None,
            arm: false,
            from: None,
            tbox: None,
            auto: None,
            autox: None,
            beat: None,
        }
    }

    /// Starts on the editor (test hook).
    pub fn first(&self) -> bool {
        self.hook.as_ref().is_some_and(|h| h.on)
    }

    fn changed(&mut self) {
        self.ver += 1;
        self.dirty = true;
        for d in &mut self.decks {
            d.k = None;
        }
        // a change that is not the drag's own ends it, also one that skipped `ed`'s halt (a moved clip ends unplaced)
        self.halt();
        if self.playing {
            self.pause();
        }
        // a shorter timeline: the playhead stays on it
        if self.t > self.pj.tl().dur() {
            self.seek(self.t);
        }
        let n = self.pj.tl().clips.len();
        if self.sel.is_some_and(|k| k >= n) {
            self.sel = n.checked_sub(1);
        }
        let paths: Vec<String> = self.clip_files();
        for p in paths {
            self.thumbs.ask(&p);
            self.info(&p);
        }
    }

    /// Paths of the files clips use.
    fn clip_files(&self) -> Vec<String> {
        let tl = self.pj.tl();
        let mut v: Vec<String> = vec![];
        for c in &tl.clips {
            let p = &tl.files[c.file as usize].path;
            if !v.contains(p) {
                v.push(p.clone());
            }
        }
        v
    }

    /// An edit's result: on success the timeline changed.
    fn did<T>(&mut self, r: edit::Result<T>) -> Option<T> {
        match r {
            Ok(v) => {
                self.err = None;
                self.changed();
                Some(v)
            }
            Err(e) => {
                self.err = Some(e.0);
                None
            }
        }
    }

    /// An edit of the project; on success the timeline changed. An open drag ends first (`halt`).
    fn ed<T>(&mut self, f: impl FnOnce(&mut Project) -> edit::Result<T>) -> Option<T> {
        self.halt();
        let r = f(&mut self.pj);
        self.did(r)
    }

    /// Ends the open drag and its undo step, before an edit that is not its own (Delete or S mid-drag, a tracking run's
    /// keys): it names its clip, key or point by index, which may be another's afterwards (its widget's id too).
    fn halt(&mut self) {
        if self.drag.take().is_some() {
            self.ctx.stop_dragging();
            if self.grp {
                self.end();
            }
        }
    }

    /// `f` as the open drag's own work (its undo group, its edits): the drag goes on through it.
    fn own<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let d = self.drag.take();
        let r = f(self);
        self.drag = d;
        r
    }

    fn begin(&mut self) {
        self.halt();
        self.pj.begin();
        self.grp = true;
    }

    fn end(&mut self) {
        self.pj.end();
        self.grp = false;
    }

    /// Appends (or inserts before clip `at`) the whole recording of `p`; returns the clip's index.
    pub fn add(&mut self, p: &Path, at: Option<usize>) -> Result<usize, String> {
        let p = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
        let w = crate::library::pair(&p);
        let main = w
            .main
            .first()
            .ok_or_else(|| format!("{}: no full-resolution file", p.display()))?;
        let (rate, dur) = crate::export::probe(main)?;
        let s = main.to_string_lossy().to_string();
        let f = self.pj.file(&s, dur).map_err(|e| e.0)?;
        let (c, tl) = (Clip::new(f, 0.0, dur), self.pj.tl());
        tl.ok(&c).map_err(|e| e.0)?;
        let k = at.unwrap_or(tl.clips.len()).min(tl.clips.len());
        // a first clip brings its frame rate: one step with the insert
        let fps = tl.is_empty() && tl.out.fps != rate;
        self.begin();
        let r = match fps {
            true => self.pj.edit_out(|o| o.fps = rate),
            false => Ok(()),
        }
        .and_then(|_| self.pj.insert(k, c));
        self.end();
        r.map_err(|e| e.0)?;
        self.sel = Some(k);
        self.changed();
        Ok(k)
    }

    pub fn split(&mut self) {
        if let Some(k) = {
            let t = self.t;
            self.ed(|p| p.split(t))
        } {
            self.sel = Some(k);
        }
    }

    pub fn freeze(&mut self) {
        let h = self.hold;
        if let Some(k) = {
            let t = self.t;
            self.ed(|p| p.freeze(t, h))
        } {
            self.sel = Some(k);
        }
    }

    pub fn delete(&mut self, k: usize) {
        self.ed(|p| p.delete(k));
    }

    pub fn mv(&mut self, from: usize, to: usize) {
        if self.ed(|p| p.mv(from, to)).is_some() {
            self.sel = Some(to);
        }
    }

    /// Brings the history up to date: edit's new steps, then the extras once they settle (no button held, no text
    /// being typed: a slider's drag or a text's edit is one step). A change drops the steps undone, as edit does.
    fn sync(&mut self) {
        if self.grp {
            return;
        }
        let (u, r) = self.pj.history();
        if u > self.eh.0 {
            self.fut.clear();
        }
        self.hist.extend((self.eh.0..u).map(|_| Step::Ed));
        self.eh = (u, r);
        if self.x != self.xs
            && !self.ctx.input(|i| i.pointer.any_down())
            && !self.ctx.text_edit_focused()
        {
            self.hist
                .push(Step::Ex(std::mem::replace(&mut self.xs, self.x.clone())));
            self.fut.clear();
        }
    }

    /// Undoes (`back`) or redoes step `s`; returns the step that goes the other way.
    fn flip(&mut self, s: Step, back: bool) -> Step {
        let s = match s {
            Step::Ed => {
                let ok = match back {
                    true => self.pj.undo(),
                    false => self.pj.redo(),
                };
                debug_assert!(ok, "the history is out of step with edit's log");
                self.eh = self.pj.history();
                Step::Ed
            }
            Step::Ex(x) => {
                self.xs = x.clone();
                Step::Ex(std::mem::replace(&mut self.x, x))
            }
        };
        self.free = None;
        self.changed();
        s
    }

    /// Undoes the last step; nothing mid-drag (edit would close the drag's step).
    pub fn undo(&mut self) {
        self.sync();
        if self.grp {
            return;
        }
        if let Some(s) = self.hist.pop() {
            let s = self.flip(s, true);
            self.fut.push(s);
        }
    }

    pub fn redo(&mut self) {
        self.sync();
        if self.grp {
            return;
        }
        if let Some(s) = self.fut.pop() {
            let s = self.flip(s, false);
            self.hist.push(s);
        }
    }

    /// Undo and redo depths.
    pub fn history(&self) -> (usize, usize) {
        (self.hist.len(), self.fut.len())
    }

    /// The track keys go on: the output timeline's or the clip's under the playhead.
    fn on(&self) -> Option<On> {
        let tl = self.pj.tl();
        match self.whole {
            true => Some(On::Out),
            false => tl.frame_at(self.t).and_then(|n| tl.find(n)).map(On::Clip),
        }
    }

    /// The view the viewer shows: the user's, or the keys' at the playhead.
    pub fn shown(&self) -> View {
        if let Some(v) = self.free {
            return v;
        }
        let tl = self.pj.tl();
        match tl.at(self.t.min(tl.dur() - 1e-6).max(0.0)) {
            Some(e) => export::view(&e.view),
            None => Look::Perspective.view(),
        }
    }

    /// Whether keys move the view at the playhead.
    fn keyed(&self) -> bool {
        let tl = self.pj.tl();
        !tl.out.view.is_empty()
            || tl
                .at(self.t.max(0.0))
                .is_some_and(|e| !e.clip.view.is_empty())
    }

    /// A key at the playhead from the viewer's view.
    pub fn key_here(&mut self) {
        let Some(on) = self.on() else {
            self.err = Some("The playhead is outside the timeline.".into());
            return;
        };
        let v = to_edit(&self.shown());
        if let Some(i) = {
            let (t, e) = (self.t, self.ease);
            self.ed(|p| p.key(on, t, v, e))
        } {
            (self.key, self.free) = (Some((on, i)), None);
        }
    }

    /// Tracking of subject `s` (`ai::subject` of a box in the viewer's perspective view) from the playhead to the end
    /// of its clip: the result becomes the clip's reframing keys, the viewer's view turned onto it.
    pub fn track(&mut self, s: track::Subject) -> Result<(), String> {
        let v = self.shown();
        if self.pano || look_of(&v) != Look::Perspective {
            return Err("Tracking: draw the box in the perspective view".into());
        }
        let (i, ..) = self
            .cur()
            .ok_or("Tracking: the frame at the playhead is not shown yet")?;
        let d = &self.decks[i];
        let (k, k0) = (d.k.ok_or("no clip")?.0, d.held.ok_or("no frame")?.k);
        let c = self.pj.tl().clips[k].clone();
        if c.hold > 0.0 || c.rev {
            return Err("Tracking follows clips that play forwards".into());
        }
        let k1 = clip::frame_at(&d.s.v.times, c.b);
        let path = self.pj.tl().files[c.file as usize].path.clone();
        let f = self
            .info(&path)
            .and_then(|i| i.files.first().cloned())
            .ok_or("no main file")?;
        if let Some(t) = &self.trk {
            t.t.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let (a, bb) = (c.a, c.b);
        let t = ai::Track::start(
            &self.g,
            &self.cx,
            &self.ctx,
            f,
            (self.x.stab, self.x.smooth),
            k0,
            k1,
            s,
        );
        self.trk = Some(Trk {
            t,
            k,
            path,
            a,
            b: bb,
            v: to_edit(&v),
            keyed: false,
        });
        (self.arm, self.err) = (false, None);
        Ok(())
    }

    /// Auto framing of the clip at the playhead (`ai::Track::framing` over its whole file): the result becomes the clip's
    /// reframing keys over all of it, the viewer's view turned onto it.
    pub fn frame(&mut self) -> Result<(), String> {
        let v = self.shown();
        if self.pano || !matches!(look_of(&v), Look::Perspective | Look::Panini) {
            return Err(
                "Auto framing keys a perspective or Panini view (not the 360 frame)".into(),
            );
        }
        let tl = self.pj.tl();
        let k = (tl.at(self.t.min(tl.dur() - 1e-6).max(0.0)))
            .ok_or("Auto framing: no clip at the playhead")?
            .k;
        let c = tl.clips[k].clone();
        if c.hold > 0.0 || c.rev {
            return Err("Auto framing frames clips that play forwards".into());
        }
        let path = tl.files[c.file as usize].path.clone();
        let f = self
            .info(&path)
            .and_then(|i| i.files.first().cloned())
            .ok_or("no main file")?;
        if let Some(t) = &self.trk {
            t.t.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let st = (self.x.stab, self.x.smooth);
        let t = ai::Track::framing(&self.g, &self.cx, &self.ctx, f, st, self.aspect());
        self.trk = Some(Trk {
            t,
            k,
            path,
            a: c.a,
            b: c.b,
            v: to_edit(&v),
            keyed: false,
        });
        (self.arm, self.err) = (false, None);
        Ok(())
    }

    /// A finished run's keys onto its clip (one undo step): keys every 0.5 s from where the run started (auto framing:
    /// the clip's start) to the clip's end replace the clip's keys there.
    fn rekey(&mut self) {
        let Some(t) = self.trk.as_mut().filter(|t| !t.keyed && t.t.st().over()) else {
            return;
        };
        t.keyed = true;
        let w = ["Tracking", "Auto framing"][t.t.auto as usize];
        match t.t.st() {
            ai::St::Done => {}
            ai::St::Failed(e) => {
                self.err = Some(format!("{w}: {e}"));
                return;
            }
            _ => return,
        }
        let (k, path, a, b) = (t.k, t.path.clone(), t.a, t.b);
        let s0 = match t.t.auto {
            true => a,
            false => (t.t.times.get())
                .and_then(|v| v.get(t.t.k0).copied())
                .unwrap_or(a),
        };
        let keys = ai::keys(&t.t, s0.max(a), b, 0.5, &t.v);
        let tl = self.pj.tl();
        if tl
            .clips
            .get(k)
            .is_none_or(|c| tl.files[c.file as usize].path != path || c.a > s0 || c.b < s0)
        {
            self.err = Some(format!("{w}: the clip changed while it was tracked"));
            return;
        }
        if keys.len() < 2 {
            self.err = Some(format!("{w}: too short a track for keys"));
            return;
        }
        let (lo, hi) = (keys[0].t, keys[keys.len() - 1].t);
        if self
            .ed(|p| {
                p.edit(k, |c| {
                    c.view
                        .retain(|x| x.t < lo - edit::GAP || x.t > hi + edit::GAP);
                    c.view.extend(keys);
                    c.view.sort_by(|x, y| x.t.total_cmp(&y.t));
                })
            })
            .is_some()
        {
            self.free = None;
        }
    }

    /// Puts an auto edit of `p` (highlights framed for the export preset, cut to the beats of the music chosen for auto
    /// edits) on the timeline once made; an empty timeline takes stabilisation `stab` (the player's) first, a timeline
    /// with clips keeps its own, which the analysis uses.
    pub fn auto(&mut self, p: &Path, stab: (gyro::Mode, f64)) {
        let f = crate::library::pair(p).main.into_iter().next();
        let Some(f) = f else {
            self.err = Some(format!("{}: auto edit needs the main file", p.display()));
            return;
        };
        if self.pj.tl().is_empty() {
            self.adopt(stab);
        }
        if let Some(a) = &self.auto {
            a.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.auto = Some(ai::Auto::start(
            &self.g,
            &self.cx,
            &self.ctx,
            f,
            (self.x.stab, self.x.smooth),
            self.aspect(),
            self.beat.clone(),
        ));
    }

    /// The export preset's flat frame, whatever the window: a flat preset's own (the viewer letterboxes to it), 16:9
    /// for a 360 one (a flat 1080p export's, as the player's `box=`).
    fn aspect(&self) -> f32 {
        match crate::export::PRESETS[self.preset] {
            (_, _, true, ..) => 16.0 / 9.0,
            (_, _, _, [w, h], ..) => w as f32 / h as f32,
        }
    }

    /// A finished auto edit's clips onto the timeline.
    fn place(&mut self) {
        let Some(a) = self.auto.take_if(|a| a.st().over()) else {
            return;
        };
        let tl = match (
            a.st(),
            a.out.lock().unwrap_or_else(|e| e.into_inner()).take(),
        ) {
            (ai::St::Done, Some(tl)) => tl,
            (st, _) => {
                if !matches!(st, ai::St::Cancelled) {
                    self.err = Some(st.line("Auto edit"));
                }
                return;
            }
        };
        match self.put(tl.clone(), a.stab) {
            Ok(_) => {
                eprintln!(
                    "x5desk editor: auto edit of {}: {} highlights, {:.2} s",
                    name(&tl.files[0].path),
                    tl.clips.len(),
                    tl.dur()
                );
                if let Some(tok) = self.autox.take() {
                    self.export(tok);
                }
            }
            Err(e) => self.err = Some(e),
        }
    }

    /// The clips of `tl` (of its first file; views in its frame stitched with `stab`) onto the end of the timeline as
    /// one undo step; an empty timeline takes its frame rate and `stab`. Its music, if any, becomes the timeline's,
    /// moved with the clips (earlier clips play without). Returns the first new clip's index.
    pub fn put(&mut self, tl: edit::Timeline, stab: (gyro::Mode, f64)) -> Result<usize, String> {
        let f0 = tl.files.first().ok_or("no file")?;
        let empty = self.pj.tl().is_empty();
        if !empty && stab != (self.x.stab, self.x.smooth) {
            return Err(format!(
                "These views are for {} stabilisation, the timeline has another: set it in Reframe first",
                gyro::MODES
                    .iter()
                    .find(|m| m.0 == stab.0)
                    .map_or("", |m| m.1)
            ));
        }
        let r = self.pj.file(&f0.path, f0.dur).and_then(|f| {
            let fps = empty && self.pj.tl().out.fps != tl.out.fps;
            let (n, t0) = (self.pj.tl().clips.len(), self.pj.tl().dur());
            self.begin();
            let mut r = match fps {
                true => self.pj.edit_out(|o| o.fps = tl.out.fps),
                false => Ok(()),
            };
            if let Some(m) = &tl.out.music
                && r.is_ok()
            {
                let mf = &tl.files[m.file as usize];
                r = self.pj.file(&mf.path, mf.dur).and_then(|file| {
                    self.pj.edit_out(|o| {
                        o.music = Some(edit::Music {
                            file,
                            offset: m.offset - t0,
                            ..m.clone()
                        })
                    })
                });
            }
            for (i, c) in tl.clips.iter().enumerate() {
                if r.is_ok() {
                    r = self.pj.insert(
                        n + i,
                        edit::Clip {
                            file: f,
                            ..c.clone()
                        },
                    );
                }
            }
            self.end();
            r.map(|_| n)
        });
        let n = self
            .did(r)
            .ok_or_else(|| self.err.clone().unwrap_or_default())?;
        if empty {
            self.adopt(stab);
        }
        (self.sel, self.free) = (Some(n), None);
        self.seek(self.pj.tl().out.fps.t(self.pj.tl().frames(n).start));
        Ok(n)
    }

    fn preset(&mut self, keys: edit::Result<Vec<Key>>) {
        let Some(on) = self.on() else { return };
        let r = keys.and_then(|k| self.pj.preset(on, self.t, &k));
        if self.did(r).is_some() {
            self.free = None;
        }
    }

    /// Moves the playhead; a keyed view follows it.
    pub fn seek(&mut self, t: f64) {
        let d = self.pj.tl().dur();
        self.t = t.clamp(0.0, d.max(0.0));
        if self.keyed() {
            self.free = None;
        }
        if self.playing {
            self.pause();
            self.play();
        }
        self.dirty = true;
    }

    pub fn play(&mut self) {
        let tl = self.pj.tl();
        if tl.is_empty() {
            return;
        }
        if self.t >= tl.dur() - tl.out.fps.t(1) {
            self.t = 0.0;
        }
        if self.keyed() {
            self.free = None;
        }
        let files = tl.files.iter().map(|f| PathBuf::from(&f.path)).collect();
        let m = sound::Mix {
            tl: tl.clone(),
            files,
            denoise: self.x.denoise,
        };
        self.snd = Some(sound::Sound::start(m, self.t, self.vol));
        self.playing = true;
        if let Some(l) = &mut self.log {
            *l = (0, Some((std::time::Instant::now(), self.t)));
        }
    }

    pub fn pause(&mut self) {
        if let (true, Some((n, Some((w, t0))))) = (self.playing, &mut self.log) {
            let (s, d) = (w.elapsed().as_secs_f64(), self.t - *t0);
            eprintln!(
                "x5desk editor: played {:.2} s of timeline in {s:.2} s, {n} frames shown ({:.1} per s), sound {}",
                d,
                *n as f64 / s.max(1e-9),
                if self.snd.as_ref().is_some_and(|s| s.live()) {
                    "on the device"
                } else {
                    "off (wall clock)"
                }
            );
            self.log = Some((0, None));
        }
        self.playing = false;
        self.snd = None;
    }

    /// Imports `p` as the music, ducked under the clips. A file whose container gives no length is decoded through on
    /// a thread to count it (a song takes about a second) and goes in once counted (`tick`).
    pub fn music(&mut self, p: &Path) -> Result<(), String> {
        self.mq.0 = None;
        let mut m = media::music::Music::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
        if m.duration > 0.0 {
            return self.song(p, m.duration);
        }
        let (tx, p) = (self.mq.1.clone(), p.to_path_buf());
        self.mq.0 = Some(p.clone());
        thread::spawn(move || {
            let (mut b, mut n) = (vec![0f32; 2 * 8192], 0usize);
            // up to the end or a read error
            while let Ok(k @ 1..) = m.read(&mut b) {
                n += k;
            }
            let _ = tx.send((p, n as f64 / media::music::RATE as f64));
        });
        Ok(())
    }

    /// Puts music file `p` of `dur` s on the timeline.
    fn song(&mut self, p: &Path, dur: f64) -> Result<(), String> {
        let f = self.pj.file(&p.to_string_lossy(), dur).map_err(|e| e.0)?;
        // a count may end mid-drag: through `ed`, which ends the drag first
        self.ed(|pj| {
            pj.edit_out(|o| {
                o.music = Some(edit::Music {
                    file: f,
                    offset: 0.0,
                    gain: 1.0,
                    duck: true,
                })
            })
        })
        .ok_or_else(|| self.err.clone().unwrap_or_default())
    }

    /// Imports `.cube` file `p` as the selected clip's look (its path is the look's id).
    pub fn cube(&mut self, p: &Path) -> Result<(), String> {
        let k = (self.sel)
            .filter(|&k| k < self.pj.tl().clips.len())
            .ok_or("Select a clip for the look.")?;
        let id = std::fs::canonicalize(p)
            .unwrap_or_else(|_| p.to_path_buf())
            .to_string_lossy()
            .to_string();
        // read again: the file may have changed since
        self.vw.cubes.remove(&id);
        Viewer::cube(&mut self.vw.cubes, &id)?;
        (self.vw.look, self.dirty) = (None, true);
        self.ed(|pj| pj.edit(k, |c| c.look = id))
            .ok_or_else(|| self.err.clone().unwrap_or_default())
    }

    pub fn save(&mut self, p: &Path) -> Result<(), String> {
        let s = doc::save(&self.pj, &self.x);
        std::fs::write(p, &s).map_err(|e| format!("{}: {e}", p.display()))?;
        (self.path, self.saved) = (Some(p.to_path_buf()), s);
        Ok(())
    }

    pub fn load(&mut self, p: &Path) -> Result<(), String> {
        let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        let (pj, x) = doc::load(&s).map_err(|e| format!("{}: {e}", p.display()))?;
        self.take(pj, x, Some(p.to_path_buf()));
        Ok(())
    }

    /// Takes over a project: no history, the playhead at the start.
    fn take(&mut self, pj: Project, x: Extra, path: Option<PathBuf>) {
        self.pause();
        self.saved = doc::save(&pj, &x);
        (self.pj, self.xs, self.x, self.path) = (pj, x.clone(), x, path);
        (self.hist, self.fut, self.eh, self.mq.0) = (vec![], vec![], (0, 0), None);
        (self.t, self.sel, self.key, self.free) = (0.0, None, None, None);
        self.changed();
    }

    pub fn modified(&self) -> bool {
        doc::save(&self.pj, &self.x) != self.saved
    }

    /// The export job of the timeline.
    pub fn job(&self) -> Result<Tl, String> {
        let tl = self.pj.tl();
        if tl.is_empty() {
            return Err("The timeline is empty.".into());
        }
        if tl.clips.iter().any(|c| c.rev && c.hold == 0.0) {
            return Err("Reversed clips do not export yet.".into());
        }
        let music = tl.out.music.as_ref().map(|m| m.file as usize);
        let files: Vec<Vec<PathBuf>> = (tl.files.iter().enumerate())
            .map(|(i, f)| match Some(i) == music {
                true => vec![PathBuf::from(&f.path)],
                false => crate::library::pair(Path::new(&f.path)).main,
            })
            .collect();
        // a clip of a proxy or another file with no main file beside it: nothing to render it from (the export's
        // face pass reads a clip's main files)
        if let Some(c) = tl.clips.iter().find(|c| files[c.file as usize].is_empty()) {
            let p = &tl.files[c.file as usize].path;
            return Err(format!("{}: no full-resolution file", name(p)));
        }
        let mut cubes = HashMap::new();
        let looks = (tl.clips.iter())
            .filter(|c| !c.look.is_empty())
            .map(|c| Viewer::cube(&mut cubes, &c.look).map(|q| (c.look.clone(), q)))
            .collect::<Result<_, String>>()?;
        let over = (self.x.over.iter())
            .map(|o| Ok((o.image()?, o.place, o.opacity)))
            .collect::<Result<_, String>>()?;
        Ok(Tl {
            name: self
                .path
                .as_ref()
                .and_then(|p| p.file_stem())
                .map_or("timeline".into(), |s| s.to_string_lossy().into()),
            tl: tl.clone(),
            files,
            looks,
            over,
            denoise: self.x.denoise,
            stab: self.x.stab,
            smooth: self.x.smooth,
        })
    }

    /// Hands the timeline to the Export screen with `tok` (`export::parse`: a preset key and options).
    pub fn export(&mut self, tok: String) {
        match self.job() {
            Ok(j) => self.ask = Some((j, tok)),
            Err(e) => self.err = Some(e),
        }
    }

    fn pkey(&self) -> PoseKey {
        (self.x.stab, self.x.smooth.to_bits())
    }

    /// The file's info, opened on first use; asks for its poses for the stabilisation setting unless they are in or
    /// being made (a thread making another setting's asks again when it is done).
    fn info(&mut self, p: &str) -> Option<&Info> {
        if !self.info.contains_key(p) {
            let i = Info::open(p);
            if let Err(e) = &i {
                self.err = Some(format!("{}: {e}", name(p)));
            }
            self.info.insert(p.into(), i);
        }
        let key = self.pkey();
        if let (Some(lock), Some(Ok(i))) = (key.0, self.info.get(p))
            && !i.posed(key)
            && !self.asked.contains_key(p)
        {
            self.asked.insert(p.into(), key);
            let (tx, path, ctx, s, g, cx) = (
                self.pq.0.clone(),
                p.to_string(),
                self.ctx.clone(),
                self.x.smooth,
                self.g.clone(),
                self.cx.clone(),
            );
            thread::spawn(move || {
                // the forward view: the clip's analysis first (kept for the process)
                let fwd = match lock {
                    stab::Lock::Forward => {
                        let f = [PathBuf::from(&path)];
                        let stop = std::sync::atomic::AtomicBool::new(false);
                        export::travel(&g, &cx, &f, &stop, &mut |_, _| {}).map(drop)
                    }
                    _ => Ok(()),
                };
                let r = fwd
                    .and_then(|_| shotlab::Clip::open(Path::new(&path)))
                    .and_then(|c| {
                        c.poses(lock, s)
                            .ok_or("no IMU record to stabilise with".into())
                    });
                let _ = tx.send((path, key, r));
                ctx.request_repaint();
            });
        }
        self.info.get(p)?.as_ref().ok()
    }

    /// Every frame on the editor (`shown`), else it pauses: replies, the clock, the frame due.
    pub fn tick(&mut self, shown: bool) {
        if !shown {
            if self.playing {
                self.pause();
            }
            return;
        }
        if let Some(h) = self.hook.take() {
            self.log = Some((0, None));
            self.apply(h);
        }
        self.thumbs.tick();
        // the music counted, unless another import came since
        while let Ok((p, d)) = self.mq.2.try_recv() {
            if self.mq.0.as_ref() == Some(&p) {
                self.mq.0 = None;
                if let Err(e) = self.song(&p, d) {
                    self.err = Some(e);
                }
            }
        }
        // a drag whose widget went away saw no end: it ends after the button's release (the release's frame still
        // reports a widget's end to it)
        if !self
            .ctx
            .input(|i| i.pointer.any_down() || i.pointer.any_released())
        {
            if matches!(self.drag, Some(Drag::Slider(_))) {
                self.drag = None;
            }
            if self.grp {
                self.end();
            }
        }
        self.sync();
        while let Ok((p, k, r)) = self.pq.1.try_recv() {
            self.asked.remove(&p);
            if k != self.pkey() {
                // the setting changed meanwhile: its poses next
                self.info(&p);
                continue;
            }
            if let Err(e) = &r {
                self.err = Some(format!("{}: stabilisation: {e}", name(&p)));
            }
            if let Some(Ok(i)) = self.info.get_mut(&p) {
                i.poses = Some((k, r.map(Into::into)));
            }
            self.dirty = true;
        }
        while let Ok(r) = self.rr.1.try_recv() {
            self.busy = false;
            match r {
                Ok((k, t)) => {
                    self.seam = Some((k, self.cx.pano(&t.create_view(&Default::default()))));
                    self.dirty = true;
                }
                Err(e) => {
                    self.err = Some(format!("RAFT stitch: {e}"));
                    self.raft_off = true;
                }
            }
        }
        if let Some((d, r)) = &self.dlg
            && let Ok(p) = r.try_recv()
        {
            let d = *d;
            self.dlg = None;
            if let Some(p) = p {
                self.picked(d, &p);
            }
        }
        if self.playing {
            let d = self.pj.tl().dur();
            if let Some(s) = &self.snd {
                self.t = s.now();
                if let Some(e) = s.err() {
                    self.err = Some(e);
                }
            }
            if self.t >= d {
                self.pause();
                self.t = d;
            }
            self.ctx.request_repaint();
        }
        self.schedule();
        // paused on a frame for a moment (its poses in): its RAFT stitch
        if !self.playing
            && !self.busy
            && !self.raft_off
            && self.posed()
            && let Some((i, o, k)) = self.cur()
            && self.decks[i].want.is_none()
            && self.seam.as_ref().is_none_or(|s| s.0 != k)
        {
            match self.since.elapsed() >= Duration::from_millis(150) {
                true => {
                    let d = &self.decks[i];
                    let h = d.held.expect("a frame shown");
                    let j = (d.s).job(&self.g, &h, o.rs, k, &mut self.snap, &self.rr.0, &self.ctx);
                    match self.rw.send(j) {
                        Ok(()) => self.busy = true,
                        Err(_) => self.raft_off = true,
                    }
                }
                false => self.ctx.request_repaint_after(Duration::from_millis(150)),
            }
        }
        if self.autoplay && !self.playing && self.settled() {
            self.autoplay = false;
            self.play();
        }
        self.rekey();
        self.place();
        // the hook's box is in the preset's frame, not in the viewer's picture: under a 360 preset that is the viewer's
        // rect, shaped by the window's size and layout
        if self.tbox.is_some() && self.settled() {
            let s = ai::subject(&self.shown(), self.aspect(), self.tbox.take().unwrap());
            let [y, x] = track::yaw_pitch(s.dir);
            eprintln!(
                "x5desk editor: box: subject at yaw {y:.2}°, pitch {x:.2}°, {:.2}° × {:.2}°",
                s.size.x, s.size.y
            );
            if let Err(e) = self.track(s) {
                eprintln!("x5desk editor: box: {e}");
                self.err = Some(e);
            }
        }
        if self.trk.as_ref().is_some_and(|t| !t.t.st().over()) || self.auto.is_some() {
            self.ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    /// Positions the decks: the clip at the playhead, and while playing the next clip's first frame ahead of the cut.
    fn schedule(&mut self) {
        let tl = self.pj.tl();
        if tl.is_empty() {
            return;
        }
        let t = self.t.min(tl.dur() - 0.5 * tl.out.fps.t(1)).max(0.0);
        let Some(e) = tl.at(t) else { return };
        let (k, src, path) = (e.k, e.src, tl.files[e.clip.file as usize].path.clone());
        let next = (self.playing && k + 1 < tl.clips.len())
            .then(|| {
                let t1 = tl.out.fps.t(tl.frames(k + 1).start);
                (t1 - t < 1.0).then(|| {
                    (
                        k + 1,
                        src0(tl, k + 1),
                        tl.files[tl.clips[k + 1].file as usize].path.clone(),
                    )
                })
            })
            .flatten();
        let Some(i) = self.deck(k, &path, src, None) else {
            return;
        };
        let (g, ver) = (self.g.clone(), self.ver);
        let d = &mut self.decks[i];
        d.k = Some((k, ver));
        let kf = clip::frame_at(&d.s.v.times, src);
        if !self.playing || d.held.is_some_and(|h| h.k > kf) || d.pos().is_none_or(|p| p + 30 < kf)
        {
            d.seek(&g, kf);
        }
        let new = d.take(&g, src);
        // a seek target queued is shown once decoded: the media thread wakes the UI when it queues, not after
        if d.want.is_some() && d.s.v.queued() {
            self.ctx.request_repaint_after(Duration::from_millis(4));
        }
        self.fresh |= new;
        if new {
            self.since = Instant::now();
        }
        if let (true, Some(l)) = (new && self.playing, &mut self.log) {
            l.0 += 1;
        }
        if let Some((k1, s1, p1)) = next {
            let ver = self.ver;
            let cont = self.decks[i].path == p1
                && self.decks[i].near(clip::frame_at(&self.decks[i].s.v.times, s1));
            if !cont
                && !self.decks.iter().any(|d| d.k == Some((k1, ver)))
                && let Some(j) = self.deck(k1, &p1, s1, Some(i))
            {
                let d = &mut self.decks[j];
                d.k = Some((k1, ver));
                let kf = clip::frame_at(&d.s.v.times, s1);
                d.seek(&g, kf);
            }
        }
        if let Some(e) = self.decks.iter().find_map(|d| d.s.v.err()) {
            self.err = Some(e);
        }
    }

    /// A deck for clip `k` of file `path` at source time `src`, not deck `keep`: the one serving it, one of the file
    /// at that frame, a few frames before, else a free or the other deck, reopened on the file if needed.
    fn deck(&mut self, k: usize, path: &str, src: f64, keep: Option<usize>) -> Option<usize> {
        let ver = self.ver;
        let ok = |i: usize| Some(i) != keep;
        if let Some(i) = (0..self.decks.len()).find(|&i| ok(i) && self.decks[i].k == Some((k, ver)))
        {
            return Some(i);
        }
        // of the file: showing the frame at once, decoding on from a few frames before (a split clip playing on),
        // else seeking
        let mine = |i: usize, m: u8| {
            let d = &self.decks[i];
            let kf = clip::frame_at(&d.s.v.times, src);
            let held = d.want.is_none() && d.held.is_some_and(|h| h.k == kf);
            ok(i) && d.path == path && [held, d.near(kf), true][m as usize]
        };
        let n = self.decks.len();
        if let Some(i) = (0..3).find_map(|m| (0..n).find(|&i| mine(i, m))) {
            return Some(i);
        }
        // else a new deck, or one that serves nothing now
        let i = match n < DECKS {
            true => None,
            false => (0..n)
                .find(|&i| ok(i) && self.decks[i].k.is_none_or(|x| x.1 != ver))
                .or_else(|| (0..n).find(|&i| ok(i))),
        };
        // the old deck goes off the UI thread: joining its media thread and freeing its rings (which waits for the
        // GPU's queue) took ≈ 50 ms there, frames lost at a cut. Its frame goes unreleased: the ring goes with it, and
        // a release staged now could ride another thread's submit past the ring's end.
        if let Some(i) = i {
            let mut d = self.decks.remove(i);
            d.held = None;
            thread::spawn(move || drop(d));
        }
        self.info(path)?;
        let inf = self.info.get(path)?.as_ref().ok()?;
        match Deck::open(&self.g, &self.cx, path, inf, &self.ctx) {
            Ok(d) => {
                self.decks.push(d);
                Some(self.decks.len() - 1)
            }
            Err(e) => {
                self.err = Some(format!("{}: {e}", name(path)));
                None
            }
        }
    }

    /// The deck showing the playhead's clip, its frame's pose and RAFT key.
    fn cur(&self) -> Option<(usize, Pose, player::Key)> {
        let tl = self.pj.tl();
        let e = tl.at(self.t.min(tl.dur() - 0.5 * tl.out.fps.t(1)).max(0.0))?;
        let i = (self.decks.iter()).position(|d| d.k == Some((e.k, self.ver)))?;
        let (d, key) = (&self.decks[i], self.pkey());
        let h = d.held?;
        let o = self.info.get(&d.path)?.as_ref().ok()?.pose(h.k, key);
        Some((i, o, (d.s.id, h.k, o.rs[0] != glam::Vec3::ZERO)))
    }

    /// Nothing pending: not playing, the frame at the playhead shown (with its RAFT stitch), thumbnails and poses in,
    /// no export to hand over.
    pub fn idle(&self) -> bool {
        !self.autoplay
            && !self.playing
            && self.settled()
            && self.auto.is_none()
            && self.tbox.is_none()
            && self.trk.as_ref().is_none_or(|t| t.keyed)
    }

    fn settled(&self) -> bool {
        let tl = self.pj.tl();
        let shown = tl.is_empty()
            || self.decks.iter().any(|d| {
                d.k.is_some_and(|x| x.1 == self.ver) && d.held.is_some() && d.want.is_none()
            });
        let raft = self.raft_off
            || (self.cur()).is_none_or(|c| self.seam.as_ref().is_some_and(|s| s.0 == c.2));
        self.hook.is_none()
            && shown
            && self.posed()
            && raft
            && !self.thumbs.busy()
            && self.ask.is_none()
            && self.mq.0.is_none()
            && !self.fresh
    }

    /// Every clip's file has its poses for the stabilisation setting (or they failed, or it has no info).
    fn posed(&self) -> bool {
        let (tl, key) = (self.pj.tl(), self.pkey());
        tl.clips.iter().all(|c| {
            (self.info.get(&tl.files[c.file as usize].path))
                .is_none_or(|i| i.as_ref().map_or(true, |i| i.posed(key)))
        })
    }

    fn apply(&mut self, h: Hook) {
        let mut r = Ok(());
        if let Some(p) = &h.project {
            r = self.load(p);
        }
        for p in &h.add {
            if r.is_ok() {
                r = self.add(p, None).map(|_| ());
            }
        }
        for _ in 0..h.undo {
            self.undo();
        }
        for _ in 0..h.redo {
            self.redo();
        }
        if let Some(t) = h.at {
            self.seek(t);
        }
        if let Some(k) = h.sel {
            self.sel = Some(k);
        }
        if let Some(t) = h.tab {
            self.tab = t;
        }
        self.pano |= h.pano;
        self.tbox = h.tbox;
        if let Some(k) = &h.preset {
            match crate::export::PRESETS.iter().position(|p| p.0 == k) {
                Some(i) => self.preset = i,
                None => r = r.and(Err(format!("no preset {k}"))),
            }
        }
        if let (Some(p), true) = (&h.save, r.is_ok()) {
            r = self.save(p);
        }
        match r {
            Err(e) => {
                eprintln!("x5desk editor: {e}");
                self.err = Some(e);
            }
            Ok(()) => {
                self.sync();
                let (u, d) = self.history();
                let tl = self.pj.tl();
                let spans: Vec<_> = (0..tl.clips.len()).map(|k| tl.frames(k)).collect();
                eprintln!(
                    "x5desk editor: {} clips, {} frames ({:.3} s) at {}/{}, undo {u} redo {d}; clip frames {spans:?}",
                    tl.clips.len(),
                    tl.len(),
                    tl.dur(),
                    tl.out.fps.num,
                    tl.out.fps.den
                );
            }
        }
        if let Some(tok) = h.export {
            self.export(tok);
            if let Some(e) = &self.err {
                eprintln!("x5desk editor: export: {e}");
            }
        }
        self.autoplay = h.play;
    }

    fn dialog(&mut self, d: Dlg) {
        if self.dlg.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let ctx = self.ctx.clone();
        let dir = self
            .path
            .as_ref()
            .and_then(|p| p.parent().map(Path::to_path_buf));
        // the portal dialog blocks: off the UI thread
        thread::spawn(move || {
            let mut f = rfd::FileDialog::new();
            if let Some(d) = dir {
                f = f.set_directory(d);
            }
            let p = match d {
                Dlg::Add => f.add_filter("X5 video", &["insv", "mp4"]).pick_file(),
                Dlg::Open => f.add_filter("x5desk project", &["x5edit"]).pick_file(),
                Dlg::Save => f
                    .add_filter("x5desk project", &["x5edit"])
                    .set_file_name("project.x5edit")
                    .save_file(),
                Dlg::Music | Dlg::Beat => f.add_filter("Music", &MUSIC).pick_file(),
                Dlg::Png | Dlg::Nadir => f.add_filter("PNG", &["png"]).pick_file(),
                Dlg::Cube => f.add_filter("Cube LUT", &["cube"]).pick_file(),
            };
            let _ = tx.send(p);
            ctx.request_repaint();
        });
        self.dlg = Some((d, rx));
    }

    fn picked(&mut self, d: Dlg, p: &Path) {
        let r = match d {
            Dlg::Add => self.add(p, None).map(|_| ()),
            Dlg::Open => self.load(p),
            Dlg::Save => self.save(p),
            Dlg::Music => self.music(p),
            Dlg::Beat => media::music::Music::open(p)
                .map(|_| self.beat = Some(p.to_path_buf()))
                .map_err(|e| format!("{}: {e}", p.display())),
            Dlg::Cube => self.cube(p),
            Dlg::Png | Dlg::Nadir => {
                let s = p.to_string_lossy().to_string();
                self.x.over.push(Over {
                    pic: Pic::Png(s),
                    place: match d {
                        Dlg::Nadir => fx::Place::Sphere {
                            yaw: 0.0,
                            pitch: -90.0,
                            fov: 40.0,
                            roll: 0.0,
                        },
                        _ => fx::Place::Flat {
                            at: [0.88, 0.1],
                            w: 0.15,
                            rot: 0.0,
                        },
                    },
                    opacity: 0.9,
                });
                self.dirty = true;
                Ok(())
            }
        };
        if let Err(e) = r {
            self.err = Some(e);
        }
    }

    /// The screen; `media`: the library's videos (drag them onto the timeline).
    pub fn ui(&mut self, ui: &mut egui::Ui, rs: &RenderState, media: &[PathBuf]) {
        if ui.ctx().memory(|m| m.focused().is_none()) {
            let (sp, s, del, k, z, y, l, r, sh) = ui.input(|i| {
                let c = i.modifiers.command;
                (
                    i.key_pressed(egui::Key::Space),
                    i.key_pressed(egui::Key::S) && !c,
                    // the key Mac keyboards label delete is Backspace (fn+delete is Delete)
                    i.key_pressed(egui::Key::Delete)
                        || cfg!(target_os = "macos") && i.key_pressed(egui::Key::Backspace),
                    i.key_pressed(egui::Key::K),
                    c && i.key_pressed(egui::Key::Z),
                    c && i.key_pressed(egui::Key::Y),
                    i.key_pressed(egui::Key::ArrowLeft),
                    i.key_pressed(egui::Key::ArrowRight),
                    i.modifiers.shift,
                )
            });
            if sp {
                if self.playing {
                    self.pause()
                } else {
                    self.play()
                }
            }
            if s {
                self.split();
            }
            if let (true, Some(k)) = (del, self.sel) {
                self.delete(k);
            }
            if k {
                self.key_here();
            }
            match (z, sh, y) {
                (true, false, _) => self.undo(),
                (true, true, _) | (_, _, true) => self.redo(),
                _ => {}
            }
            if l || r {
                let dt = self.pj.tl().out.fps.t(1);
                self.seek(self.t + if l { -dt } else { dt });
            }
        }
        egui::Panel::top("ed bar").show(ui, |ui| self.bar(ui));
        egui::Panel::left("ed media")
            .default_size(200.0)
            .show(ui, |ui| self.media(ui, media));
        egui::Panel::right("ed inspector")
            .default_size(330.0)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for (t, n) in TABS {
                        ui.selectable_value(&mut self.tab, t, n);
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .auto_shrink(false)
                    .show(ui, |ui| match self.tab {
                        Tab::Clip => self.clip_ui(ui),
                        Tab::View => self.view_ui(ui),
                        Tab::Colour => self.colour_ui(ui),
                        Tab::Audio => self.audio_ui(ui),
                        Tab::Over => self.over_ui(ui),
                    });
            });
        egui::Panel::bottom("ed timeline")
            .resizable(false)
            .show(ui, |ui| self.timeline(ui));
        egui::CentralPanel::no_frame().show(ui, |ui| self.viewer(ui, rs));
    }

    fn bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("New").clicked() {
                let pj = Project::new(FPS).expect("a valid frame rate");
                self.take(pj, Extra::default(), None);
            }
            if ui.button("Open…").clicked() {
                self.dialog(Dlg::Open);
            }
            if ui
                .add_enabled(self.path.is_some(), egui::Button::new("Save"))
                .clicked()
                && let Some(p) = self.path.clone()
                && let Err(e) = self.save(&p)
            {
                self.err = Some(e);
            }
            if ui.button("Save as…").clicked() {
                self.dialog(Dlg::Save);
            }
            ui.separator();
            let (u, r) = self.history();
            if ui
                .add_enabled(u > 0, egui::Button::new("Undo"))
                .on_hover_text(if cfg!(target_os = "macos") {
                    "Cmd+Z"
                } else {
                    "Ctrl+Z"
                })
                .clicked()
            {
                self.undo();
            }
            if ui
                .add_enabled(r > 0, egui::Button::new("Redo"))
                .on_hover_text(match cfg!(target_os = "macos") {
                    true => "Cmd+Shift+Z",
                    false => "Ctrl+Shift+Z",
                })
                .clicked()
            {
                self.redo();
            }
            ui.label(format!("{u} / {r}"))
                .on_hover_text("Undo and redo steps");
            ui.separator();
            if ui
                .toggle_value(&mut self.pano, "360 frame")
                .on_hover_text("Preview the whole sphere as a 360 export renders it")
                .changed()
            {
                self.dirty = true;
            }
            ui.separator();
            egui::ComboBox::from_id_salt("ed preset")
                .selected_text(crate::export::PRESETS[self.preset].1)
                .show_ui(ui, |ui| {
                    for (i, p) in crate::export::PRESETS.iter().enumerate() {
                        ui.selectable_value(&mut self.preset, i, p.1);
                    }
                });
            ui.checkbox(&mut self.raft, "RAFT")
                .on_hover_text("RAFT seam on every frame (slower)");
            if ui
                .add_enabled(
                    !self.pj.tl().is_empty(),
                    egui::Button::new("Export timeline"),
                )
                .clicked()
            {
                let tok = format!(
                    "{}{}",
                    crate::export::PRESETS[self.preset].0,
                    if self.raft { "" } else { ":noraft" }
                );
                self.export(tok);
            }
            ui.separator();
            let tl = self.pj.tl();
            ui.label(format!(
                "{}{} · {} clips · {} · {:.2} fps",
                self.path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .map_or("Untitled".into(), |n| n.to_string_lossy()),
                if self.modified() { " *" } else { "" },
                tl.clips.len(),
                mmss(tl.dur()),
                tl.out.fps.num as f64 / tl.out.fps.den as f64
            ));
        });
        if let Some(a) = &self.auto {
            ui.horizontal(|ui| {
                ui.label(a.st().line("Auto edit"));
                if ui.button("Cancel").clicked() {
                    a.stop.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            });
        }
        if let Some(e) = &self.err {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        if let Some(e) = &self.vw.err {
            ui.colored_label(ui.visuals().error_fg_color, format!("Overlay: {e}"));
        }
        if let Some(e) = &self.vw.bad {
            ui.colored_label(ui.visuals().error_fg_color, format!("Look: {e}"));
        }
    }

    fn media(&mut self, ui: &mut egui::Ui, media: &[PathBuf]) {
        ui.heading("Media");
        if ui.button("Add file…").clicked() {
            self.dialog(Dlg::Add);
        }
        ui.label("Drag a recording onto the timeline, or press +.");
        ui.separator();
        let (mut add, mut auto) = (None, None);
        egui::ScrollArea::vertical()
            .id_salt("ed media list")
            .show(ui, |ui| {
                for p in media {
                    ui.horizontal(|ui| {
                        if ui
                            .small_button("+")
                            .on_hover_text("Append to the timeline")
                            .clicked()
                        {
                            add = Some(p.clone());
                        }
                        let n = p
                            .file_name()
                            .map_or(String::new(), |n| n.to_string_lossy().into());
                        ui.dnd_drag_source(egui::Id::unique(("ed media", p)), p.clone(), |ui| {
                            ui.label(n).on_hover_text(p.to_string_lossy());
                        })
                        .response
                        .context_menu(|ui| {
                            if ui
                                .button("Auto edit")
                                .on_hover_text(
                                    "Its highlights, framed automatically, onto the timeline",
                                )
                                .clicked()
                            {
                                auto = Some(p.clone());
                            }
                        });
                    });
                }
                if media.is_empty() {
                    ui.label("Open recordings in the library first, or add a file.");
                }
            });
        if let Some(p) = add
            && let Err(e) = self.add(&p, None)
        {
            self.err = Some(e);
        }
        if let Some(p) = auto {
            self.auto(&p, (self.x.stab, self.x.smooth));
        }
        ui.separator();
        ui.label("Auto edit music");
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Choose…")
                .on_hover_text(
                    "MP3, FLAC, WAV, Ogg Vorbis, M4A: it plays under an auto edit's highlights, ducked under their \
                     audio, the cuts on its beats",
                )
                .clicked()
            {
                self.dialog(Dlg::Beat);
            }
            if self.beat.is_some() && ui.button("Remove").clicked() {
                self.beat = None;
            }
        });
        match &self.beat {
            Some(p) => ui.label(name(&p.to_string_lossy())),
            None => ui.label("No music"),
        };
    }

    fn viewer(&mut self, ui: &mut egui::Ui, rs: &RenderState) {
        let (rect, resp) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let mut v = self.shown();
        let look = if self.pano {
            Look::Equirect
        } else {
            look_of(&v)
        };
        // a flat export's frame: what the keys show in it
        let (_, _, p360, [w, h], ..) = crate::export::PRESETS[self.preset];
        let r = match (self.pano, p360) {
            (false, false) => {
                let a = w as f32 / h as f32;
                let x = rect.width().min(rect.height() * a);
                egui::Rect::from_center_size(rect.center(), egui::vec2(x, x / a))
            }
            _ => look.fit(rect),
        };
        let mut live = None;
        if self.arm && !self.pano && look == Look::Perspective {
            let (l, b) = ai::drag_box(ui, &resp, r, &mut self.from);
            live = l;
            if let Some(b) = b
                && let Err(e) = self.track(ai::subject(&v, r.aspect_ratio(), b))
            {
                self.err = Some(e);
            }
        } else if !self.pano && look.steer(ui, &resp, r, &mut v) {
            if resp.double_clicked() {
                v = Look::Perspective.view();
            }
            (self.free, self.dirty) = (Some(v), true);
        }
        let ppp = ui.ctx().pixels_per_point();
        let size = [r.width(), r.height()].map(|x| ((x * ppp).round() as u32).max(1));
        if self.fresh || self.dirty || size != self.size {
            // decks as of this edit: a change unbinds them (`changed`), and drawn unbound the frame went black for good
            // when it stayed the same (no new frame to set `fresh` once bound again)
            self.schedule();
            let tl = self.pj.tl();
            let e = tl.at(self.t.min(tl.dur() - 0.5 * tl.out.fps.t(1)).max(0.0));
            let c = self.cur();
            // the playhead's frame on its way (a seek, a cut): the picture shown stays until it comes
            let wait = c.is_none()
                && e.is_some_and(|e| {
                    (self.decks.iter()).any(|d| d.k == Some((e.k, self.ver)) && d.want.is_some())
                });
            if !wait {
                let (look, amt, adj) = e.map_or((String::new(), 1.0, fx::Params::default()), |e| {
                    (e.clip.look.clone(), e.clip.amt, e.clip.adj)
                });
                let seam = (c.as_ref())
                    .and_then(|c| self.seam.as_ref().filter(|s| !self.playing && s.0 == c.2))
                    .map(|s| &s.1);
                let vv = if self.pano {
                    View {
                        mode: Mode::Equirect,
                        ..View::default()
                    }
                } else {
                    v
                };
                let d = c.map(|(i, o, _)| (&mut self.decks[i], o));
                self.vw.render(
                    rs,
                    size,
                    d,
                    seam,
                    &vv,
                    self.pano,
                    &adj,
                    (&look, amt),
                    &self.x.over,
                    self.t,
                );
                (self.fresh, self.dirty, self.size) = (false, false, size);
            }
        }
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 0.0, egui::Color32::BLACK);
        if let Some(t) = &self.vw.tgt {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            p.image(t.id, r, uv, egui::Color32::WHITE);
        }
        if let Some(b) = live {
            ai::paint_box(&p, b);
        }
        // the tracked box on the frame shown, in the viewer's view
        if let (Some(t), Some((i, ..)), false) = (&self.trk, self.cur(), self.pano)
            && look_of(&v) == Look::Perspective
            && let Some(h) = self.decks[i].held
            && self.decks[i].path == t.path
            && let Some(s) = t.t.at(h.k)
        {
            for l in ai::outline(&v, r, s.raw, s.size) {
                p.line(l, egui::Stroke::new(2.0, ai::colour(s.state)));
            }
        }
        let tl = self.pj.tl();
        let mut lines = vec![format!("{} / {}", mmss(self.t), mmss(tl.dur()))];
        if self.arm {
            lines.push("Tracking: drag a box around the subject".into());
        }
        if let Some(t) = self.trk.as_ref().filter(|t| !t.keyed) {
            lines.push(
                t.t.st()
                    .line(["Tracking", "Auto framing"][t.t.auto as usize]),
            );
        }
        if let Some(a) = &self.auto {
            lines.push(a.st().line("Auto edit"));
        }
        if tl.is_empty() {
            lines.push("Drag recordings from Media onto the timeline.".into());
        }
        if self.decks.iter().any(|d| d.want.is_some()) {
            lines.push("Decoding…".into());
        }
        if self.free.is_some() && !self.pano {
            lines.push(format!(
                "View not keyed: press K for a key at {} ({})",
                mmss(self.t),
                if self.whole {
                    "timeline track"
                } else {
                    "clip track"
                }
            ));
        }
        if !self.posed() {
            lines.push("Stabilising…".into());
        }
        if let Some(p) = &self.mq.0 {
            lines.push(format!("Reading {}…", name(&p.to_string_lossy())));
        }
        if let (false, false, Some(c)) = (self.playing, self.raft_off, self.cur()) {
            lines.push(match &self.seam {
                Some(s) if s.0 == c.2 => "RAFT stitch".into(),
                _ if self.busy => "RAFT stitching…".into(),
                _ => "Template seam".into(),
            });
        }
        let mut y = rect.top() + 6.0;
        for l in lines {
            let g = p.layout(
                l,
                egui::FontId::monospace(12.0),
                egui::Color32::WHITE,
                rect.width() - 12.0,
            );
            let at = egui::pos2(rect.left() + 6.0, y);
            p.rect_filled(
                egui::Rect::from_min_size(at, g.size()).expand(3.0),
                3.0,
                egui::Color32::from_black_alpha(160),
            );
            y += g.size().y + 8.0;
            p.galley(at, g, egui::Color32::WHITE);
        }
    }

    fn timeline(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            if ui
                .button(if self.playing { "Pause" } else { "Play" })
                .on_hover_text("Space")
                .clicked()
            {
                if self.playing {
                    self.pause()
                } else {
                    self.play()
                }
            }
            if ui
                .button("Split")
                .on_hover_text("Split at the playhead (S)")
                .clicked()
            {
                self.split();
            }
            if ui
                .button("Freeze")
                .on_hover_text("Hold the frame at the playhead")
                .clicked()
            {
                self.freeze();
            }
            ui.add(
                egui::DragValue::new(&mut self.hold)
                    .range(0.1..=10.0)
                    .speed(0.05)
                    .suffix(" s"),
            );
            if ui
                .add_enabled(self.sel.is_some(), egui::Button::new("Delete"))
                .on_hover_text("Del")
                .clicked()
                && let Some(k) = self.sel
            {
                self.delete(k);
            }
            ui.separator();
            ui.label("Zoom");
            ui.add(
                egui::Slider::new(&mut self.zoom, 4.0..=400.0)
                    .logarithmic(true)
                    .show_value(false),
            );
            if ui.button("Fit").clicked() {
                let d = self.pj.tl().dur().max(1.0) as f32;
                self.zoom = ((ui.available_width().max(400.0) + 300.0) / d).clamp(4.0, 400.0);
            }
            ui.monospace(format!("{} / {}", mmss(self.t), mmss(self.pj.tl().dur())));
        });
        let mut acts = vec![];
        let h: f32 = ROWS.iter().sum::<f32>() + 12.0;
        // following the playhead jumps: an animated scroll asked for again before it ends overshoots
        egui::ScrollArea::horizontal()
            .id_salt("ed tl")
            .animated(false)
            .show(ui, |ui| {
                let tl = self.pj.tl();
                let w = (tl.dur() as f32 * self.zoom + 240.0).max(ui.available_width());
                let (rect, _) = ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::hover());
                // a playhead that moved (or a view resized: the first frames' window is not the final one) scrolls
                // into view, over frames until it is; a timeline scrolled by hand stays where it is
                let seen = (self.t, ui.clip_rect().width());
                if seen != self.seen {
                    let x = rect.left() + 8.0 + self.t as f32 * self.zoom;
                    match ui.clip_rect().x_range().contains(x) {
                        true => self.seen = seen,
                        false => ui.scroll_to_rect(
                            egui::Rect::from_min_max(
                                egui::pos2(x - 30.0, rect.top()),
                                egui::pos2(x + 30.0, rect.bottom()),
                            ),
                            None,
                        ),
                    }
                }
                self.paint(ui, rect, &mut acts);
            });
        for a in acts {
            self.act(a);
        }
    }

    /// Does what a timeline widget asked for.
    fn act(&mut self, a: Act) {
        match a {
            Act::Seek(t) => self.seek(t),
            // scrubbing plays nothing: restarting the sound for every position would rebuild its stream
            Act::Scrub(true) if self.playing => {
                self.resume = true;
                self.pause();
            }
            Act::Scrub(false) if self.resume => {
                self.resume = false;
                self.play();
            }
            Act::Scrub(_) => {}
            Act::Sel(k) => self.sel = Some(k),
            Act::Mv(a, b) => self.mv(a, b),
            Act::Trim(k, a, b) => {
                self.own(|e| e.ed(|p| p.trim(k, a, b)));
            }
            Act::Hold(k, h) => {
                self.own(|e| e.ed(|p| p.edit(k, |c| c.hold = h)));
            }
            Act::Begin => self.own(Editor::begin),
            Act::End => self.end(),
            Act::MvKey(on, i, t) => {
                if let Some(j) = self.own(|e| e.ed(|p| p.mv_key(on, i, t))) {
                    self.key = Some((on, j));
                    if let Some(Drag::Key(_, n, _)) = &mut self.drag {
                        *n = j;
                    }
                }
            }
            Act::KeySel(on, i) => {
                self.key = Some((on, i));
                if let Some(t) = self
                    .pj
                    .tl()
                    .track(on)
                    .and_then(|ks| ks.get(i))
                    .and_then(|k| self.pj.tl().from_track(on, k.t))
                {
                    self.seek(t);
                }
            }
            Act::Add(p, k) => {
                if let Err(e) = self.add(&p, Some(k)) {
                    self.err = Some(e);
                }
            }
            Act::Speed(k, v) => {
                self.own(|e| e.ed(|p| p.speed(k, v)));
            }
            Act::Vol(k, v) => {
                self.own(|e| e.ed(|p| p.edit(k, |c| c.vol = v)));
            }
        }
    }

    /// Draws the timeline into `rect` and collects what its widgets ask for.
    fn paint(&mut self, ui: &mut egui::Ui, rect: egui::Rect, acts: &mut Vec<Act>) {
        let tl = self.pj.tl();
        let (z, x0) = (self.zoom, rect.left() + 8.0);
        let xt = |t: f64| x0 + t as f32 * z;
        let tx = |x: f32| ((x - x0) / z) as f64;
        let mut y = rect.top();
        let row: Vec<egui::Rect> = ROWS
            .iter()
            .map(|&hh| {
                let r = egui::Rect::from_min_max(
                    egui::pos2(rect.left(), y),
                    egui::pos2(rect.right(), y + hh),
                );
                y += hh + 3.0;
                r
            })
            .collect();
        let p = ui.painter_at(rect);
        let vis = ui.visuals().clone();
        let (fg, weak, sel) = (
            vis.text_color(),
            vis.weak_text_color(),
            vis.selection.stroke.color,
        );
        p.rect_filled(rect, 2.0, vis.extreme_bg_color);
        // ruler: ticks at a spacing ≥ 60 px
        let step = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
            .into_iter()
            .find(|s| *s as f32 * z >= 60.0)
            .unwrap_or(120.0);
        let mut s = 0.0;
        while xt(s) < rect.right() {
            let x = xt(s);
            p.line_segment(
                [
                    egui::pos2(x, row[0].bottom() - 6.0),
                    egui::pos2(x, row[0].bottom()),
                ],
                (1.0, weak),
            );
            p.text(
                egui::pos2(x + 2.0, row[0].top()),
                egui::Align2::LEFT_TOP,
                mmss(s),
                egui::FontId::monospace(10.0),
                weak,
            );
            s += step;
        }
        let ruler = ui.interact(
            row[0],
            ui.make_persistent_id("ed ruler"),
            egui::Sense::click_and_drag(),
        );
        if ruler.drag_started() {
            acts.push(Act::Scrub(true));
        }
        if (ruler.clicked() || ruler.dragged())
            && let Some(q) = ruler.interact_pointer_pos()
        {
            acts.push(Act::Seek(tx(q.x).max(0.0)));
        }
        if ruler.drag_stopped() {
            acts.push(Act::Scrub(false));
        }
        let ptr = ui.ctx().pointer_interact_pos();
        let font = egui::FontId::proportional(11.0);
        // clips
        let vr = row[2];
        // clip centres; a clip dropped after a move: index, pointer x
        let (mut mid, mut drop) = (vec![], None);
        for (k, c) in tl.clips.iter().enumerate() {
            let n = tl.frames(k);
            let (a, b) = (tl.out.fps.t(n.start), tl.out.fps.t(n.end));
            let r = egui::Rect::from_min_max(
                egui::pos2(xt(a), vr.top()),
                egui::pos2(xt(b), vr.bottom()),
            );
            mid.push(r.center().x);
            let path = &tl.files[c.file as usize].path;
            let cp = p.with_clip_rect(r.shrink(1.0));
            cp.rect_filled(r, 3.0, vis.widgets.inactive.bg_fill);
            // thumbnails, each from the source time at its middle
            let tw = vr.height() * thumbs::W as f32 / thumbs::H as f32;
            let mut x = r.left();
            while x < r.right() {
                let s = match c.hold > 0.0 {
                    true => c.a,
                    false => tl.content(k, (tx(x + tw / 2.0) - a).clamp(0.0, b - a)),
                };
                if let Some(t) = self.thumbs.at(path, s) {
                    let tr = egui::Rect::from_min_size(
                        egui::pos2(x, r.top()),
                        egui::vec2(tw, r.height()),
                    );
                    cp.image(
                        t.id(),
                        tr,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }
                x += tw;
            }
            // a tracking run on this clip's file: its states along the clip's bottom
            if let Some(t) = self
                .trk
                .as_ref()
                .filter(|t| &t.path == path && c.hold == 0.0)
                && let Some(ts) = t.t.times.get()
            {
                let y = (r.bottom() - 5.0)..=(r.bottom() - 1.0);
                ai::strip(&cp, &t.t, y.into(), |f| {
                    let s = ts.get(f).copied()?;
                    tl.from_track(On::Clip(k), s.clamp(c.a, c.b)).map(xt)
                });
            }
            // the speed curve on a log scale (slowest bottom, fastest top) with its keys
            let (lo, hi) = (edit::MIN_SPEED.ln(), edit::MAX_SPEED.ln());
            let ys =
                |v: f64| r.bottom() - 4.0 - ((v.ln() - lo) / (hi - lo)) as f32 * (r.height() - 8.0);
            if !c.speed.is_empty() && c.hold == 0.0 {
                let pts: Vec<egui::Pos2> = (0..=((r.width() / 3.0) as usize))
                    .map(|i| {
                        let x = r.left() + i as f32 * 3.0;
                        egui::pos2(x, ys(c.rate(tl.content(k, (tx(x) - a).clamp(0.0, b - a)))))
                    })
                    .collect();
                cp.add(egui::Shape::line(
                    pts,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 190, 60)),
                ));
                for &(s, v) in &c.speed {
                    if let Some(t) = tl.from_track(On::Clip(k), s) {
                        cp.circle_filled(
                            egui::pos2(xt(t), ys(v)),
                            3.5,
                            egui::Color32::from_rgb(255, 190, 60),
                        );
                    }
                }
            }
            let mut tag = name(path).to_string();
            if c.hold > 0.0 {
                tag = format!("Freeze {:.2} s · {tag}", c.hold);
            } else if let (Some(lo), Some(hi)) = (
                c.speed.iter().map(|s| s.1).reduce(f64::min),
                c.speed.iter().map(|s| s.1).reduce(f64::max),
            ) {
                tag = match lo == hi {
                    true => format!("{lo:.2}× · {tag}"),
                    false => format!("{lo:.2}–{hi:.2}× · {tag}"),
                };
            }
            if !c.look.is_empty() {
                tag += &format!(" · {}", name(&c.look));
            }
            let g = cp.layout_no_wrap(tag, font.clone(), egui::Color32::WHITE);
            let at = r.left_top() + egui::vec2(4.0, 3.0);
            cp.rect_filled(
                egui::Rect::from_min_size(at, g.size()).expand(2.0),
                2.0,
                egui::Color32::from_black_alpha(170),
            );
            cp.galley(at, g, egui::Color32::WHITE);
            let st = if self.sel == Some(k) {
                egui::Stroke::new(2.0, sel)
            } else {
                egui::Stroke::new(1.0, weak)
            };
            p.rect_stroke(r, 3.0, st, egui::StrokeKind::Inside);
            let body = ui.interact(
                r,
                ui.make_persistent_id(("ed clip", k)),
                egui::Sense::click_and_drag(),
            );
            if body.clicked() {
                acts.push(Act::Sel(k));
                if let Some(q) = body.interact_pointer_pos() {
                    acts.push(Act::Seek(tx(q.x)));
                }
            }
            if body.drag_started() {
                self.drag = Some(Drag::Move(k));
                acts.push(Act::Sel(k));
            }
            // a drop is a release: egui also ends a drag on Escape, the button still down, which cancels the move
            if body.drag_stopped()
                && let Some(Drag::Move(from)) = self.drag.take()
                && ui.input(|i| i.pointer.any_released())
            {
                drop = body.interact_pointer_pos().map(|q| (from, q.x));
            }
            // trim handles on the edges
            for right in [false, true] {
                let hr = match right {
                    false => egui::Rect::from_min_max(
                        r.left_top(),
                        egui::pos2(r.left() + 6.0, r.bottom()),
                    ),
                    true => egui::Rect::from_min_max(
                        egui::pos2(r.right() - 6.0, r.top()),
                        r.right_bottom(),
                    ),
                };
                if !right && c.hold > 0.0 {
                    continue;
                }
                let h = ui
                    .interact(
                        hr,
                        ui.make_persistent_id(("ed trim", k, right)),
                        egui::Sense::drag(),
                    )
                    .on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
                if h.hovered() || h.dragged() {
                    p.rect_filled(hr, 2.0, sel);
                }
                if h.drag_started()
                    && let Some(q) = h.interact_pointer_pos()
                {
                    acts.push(Act::Begin);
                    self.drag = Some(Drag::Trim(k, right, c.clone(), q.x));
                }
                if h.dragged()
                    && let (Some(Drag::Trim(j, rt, c0, x1)), Some(q)) =
                        (&self.drag, h.interact_pointer_pos())
                    && *j == k
                    && *rt == right
                {
                    let (dt, fd, f) = (
                        (q.x - x1) as f64 / z as f64,
                        tl.out.fps.t(1),
                        tl.files[c0.file as usize].dur,
                    );
                    if c0.hold > 0.0 {
                        acts.push(Act::Hold(k, (c0.hold + dt).clamp(fd, edit::MAX_HOLD)));
                    } else if right {
                        let b = c0.src((c0.dur() + dt).max(fd)).min(f);
                        acts.push(Act::Trim(k, c0.a, b.max(c0.a + fd * c0.rate(c0.a))));
                    } else {
                        let a = match dt >= 0.0 {
                            true => c0.src(dt),
                            false => c0.a + dt * c0.rate(c0.a),
                        };
                        acts.push(Act::Trim(k, a.clamp(0.0, c0.b - fd * c0.rate(c0.b)), c0.b));
                    }
                }
                if h.drag_stopped() {
                    acts.push(Act::End);
                    self.drag = None;
                }
            }
            // speed keys: sideways in time (the clip's timing at the drag's start), up and down in speed
            for (i, &(s, v)) in c.speed.iter().enumerate().filter(|_| c.hold == 0.0) {
                let Some(t) = tl.from_track(On::Clip(k), s) else {
                    continue;
                };
                let kh = ui
                    .interact(
                        egui::Rect::from_center_size(
                            egui::pos2(xt(t), ys(v)),
                            egui::vec2(10.0, 10.0),
                        ),
                        ui.make_persistent_id(("ed speed", k, i)),
                        egui::Sense::drag(),
                    )
                    .on_hover_cursor(egui::CursorIcon::Grab)
                    .on_hover_ui(|ui| {
                        ui.label(format!("{v:.2}× at source {}", mmss(s)));
                    });
                // from where the button went down: the key keeps under the pointer through the drag threshold
                if kh.drag_started()
                    && let Some(q) = ui.input(|i| i.pointer.press_origin())
                {
                    acts.push(Act::Begin);
                    self.drag = Some(Drag::Speed(k, i, c.clone(), q.x));
                }
                if kh.dragged()
                    && let (Some(Drag::Speed(j, ii, c0, x1)), Some(q)) =
                        (&self.drag, kh.interact_pointer_pos())
                    && (*j, *ii) == (k, i)
                {
                    let dt = (q.x - x1) as f64 / z as f64;
                    let u = c0.out(c0.speed[i].0) + if c0.rev { -dt } else { dt };
                    let a = i.checked_sub(1).map_or(c0.a, |h| c0.speed[h].0 + edit::GAP);
                    let b = c0.speed.get(i + 1).map_or(c0.b, |n| n.0 - edit::GAP);
                    let f = ((r.bottom() - 4.0 - q.y) / (r.height() - 8.0)) as f64;
                    let mut keys = c0.speed.clone();
                    keys[i] = (
                        c0.src(u.max(0.0)).max(a).min(b),
                        (lo + f * (hi - lo))
                            .exp()
                            .clamp(edit::MIN_SPEED, edit::MAX_SPEED),
                    );
                    acts.push(Act::Speed(k, keys));
                }
                if kh.drag_stopped() {
                    acts.push(Act::End);
                    self.drag = None;
                }
            }
        }
        if let Some((from, x)) = drop {
            let to = (0..mid.len()).filter(|&j| j != from && mid[j] < x).count();
            if to != from {
                acts.push(Act::Mv(from, to));
            }
        }
        // where a dragged clip or recording would go
        let moving = matches!(self.drag, Some(Drag::Move(_)));
        let media = egui::DragAndDrop::has_payload_of_type::<PathBuf>(ui.ctx());
        if let Some(q) = ptr.filter(|q| (moving || media) && rect.contains(*q)) {
            let i = mid.iter().filter(|&&m| m < q.x).count();
            let x = match (i, tl.clips.len()) {
                (0, _) => xt(0.0),
                (i, _) => xt(tl.out.fps.t(tl.frames(i - 1).end)),
            };
            p.line_segment(
                [
                    egui::pos2(x, vr.top() - 2.0),
                    egui::pos2(x, vr.bottom() + 2.0),
                ],
                egui::Stroke::new(3.0, sel),
            );
            if media
                && ui.input(|i| i.pointer.any_released())
                && let Some(f) = egui::DragAndDrop::take_payload::<PathBuf>(ui.ctx())
            {
                acts.push(Act::Add((*f).clone(), i));
            }
        }
        // view keys of the track keys go on
        let kr = row[1];
        let tracks: Vec<On> = match self.whole {
            true => vec![On::Out],
            false => (0..tl.clips.len()).map(On::Clip).collect(),
        };
        for on in tracks {
            for (i, key) in tl.track(on).unwrap_or_default().iter().enumerate() {
                let Some(t) = tl.from_track(on, key.t) else {
                    continue;
                };
                let c = egui::pos2(xt(t), kr.center().y);
                let d = 6.0;
                let picked = self.key == Some((on, i));
                p.add(egui::Shape::convex_polygon(
                    vec![
                        c + egui::vec2(0.0, -d),
                        c + egui::vec2(d, 0.0),
                        c + egui::vec2(0.0, d),
                        c + egui::vec2(-d, 0.0),
                    ],
                    if picked {
                        sel
                    } else {
                        egui::Color32::from_rgb(90, 170, 255)
                    },
                    egui::Stroke::new(1.0, fg),
                ));
                // the dragged key keeps its widget id while passing others (it swaps with the key now at its start
                // index): egui ends a drag whose widget's id goes
                let id = match self.drag {
                    Some(Drag::Key(o, j, i0)) if o == on && i == j => i0,
                    Some(Drag::Key(o, j, i0)) if o == on && i == i0 => j,
                    _ => i,
                };
                let kh = ui.interact(
                    egui::Rect::from_center_size(c, egui::vec2(2.0 * d, 2.0 * d)),
                    ui.make_persistent_id((
                        "ed key",
                        on == On::Out,
                        if let On::Clip(k) = on { k } else { 0 },
                        id,
                    )),
                    egui::Sense::click_and_drag(),
                );
                if kh.clicked() {
                    acts.push(Act::KeySel(on, i));
                }
                if kh.drag_started() {
                    acts.push(Act::Begin);
                    self.drag = Some(Drag::Key(on, i, i));
                }
                if kh.dragged()
                    && let (Some(Drag::Key(o, j, _)), Some(q)) =
                        (&self.drag, kh.interact_pointer_pos())
                    && *o == on
                {
                    acts.push(Act::MvKey(on, *j, tx(q.x)));
                }
                if kh.drag_stopped() {
                    acts.push(Act::End);
                    self.drag = None;
                }
            }
        }
        // audio: each clip's gain envelope (0..2), muted and frozen clips flat at the bottom
        let ar = row[3];
        for (k, c) in tl.clips.iter().enumerate() {
            let n = tl.frames(k);
            let (a, b) = (tl.out.fps.t(n.start), tl.out.fps.t(n.end));
            let r = egui::Rect::from_min_max(
                egui::pos2(xt(a), ar.top()),
                egui::pos2(xt(b), ar.bottom()),
            );
            p.rect_filled(
                r.shrink2(egui::vec2(1.0, 0.0)),
                2.0,
                vis.widgets.noninteractive.bg_fill,
            );
            let gy = |g: f32| r.bottom() - 2.0 - (g / 2.0).clamp(0.0, 1.0) * (r.height() - 4.0);
            let pts: Vec<egui::Pos2> = (0..=((r.width() / 3.0) as usize))
                .map(|i| {
                    let x = (r.left() + i as f32 * 3.0).min(r.right());
                    let g = tl
                        .at(tx(x).min(b - 1e-6))
                        .filter(|e| e.k == k)
                        .map_or(0.0, |e| e.gain);
                    egui::pos2(x, gy(g))
                })
                .collect();
            let col = if c.mute || c.hold > 0.0 {
                weak
            } else {
                egui::Color32::from_rgb(110, 210, 120)
            };
            p.add(egui::Shape::line(pts, egui::Stroke::new(1.5, col)));
            // volume points: sideways in content time, up and down in gain (0..2)
            for (i, &(x, g)) in c.vol.iter().enumerate().filter(|_| c.hold == 0.0) {
                let Some(t) = tl.from_track(On::Clip(k), x) else {
                    continue;
                };
                let at = egui::pos2(xt(t), gy(g));
                p.circle_filled(at, 3.0, col);
                let vh = ui
                    .interact(
                        egui::Rect::from_center_size(at, egui::vec2(10.0, 10.0)),
                        ui.make_persistent_id(("ed vol", k, i)),
                        egui::Sense::drag(),
                    )
                    .on_hover_cursor(egui::CursorIcon::Grab)
                    .on_hover_ui(|ui| {
                        ui.label(format!("gain {g:.2} at source {}", mmss(x)));
                    });
                if vh.drag_started() {
                    acts.push(Act::Begin);
                    self.drag = Some(Drag::Vol(k, i));
                }
                if vh.dragged()
                    && let (Some(Drag::Vol(j, ii)), Some(q)) =
                        (&self.drag, vh.interact_pointer_pos())
                    && (*j, *ii) == (k, i)
                {
                    let lo = i.checked_sub(1).map_or(c.a, |h| c.vol[h].0 + edit::GAP);
                    let hi = c.vol.get(i + 1).map_or(c.b, |n| n.0 - edit::GAP);
                    let s = (tl.to_track(On::Clip(k), tx(q.x).clamp(a, b))).unwrap_or(x);
                    let mut v = c.vol.clone();
                    v[i] = (
                        s.max(lo).min(hi),
                        (2.0 * (r.bottom() - 2.0 - q.y) / (r.height() - 4.0)).clamp(0.0, 2.0),
                    );
                    acts.push(Act::Vol(k, v));
                }
                if vh.drag_stopped() {
                    acts.push(Act::End);
                    self.drag = None;
                }
            }
            if c.mute {
                p.text(
                    r.left_top() + egui::vec2(3.0, 1.0),
                    egui::Align2::LEFT_TOP,
                    "muted",
                    font.clone(),
                    weak,
                );
            }
        }
        // music
        let mr = row[4];
        if let Some(m) = &tl.out.music {
            let f = &tl.files[m.file as usize];
            let (a, b) = ((-m.offset).max(0.0), (f.dur - m.offset).min(tl.dur()));
            if b > a {
                let r = egui::Rect::from_min_max(
                    egui::pos2(xt(a), mr.top()),
                    egui::pos2(xt(b), mr.bottom()),
                );
                p.rect_filled(r, 3.0, egui::Color32::from_rgb(120, 80, 170));
                p.text(
                    r.left_center() + egui::vec2(4.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    format!(
                        "Music: {} · gain {:.2}{}",
                        name(&f.path),
                        m.gain,
                        if m.duck { " · ducked" } else { "" }
                    ),
                    font.clone(),
                    egui::Color32::WHITE,
                );
            }
        } else {
            p.text(
                mr.left_center() + egui::vec2(8.0, 0.0),
                egui::Align2::LEFT_CENTER,
                "No music (Audio tab: Import…)",
                font.clone(),
                weak,
            );
        }
        // the playhead
        let x = xt(self.t);
        p.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(2.0, egui::Color32::from_rgb(240, 70, 60)),
        );
    }

    /// One undo step per drag of the widget whose response is `r`; `f` edits when it changed.
    fn step(&mut self, r: &egui::Response, f: impl FnOnce(&mut Project) -> edit::Result<()>) {
        self.slide(r, r.changed(), |e| {
            let x = f(&mut e.pj);
            e.did(x);
        });
    }

    /// A drag of the slider (or sliders) whose response is `r` as one undo step and a drag of the editor's: `f` edits
    /// when `ch`, as the drag's own work. A drag that another edit ended (`halt`: Delete or S mid-drag) edits no more,
    /// also not in the frame egui reports its end: the clip it names by index may be another's now.
    fn slide(&mut self, r: &egui::Response, ch: bool, f: impl FnOnce(&mut Self)) {
        if r.drag_started() {
            self.begin();
            self.drag = Some(Drag::Slider(r.id));
        }
        let mine = matches!(self.drag, Some(Drag::Slider(id)) if id == r.id);
        if ch && (mine || !(r.dragged() || r.drag_stopped())) {
            self.own(f);
        }
        if mine && r.drag_stopped() {
            self.drag = None;
            self.end();
        }
    }

    fn clip_ui(&mut self, ui: &mut egui::Ui) {
        let tl = self.pj.tl();
        let Some(k) = self.sel.filter(|&k| k < tl.clips.len()) else {
            ui.label("Select a clip on the timeline.");
            return;
        };
        let (mut c, f, n) = (
            tl.clips[k].clone(),
            tl.files[tl.clips[k].file as usize].clone(),
            tl.frames(k),
        );
        let (fd, total) = (tl.out.fps.t(1), tl.clips.len());
        ui.heading(format!("Clip {} of {total}", k + 1));
        ui.label(name(&f.path)).on_hover_text(&f.path);
        ui.label(format!(
            "{} frames · {} on the timeline from {}",
            n.end - n.start,
            mmss(c.dur()),
            mmss(tl.out.fps.t(n.start))
        ));
        ui.horizontal_wrapped(|ui| {
            if ui.button("Split").clicked() {
                self.split();
            }
            if ui.button("Duplicate").clicked() {
                self.ed(|p| p.dup(k));
            }
            if ui.add_enabled(k > 0, egui::Button::new("◀ Move")).clicked() {
                self.mv(k, k - 1);
            }
            if ui
                .add_enabled(k + 1 < total, egui::Button::new("Move ▶"))
                .clicked()
            {
                self.mv(k, k + 1);
            }
            if ui.button("Delete").clicked() {
                self.delete(k);
            }
        });
        if self.pj.tl().clips.get(k) != Some(&c) {
            return;
        }
        ui.separator();
        if c.hold > 0.0 {
            ui.label(format!("Freeze frame of source {}", mmss(c.a)));
            let r = ui.add(egui::Slider::new(&mut c.hold, fd..=10.0).text("Hold (s)"));
            let h = c.hold;
            self.step(&r, |pj| pj.edit(k, |c| c.hold = h));
            return;
        }
        ui.label("Source in and out");
        let (mut a, mut b) = (c.a, c.b);
        let ra = ui.add(egui::Slider::new(&mut a, 0.0..=f.dur).text("In (s)"));
        let rb = ui.add(egui::Slider::new(&mut b, 0.0..=f.dur).text("Out (s)"));
        let (a, b) = (a.min(c.b - fd), b.max(c.a + fd));
        self.step(&ra, |pj| pj.trim(k, a, c.b));
        self.step(&rb, |pj| pj.trim(k, c.a, b));
        ui.separator();
        ui.label("Speed");
        let ramp = c.speed.len() > 1;
        if !ramp {
            let mut s = c.speed.first().map_or(1.0, |s| s.1);
            let r = ui.add(
                egui::Slider::new(&mut s, edit::MIN_SPEED..=edit::MAX_SPEED)
                    .logarithmic(true)
                    .text("×"),
            );
            let a = c.a;
            self.step(&r, |pj| {
                pj.speed(k, if s == 1.0 { vec![] } else { vec![(a, s)] })
            });
        }
        let mut keys = self.pj.tl().clips[k].speed.clone();
        let mut ch = None;
        egui::Grid::new("ed speed keys")
            .num_columns(3)
            .show(ui, |ui| {
                for i in 0..keys.len() {
                    ui.label(format!("{:.2} s", keys[i].0 - c.a));
                    let r = ui.add(
                        egui::Slider::new(&mut keys[i].1, edit::MIN_SPEED..=edit::MAX_SPEED)
                            .logarithmic(true)
                            .suffix("×"),
                    );
                    if r.changed() || r.drag_started() || r.drag_stopped() {
                        ch = Some(r);
                    }
                    if ui
                        .small_button("Delete")
                        .on_hover_text("Delete this speed key")
                        .clicked()
                    {
                        keys.remove(i);
                        ch = None;
                        let v = keys.clone();
                        self.ed(|p| p.speed(k, v));
                        break;
                    }
                    ui.end_row();
                }
            });
        if let Some(r) = ch {
            let v = keys.clone();
            self.step(&r, |pj| pj.speed(k, v));
        }
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Speed key at playhead")
                .on_hover_text("Ramp keys sit on source time; speed is linear between them")
                .clicked()
            {
                let tl = self.pj.tl();
                match tl
                    .to_track(On::Clip(k), self.t)
                    .filter(|x| *x > c.a && *x < c.b)
                {
                    Some(x) => {
                        let mut v = tl.clips[k].speed.clone();
                        let s = tl.clips[k].rate(x);
                        if v.is_empty() {
                            v.push((c.a, 1.0));
                        }
                        let i = v.partition_point(|p| p.0 < x);
                        v.insert(i, (x, s));
                        v.dedup_by(|q, p| q.0 - p.0 < edit::GAP);
                        self.ed(|p| p.speed(k, v));
                    }
                    None => self.err = Some("The playhead is not inside this clip.".into()),
                }
            }
            if ui.button("Ramp 1×–2×–1×").clicked() {
                let (a, b) = (c.a, c.b);
                let m = 0.25 * (b - a);
                self.ed(|p| p.speed(k, vec![(a, 1.0), (a + m, 2.0), (b - m, 2.0), (b, 1.0)]));
            }
            if ui
                .button("1×")
                .on_hover_text("Remove the speed keys")
                .clicked()
            {
                self.ed(|p| p.speed(k, vec![]));
            }
        });
        ui.separator();
        let mut rev = c.rev;
        if ui
            .checkbox(&mut rev, "Reverse")
            .on_hover_text("Plays the clip backwards (the export does not do reversed clips yet)")
            .changed()
        {
            self.ed(|p| p.edit(k, |c| c.rev = rev));
        }
        ui.separator();
        ui.label("Effects (replaces this clip)");
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.lapse)
                    .range(1.0..=16.0)
                    .speed(0.1)
                    .prefix("speed ")
                    .suffix("×"),
            );
            ui.add(
                egui::DragValue::new(&mut self.turns)
                    .range(-8.0..=8.0)
                    .speed(0.05)
                    .prefix("turns "),
            );
            ui.add(
                egui::DragValue::new(&mut self.every)
                    .range(0.05..=5.0)
                    .speed(0.01)
                    .prefix("step ")
                    .suffix(" s"),
            );
        });
        let v = to_edit(&self.shown());
        let (a, b, file) = (c.a, c.b, c.file);
        let ramp = (0.15 * (b - a)).min(1.0);
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Orbit Lapse")
                .on_hover_text("Hyperlapse orbiting from a tiny planet")
                .clicked()
            {
                let r = shotlab::cut::orbit(file, a, b, self.lapse, ramp, v, self.turns, 1.0);
                self.shot(k, r.map(|c| vec![c]));
            }
            if ui
                .button("Roll Lapse")
                .on_hover_text("Hyperlapse rolling the world over")
                .clicked()
            {
                let r = shotlab::cut::roll(file, a, b, self.lapse, ramp, v, self.turns);
                self.shot(k, r.map(|c| vec![c]));
            }
            if ui
                .button("Stop Motion")
                .on_hover_text("Freeze frames a step of source apart")
                .clicked()
            {
                let hold = (self.every / self.lapse).max(fd);
                let r = shotlab::cut::stop_motion(file, a, b, self.every, hold);
                self.shot(k, r);
            }
        });
    }

    /// Replaces clip `k` with `r`'s clips as one undo step.
    fn shot(&mut self, k: usize, r: edit::Result<Vec<Clip>>) {
        // checked before the first edit, so none fails halfway: edit's undo could take such a step back only by
        // leaving it to redo
        let tl = self.pj.tl();
        let r = r.and_then(|v| {
            tl.clips
                .get(k)
                .ok_or_else(|| edit::Error(format!("no clip {k}")))?;
            v.iter().try_for_each(|c| tl.ok(c))?;
            Ok(v)
        });
        let Some(v) = self.did(r) else { return };
        self.begin();
        let mut r = self.pj.delete(k);
        for (i, c) in v.into_iter().enumerate() {
            if r.is_ok() {
                r = self.pj.insert(k + i, c);
            }
        }
        self.end();
        self.did(r);
    }

    fn view_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Reframe");
        ui.horizontal(|ui| {
            ui.label("Keys on");
            ui.selectable_value(&mut self.whole, false, "the clip").on_hover_text("Keys on the clip's own track move with its picture through trims and speed changes");
            ui.selectable_value(&mut self.whole, true, "the timeline").on_hover_text("Keys on output time, for clips without keys of their own");
        });
        let mut v = self.shown();
        let v0 = v;
        ui.horizontal_wrapped(|ui| {
            for (l, n) in Look::ALL {
                if ui.selectable_label(look_of(&v) == l, n).clicked() {
                    v = View {
                        yaw: v.yaw,
                        ..l.view()
                    };
                }
            }
        });
        egui::Grid::new("ed view").num_columns(2).show(ui, |ui| {
            for (x, n, lo, hi) in [
                (&mut v.yaw, "Yaw", -180.0, 360.0),
                (&mut v.pitch, "Pitch", -90.0, 90.0),
                (&mut v.roll, "Roll", -180.0, 180.0),
            ] {
                ui.label(n);
                ui.add(
                    egui::DragValue::new(x)
                        .range(lo..=hi)
                        .speed(0.5)
                        .suffix("°"),
                );
                ui.end_row();
            }
            let (a, b) = look_of(&v).fov();
            let (a, b) = (a.min(v.fov), b.max(v.fov));
            ui.label("Field of view");
            ui.add(egui::Slider::new(&mut v.fov, a..=b).suffix("°"));
            ui.end_row();
            if v.mode == Mode::Planet {
                ui.label("Planet distance");
                ui.add(egui::Slider::new(&mut v.planet, 0.0..=1.0))
                    .on_hover_text("1 stereographic (tiny planet), 0 rectilinear");
                ui.end_row();
            }
            if look_of(&v) == Look::Panini {
                ui.label("Panini");
                ui.add(egui::Slider::new(&mut v.panini, 0.01..=1.0));
                ui.end_row();
            }
        });
        if v.yaw != v0.yaw
            || v.pitch != v0.pitch
            || v.roll != v0.roll
            || v.fov != v0.fov
            || v.mode != v0.mode
            || v.planet != v0.planet
            || v.panini != v0.panini
        {
            (self.free, self.dirty) = (Some(v), true);
        }
        ui.horizontal_wrapped(|ui| {
            egui::ComboBox::from_id_salt("ed ease")
                .selected_text(EASES.iter().find(|e| e.0 == self.ease).map_or("", |e| e.1))
                .show_ui(ui, |ui| {
                    for (e, n) in EASES {
                        ui.selectable_value(&mut self.ease, e, n);
                    }
                });
            if ui
                .button("Key at playhead")
                .on_hover_text("K: the viewer's view as a key")
                .clicked()
            {
                self.key_here();
            }
        });
        let Some(on) = self.on() else {
            ui.label("Move the playhead onto a clip.");
            return;
        };
        let tl = self.pj.tl();
        let keys = tl.track(on).unwrap_or_default().to_vec();
        ui.label(match on {
            On::Out => format!("{} timeline keys", keys.len()),
            On::Clip(k) => format!("{} keys on clip {}", keys.len(), k + 1),
        });
        let mut act = None;
        egui::Grid::new("ed keys")
            .num_columns(4)
            .striped(true)
            .show(ui, |ui| {
                for (i, key) in keys.iter().enumerate() {
                    let t = tl.from_track(on, key.t);
                    let [y, p, _] = key.v.ypr();
                    let lab = format!(
                        "{} · {y:.0}°/{p:.0}° · {:.0}°",
                        t.map_or("–".into(), mmss),
                        key.v.fov
                    );
                    if ui
                        .selectable_label(self.key == Some((on, i)), lab)
                        .clicked()
                    {
                        act = Some((i, None, true));
                    }
                    let mut e = key.e;
                    egui::ComboBox::from_id_salt(("ed key ease", i))
                        .width(90.0)
                        .selected_text(EASES.iter().find(|x| x.0 == e).map_or("", |x| x.1))
                        .show_ui(ui, |ui| {
                            for (x, n) in EASES {
                                ui.selectable_value(&mut e, x, n);
                            }
                        });
                    if e != key.e {
                        act = Some((i, Some(e), false));
                    }
                    if ui
                        .small_button("Delete")
                        .on_hover_text("Delete this key")
                        .clicked()
                    {
                        act = Some((i, None, false));
                    }
                    ui.end_row();
                }
            });
        match act {
            Some((i, None, true)) => {
                self.key = Some((on, i));
                if let Some(t) = self.pj.tl().from_track(on, keys[i].t) {
                    self.seek(t);
                    self.free = None;
                }
            }
            Some((i, Some(e), _)) => {
                let r = match on {
                    On::Out => self.pj.edit_out(|o| o.view[i].e = e),
                    On::Clip(k) => self.pj.edit(k, |c| c.view[i].e = e),
                };
                self.did(r);
            }
            Some((i, None, false)) => {
                self.ed(|p| p.unkey(on, i));
                self.key = None;
            }
            None => {}
        }
        ui.separator();
        ui.label("Tracking and auto framing");
        ui.horizontal_wrapped(|ui| {
            match self.trk.as_ref().map(|t| t.t.st()) {
                Some(st) if !st.over() => {
                    let auto = self.trk.as_ref().is_some_and(|t| t.t.auto);
                    ui.label(st.line(if auto { "Auto framing" } else { "Tracking" }));
                    if ui.button("Cancel").clicked() {
                        (self.trk.as_ref().unwrap().t.stop).store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                st => {
                    let ok = !self.pano && look_of(&self.shown()) == Look::Perspective;
                    ui.add_enabled_ui(ok, |ui| {
                        ui.toggle_value(&mut self.arm, "Track a subject")
                            .on_hover_text(
                                "Drag a box around a subject in the viewer: it is followed from the playhead to the \
                                 clip's end, and the clip gets keys that keep it in view",
                            )
                            .on_disabled_hover_text("Tracking needs the perspective view (not the 360 frame)");
                    });
                    if ui
                        .button("Auto framing")
                        .on_hover_text(
                            "The clip at the playhead gets keys over all of it that follow its main subject, else the \
                             direction of travel (its whole recording is analysed first)",
                        )
                        .clicked()
                        && let Err(e) = self.frame()
                    {
                        self.err = Some(e);
                    }
                    if let Some(st) = st {
                        ui.label(st.line("Last track"));
                    }
                }
            }
        });
        ui.separator();
        ui.label("Camera movements from the viewer's view, at the playhead");
        ui.add(
            egui::Slider::new(&mut self.pdur, 0.2..=20.0)
                .logarithmic(true)
                .text("Length (s)"),
        );
        let (v, d) = (to_edit(&self.shown()), self.pdur);
        let right = edit::View {
            q: (glam::Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2) * v.q).normalize(),
            ..v
        };
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Orbit")
                .on_hover_text("Turns about the vertical (turns: Clip tab)")
                .clicked()
            {
                self.preset(edit::view::preset::orbit(v, d, self.turns));
            }
            if ui
                .button("Whip pan")
                .on_hover_text("A fast pan 90° to the right")
                .clicked()
            {
                self.preset(edit::view::preset::whip(v, right, d, 20.0));
            }
            if ui.button("Zoom in").clicked() {
                self.preset(edit::view::preset::zoom(v, (v.fov * 0.5).max(10.0), d));
            }
            if ui.button("Push in").clicked() {
                self.preset(edit::view::preset::push(v, d, 0.7));
            }
            if ui.button("Planet unfold").clicked() {
                self.preset(edit::view::preset::planet(v, d, false));
            }
            if ui.button("Planet fold").clicked() {
                self.preset(edit::view::preset::planet(v, d, true));
            }
        });
        ui.separator();
        ui.label("Stabilisation (whole timeline)");
        let (mut lock, mut smooth) = (self.x.stab, self.x.smooth);
        ui.horizontal_wrapped(|ui| {
            egui::ComboBox::from_id_salt("ed stab")
                .selected_text(gyro::MODES.iter().find(|m| m.0 == lock).map_or("", |m| m.1))
                .show_ui(ui, |ui| {
                    for (m, n) in gyro::MODES {
                        ui.selectable_value(&mut lock, m, n);
                    }
                });
            ui.add_enabled(
                lock.is_some(),
                egui::Slider::new(&mut smooth, 0.0..=gyro::SMOOTH_MAX)
                    .step_by(0.05)
                    .suffix(" s")
                    .text("Smoothing"),
            );
        });
        self.stab(lock, smooth);
    }

    /// An empty timeline takes stabilisation `st` with the clips that come with it: no undo step of its own.
    fn adopt(&mut self, st: (gyro::Mode, f64)) {
        self.stab(st.0, st.1);
        (self.xs.stab, self.xs.smooth) = st;
    }

    /// Sets the stabilisation (whole timeline) and asks for its poses.
    pub fn stab(&mut self, lock: gyro::Mode, smooth: f64) {
        if (lock, smooth) != (self.x.stab, self.x.smooth) {
            (self.x.stab, self.x.smooth, self.dirty) = (lock, smooth, true);
            for p in self.clip_files() {
                self.info(&p);
            }
        }
    }

    fn colour_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Colour");
        let tl = self.pj.tl();
        match self.sel.filter(|&k| k < tl.clips.len()) {
            Some(k) => {
                let c = tl.clips[k].clone();
                ui.label(format!("Look of clip {}", k + 1));
                let mut id = c.look.clone();
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("ed look")
                        .selected_text(if id.is_empty() { "None" } else { name(&id) })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut id, String::new(), "None");
                            for l in fx::Look::ALL {
                                ui.selectable_value(&mut id, l.name().to_string(), l.name());
                            }
                            // the .cube files imported
                            let mut own: Vec<&String> = (self.vw.cubes.keys())
                                .filter(|c| fx::Look::ALL.iter().all(|l| l.name() != *c))
                                .collect();
                            own.sort();
                            for c in own {
                                ui.selectable_value(&mut id, c.clone(), name(c))
                                    .on_hover_text(c);
                            }
                        });
                    if ui
                        .button("Import .cube…")
                        .on_hover_text("A 1D or 3D LUT (Adobe / Resolve .cube) as this clip's look")
                        .clicked()
                    {
                        self.dialog(Dlg::Cube);
                    }
                });
                if id != c.look {
                    self.ed(|p| p.edit(k, |c| c.look = id.clone()));
                }
                let mut amt = c.amt;
                let r = ui.add_enabled(
                    !c.look.is_empty(),
                    egui::Slider::new(&mut amt, 0.0..=1.0).text("Intensity"),
                );
                self.step(&r, |pj| pj.edit(k, |c| c.amt = amt));
                if ui.button("Look on every clip").clicked() {
                    self.begin();
                    for j in 0..self.pj.tl().clips.len() {
                        let id = c.look.clone();
                        let _ = self.pj.edit(j, |x| (x.look, x.amt) = (id, c.amt));
                    }
                    self.end();
                    self.changed();
                }
                ui.separator();
                ui.label(format!("Adjustments of clip {}", k + 1));
                let mut a = c.adj;
                let r = (doc::ADJ.iter().zip(doc::adj(&mut a)))
                    .map(|((_, n, lo, hi), v)| ui.add(egui::Slider::new(v, *lo..=*hi).text(*n)))
                    .reduce(|p, q| p | q)
                    .expect("adjustments");
                if ui.button("Reset").clicked() {
                    a = fx::Params::default();
                }
                // the picture changes, the clips stay: no `changed` (playback goes on)
                self.slide(&r, a != c.adj, |e| {
                    let x = e.pj.edit(k, |c| c.adj = a);
                    e.err = x.err().map(|e| e.0);
                    e.dirty = true;
                });
                if ui.button("Adjustments on every clip").clicked() {
                    self.begin();
                    for j in 0..self.pj.tl().clips.len() {
                        let _ = self.pj.edit(j, |x| x.adj = a);
                    }
                    self.end();
                    self.dirty = true;
                }
            }
            None => {
                ui.label("Select a clip for its look and adjustments.");
            }
        }
    }

    fn audio_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Audio");
        let tl = self.pj.tl();
        if let Some(k) = self
            .sel
            .filter(|&k| k < tl.clips.len() && tl.clips[k].hold == 0.0)
        {
            let c = tl.clips[k].clone();
            ui.label(format!("Clip {}", k + 1));
            let mut mute = c.mute;
            if ui.checkbox(&mut mute, "Mute").changed() {
                self.ed(|p| p.edit(k, |c| c.mute = mute));
            }
            if c.vol.len() <= 1 {
                let mut g = c.vol.first().map_or(1.0, |v| v.1);
                let r = ui.add(egui::Slider::new(&mut g, 0.0..=4.0).text("Volume"));
                let a = c.a;
                self.step(&r, |pj| {
                    pj.edit(k, |c| c.vol = if g == 1.0 { vec![] } else { vec![(a, g)] })
                });
            }
            let mut pts = c.vol.clone();
            let mut ch = None;
            egui::Grid::new("ed vol").num_columns(3).show(ui, |ui| {
                for i in 0..pts.len().min(64) {
                    ui.label(format!("{:.2} s", pts[i].0 - c.a));
                    let r = ui.add(egui::Slider::new(&mut pts[i].1, 0.0..=4.0));
                    if r.changed() || r.drag_started() || r.drag_stopped() {
                        ch = Some(r);
                    }
                    if ui.small_button("Delete").clicked() {
                        pts.remove(i);
                        let v = pts.clone();
                        self.ed(|p| p.edit(k, |c| c.vol = v));
                        ch = None;
                        break;
                    }
                    ui.end_row();
                }
            });
            if let Some(r) = ch {
                let v = pts.clone();
                self.step(&r, |pj| pj.edit(k, |c| c.vol = v));
            }
            ui.horizontal_wrapped(|ui| {
                if ui.button("Point at playhead").clicked() {
                    let tl = self.pj.tl();
                    match tl
                        .to_track(On::Clip(k), self.t)
                        .filter(|x| *x >= c.a && *x <= c.b)
                    {
                        Some(x) => {
                            let g = tl.at(self.t).map_or(1.0, |e| e.gain.max(0.0));
                            let mut v = c.vol.clone();
                            v.retain(|p| (p.0 - x).abs() >= edit::GAP);
                            let i = v.partition_point(|p| p.0 < x);
                            v.insert(i, (x, if c.mute { 1.0 } else { g }));
                            self.ed(|p| p.edit(k, |c| c.vol = v));
                        }
                        None => self.err = Some("The playhead is not inside this clip.".into()),
                    }
                }
                let fade = (0.5f64).min(0.5 * (c.b - c.a));
                if ui.button("Fade in").clicked() {
                    let mut v = c.vol.clone();
                    v.retain(|p| p.0 > c.a + fade);
                    let g = c.vol.first().map_or(1.0, |p| p.1);
                    v.splice(0..0, [(c.a, 0.0), (c.a + fade, g)]);
                    self.ed(|p| p.edit(k, |c| c.vol = v));
                }
                if ui.button("Fade out").clicked() {
                    let mut v = c.vol.clone();
                    v.retain(|p| p.0 < c.b - fade);
                    let g = c.vol.last().map_or(1.0, |p| p.1);
                    v.extend([(c.b - fade, g), (c.b, 0.0)]);
                    self.ed(|p| p.edit(k, |c| c.vol = v));
                }
                if ui.button("Flat").clicked() {
                    self.ed(|p| p.edit(k, |c| c.vol.clear()));
                }
            });
        } else {
            ui.label("Select a clip for its volume (freeze frames are silent).");
        }
        ui.separator();
        ui.label("Music");
        let m = self.pj.tl().out.music.clone();
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Import…")
                .on_hover_text("MP3, FLAC, WAV, Ogg Vorbis, M4A")
                .clicked()
            {
                self.dialog(Dlg::Music);
            }
            if m.is_some() && ui.button("Remove").clicked() {
                let r = self.pj.edit_out(|o| o.music = None);
                self.did(r);
            }
        });
        if let Some(mut m) = m {
            let f = self.pj.tl().files[m.file as usize].clone();
            ui.label(format!("{} · {}", name(&f.path), mmss(f.dur)));
            let r = ui.add(egui::Slider::new(&mut m.gain, 0.0..=2.0).text("Gain"));
            let g = m.gain;
            self.step(&r, |pj| pj.edit_out(|o| o.music.as_mut().unwrap().gain = g));
            let r = ui.add(
                egui::DragValue::new(&mut m.offset)
                    .range(-600.0..=f.dur)
                    .speed(0.05)
                    .prefix("starts at music ")
                    .suffix(" s"),
            );
            let o = m.offset;
            self.step(&r, |pj| {
                pj.edit_out(|x| x.music.as_mut().unwrap().offset = o)
            });
            if ui.checkbox(&mut m.duck, "Duck under clip audio").changed() {
                let d = m.duck;
                let r = self.pj.edit_out(|o| o.music.as_mut().unwrap().duck = d);
                self.did(r);
            }
        }
        ui.separator();
        let mut on = self.x.denoise.is_some();
        let mut s = self.x.denoise.unwrap_or(1.0);
        ui.horizontal(|ui| {
            ui.checkbox(&mut on, "Denoise")
                .on_hover_text("RNNoise on the clips' audio (preview and export)");
            ui.add_enabled(on, egui::Slider::new(&mut s, 0.0..=1.0).text("Strength"));
        });
        if on.then_some(s) != self.x.denoise {
            self.x.denoise = on.then_some(s);
            if self.playing {
                self.pause();
                self.play();
            }
        }
        ui.separator();
        if ui
            .add(egui::Slider::new(&mut self.vol, 0.0..=1.0).text("Preview volume"))
            .changed()
            && let Some(s) = &self.snd
        {
            s.vol(self.vol);
        }
        if let Some(s) = &self.snd
            && !s.live()
        {
            ui.label("No audio output: the preview runs silent.");
        }
    }

    fn over_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Overlays");
        ui.label(
            "Drawn on every frame after the colour; sphere placements belong to a 360 export.",
        );
        ui.horizontal_wrapped(|ui| {
            if ui.button("Text").clicked() {
                self.x.over.push(Over {
                    pic: Pic::Text {
                        s: "x5desk".into(),
                        px: 64.0,
                        rgba: [255; 4],
                    },
                    place: fx::Place::Flat {
                        at: [0.5, 0.88],
                        w: 0.3,
                        rot: 0.0,
                    },
                    opacity: 1.0,
                });
            }
            if ui
                .button("Sticker / watermark…")
                .on_hover_text("A PNG")
                .clicked()
            {
                self.dialog(Dlg::Png);
            }
            if ui
                .button("Nadir patch")
                .on_hover_text("A disc over the tripod at the bottom of the sphere")
                .clicked()
            {
                self.x.over.push(Over {
                    pic: Pic::Disc([24, 24, 24, 255]),
                    place: fx::Place::Sphere {
                        yaw: 0.0,
                        pitch: -90.0,
                        fov: 40.0,
                        roll: 0.0,
                    },
                    opacity: 1.0,
                });
            }
            if ui
                .button("Nadir logo…")
                .on_hover_text("A PNG over the tripod")
                .clicked()
            {
                self.dialog(Dlg::Nadir);
            }
        });
        let o0 = self.x.over.clone();
        let mut del = None;
        for (i, o) in self.x.over.iter_mut().enumerate() {
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(match &o.pic {
                    Pic::Text { .. } => "Text".to_string(),
                    Pic::Png(p) => name(p).to_string(),
                    Pic::Disc(_) => "Nadir disc".to_string(),
                });
                if ui.small_button("Delete").clicked() {
                    del = Some(i);
                }
            });
            match &mut o.pic {
                Pic::Text { s, px, rgba } => {
                    ui.text_edit_singleline(s);
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(px).range(8.0..=400.0).suffix(" px"));
                        ui.color_edit_button_srgba_unmultiplied(rgba);
                    });
                }
                Pic::Disc(c) => {
                    ui.color_edit_button_srgba_unmultiplied(c);
                }
                Pic::Png(_) => {}
            }
            ui.horizontal(|ui| {
                let flat = matches!(o.place, fx::Place::Flat { .. });
                if ui.selectable_label(flat, "Flat").clicked() && !flat {
                    o.place = fx::Place::Flat {
                        at: [0.5, 0.5],
                        w: 0.3,
                        rot: 0.0,
                    };
                }
                if ui.selectable_label(!flat, "Sphere").clicked() && flat {
                    o.place = fx::Place::Sphere {
                        yaw: 0.0,
                        pitch: 0.0,
                        fov: 30.0,
                        roll: 0.0,
                    };
                }
            });
            match &mut o.place {
                fx::Place::Flat { at, w, rot } => {
                    ui.add(egui::Slider::new(&mut at[0], 0.0..=1.0).text("x"));
                    ui.add(egui::Slider::new(&mut at[1], 0.0..=1.0).text("y"));
                    ui.add(egui::Slider::new(w, 0.01..=1.0).text("Width"));
                    ui.add(egui::Slider::new(rot, -180.0..=180.0).text("Rotation"));
                }
                fx::Place::Sphere {
                    yaw,
                    pitch,
                    fov,
                    roll,
                } => {
                    ui.add(egui::Slider::new(yaw, -180.0..=180.0).text("Yaw"));
                    ui.add(egui::Slider::new(pitch, -90.0..=90.0).text("Pitch"));
                    ui.add(egui::Slider::new(fov, 1.0..=170.0).text("Size (°)"));
                    ui.add(egui::Slider::new(roll, -180.0..=180.0).text("Roll"));
                }
            }
            ui.add(egui::Slider::new(&mut o.opacity, 0.0..=1.0).text("Opacity"));
        }
        if let Some(i) = del {
            self.x.over.remove(i);
        }
        if self.x.over != o0 {
            self.dirty = true;
        }
    }
}
