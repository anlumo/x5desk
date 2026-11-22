//! Settings screen and the app's preferences, `key=value` lines in `<config>/x5desk/settings`: the default export
//! preset, the audio output, the thumbnail cache's limit, and this installation's client id (the camera remembers it
//! once its user allows the connection, over Bluetooth and Wi-Fi). The download folder and auto-connect stay in the
//! camera screen's own file (`camera`); this page edits them through it. The UI is plain English, dates read
//! YYYY-MM-DD, sizes in decimal units.

use crate::{album, camera::Cam, export};
use cpal::traits::{DeviceTrait, HostTrait};
use eframe::egui;
use std::{
    fs::File,
    io::Read,
    path::PathBuf,
    sync::{
        Condvar, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    thread,
};
use video::Gpu;

#[derive(Clone, Debug, PartialEq)]
pub struct Prefs {
    /// export preset key (`export::PRESETS`)
    pub preset: Option<String>,
    /// cpal output device id; None: the system default
    pub audio: Option<String>,
    /// thumbnail cache limit, bytes
    pub cache: u64,
    pub id: String,
    file: Option<PathBuf>,
}

/// The output the player, the live outputs and the editor's preview play on ([`output`]): [`Prefs::audio`], set by
/// the Settings screen.
static AUDIO: Out = Out::new(media::audio::device);

/// An output device looked up off the UI thread (an id lists ALSA's devices): the newest pick looked up so far (its
/// id, the device with that id when there was one), the picks made, the newest one looked up; the lookup
/// (`media::audio::device`).
struct Out {
    m: Mutex<(Option<String>, Option<cpal::Device>, u32, u32)>,
    c: Condvar,
    look: fn(Option<&str>) -> Option<cpal::Device>,
}

/// A pick's lookup result, handed over when it drops: also when the lookup panics, which left every later `get` (each
/// file opened) waiting.
struct Done(&'static Out, u32, Option<String>, Option<cpal::Device>);

impl Drop for Done {
    fn drop(&mut self) {
        let mut m = self.0.m.lock().unwrap_or_else(|e| e.into_inner());
        // an earlier pick whose lookup ends later changes nothing
        if self.1 > m.3 {
            (m.0, m.1, m.3) = (self.2.take(), self.3.take(), self.1);
        }
        self.0.c.notify_all();
    }
}

/// Whether `d` is the device with this id (as [`Prefs::audio`] keeps it).
fn is(d: &cpal::Device, id: &str) -> bool {
    d.id().is_ok_and(|x| x.to_string() == id)
}

impl Out {
    const fn new(look: fn(Option<&str>) -> Option<cpal::Device>) -> Out {
        Out {
            m: Mutex::new((None, None, 0, 0)),
            c: Condvar::new(),
            look,
        }
    }

    /// Looks up the device with this id (as [`Prefs::audio`] keeps it) on a thread of its own. Unknown or None: the
    /// system default, as it is at each `get`.
    fn pick(&'static self, id: Option<String>) {
        let n = {
            let mut m = self.m.lock().unwrap();
            m.2 += 1;
            m.2
        };
        thread::spawn(move || {
            let mut d = Done(self, n, id, None);
            if let Some(i) = d.2.as_deref() {
                d.3 = (self.look)(Some(i)).filter(|x| is(x, i));
            }
        });
    }

    /// The newest pick's device, else the system default; waits while its lookup runs. macOS: a picked device that was
    /// not there at the pick or is gone since (the device kept names it by CoreAudio's id, which lasts while it is
    /// connected) is looked up again here, and kept when found: one back under a new id plays there again.
    fn get(&self) -> Option<cpal::Device> {
        let def = || cpal::default_host().default_output_device();
        let m = self.c.wait_while(self.m.lock().unwrap(), |m| m.3 < m.2);
        let m = m.unwrap();
        let d = m.1.clone();
        #[cfg(target_os = "macos")]
        if let Some(i) = (m.0.clone()).filter(|i| !d.as_ref().is_some_and(|d| is(d, i))) {
            let (n, look) = (m.2, self.look);
            drop(m);
            // a lookup that panics gives the default, as at the pick
            let d = std::panic::catch_unwind(|| look(Some(&i)));
            let d = d.ok().flatten().filter(|x| is(x, &i));
            let mut m = self.m.lock().unwrap();
            // unless a newer pick came meanwhile
            if m.2 == n {
                m.1 = d.clone();
            }
            drop(m);
            return d.or_else(def);
        }
        drop(m);
        d.or_else(def)
    }
}

/// The output device the user picked, else the system default. Waits for a lookup under way (just after a pick):
/// never call it on the UI thread.
pub fn output() -> Option<cpal::Device> {
    AUDIO.get()
}

/// A random (version 4) UUID.
fn uuid() -> String {
    let mut b = [0u8; 16];
    if let Err(e) = File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut b)) {
        eprintln!("x5desk: /dev/urandom: {e}");
    }
    (b[6], b[8]) = (b[6] & 0x0f | 0x40, b[8] & 0x3f | 0x80);
    let h: String = b.iter().map(|x| format!("{x:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    )
}

impl Prefs {
    /// `None`: nothing persists (test hooks); a missing id is made and saved.
    pub fn load(file: Option<PathBuf>) -> Prefs {
        let mut p = Prefs {
            preset: None,
            audio: None,
            cache: album::CAP,
            id: String::new(),
            file,
        };
        let t = p
            .file
            .as_ref()
            .and_then(|f| std::fs::read_to_string(f).ok())
            .unwrap_or_default();
        for l in t.lines() {
            match l.split_once('=') {
                Some(("preset", v)) => p.preset = Some(v.into()),
                Some(("audio", v)) => p.audio = Some(v.into()),
                Some(("cache", v)) => p.cache = v.parse().unwrap_or(p.cache),
                Some(("id", v)) => p.id = v.into(),
                _ => {}
            }
        }
        if p.id.is_empty() {
            p.id = uuid();
            p.save();
        }
        p
    }

    pub fn save(&self) {
        let Some(f) = &self.file else { return };
        let mut s = String::new();
        for (k, v) in [
            ("preset", self.preset.clone()),
            ("audio", self.audio.clone()),
            ("cache", Some(self.cache.to_string())),
            ("id", Some(self.id.clone())),
        ] {
            if let Some(v) = v {
                s += &format!("{k}={v}\n");
            }
        }
        let t = f.with_extension("new");
        let r = f
            .parent()
            .map_or(Ok(()), crate::private)
            .and_then(|_| std::fs::write(&t, s))
            .and_then(|_| std::fs::rename(&t, f));
        if let Err(e) = r {
            eprintln!("{}: {e}", f.display());
        }
    }

    /// The thumbnail cache at its limit; none when nothing persists (test hooks: the user's cache is neither read,
    /// written nor trimmed).
    pub fn thumbs(&self) -> album::Cache {
        let mut c = match self.file {
            Some(_) => album::Cache::new(),
            None => album::Cache::at(None),
        };
        c.cap = self.cache;
        c
    }

    /// The default preset's index.
    pub fn preset(&self) -> Option<usize> {
        let k = self.preset.as_deref()?;
        export::PRESETS.iter().position(|p| p.0 == k)
    }
}

/// Cache limits offered, MB.
const CAPS: [u64; 5] = [250, 500, 1000, 2000, 5000];

fn mb(b: u64) -> String {
    match b {
        0..1_000_000_000 => format!("{:.1} MB", b as f64 / 1e6),
        _ => format!("{:.2} GB", b as f64 / 1e9),
    }
}

pub struct Settings {
    pub p: Prefs,
    /// the shared device: name and value per row
    gpu: Vec<(&'static str, String)>,
    /// its decode checks' counts, which change ([`checks`])
    #[cfg(target_os = "linux")]
    checks: std::sync::Arc<video::gpu::Checks>,
    cache: album::Cache,
    /// the thumbnail cache's files and bytes, measured when the page opens
    usage: Option<Usage>,
    /// the cache thread (a directory of up to ~5000 files): trims to a limit (None: only measures), sends the usage
    io: Option<(Sender<Option<u64>>, Receiver<Usage>)>,
    /// output devices (id, name), listed on a thread when the page first opens: ALSA takes a while
    outs: Vec<(String, String)>,
    listing: Option<Receiver<Vec<(String, String)>>>,
    listed: bool,
    /// the frame it was last drawn in
    shown: Option<u64>,
}

type Usage = (usize, u64);

/// The driver's name and version as wgpu has them, the backend when it has none (Metal: the system's own).
pub fn driver(i: &wgpu::AdapterInfo) -> String {
    let v: Vec<&str> = [&i.driver, &i.driver_info]
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    match v.is_empty() {
        true => format!("{:?}", i.backend),
        false => v.join(" "),
    }
}

/// The Graphics page's decode checks row: the pictures a redo recovered and those that could not be decoded since the
/// start, the memory the redos read (the host's fails more often); none where decodes aren't checked (RADV reports no
/// decode status).
#[cfg(target_os = "linux")]
fn checks(c: &video::gpu::Checks) -> Option<String> {
    use std::sync::atomic::Ordering::Relaxed;
    let (r, f) = (c.redone.load(Relaxed), c.failed.load(Relaxed));
    let m = c
        .read()
        .map_or(String::new(), |m| format!(" (redone from {m} memory)"));
    let s = if r == 1 { "" } else { "s" };
    c.on.then(|| format!("{r} picture{s} recovered{m}, {f} could not be decoded"))
}

/// Codec operations of a video queue, as text.
#[cfg(target_os = "linux")]
fn ops(f: ash::vk::VideoCodecOperationFlagsKHR) -> String {
    use ash::vk::VideoCodecOperationFlagsKHR as V;
    let v: Vec<&str> = [
        (V::DECODE_H264, "H.264"),
        (V::DECODE_H265, "H.265"),
        (V::ENCODE_H264, "H.264"),
        (V::ENCODE_H265, "H.265"),
    ]
    .iter()
    .filter(|x| f.contains(x.0))
    .map(|x| x.1)
    .collect();
    if v.is_empty() {
        "none".into()
    } else {
        v.join(", ")
    }
}

impl Settings {
    pub fn new(g: &Gpu, p: Prefs) -> Settings {
        let i = g.adapter.get_info();
        let gpu = vec![
            ("Adapter", i.name.clone()),
            (
                "Type",
                format!(
                    "{:?} ({:04x}:{:04x}), {:?}",
                    i.device_type, i.vendor, i.device, i.backend
                ),
            ),
            ("Driver", driver(&i)),
        ];
        #[cfg(target_os = "linux")]
        let gpu = [
            gpu,
            vec![
                ("Video decode", ops(g.dec.ops)),
                (
                    "Video encode",
                    g.enc.as_ref().map_or("none".into(), |q| ops(q.ops)),
                ),
                (
                    "DMA-BUF export",
                    match g.mem_fd.is_some() {
                        true => "yes (OBS, PipeWire camera)".into(),
                        false => "no".into(),
                    },
                ),
            ],
        ]
        .concat();
        #[cfg(target_os = "macos")]
        let gpu = [
            gpu,
            vec![
                ("Video decode", "VideoToolbox: H.264, H.265".into()),
                ("Video encode", "VideoToolbox: H.264, H.265".into()),
                (
                    "DMA-BUF export",
                    "no (OBS, PipeWire camera: Linux only)".into(),
                ),
            ],
        ]
        .concat();
        let cache = p.thumbs();
        AUDIO.pick(p.audio.clone());
        Settings {
            p,
            gpu,
            #[cfg(target_os = "linux")]
            checks: g.checks.clone(),
            cache,
            usage: None,
            io: None,
            outs: vec![],
            listing: None,
            listed: false,
            shown: None,
        }
    }

    /// Has the cache thread trim the cache to `cap` (None: only measure it) and measure it.
    fn disk(&mut self, ctx: &egui::Context, cap: Option<u64>) {
        let (tx, _) = self.io.get_or_insert_with(|| {
            let ((tx, rx), (utx, urx)) = (mpsc::channel(), mpsc::channel());
            let (c, ctx) = (self.cache.clone(), ctx.clone());
            thread::spawn(move || {
                for cap in rx {
                    if let Some(x) = cap {
                        c.trim(x)
                    }
                    if utx.send(c.usage()).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
            });
            (tx, urx)
        });
        let _ = tx.send(cap);
    }

    /// `frame`: the UI frame number (a page shown again after a gap is measured again).
    pub fn ui(&mut self, ui: &mut egui::Ui, frame: u64, cam: &mut Cam, exp: &mut export::Exp) {
        if let Some((_, rx)) = &self.io {
            self.usage = rx.try_iter().last().or(self.usage);
        }
        if self.shown.map(|s| s + 1) != Some(frame) {
            self.disk(ui.ctx(), None);
        }
        self.shown = Some(frame);
        if !self.listed {
            self.listed = true;
            let (tx, rx) = mpsc::channel();
            let c = ui.ctx().clone();
            thread::spawn(move || {
                let v = cpal::default_host()
                    .output_devices()
                    .map(|d| {
                        d.filter_map(|d| {
                            let n = d.description().ok()?.name().to_string();
                            Some((d.id().ok()?.to_string(), n))
                        })
                        .collect()
                    })
                    .unwrap_or_default();
                let _ = tx.send(v);
                c.request_repaint();
            });
            self.listing = Some(rx);
        }
        if let Some(v) = self.listing.as_ref().and_then(|r| r.try_recv().ok()) {
            (self.outs, self.listing) = (v, None);
        }
        let grid = |ui: &mut egui::Ui, id: &str, f: &mut dyn FnMut(&mut egui::Ui)| {
            egui::Grid::new(id)
                .num_columns(2)
                .spacing([24.0, 8.0])
                .min_col_width(160.0)
                .show(ui, |ui| f(ui));
        };
        let weak = ui.visuals().weak_text_color();
        egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            ui.heading("General");
            grid(ui, "set general", &mut |ui| {
                ui.label("Language");
                ui.label("English. Dates read year-month-day, sizes are in decimal units (1 MB = 1 000 000 bytes).");
                ui.end_row();
                ui.label("Download folder");
                ui.horizontal_wrapped(|ui| {
                    ui.monospace(cam.dest().display().to_string());
                    if ui.button("Change…").clicked() {
                        cam.pick_dest();
                    }
                });
                ui.end_row();
                ui.label("");
                ui.colored_label(
                    weak,
                    "Camera downloads and card imports go here; the Library lists the recordings in it.",
                );
                ui.end_row();
                ui.label("Default export preset");
                let cur = self.p.preset();
                egui::ComboBox::from_id_salt("set preset")
                    .selected_text(export::PRESETS[cur.unwrap_or(0)].1)
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        for (i, x) in export::PRESETS.iter().enumerate() {
                            if ui.selectable_label(cur.unwrap_or(0) == i, x.1).clicked() {
                                self.p.preset = Some(x.0.into());
                                self.p.save();
                                // settings that still are a preset's follow; changed ones stay
                                if exp.set.preset.is_some() {
                                    exp.set.apply(i)
                                }
                            }
                        }
                    });
                ui.end_row();
            });
            ui.add_space(12.0);
            ui.heading("Audio");
            grid(ui, "set audio", &mut |ui| {
                ui.label("Output");
                let name = |id: &Option<String>| match id {
                    None => "System default".to_string(),
                    Some(i) => (self.outs.iter().find(|o| o.0 == *i)).map_or(i.clone(), |o| o.1.clone()),
                };
                let mut pick = None;
                egui::ComboBox::from_id_salt("set audio out")
                    .selected_text(name(&self.p.audio))
                    .width(320.0)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(self.p.audio.is_none(), "System default").clicked() {
                            pick = Some(None);
                        }
                        for (i, n) in &self.outs {
                            if ui.selectable_label(self.p.audio.as_ref() == Some(i), n).on_hover_text(i).clicked() {
                                pick = Some(Some(i.clone()));
                            }
                        }
                        if self.listing.is_some() {
                            ui.label("Listing devices…");
                        }
                    });
                if let Some(a) = pick {
                    AUDIO.pick(a.clone());
                    self.p.audio = a;
                    self.p.save();
                }
                ui.end_row();
                ui.label("");
                ui.colored_label(
                    weak,
                    "The player and the live outputs play here from the next file opened, the editor's preview from \
                     its next start.",
                );
                ui.end_row();
            });
            ui.add_space(12.0);
            ui.heading("Thumbnail cache");
            grid(ui, "set cache", &mut |ui| {
                ui.label("In use");
                ui.horizontal(|ui| {
                    match (self.cache.dir(), self.usage) {
                        (None, _) => ui.label("no cache folder"),
                        (_, Some((n, b))) => ui.label(format!("{n} thumbnails, {}", mb(b))),
                        _ => ui.label("measuring…"),
                    };
                    if ui.button("Clear").on_hover_text("Thumbnails come from the camera again").clicked() {
                        self.disk(ui.ctx(), Some(0));
                    }
                });
                ui.end_row();
                ui.label("Limit");
                let mut c = self.p.cache;
                egui::ComboBox::from_id_salt("set cache cap")
                    .selected_text(mb(c))
                    .show_ui(ui, |ui| {
                        for m in CAPS {
                            ui.selectable_value(&mut c, m * 1_000_000, mb(m * 1_000_000));
                        }
                        // the limit of an older version, or one set by hand
                        if !CAPS.contains(&(self.p.cache / 1_000_000)) {
                            ui.selectable_value(&mut c, self.p.cache, mb(self.p.cache));
                        }
                    });
                if c != self.p.cache {
                    (self.p.cache, self.cache.cap) = (c, c);
                    self.p.save();
                    // the least recently used go first
                    self.disk(ui.ctx(), Some(c));
                }
                ui.end_row();
                ui.label("Folder");
                ui.monospace(self.cache.dir().map_or("none".into(), |d| d.display().to_string()));
                ui.end_row();
            });
            ui.add_space(12.0);
            ui.heading("Graphics");
            grid(ui, "set gpu", &mut |ui| {
                for (k, v) in &self.gpu {
                    ui.label(*k);
                    ui.label(v);
                    ui.end_row();
                }
                #[cfg(target_os = "linux")]
                if let Some(v) = checks(&self.checks) {
                    ui.label("Decode checks");
                    ui.label(v);
                    ui.end_row();
                    ui.label("");
                    ui.colored_label(
                        weak,
                        "The driver reports whether each picture finished decoding. One it left unfinished is decoded \
                         again before it is shown or exported: recovered pictures are correct. One still unfinished \
                         after three tries is used as it came out. Counted since x5desk started.",
                    );
                    ui.end_row();
                }
            });
            ui.add_space(12.0);
            ui.heading("Camera link");
            grid(ui, "set id", &mut |ui| {
                ui.label("Client id");
                ui.monospace(&self.p.id);
                ui.end_row();
                ui.label("");
                ui.colored_label(
                    weak,
                    "The camera remembers this computer by it once you allow the connection on the camera.",
                );
                ui.end_row();
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Saved and read back; a new file gets an id of its own, kept across loads; unknown lines are skipped.
    #[test]
    fn persist() {
        let d = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-settings-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let f = d.join("settings");
        let mut p = Prefs::load(Some(f.clone()));
        assert_eq!(
            (p.preset(), p.audio.as_ref(), p.cache),
            (None, None, album::CAP)
        );
        let id = p.id.clone();
        assert!(id.len() == 36 && id.as_bytes()[14] == b'4', "{id}");
        assert_ne!(id, Prefs::load(None).id);
        (p.preset, p.audio, p.cache) = (
            Some("flat1080".into()),
            Some("alsa:pipewire".into()),
            500_000_000,
        );
        p.save();
        std::fs::write(&f, std::fs::read_to_string(&f).unwrap() + "colour=blue\n").unwrap();
        let q = Prefs::load(Some(f.clone()));
        assert_eq!(q, p);
        assert_eq!(q.preset(), Some(4));
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The output device, looked up off the UI thread: every listed id as the page stores it finds that device
    /// (cpal's `device_by_id` took ALSA's `sysdefault:CARD=x` and `usbstream:CARD=x` for other ids and played on the
    /// default), an unknown one gives the default, and of two picks in a row the later wins whichever lookup ends
    /// first.
    #[test]
    fn out() {
        // not AUDIO: other tests play on it
        static O: Out = Out::new(media::audio::device);
        let h = cpal::default_host();
        let id = |d: Option<cpal::Device>| Some(d?.id().ok()?.to_string());
        let outs: Vec<String> = (h.output_devices().into_iter().flatten())
            .filter_map(|d| id(Some(d)))
            .collect();
        let t0 = std::time::Instant::now();
        let bad: Vec<&String> = (outs.iter())
            .filter(|i| {
                O.pick(Some(i.to_string()));
                id(O.get()).as_ref() != Some(i)
            })
            .collect();
        println!(
            "{} output devices, {:.1} ms a lookup; found as another: {bad:?}",
            outs.len(),
            t0.elapsed().as_secs_f64() * 1e3 / outs.len().max(1) as f64
        );
        assert!(bad.is_empty());
        if let [.., a, b] = &outs[..] {
            O.pick(Some(a.clone()));
            O.pick(Some(b.clone()));
            assert_eq!(id(O.get()).as_ref(), Some(b));
        }
        O.pick(Some("alsa:no such device".into()));
        assert_eq!(id(O.get()), id(h.default_output_device()));
    }

    /// A lookup that panics still counts as done: `get` gives the system default instead of waiting for good (every
    /// file opened after it would hang), and the next pick is looked up as usual.
    #[test]
    fn out_panic() {
        static O: Out = Out::new(|i| match i {
            Some("panic") => panic!("a lookup that panics (this test)"),
            i => media::audio::device(i),
        });
        let id = |d: Option<cpal::Device>| Some(d?.id().ok()?.to_string());
        let def = id(cpal::default_host().default_output_device());
        let get = || {
            let (tx, rx) = mpsc::channel();
            thread::spawn(move || tx.send(id(O.get())));
            rx.recv_timeout(std::time::Duration::from_secs(10))
        };
        O.pick(Some("panic".into()));
        assert_eq!(get(), Ok(def.clone()), "waits for the lookup that panicked");
        O.pick(None);
        assert_eq!(get(), Ok(def));
    }

    /// macOS: a picked device that was not there at its pick, or whose kept device now names another one or none
    /// (CoreAudio's ids last while a device is connected), is looked up again at the next `get` and kept: one back
    /// under a new id plays there again.
    #[cfg(target_os = "macos")]
    #[test]
    fn out_again() {
        static O: Out = Out::new(media::audio::device);
        let id = |d: &cpal::Device| d.id().unwrap().to_string();
        let outs: Vec<cpal::Device> =
            (cpal::default_host().output_devices().into_iter().flatten()).collect();
        let Some(a) = outs.first() else {
            eprintln!("no output device: skipped");
            return;
        };
        let kept = || O.m.lock().unwrap().1.as_ref().map(id);
        *O.m.lock().unwrap() = (Some(id(a)), None, 1, 1);
        let t0 = std::time::Instant::now();
        let got = O.get().as_ref().map(id);
        let (ms, n) = (t0.elapsed().as_secs_f64() * 1e3, outs.len());
        println!("{n} output devices; {ms:.2} ms to look one up again");
        assert_eq!(got, Some(id(a)), "not there at the pick");
        assert_eq!(kept(), Some(id(a)));
        if let Some(b) = outs.get(1) {
            O.m.lock().unwrap().1 = Some(b.clone());
            let got = O.get().as_ref().map(id);
            assert_eq!(got, Some(id(a)), "kept as another device");
            assert_eq!(kept(), Some(id(a)));
        }
    }

    /// The page reads and trims the thumbnail cache on a thread of its own: its first frame (pass 1 here, which the
    /// page took for a continuation and never measured) has no number yet, the usage follows, Clear's request (limit
    /// 0) empties the cache; a test hook's prefs have no cache.
    #[test]
    fn cache_thread() {
        let d = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-thumbs-{}", std::process::id()));
        let t = d.join("thumbs");
        std::fs::create_dir_all(&t).unwrap();
        for i in 0..3 {
            std::fs::write(t.join(format!("{i:016x}")), [0u8; 100]).unwrap();
        }
        let g = std::sync::Arc::new(Gpu::new().unwrap());
        let cx = stitch::Ctx::new(&g.device, &g.queue);
        let ctx = egui::Context::default();
        // one pass per frame: a second one (the grids' first layout) would take the thread's answer in the same frame
        ctx.options_mut(|o| o.max_passes = 1.try_into().unwrap());
        let p = Prefs::load(None);
        assert!(p.thumbs().dir().is_none());
        let mut cam = Cam::new(
            &ctx,
            None,
            &[],
            false,
            None,
            &p,
            None,
            (crate::bt::sim(false), vec![]),
            false,
        );
        let mut exp = export::Exp::new(&g, &cx, &ctx, Some(d.join("out")), false);
        let mut s = Settings::new(&g, p);
        s.cache = album::Cache::at(Some(t.clone()));
        let mut n = 0;
        let mut frame = |s: &mut Settings| {
            n += 1;
            let mut o = ctx.run_ui(Default::default(), |ui| s.ui(ui, n, &mut cam, &mut exp));
            // no renderer takes them
            o.textures_delta.clear();
            s.usage
        };
        type F<'a> = dyn FnMut(&mut Settings) -> Option<Usage> + 'a;
        let until = |s: &mut Settings, f: &mut F, u| {
            let t0 = std::time::Instant::now();
            while f(s) != Some(u) {
                assert!(t0.elapsed().as_secs() < 10, "{:?}", s.usage);
                thread::sleep(std::time::Duration::from_millis(5));
            }
        };
        assert_eq!(frame(&mut s), None, "the UI thread read the cache");
        until(&mut s, &mut frame, (3, 300));
        s.disk(&egui::Context::default(), Some(0));
        until(&mut s, &mut frame, (0, 0));
        assert_eq!(std::fs::read_dir(&t).unwrap().count(), 0);
        drop((s, frame));
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Graphics' decode checks row: none where decodes aren't checked (RADV); where they are (NVIDIA), pictures
    /// `Decoder::fake` marks unfinished are redone, and the device's count over two decoders, one gone already, rises
    /// by the pictures they recovered, which the row says.
    #[cfg(target_os = "linux")]
    #[test]
    fn decode_checks() {
        use std::sync::atomic::Ordering::Relaxed;
        use video::{Codec, Decoder, check};
        let g = Gpu::new().unwrap();
        let s = Settings::new(&g, Prefs::load(None));
        let row = || checks(&s.checks);
        if !g.checks.on {
            println!("{}: decodes not checked, no row", g.adapter.get_info().name);
            return assert_eq!(row(), None);
        }
        let (side, mut sum) = (256, 0);
        assert_eq!(
            row().as_deref(),
            Some("0 pictures recovered, 0 could not be decoded")
        );
        for _ in 0..2 {
            let mut d =
                Decoder::new(&g, Codec::H264, &check::sets(side, true), side, side).unwrap();
            assert!([1, 2, 4].iter().all(|&f| d.fake(f, false)));
            for k in 0..3 {
                for au in [check::flat(side, k, 128), check::skip(side, 1)] {
                    let f = d.decode(&g, &[&au]).unwrap();
                    d.release(&g, f);
                }
            }
            let t0 = std::time::Instant::now();
            while d.ready() < d.frame {
                assert!(
                    t0.elapsed().as_secs() < 10,
                    "{} of {} ready",
                    d.ready(),
                    d.frame
                );
                thread::sleep(std::time::Duration::from_millis(5));
            }
            assert!(
                d.redone() >= 3 && d.failed() == 0,
                "{} redone, {} failed",
                d.redone(),
                d.failed()
            );
            sum += d.redone();
        }
        let r = row().unwrap();
        println!("{sum} redone: {r}");
        assert_eq!(g.checks.redone.load(Relaxed), sum);
        assert!(
            r.starts_with(&format!("{sum} pictures recovered (redone from ")),
            "{r}"
        );
        assert!(r.ends_with(", 0 could not be decoded"), "{r}");
    }
}
