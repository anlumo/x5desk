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
        Mutex,
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

/// The output the editor's preview plays on ([`output`]): [`Prefs::audio`], set by the Settings screen.
static AUDIO: Mutex<Option<String>> = Mutex::new(None);

/// The output device the user picked, else the system default.
pub fn output() -> Option<cpal::Device> {
    dev(AUDIO.lock().unwrap().clone())
}

/// The output device with this id (as [`Prefs::audio`] keeps it), else the system default.
fn dev(id: Option<String>) -> Option<cpal::Device> {
    let h = cpal::default_host();
    id.and_then(|s| h.device_by_id(&s.parse().ok()?))
        .or_else(|| h.default_output_device())
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

/// Codec operations of a video queue, as text.
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
            ("Driver", format!("{} {}", i.driver, i.driver_info)),
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
        ];
        let cache = p.thumbs();
        AUDIO.lock().unwrap().clone_from(&p.audio);
        Settings {
            p,
            gpu,
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
                    AUDIO.lock().unwrap().clone_from(&a);
                    self.p.audio = a;
                    self.p.save();
                }
                ui.end_row();
                ui.label("");
                ui.colored_label(
                    weak,
                    "The editor's preview plays here from its next start. The player and the live outputs use the \
                     system default.",
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
        // an id as the page stores it finds its device again; an unknown one falls back to the default
        let h = cpal::default_host();
        let id = |d: Option<cpal::Device>| Some(d?.id().ok()?.to_string());
        let outs: Vec<String> = (h.output_devices().into_iter().flatten())
            .filter_map(|d| id(Some(d)))
            .collect();
        println!("{} output devices, e.g. {:?}", outs.len(), outs.first());
        if let Some(i) = outs.last() {
            assert_eq!(id(dev(Some(i.clone()))).as_ref(), Some(i));
        }
        let def = id(h.default_output_device());
        assert_eq!(id(dev(Some("alsa:no such device".into()))), def);
        std::fs::remove_dir_all(&d).unwrap();
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
            (crate::bt::sim(), vec![]),
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
}
