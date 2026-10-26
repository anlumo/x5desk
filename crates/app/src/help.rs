//! Help screen: keyboard and mouse, a short guide per screen, and About: versions, and the licences of every bundled
//! model and table with what is known about their training data (texts from `crates/*/models/`).

use eframe::egui;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Page {
    Keys,
    Library,
    Editor,
    Lab,
    Camera,
    Capture,
    Live,
    Export,
    Settings,
    About,
}

impl Page {
    pub fn parse(s: &str) -> Option<Page> {
        PAGES
            .iter()
            .find(|p| p.1.eq_ignore_ascii_case(s) || p.2 == s)
            .map(|p| p.0)
    }
}

/// Page, title, hook name.
const PAGES: [(Page, &str, &str); 10] = [
    (Page::Keys, "Keyboard and mouse", "keys"),
    (Page::Library, "Library", "library"),
    (Page::Editor, "Editor", "editor"),
    (Page::Lab, "Shot Lab", "lab"),
    (Page::Camera, "Camera", "camera"),
    (Page::Capture, "Capture", "capture"),
    (Page::Live, "Live", "live"),
    (Page::Export, "Export", "export"),
    (Page::Settings, "Settings", "settings"),
    (Page::About, "About", "about"),
];

/// Where, keys, what.
const KEYS: [(&str, &str, &str); 20] = [
    ("Viewers", "Drag", "Look around (360 views)"),
    (
        "Viewers",
        "Drag (Deep Track on)",
        "Draw a box around the subject to track",
    ),
    ("Viewers", "Scroll", "Zoom: narrower or wider field of view"),
    ("Viewers", "Double-click", "Reset the view"),
    ("Player", "Space", "Play / pause"),
    ("Player", "← / →", "One frame back / forward"),
    ("Plain videos", "Space", "Play / pause"),
    ("Editor", "Space", "Play / pause"),
    ("Editor", "S", "Split the clip at the playhead"),
    ("Editor", "Delete", "Delete the selected clip"),
    (
        "Editor",
        "K",
        "Reframing key at the playhead from the viewer's view",
    ),
    ("Editor", "Ctrl+Z", "Undo"),
    ("Editor", "Ctrl+Shift+Z, Ctrl+Y", "Redo"),
    ("Editor", "← / →", "Playhead one frame back / forward"),
    ("Album", "Click", "Select an item"),
    (
        "Album",
        "Ctrl+click",
        "Add an item to the selection or take it out",
    ),
    (
        "Album",
        "Shift+click",
        "Select the range from the last click",
    ),
    (
        "Album",
        "Double-click",
        "Open a downloaded video (on a card: straight from the card)",
    ),
    (
        "Window",
        "Drop a file",
        "Open it (on the Editor: add it to the timeline)",
    ),
    (
        "Window",
        "Ctrl+Plus / Ctrl+Minus / Ctrl+0",
        "Make the interface larger / smaller / normal",
    ),
];

/// Per screen: paragraphs.
fn guide(p: Page) -> &'static [&'static str] {
    match p {
        Page::Library => &[
            "Open an X5 recording (.insv, .lrv) or photo (.jpg, .insp) with Open…, by dropping it on the window, or \
             from the Recent and Downloaded lists on the left. The files of one recording (both lenses, the proxy) \
             open together, whichever you pick.",
            "The viewer stitches the two lenses live. Pick the view below it: perspective, Panini, equirect or tiny \
             planet. Stabilisation follows the camera's gyro: off, free, horizon lock or direction lock, with a \
             smoothing time.",
            "When the full resolution falls behind, playback switches to the proxy. Paused, the frame shows with the \
             AI seam (RAFT).",
            "Export… takes the file and the current view to the Export screen; Grab 360 and Grab view make a JPEG of \
             the frame shown there. A file's menu in the lists exports it or adds it to the editor.",
            "Deep Track: turn it on and draw a box around a subject in the perspective view. It is tracked from that \
             frame to the end on the AI models (loaded on first use, a minute's warm-up the first time); the bar \
             under the seek bar shows where it was tracked (green), checked by the detector (teal), missing (amber), \
             lost (red) and found again (blue). Then the view follows it, and a flat export takes that follow view.",
            "Blur faces previews the export's privacy blur on the paused frame. Auto edit makes an editor timeline of \
             the clip's highlights, framed automatically.",
        ],
        Page::Editor => &[
            "Drag recordings from the media list onto the timeline. Trim clips at their edges, split them at the \
             playhead (S), reorder them by dragging, delete the selected one (Delete).",
            "Reframe: set the view in the viewer and press K for a key; keys ease between views, or apply a camera \
             movement preset or a Shot Lab cut. Speed ramps and freeze frames are on the Clip tab.",
            "Colour: a look (ours or an imported .cube) and adjustments per clip, each with a button that puts it on \
             every clip. Audio: clip volume, mute, music that ducks under the clips' audio, voice denoise. Overlays: \
             text, a PNG sticker or watermark, a nadir patch over the tripod.",
            "Undo and redo cover every step. Save the project to a file, and Export timeline hands it to the Export \
             screen's queue.",
            "AI: on the Reframe tab, Track a subject and draw a box around it in the viewer; it is followed from the \
             playhead to the clip's end and the clip gets keys that keep it in view (its tracking states show along \
             the clip). A media file's menu (right click) has Auto edit: its highlights, framed automatically, onto \
             the timeline.",
        ],
        Page::Lab => &[
            "Shot Lab works on the clip open in the Library. Pick an effect, set its range and parameters; Preview \
             shows what it makes at the range's out point (the editing presets: four moments through their views).",
            "The passes (Remove Passerby, Light Trails, Star Lapse, Frame Delay, Shadow Clone, Clone Loop, Parallel \
             Planet) export as 360 videos. Those that need a still camera check the gyro and warn when it turned; \
             they stitch with the direction lock. The editing presets (Stop Motion, Fly Lapse, Flip My Day, Center \
             Stage, Spin Me Around, Fly Out, Dolly Zoom) export flat or go onto the editor's timeline.",
            "Shadow Clone, Clone Loop and the subject presets follow the subject tracked with Deep Track in the \
             player. With Blur faces on in the Export screen, a pass blurs the faces before the effect runs.",
        ],
        Page::Camera => &[
            "Connect the X5 by USB and choose Android App on the camera's USB menu (app mode). The camera connects \
             automatically (Auto-connect); battery and card space show at the top, and a low battery warns on every \
             screen.",
            "Album: the camera's files grouped the way the official app groups them, with thumbnails. Select with \
             click, Ctrl+click and Shift+click; Download copies the files into the download folder, four connections \
             at a time; a stopped download resumes where it stopped, also after a restart. Favourite and Delete act \
             on the camera after a confirmation.",
            "Mass storage: when the X5 is plugged in as a USB disk (or its card is in a reader) and the desktop mounts \
             it, the album offers it as a source. Import copies the selected items into the download folder; an \
             interrupted import resumes.",
            "Camera: model, firmware, serial, clock (Sync time sets it), the camera's Wi-Fi name and password, and the \
             SD card format.",
            "Wireless: Bluetooth (scan, connect, pair, wake from standby, read the access point) and Wi-Fi station \
             mode (the camera joins your network, then x5desk connects to it there). The X5 ignores Bluetooth while \
             it is connected by USB: use USB then.",
        ],
        Page::Capture => &[
            "The camera's live preview, stitched live in the same viewer as the player. The controls come from the \
             camera itself: its modes and, for the mode on, its settings with the values it allows.",
            "The shutter takes a photo or starts and stops a recording (timelapse too); the self-timer counts down \
             here. A highlight mark flags the moment in the recording.",
            "The X5's Webcam Mode (UVC) can be the source instead: it shows the camera's own equirect or flat picture.",
            "Settings the app has not verified on a real X5 are marked ⚠.",
        ],
        Page::Live => &[
            "Send a source live: the player (its view or the 360 picture), the camera's live preview, or its Webcam \
             Mode.",
            "RTMP: enter the server URL and stream key, choose resolution, frame rate and bitrate, then go live. OBS: \
             add the x5desk source in OBS (plugin x5desk-obs.so); frames arrive without copies. PipeWire camera: \
             other apps see x5desk as a camera.",
            "The outputs keep running while the window is hidden or minimised.",
        ],
        Page::Export => &[
            "A job comes from the Library (a file and its view), from a frame grab, or from the editor's timeline. \
             Pick a preset or set resolution, codec (H.265 8 or 10-bit, H.264), bitrate or constant quality, and \
             frame rate.",
            "360 exports carry spherical metadata; flat ones follow the view. Options: AI seam (RAFT), colour \
             matching between the lenses, stabilisation, a look and adjustments, and privacy blur of faces.",
            "Jobs run one after another; each shows its progress and can be cancelled. Open shows the result in the \
             Library.",
        ],
        Page::Settings => &[
            "The download folder (camera downloads and card imports), the default export preset, the audio output of \
             the player, the live outputs and the editor's preview, and the thumbnail cache's limit and clearing.",
            "Graphics shows the shared Vulkan device: decode, encode and DMA-BUF export capabilities. The client id \
             is what the camera remembers when you allow this computer on it.",
            "Settings are kept in ~/.config/x5desk; thumbnails in ~/.cache/x5desk.",
        ],
        Page::Keys | Page::About => &[],
    }
}

/// A bundled model or table: name, where it serves, licence, copyright and source, what is known about its data,
/// licence text.
struct Model(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
);

const MODELS: [Model; 7] = [
    Model(
        "RAFT-small",
        "seam flow (stitch)",
        "BSD-3-Clause",
        "torchvision weights Raft_Small_Weights.C_T_V2, exported to ONNX; Copyright (c) Soumith Chintala 2016",
        "⚠ Trained on FlyingChairs and FlyingThings3D, synthetic datasets the University of Freiburg provides for \
         research only (\"Any commercial use is prohibited\"); the weights are published under BSD-3-Clause by \
         torchvision, which notes that pre-trained weights may carry their training data's terms. Check the datasets' \
         terms before any commercial use.",
        include_str!("../../stitch/models/LICENSE-torchvision"),
    ),
    Model(
        "YOLOX-nano",
        "subject detection (track)",
        "Apache-2.0",
        "Megvii-BaseDetection/YOLOX release 0.1.1rc0; Copyright (c) 2021-2022 Megvii Inc.",
        "Trained on COCO 2017.",
        include_str!("../../track/models/LICENSE-yolox"),
    ),
    Model(
        "VitTrack",
        "single-subject tracking (track)",
        "Apache-2.0",
        "OpenCV model zoo, object_tracking_vittrack_2023sep (Pengyu Liu, GSoC 2023)",
        "⚠ The zoo does not name its training data; its contributor's training configuration trains on LaSOT, \
         GOT-10k, COCO 2017 and TrackingNet, and GOT-10k is released under CC BY-NC-SA 4.0 (research use; commercial \
         use only by arrangement with its authors). The weights are published under Apache-2.0. Check the datasets' \
         terms before any commercial use.",
        include_str!("../../track/models/LICENSE-vittrack"),
    ),
    Model(
        "YuNet",
        "face detection for the privacy blur (track)",
        "MIT",
        "OpenCV model zoo, face_detection_yunet_2023mar; Copyright (c) 2020 Shiqi Yu",
        "⚠ Trained on WIDER FACE, released under CC BY-NC-ND (non-commercial); the weights are published under MIT \
         by their authors. Check the dataset's terms before any commercial use.",
        include_str!("../../track/models/LICENSE-yunet"),
    ),
    Model(
        "PP-HumanSeg v1 (mobile)",
        "person segmentation for Shot Lab (shotlab)",
        "Apache-2.0",
        "PaddleSeg via the OpenCV model zoo, human_segmentation_pphumanseg_2023mar; Copyright (c) 2021 PaddlePaddle \
         Authors",
        "Trained on large-scale portrait data its authors do not name.",
        include_str!("../../shotlab/models/LICENSE-pphumanseg"),
    ),
    Model(
        "RNNoise 0.2",
        "voice denoise (media)",
        "BSD-3-Clause",
        "Xiph.Org rnnoise model data; Copyright (c) 2007-2017, 2024 Jean-Marc Valin, (c) 2023 Amazon, (c) 2017 \
         Mozilla, (c) 2005-2017 Xiph.Org Foundation, (c) 2003-2004 Mark Borgerding",
        "Trained on public datasets (RNNoise's datasets.txt).",
        include_str!("../../media/models/LICENSE-rnnoise"),
    ),
    Model(
        "vo-aacenc tables",
        "AAC Huffman codebooks of the AAC encoder (media)",
        "Apache-2.0",
        "VisualOn's AAC encoder (mstorsjo/vo-aacenc); Copyright 2003-2010 VisualOn",
        "Code tables, not a trained model.",
        include_str!("../../media/src/aacenc/NOTICE-vo-aacenc"),
    ),
];

pub struct Help {
    pub page: Page,
    /// the shared device and its driver
    gpu: String,
    os: String,
}

impl Help {
    pub fn new(g: &video::Gpu, page: Option<Page>) -> Help {
        let i = g.adapter.get_info();
        let rel = |f: &str, k: &str| {
            let s = std::fs::read_to_string(f).unwrap_or_default();
            let v = s
                .lines()
                .find_map(|l| l.strip_prefix(k))
                .map(|v| v.trim_matches('"').to_owned());
            v.unwrap_or_else(|| s.trim().to_owned())
        };
        Help {
            page: page.unwrap_or(Page::Keys),
            gpu: format!("{}, {} {}", i.name, i.driver, i.driver_info),
            os: format!(
                "{}, Linux {}",
                rel("/etc/os-release", "PRETTY_NAME="),
                rel("/proc/sys/kernel/osrelease", "")
            ),
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("help pages")
            .resizable(false)
            .exact_size(170.0)
            .show(ui, |ui| {
                ui.add_space(6.0);
                for (p, t, _) in PAGES {
                    ui.selectable_value(&mut self.page, p, t);
                }
            });
        egui::CentralPanel::default_margins().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| {
                    ui.set_max_width(760.0);
                    ui.heading(PAGES.iter().find(|p| p.0 == self.page).map_or("", |p| p.1));
                    ui.add_space(8.0);
                    match self.page {
                        Page::Keys => self.keys(ui),
                        Page::About => self.about(ui),
                        p => {
                            for t in guide(p) {
                                ui.label(*t);
                                ui.add_space(6.0);
                            }
                        }
                    }
                });
        });
    }

    fn keys(&self, ui: &mut egui::Ui) {
        egui::Grid::new("help keys")
            .num_columns(3)
            .striped(true)
            .spacing([24.0, 6.0])
            .show(ui, |ui| {
                let mut last = "";
                for (w, k, d) in KEYS {
                    ui.label(if w == last { "" } else { w });
                    last = w;
                    ui.monospace(k);
                    ui.label(d);
                    ui.end_row();
                }
            });
        ui.add_space(8.0);
        ui.label("The editor's keys act while no text field has the focus.");
    }

    fn about(&self, ui: &mut egui::Ui) {
        let weak = ui.visuals().weak_text_color();
        egui::Grid::new("help about")
            .num_columns(2)
            .spacing([24.0, 6.0])
            .show(ui, |ui| {
                for (k, v) in [
                    (
                        "x5desk",
                        format!(
                            "{}, licence {}",
                            env!("CARGO_PKG_VERSION"),
                            env!("CARGO_PKG_LICENSE")
                        ),
                    ),
                    ("Built with", env!("X5_RUSTC").into()),
                    ("Libraries", env!("X5_DEPS").into()),
                    ("Graphics", self.gpu.clone()),
                    ("System", self.os.clone()),
                ] {
                    ui.label(k);
                    ui.add(egui::Label::new(v).wrap());
                    ui.end_row();
                }
            });
        ui.add_space(8.0);
        ui.label(
            "A desktop app for the Insta360 X5, independent of Insta360. It talks to the camera with the camera's own \
             messages and uses no Insta360 software, models or online service; lens calibration comes from each \
             file.",
        );
        ui.add_space(12.0);
        ui.heading("Bundled models and tables");
        ui.add_space(4.0);
        for m in &MODELS {
            ui.horizontal_wrapped(|ui| {
                ui.strong(m.0);
                ui.label(format!("· {} · {}", m.1, m.2));
            });
            ui.colored_label(weak, m.3);
            match m.4 {
                "" => {}
                n if n.starts_with('⚠') => drop(ui.colored_label(ui.visuals().warn_fg_color, n)),
                n => drop(ui.label(n)),
            }
            ui.collapsing(format!("{} licence text", m.0), |ui| {
                ui.add(egui::Label::new(egui::RichText::new(m.5).monospace().size(11.0)).wrap());
            });
            ui.add_space(6.0);
        }
        ui.label(
            "Person re-identification (track) uses no model: LOMO, a hand-made appearance feature from Liao et al., \
             \"Person Re-identification by Local Maximal Occurrence Representation and Metric Learning\" (CVPR 2015, \
             arXiv:1406.4216), written for this app from the paper, with no weights and no training data. It replaces \
             OSNet-AIN, whose training data (MSMT17) is for research use only.",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages() {
        assert_eq!(Page::parse("about"), Some(Page::About));
        assert_eq!(Page::parse("Keyboard and mouse"), Some(Page::Keys));
        assert_eq!(Page::parse("nope"), None);
        // every screen has a guide, every model its licence text
        for (p, ..) in PAGES {
            assert!(
                matches!(p, Page::Keys | Page::About) || !guide(p).is_empty(),
                "{p:?}"
            );
        }
        for m in &MODELS {
            assert!(
                m.5.len() > 900 && (m.5.contains("Copyright") || m.5.contains("License")),
                "{}",
                m.0
            );
        }
        assert!(env!("X5_DEPS").contains("eframe 0.37"));
    }
}
