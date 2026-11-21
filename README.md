# x5desk

An unofficial desktop app for the Insta360 X5 360° camera: a player, an editor, camera control, live output and
export. It runs on Linux (Wayland or X11, Vulkan) and on macOS (Metal); everything from decoding to the AI models runs
on the GPU, and no FFmpeg is involved.

x5desk is a personal project. It is not affiliated with, endorsed by or supported by Insta360 (Arashi Vision Inc.).
"Insta360" and "X5" are their trademarks and are used here only to say which camera the app works with.

## What it does

- **Library and player.** Opens X5 recordings (`.insv`, `.lrv`) and photos (`.insp`, `.jpg`) and stitches the two
  lenses live. Views: perspective, Panini, equirect, tiny planet. Stabilisation from the camera's gyro: free, horizon
  lock, direction lock, or a forward view along the direction of travel. The seam is aligned by optical flow (RAFT).
  Frame grabs as JPEG, with noise reduction on night footage.
- **Tracking and framing.** Follow a subject you box or click on a paused frame (Deep Track), or let auto framing pick
  the clip's main subject. Faces and licence plates can be blurred. Highlights become an auto edit, cut to a music
  track's beats if you give it one.
- **Library search.** Type words to rank your recordings by what their pictures show; tags (night, ride, city, …) and
  near-duplicates come from the same model.
- **Editor.** A timeline with trim, split, speed ramps and freeze frames; reframing with keyframes and camera movement
  presets; looks (`.cube` LUTs) and colour adjustments per clip; music that ducks under the clips' audio, voice
  denoise; text, stickers, watermark and a nadir patch; undo and project files.
- **Shot Lab.** Effects on a clip: Remove Passerby, Light Trails, Star Lapse, Frame Delay, Shadow Clone, Clone Loop,
  Parallel Planet, Sky Swap, and editing presets such as Dolly Zoom, Fly Out or Overtake.
- **Camera.** Over USB, Wi-Fi or Bluetooth: the album with thumbnails and resumable downloads, import from the card,
  camera information and clock, and capture with the live preview, the camera's modes and settings, the shutter and
  timelapse. The X5's Webcam Mode works as a source as well.
- **Live.** Send the player, the live preview or the webcam picture to an RTMP server, to OBS (a source plugin) or to
  other apps as a PipeWire camera.
- **Export.** H.265 (8 or 10 bit) and H.264, as 360 video with spherical metadata or flat along the view, with the
  options above, in a queue.
- **For AI agents.** `x5desk mcp` is an MCP server that lets an agent list and download the album, render stills and
  run exports, in tiers you enable (`x5desk mcp --help`, `x5desk mcp --print-config`).

The app's Help tab describes each screen. Settings x5desk has not been run with on a real X5 are marked ⚠ in the
app. `ARCHITECTURE.md` describes the code, `TODO.md` what is open.

## Requirements

- Linux on x86-64 with Wayland or X11, PipeWire 1.2 or newer, ALSA and D-Bus (BlueZ for Bluetooth); file dialogs
  come from xdg-desktop-portal.
- A GPU whose Vulkan driver decodes and encodes H.265 and H.264 (Vulkan Video). x5desk is developed on AMD graphics
  with Mesa's RADV (Mesa 26). On NVIDIA it has been run on one card, an RTX 4090 with driver 620.73, without a
  display attached: stills and exports work there. Intel graphics are untested. Settings → Graphics shows what the
  app found; on a machine with two GPUs the app takes the faster one (`VK_ICD_FILENAMES` picks another).
- An Insta360 X5. Other Insta360 cameras have not been tried.
- macOS 15 or later also works (Metal, VideoToolbox); see "macOS" in `ARCHITECTURE.md`. There is no Windows version.

## Install

Download `x5desk-<version>-x86_64-linux.tar.gz` from the Releases page, unpack it and run `x5desk`. The binary is
built in Debian 13, so it needs glibc 2.41 or newer (Debian 13, Ubuntu 25.04, Fedora 42, Arch and later). The archive
also holds this README, the licence and the licence texts of everything bundled (`licenses/`).

To use the camera over USB, choose **Android App** on the camera's USB menu. x5desk talks to the camera directly, so
your user needs access to the USB device. If the camera is not found, add a udev rule and plug the camera in again:

```
# /etc/udev/rules.d/70-x5desk.rules
SUBSYSTEM=="usb", ATTR{idVendor}=="2e1a", TAG+="uaccess"
```

## Build from source

You need:

- Rust 1.98.1 (what x5desk is developed with) or newer, from [rustup](https://rustup.rs).
- Git LFS. Two model files are stored in LFS; without it the checkout holds pointer files and the build fails while
  importing the models.
- A C toolchain, pkg-config, clang (for bindgen) and the development files of ALSA, D-Bus and PipeWire 1.2 or newer
  (Ubuntu 24.04's PipeWire 1.0 is too old: the link fails):

  ```bash
  # Debian 13, Ubuntu 25.04 or newer
  sudo apt install build-essential pkg-config clang libclang-dev git-lfs libasound2-dev libdbus-1-dev libpipewire-0.3-dev
  ```

  ```bash
  # Arch
  sudo pacman -S --needed base-devel clang git-lfs alsa-lib dbus pipewire
  ```

Then:

```bash
git lfs install
```

```bash
git clone https://github.com/anlumo/x5desk.git
```

```bash
cd x5desk && cargo build --release --locked -p app
```

The app is `target/release/x5desk`. The workspace's release profile keeps line tables, which makes that binary about
1.1 GB; `CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_PROFILE_RELEASE_STRIP=debuginfo cargo build --release --locked -p app`
builds it without them.

The release archive is built by `.github/release-linux.sh` in a Debian 13 container, on a GitHub runner for every
`v<version>` tag (`.github/workflows/release.yml`) or by hand:

```bash
docker run --rm -v "$PWD:/src" -e CARGO_HOME=/src/target/cargo -e RUST=1.98.1 -e DIR=x5desk-local-x86_64-linux -e CARGO_PROFILE_RELEASE_DEBUG=0 -e CARGO_PROFILE_RELEASE_STRIP=debuginfo debian:13 /src/.github/release-linux.sh
```

The OBS source plugin is built separately and is not part of the release archive: `cargo build --release --locked -p
obs` makes `target/release/libx5desk_obs.so`, which OBS loads as
`~/.config/obs-studio/plugins/x5desk-obs/bin/64bit/x5desk-obs.so`.

`cargo test --workspace` runs the tests. Those that need footage read X5 recordings of your own from the folder named
by `X5_SAMPLES` and skip without it; the repository contains no footage.

## Licence

x5desk is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE): you may use, change and share it for any
non-commercial purpose.

Required Notice: Copyright 2026 Andreas Monitzer (https://github.com/anlumo/x5desk)

It is non-commercial because of what it bundles. The neural networks compiled into the app are published under
permissive licences, but several were trained on datasets whose terms allow non-commercial use only, and one on data
that states no licence at all:

| Model | Used for | Licence of the weights | Training data to know about |
|---|---|---|---|
| RAFT-small (torchvision) | seam alignment | BSD-3-Clause | FlyingChairs, FlyingThings3D: research only |
| YOLOX-nano (Megvii) | detecting subjects | Apache-2.0 | COCO 2017 |
| VitTrack (OpenCV zoo) | tracking a subject | Apache-2.0 | includes GOT-10k: CC BY-NC-SA 4.0 |
| YuNet (OpenCV zoo) | face detection for the blur | MIT | WIDER FACE: non-commercial |
| YOLOX-nano licence plates (this project's fine-tune) | plate detection for the blur | Apache-2.0 (YOLOX) | open datasets, own footage, and one dataset without a stated licence |
| PP-HumanSeg (PaddleSeg) | person segmentation in Shot Lab | Apache-2.0 | a Baidu dataset |
| EfficientViT-seg B1 (MIT HAN Lab) | sky segmentation for Sky Swap | Apache-2.0 | ADE20K, ImageNet: non-commercial research and education |
| TinyCLIP ViT-39M/16 (Microsoft) | library search | MIT | YFCC-15M (partly non-commercial), distilled from a LAION-2B model |
| RNNoise (Xiph.Org) | voice denoise | BSD-3-Clause | public datasets |

The app's About page shows each model's source, licence text and training data in full; the `models/README` files in
`crates/track`, `crates/stitch`, `crates/shotlab` and `crates/media` record where every file came from and how it was
converted. No Insta360 model is used.

The Rust crates the app is built from are under permissive licences (MIT, Apache-2.0, BSD, ISC, Zlib, Unicode-3.0),
a few under MPL-2.0 (Symphonia and three small crates, used unmodified from crates.io). The release archive carries
their licence texts in `licenses/crates`, with an index of every crate, its licence and its source.

x5desk contains its own AAC encoder and reads and writes H.264 and H.265 streams through the GPU. These formats are
covered by patents in some countries; the licence above grants no rights to third parties' patents.

## Why releasing this app is legitimate

x5desk is an independent program that works with a camera its user owns. This is how the author understands the legal
position; it is not legal advice.

- **It is its own work.** All code in this repository was written for this project. The repository contains no
  Insta360 source code, binaries, firmware, AI models, artwork, LUTs or footage, and the app uses neither Insta360's
  SDK nor its accounts or servers. Stitching, stabilisation, tracking and the effects are implemented here, with the
  open models listed above.
- **What it shares with the camera is an interface.** To talk to the camera and to read its files, a program has to use
  the message layouts the camera expects over USB, Wi-Fi and Bluetooth (`crates/proto/proto`) and the layout of the
  metadata in its recordings. That is interface information, and it is included only as far as working with the camera
  requires. Copyright protects the code of a program, not the protocols and file formats it uses: in the EU, Directive
  2009/24/EC excludes the ideas and principles behind a program's interfaces from protection (Article 1(2)), and the
  Court of Justice held that a program's functionality and its data file formats are not protected expression (SAS
  Institute v World Programming, C-406/10). US law draws the same line (17 U.S.C. § 102(b)).
- **Finding out how to interoperate is permitted.** The same Directive lets the lawful user of a program observe,
  study and test it to determine the ideas and principles behind it (Article 5(3)) and obtain the information needed
  to make an independently created program work with it (Article 6), and it voids contract terms that say otherwise
  (Article 8). In the US, 17 U.S.C. § 1201(f) and the case law on interoperability (Sega v. Accolade, Sony v.
  Connectix) are the counterparts.
- **Nothing is circumvented.** The camera's recordings are ordinary MP4 and JPEG files with unencrypted metadata, and
  its protocol is not encrypted. x5desk breaks no encryption, copy protection or licence check and does not touch the
  camera's firmware. It connects the way the camera offers: over USB once its owner picks the mode on the camera's
  menu, over Wi-Fi once the owner allows the computer on the camera's screen.
- **The names say what it is compatible with.** "Insta360" and "X5" identify the camera; nothing suggests that the
  app comes from or is approved by Insta360.

What this means for you: use x5desk with your own camera and for non-commercial purposes, and expect no warranty.
It is not the vendor's software, so it can behave differently from the official app and may stop working with a
firmware update.

If you hold rights in something here and see a problem, please open an issue.
