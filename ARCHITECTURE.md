# x5desk

Desktop app for the Insta360 X5: player, camera control, live output, export. Linux (Wayland, RADV) first.
The plan with decisions, measurements and milestones is `../insta360/PLAN.md`; the camera protocol and file formats
are documented in `../insta360/re/` (`PROTOCOL.md`, `analysis/*.md`).

## Rules

- No FFmpeg. Video goes through Vulkan Video on the one shared device.
- No Insta360 data in this repo unless the code needs it to run (the `.proto` files the app uses). Lens calibration
  comes from each file's trailer, never from the repo. Seam flow uses RAFT (open model), no Insta360 models.
- The UI thread never paces playback. Media threads decode ahead of the audio clock.
- Tests that need sample footage read the directory in `X5_SAMPLES` (`v635.insv`, `v606.insv`, `v627.insv`,
  `v635.lrv`) and return early with a note when it is unset.
- Tests that make paths in `$XDG_RUNTIME_DIR` name them `<name>-<pid>` and remove them through a drop guard, also
  when they fail.

## Crates

| Crate | Content |
|---|---|
| `video` | `Gpu` (one Vulkan device via wgpu-hal with decode and encode queues), H.265 / H.264 parsers, Vulkan Video decode and encode, NV12 / P010 textures for wgpu |
| `media` | MP4 demux and mux (`sv3d` / `st3d`), `.insv` / `.insp` trailer (calibration, gyro, exposure), AAC decode (symphonia) and AAC-LC encode (`aacenc`), editor mixer with ducking and limiter (`mix`), music import (`music`), resampler, RNNoise denoise, audio output and master clock (cpal; `audio::Player::open_on` plays on a picked device, `audio::device(id)` resolves the settings' device id by exact match: it lists the devices, so resolve when the choice changes, never per open), MJPEG decode, UVC capture of the X5's Webcam Mode |
| `proto` | Protobuf messages for the camera (prost, compiled with protox from `proto/`) |
| `link` | Camera link: UCD framing, session, requests, notifications, HTTP tunnel with several connections, downloads; USB (nusb), Wi-Fi (TCP); BLE lives in `ble`. Connections start as `link::Opening` (dropping it cancels the sync); `Error::Silent` after 30 s without a sync answer; feature `sim`: a simulated camera |
| `stitch` | Dual fisheye → equirect / perspective on the shared device from NV12 / P010 textures: template seam, RAFT flow, colour matching, stabilisation rotation |
| `stab` | Stabilisation maths: trailer gyro / accelerometer → per-frame orientation, smoothing (FlowState-like), horizon and direction lock, rolling-shutter terms |
| `fx` | Image effects on the shared device: colour adjustments (incl. brightness, brilliance), automatic colour and underwater correction, `.cube` LUTs and our own looks, defringe (purple / green / blue), clarity, overlays (watermark, stickers, text) |
| `edit` | Editor model: timeline, clips with a look and colour adjustments each (`fx::Params`: edit depends on fx), trim / split / speed / freeze, reframing keyframes and camera-movement presets, undo (a drag that ends where it started leaves no undo step and keeps the redo list), project files |
| `export` | Render jobs: decode → stitch → stabilise → effects → encode → mux, 360 and flat, queue; editor timelines (`Cut::Edit`: speed ramps, freeze frames, views, each clip's adjustments and look via `fx`, audio mixed with music, RNNoise-denoised after the time-stretch when `Edit::denoise` is set, and AAC-encoded, or copied for plain 1× cuts without denoise); photo frame grabs (JPEG with GPano). `export::locks(gate, burn)` hands it the app's GATE / BURN; `Job::flat` keeps a flat 2:1 picture from a 360 hook untagged |
| `track` | AI with open models (burn on the shared device): subject tracker on the sphere with re-identification (LOMO colour and texture histograms with Fisher weights learnt per session, no model), detector, auto framing, face privacy blur regions, highlights → `edit::Timeline` |
| `ble` | Camera BLE: scan, wake, pairing, Wi-Fi credentials (btleplug). The X5's wake-up brings up its radios and session, never its screen. Feature `sim`: `Ble::sim`, a simulated adapter and camera on the real session code |
| `live` | FLV and RTMP client, OBS frame protocol (Unix socket + DMA-BUF fds), PipeWire camera node |
| `obs` | OBS source plugin (`cdylib`) that imports our frames |
| `app` | egui (eframe 0.37) UI, binary `x5desk` |

## Shared device contract (`video::Gpu`)

- `Gpu::new()` opens the Vulkan adapter through wgpu-hal with every adapter feature, the experimental features,
  `TEXTURE_FORMAT_NV12`, the video decode and encode queue families, and the external memory / semaphore
  extensions (`VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`,
  `VK_KHR_external_semaphore_fd`, `VK_EXT_queue_family_foreign`).
- `Gpu::new` serialises instance creation (`gpu.rs` `SCAN`): the loader runs every ICD's negotiate function from
  overlapping scans, and NVIDIA's ICD (installed on this AMD host) crashes when two threads hit its lazy init. Any
  other `wgpu::Instance::new` in the process (eframe's `CreateNew`) bypasses it and can race.
- Public fields: `instance`, `adapter`, `device`, `queue` (wgpu) and the raw Vulkan handles the video and live code
  need. The app hands the wgpu handles to eframe with `WgpuSetup::Existing`.
- Decoded pictures are NV12 (8-bit) or P010 (H.265 Main 10) `wgpu::Texture`s: the first SPS's bit depth decides,
  `texture(f).format()` tells, and a picture of the other depth is `Error::Bad`. Plane views are `R8Unorm` /
  `Rg8Unorm` or `R16Unorm` / `Rg16Unorm`; a view with the plane aspect and no format gets them. A P010 view reads
  the 10-bit code d as 64·d / 65535. Home state is `RESOURCE`.
  Before each wgpu submit that samples picture `f`, call the decoder's `acquire(&gpu, f)` (stages the timeline wait
  `done ≥ f + 1`). After the last submit that samples it, call `release(&gpu, f)` (stages the signal `free = f + 1`).
  wgpu attaches staged semaphores to the next submit on the queue from any thread: a wait taken by an earlier submit
  and a signal taken by a later one are both still correct in submission order, so neither needs a lock. A paused
  view re-acquires the same frame each redraw and releases it when it moves on. `release(f)` frees `f` and every
  earlier frame of that ring (one timeline value), so holders release in frame order. There is no flush call: a
  decoder that finds its slot blocked by a release still staged sends it with an empty submit itself.
- H.264 input may miss whole access units (the live preview drops packets when the decode thread falls behind):
  frame_num gaps infer non-existing frames (8.2.5.2) that stand in with the pushed-out picture, and an IDR always
  recovers a full DPB. Loss shows as concealment until the next IDR, never as a refused stream.
- Anything else that stages a semaphore signal on wgpu's queue (DMA-BUF export, encoder input) stages it only after
  the submit it covers, never before: a signal staged first can be taken by another thread's earlier submit.
- Stitch sources are whole NV12 / P010 textures or rects in one (full range, BT.709; `Ctx::planes` also takes
  R16 / RG16 views as P010; side by side: `.lrv` and live preview, lens 0 left; the
  `.lrv` needs `Frame::shift`); the preview also draws from an RGBA equirect (paused RAFT stitch, Webcam Mode).
- Encoder input: `Encoder::input` (`Arc<Input>`) is a ring of `encode::SLOTS` NV12 / P010 textures, home state
  `COPY_DST`; producers fill them by copying (wgpu has no NV12 / P010 storage). Before the first wgpu submit that
  writes frame `f` call `input.acquire(&gpu, f)` (`Err(Full)` after 50 ms if the encoder still holds the slot); after
  the last one call `input.release(&gpu, f)`, then hand `f` to the encoder thread, which calls `encode(&gpu, pts)` and
  `packet(&mut Packet, ns)`. Frames are released in order. `Input::key()` makes the next frame an IDR. `Cfg.vbv` is
  the rate control buffer in ms (1000 for files, 100 for low-latency RTMP). The VCN firmware's CBR (RADV and
  radeonsi alike) runs 4 to 10 % over at 100 ms and near the target from about 500 ms on ordinary content (`video`
  test `cbr`); initial occupancy, GOP flags and GOP-remaining hints change nothing. RADV: no B-frames, quality level 3 of
  H.265 under rate control writes corrupt slices (excluded by `Encoder::levels`).
- DMA-BUF export (`live::dmabuf`): `Image::acquire()` before the first submit that writes an exported image (content
  undefined afterwards), `Image::release()` after the last one; it returns the sync_file and also attaches it to the
  DMA-BUF as its write fence, so implicit-sync consumers (PipeWire clients) are correct too.
- A command encoder is either raw (`as_hal_mut`) or wgpu, never both.
- Video decode holds back the same process's graphics submissions by the decode queue's backlog (RADV / amdgpu;
  another process decoding at full speed does not). A decode burst, such as the editor's next deck decoding from
  its keyframe to the next clip's first frame, delays the UI's frames by 40–190 ms: `vkAcquireNextImageKHR` waits
  for the release of a frame the GPU has not finished.
- The player's paused RAFT snapshot copies the lens planes as R8 / RG8, or R16 / RG16 for P010 (2 bytes per sample
  in the row stride).

## App (`crates/app`)

- One eframe window on `video::Gpu` (`WgpuSetup::Existing`). A side panel switches between Library (local files and
  player), Editor, Shot Lab, Camera (panes Album, Downloads, Camera, Wireless), Capture (live preview,
  `/CONF`-driven controls, shutter), Live (RTMP, OBS, PipeWire camera), Export, Settings (download folder, default
  export preset, audio output of the player and the editor's preview, thumbnail cache, GPU info, client id) and Help (keys, a guide per screen, About with
  licences). A camera battery at 20 % or less and not charging shows as a warning on every screen (`camera::low`).
- One module per screen: `library.rs` / `player.rs`; `editor.rs` (`deck.rs`, `sound.rs`, `thumbs.rs`, `doc.rs`);
  `lab.rs` (Shot Lab on crate shotlab); `camera.rs` (`album.rs`, `dl.rs`; `disk.rs`: a mounted card found through
  the mount table and the sysfs product string, listed from its trailers, imported as `file://` download entries;
  `bt.rs`: Bluetooth and station-mode Wi-Fi: join, a read of the camera's existing station connection over BLE, a
  typed camera address); `capture.rs` (worker on the camera session through `camera::Tap`;
  `conf.rs`: the `/CONF` model and its wire values; `feed.rs`: live preview decode thread, pictures handed over under
  one lock and taken as they fall due, paced on the camera clock from the slowest of the last 32 deliveries, IMU
  levelling; `webcam.rs`: Webcam Mode via `media::uvc` + MJPEG; `replay.rs`: link's simulated camera streaming a
  recorded preview, every loop from its first IDR (leading packets stamped 0 are sent again a frame apart)); `live.rs` (sources `live::Pick {Player, Preview, Webcam}`; `live::Source::tick(at)` takes the
  tick's scheduled `CLOCK_MONOTONIC` time); `export.rs` with the `photo.rs` and `movie.rs` viewers (`movie.rs`: plain videos such as our exports, NV12 or
  P010 to RGBA through BT.709 limited range); `ai.rs`;
  `settings.rs`; `help.rs`. Work runs on the threads in PLAN.md §2; the UI thread only reads the newest state and
  draws.
- AI (`ai.rs`): `ai::MODELS` loads one `track::Ai` per run and device on first use (warm-up one network call at a
  time under BURN), shared by Deep Track, auto edit and every face pass. Deep Track directions are in the frame of the
  track's stabilisation and are turned into any other frame with `Track::turn` / `ai::rots`. The privacy blur also
  covers editor timelines (one face pass per clip) and Shot Lab passes (before the effect).
- Editor projects are edit's JSON plus a top-level `"x5desk"` object (overlays, denoise, stabilisation). A clip's
  look (one of fx's looks by name or a `.cube` path) and colour adjustments (per clip `"adjust"`: the `fx::Params`
  fields that differ from the defaults) live in edit's clips; a file from before per-clip adjustments holds the
  timeline's in `"x5desk"."adjust"`, which load onto every clip without its own. The Colour tab's sliders edit the
  selected clip through edit's log without `changed()` (playback goes on). Denoise runs in the export
  (`export::Edit::denoise`) as in the preview: `media::denoise::Stereo` after the time-stretch, its delay dropped.
  The undo history interleaves edit's log with steps for those extras; a drag that ends where it started leaves
  both as they were, the steps undone included. The player and the editor share one RAFT worker (`player::still`).
- The editor's viewer redraws only from decks bound for the current edit (`schedule` runs before each redraw); while
  the playhead's frame is still decoding (seek, cut) the last picture stays.
- Audio output: the device picked in Settings is looked up off the UI thread (`settings::output`:
  `media::audio::device` on a thread per pick; a caller waits while the newest pick's lookup runs, so it is never
  called on the UI thread). The player opens its audio on it with `media::audio::Player::open_on` (a failing stream
  falls back to the system default; no AAC track or no device: the silent clock), and the live outputs follow the
  player's clock. The editor's preview plays on it through cpal directly (`editor/sound.rs`): its stream is looked
  up, opened and closed on the sound's own thread, one stream at a time (`sound::DEV`), and its clock holds at the
  start position until the device plays.
- Disk work stays off the UI thread: the Settings screen measures and trims the thumbnail cache on a thread of its
  own; the Downloads pane takes whether an import waits for its card from the camera worker's mounted volumes. The
  download queue file is written only when its text changes.
- Config: every save to `<config>/x5desk` makes that dir owner-only (`main.rs` `private`), also when it exists
  already; the `live` file (stream key) is 0600. A stream key in the RTMP URL (key field empty: the last path
  segment after the app) is drawn masked and handed to `live::rtmp` as the key, never inside the URL.
- Device policy (`main.rs` `GATE: RwLock`, `BURN: Mutex`): the app's own submitting threads hold `GATE` shared around
  each piece of GPU work (live output thread per pass, RTMP encoder per frame, export threads and decoders, photo
  exports); threads whose work runs through burn (RAFT worker, the AI models, shotlab's person network) hold `BURN`
  for a whole job instead. Lock order: GATE is dropped before BURN is taken; BURN never nests. A new submitter takes
  one of them.
- The app owns the device's uncaptured-error handler (`main.rs` `errors`). A `Surface::configure` that fails with
  `GpuWaitTimeout` (another thread submitted during its idle wait; native Wayland never reports the stale swapchain
  `Outdated`) is retried in the next painted frame: egui is made to reconfigure by flipping
  `desired_maximum_frame_latency` between `Some(2)` and `None`, and the UI thread holds BURN, then GATE, exclusively
  from the end of that frame's UI until egui has configured and acquired. wgpu-core drops the map / work-done
  callbacks a failed configure collected, so a `map_async` user treats a dropped callback as a finished map.
  Everything else panics. burn attaches once per device through `stitch::attach`, which installs cubecl's handler
  (and device-lost callback) on first attach only; the app calls it before installing its own, and RAFT, shotlab's
  segmentation and the one `track::Ai` in `ai::MODELS` reuse that attachment.
- Test hooks: `X5DESK_SHOT=<file.png>` (plus `X5DESK_SHOT_AFTER=<s>`) saves a screenshot of the window through egui's
  `ViewportCommand::Screenshot` and exits; the Capture screen is ready for it once the capture worker reported on the
  camera's session (after its `/CONF` attempt, which may fail). `X5DESK_TEST=<tokens>` drives the app; under it
  nothing of the user's config or cache is read, written or trimmed (app and camera settings, the recent list; the
  download queue is the run's own; no thumbnail cache: `settings::Prefs::thumbs`), and `dest=` is also the download
  folder. Tokens:
  - player: `play`, `loop`, `vol=`, `view=`, `seek=`, `proxy`, `stats=<s>`, `novsync`, `full`,
    `stab=off|free|horizon|direction`, `smooth=<s>`, `aim=<yaw>/<pitch>/<fov>`, `max`;
  - camera: `camera=album|downloads|settings|wireless`, `dl=<uri>`, `mount=<dir>` (repeatable fake card; the real
    mount table is not read), `import`, `ble=sim`, `ble=sta` (the simulated X5 on HomeNet from an earlier
    session: connect, pair, read its station connection; no join), `battery=<%>` (the `replay=` camera's level);
  - capture: `capture`, `replay=<rec.h264>`, `conf=<file>`, `webcam[=1920|<file.mjpg>]`, `cstats=<s>`, `yaw=<deg>`,
    `level=off`, `pace=off` (viewer only), `src=preview|webcam`;
  - live: `live`, `end` (Live screen scrolled to its end), `rtmp=<url>`, `res=`, `fps=`, `kbps=`, `obs`, `pwcam`,
    `grab=`, `min=`;
  - export: `export=<preset>[:faces]`, `frame=`, `dest=`;
  - editor: `editor`, `project=<file>`, `clip=<file>` (repeatable), `undo=<n>`, `redo=<n>`, `at=<s>`, `sel=<k>`,
    `etab=clip|reframe|colour|audio|overlays`, `e360`, `epreset=<preset>`, `esave=<file>`,
    `tlexport=<preset>[:noraft][:qp=<n>]`, `eplay`;
  - AI and Shot Lab: `box=<x0>/<y0>/<x1>/<y1>`, `restab=off|free|horizon|direction[/<σ>]`, `ebox=…`, `blur`,
    `autoedit[=<preset>[:faces]]`, `lab=<effect>`, `lrange=<a>/<b>`, `lpre`, `lexp=<preset>[:faces]`;
  - other screens: `settings`, `help[=keys|about|<screen>]`.

  Real-camera tests that change the camera's state and restore it: `X5_CAMERA_WRITE=<dir>`
  (`capture::tests::real_writes`), `X5_CAMERA_ILOG=<dir>` with `X5_ILOG_EXPO=<iso>,<shutter>`
  (`capture::tests::real_ilog`). With the display off (DPMS), Wayland sends no frame callbacks and windows stall: run
  tests under XWayland (`env -u WAYLAND_DISPLAY`).
- OBS plugin (`crates/obs`, `libx5desk_obs.so`, installed as `x5desk-obs.so`): source `x5desk_source`, reads the
  app's socket (`$XDG_RUNTIME_DIR/x5desk/obs.sock`) with `live::obs::Client`; it announces the modifiers OBS imports and
  sends `done` only after OBS's GPU fence for that buffer.
