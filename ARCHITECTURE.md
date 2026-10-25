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

## Crates

| Crate | Content |
|---|---|
| `video` | `Gpu` (one Vulkan device via wgpu-hal with decode and encode queues), H.265 / H.264 parsers, Vulkan Video decode and encode, NV12 textures for wgpu |
| `media` | MP4 demux and mux (`sv3d` / `st3d`), `.insv` / `.insp` trailer (calibration, gyro, exposure), AAC decode (symphonia) and AAC-LC encode (`aacenc`), editor mixer with ducking and limiter (`mix`), music import (`music`), resampler, RNNoise denoise, audio output and master clock (cpal), MJPEG decode, UVC capture of the X5's Webcam Mode |
| `proto` | Protobuf messages for the camera (prost, compiled with protox from `proto/`) |
| `link` | Camera link: UCD framing, session, requests, notifications, HTTP tunnel with several connections, downloads; USB (nusb), Wi-Fi (TCP); BLE lives in `ble` |
| `stitch` | Dual fisheye → equirect / perspective on the shared device from NV12 textures: template seam, RAFT flow, colour matching, stabilisation rotation |
| `stab` | Stabilisation maths: trailer gyro / accelerometer → per-frame orientation, smoothing (FlowState-like), horizon and direction lock, rolling-shutter terms |
| `fx` | Image effects on the shared device: colour adjustments (incl. brightness, brilliance), automatic colour and underwater correction, `.cube` LUTs and our own looks, defringe (purple / green / blue), clarity, overlays (watermark, stickers, text) |
| `edit` | Editor model: timeline, clips, trim / split / speed / freeze, reframing keyframes and camera-movement presets, undo, project files |
| `export` | Render jobs: decode → stitch → stabilise → effects → encode → mux, 360 and flat, queue; editor timelines (`Cut::Edit`: speed ramps, freeze frames, views, looks via `fx`, audio mixed with music and AAC-encoded, or copied for plain 1× cuts); photo frame grabs (JPEG with GPano) |
| `track` | AI with open models (burn on the shared device): subject tracker on the sphere with re-identification, detector, auto framing, face privacy blur regions, highlights → `edit::Timeline` |
| `ble` | Camera BLE: scan, wake, pairing, Wi-Fi credentials (btleplug) |
| `live` | FLV and RTMP client, OBS frame protocol (Unix socket + DMA-BUF fds), PipeWire camera node |
| `obs` | OBS source plugin (`cdylib`) that imports our frames |
| `app` | egui (eframe 0.37) UI, binary `x5desk` |

## Shared device contract (`video::Gpu`)

- `Gpu::new()` opens the Vulkan adapter through wgpu-hal with every adapter feature, the experimental features,
  `TEXTURE_FORMAT_NV12`, the video decode and encode queue families, and the external memory / semaphore
  extensions (`VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`,
  `VK_KHR_external_semaphore_fd`, `VK_EXT_queue_family_foreign`).
- Public fields: `instance`, `adapter`, `device`, `queue` (wgpu) and the raw Vulkan handles the video and live code
  need. The app hands the wgpu handles to eframe with `WgpuSetup::Existing`.
- Decoded pictures are NV12 `wgpu::Texture`s (plane views `R8Unorm` / `Rg8Unorm`, home state `RESOURCE`).
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
- Stitch sources are whole NV12 textures or rects in one (side by side: `.lrv` and live preview, lens 0 left; the
  `.lrv` needs `Frame::shift`); the preview also draws from an RGBA equirect (paused RAFT stitch, Webcam Mode).
- Encoder input: `Encoder::input` (`Arc<Input>`) is a ring of `encode::SLOTS` NV12 / P010 textures, home state
  `COPY_DST`; producers fill them by copying (wgpu has no NV12 / P010 storage). Before the first wgpu submit that
  writes frame `f` call `input.acquire(&gpu, f)` (`Err(Full)` after 50 ms if the encoder still holds the slot); after
  the last one call `input.release(&gpu, f)`, then hand `f` to the encoder thread, which calls `encode(&gpu, pts)` and
  `packet(&mut Packet, ns)`. Frames are released in order. `Input::key()` makes the next frame an IDR. `Cfg.vbv` is
  the rate control buffer in ms (1000 for files, 100 for low-latency RTMP). RADV: no B-frames, quality level 3 of
  H.265 under rate control writes corrupt slices (excluded by `Encoder::levels`).
- DMA-BUF export (`live::dmabuf`): `Image::acquire()` before the first submit that writes an exported image (content
  undefined afterwards), `Image::release()` after the last one; it returns the sync_file and also attaches it to the
  DMA-BUF as its write fence, so implicit-sync consumers (PipeWire clients) are correct too.
- A command encoder is either raw (`as_hal_mut`) or wgpu, never both.

## App (`crates/app`)

- One eframe window on `video::Gpu` (`WgpuSetup::Existing`). A side panel switches between Library (local files and
  player), Camera (link, album, downloads, settings), Capture (live preview, `/CONF`-driven controls, shutter), Live
  (RTMP, OBS, PipeWire camera) and Export.
- One module per screen (`library.rs` / `player.rs`, `camera.rs`, `capture.rs`, `live.rs`, `export.rs`). Work runs on
  the threads in PLAN.md §2 (media per open clip, audio clock, camera I/O, export); the UI thread only reads the newest
  state and draws.
- Test hooks: `X5DESK_SHOT=<file.png>` (plus `X5DESK_SHOT_AFTER=<s>`) saves a screenshot of the window through egui's
  `ViewportCommand::Screenshot` and exits. `X5DESK_TEST=<tokens>` drives the first opened file (`play`, `loop`,
  `vol=`, `view=`, `seek=`, `proxy`, `stats=<s>`, `novsync`, `full`, `stab=off|free|horizon|direction`, `smooth=<s>`,
  `camera=album|downloads|settings`, `dl=<uri>`). With the display off (DPMS), Wayland sends no
  frame callbacks and windows stall: run tests under XWayland (`env -u WAYLAND_DISPLAY`).
- The app owns the device's uncaptured-error handler (`main.rs` `errors`). burn attaches once per device through
  `stitch::attach`, which installs cubecl's handler (and device-lost callback) on first attach only; the app calls it
  before installing its own, and RAFT and `track::Ai` reuse that attachment.
- OBS plugin (`crates/obs`, `libx5desk_obs.so`, installed as `x5desk-obs.so`): source `x5desk_source`, reads the
  app's socket (`$XDG_RUNTIME_DIR/x5desk/obs.sock`) with `live::obs::Client`; it announces the modifiers OBS imports and
  sends `done` only after OBS's GPU fence for that buffer.
