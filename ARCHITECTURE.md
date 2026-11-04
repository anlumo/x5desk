# x5desk

Desktop app for the Insta360 X5: player, camera control, live output, export. Linux (Wayland, RADV) first; macOS (Metal, VideoToolbox) as well.
The plan with decisions, measurements and milestones is `../insta360/PLAN.md`; the camera protocol and file formats
are documented in `../insta360/re/` (`PROTOCOL.md`, `analysis/*.md`).

## Rules

- No FFmpeg. Video goes through Vulkan Video (Linux) or VideoToolbox (macOS) on the one shared device.
- No Insta360 data in this repo unless the code needs it to run (the `.proto` files the app uses). Lens calibration
  comes from each file's trailer, never from the repo. Seam flow uses RAFT (open model), no Insta360 models.
- The UI thread never paces playback. Media threads decode ahead of the audio clock.
- Tests that need sample footage read the directory in `X5_SAMPLES` (`v635.insv`, `v606.insv`, `v627.insv`,
  `v635.lrv`) and return early with a note when it is unset. `track`'s `tests/footage.rs` `subject` with `X5_REF` (a
  reference track checked by eye) asserts the box on the subject itself on half the frames, without it just a box.
  Track test results depend on the kernels burn's autotune picks (other kernels round VitTrack's confidence across
  its thresholds differently): a cold cubecl slot (App, burn's cache) tunes them under whatever else runs on the GPU.
- Tests on numbered clips (`clip::tests::numbered`, made with the ffmpeg CLI: the B-frame, open-GOP and 10-bit tests of
  the app's `clip.rs`, movie's B-frame test) skip with a note when ffmpeg is missing or cannot make the clip (an encoder
  or filter it lacks, drawtext without fontconfig).
- `link`'s `holes_fs` checks that a download on FAT or exFAT runs one connection in a dir on one named by
  `X5_NOHOLES` (on macOS a mounted `hdiutil create -fs ExFAT` image will do), without it only the temp dir's four.
- Tests that make paths in `$XDG_RUNTIME_DIR` name them `<name>-<pid>` and remove them through a drop guard, also
  when they fail. The test hook's burn cache is the exception: `x5desk-cubecl/<n>`, shared by every hook run on
  purpose, one process to a slot (`flock`), kept.
- Concurrent test runs share `X5_FIXTURES`, `X5_OUT` and the temp dir: stitch, fx, the player's proxy and live's
  `tests/rtmp.rs` write a fixture cached under a fixed name under a name of the process (and thread) and rename it into
  place whole, and live's `tests/rtmp.rs` names what its listeners receive `<pid>-<name>`; other fixed-name caches
  (stab's `.tracks`) are still written in place. Within one process, tests name their dirs apart: a
  `clip::tests::Tmp` (`x5desk-<name>-<pid>`) is removed whole when it drops, also under another test using that dir.
- The RTMP tests' `ffmpeg -listen` listeners (app `live.rs`, live `tests/rtmp.rs`) and `pw_gst`'s gst-launch are
  each held in a `Kid` drop guard, which kills and reaps the process also when the test fails; the listeners' `Kid`
  removes the file it received too, and gst-launch's output has a `Run` guard. `Kid` spawns with `PR_SET_PDEATHSIG`
  (SIGKILL), so a test process that is killed or aborts leaves none running. The signal follows the spawning thread:
  the test's own thread spawns and owns the child. Both take a listener's port from `bind(0)` (`serve`; a reconnect
  test's second listener takes the first one's again), trust the listener only once a listening socket on that port is
  one of ffmpeg's own fds (its inode among `/proc/<pid>/fd`), and move to another port, 5 in all, if ffmpeg exits first.

## Crates

| Crate | Content |
|---|---|
| `video` | `Gpu` (one Vulkan device via wgpu-hal with decode and encode queues; macOS: the Metal device), H.265 / H.264 parsers, Vulkan Video (macOS: VideoToolbox, `mac/`) decode and encode, NV12 / P010 pictures for wgpu (`Ring::planes`) |
| `media` | MP4 demux and mux (`sv3d` / `st3d`), `.insv` / `.insp` trailer (calibration, gyro, exposure), AAC decode (symphonia) and AAC-LC encode (`aacenc`), editor mixer with ducking and limiter (`mix`), music import (`music`), resampler, RNNoise denoise, audio output and master clock (cpal; `audio::Player::open_on` plays on a picked device, `audio::device(id)` resolves the settings' device id by exact match: it lists the devices, so resolve when the choice changes, never per open; macOS: the app's settings look a gone or missing picked device up again on the opening thread), `audio::Conv` to a stream's format (`Conv::on`: macOS the device's own rate and up to 2 channels, else the source's; the clock counts source frames), MJPEG decode, baseline JPEG encode (`jpeg`), UVC capture of the X5's Webcam Mode |
| `proto` | Protobuf messages for the camera (prost, compiled with protox from `proto/`) |
| `link` | Camera link: UCD framing, session, requests, notifications, HTTP tunnel with several connections, downloads (each connection writes its range at its offset of `<dst>.part`; on FAT and exFAT, which hold no holes and zero-fill up to an out-of-order write, one connection writes in order; Windows marks the file sparse first); USB (nusb), Wi-Fi (TCP); BLE lives in `ble`. Connections start as `link::Opening` (dropping it cancels the sync); `Error::Silent` after 30 s without a sync answer; `Camera::set_time` sends the app's two SET_OPTIONS at once and returns both pending answers; `scan` / `watch` report `Plug::Disk` for an Insta360 product string under vendor 070a (the X5 in U-disk mode, or the first seconds of app mode); feature `sim`: a simulated camera (a handler status 0 leaves the request unanswered) |
| `stitch` | Dual fisheye → equirect / perspective on the shared device from NV12 / P010 textures: template seam, RAFT flow, colour matching, stabilisation rotation; burn's attachment per device (`attach`) and its cache dir (`cache`: a slot of the process's own, else memory only) |
| `stab` | Stabilisation maths: trailer gyro / accelerometer → per-frame orientation, smoothing (FlowState-like), horizon and direction lock, rolling-shutter terms |
| `fx` | Image effects on the shared device: colour adjustments (incl. brightness, brilliance), automatic colour and underwater correction, `.cube` LUTs and our own looks, defringe (purple / green / blue), clarity, overlays (watermark, stickers, text) |
| `edit` | Editor model: timeline, clips with a look and colour adjustments each (`fx::Params`: edit depends on fx), trim / split / speed / freeze, reframing keyframes and camera-movement presets, undo (a drag that ends where it started leaves no undo step and keeps the redo list), project files |
| `export` | Render jobs: decode → stitch → stabilise → effects → encode → mux, 360 and flat, queue; editor timelines (`Cut::Edit`: speed ramps, freeze frames, views, each clip's adjustments and look via `fx`, audio mixed with music, RNNoise-denoised after the time-stretch when `Edit::denoise` is set, and AAC-encoded, or copied for plain 1× cuts without denoise); photo frame grabs (JPEG with GPano; the encoder is `media::jpeg`, also reachable as `export::jpeg`). `export::locks(gate, burn)` hands it the app's GATE / BURN; `Job::flat` keeps a flat 2:1 picture from a 360 hook untagged |
| `track` | AI with open models (burn on the shared device; macOS: the face detector through Core ML, `coreml.rs`): subject tracker on the sphere with re-identification (LOMO colour and texture histograms with Fisher weights learnt per session, no model), detector, auto framing, face privacy blur regions, highlights → `edit::Timeline`. The tracker finds a lost subject again only within its reach (`Seen::prior`, `Seen::speed`: the speed, in its own heights a second, it needs to be a candidate since it was last seen for sure, its heading carried on for up to a second; beyond what it reaches setting off at a walk and speeding up by 3 heights/s² to a sprint's 3, it is someone else; within it, its similarity loses 0.1 × (speed / 1 height a second)²), and a person only with the pick's template's box on it (`vit::hit`: never on the detector's box alone); a person at least as like one `find` took and the tracker lost again before VitTrack was sure of them as like the subject (`vit::Tried`, the last 8; not added to the gallery, as the subject can be one) needs VitTrack 0.5, not 0.25 (`Tried::need`, also for whoever the tracker takes on its own after a gap). `Tracker::fps` is the clip's frame rate (30 until set) |
| `ble` | Camera BLE: scan, wake, pairing, Wi-Fi credentials (btleplug). The X5's wake-up brings up its radios and session, never its screen. Feature `sim`: `Ble::sim`, a simulated adapter and camera on the real session code ; macOS: `Error::Off` / `Error::Denied` and `Event::Adapter` (Bluetooth off, not allowed, back) |
| `live` | FLV and RTMP client, OBS frame protocol (Unix socket + DMA-BUF fds), PipeWire camera node (the last three Linux only: elsewhere `stub.rs` keeps their API and refuses to start) |
| `obs` | OBS source plugin (`cdylib`) that imports our frames |
| `cam` | The app's camera code without its UI, so the app and a headless host run the same session code: the camera worker (`camera`: session, clock, album and thumbnails, download queue, commands, mounted cards) and the capture worker (`capture`: mode, settings and shutter from `/CONF`), with `album`, `dl`, `disk`, `conf`, `work` (a recording's files), `dirs` (config, cache and video folders; on Windows the profile's), `awake`, `mono`. The workers guard what any host sends: no delete while the camera records or captures (`Info::busy`, and the capture status read before the delete), a setting only to a value `/CONF` allows now (`St::allowed`), no start of a recording or timelapse at 5 % battery or less unless charging (`camera::empty`), an explicit `Start` / `Stop` (the camera's capture state read first, nothing sent when it already is so); a connect sets the camera clock only when asked to (`Cmd::Connect { clock }`, the app's own always; its retries and its reconnects after a lost link, by the 2 s timer or the camera back on the bus, the same). Outside Linux and macOS: no card found (a card source given still lists, its URIs `/`-separated); not unix: the camera clock never set (no IANA zone known; said once), the favourite timestamp from the system's zone rules (Windows: `GetTimeZoneInformation`). Feature `sim`: a simulated X5 with a fake card for the tests above it. No egui, wgpu or video |
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
  decoder that finds its slot blocked by a release still staged sends it with an empty submit itself. A ring holds
  `Ring::slots()` frames, `decode::OUT` (4) unless made by `Decoder::with_out(…, n)` (n ≥ OUT); frame `f` sits in
  slot `f % slots()`, and consumers bind per slot.
- H.264 input may miss whole access units (the live preview drops packets when the decode thread falls behind):
  frame_num gaps infer non-existing frames (8.2.5.2) that stand in with the pushed-out picture, and an IDR always
  recovers a full DPB. Loss shows as concealment until the next IDR, never as a refused stream.
- A consumer that seeks calls `Decoder::restart()` before the access unit decoding restarts at: the DPB is emptied,
  the next decode resets the video session (VCN keeps state of its own: with only the DPB emptied, H.264 pictures some
  frames after a restart came out wrong), and the POC / frame_num state starts over, as in a new decoder; an H.265 CRA
  there is handled as a BLA (HandleCraAsBlaFlag). Leading pictures that may reference pictures from before where
  decoding started or restarted return `Error::Skip`: nothing decoded, no frame index used. These are the RASL
  pictures of an H.265 IRAP with NoRaslOutputFlag (that CRA, a BLA, a CRA first in the stream or after an end of
  sequence) and the H.264 pictures shown before the non-IDR I picture decoding started or restarted at (a recovery
  point; libx264's `open-gop` marks such I pictures as MP4 sync samples); the H.264 ones are still marked, a
  reference as a non-existing frame.
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
- Freeing GPU memory while decodes are queued holds the process's graphics back (amdgpu): the page-table update that
  unmaps it waits for every submission queued in the process's VM, decodes included, and later graphics submissions
  wait for that update (another process's decodes are not in our VM). During playback the decode queue never drains
  (two 3840² lenses keep a VCN busy), so memory freed then delays the UI's frames by the backlog
  (`vkAcquireNextImageKHR` waits; 40–190 ms at the editor's cuts, where a deck is dropped while the next one decodes
  from its keyframe). So a dropped `Decoder` parks its session, DPB, bitstream and command buffers, and its `Ring`
  its textures, in `Gpu`'s bin (`decode::Bin`, two of each per shape: codec, profile, bit depth, size, DPB slots,
  ring size; freed with the `Gpu`), and the next decoder of the shape takes them (new timelines, frames from 0):
  nothing may sample a ring's textures after its last `Arc` goes. A new `Decoder` grows its command buffers' upload
  memory before any decode (RADV frees outgrown upload buffers at a command buffer's next reset).
- The player's paused RAFT snapshot copies the lens planes as R8 / RG8, or R16 / RG16 for P010 (2 bytes per sample
  in the row stride).

## macOS (`video::mac`)

- wgpu runs on Metal (`Gpu::new`: every adapter feature, the experimental ones; needs `TEXTURE_FORMAT_16BIT_NORM`).
  burn compiles its kernels to MSL (stitch's manifest picks burn's `metal` feature there, `vulkan` on Linux). Metal
  compiles shaders with fast math (wgpu leaves the default): `pow` is approximate, ≈ 4 10-bit levels in fx's `blend`.
- wgpu's Metal backend has no NV12 / P010 textures, so a decoder slot is two textures, luma R8 / R16 and chroma RG8 /
  RG16 (half size). Consumers take a frame's planes through `Ring::planes(f)` (views), `Ring::plane(f, i)` (copies)
  and `Ring::textures(f)` (transitions), which both backends have; `Ring::texture(f)` (one NV12 / P010 texture) is
  Linux only. `stitch::Ctx::rect` takes the two views; `stitch::views` makes them from an NV12 / P010 texture.
- Decode: the parameter set parsers still do picture management (POC, references, `Error::Skip` for leading
  pictures, the same as on Linux), VideoToolbox decodes every other picture synchronously into an IOSurface of the
  stream's own range and depth (no conversion), and a copy on wgpu's queue moves its planes into the slot. All of it
  is on wgpu's one queue, which Metal orders by resource, so `acquire` stages nothing and `release` is a counter the
  decoder waits on (`Full` after 50 ms as on Linux). `restart` makes a new session. A picture VideoToolbox fails on
  still takes its frame and shows the frame before it (a copy of that slot: concealment, as on Linux;
  `Decoder::failed` counts them); with nothing decoded since the start or a `restart` it is an error. The next random
  access point (H.264 IDR or I picture, H.265 IRAP) after a failure gets a new session; while VideoToolbox has nothing
  to decode against (its session invalidated, -12903; a random access point failed), the pictures before the next one
  are concealed without decoding. Each copy flushes the CVMetalTextureCache (`Gpu::flush`; CoreVideo evicts only
  then). `Decoder::took`: the decode's own time once it has its slot (parsing, VideoToolbox, the copy), which the
  preview's pacing and decode stats use in place of Linux's done timeline. The parsers step over scaling lists here
  (VideoToolbox applies them; its own HEVC encoder writes them), rejecting a delta_scale outside −128..=127; on Linux
  they stay refused.
- `Decoder::gated(&GATE)`: the decoder holds GATE shared around its copy submits only, not the decode, waiting for it
  at most 50 ms (a GATE holder may be joining its thread). For consumers that decode outside GATE (the live preview,
  `clip::Video`); one that holds GATE around `decode` (crate export, shotlab's frames) must not, since std's RwLock
  queues a reader behind a waiting writer.
- Encode: each input slot is a CVPixelBuffer whose planes wgpu writes through Metal textures over the IOSurface;
  `copy_buffer` copies stitch's YUV buffer with a render pass (any even pitch; the buffer needs STORAGE usage).
  `release` stages an MTLSharedEvent signal (`filled`) on wgpu's queue as Linux stages its semaphore; `encode` waits
  for it, forces IDRs where the GOP or `Input::key` wants them, frame reordering off (dts = pts). `params` come from a
  picture a throwaway session encodes in `new`; each session's first sample must carry them, or `packet` fails.
  Hardware encoders only: H.264 7680 × 3840 fails on the M4 rather than run in software (about 1 fps). Every
  session is RealTime: off, as Apple advises for files, an 8K H.265 Main 10 stream encodes at 16.6 instead of 30.3
  fps on the M4 for 0.2 dB (`tests/mac.rs` `speed`). A `Cfg.vbv` under 1000 is a low-latency stream (RTMP); a file's
  session is asked to hold back one frame, an RTMP one none, but the M4's encoders don't take MaxFrameDelayCount (and
  hold nothing back with frame reordering off): `encode` makes one that holds `SLOTS` − 1 emit them, or the
  producer's `acquire` would wait for good, and `packet` makes a file's emit what it holds after 20 ms of waiting; a
  session invalidated with frames held back loses them there. `Rate::Cbr` is AverageBitRate with DataRateLimits
  of two buffers per `vbv` window (one undershoots by 30 %), not ConstantBitRate (it pads, overshoots 32 % with
  DataRateLimits, and its VBVBufferDuration changes no IDR); no mode caps an IDR at the buffer. `Rate::Qp` is
  VideoToolbox's min = max frame QP. `levels` is 1. `Packet::key` is the sample's sync flag. A frame VideoToolbox
  drops, or loses with an invalidated session, comes back without data and never as key, and the next frame is an IDR
  if the lost one was to be: RTMP sends no tag for it, `Mux::write` lengthens the previous sample. kVTInvalidSessionErr
  (sleep, a media server restart) makes `encode` start a new session at that frame with an IDR.
- USB app mode: macOS sets a device's configuration only for a driver of its own, and none takes the X5's app mode
  (`2e1a:0002`), so it stays unconfigured, without interfaces. `link::usb`'s `plug` counts such a device as
  `Plug::App` there, and `Usb::open` sets the configuration with the vendor interface 0 before claiming it (Linux's
  kernel sets it itself). The X5 gives up after about 15 s without a session ("switching mode failed" on its screen)
  and goes back to `070a:4027`: the app has to be running while the camera switches.
- BLE (btleplug's CoreBluetooth backend): a `Dev` is the peripheral's UUID. btleplug forgets a peripheral at every
  disconnect (also a failed or timed-out connect); a connect that misses it in `peripherals()` takes it back with
  `retrieve_peripherals` (`retrievePeripheralsWithIdentifiers`), so a listed camera reconnects without a scan.
  CoreBluetooth accepts a scan or connect in any state, so both check the adapter state first: `Error::Off`, or
  `Error::Denied` (btleplug folds unauthorized, unsupported and resetting into one state; `Ble::open` maps its
  PermissionDenied to it too). Each state update arrives as `Event::Adapter` and forgets the scan; a scan or find
  that was running starts again once Bluetooth is back (CoreBluetooth never resumes one), and so does one the
  Wireless pane asked for while it was off or not available (`bt.rs` `again`). Notifications and adapter events come
  through broadcast channels of 16 that drop what overflows: a pump task per link moves notifications to an unbounded
  channel as they come (also while the link writes) and asks `is_connected` every 2 s, since a lost
  DeviceDisconnected would leave the link up.
- Audio: cpal's CoreAudio backend sets a device's physical format, else its nominal rate, to the rate a stream asks
  for, for every app and for good, and fails on a rate the device lacks. Streams therefore open at the device's own
  rate and up to 2 of its channels (`media::audio::Conv::on`, from `default_output_config`); the player resamples its
  AAC in the decode thread and the editor its 48 kHz mix, and stream frame n is source time n / rate, so the clocks
  keep source time. A cpal device is a CoreAudio object id, which lasts while the device is connected:
  `settings::output` checks the kept device's UID and looks a picked device up again on the opening thread when it is
  gone or was missing at its pick (under 1 ms); the editor's preview falls back to the system default as the player
  does. CoreAudio pauses a stream on a picked device that goes away or changes rate under it: the player's error
  callback then hands its clock to a stand-in on wall time (`Shared::lost`, from a seek to where it stood), so
  playback and seeks go on in silence (on Linux too).
- Faces: `track`'s face pass runs YuNet through Core ML on the Neural Engine and the CPU (`track/src/coreml.rs`,
  `models/yunet_b8.mlpackage`: batch 8, FP16, made from the ONNX by `models/coreml.py`; `models/README`), leaving the
  GPU to the stitcher. The compiled model is kept in `x5desk-coreml` in the temp dir (made 0700; used only if it is
  this user's and nobody else can write to it, else compiled into a dir of the process's own), one per package (its
  hash), compiled beside it and renamed into place by the first process that needs it, a stale one moved aside first;
  Core ML keeps its own Neural Engine cache under `~/Library/Caches/<process name>`, test-hook runs too. The first face
  pass loads it (≈ 0.6 s, against burn's 11 s of kernel compiles). The crops reach it through a mapped copy of the crop
  pass's buffer; its scores and boxes come back in burn's layout, so the decoding after is the same. Every Core ML call
  runs in an autorelease pool (the app's threads have none: a prediction autoreleases 12 MB of outputs). M4, footage
  medians (`tests/footage.rs` `privacy`, `X5_YUNET=burn` for burn's): 42 ms per frame of 20 tiles instead of 80, a run
  of up to 8 tiles with copies, read-back and decoding 11.5 ms instead of 24 (Core ML's prediction of 8 alone: 9 ms),
  the same faces and tracks. `Ai::coreml` (on) picks it, `Ai::coreml_used` tells whether it runs; a model that won't
  load, or a run it fails, leaves burn's YuNet, which the other platforms run.
- App Nap: long work holds an `awake::Hold` (one ref-counted NSProcessInfo activity, user initiated, idle system sleep
  allowed; a no-op elsewhere): each job on the export thread (exports, frame grabs, Shot Lab exports), Deep Track and
  auto edit, Shot Lab previews, the live outputs while one runs (RTMP, a watched OBS output, the PipeWire camera; not
  the Live screen's preview alone), and the running download or card import (`dl::Cur`).
- Keys: egui's `command` is Cmd here (egui-winit), so Help and the editor's tooltips name Cmd for the shortcuts, Ctrl
  elsewhere (`help.rs` `cmd!`); Ctrl-click is a plain click. The key Mac keyboards label delete is Backspace to winit
  (fn+delete is Delete): the editor's clip delete takes both. wgpu's Metal adapter reports no driver: Settings' Driver
  row and About's Graphics line show the backend then (`settings::driver`).
- Elsewhere in the app: Webcam Mode (V4L2 UVC), OBS, the PipeWire camera and the BLE wake advertisement (BlueZ) are
  Linux only. Mounted cards come from the kernel's mount table (`getfsstat`, `MNT_NOWAIT`: no file system is asked,
  so a share whose server went away holds nothing up): block devices under `/Volumes` that Finder shows (not
  `nobrowse`) with a `DCIM` tree, only then looked at. The camera's is the one whose USB device (the first IOKit
  registry entry up from its IOMedia with `idVendor` and a product string) calls itself `Insta360 …`, as Linux's
  sysfs test (registry only, no I/O on the disk). Config and cache default to `~/Library/Application Support` and
  `~/Library/Caches` (the XDG variables still win), downloads and exports to `~/Movies`.
- Bundle: `cargo packager --release` (cargo-packager 0.11, `[package.metadata.packager]` in `crates/app/Cargo.toml`)
  builds the release binary and makes `target/release/x5desk.app` and `x5desk_<version>_aarch64.dmg`. Info.plist:
  cargo-packager's keys (identifier `io.github.anlumo.x5desk`, versions, a UTC time stamp as the build number,
  `LSMinimumSystemVersion` 15.0 for the Core ML package, the video category) with `crates/app/macos/Info.plist` merged
  over them: the usage strings for Bluetooth (without it macOS kills the app at its first scan), the local network (the
  camera's Wi-Fi, RTMP on the LAN), removable volumes, Documents, Downloads and Desktop. Signed with the hardened
  runtime and a timestamp by `macos.signing-identity`, the maintainer's Apple Development certificate: macOS's
  privacy grants follow the signature, which an ad hoc one (`-`) changes with every build. A Developer ID identity and
  Apple credentials in the environment sign it for anyone and notarize it. The disk image's window is laid out by
  Finder through AppleScript, which asks for Automation permission once; `CI=true` skips that, as where nobody can
  answer (a locked screen). Launched from Finder its stdout and stderr go nowhere.
- Icon: `crates/app/icon/icon.swift` (CoreGraphics, run by hand; needs Xcode's actool) renders it and writes what is
  checked in beside it: `Assets.car`, the asset catalog Info.plist's `CFBundleIconName` names (macOS 27 shows an icon
  that only comes as an icns small on a grey plate at 16 and 32 px on a 1x screen), `AppIcon.icns` for what reads no
  catalog (the dmg's volume), and `x5desk.png`, the window icon (X11, Windows; winit 0.30 takes none on Wayland, where
  the compositor would go by the app id's desktop file, which the repo has none of) and a bare binary's Dock icon.
  eframe puts egui's logo in the Dock unless given one; inside a bundle (`NSBundle` has an identifier) the app gives it
  the empty default instead, so the catalog's icon stays, rendered as Finder shows it.
- Opening files: Info.plist declares what the Library opens by extension (the system's MP4 and JPEG types also take
  in `.m4v` and `.jpe`): `.insv`/`.lrv` and `.insp` as Insta360 Studio does (Editor, the default rank, so the user
  picks which app opens them), `.mp4`, `.jpg` and `.jpeg` as Viewer, Alternate (offered in Open With, never their
  default). Finder, the Dock and `open` send them as `kAEOpenDocuments` Apple events, which winit
  0.30 leaves to AppKit, and AppKit has nobody to give them to. `finder.rs` replaces that one handler from
  NSApplicationWillFinishLaunchingNotification, the only point that gets the launch's files too (AppKit installs its
  own handlers just before, and dispatches the launch's event just after), queues the paths and wakes the UI, which
  takes them each frame as it does dropped files, un-minimising the window.
- Tests: `video`'s `tests/mac.rs` round-trips synthetic pictures through encode and decode (H.264, H.265, Main 10,
  RTMP's low-latency settings) with key requests, checked with `ffprobe`; `cbr` checks the rate within 10 % at 100 ms
  and 1 s and no filler, `renew` an invalidated session (`Encoder::invalidate`, a hidden hook), `hardware` every
  export preset on the media engine and no software H.264 at 8K. `tests/mac_decode.rs` decodes hand-made H.264
  (Baseline, I_PCM, one luma value per picture): a failed picture concealed with the one before, never the slot's old
  content or a pre-restart picture; the gate; scaling lists cut short or out of range rejected. Vulkan-only tests are
  `cfg(target_os = "linux")`. Test tools that read `/proc` (the RTMP listeners' readiness) use `lsof` elsewhere;
  `PR_SET_PDEATHSIG` is Linux only.

## App (`crates/app`)

- One eframe window on `video::Gpu` (`WgpuSetup::Existing`). A side panel switches between Library (local files and
  player), Editor, Shot Lab, Camera (panes Album, Downloads, Camera, Wireless), Capture (live preview,
  `/CONF`-driven controls, shutter), Live (RTMP, OBS, PipeWire camera), Export, Settings (download folder, default
  export preset, audio output of the player and the editor's preview, thumbnail cache, GPU info, client id) and Help (keys, a guide per screen, About with
  licences). A camera battery at 20 % or less and not charging shows as a warning on every screen (`camera::low`).
- One module per screen: `library.rs` / `player.rs`; `editor.rs` (`deck.rs`, `sound.rs`, `thumbs.rs`, `doc.rs`);
  `lab.rs` (Shot Lab on crate shotlab); `camera.rs` (its worker, `cam::camera`, connects in the app's order, PROTOCOL.md
  §5.1: identity, options, capture status, then the clock's two SET_OPTIONS (§5.3) on every USB and Wi-Fi connect, not
  waited for: polled in its tick; once both were answered, a refusal too, the options are read again, so the Clock row
  shows the clock as the camera has it (set, even when only the zone was refused); not when an answer never came or
  came garbled (timeout: GET_OPTIONS would hold the worker; link lost; undecodable reply); then the connect note and a
  low-battery note come; a refusal or timeout adds an error note and the session stays; then the Capture screen's tap
  and the listing; Sync time takes the same path; at quit only the running download's end is collected (`dl::end`),
  nothing new starts; `cam`'s `album`,
  `dl`; `disk`: a mounted card found through the mount table and the sysfs product string, listed with one
  trailer per item (its first file; empty files left out), imported as `file://` download entries; the bar takes
  U-disk mode from a mounted disk of the camera's own, else from the scan's `Plug::Disk` (`070a:4027`, also the first
  seconds of app mode); `bt.rs`: Bluetooth and station-mode Wi-Fi: join, a read of the camera's existing station
  connection over BLE, a typed camera address; the scan row says when Bluetooth is off or not allowed, macOS: System
  Settings > Privacy & Security > Bluetooth); `capture.rs` (worker `cam::capture` on the camera session through
  `camera::Tap`; its reports carry the session (`Tap::sid`) they belong to, and the screen drops those of an earlier
  one; the shutter sends a photo, or the start or stop its button shows; `cam::conf`: the `/CONF` model and its wire
  values; `feed.rs`: live preview decode thread, pictures handed over under one lock and
  taken as they fall due, paced on the camera clock from the slowest of the last 32 deliveries, IMU levelling;
  `webcam.rs`: Webcam Mode via `media::uvc` + MJPEG; `replay.rs`: link's simulated camera streaming a recorded
  preview, every loop from its first IDR (leading packets stamped 0 are sent again a frame apart)); `live.rs`
  (sources `live::Pick {Player, Preview, Webcam}`; `live::Source::tick(at)` takes the tick's scheduled
  `CLOCK_MONOTONIC` time); `export.rs` (an output name is numbered past one on disk, one whose `.part` a job in any
  process is writing, and one a queued job will write) with the `photo.rs` and `movie.rs` viewers (`movie.rs`: plain
  videos such as our exports or B-frame MP4s, NV12 or P010 to RGBA through BT.709 limited range); `ai.rs`;
  `settings.rs`; `help.rs`.
  Work runs on the threads in PLAN.md §2; the UI thread only reads the newest state and draws.
- AI (`ai.rs`): `ai::MODELS` loads one `track::Ai` per run and device on first use (warm-up: the tracker and the
  detector at every batch size, one network call at a time under BURN; re-identification runs no network), shared by
  Deep Track, auto edit and every face pass. Deep Track sets `Tracker::fps` from the clip (`shotlab::Clip::meta.fps`);
  its directions are in the frame of the track's stabilisation and are turned into any other frame with
  `Track::turn` / `ai::rots`. The privacy blur also covers editor timelines (one face pass per clip) and Shot Lab
  passes (before the effect).
- Editor projects are edit's JSON plus a top-level `"x5desk"` object (overlays, denoise, stabilisation). A clip's
  look (one of fx's looks by name or a `.cube` path) and colour adjustments (per clip `"adjust"`: the `fx::Params`
  fields that differ from the defaults) live in edit's clips; a file from before per-clip adjustments holds the
  timeline's in `"x5desk"."adjust"`, which load onto every clip without its own. The Colour tab's sliders edit the
  selected clip through edit's log without `changed()` (playback goes on). Denoise runs in the export
  (`export::Edit::denoise`) as in the preview: `media::denoise::Stereo` after the time-stretch, its delay dropped.
  The undo history interleaves edit's log with steps for those extras; a drag that ends where it started leaves
  both as they were, the steps undone included. The player and the editor share one RAFT worker (`player::still`).
- `clip::Video` (the player's, the editor decks' and the movie viewer's video) numbers frames in presentation order
  (`times` ascending, `Pic::k`). With B-frames the media thread holds a decoded frame back until those shown before
  it are queued, and a frame's release stops below the lowest decoder frame of the frames after it (`Pic::r`): a
  ring's release is one timeline in decode order. A seek starts at the last sync sample at or before the target in
  decode order that is not shown after it (an open GOP's leading frames need the GOP before). A seek that restarts
  decoding there (`Plan::Go`) calls `Decoder::restart` on each decoder: the restart's leading frames (a CRA's RASL
  frames, those before an H.264 non-IDR I picture) show before it, below the target, and their `Error::Skip` is no
  frame, no error. A clip whose first sample has leading pictures the decoder skips (a stream copy cut at an open
  GOP's CRA or non-IDR I picture) leaves those samples out of its frames (`times`); the media thread still hands them
  to the decoder. A clip's decoders get as many ring frames as reordering keeps decoded for a consumer holding one
  frame (2 without B-frames, 4 for libx264 `-bf 1`, 7 for its defaults) plus 2 to decode ahead (`Decoder::with_out`).
- The editor's viewer redraws only from decks bound for the current edit (`schedule` runs before each redraw); while
  the playhead's frame is still decoding (seek, cut) the last picture stays. For a frame it takes the deck holding
  that frame (no seek pending) first, then one a few frames before it (decoding on), then any deck of the file
  (seeking).
- Timeline drags (clip move, trim handle, speed key, volume point, view key) name their target by index: an edit that
  is not the drag's own ends the drag first, with its undo step (`Editor::halt` from `ed`, `begin` and `changed`; the
  drag's own edits go through `Editor::own`). Escape cancels a clip move (a drop is a release). A timeline that gets
  shorter keeps the playhead on it. Inspector slider drags (Hold, In / Out, Speed and its keys, Intensity, the Colour
  adjustments as one, Volume and its points, music Gain and offset: `Editor::slide`) are editor drags too
  (`Drag::Slider`): another edit ends them with their undo step, and the end egui reports for them in that frame
  edits nothing; a slider drag whose widget went away ends a frame after the button's release.
- Audio output: the device picked in Settings is looked up off the UI thread (`settings::output`:
  `media::audio::device` on a thread per pick; a caller waits while the newest pick's lookup runs, so it is never
  called on the UI thread; a lookup that panics still counts as done, through a drop guard). The player opens its
  audio on it with `media::audio::Player::open_on` (a stream that fails to build or start, an `io::Error` holding a
  `cpal::Error`, falls back to the system default; any other error, such as no AAC track, a file the decoder refuses
  or no device, gives the silent clock), and the live outputs follow the player's clock. The editor's preview plays
  on it through cpal directly (`editor/sound.rs`): its stream is looked up, opened and closed on the sound's own
  thread, one stream at a time (`sound::DEV`; a Sound dropped during its mixer's setup opens no more files), and its
  clock holds at the start position until the device plays.
- Disk work stays off the UI thread: the Settings screen measures and trims the thumbnail cache on a thread of its
  own; the Downloads pane takes whether an import waits for its card from the camera worker's mounted volumes, and an
  import whose card is mounted without its file fails ("not on the card"). The Library looks for the files of its
  Recent and Downloaded lists on a thread every 2 s and greys out the ones not there (on a card that is not mounted),
  keeping them out of the editor's media, until they are back. The download queue file is written only when its
  text changes.
- Config: every save to `<config>/x5desk` makes that dir owner-only (`cam::dirs::private`), also when it exists
  already; the `live` file (stream key) is 0600. A stream key in the RTMP URL (key field empty: the last path
  segment after the app, trailing '/' ignored as `live::rtmp::Rtmp::start` does, whitespace around the URL trimmed by
  the app first: `live::dest`, `live::mask`) is drawn masked and handed to `live::rtmp` as the key, never inside the
  URL. `Rtmp::start`'s errors quote the URL without the key; given no key and no segment after the app, it fails with
  "no stream key", quoting the server only (`<scheme>://<host>[:<port>]`).
- Device policy (`main.rs` `GATE: RwLock`, `BURN: Mutex`): the app's own submitting threads hold `GATE` shared around
  each piece of GPU work (live output thread per pass, RTMP encoder per frame, export threads and decoders, photo
  exports; macOS: `clip::Video`'s and the live preview's decoders around their copies, `Decoder::gated`); threads whose work runs through burn (RAFT worker, the AI models, shotlab's person network) hold `BURN`
  for a whole job instead. Lock order: GATE is dropped before BURN is taken; BURN never nests. A new submitter takes
  one of them.
- burn's cache: burn keeps its device measurements and autotune results (burn feature `persistence`) in a slot of
  `<cache>/x5desk/cubecl` (`stitch::cache(Option<&Path>)`, called in `main` before `stitch::attach`). Test-hook runs
  share `$XDG_RUNTIME_DIR/x5desk-cubecl` (the temp dir without it), so they start warm too, if it is a dir of this
  user's, not a symlink, that no other user can write to (`cam::dirs::mine`, made 0700 when missing); any other burn
  user (tests) gets cubecl's own root from `stitch::attach` (`target/environment` of the outermost Cargo workspace
  above the working dir, else the user cache dir). A slot is the first `<n>` (made 0700) whose `lock` the process can
  `flock`, never through a symlink another user planted (the lock made with `O_EXCL`, else taken only if `lstat` shows
  a plain file): cubecl-environment 0.11's database (turso 0.8, experimental multi-process WAL) loses acknowledged
  writes, reads corrupt pages and panics when processes share it. The lock goes with the process however it ends, so
  slots are reused and never outnumber the processes that ran at once (nothing trims them). With no dir (a hook dir
  refused, neither `XDG_CACHE_HOME` nor `HOME`) or no slot locked (no locks on that file system, no write access),
  cubecl's root is `/dev/null`, where its database can't be opened, and cubecl keeps it all in memory. Build records
  are off. With a warm slot RAFT's first job only compiles its kernels (≈ 2 s instead of ≈ 10 s on a quiet GPU; 4–9 s
  instead of 22–46 s under other load). A configure retry (below) that lands in a burn job still waits for the job's
  end, the UI frozen meanwhile.
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
  download queue is the run's own; no thumbnail cache: `settings::Prefs::thumbs`; burn's cache is the hook's shared
  `x5desk-cubecl` dir, or memory only; macOS: Core ML keeps its Neural Engine cache in `~/Library/Caches/x5desk`
  anyway, named after the process, and the face model's compiled form is the temp dir's shared one), and `dest=` is
  also the download folder. Tokens:
  - player: `play`, `loop`, `vol=`, `view=`, `seek=`, `proxy`, `stats=<s>`, `novsync`, `full`,
    `stab=off|free|horizon|direction`, `smooth=<s>`, `aim=<yaw>/<pitch>/<fov>`, `max` (`play`, `seek=` and `vol=`
    also drive the viewer of plain videos, `movie.rs`. `max` maximises the window, asked every 250 ms whenever it is
    not maximised, 12 times in all (about 3 s): an X11 window manager ignores a maximise for a window it has not mapped
    (eframe maps it after its first frame), and COSMIC un-maximises it when another window opens on the workspace.
    Under XWayland on COSMIC's tiling the window is otherwise tiled, 602 × 996 here; with `max` it fills the output's
    work area, 2488 × 1012 on a 2560 × 1080 output. The screenshot is taken in a frame the window is maximised in;
    once the asks are spent the run logs the size it got and goes on);
  - camera: `camera=album|downloads|settings|wireless`, `dl=<uri>`, `mount=<dir>` (repeatable fake card; neither the
    mount table nor what is mounted, `/Volumes` on macOS, is read: `disk::Watch::hook`), `import`, `ble=sim`, `ble=sta` (the simulated X5 on HomeNet from an earlier
    session: connect, pair, read its station connection; no join), `battery=<%>` (the `replay=` camera's level);
  - capture: `capture`, `replay=<rec.h264>`, `conf=<file>`, `webcam[=1920|<file.mjpg>]`, `cstats=<s>`, `yaw=<deg>`,
    `level=off`, `pace=off` (viewer only), `src=preview|webcam`;
  - live: `live`, `end` (Live screen scrolled to its end), `rtmp=<url>`, `res=`, `fps=`, `kbps=`, `obs`, `pwcam`,
    `grab=`, `min=`;
  - export: `export=<preset>[:faces]`, `frame=`, `dest=`;
  - editor: `editor`, `project=<file>`, `clip=<file>` (repeatable), `undo=<n>`, `redo=<n>`, `at=<s>`, `sel=<k>`,
    `etab=clip|reframe|colour|audio|overlays`, `e360`, `epreset=<preset>`, `esave=<file>`,
    `tlexport=<preset>[:noraft][:qp=<n>]`, `eplay`;
  - AI and Shot Lab: `box=<x0>/<y0>/<x1>/<y1>`, `restab=off|free|horizon|direction[/<σ>]`, `ebox=<x0>/<y0>/<x1>/<y1>`,
    `blur`, `autoedit[=<preset>[:faces]]`, `lab=<effect>`, `lrange=<a>/<b>`, `lpre`, `lexp=<preset>[:faces]` (`box=`
    is 0..1 of the player's view at 16:9, as a flat 1080p export frames it, whatever the window's size or layout; the
    hook prints its subject, `x5desk: box: subject at yaw …, pitch …, <w>° × <h>°`, so runs can be compared; `ebox=`
    is 0..1 of the `epreset=` export preset's flat frame, whatever the window: a flat preset's own, as the viewer
    letterboxes it, 16:9 for a 360 one as for `box=`; it prints its subject as `x5desk editor: box: …`);
  - other screens: `settings`, `help[=keys|about|<screen>]`.

  Real-camera tests that change the camera's state and restore it: `X5_CAMERA_WRITE=<dir>`
  (`capture::tests::real_writes`), `X5_CAMERA_ILOG=<dir>` with `X5_ILOG_EXPO=<iso>,<shutter>`
  (`capture::tests::real_ilog`). With the display off (DPMS), Wayland sends no frame callbacks and windows stall: run
  tests under XWayland (`env -u WAYLAND_DISPLAY`).
- OBS plugin (`crates/obs`, `libx5desk_obs.so`, installed as `x5desk-obs.so`): source `x5desk_source`, reads the
  app's socket (`$XDG_RUNTIME_DIR/x5desk/obs.sock`) with `live::obs::Client`; it announces the modifiers OBS imports and
  sends `done` only after OBS's GPU fence for that buffer.
