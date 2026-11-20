# x5desk

Desktop app for the Insta360 X5: player, camera control, live output, export. Linux (Wayland, RADV) first; macOS (Metal, VideoToolbox) as well.
The plan with decisions, measurements and milestones is `../insta360/PLAN.md`; the camera protocol and file formats
are documented in `../insta360/re/` (`PROTOCOL.md`, `analysis/*.md`).

## Rules

- No FFmpeg. Video goes through Vulkan Video (Linux) or VideoToolbox (macOS) on the one shared device.
- No Insta360 data in this repo unless the code needs it to run (the `.proto` files the app uses). Lens calibration
  comes from each file's trailer, never from the repo. Seam flow uses RAFT (open model), no Insta360 models.
- The UI thread never paces playback. Media threads decode ahead of the audio clock.
- Tests that need sample footage read the directory in `X5_SAMPLES` (`v635.insv`, `v606.insv`, `v627.insv`, `v635.lrv`)
  and return early with a note when it is unset. The app's `faces`, `plates` and `timeline_faces` (and `ai`'s
  `track_v606` and `auto_v635`) check what those clips show; its blur parity tests (`privacy_parity`,
  `agent_grab_parity`) compare two passes, so any X5 recording will do (`v635.insv`, else the first `.insv`;
  `X5_PLATES_AT`, `X5_BLUR_AT`: a time with vehicles, with a face or plate). `track`'s `tests/footage.rs` `subject` with
  `X5_REF` (a reference track checked by eye) asserts the box on the subject itself on half the frames, without it just
  a box. `track`'s `tests/footage.rs` `plates` checks the plate pass against plates labelled by eye (`<clip>_plates.txt`
  in `X5_FIXTURES`, never in git) and skips each clip without them. Track test results depend on the kernels burn's
  autotune picks (other kernels round VitTrack's confidence across its thresholds differently): a cold cubecl slot (App,
  burn's cache) tunes them under whatever else runs on the GPU.
- Tests on numbered clips (`clip::tests::numbered`, made with the ffmpeg CLI: the B-frame, open-GOP and 10-bit tests of
  the app's `clip.rs`, movie's B-frame test) skip with a note when ffmpeg is missing or cannot make the clip (an encoder
  or filter it lacks, drawtext without fontconfig).
- `link`'s `holes_fs` checks that a download on FAT or exFAT runs one connection in a dir on one named by
  `X5_NOHOLES` (on macOS a mounted `hdiutil create -fs ExFAT` image will do), without it only the temp dir's four.
- `media::part`'s `finish_cases` runs in a dir named by `X5_FAT_DIR` too (on FAT or exFAT, where macOS renumbers a
  file as it is written; such an image will do), and mcp's `tests/render.rs` `fat_refused` only there.
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
| `video` | `Gpu` (one Vulkan device via wgpu-hal with decode and encode queues; macOS: the Metal device; `Gpu::trim`: an idle host's parked decoder memory freed, macOS the texture cache flushed), H.265 / H.264 parsers, Vulkan Video (macOS: VideoToolbox, `mac/`) decode and encode, NV12 / P010 pictures for wgpu (`Ring::planes`); `Decoder::failed`, the pictures concealed (macOS; 0 on Linux, where one that fails is an error); `check`, a host's GPU self-test: an exact decode of hand-made H.264 High I_PCM pictures (`pcm`), the X5's H.265 at its size from the driver's caps without a session (`h265`), the decode and encode caps per profile, H.265 Main, Main 10, H.264 High (`caps`), one 640 × 360 picture through an encoder (`encode`), and its test streams (`sets`, `flat`, `skip`, `broken`; the bit writer gains `se`, `align`) |
| `media` | MP4 demux and mux (`sv3d` / `st3d`; `Mux::new` writes into a file made elsewhere, emptied first, `Mux::finish_with` asks a stop before each 4 MiB chunk of the faststart shift, `Mp4::from_file` reads an open file), output files beside their name (`part`: `<p>.part` made `O_CREAT\|O_EXCL`, never through a link, read + write, 0o666 under the umask, known by its identity from then on, `Id`: dev, inode, birth time, inode and birth alone when dev differs, Windows the volume serial and file index; `publish` never replaces: Linux `renameat2(RENAME_NOREPLACE)`, macOS `renamex_np(RENAME_EXCL)`, Windows `MoveFileExW` without replacing, where the file system lacks it a hard link and an unlink, without hard links a look and a rename, said once per volume; `settle`, the one cleanup by identity: dst, then the `.part`, then dst again; `finish`, a job's end: the stop checked last, the `.part` still the open file as it is now, the publish, every error removing it only if it is that file; `free`: nothing at a name or its `.part`, a dangling link included; `avail`: free space and volume; `fat`: FAT or exFAT, whose inode numbers may change across mounts, on macOS as a file is written), an RGBA area downscaler (`scale`: an agent's still preview), `.insv` / `.insp` trailer (calibration, gyro, exposure), AAC decode (symphonia) and AAC-LC encode (`aacenc`), editor mixer with ducking and limiter (`mix`), music import (`music`), resampler, RNNoise denoise, audio output and master clock (cpal; `audio::Player::open_on` plays on a picked device, `audio::device(id)` resolves the settings' device id by exact match: it lists the devices, so resolve when the choice changes, never per open; macOS: the app's settings look a gone or missing picked device up again on the opening thread), `audio::Conv` to a stream's format (`Conv::on`: macOS the device's own rate and up to 2 channels, else the source's; the clock counts source frames), MJPEG decode, baseline JPEG encode (`jpeg`), UVC capture of the X5's Webcam Mode |
| `proto` | Protobuf messages for the camera (prost, compiled with protox from `proto/`) |
| `link` | Camera link: UCD framing, session, requests, notifications, HTTP tunnel with several connections, downloads (each connection writes its range at its offset of `<dst>.part`; on FAT and exFAT, which hold no holes and zero-fill up to an out-of-order write, one connection writes in order; Windows marks the file sparse first); USB (nusb), Wi-Fi (TCP); BLE lives in `ble`. Connections start as `link::Opening` (dropping it cancels the sync); `Error::Silent` after 30 s without a sync answer; `Camera::set_time` sends the app's two SET_OPTIONS at once and returns both pending answers; `scan` / `watch` report `Plug::Disk` for an Insta360 product string under vendor 070a (the X5 in U-disk mode, or the first seconds of app mode); feature `sim`: a simulated camera (a handler status 0 leaves the request unanswered) |
| `stitch` | Dual fisheye → equirect / perspective on the shared device from NV12 / P010 textures: template seam, RAFT flow, colour matching, stabilisation rotation; burn's attachment per device (`attach`) and its cache dir (`cache`: a slot of the process's own, else memory only) |
| `stab` | Stabilisation maths: trailer gyro / accelerometer → per-frame orientation, smoothing (FlowState-like), horizon and direction lock, the forward view (`fwd`: the direction of travel from the feature motion the gyro leaves, its focus of expansion fitted per pair of frames; `Lock::Forward` follows it, smoothed, and keeps the horizon lock's heading, turned by the last offset, below walking pace), rolling-shutter terms |
| `fx` | Image effects on the shared device: colour adjustments (incl. brightness, brilliance), automatic colour and underwater correction, `.cube` LUTs and our own looks, defringe (purple / green / blue), clarity, overlays (watermark, stickers, text) |
| `edit` | Editor model: timeline, clips with a look and colour adjustments each (`fx::Params`: edit depends on fx), trim / split / speed / freeze, reframing keyframes and camera-movement presets, undo (a drag that ends where it started leaves no undo step and keeps the redo list), project files |
| `export` | Render jobs: decode → stitch → stabilise → effects → encode → mux, 360 and flat, queue; the forward view's analysis (`flow`: every 15th of a second stitched 1024 wide with the gyro's rotation out and its corners tracked by `stab::vis`; `travel`: the direction of travel per pair, kept for the process per file, which a job with `Lock::Forward` runs in its plan unless the player, the Export screen or auto edit ran it first); editor timelines (`Cut::Edit`: speed ramps, freeze frames, views, each clip's adjustments and look via `fx`, audio mixed with music, RNNoise-denoised after the time-stretch when `Edit::denoise` is set, and AAC-encoded, or copied for plain 1× cuts without denoise); photo frame grabs (JPEG with GPano; the encoder is `media::jpeg`, also reachable as `export::jpeg`). `export::locks(gate, burn)` hands it the app's GATE / BURN; `Job::flat` keeps a flat 2:1 picture from a 360 hook untagged. Output: `<path>.part` made new at the job's start (`media::part::create`; one there fails it, "being written by another export") or reserved by the caller (`run_into`'s `Dest`), written through the open file, published only if nothing is at `path` and `stop` is still clear after the last write, every error removing it and only it (`media::part::finish`; the job's closures dropped first, one that panics as it drops failing the job). `Dest::strict`: the MP4's frame count or the still's JPEG decode and size checked before the publish, and a concealed picture (`Progress::concealed`, from `Decoder::failed`) fails the job (`CONCEALED`) unless `frozen_ok`. `stop` reaches the decode pass (its catch-up from the sync sample too), the renderer (a 100 ms `recv_timeout`: a stalled decode), the encoder's drain and the faststart shift. `Queue`: `with(name)` (queues on one device run side by side), `push_into` (a reserved `.part`; Err: the thread gone, the `.part` removed), a queued job's cancel final at once (its `.part` removed), a panic failing its job alone (`catch_unwind` per job), a device poll after each job (`lost`), `busy`, `alive`, `trim` (its RAFT model dropped), `forget`, `detach` (no join; `Drop` joins). `source` reads a recording as a job would, without the GPU (`Src`: rate, frames, length, each lens's codec, depth and size, IMU, AAC); `scoped` is public (a host's GPU work). `Job::pre`: what a job's frames need first (an agent's blur pass), run once on its render thread in an error scope of its own, after the plan and before RAFT loads, with the job's stop (looked at again after it); a hook it hands back runs on each frame before `Job::fx`. Feature `synth`: a synthetic X5 recording for the tests of this crate and those above it (`synth::insv`, `damaged`: both lenses as H.264 High tracks of `video::check`'s I_PCM pictures, an IDR every 30 frames, 29.97 fps, a V3 trailer with a synthetic calibration, no IMU, no audio) |
| `track` | AI with open models (burn on the shared device; macOS: the face detector through Core ML, `coreml.rs`): subject tracker on the sphere with re-identification (LOMO colour and texture histograms with Fisher weights learnt per session, no model), detector, auto framing, face and licence plate privacy blur regions (plates: our fine-tune of YOLOX-nano, `models/README`, on square crops around the vehicles YOLOX finds at score 0.1 or more: `Ai::plates`, `plate_crops`; regions through the same `Hide` and `Blur` as faces), highlights → `edit::Timeline`. The tracker finds a lost subject again only within its reach (`Seen::prior`, `Seen::speed`: the speed, in its own heights a second, it needs to be a candidate since it was last seen for sure, its heading carried on for up to a second; beyond what it reaches setting off at a walk and speeding up by 3 heights/s² to a sprint's 3, it is someone else; within it, its similarity loses 0.1 × (speed / 1 height a second)²), and a person only with the pick's template's box on it (`vit::hit`: never on the detector's box alone); a person at least as like one `find` took and the tracker lost again before VitTrack was sure of them as like the subject (`vit::Tried`, the last 8; not added to the gallery, as the subject can be one) needs VitTrack 0.5, not 0.25 (`Tried::need`, also for whoever the tracker takes on its own after a gap). `Tracker::fps` is the clip's frame rate (30 until set) |
| `ble` | Camera BLE: scan, wake, pairing, Wi-Fi credentials (btleplug). The X5's wake-up brings up its radios and session, never its screen. Feature `sim`: `Ble::sim`, a simulated adapter and camera on the real session code ; macOS: `Error::Off` / `Error::Denied` and `Event::Adapter` (Bluetooth off, not allowed, back) |
| `live` | FLV and RTMP client, OBS frame protocol (Unix socket + DMA-BUF fds), PipeWire camera node (the last three Linux only: elsewhere `stub.rs` keeps their API and refuses to start) |
| `obs` | OBS source plugin (`cdylib`) that imports our frames |
| `cam` | The app's camera code without its UI, so the app and a headless host run the same session code: the camera worker (`camera`: session, clock, album and thumbnails, download queue, commands, mounted cards; at its end a running download, a card import too, goes back to the queue stopped, `dl::Queue::quit`, one whose cancel it had yet to see paused, and kept so whatever ends its transfer; the queue file leaves out an entry its lines cannot hold: a line break in a URI or path, a tab in a URI, a path that is no UTF-8) and the capture worker (`capture`: mode, settings and shutter from `/CONF`; a command sent as `Cmd::Ask` gets its outcome, `capture::Out`, in the report after it under the caller's number, kept across sessions, refused unless the camera is as the caller checked it: mode, lens block, a sub-mode the protocol knows, `capture::Seen`; a start whose capture state cannot be read first, or that the camera answers with an error, sent or began nothing, `Out::No`, a failure after it went out may have begun, `Out::Failed`; `St::vals` names the mode whose values the last load read, and a setting is written, a photo taken, only with them, read once more first when a load failed; the 10 s status poll reads the capture state too, and loads again while a load failed, at most once a minute; a `/CONF` that did not parse is not fetched again in the session; a reload the session's close cuts short notes nothing), with `album`, `dl`, `disk`, `conf`, `work` (a recording's files), `dirs` (config, cache and video folders; on Windows the profile's), `awake` (App Nap's `hold`; `keep`, for an agent's renders: an activity of its own that keeps idle system sleep off too; both nothing elsewhere), `mono`. The workers guard what any host sends: no delete while the camera records or captures (`Info::busy`, and the capture status read before the delete), a setting only to a value `/CONF` allows now (`St::allowed`), no mode or setting change while the camera captures (its capture state read first), no exposure write without the camera's exposure values (a default one would reset the program and shutter), no start of a recording or timelapse at 5 % battery or less unless charging (`camera::empty`), an explicit `Start` / `Stop` (the camera's capture state read first, nothing sent when it already is so); a connect sets the camera clock only when asked to (`Cmd::Connect { clock }`, the app's own always; its retries and its reconnects after a lost link, by the 2 s timer or the camera back on the bus, the same). The camera's listing leaves out files whose names are no safe file names (`album::safe`: a separator, `:`, `..`, a control character, a trailing dot or space, a Windows device name), in a note, so no host joins them to a folder. Outside Linux and macOS: no card found (a card source given still lists, its URIs `/`-separated); not unix: the camera clock never set (no IANA zone known; said once), the favourite timestamp from the system's zone rules (Windows: `GetTimeZoneInformation`). `conf` also holds what ran on an X5 (360, fw v1.13.21): the settings `CHECKED`, the mode switches `SWITCHED`, the shutters `SHOT`; `conf::known`: sub-mode values the protocol knows (one it does not reads as VIDEO_NORMAL or PHOTO_SINGLE); `conf::show`: a value as people read it. Feature `sim`: a simulated X5 with a fake card for the tests above it, capture on request (`sim::capture`: a `/CONF` of tested and untested modes and settings, sub-modes as raw values, photography options per function mode, photos and clips added with their notifications). No egui, wgpu or video |
| `mcp` | x5desk as an MCP server (rmcp 3.8, stdio framing; clients on 2025-11-25 and 2026-07-28). `host`: the headless host, cam's camera worker following USB but never connecting on its own or setting the camera clock, its capture worker kept on the same session (a thread of its own, recording every asked command's outcome by number; a report of the connection come or gone moves the state's session on at once, so a gone session's capture state is never read as the camera's, while that session's late reports still end a recording this host started), the download queue, the jobs and the render jobs (kept in `<config>/x5desk` for the next host, see Agents), cards watched, a pump thread turning the worker's reports into the state the tools wait on, a poller following this host's renders (their records, the watchdog, a suspect GPU's probe, the idle trim, idle sleep kept off); the camera is released after 5 min without a call (not while a call runs, a download runs or can start: a card import whose card went does not count, or the camera captures; renders don't hold it) and by `Host::shutdown`, which first cancels this host's renders (waited for up to `RENDERS`, 8 s, then settled by identity) and stops a recording this host started; a stopped download keeps its `.part`, which downloading the item again resumes, in a later host too. One `Host` per process, one rmcp service per connection over it (`Host::serve` on any reader and writer, narrowed by that client's own `--tiers` if any). `server`: the tools, a table by tier (`read`, `local`, `camera`, `delete`; `untested` has none, it widens `camera`) that `tools/list` and `tools/call` both filter by the enabled tiers, the GPU tools only with a backend (`server::listed`), the capture tools and the camera's file writes in `server::capture`, the GPU tools and the render jobs in `server::render`; results are JSON built field by field (never the access point's password or the Wi-Fi client id), thumbnails JPEG; a Wi-Fi connect goes only to a private or link-local address. `config`: the tiers in `<config>/x5desk/mcp` (0600), `--tiers` only narrowing, re-read every 2 s, a change sent as `tools/list_changed` (2026-07-28: on `subscriptions/listen`); its writers (`enable`, `disable`, a host keeping its Wi-Fi id) take turns under an `flock` on `mcp.lock` (0600), each through a temp name of its own. `job`: downloads and renders followed by `<host id>-<n>` handles, of the host that made the job, numbered over both kinds (`Jobs::which`; downloads kept with the queue as a JSON object a line, so they work in later hosts; one that ended and is not kept, an ended host's or one this host pruned, says so); jobs of one item share a file's download while it waits or runs; one that stopped, failed, or finished and went is the job's alone that starts it again (the older jobs show it as it was then); a cancel stops only the job's own downloads that no other uncancelled job shares, and downloading an item again resumes its stopped files under their names, the finished ones kept. The jobs are locked before the queue. `paths`: writes only into `<videos>/x5desk/agent` (subfolders without `..`, `:` or control characters, made only where no link leads out of it: the part that exists is checked first; names in use, a dangling link too, or being written numbered alike per item, camera names checked again), reads only of X5 files under the canonicalised read roots (the agent folder, the app's `dest=` and `<videos>/x5desk` looked up at each use, cards, `--allow-dir`s), a path outside them answered alike whether it exists or not; on Windows paths shown without `\\?\`. `scrub`: every line written to a client and to the log with each secret, raw and JSON-escaped, masked (none under 4 bytes); a client's line stays JSON, only its strings and keys masked (a result's text, JSON in a string, as JSON too), numbers as they are, one that does not parse masked as bytes. `log`: `<cache>/x5desk/mcp.log`, a line per call, 1 MB then `.1`. `sim`: `--sim`, cam's simulated X5 (an access-point password the host learns only from it, and a marker it knows from its start, both looked for by the scrub tests) and a fake card in temp dirs. `render`: the seam to the GPU, plain data (`Render`, `Run`, `Catalog`, `Plan`, `GpuInfo`, `Found`; `Backend`, which the app hands `mcp::main`: no GPU dependency here, no GPU tools without one). `renders`: the render records in `mcp-renders`, their crash cleanup by identity. The process side (Agents, below): `cli` (`x5desk mcp …`, its `gpu-check` child), `out` (the stdout guard), `endpoint` (the per-user socket or pipe, the election), `front` (stdio's lines to the host of this process or another's, the handshake replayed, re-election, `--tiers` for its own client only, its own host told of stdin's end first, `Host::closing`, the answers passed on after it) |
| `app` | egui (eframe 0.37) UI, binary `x5desk` (`x5desk mcp …` hands over to `mcp::main` first thing in `main`, the app's engine, `export::agent`, as its GPU backend; `_exit` after it while a GPU thread may be in the driver) |

## Shared device contract (`video::Gpu`)

- `Gpu::new()` opens the Vulkan adapter through wgpu-hal with every adapter feature, the experimental features,
  `TEXTURE_FORMAT_NV12`, the video decode and encode queue families, and the external memory / semaphore
  extensions (`VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`,
  `VK_KHR_external_semaphore_fd`, `VK_EXT_queue_family_foreign`).
- `Gpu::new` serialises instance creation (video's `gpu.rs` `SCAN`): the loader runs every ICD's negotiate function from
  overlapping scans, and NVIDIA's ICD (installed on this AMD host) crashes when two threads hit its lazy init. Any
  other `wgpu::Instance::new` in the process (eframe's `CreateNew`) bypasses it and can race.
- Public fields: `instance`, `adapter`, `device`, `queue` (wgpu) and the raw Vulkan handles the video and live code
  need. The app opens it in `gpu.rs` (`gpu::open`, the window's and the headless MCP host's one setup, App) and hands
  the wgpu handles to eframe with `WgpuSetup::Existing`.
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
- `Decoder::failed()`: the pictures that showed the frame before them, which crate export's strict jobs fail on. macOS
  conceals a picture VideoToolbox fails on and counts it; here it is 0, as a picture that fails is an error.
- Anything else that stages a semaphore signal on wgpu's queue (DMA-BUF export, encoder input) stages it only after
  the submit it covers, never before: a signal staged first can be taken by another thread's earlier submit.
- Stitch sources are whole NV12 / P010 textures or rects in one (full range, BT.709; `Ctx::planes` also takes
  R16 / RG16 views as P010; side by side: `.lrv` and live preview, lens 0 left; the
  `.lrv` needs `Frame::shift`); the preview also draws from an RGBA equirect (paused RAFT stitch, Webcam Mode).
- Encoder input: `Encoder::input` (`Arc<Input>`) is a ring of `encode::SLOTS` NV12 / P010 textures, home state
  `COPY_DST`; producers fill them by copying (wgpu has no NV12 / P010 storage). Before the first wgpu submit that writes
  frame `f` call `input.acquire(&gpu, f)` (`Err(Full)` after 50 ms if the encoder still holds the slot); after the last
  one call `input.release(&gpu, f)`, then hand `f` to the encoder thread, which calls `encode(&gpu, pts)` and
  `packet(&mut Packet, ns)`. Frames are released in order. `Input::key()` makes the next frame an IDR. `Cfg.vbv` is the
  rate control buffer in ms (1000 for files, 100 for low-latency RTMP). The VCN firmware's CBR (RADV and radeonsi alike)
  runs 4 to 10 % over at 100 ms and near the target from about 500 ms on ordinary content (`video` test `cbr`); initial
  occupancy, GOP flags and GOP-remaining hints change nothing. RADV: no B-frames, quality level 3 of H.265 under rate
  control writes corrupt slices (excluded by `Encoder::levels`).
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
  ring size; freed with the `Gpu`, or by `Gpu::trim`: an idle MCP host's, Agents; running decoders keep theirs), and
  the next decoder of the shape takes them (new timelines, frames from 0): nothing may sample a ring's textures after
  its last `Arc` goes. A new `Decoder` grows its command buffers' upload memory before any decode (RADV frees outgrown
  upload buffers at a command buffer's next reset).
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
  then); `Gpu::trim` is that flush (no decoder memory of ours is parked here). `Decoder::took`: the decode's own time
  once it has its slot (parsing, VideoToolbox, the copy), which the preview's pacing and decode stats use in place of
  Linux's done timeline. The parsers step over scaling lists here (VideoToolbox applies them; its own HEVC encoder
  writes them), rejecting a delta_scale outside −128..=127; on Linux they stay refused.
- GPU self-test (`video::check`, Agents): VideoToolbox has no query for sizes or Main 10, so `h265` asks
  `VTIsHardwareDecodeSupported` and `caps` is a fixed table: decode, for the codecs with a hardware decoder, from
  16 × 16 up to the app's limits (H.264 4096, H.265 8192: `H264_MAX`, `H265_MAX`); encode in CBR, VBR and constant
  QP, QP 0-51, no bit rate reported. The pixel check also catches VideoToolbox's concealment, a picture it returns `Ok`.
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
  Core ML keeps its own Neural Engine cache under `~/Library/Caches/<process name>` (of `CFFIXED_USER_HOME` when set,
  not `HOME`), test-hook runs too. The first face pass loads it (≈ 0.6 s, against burn's 11 s of kernel compiles). The
  crops reach it through a mapped copy of the crop pass's buffer; its scores and boxes come back in burn's layout, so
  the decoding after is the same. Every Core ML call runs in an autorelease pool (the app's threads have none: a
  prediction autoreleases 12 MB of outputs). M4, footage medians (`tests/footage.rs` `privacy`, `X5_YUNET=burn` for
  burn's): 42 ms per frame of 20 tiles instead of 80, a run of up to 8 tiles with copies, read-back and decoding 11.5 ms
  instead of 24 (Core ML's prediction of 8 alone: 9 ms), the same faces and tracks. `Ai::coreml` (on) picks it,
  `Ai::coreml_used` tells whether it runs; a model that won't load, or a run it fails, leaves burn's YuNet, which the
  other platforms run. The plate pass stays on burn's GPU (YOLOX, the plate detector).
- App Nap: long work holds an `awake::Hold` (one ref-counted NSProcessInfo activity, user initiated, idle system sleep
  allowed; a no-op elsewhere): each job on the export thread (exports, frame grabs, Shot Lab exports), Deep Track and
  auto edit, Shot Lab previews, the live outputs while one runs (RTMP, a watched OBS output, the PipeWire camera; not
  the Live screen's preview alone), and the running download or card import (`dl::Cur`). An MCP host's renders hold
  an `awake::Keep` instead (its poller, while a render job of the host is unfinished): an activity of its own, user
  initiated, that keeps idle system sleep off too (the display may sleep), as nobody may be at the keyboard and
  a sleep invalidates VideoToolbox's sessions (pictures concealed until the next random access point); no hold's end
  weakens it.
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
  (`video::check`'s streams in Constrained Baseline, I_PCM, one luma value per picture): a failed picture concealed
  with the one before, never the slot's old content or a pre-restart picture; the gate; scaling lists cut short or out
  of range rejected. Vulkan-only tests are `cfg(target_os = "linux")`. Test tools that read `/proc` (the RTMP
  listeners' readiness) use `lsof` elsewhere; `PR_SET_PDEATHSIG` is Linux only.

## App (`crates/app`)

- One eframe window on `video::Gpu` (`WgpuSetup::Existing`). A side panel switches between Library (local files and
  player), Editor, Shot Lab, Camera (panes Album, Downloads, Camera, Wireless), Capture (live preview, `/CONF`-driven
  controls, shutter), Live (RTMP, OBS, PipeWire camera), Export, Settings (download folder, default export preset, audio
  output of the player and the editor's preview, thumbnail cache, GPU info, client id) and Help (keys, a guide per
  screen, About with licences). A camera battery at 20 % or less and not charging shows as a warning on every screen
  (`camera::low`).
- One module per screen: `library.rs` / `player.rs`; `editor.rs` (`deck.rs`, `sound.rs`, `thumbs.rs`, `doc.rs`);
  `lab.rs` (Shot Lab on crate shotlab); `camera.rs` (its worker, `cam::camera`, connects in the app's order, PROTOCOL.md
  §5.1: identity, options, capture status, then the clock's two SET_OPTIONS (§5.3) on every USB and Wi-Fi connect, not
  waited for: polled in its tick; once both were answered, a refusal too, the options are read again, so the Clock row
  shows the clock as the camera has it (set, even when only the zone was refused); not when an answer never came or came
  garbled (timeout: GET_OPTIONS would hold the worker; link lost; undecodable reply); then the connect note and a
  low-battery note come; a refusal or timeout adds an error note and the session stays; then the Capture screen's tap
  and the listing; Sync time takes the same path; at quit only the running download's end is collected (`dl::end`),
  nothing new starts; `cam`'s `album`, `dl`; `disk`: a mounted card found through the mount table and the sysfs product
  string, listed with one trailer per item (its first file; empty files left out), imported as `file://` download
  entries; the bar takes U-disk mode from a mounted disk of the camera's own, else from the scan's `Plug::Disk`
  (`070a:4027`, also the first seconds of app mode); `bt.rs`: Bluetooth and station-mode Wi-Fi: join, a read of the
  camera's existing station connection over BLE, a typed camera address; the scan row says when Bluetooth is off or not
  allowed, macOS: System Settings > Privacy & Security > Bluetooth); `capture.rs` (worker `cam::capture` on the camera
  session through `camera::Tap`; its reports carry the session (`Tap::sid`) they belong to, and the screen drops those
  of an earlier one; the shutter sends a photo, or the start or stop its button shows; `cam::conf`: the `/CONF` model
  and its wire values, the lists of what ran on an X5, which the screen marks ⚠ when not in them; `feed.rs`: live
  preview decode thread, pictures handed over under one lock and taken as they fall due, paced on the camera clock from
  the slowest of the last 32 deliveries, IMU levelling; `webcam.rs`: Webcam Mode via `media::uvc` + MJPEG; `replay.rs`:
  link's simulated camera streaming a recorded preview, every loop from its first IDR (leading packets stamped 0 are
  sent again a frame apart)); `live.rs` (sources `live::Pick {Player, Preview, Webcam}`; `live::Source::tick(at)` takes
  the tick's scheduled `CLOCK_MONOTONIC` time); `export.rs` (an output name is numbered past one on disk, a dangling
  link too, one whose `.part` a job in any process is writing, and one a queued job will write, `media::part::free`; a
  clip's job built by `span` and `clip_job` and named by `stem_tag`, which the agents' engine `export/agent.rs` shares,
  so both build the same job; `Keep::video` runs it through `export::run_into`, `Task.dest` a reserved `.part`'s place,
  M4) with the `photo.rs` and `movie.rs` viewers (`movie.rs`: plain videos such as our exports or B-frame MP4s, NV12
  or P010 to RGBA through BT.709 limited range); `ai.rs`; `settings.rs`; `help.rs`; `gpu.rs` (the process's one GPU
  setup, the window's and the MCP server's: below). Work runs on the threads in PLAN.md §2; the UI thread only reads
  the newest state and draws.
- GPU setup (`gpu::open`, once per process): the device (`video::Gpu::new`), burn's cache dir (`stitch::cache`, before
  burn attaches: else cubecl keeps it under its Target root, an MCP client's project's `target/`), burn attached
  (`stitch::attach`, first, as cubecl installs its own error handler then), the error policy (`gpu::Policy`), then
  `export::locks(&GATE, &BURN)`. The window's policy is `Panic` (`errors`, below; a panic in the attach ends the app);
  a headless host's is `Log` (Agents): each uncaptured error logged and counted, never a panic (the handler runs on
  whatever thread made the failing call, a lane's or a tokio worker's, in the process that holds the camera), an
  out-of-memory or internal one making the GPU suspect, a panic in the attach an error (RAFT off). wgpu hands a lost
  device to no handler and cubecl only logs it (its device-lost callback stays: replacing it blinds burn), so a
  headless host also installs a `log` logger (warn and up into its `gpu::Watch`'s sink, the newest host's log, 20 lines
  a minute), and cubecl's line "the wgpu device was lost" makes the GPU suspect.
- AI (`ai.rs`): `ai::MODELS` loads one `track::Ai` per run and device on first use (warm-up: the tracker and the
  detector at every batch size, one network call at a time under BURN; re-identification runs no network), shared by
  Deep Track, auto edit and every privacy pass. Deep Track sets `Tracker::fps` from the clip
  (`shotlab::Clip::meta.fps`); its directions are in the frame of the track's stabilisation and are turned into any
  other frame with `Track::turn` / `ai::rots`. The privacy blur also covers editor timelines (one face pass per clip)
  and Shot Lab passes (before the effect). The privacy pass (the export's `Keep::privacy` over `export::privacy_pass`,
  the player's preview `ai::privacy`) finds faces and/or licence plates every 3rd frame, each linked by a `Hide` of its
  own; with plates it stitches 7680 wide and finds faces there too, else 3072. `Keep::privacy` holds `ai::MODELS` for
  the whole pass (`Hold::Pass`); the agents' lanes run the same pass from `Job::pre` (Agents) and hold the models per
  searched frame (`Hold::Frame`, through `Models::with_stop`, whose wait for them sees the job's stop), so the other
  lane waits out one detection, not a pass. `Models::trim` drops them when nobody holds them (an idle MCP host's trim).
  The player's Blur faces and Blur license plates previews carry into the export (`Spec::faces`, `Spec::plates`).
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
- Device policy (`main.rs` `GATE: RwLock`, `BURN: Mutex`, handed to crate export by `gpu::open`): the app's own
  submitting threads hold `GATE` shared around each piece of GPU work (live output thread per pass, RTMP encoder per
  frame, export threads and decoders, photo exports, the MCP engine's lanes and its probe; macOS: `clip::Video`'s and
  the live preview's decoders around their copies, `Decoder::gated`); threads whose work runs through burn (RAFT
  worker, the AI models, shotlab's person network) hold `BURN` for a whole job instead. Lock order: GATE is dropped
  before BURN is taken; BURN never nests. A new submitter takes one of them. A headless host has no surface, so its
  GATE never has a writer; its two lanes' RAFT flows take turns at BURN (a RAFT still waits out an export's first
  kernel compile).
- burn's cache: burn keeps its device measurements and autotune results (burn feature `persistence`) in a slot of
  `<cache>/x5desk/cubecl` (`stitch::cache(Option<&Path>)`, which `gpu::open` calls before `stitch::attach`; the first
  call wins); a real MCP host's in `<cache>/x5desk/cubecl-mcp`, apart from the window's slots (so neither takes the
  other's warm one) and warm across sessions. Test-hook runs (the app's `X5DESK_TEST`, `x5desk mcp` under `--sim` or
  `X5DESK_TEST`) share `$XDG_RUNTIME_DIR/x5desk-cubecl` (the temp dir without it), so they start warm too, if it is a
  dir of this user's, not a symlink, that no other user can write to (`cam::dirs::mine`, made 0700 when missing); any
  other burn user (tests) gets cubecl's own root from `stitch::attach` (`target/environment` of the outermost Cargo
  workspace above the working dir, else the user cache dir). A slot is the first `<n>` (made 0700) whose `lock` the
  process can `flock`, never through a symlink another user planted (the lock made with `O_EXCL`, else taken only if
  `lstat` shows a plain file): cubecl-environment 0.11's database (turso 0.8, experimental multi-process WAL) loses
  acknowledged writes, reads corrupt pages and panics when processes share it. The lock goes with the process however
  it ends, so slots are reused and never outnumber the processes that ran at once (nothing trims them). With no dir (a
  hook dir refused, neither `XDG_CACHE_HOME` nor `HOME`) or no slot locked (no locks on that file system, no write
  access), cubecl's root is `/dev/null`, where its database can't be opened, and cubecl keeps it all in memory. Build
  records are off. With a warm slot RAFT's first job only compiles its kernels (≈ 2 s instead of ≈ 10 s on a quiet
  GPU; 4–9 s instead of 22–46 s under other load). A configure retry (below) that lands in a burn job still waits for
  the job's end, the UI frozen meanwhile.
- The window owns the device's uncaptured-error handler (`main.rs` `errors`). A `Surface::configure` that fails with
  `GpuWaitTimeout` (another thread submitted during its idle wait; native Wayland never reports the stale swapchain
  `Outdated`) is retried in the next painted frame: egui is made to reconfigure by flipping
  `desired_maximum_frame_latency` between `Some(2)` and `None`, and the UI thread holds BURN, then GATE, exclusively
  from the end of that frame's UI until egui has configured and acquired. wgpu-core drops the map / work-done
  callbacks a failed configure collected, so a `map_async` user treats a dropped callback as a finished map.
  Everything else panics. burn attaches once per device through `stitch::attach`, which installs cubecl's handler
  (and device-lost callback) on first attach only; `gpu::open` calls it before installing the policy's own, and RAFT,
  shotlab's segmentation and the one `track::Ai` in `ai::MODELS` reuse that attachment.
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
    mount table nor what is mounted, `/Volumes` on macOS, is read: `disk::Watch::hook`), `import`, `ble=sim`, `ble=sta`
    (the simulated X5 on HomeNet from an earlier session: connect, pair, read its station connection; no join),
    `battery=<%>` (the `replay=` camera's level);
  - capture: `capture`, `replay=<rec.h264>`, `conf=<file>`, `webcam[=1920|<file.mjpg>]`, `cstats=<s>`, `yaw=<deg>`,
    `level=off`, `pace=off` (viewer only), `src=preview|webcam`;
  - live: `live`, `end` (Live screen scrolled to its end), `rtmp=<url>`, `res=`, `fps=`, `kbps=`, `obs`, `pwcam`,
    `grab=`, `min=`;
  - export: `export=<preset>[:faces][:plates]`, `frame=`, `dest=`;
  - editor: `editor`, `project=<file>`, `clip=<file>` (repeatable), `undo=<n>`, `redo=<n>`, `at=<s>`, `sel=<k>`,
    `etab=clip|reframe|colour|audio|overlays`, `e360`, `epreset=<preset>`, `esave=<file>`,
    `tlexport=<preset>[:noraft][:qp=<n>]`, `eplay`;
  - AI and Shot Lab: `box=<x0>/<y0>/<x1>/<y1>`, `restab=off|free|horizon|direction[/<σ>]`, `ebox=<x0>/<y0>/<x1>/<y1>`,
    `blur`, `plates` (the player's plate blur preview), `autoedit[=<preset>[:faces][:plates]]`, `lab=<effect>`,
    `lrange=<a>/<b>`, `lpre`, `lexp=<preset>[:faces][:plates]` (`box=`
    is 0..1 of the player's view at 16:9, as a flat 1080p export frames it, whatever the window's size or layout; the
    hook prints its subject, `x5desk: box: subject at yaw …, pitch …, <w>° × <h>°`, so runs can be compared; `ebox=`
    is 0..1 of the `epreset=` export preset's flat frame, whatever the window: a flat preset's own, as the viewer
    letterboxes it, 16:9 for a 360 one as for `box=`; it prints its subject as `x5desk editor: box: …`);
  - other screens: `settings`, `help[=keys|about|<screen>]`.

  Real-camera tests that change the camera's state and restore it: `X5_CAMERA_WRITE=<dir>`
  (`capture::tests::real_writes`), `X5_CAMERA_ILOG=<dir>` with `X5_ILOG_EXPO=<iso>,<shutter>`
  (`capture::tests::real_ilog`). `x5desk mcp` takes `X5DESK_TEST` too (Agents, below). With the display off (DPMS),
  Wayland sends no frame callbacks and windows stall: run tests under XWayland (`env -u WAYLAND_DISPLAY`).
- OBS plugin (`crates/obs`, `libx5desk_obs.so`, installed as `x5desk-obs.so`): source `x5desk_source`, reads the
  app's socket (`$XDG_RUNTIME_DIR/x5desk/obs.sock`) with `live::obs::Client`; it announces the modifiers OBS imports and
  sends `done` only after OBS's GPU fence for that buffer.

## Agents (`x5desk mcp`, crate `mcp`)

- CLI, dispatched in the app's `main` before `finder::watch`, `gpu::open` and anything that prints, the app's engine
  its GPU tools' backend (`export::agent::Agent::headless()`, which only allocates) (`x5desk <file>` and macOS's
  `-psn_` stay as they were; a file named `mcp` opens as `./mcp`):
  - `x5desk mcp [--sim] [--tiers a,b] [--allow-dir DIR]…` serves MCP on stdio. `--tiers` only narrows the enabled
    set (twice: both narrow); `--allow-dir` adds a read root (canonicalised; a missing one exits 1). `--sim`: cam's
    simulated X5, a fake card, agent folder, log, queue and jobs in a temp dir that goes with the host.
  - `x5desk mcp enable --tiers read,local[,camera][,delete][,untested]` writes `<config>/x5desk/mcp` (the set replaced;
    the dir owner-only, the file 0600; under an `flock` on `mcp.lock`, so a host storing its Wi-Fi id meanwhile loses
    neither) and prints what is enabled; `disable` turns every tier off. Hosts follow within 2 s.
  - `x5desk mcp --print-config` prints `claude mcp add --transport stdio x5desk -- <exe> mcp` and Claude Desktop's
    `mcpServers` entry for `current_exe()`, to be added while Claude Desktop is quit (on Linux it writes the file
    over while it runs); exit 1 from a disk image (`/Volumes/…`) or a translocated copy (`/AppTranslocation/`), paths
    that are gone later.
  - `x5desk mcp gpu-check` (not in the usage): the GPU self-test child (The GPU, below). Its stdout guarded as a
    server's, one JSON line on it, `{"v":1,"ok":true,"info":…}` or `{"v":1,"ok":false,"step":…,"why":…}`
    (`cli::gpu_check`, read back by `gpu_checked`; a panic in the test a failure without a step); it `_exit`s once
    the line is out, at stdin's EOF (its parent gone) or after 25 s.
  - Wrong arguments: the usage on stderr, exit 2. A server exits 0 when its client goes (stdin's end; SIGTERM,
    SIGINT and SIGHUP unless it started with them ignored, as under `nohup`, which stays so; Windows: Ctrl-C,
    Ctrl-Break, its console closed, logoff, shutdown) and 1 when it could not start. Started with SIGCHLD ignored, a
    server sets it back to its default (Linux would reap the `gpu-check` child itself, and waiting for it fail).
- stdout is the protocol's. First thing when serving, `out::guard` keeps a copy of stdout for the protocol and points
  stdout at stderr (unix: `F_DUPFD_CLOEXEC` of fd 1, then `dup2(2, 1)`, `/dev/null` without a stderr; Windows: a
  duplicated handle, then `SetStdHandle(STD_OUTPUT_HANDLE, <stderr>)`, which std's `println!` follows as it looks the
  handle up per write; the C runtime's fd 1 does not). stdin is read and stdout written on threads of their own, a
  line at a time.
- Hosts. One per user owns the camera; the first `x5desk mcp` binds the endpoint and hosts, later ones proxy to it. Each
  process is a front (`front`) on its own stdio that passes whole JSON-RPC lines (16 MiB at most, as rmcp) to a host:
  its own over a duplex (served as `stdio`) or another's over the endpoint (`endpoint <n>`; one rmcp service per
  connection over the shared host). The front keeps the client's `initialize` and `notifications/initialized`
  (2025-11-25 handshakes once) and replays them to every later host under an id of its own, the answer swallowed;
  2026-07-28 clients send none. When the host goes, each request it had not answered gets -32603 "host ended; try
  again", and the front elects again (bind and host, or connect to whoever did), the switch on stderr; a host that drops
  a front before a word is tried again after 50 ms, doubling to 2 s. At stdin's end (a leading BOM of a line is cut off,
  as rmcp does) a front sends nothing more, so its host sees the end too (a host of its own process first stops its
  renders, `Host::closing`, so a call waiting for one answers "the host is shutting down" before the drain), and passes
  on the answers to the requests still open until each has one, the link ends or 5 s pass (as rmcp's drain); then it
  ends without electing again. A request the client cancelled (`notifications/cancelled`, passed on) is open no more:
  rmcp drops an answer not sent yet (one that comes is passed on; with `--tiers` every tool list is narrowed). A host's
  stdin end (its client gone) ends its process that way: the listener closed and its socket removed first (so a
  proxy's election finds no host), then `Host::shutdown` (its renders cancelled and settled, a proxied client's too:
  The host's end, below; connections ended, a recording this host started stopped, the camera released, a running
  download back in the queue with its `.part`, the queue, the jobs and the render records saved for the next host),
  exit 0; a signal ends it at once, without the drain. An endpoint that fails (no home, a socket file of another user's,
  a path too long) is said on stderr and the process hosts without one. `--tiers` narrows only its own process's client,
  whichever host it has: the front answers calls of tools outside them itself, and sends the others with its `--tiers`
  in their `_meta` (`x5desk/tiers`, `front::NARROW`), which the host narrows that call by (so `untested`, which has no
  tools, too; a client can only narrow itself so) (also a call of a name it does not know or of none, -32602; under
  `--tiers` a line that is no JSON object never reaches the host, a batch's requests answered -32600) and leaves those
  tools out of `tools/list`. A hosting process serves the endpoint's connections with the tiers enabled and only its own
  `stdio` narrowed (`Host::serve`'s `narrow`: its `status`, instructions and errors name the narrowed set, and a call
  left out names `--tiers`, not `enable`); through a proxy `status` names the host's tiers. `--allow-dir` is the host's:
  a proxy's counts only once it hosts, and then for every client of the host.
- Kept queue and jobs. In `<config>/x5desk` (under `--sim` and `X5DESK_TEST` their temp dirs): `mcp-downloads`, cam's
  queue file format (the app's own queue, `downloads`, stays apart), `mcp-jobs`, a JSON object per job and line (handle,
  keys, files with the state a later job took them over in and, once a file's download finished, the bytes the camera
  sent; creation time, cancelled), and `mcp-renders`, the render jobs' (Renders, below); lines that do not parse (no
  UTF-8, no job, a time the system cannot hold) are left out and logged. A host holds `mcp-downloads.lock` (0600,
  `flock` / `LockFileEx`) from its start until its worker is gone and the files are saved the last time; a worker a
  camera call holds up (30 s at most) is waited for 5 s (`ENDING`), then the queue and the jobs are saved without it,
  its queued downloads paused in memory so it starts none, and the lock let go. The next host, elected while the last
  one still ends, waits for the lock up to 15 s (`HANDOVER`: the end's 5 s for its recording, `STOPPING`, and 5 s for
  its worker; its renders' 8 s, `RENDERS`, run alongside them from its start); one whose wait runs out (another
  endpoint's host, a hung process) keeps its queue in memory and logs it, as one does whose files are there but cannot
  be read (a queue file that is no UTF-8 is moved to `mcp-downloads.bad` instead). What it takes is checked, a forged or
  hand-edited file in mind: only downloads and jobs into the agent folder (lexically, or its canonical path; no `..`, a
  safe name) from a camera URI or a card's `DCIM` (no `..`); a download whose file is there without its `.part` loads as
  done (it finished uncollected, or the file came meanwhile: never fetched over it). Every change is saved (the queue
  first, so a kept job has its files in it), and the worker saves at the end of each run. At the worker's end every
  running download, a card import too, goes back to the queue stopped (`dl::Queue::quit`; nothing writes its `.part`
  after), so the next host resumes it: queued camera downloads once the camera is connected, a card import at once, a
  cancelled one when its item is downloaded again (`place` finds its entry). It keeps every unfinished job, those with a
  stopped file whose `.part` is there (their finished files stay in the queue, so the item resumes at its number) and
  the newest 100 (`job::KEEP`), and of the queue what waits, what a kept job has and a stopped download whose `.part` is
  there; a finished file's size is read for the jobs that show it, and its host id is one no kept handle of either kind
  uses.
- Endpoint. Linux `$XDG_RUNTIME_DIR/x5desk/mcp.sock`, without it `<cache>/x5desk/mcp.sock`; macOS
  `<config>/x5desk/mcp.sock`; the dir owner-only (`cam::dirs::private`); a path that does not fit `sun_path` (104
  bytes on macOS, 108 on Linux, NUL included) moves to `/tmp/x5desk-<uid>/mcp.sock` in a dir of the user's alone
  (`cam::dirs::mine`). The election runs under an `flock` on `<socket>.lock`: connect if a socket of this user's
  answers, else remove the stale one and bind (socket 0600). A client checks the socket file's owner before
  connecting and the peer's uid after; a host closes connections of other users (`peer_cred`). A host removes its
  socket at its end under the lock, only while it is still its own (device and inode) and without waiting for the
  lock. Windows (type-checked only): `\\.\pipe\x5desk-mcp-<user SID>`, the first instance made with
  `first_pipe_instance` (of two starters one hosts, the other connects), `reject_remote_clients`, a DACL of the
  current user alone (`D:P(A;;GA;;;<SID>)`), the next instance made before a connection is handed off; a client
  checks that the server process (`GetNamedPipeServerProcessId`) runs as its user.
- Tiers: `read`, `local`, `camera`, `delete`, `untested`, enabled once, no per-call approvals; `delete` and `untested`
  only by name; nothing enabled: no tools and instructions saying how. `untested` has no tools: it lets the camera tier
  use modes, settings and shutters x5desk has not run on an X5 (`cam::conf`'s lists), and favourites (the X5 on fw
  v1.13.21 refused them). The GPU tools: `catalog` in `read`, `render_still` and `export` in `local`, listed and run
  only with a backend (`server::listed`: the app's engine; none in mcp alone, as in its own tests (Windows: later),
  where calling one says "this x5desk has no GPU tools"). `--sim` and `X5DESK_TEST` take the tiers from
  `X5DESK_MCP_TIERS` (a comma list), else all but `delete` and `untested`, and read no config file; without `--sim`
  (`Opts::dir`, a real camera) the environment gives at most `read,local`. An accepted risk of `camera`: a Wi-Fi
  connect (`camera_connect`) sends the client id the camera authorized for the app (its `id=`, else the agents' own
  `id` in `<config>/x5desk/mcp`), only to a private or link-local address (`lan()`), but a device on that network
  answering as a camera gets it.
- Capture and the camera's files (`server::capture`). `capture_status` (read) shows the mode, the modes and the current
  mode's settings with their values (and how they read), the values allowed now (one allowed: fixed by what it depends
  on), whether each ran on an X5, whether the camera captures and whether this host started it. The camera tier:
  `capture_mode` (a name or label; not a live mode), `capture_set` (a value as `/CONF` spells it, a number or boolean
  spelt so; answered once the camera reports it), `capture_photo` (the self-timer counted down by the host, at most 30
  s, a cancel before the shutter taking none, also while it waits its turn), `capture_record` (start; start with
  `seconds`: the host's timer stops it, again every 5 s while that fails, also once a lost link is back, up to 45 s the
  call answers the clip, refused while this host renders or a run of it was settled stuck, Renders below; stop; wait),
  `capture_mark`, `camera_favourite` (needs `untested`); the delete tier: `camera_delete`. What each checks goes first,
  nothing sent when it fails: the camera connected and its capture state read (waited for up to 15 s after a connect;
  writes and the shutter with the mode's values), the `/CONF` lists and the values allowed now, not while the camera
  captures (capture tools, favourites, deletes), not while a recording this host started or its start is on the camera
  (another start, a mode or setting change, a delete, the release), and what ran on an X5: 360, a sub-mode the protocol
  knows, listed in `/CONF`, in `CHECKED` / `SWITCHED` / `SHOT` (highlight marks: VIDEO_NORMAL), else only with
  `untested`. The camera-writing tools take turns (`Shared::act`); a capture command goes as asked (numbered, with the
  camera as it was checked, `Seen`), its own outcome waited for, and a tool finding an earlier one unanswered says so. A
  recording this host started (`host::Rec`, from the moment its start is asked, kept across that camera's sessions: a
  link lost while it records) ends with its stop's outcome, a refused start, the camera saying it stopped, its capture
  state back to idle in a session that showed it capturing, a new session's first whole read showing it idle (a load
  that failed says nothing), or a capture state read more than `host::LATE` (10 s) after its start went out (a timeout
  counts as gone out) still showing none; a stop meanwhile that finds nothing keeps it and is tried again once it shows
  (the timer, every 5 s; `Rec::due`, which makes no untimed recording timed: renders go on). A capture found running
  after a lost link is taken as this host's (the camera's capture time cannot tell: a recording the user started while
  the link was gone is stopped with it). The host's end stops it on the session directly, reading the capture state
  until the camera captures (then stopped), the recording is known to have ended, or `STOPPING` (5 s) passed (the log
  says when it may still run); while it is on the camera, the camera is not released for idleness. A recording another
  process's client of this host started also stops when the host ends. `camera_delete`: the camera's album only (a
  card's: refused), every key listed (else nothing deleted), none being downloaded; while it runs no other camera write,
  `card_open` or `card_close` goes (its note ends it); its answer is judged on the camera's listing after its note: per
  item deleted, partly (files gone and left) or not, how each was kept here (`verified`: a finished download of that
  very URI whose file has the size the camera sent, as the jobs keep it, so in later hosts too; `by_name`: files of
  those names in the agent folder or, on a real host, the app's download folders; `none`), the never downloaded named;
  `downloaded_only` keeps those not verified; a key this host deleted answers as already deleted.
- GPU tools (`server::render`). `catalog` (read): what the other two accept, from the app's tables (presets, the video
  sizes: 360 widths without the photos' 11904 and the flat sizes, codecs, rates and ranges, frame rates, looks with
  their names and aliases, the editor's adjustments, stabilisation modes, the flat projections, the still's preset-free
  defaults: 360 at 7680, flat 1920 × 1080, quality 92; the blur targets, `faces` and `plates` with their aliases, and
  the pass: one source frame in `blur_every` (3) searched, best effort, what it costs, unavailable with why while the
  GPU is up without burn; what agents can't do yet: follow, photos, LUTs, overlays, timelines, Shot Lab), narrowed to
  what the device takes once it is up, and the GPU's state; it never starts the GPU. `render_still` and `export` (local)
  take an X5 recording under the read roots (an `.insv`, paired by `cam::work::pair`: an `.lrv` or `_10_` file finds the
  `.insv` beside it; a proxy alone, a photo or an `.mp4` refused, saying what to give) and write into the agent folder
  or a `subdir` of it, under `name` (a safe file name of 120 bytes at most, the kind's extension stripped, not in a
  camera file's pattern) or `<stem>_<tag>` (the app's `stem_tag`; blurred `<stem>_blur_<tag>`, agents' only, so a
  blurred file is never taken for the sharp one), numbered past a name in use. `blur` (both tools; off unless asked):
  catalog blur ids or aliases, one or a list, or `true` for both. A call is checked in order before the GPU starts: its
  arguments (a feature agents can't use yet, the other tool's field, or a word agents may guess for `blur`, such as
  `faces`, `face_blur` or `redact`, said by name, one set to no told to leave it out; a `blur` of the wrong type), every
  id and range against the catalog (an export's video size from its lists only; a still's free: 360 a multiple of 8, 512
  to 8192 wide, flat 16 to 8192 a side), the host (its jobs file its own, not ending, no timed recording of its own),
  the source, the backend's check of it (`Render::check`: trailer, both lenses, calibration, IMU, rate, length, tracks;
  stabilisation `horizon` by default when it has IMU data, else `off`; with blur: a recording in a file per lens and
  `mcp-noattach` refused, the clip opened as the pass reads it, shotlab's IMU: an asked stabilisation it has none for
  refused, a defaulted one `off`, and the frames its pass searches counted, `Plan::search`), the device's limits once it
  is up (the texture size, the source's tracks among what it decodes, the codec, size and rate among what it encodes,
  burn there for RAFT and blur), then the destination (made only inside the agent folder, UTF-8, not on FAT or exFAT,
  its free space) and the reservation (Renders, below). An export starts from `pano57` unless it names a preset: none of
  the GUI's settings reach agents. RAFT is the default seam unless the GPU is up without burn (then `raft: true` is
  refused with why); `blur` is then refused with why too (`render::no_blur`: mcp's device check, the backend's check
  under `mcp-noattach` and its submit), with no default to fall back to, so a blurred job parked while the GPU came up
  so fails at its submit: a blur asked for is never dropped. `render_still` waits up to `wait_s` (50 s, also the most:
  `server::WAIT_MAX`) and answers a done still as `album_thumbnail` answers: its JSON in a text block and a preview
  after it (a long edge of 1568, under 1 MB at quality 85, 70 or 55; the newest 8 kept), no structured content; else the
  job, which `job_wait` finishes with the image; once reserved, the call's cancel leaves the still to finish. `export`
  answers its job at once with the length, frames and size and a time guess from the last done export of that size and
  seam (its blur pass taken out: `renders::render_secs`, a record skipped when under 1 s or 10 frames are left), for a
  blurred one the pass first, from the last pass of those targets on this server (else "roughly 2 to 2.5 times the
  span's length on a Radeon 8060S", as measured there; the tool texts say so too). Agent jobs are strict: an export
  fails at the first picture that could not be decoded (after the Mac slept, a damaged GOP), its error naming
  `allow_frozen: true`, which exports it with repeated frames and counts them (`concealed_frames`); a still ends done
  with `concealed`. The job tools take both kinds under one numbering: `job_status` and `job_wait` (50 s a call at
  most, then called again) in frames, with the settings and a phase (`gpu_starting`, `waiting`, `detecting`: the blur
  pass, `setting_up`, `compiling`: the first RAFT flow, `frames`, `finishing`), progress the frames searched and made
  plus a creep while none comes (a quarter of what is left to the next every 5 s) out of the frames to search (the
  record's `search`, so the total holds from the first answer) + the frames + 1, a notification a second at most (while
  detecting the frames searched and the pass's time left, before the first "loading the detectors"). A blurred job's
  result has a `blur` object: its targets, the frames searched and to search, while detecting the pass's frames/s and
  seconds left (the top-level `fps` and `eta_s`, the frames', null then), once done what it found (detections and
  tracks per target asked, the frames blurred of the span's, the pass's seconds; `End::found` in the record; one that
  ended before done keeps the frames it searched, `End::searched`) and a note: detection is best effort (it can miss
  small, distant, turned-away or partly hidden faces and plates), as the schema, the tool texts and the catalog say
  too; a job whose pass kept no track (plate detections `Hide::plates` does not hide count for nothing) ends done, its
  note saying nothing was blurred. A still's pass searches one frame, where no plate track can pass
  `Hide::plates`, so its plates are a photo's (`Ai::plates_still`: the sure ones), as the tool texts say. `job_list` newest first, `kind` and
  `limit` (50, at most 500), render rows short; `job_cancel` ends a render with no file and no `.part` (waited for 3 s,
  else "stopping at its next frame"). `status` shows the GPU without starting it and counts both kinds; a card's
  `album_list` items carry their first file's path.
- The GPU (`export::agent`, the app's engine behind `mcp::render::Render`; mcp has no GPU dependency, and its side is
  tested on a fake). One engine per process (`ENG`, never dropped: dropping an `export::Queue` joins its thread, which
  may be in the driver), its device opened at the first `render_still` or `export` that passed its checks (never by
  `catalog`, `status`, a proxy or a refused call) and kept until the process ends. The host parks such jobs, and one
  task of its own starts the GPU on a blocking thread (`Start`: burn's base, the child's executable, the log sink,
  `halt`; a start under way is waited for), then submits them, or fails them with why. First the self-test in a child,
  `<exe> mcp gpu-check` (Linux `/proc/self/exe`, macOS `current_exe`; run in the temp dir; its stdin a pipe, so a
  killed host takes it along; its stdout's line read back, 16 KiB of its stderr into the log; killed after 20 s or at
  the host's end), so a driver that crashes or hangs at its first init, or decodes wrongly, takes the child down, not
  the host and its camera session. Its steps: (1) the device (Linux, with no adapter at all: no render node, a VM's or
  container's; one the user can't open, the `render` group; else the Vulkan driver), (2) the pipelines
  (`stitch::Ctx::new`), (3) the X5's H.265 Main at 3840 × 3840 and the decode caps
  per profile (`video::check::h265`, `caps`; Intel's coinciding DPB mode refused; the NV12 / P010 output query only
  logged), (4) the pixels (`check::pcm` at 256 × 256: three H.264 High I_PCM IDRs, luma 16, 128 and 235, each with a
  skipped P picture, more frames than the ring has, every luma sample read back exact, none concealed; skipped when the
  decoder refuses High), (5) the encode caps and one picture through each encoder (one that fails is left out: with
  none, stills only); it never attaches burn, so it takes no cubecl slot. Then in process: `gpu::open` with the
  host's policy (burn attached in it; failed or turned off: RAFT off), the pipelines, the caps (the child's when its
  adapter's vendor, device and driver are the host's, else steps 3 and 5 again) and the lanes; one log line names the
  adapter, backend, driver and which of `DRI_PRIME`, `__NV_PRIME_RENDER_OFFLOAD`, `VK_ICD_FILENAMES` and `WGPU_*` are
  set. A start that failed before the process had a device is tried again at the next GPU call after 60 s, then 10
  min; three crashed or hung children turn GPU tools off for the process, as does any failure once it has a device (a
  second device and attach would leak burn's runtime); a halted start counts nothing, a panic in it is a failure. Two
  lanes, each an `export::Queue` on the one device (`x5-still`, `x5-export`), so a still never waits behind an export.
  A job is the one the Export screen would queue (`set_of` sets what its widgets would, the Blur boxes too; `span`,
  `clip_job`), written into the `.part` mcp reserved (`Dest`, strict), its frames checked against the plan's at the
  submit ("the source changed since it was checked"). A blurred job runs the Export screen's privacy pass first, on its
  lane (`Job::pre`: serialised with the lane's GPU work, its stop the job's, the lane `busy`), with the plan's
  stabilisation and targets at the source's rate, in a `scoped` of its own (a wgpu error or panic in it fails the job at
  once, "the blur pass: …"; never exported without it), the models held per searched frame (`ai::MODELS` serves one at a
  time: a still waits out one detection of the export lane, not its pass); then the pass's hook blurs each frame in the
  job's fixed view. Its first use in the process loads the detectors (YuNet on Core ML ≈ 0.6 s, on burn 11-17 s of
  compiles; YOLOX and the plate detector compile per new batch size, also mid-pass), none of it cancellable. Blur is
  gated on burn as RAFT is: `track::Ai::new` would attach burn, and cubecl's panicking handler with it over the host's
  `Log` policy. On the M4 with the earlier plate detector (YOLOv9-t; the user's recording, 2880² lenses; faces and
  plates, so 7680 wide) the pass searched 6.25-6.3 frames/s, about 19 source frames/s: a 10 s span detected for about
  16 s before its first frame, a whole 818 s clip for about 22 min; a blurred 360 still at 3840 took about 1 s once
  warm (its pass 0.4 s; the first with every cache cold 28 s), the first searched frame came about 0.5 s after the
  submit (3 s when the server loads the detectors), and blur's `check` adds about 55 ms. On Linux with ours (Radeon
  8060S, faces and plates, `x5_blur` on 2026-11-13) a 3 s span searches 5.1 frames/s on v635 and 4.1 on v606 (6.1 and
  7.5 s, 2 and 2.5 times the span); a blurred 360 still at 3840 takes 0.6-1.6 s once warm (its pass 0.2-0.8 s; the
  first with a cold cache 10-11 s); over the test (passes, stills, 4K exports) the host's RSS peaks at 356 MiB, the
  Radeon's VRAM at 1.4 GiB above the desktop's and GTT at 0.25 GiB. burn's base: `<cache>/x5desk/cubecl-mcp`, under
  `--sim` and `X5DESK_TEST` the test hook's `x5desk-cubecl` (App).
- GPU faults. The headless policy (`gpu::Policy::Log`, App) never panics on an uncaptured error; each lane polls the
  device after a job (an error there, or its thread gone, makes the GPU suspect too). The poller (one task per host,
  every 200 ms from its first render job until it ends) writes each run's state into its record (readers never ask the
  backend), settles each that ended (a done still once its preview is kept), and probes the GPU at once after a job
  failed as a lost device fails one or a watchdog cancel, and a suspect one at most every 30 s: `poll(Poll)`, then a
  4 × 4 clear read back within 5 s. A failed probe makes it `Dead`: GPU tools off until the server restarts (Claude
  Code: `/mcp`, reconnect), every run of the host cancelled and settled failed 15 s later at the latest. The watchdog
  cancels a run that does not move for 120 s making frames, or 180 s in its blur pass (a searched frame is a move; the
  first loads the detectors), setting up or in the first RAFT compile (never while it finishes its file: the faststart
  shift reads its stop per 4 MiB), whatever the probe says (a decode stuck on its own queue passes one). A run not ended
  15 s after such a cancel or a lost device's (`GRACE`) is settled failed, its lane wedged when the watchdog stopped it
  (the lane's queued jobs failed, new ones refused until the server restarts; the other lane works), and from then on
  timed recordings are refused (`Shared::wedged`: its thread may still be in the driver, which may take the process
  down). After 5 min without an unfinished render job of the host (once per idle spell) the GPU is trimmed: `Gpu::trim`,
  each lane drops its RAFT model (about 150 ms to load again), and the blur's detectors go unless a pass holds them
  (`Models::trim`, under BURN: on the M4 with the earlier plate detector a faces and plates pass left about 160 MB of
  them, faces alone 135 MB, not measured with ours or on Linux; they load again in about 20 ms, burn's kernels kept). A
  blurred job waiting for the models sees its stop (`Models::with_stop`), so a detection wedged on the other lane
  leaves it cancellable, and once that lane is wedged blur is refused while it holds them (`Models::holder`,
  `GpuInfo::blur_stuck`: the backend's check and submit, mcp's device check, the catalog and `status`;
  `render::stuck_blur`), and RAFT too, whose flows would wait for its BURN past their stop: off by default, refused when
  asked (`render::stuck_raft`); a decoder wedged in the pass (shotlab's frames never look at the stop) goes the way of a
  wedged driver. While one is unfinished the poller holds `cam::awake::keep()` (macOS: no idle sleep; Linux: nothing
  yet).
- Renders: records and files. Render jobs are kept in `mcp-renders` beside `mcp-jobs`, under its lock, a JSON object a
  line (an older x5desk never reads it), their handles from the jobs' numbering; never resumed by another host. A host
  keeps every unfinished one and the newest 100 finished (`renders::KEEP`, its own pruned as they end; downloads keep
  their own 100). Renders don't hold the camera. The reservation, under `jobs` → `q` → `runs`: the host not ending
  (`closing`, set by the end's sweep under `jobs`), at most 16 unfinished render jobs, the free space
  (`media::part::avail`) after what this host's other jobs on the volume still need (their estimates less what they
  wrote) at least the job's estimate plus 2 GiB (a still: 256 MiB); the record saved before its `.part` exists (a host
  whose records are in memory only, its jobs files another host's or unreadable, refuses GPU tools; a record that can't
  be saved refuses the call); the `.part` made new at the first free name (numbered past a file, a `.part`, a dangling
  link, an unfinished download or render there) and its identity saved. The estimate: a still w × h / 2 bytes; an export
  its rate and 10 % over, or at a QP 1 bit a pixel at QP 24 for a 360 (as if 3840 wide: its bits grow as the square root
  of its pixels) and 1.75 for a flat one, doubling every 6 QP lower; then the AAC and a still's worth. The `.part`
  (`media::part::create`: `O_EXCL`, never through a link) is handed to the job, which writes through it and ends in
  `media::part::finish`: the stop checked after the last write, the `.part` still the open file, a publish that never
  replaces (a file that appeared meanwhile fails the job and stays as it was), every other end removing it by identity
  only. Every cleanup (the poller's, a cancel's, a failed submit's, the end's, a lost device's, the next host's) goes by
  that identity (`media::part::settle`), so a file that took either name meanwhile (a download's, the GUI's, the user's)
  is never removed. An agent folder or subfolder on FAT or exFAT is refused: their file numbers can change across
  mounts, on macOS as a file is written, so a record can't follow its file. The next host settles what a crashed or
  killed one left, by identity, before it prunes: a published file reads done (a `.part` name the link fallback left
  beside it removed), anything else cancelled ("its host ended before it finished"), its `.part` removed if it is still
  that file; a record without an identity (a crash in the µs before it was saved) or on FAT or exFAT unlinks nothing,
  its `.part` left and logged; records not from absolute sources to an `.mp4` or `.jpg` in the agent folder are left
  out. Downloads' `.part` files are never touched. Timed recordings and renders refuse each other, both counted under
  `st` (`rendering`, `timed`): a crash mid-render would lose a timed recording's stop, this host's timer; untimed ones
  coexist (a crash leaves one running, which `capture_record stop` in the next host ends).
- The host's end with renders. The pump's exit (`Host::shutdown`, `Drop`, a dead worker) sweeps first, as
  `Host::closing` does earlier at the hosting client's stdin end: no job is reserved any more, a parked one is settled
  at once, every run told to stop (no lock held), a GPU start halted (its child killed within 50 ms). Then the
  recording's stop and the worker's end, while the renders wind down; then each run that ended is settled, every 50 ms
  until none is left or 8 s after the sweep (`RENDERS`), and the rest as they are, by identity (cancelled, logged as
  "still in <phase> at the host's end"): a writer still in the driver writes into an orphan inode and publishes
  nothing. Nothing joins a GPU thread, so the end stays within `HANDOVER`. Once `mcp::main` returns, the app
  `_exit`s (no atexit, no driver teardown) while a GPU thread may still be in the driver (`export::agent::busy`: a
  lane with a job or a trim, a `Dead` engine's too, a probe or trim running, the engine starting; once it has looked,
  none starts), else it `exit`s.
- Logs: `<cache>/x5desk/mcp.log` (0600, scrubbed, a line per call and per host event, 1 MB then `.1`; the GPU's
  lines as `gpu: …`: its start, the self-test child's stderr and its pass, uncaptured errors, warn-level log lines, 20
  a minute; each render's reservation and end, a blurred one's with what its pass found in the GUI note's words and its
  frames/s without the pass); fronts say what they host or proxy, and every switch, on stderr.
- Test hooks: `X5DESK_MCP_ENDPOINT=<path or pipe name>` overrides the endpoint; without it `--sim` and `X5DESK_TEST` use
  none (they never probe or bind the user's). `X5DESK_TEST` without `--sim` keeps the agent folder and log in
  `$TMPDIR/x5desk-mcp-<pid>` and no thumbnail cache (`Opts::dir`), and its token `mcp-print` prints a line through
  `println!` before serving (which the guard puts on stderr); with `--sim`, its token `mcp-sim-dir` keeps the
  simulated camera's dirs (agent folder, log, records) in `$TMPDIR/x5desk-mcp-<pid>`, for a process test to read
  after the end, and `mcp-noattach` keeps burn off the device, as a failed attach would (RAFT off). The app's
  `tests/mcp.rs` runs the real binary (`--sim`, endpoint, home, config, cache and temp dirs in a temp dir of its own):
  every stdout line JSON-RPC and the forced print on stderr, stdin's end, SIGTERM and SIGHUP ending a host with exit 0
  and its socket and temp dir gone (one started with SIGHUP and SIGINT ignored surviving them, answering and ending at
  stdin's end), a client that writes its requests and closes stdin at once answered (by its own host and through a
  proxy), three processes sharing one host (a client of the second on 2025-11-25, of the third on 2026-07-28) and, its
  process killed, the other two electing one of them, the 2025-11-25 handshake replayed, a host started with
  `--tiers read` narrowing only its own client, and the command line (usage, `enable` / `disable`, `--print-config`); a
  socket path that would not fit `sun_path` (a long `$TMPDIR`) moves to `/tmp/x5mcp-<test>-<pid>/`, a Windows pipe is
  named after its test. The crate's `tests/tools.rs` runs the tools over a duplex on cam's simulated X5 (a connection
  narrowed by its own `--tiers` too; jobs sharing a download, and taking over one that stopped, failed or finished and
  went; a host ending mid-download while the next one waits for the queue, which resumes the `.part` once connected, and
  a third host finding the jobs as they ended; a host whose worker a camera call holds up handing them on after
  `ENDING`; render jobs an earlier host left, settled and listed beside the downloads; the GPU tools listed only with a
  backend); `front`'s unit tests play hosts (the handshake replayed, an open request answered, `--tiers` in a proxy
  against a BOM, batches, unknown names, the answers after stdin's end, a request the client cancelled not waited for);
  `endpoint`'s the election on a temp socket (in `/tmp` for a long `$TMPDIR`). Opt-in on a real X5 in Android App mode
  over USB: `X5_CAMERA=<scratch dir>` (`tests/camera.rs`: a host with its agent folder and log there connects without
  the clock, lists the album, downloads its smallest item and checks its size; a second host of the process is refused
  the camera, "in use by another app"; the camera is released); `X5_CAMERA_AGENT=<scratch dir>` (an agent's writes in
  the tested modes and settings, the camera starting idle in one of them: white balance changed and set back, a photo, a
  timed clip, a clip the host's end stops, a highlight mark, then the test's own items deleted; mode, white balance and
  file list as they began, also when a check fails; not `X5_CAMERA_WRITE`, whose app test sets the clock). The crate's
  `tests/capture.rs` runs the capture tools on the simulated X5 with capture: the checks before anything goes out, what
  goes out and comes back, the `untested` tier (and a client's `--tiers` leaving it out), the self-timer, a command
  whose camera call hangs and the next tool's own outcome, a start refused or sent with the capture state unreadable,
  one the camera begins late (the end waits for it and stops it), a timed stop retried after a lost link and one during
  a slow start, a start never begun (cleared after 10 s), a call narrowed by a proxy's `--tiers`, a reconnect whose
  first read fails and one 20 s into a recording, deletes and their answers, verified copies in a later host and a
  changed one, the host's end stopping its recording (also after a lost link), a recording keeping the camera from the
  idle release, the user's own recording left alone; no request ever sets the camera's clock or erases its card.
- GPU tests. mcp's `tests/render.rs` runs the GPU tools over a duplex on a fake backend (no GPU, no samples: every OS):
  `Fake` stands in for the app's engine with a catalog, a check of small text recordings in an `--allow-dir`, a start
  that can fail, take a while or panic, two lanes that write into the handed `.part` and end through
  `media::part::finish` as `export::run_into` does, and runs that can hang ignoring their stop. It covers the refusals
  before the GPU starts, the job limit and free space, timed recordings against renders (and a stop before the camera
  began), names never written over, cancels (running, submitting, parked), submits and every backend call failing or
  panicking, the still's image (also after its call's cancel), the host's end (runs running, queued, submitting, stuck
  in the driver: settled within 11 s, nothing joined), a call held in the check at the end, a record that can't be
  saved, FAT refused (`X5_FAT_DIR`), the next host's cleanup by identity (also after a host killed mid-export in a child
  process of the test), pruning by kind, the job tools by kind, a start that fails, rising progress, blur (the Fake's
  pass searches first: the refusals, its status and progress out of the frames to search + the frames + 1, a cancel and
  the host's end mid-pass, one held in its pass, a GPU up without burn; `tests/tools.rs`: a blurred record a later host
  lists) and the scrub's marker through every argument; `server::render`'s unit tests have the checks, the parking, the
  poller's probes and trims, the watchdog (a searched frame a move too), a dead GPU, the end and `Host::closing` at
  their `cfg(test)` times (1 s), the blur's result and time guess. `render`'s own `Fake` (a self-test that passes, fails
  or panics) drives `cli`'s `gpu-check` line and `server`'s listing; `cli`'s `reaps` starts a server's SIGCHLD ignored.
  The app's `tests/mcp.rs` runs the GPU tools on this machine's GPU, one server at a time, on `export::synth`'s
  recording (300 frames of 256² a lens): `gpu-check`'s one line, `mcp-noattach` (a still without RAFT, `raft: true` and
  blur refused), a 360 still and two exports followed to done, a cancel at 10 frames or more leaving nothing, a still
  overtaking a RAFT export (the lanes), blurred a still and an export followed to done (`gpu_blur_synth`), one cancelled
  in its pass and a still done while a whole-clip export still detects (`gpu_blur_lanes`, no RAFT: the models held per
  frame), stdin's end and SIGTERM mid-export, SIGTERM in the first RAFT compile (burn's cache cold: the end waits
  `RENDERS`, settles and `_exit`s), a proxied client's export ending with the hosting client's host; their
  `XDG_RUNTIME_DIR` is a kept `<temp>/x5mcp-gpu-run`, so the hook's cubecl cache stays warm across them, and on macOS
  `CFFIXED_USER_HOME` the test's home. Its `x5_samples` (`X5_SAMPLES`: `v635.insv`, else its first `.insv`) is the
  milestone on real footage: the catalog, RAFT stills 360 at 3840 and flat 1920 × 1080, 2-5 s exported flat 1280 × 720
  H.264 and as `pano4k` with RAFT, a whole-clip `pano57` cancelled at 30 frames or more, stdin's end with an export
  running (the files copied to `X5_EXPORT_OUT` when set). Its `x5_blur` (`X5_SAMPLES` too) asserts nothing of the
  content: blurred 360 stills across the clip find something and differ from the sharp one past its noise floor
  (rendered twice), a 3 s export's frames between the searched ones keep the blur (`Hide`'s bridge), a whole-clip pass
  with a still overtaking it, cancelled, and at stdin's end. Below them: export's `tests/export.rs` on the synthetic
  recording (the handed `.part` published, nothing replaced, a link at the `.part`, a stop in the last frame and in the
  faststart shift, a queued cancel, panics in a job and as it drops, two lanes at once, `detach`, strict and `frozen_ok`
  on a damaged recording; `Job::pre` first, once, on the lane, and a cancel, an error, a wgpu error or a panic in it or
  its hook failing the job with nothing left), `video::check`'s GPU tests (the streams, the pixel check at 256² and
  400², the X5's H.265 Main and Main 10, every profile's caps and encode probe), the app's `export::agent` unit tests
  (the catalog's tables, `set_of`, ranges, the settings shown, `busy` while a lane or a probe may be in the driver, the
  size estimates against sizes this Mac's encoders made; blur's check, a blurred run on the tests' device, `blur_run`:
  detecting, then frames, what it found, a cancel in its pass, a still after the trim (which dropped the detectors);
  `blur_stuck`: blur refused at check, submit and in the GPU's info while a wedged lane holds the models, RAFT off by
  default and refused at submit; `agent_grab_parity`: an agent's blurred still the Export screen's grab byte for byte,
  `X5_SAMPLES`),
  `export::tests::parity` (`span` and `clip_job` build the jobs `Keep::video` built), `privacy_synth` (the pass's counts
  and note on synth, recorded before the pass was factored out), `privacy_parity` (`Keep::privacy` against that old
  code, `X5_SAMPLES`, each case finding each kind it asks for: its plates-only second from `X5_PLATES_AT` s, 1 by
  default) and `ai`'s `with_stop` and `trim` (loaded models dropped). The app's tests that load Core ML in the test
  process (a face pass: `export::tests`' `plates`, `faces`, `timeline_faces`, `gate`, `privacy_synth` and
  `privacy_parity`, `export::agent`'s `blur_run` and `agent_grab_parity`, `ai`'s `auto_v635`) skip on macOS without
  `CFFIXED_USER_HOME` (`ai::tests::no_cf_home`), so they never write the real `~/Library/Caches`: set it, with the rest
  of a scratch home, for cargo runs there. Crate track's do too (their own `no_cf_home`): `tests::coreml`, and
  footage's `privacy`, `plates` and `highlights`.
