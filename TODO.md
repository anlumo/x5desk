# Todo

Merged on 2026-11-10 from the comparison with golblick, `target/plates-train/PLAN.md`, the open risks of the desktop
app plan (`insta360/PLAN.md` §5, 2026-10-22) and the feature inventory's open questions
(`insta360/re/analysis/app_features.md` §c). Items already done in the code were left out.

## Stitch: takeaways from golblick

From comparing our stitch with golblick's notes
(<https://github.com/AkshayRao27/golblick/blob/main/docs/formats/insta360-agent-notes.md#stitching-the-seam>;
comparison page <https://claude.ai/artifact/WzKEEXqGvqmLj1ojXMq9p8>). Their figures are theirs, on their clips.

- [x] **Two bands on the template path.** Without flow (live preview `fs`, exports with flow off) `stitch.wgsl`
  `seam` hands detail over across a narrow fade, min(3°, max(1.5°, 4 rows)), and the lenses' difference blurred
  over about 3° across the 6° ramp (golblick); the blur comes from `low`, a pass over each lens into blocks about 3°
  across (decoded NV12 has no mips), recorded with the stitch and in `Preview::colour`. The flow path (`shade`,
  `main`, `ff`) is untouched: RAFT frames identical before / after. 24 frames each of 635 / 606 / 627 (5.7K, colour
  fit on), against the single ramp: ghosting 1.66 / 1.59 / 0.80 → 0.56 / 0.54 / 0.32 levels and seam error
  (gradient domain) 64 / 52 / 37 → 32 / 26 / 19, lower on every frame; the low band's step 2.20 / 0.92 / 0.30 →
  1.96 / 0.72 / 0.33 (627 worse). Cost on the Radeon: +0.3 ms a 5.7K frame (0.7 → 1.0 ms), +0.2 ms a new
  preview frame. Open: on these clips a lone 1.5° fade scores as well; the low band pays only on a brightness step
  between the lenses (lens 1 × 0.8, colour matching off: step 4.4 lone fade, 3.0 two bands, 3.1 single ramp), and it
  spreads a very near object's edge into a soft 6° haze (635 railing). Scoring: `target/bands-scratch/harness`.
- [ ] **Lens guards.** With guards on, the mask at 200° (`offset.rs` `pack`, q[16]) blends the guard's rim into the
  overlap. golblick reads a stored guard factor of 1.0225 (field of view about 2.25 % narrower) and picks it per
  photo by scoring the overlap at both angles. Find the field, shrink the image circle when set; needs a clip shot
  with guards.
- [ ] **Score colour matching against the app's export.** golblick found lens balancing worse than none against the
  vendor player (error 4.0 with, 2.4 without; near content moved 10–13 levels). Ours (`Opts::colour`, on by default:
  `app/src/export.rs` `colour: true`) was never scored that way. Error against an app export of the same clip, on
  vs off; turn the default off if off wins.
- [x] **Decline rule for the flow.** golblick keeps the straight seam unless the alternative cuts the disagreement
  by 10 % (and the straight seam disagrees by 2 grey levels or more). Ours (`Flow::decline`, on in `Flow::RAFT`;
  `post.wgsl` `judge`): per 10° stretch of the seam, on the belt RAFT saw (no lens samples, no readback), the mean grey
  |lens 0 − lens 1| over ±3° straight and warped with the new flow, smoothed over time (0.1); a stretch failing either
  test fades its flow to the template through the flow's own smoothing, weights linear between stretch centres.
  Decided per frame the stretches flipped on 3–20 % of frames, and each flip moves the flow by its full size: wobble
  +14 % / +12 % on v635 / v606 H. A per-frame decision over the whole seam never declined on these clips. Scored by
  `tests/stitch.rs` `seam_score` (ignored; the belt dump's seam ±3° and wobble, sampled from the lens pictures on the
  CPU), 40 frames at the start / middle, RAFT always → rule: seam v635 8.17 → 8.19 / 7.76 → 7.79, v606 6.80 → 6.80 /
  5.14 → 5.14, v627 (night) 4.79 → 4.81 / 4.04 → 4.10; wobble 0.410 → 0.372 / 0.388 → 0.388, 0.412 → 0.411 / 0.404 →
  0.398, 0.326 → 0.276 / 0.184 → 0.164. So the wobble drops on all six (−5 % mean) but the seam rises on four (+0.3 to
  +1.4 %, mean +0.4 %): where the 2-level floor declines dark ground the flow still helped, and on flat walls where it
  cut about 10 %. Night bike rides (669, 713): seam −2.2 to +0.4 %, wobble −24 to −36 %; there RAFT mangles close
  bike parts and lights the template leaves whole. Declined: 1–13 % of stretches by day, 32–52 % at night. Export
  speed (v635 5760 Main 10, Radeon) unchanged: 12.74 / 12.80 fps against 12.77 / 13.01. Open: the belt reads 20–30 %
  less disagreement than full resolution, so the floor acts at about 2.5 levels there.
- [ ] **Levelling check without a rig.** Correlate the stitch with the camera's own stitched preview (trailer
  thumbnail) to check the IMU axes and the Mahony gains independently. golblick: FFT correlation, global 256 × 128
  then 512 × 256, 47 of 48 X5 stills aligned.

## Plate detector (branch `plates-yolox`, worktree `target/plates-wt`)

Training and evaluation tools in `target/plates-train` (`PLAN.md` there). Held out for good: the 2026-03-04 night
ride; held out of the flag training: 2026-04-16 and the sample clips v606 / v627.

- [x] **`Hide`'s timing** (c856743). Most "wrong blur" flags weren't detector mistakes: a plate passed within
  metres sweeps about 4° a searched frame, beyond the linking reach, so each detection became a track of its own,
  held still for 0.25 s on both sides (a row of blurs beside the plate, 722 frame 2829). Fixed in `privacy.rs`
  (`SWEEP` 30°/s for a track's second detection within `SOON` 0.25 s, `CARRY` 0.15 s, `BRIDGE` 0.5 s; plates only).
  Third gallery (`scripts/views3.sh`, changed views only via `views_diff.py`), reviewed 2026-11-11: 18 wrong blurs in
  17 of 78 views, no missed plate; blurs over a plate hidden for a few frames are wanted. Most were detector mistakes
  flagged before at nearly the same spot; the new ones came from `SWEEP` over gaps of 0.3–0.8 s, joining a stray
  detection to an unrelated plate track, hence `SOON`. Rendered again with `SOON` (`views3.sh views4`; the Rust output
  matches `rule_eval.py`'s replica): under the current rule blurs on nothing 23 → 14, on unreadable plates 26 → 19,
  plates covered 239 → 241, marks uncovered 8 → 7; third gallery 14 → 10 flagged, none of 74 lost; `plates_flags`
  89 → 71 of 1,003 still blurred, 265 of 280 kept; `plates_rides` 2 of 85 uncovered (was 4; both unreadable), none
  of 32 readable; `plates` unchanged; models/README updated. Left: a carry past a car that turns its plate away (740 frame 3078),
  a backward carry from a plate that comes out from behind the camera mount onto a headlight (718 frame 2258), and
  detector mistakes the timing only moves (717, 737, 741, 689). Faces still use the old timing.
- [ ] **Hard-negative fine-tune: shelved.** Only 56 of the 344 junk flags had a detection at the flagged frame, and
  even those included real plates (the flag judged the blur `Hide` drew, not the detection), so the labels would teach
  the detector to miss plates. Revisit only with blurs that sit on a detection of the very same frame. Tools stay:
  `dump_flags.sh`, `make_flags.py`, `train_flags.sh` (watchdogs fixed to kill the whole scope).
- [ ] **The 669 false positive.** The gap between a scooter rider and the scooter (2026-03-04, held out): a real
  detector mistake, still open with the fine-tune shelved.
- [x] **Re-pick `Hide`'s plate rule** (0018c58). A track is now also hidden
  with `MANY` 8 detections of score `VERY` 0.8 or more within one second, at any width (`privacy.rs`); `TWICE` sure
  detections stay. On the reviewed views (`rule_eval.py`): blurs on nothing 14 → 17 (a car seat behind glass 720/803,
  a white wall 690/2490, a car's flank 691/934), on unreadable plates 19 → 22, plates covered 241 → 338, marks
  uncovered 7 → 2 (692/56 and 717/2638, which no rule reaches). Dropping the width gate outright: 34–36 blurs on
  nothing. A single high score adds wrong blurs and no plate (689 frame 807 was covered already). Whole segments
  (28 min): 359 more hidden tracks (2,113), blurred track-time +18.6 %; 19 of 20 sampled additions are car plates of
  13–28 px. `plates_flags` 38 of 741 wrong still blurred (was 32), 205 of 216 kept, no mark uncovered (was 5 of 8);
  `plates_rides` 2 of 85 (both unreadable), 0 of 32 readable; `plates` unchanged. A 60 fps clip gets twice the
  chances at 8 detections; the variant "80 % of the second's searched frames" (not built) scores 13 / 18 / 276 with
  every coverable mark covered. Fourth gallery (`views6d`, software decode; flags in `views6d/flagged`): the rule
  adds 159 regions in 121 views and removes none; the user flagged one of them (the car seat 720/803) and marked
  no missed plate. Three more flags are regions the old rule draws too (720/803 a second one, 689/2141, 689/2999).
  The 80 % variant drops the car seat (15 of its second's 20 searched frames) and with it 86 of the 159, all at
  60 fps; 75 % keeps the seat and still drops 69. So the count rule stands unless the user says otherwise; the
  car seat is the detector's mistake. The user kept the count rule (2026-11-13).
- [x] **`flags_fx.py` and the fixtures.** Regions are labelled by the views that show them whole (wrong in all of
  them: 741; plate in none flagged: 572, of which 216 are 30 px or wider; the rest, 462, left out), with both
  galleries' flags and the 8 marks (`eval-fx/flags_regions.txt`, `flags_marks.txt`; the old set in `eval-fx/flags1`).
- [ ] **Plates no vehicle crop holds are never searched.** The van plate of 717 frame 2638 (about 50 px) is one: the
  van's front sits between the rider's legs and the vehicle model never scores it above 0.2, while the plate model
  finds the plate at 0.79–0.91 on any crop of 20° or less centred on it (0.42–0.85 at 24°, nothing from 48°). The
  vehicle detector also misses close-ups (in the downloaded stills: 1080 of 1747 greenpack, 455 of 915 Tgmt) and cars
  passing within metres (692 frame 56). A search that doesn't depend on a vehicle needs a plate detector that knows
  scenes without one first: ours was trained on vehicle crops only.
  - Tried and dropped: a ring of 18 crops of 22° along the horizon at pitch −3° (the sure plates' median; 95 % of
    them lie within its ±11°) next to the vehicle crops, each half on alternate searched frames
    (`target/crop-scratch/ring.patch`). It covers both marks, the regions squarely on the plates, but shown crops
    without a vehicle the plate model fires on lettering, signs and posters, the rider's own legs, bumper corners and
    tail lights. On the reviewed views (a run on the 4090, both searches on the same vehicles): 35 new regions, by eye
    5 on plates, 4 on unreadable plates, 26 on nothing (a rental truck's side lettering became a 15° plate, also in
    `plates`: v606's area holding no labelled plate 0.17 → 75.8 deg² a labelled frame), 5 gone (1 plate, 1
    unreadable, 3 nothing); `rule_eval.py` blurs on nothing 17 → 21, on unreadable plates 22 → 20, plates covered
    338 → 334, marks uncovered 2 → 0, regions on no labelled spot 14 → 40; `plates_rides` 2 → 1 of 85 uncovered (a
    49 px plate nobody can read), readable 0 of 32 either way; `plates` covers the same labelled plates. Cost on the
    Radeon (vehicle detection and plate search a searched frame): a quiet clip 3.0 → 12.0 crops, 29.1 → 37.5 ms
    (34.4 → 26.6 frames/s), a busy one 20.4 → 29.4 crops (41 at most), 42.8 → 51.2 ms (23.4 → 19.5 frames/s); on the
    4090 over all reviewed frames 13.2 → 22.2 crops, 38.3 → 50.0 ms. The whole ring on every frame: 58 regions on no
    labelled spot (61 new regions, 42 on nothing by eye) and 9–10 ms more again. Skipping ring crops a vehicle crop
    already shows saves nothing (1.4 % of them lie 90 % inside one). Next if retried: hard negatives from the ring's
    own hits (their labels are clean, unlike the flags), then measure again.
  - Unlevelled exports (stabilisation off): the camera's mount tilts the stitch's horizon by about 60° on the rides
    (8–12° on the walks; from the IMU, not checked on an image), and every crop of the plate search is upright in the
    pano, so the search there sees vehicles and plates on their side. Searching a levelled stitch and turning the
    regions into the export's frame would fix the ring and the vehicle crops alike.
  - Tried and dropped: a crop where a live plate track is predicted (`target/crop-scratch/item2-track3.patch`). It
    closes 692 frame 56's gap, but on the reviewed views it trades 3 flagged blurs for 8 new ones (a parking sign's
    logo, bumper corners, a tail light) at +0.95 crops and +1.5 ms a searched frame, and 717's plate has no
    detection within 15° to follow.
- [ ] **Corrupt frames from the hardware decoder.** The footage tests decoded with `ffmpeg -hwaccel vulkan` on the GPU
  the networks run on; while they ran, a run of frames up to the next key frame now and then came back corrupt (half
  the luma pixels off, 17–19 levels on average), differently in each process, and the stitcher's smoothed colour fit
  carried it to the end of the clip. This was the run-to-run "noise" of every unpaired GPU comparison so far; the
  networks and the NMS are bit-exact on the same input, except in the first four clips a process searches, where
  0.4–8 % of the detections differ in the third decimal from run to run (whichever clips come first; no rule outcome
  changed; burn's kernel autotune at start-up is the suspect, unproven). Fix in `crates/track/tests/footage.rs` (software decode on Linux; `plate_views`
  takes 1.8× as long; 1fb1159). The app's own export is clean: 12 exports with plates of two clips (722 at
  60 fps, 604 at 30 fps; `Keep::video`, on the Radeon 8060S) gave the same regions and the same video every time, and
  all 9,600 lens frames hashed match ffmpeg's software decode (`target/dec-scratch`, probe hooks in
  `probe-hooks.patch`). `crates/stab/tests/footage.rs` decoded the same way and got the same fix (bc5725e; its four tests pass on
  fresh tracks, 937 s). Open: driver or the 4090's link (ffmpeg's
  Vulkan decode alone against a compute load from a plain process would tell); the validation layers aren't
  installed, so crate video's sync was only read against the spec (no defect found).
- [ ] **No export on the RTX 4090.** Started on the 4090, the export fails the GPU self-test at step 3: NVIDIA's
  driver offers H.265 decode only with DPB and output images coinciding, and crate video requires them distinct. On
  this host the app therefore decodes and runs its networks on the Radeon (`VK_ICD_FILENAMES`). Support coinciding
  images, or pick the adapter by what it can decode.
- [x] **Blur for the MCP server's agents.** e3e1dfe brought onto main:
  `render_still` and `export` take `blur`, run the Export screen's pass (`export::privacy_pass`: main's `PMIN` 0.1 and
  `Hide::plates`) from `Job::pre` on their lanes, and say what it found. On the Radeon (`x5_blur`, 3 s spans) it
  searches 5.1 frames/s on v635 and 4.1 on v606 (the YOLOv9-t detector: 5.9 / 4.6), 2–2.5 times the span; the host
  peaks at 356 MiB RSS, 1.4 GiB VRAM. A pass over one frame (a frame grab, a still) judges its plates as a photo's
  (`plates_still`): `Hide::plates` never hides a plate seen once, so until then a grab blurred no plate.
  `export::tests::plates` asked for a region on v606's Mercedes, which the detector finds at 0.75 but 20 px wide, under
  the photo rule; it now asserts none. Left: no sample clip holds a plate a still blurs, so that side is untested;
  the clips hold almost no plate an export hides either (a pass over all of v635: 5 detections, none hidden), so
  `privacy_parity` compares plate detections only; the M4 figures (1.6 times the span, 160 MB freed by the trim)
  predate our detector.
- [ ] **Tests that fail on main.** Fixed: the app's MCP test `x5_samples` asked for 10–40 s, written against a
  62.9 s clip (its proof run of 11-07 had no v635 and took the first `.insv`, v603); v635 is 5.84 s and the server
  refuses a range past the end by design, so the export cut short by stdin's end now takes the whole clip.
  `live::tests::rtmp` and `rtmp_backlog` named their file by the frame rate alone; now by both of `stream`'s
  parameters (`live-rtmp<fps>-<back>.flv`). Still failing, from the Radeon's CBR rather than our code: crate video's
  `cbr` (H.264 +9.2 %, H.265 +13.2 % at a 1000 ms buffer, 2 % allowed) and `h265_lens0` (+10.8 % against 10 %, and
  its first IDR at 32.8 dB against 33), crate export's `segments` (its first IDR at 32.6 dB against 35). Between 10-27
  and 11-03 the encoder's rate control changed: for the same pictures from a 250 ms buffer on, the stream's first IDR
  comes out smaller (`segments`: 1080 → 732 kB, 38.5 → 32.6 dB, the JPEG grab beside it byte-identical) and the
  P-frames run over their share and never pay it back; 100 ms streams and VBR stay byte-identical. The code of 10-27
  (0c67ef9) and of the first commit (495bbc4) write today's bytes; Mesa 26.2.6 lists no RADV encode change and 26.3.1
  (today) changes none of these bytes; the system update of 11-03 10:55 brought linux-firmware 20261019 (the VCN
  firmware runs the rate control) and OBS 33: the firmware is the likeliest cause, unconfirmed. App exports (CBR,
  1000 ms) come out about 9 % over the rate asked, their first IDR worse. **Decide:** report upstream, a per-vendor
  tolerance, or leave them failing; re-run `cbr` after the reboot into kernel 7.3.5. Also failing: obs's
  `tests::layout` since OBS 33.0.1 (11-03: `LIBOBS_API_VER` ours 32.2.2; every layout matches), and four media tests
  since FFmpeg 9.1 (a test tool, today): `encodes_v635_like_ffmpeg` and `rates_across_samples` (FFmpeg's AAC encoder
  moved: 43.9 → 50.4 dB SNR at 192k, 32.5 → 40.9 dB at 128k; ours unchanged), `remux_edit_lists` (9.1's source cuts
  the AAC padding with its audio edit, 3.0 s of 3.042 s; our remux rewrites the edit as media less priming,
  3.0186 s), `music::tests::formats` (9.1's `dash.m4a` decodes 820 padding samples too many).
- [ ] **Wrap up.** `plates_flags`, `plates_rides`, `plates` and the still checks; `cargo fmt`; models/README
  (sha256, data, results), the About page and `PLAN.md`; commit, merge and push on the user's word.
- [ ] **Decide:** remove the old yolo-v9 plate model files? They stay in the tree unbuilt, and
  `scripts/score_tiles.py` still reads the old ONNX.

## Player

- [ ] **Wayland frame pacing with the display on.** Open since the plan; I found no measurement on a lit native
  Wayland display (the tests run under XWayland, as Wayland sends no frame callbacks while the display is off).

## Colour

- [ ] **I-Log to Rec. 709.** The I-Log curve is unpublished and there is no preset (`fx/src/cube.rs`): fit its
  parameters on footage shot in I-Log, which needs a clip from the user.

## Other platforms

- [ ] **macOS virtual camera.** The macOS build signs and notarizes with a Developer ID; the camera extension isn't
  written (plan §3.2).
- [ ] **Windows port.** Type-checked only so far. Needs the WinUSB install, Vulkan Video per GPU vendor, the virtual
  camera (its frames cross into the Frame Server's LocalService account) and the MCP server's tests.

## AI features

Compared with the vendor's apps (2026-11-13). Here: Deep Track (VitTrack, YOLOX-nano, LOMO re-identification), face
and licence plate blur (YuNet, our YOLOX-nano plate fine-tune), auto edit (heuristic highlights, auto framing), seam
flow on exports and paused frames (RAFT-small), Shadow Clone and Clone Loop (PP-HumanSeg), voice denoise on editor
timelines (RNNoise).

- [ ] **Wind noise on every export.** RNNoise runs on editor timelines only, off by default; the vendor's app applies
  wind reduction by default to recordings the camera marks as AI audio. Offer it on library exports, and score
  DeepFilterNet 3 (MIT / Apache-2.0, to verify) against RNNoise on ride audio: wind, engine, speech.
- [ ] **Forward view.** The view along the direction of travel (the vendor's AI Forward), not the camera's heading,
  which on a helmet is the head's. From the focus of expansion of the image motion left after the gyro's rotation
  (`stab::vis` tracks), smoothed, horizon lock below walking pace; GPS course where the file has it. A mode for the
  player, exports and auto edit. No model.
- [ ] **Highlight marks in auto edit.** Marks set while recording (Capture's highlight mark, the phone app's) in
  trailer record 10 become highlights auto edit always keeps, and show on the seek bar and the editor's timeline.
  Needs a marked clip to confirm the record and its layout on X5 files (Camera: "Unconfirmed on X5 firmware"). If X5
  files carry the camera's own framing (records 18 / 26), read it as a second framing source.
- [ ] **Overtakes.** Vehicles passing within metres (the vendor's Overtaker): YOLOX vehicle tracks crossing from
  ahead to behind or back, as a Shot Lab preset that follows the passing vehicle and as a highlight signal for auto
  edit. Bounded by the vehicle detector missing cars that close (plate item "Plates no vehicle crop holds").
- [ ] **Music in auto edit.** A music track under the highlights, cuts moved onto its beats (onset strength and
  dynamic-programming beat tracking, no model), ducked as in the editor.
- [ ] **Auto framing for a whole clip.** `track::Framer` frames only auto edit's highlights; offer it for the whole
  clip as a follow view (player, flat export) and as editor keys, with the forward view where no subject wins.
- [ ] **Pick by click.** The detector's people and animals boxed on the paused frame, a click starting Deep Track
  (the vendor's Identify Tracking Targets); `Ai::pick` exists, nothing calls it.
- [ ] **Seam flow while playing.** RAFT runs on paused frames and exports only; the vendor's player keeps its AI
  seam while playing. RAFT on every n-th frame during playback, the flow held between; cost unmeasured beside two 4K
  decodes.
- [ ] **Frame grab denoise.** The vendor denoises stills saved from video, we don't. A GPU non-local-means pass in
  fx before the JPEG encoder.
- [ ] **Sky Swap.** Sky segmentation on perspective tiles of the sphere (e.g. SegFormer-B0 on ADE20K, NVIDIA's
  non-commercial licence, to verify), feathered and smoothed over time, a sky panorama in its place; a Shot Lab pass.
- [ ] **Library search.** Tags and text search from an image embedding of each clip's thumbnails (e.g. SigLIP,
  Apache-2.0 per its model card, to verify); near-duplicates from the same embedding.
- Not worth doing: AI Retouch, AI Warp, AI templates, AI Vlog and 3D Space (generated on the vendor's servers);
  beauty filter, Bullet Time, Nose Mode, Face Off, Skeleton, Jump / Big Man / Running Planet, Swing Sticky Jump,
  Similarity, Circle Clone, Parallel World, the TimeShift preset (selfie-stick tricks, pose models or bullet-time
  files, little use on ride footage; speed ramps cover TimeShift); InstaFrame tracking during capture (only while
  tethered); AI Director (the camera's own analysis, unconfirmed on the X5); learned colour, automatic underwater
  detection and model-based defringe (the classical Auto colour, Underwater and fringe controls are in); multi-frame
  video denoise (the camera denoises); dive-case removal (diving only); freeze-frame plane detection (purpose
  unknown); a learned highlight scorer (no open model for 360 ride footage; marks, motion and overtakes first); an
  automatic selfie view (Deep Track on the person does it). Lens guard detection is the Stitch item "Lens guards".

## Camera

Tests on the real X5 send only request types the app sends; never format the SD card.

- [ ] **Wi-Fi tests in station mode** on the home network, which needs its credentials from the user. Never join the
  camera's own access point: it takes this machine offline.
- [ ] **Activation.** How the firmware reacts to option 19 = 0 or 100 (`SET_OPTIONS[19]`): does it lock features?
- [ ] **Settings the app never writes for the X5.** Does the camera accept `MUTE`, `FLICKER_GLOBAL`,
  `CAMERA_LANGUAGE`, `VIDEO_ENCODE_TYPE` and `BUTTON_PRESS_OPTIONS`?
- [ ] **Self-timer.** x5desk counts down on the host and the camera stores option 26: does the camera wait as well
  (a double delay)?
- [ ] **Unconfirmed on X5 firmware.** Favourites sync (msg 176 / 201); AEB grouping and its `hdr_process_type`; the
  trailer records EDITOR 18, APEI 26 and HIGHLIGHT 10 on real files; which capture setting gives `audio_mode` 9, 10
  or 11.
- [ ] **Firmware file.** Format and signature checks of the X5 `.bin`; msg 235 is in the catalog, but the traced
  upload sends no control message before `/upload_fw`. Only matters for firmware updates.

## Host

- [ ] **RTX 4090 PCIe link.** Thousands of link replays; training stalls in `cuStreamSynchronize`, and a crop dump
  lost the device on 2026-11-10 (Xid 32 in ffmpeg's decoder). Cap the link at Gen3, or change the OcuLink cable or
  adapter.
