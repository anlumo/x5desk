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
  vs off; turn the default off if off wins. No app export of a clip we hold exists yet. Proxy (stab footage test
  `level`, 2026-11-20), the camera's own stitched thumbnail after alignment, 320 × 160 (low frequencies only),
  per 15° of seam the step lens 1 − lens 0 over 3°–15° either side: ours off within 0.44 levels (median) of the
  camera's, on 1.20 (12 handheld clips, on worse on 11); scooter clips, rider and scooter close to the seam, 0.71 /
  8.18 (35, day and night, on worse on all).
  Ours off matches the camera's step within 0.2–0.6 levels, i.e. the camera balances nothing in its own stitch;
  that is all this shows, the app's export runs its own matching. Weak, so the default stays on. Needed: the app's
  360° exports of v635, v606 and 2026-03-04 `_672` (equirect at the largest size, no filter, colour preset,
  enhancement or denoise, horizon levelling off), scored with the same alignment against ours, colour on and off.
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
- [x] **Levelling check without a rig.** Correlate the stitch with the camera's own stitched preview (trailer
  thumbnail) to check the IMU axes and the Mahony gains independently. golblick: FFT correlation, global 256 × 128
  then 512 × 256, 47 of 48 X5 stills aligned. Done as stab footage test `level` (2026-11-20): the thumbnail is the
  camera's levelled equirect (NV12 1280 × 640, BT.709, framed on lens 1 like ours) of the frame 2.97 s in where the
  camera marks `trim_start` 3 s, 1 s into a dash-cam segment, else frame 0; our horizon-levelled stitch of it is
  aligned by correlation (yaw × tilt ±30° at 160 × 80, then 320 × 160 and 640 × 320; a known turn comes back within
  0.02°). 47 of 48 clips aligned (`X5_LEVEL`). Handheld / stick (12): residual tilt unrelated to the camera's tilt
  (−30…77°), so axes and signs hold. Scooter rides (35 dash-cam clips, 0.3–0.7 g of vibration) were 6.0° rms, up to
  16°: `Track::new` now averages the accelerometer over ±50 ms for the gate and the correction and runs a second
  backward pass (a 0.5 s seed added nothing on top): rides 2.25° rms (median 4.8° → 1.3°, max 6.0°, better on 27),
  handheld 0.98° → 0.91° (better on 7 of 12); image horizon v606 0.78° → 0.96°, v627 1.10° → 1.00°. Left: rides
  still up to 6° (2026-04-13 `_723`, 04-16 `_738`, 03-11 `_720`: 4.8–6.0°); the image now prefers Kp 0.05 / Ki 0.001 by 0.08°, the camera the current gains,
  kept; 692's thumbnail frame is not 1 s in. The live preview's filter (`app/src/feed.rs` `Att`) averages the same way
  (the last 100 ms) and starts from its first window: on seven recordings' IMU at the stream's rate its gravity is
  0.5–4.9° rms off the files' filter's (raw samples: 2.5–16.7°; test `ride`); not tried on the camera's stream.

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
  tolerance, or leave them failing; re-run `cbr` after the reboot into kernel 7.3.5. Fixed since, in crates obs and
  media: obs's `tests::layout` wanted our declared libobs API (32.2.2) equal to the installed one (33.0.1 since
  11-03), but libobs refuses only a module whose major.minor is newer than its own (`obs_open_module`, 32.0.0 to
  33.0.1); the test now checks that, and the plugin keeps declaring 32.2 (libobs 33.0.1 loads it, refuses a module
  declaring 33.1; declaring 33 would shut out OBS 32). `remux_edit_lists`: the remux applied only the start of the
  source's edit, and FFmpeg 9.1 cuts the AAC remainder with the edit's end alone (9.0.2 also shortened the last
  frame), so the remux played 820 samples more; `Track::end` gives the edit's end and the remux ends the track there.
  The export copies X5 audio, which has no edit list, and ends it with the video. `music::tests::formats`: 9.1's
  `dash.m4a` states its remainder only in `sidx` and the MPD; FFmpeg and mpv play those 820 samples too, as our
  reader does, so the test expects the file's samples. `encodes_v635_like_ffmpeg` and `rates_across_samples` held
  our AAC encoder to the installed FFmpeg's (9.1's gained 3–12 dB SNR); now to our own figures of October, judged
  then against FFmpeg 9.0.2, less 1 dB, with FFmpeg's printed beside them; ours unchanged.
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
and licence plate blur (YuNet, our YOLOX-nano plate fine-tune), auto edit (heuristic highlights, auto framing, cuts on
the music's beats), seam flow on exports and paused frames (RAFT-small), Shadow Clone and Clone Loop (PP-HumanSeg),
voice denoise on editor timelines and exports (RNNoise).

- [x] **Wind noise on every export.** Library exports take the editor's RNNoise (`Set::denoise` into
  `export::Edit::denoise`): the Export screen's Audio row, the MCP `export` tool's `denoise` (0–1; catalog: denoise).
  Off by default: the vendor's app wind-reduces recordings whose trailer `audio_mode` (#61) is 9, AI wind reduction,
  but all 46 X5 files we have carry 9 (firmware 1.7.43–1.13.21), so the mark tells nothing apart, and at strength 1 a
  ride without speech exports nearly silent. Those recordings are dual mono (L = R). Score (no clean reference) on the
  five ride dates, wind and motor whine, no speech; speech only in 910, 631 and 605: at strength 1 both RNNoise and
  DeepFilterNet 3 take a ride 35–45 dB down below 300 Hz and ~30 dB above; the strength mixes linearly (0.5: −6 dB).
  Clean LibriSpeech over ride noise, −10…+5 dB: DFN3 SI-SDR 11.2–20.6 dB against RNNoise's 6.1–13.1, PESQ-WB
  1.62–2.88 against 1.33–2.24; on real speech the two are close. Cost, one core: RNNoise RTF 0.038, DFN3 0.030 (its
  tract binary). DFN3: code and weights MIT or Apache-2.0 (README, since 2026-11-06).
- [ ] **Replace RNNoise by DeepFilterNet 3** (the user decides). Better on speech under ride noise (above), same cost.
  About 2.1 M parameters, 8.6 MB fp32 ONNX in three graphs (Conv, ConvTranspose, GRU, Einsum), which burn-onnx 0.22
  turns into code that compiles (not run); around them a port of libDF's DSP (STFT 960 / 480, ERB and complex
  features with running norms, the deep filter), or the `deep_filter` crate with tract as a dependency. Neither
  removes wind alone: both are speech denoisers.
- [x] **Forward view.** `stab::Lock::Forward`, "Forward" beside the locks in the player, the Export screen, the
  editor and the MCP's `stabilisation` (`forward`, catalog and docs); auto edit's framer looks along it where no
  subject wins (`track::Framer::ahead`: none yet, or the subject gone for `GONE`). The direction of travel from the
  focus of expansion of the image motion the gyro leaves: `export::flow` stitches a frame every 1/15 s 1024 wide with
  the gyro's rotation out and tracks its corners between 10° and 160° polar (`stab::vis`); `stab::fwd::fit` fits the
  translation per pair (RANSAC over the great-circle constraint t · (a × b) = 0 with the expansion's sign, then least
  squares on the inliers); features without parallax (under 0.09°: sky, far scenery) or that keep their place
  relative to the camera body (the rider, the scooter) don't vote, moving objects are outliers. `fwd::yaws` smooths it
  on the circle (Gaussian, robustly reweighted; σ the smoothing setting, at least 0.25 s); below walking pace (speed
  over camera height under 0.3/s, from the ground features; v606 walking 0.4–0.6, standing 0.03–0.2) the horizon
  lock's heading turned by the travel's usual offset from it (±2 s around the stop), faded over 1 s at the edges. The
  analysis is kept per file for the process (`export::travel`): the player runs it on its worker (the line counts the
  frames; leaving the mode stops it), the Export screen before the privacy pass, auto edit before its detections, a
  job's plan otherwise (MCP). Measured (`crates/export/tests/forward.rs`; figures by
  `target/forward-scratch/py/measure.py`, `sigma.py`): on eight rides of five days (scooter stem mount: the camera's
  heading is the travel's plus steering) the travel runs 12.3–13.9° off the camera's heading (9.5° on 03-11),
  per pair 0.7–1.6° (median) from that, over 10° on 0–1.7 % of travelling pairs; at σ 0.5 s the view stays within
  2.6–6.7° (p95) of camera + offset on straight stretches, with 1.4–3.4 °/s of heading jitter there against the horizon
  lock's 1.7–2.8 (smoother on 7 of 8), and no lag (yaw-rate cross-correlation 0 ± 50 ms). σ 0.25 / 0.5 / 0.75 / 1 s:
  jitter 3.2 / 2.0 / 1.6 / 1.4 °/s, turns rounded by 1.6 / 3.1 / 6.6 / 12.0° (p95); 0.5 s (the default) is the knee.
  v606 travels 61 % of the time (standing before 5 s and after 26 s), v627 and v635 never. Contact sheets (forward view
  over the horizon lock's every 3 s, the raw travel as a red ring): `target/forward-scratch/out/*_forward.jpg`; the
  roads' and paths' vanishing points sit at the forward view's centre. Cost on the Radeon 8060S: 27–38 s a minute of
  footage, the decoding alone 34 (2048 wide: 116–184 s, no better). GPS: none of the 11 clips has a GPS record
  (record 7, 0 bytes), so no course over ground is used. Open: no helmet clip to try; the vendor's AI Forward not
  compared; on 738 the road's vanishing point, where the travel sits, drifts from 13° above to 13° below the horizon:
  the gravity estimate's tilt (`Track::new`, branch `level-check`), which the forward view inherits as every lock
  does; the travel's elevation on roads could check the level.
- [ ] **Highlight marks in auto edit.** Marks set while recording (Capture's highlight mark, the phone app's) in
  trailer record 10 become highlights auto edit always keeps, and show on the seek bar and the editor's timeline.
  Needs a marked clip to confirm the record and its layout on X5 files (Camera: "Unconfirmed on X5 firmware"). If X5
  files carry the camera's own framing (records 18 / 26), read it as a second framing source.
- [x] **Overtakes.** Vehicles overtaking within metres (the vendor's Overtaker): `track::Framer::passes` on the
  detections of the one pass auto framing and auto edit share (`ai::scan`, no second detection loop), in the travel's
  frame (`Framer::fwd`: the forward view's level poses turned into the pass's stitch frame; `ahead` is its centre).
  Vehicle boxes (bicycle, car, motorcycle, bus, truck; a box whose centre lies inside a wider one's on the same side is
  part of it, cut at a tile's edge) are linked into tracks of their own, any vehicle class, on one side: the box
  nearest where the last motion puts the track, within 1.5× of its lateral distance (the box's lower edge on the
  ground, in camera heights; the median of its last three); a track unseen for more than 2 key frames ends (the
  framer's tracklets, per class and by angle only, hop from a passing car onto the parked one it crosses). An
  overtake is a track going from the rear quarter (within 45° of straight behind) to the front one, its median
  lateral distance between them 3 camera heights or less, at 12 camera heights a second or slower (the median rate of
  the cotangent of its azimuth, on the ground along / lateral distance, by that distance), within 8 s; a vehicle seen
  twice passes once. `Pass`: its frames (last behind, first ahead, abeam), side, distance, the track's detections.
  Parked cars cannot do that (they only ever go front to rear), so an overtake needs no speed. We-pass events are not
  counted: from the same rate, the lateral distance and our speed over the camera's height (`stab::fwd::Est::ground`)
  a parked car's speed came out at −0.2 to 0.5 of ours (0.2–0.5 along a close row parked at an angle), the ground
  speed's per-pair estimates are bimodal (a low median made parked cars look oncoming), and no moving vehicle we
  passed turned up to set a threshold against. Auto edit: `Sig::Pass`, 1 over each pass's span, weight 0.2 with the
  other four scaled by 0.8 (their ratios kept: a clip without passes gets the same picks). Shot Lab: Overtake, an
  editing preset (`shotlab::cut::overtake`: keys on every detection of the vehicle at a fixed field of view, 80° by
  default, as fitting its height would zoom from a few degrees to over 100), offered (enabled in the list) once the
  player's Auto framing run on the clip (`ai::Track::passes`, `Library::framing`) found one: a Vehicle list, the range
  its pass and a second either side, the run's stabilisation. Measured on all 36 scooter rides (five days, 35 min,
  60 fps, horizon lock; dumps, contact sheets and scripts in `target/wind-scratch/overtake`): 7 overtakes, 2026-03-11
  one (night), 04-16 two, 04-29 four (691 one, 693 three), 03-04 (night) and 04-13 none; by eye all 7 true (cars and
  vans overtaking on the left 1.8–2.8 camera heights away in 1.2–5.0 s, one a van creeping past in queued traffic).
  Left out by the rules: a track hopping along a parked row (743, "passing" at 39 camera heights a second), a moped
  that waited 30 s beside us at the lights (669). Within 6 camera heights 8 more (cars in the next lane, 3.1–6.0
  away): the 7 viewed true. Misses, from sheets of a frame a second (the band from +35° to −55°, the travel in the
  middle): 738 none of 2, 692 none (no overtake), 693 2 of 5 (a coach bus and a van passing within about a metre while
  we waited: the bus unseen abeam for 1.2 s, the van's boxes broke into parts whose distances jumped by over 1.5×).
  The detector misses close vehicles: of the tracks of vehicles within 3 camera heights that come onto the side from
  the front quarter, 25 of 49 on the left reach the rear one (most lost 45–80° off the travel), 7 of 217 on the right,
  where the rider's legs hide the right rear (about +100° to +170° on the stem mount): overtakes on the right are not
  seen. Auto edit with and without the signal (30 % of the ride): 738 adds the second overtake (the first was in
  already), 691 and 03-11 718 cover theirs (691 half before), 693 keeps one of three (its 18 s are full: the other
  two lose to its start); 717 (none) picks the same. Cost: 0.06–2.6 ms a ride in release (median 1.1 ms), nothing
  beside the pass's ~60 s a minute. Tried and dropped: crossing abeam on the framer's tracklets (9 events on 717 alone,
  all parked cars or hops), a speed share for we-pass events (above). Tests: `track::tests::passes` (synthetic:
  overtaking on the left past a parked row and on the right; a parked row, a car we pass, oncoming, 5 away, waiting
  8 s beside us; two classes pass once; nothing without the travel's frame), `track::tests::pass_signal`,
  `lab::tests::table` and `presets_v635` (Overtake's range and keys), `ai::tests::overtakes_ride` (738's two;
  `X5_RIDES`). Open: overtakes on the right; very close big vehicles; 3 camera heights or 6 (the next lane); Shot Lab
  needs the player's Auto framing run first; the GUI (the list entry, the Vehicle list) untested by hand.
- [x] **Music in auto edit.** The editor's Media panel picks it (Auto edit music: Choose…, the editor's music
  formats; the player's Auto edit uses it too; not in the MCP, which has no auto edit). It plays from its start under
  the highlights as the timeline's music, gain 1 and ducked as the editor's import puts it on (`edit::Music`, the
  editor's mixer path); appended to a timeline with clips it moves with them and replaces the timeline's. Each cut,
  the edit's end too, moves onto a beat (`track::fit`): in turn each highlight's end to the beat, on a whole frame,
  that its length reaches (at least 1 s, at most its scored span `Highlight::lo..hi`: on while the score stays at the
  median, at most halfway to its neighbours), nearest in beat periods less 0.4 per rms of the beat's bass strength (up
  to 2); it ends earlier, or later and past its span's end starts earlier. Without music nothing changed. The editor
  fades no music (its mixer stops it at the timeline's end), so the music stops there, on a beat. Beats
  (`media::beat`, after Ellis 2007, no model): mean positive flux of a 40-band log mel spectrum (32 ms Hann, 4 ms hop
  at 48 kHz on crate media's FFT, floored 80 dB under the peak), high-passed at 0.4 Hz; the tempo from its
  autocorrelation under a log-Gaussian prior (120 BPM, 1.4 octaves) plus half its double's (TPS2), 40–240 BPM;
  Ellis's dynamic programme (α 100, predecessors p/2 to 2p back), weak beats at the ends dropped; a beat's strength
  is the envelope of the four bass bands (to 200 Hz). Measured on synthetic tracks only (`beat::tests`; no music on
  this machine, none in git), 39 of 30 s at 70–180 BPM (click tracks in noise; kick, snare, hi-hat, bass, a melody and
  a chord pad in noise at −30 dB, straight and swung; three after 4 s of silence): F-measure (±70 ms) clicks 0.999
  (min 0.990), drums 0.973, swung 0.968; tempo right on 37, an octave low on 2 (drums and swung at 180 → 90 BPM, F
  0.657), no 3:2 or other error; beats 0 ms (clicks) and 4 ms (drums) late on average. The bass strength puts beats 1
  and 3 (kick) over 2 and 4 (snare) in 111 of 184 bars (median ratio 2.2), all bands the backbeat (0 of 184, 0.43):
  cuts lean to the kick's beats, downbeat-ish only where the kick falls on 1 and 3. Real music: unverified. Cost: the
  39 tracks, synthesis included, in 4.8 s on one core. Tests: `track::tests::beats` (every cut within a frame of a
  beat at 70 / 120 / 180 BPM, 30 and 29.97 fps; clips inside their spans, 1 s or longer; the total within a period
  per clip), `ai::tests::auto_v635_music` (v635 to a 100 BPM drum loop: its cut 4.9 ms off a beat),
  `editor::tests::put_music`. Open: a music fade at the edit's end (the editor has none); appended to a timeline at
  another frame rate than the clip's, each clip's frame rounding can move the cuts after it by up to half a frame.
- [x] **Auto framing for a whole clip.** `ai::Track::framing`: auto edit's pass (`ai::scan`, factored out of
  `Auto::run`, one detection loop for both: the direction of travel, then YOLOX over the sphere every 10th frame into
  `track::Framer`, the forward view as its `ahead`) over the whole clip, the framer's path as a Deep Track run of its
  own (`Track::auto`: from frame 0, no samples), so it reaches Deep Track's three consumers through the same
  mechanism: the player's Auto framing tick beside Deep Track (the analysis on the first tick, its progress on the
  view, kept while the clip is open: unticked and ticked again it follows at once; one follow view at a time, a drag
  on the view ends it); the follow view a flat export takes (Export screen: `autoframe_` name, "framed automatically";
  MCP `export`'s `auto_framing`, flat only, catalog and docs: the agents' lanes run the pass in the job's `pre` with
  the models held per frame, `autoframe_` name); the editor's Auto framing (Reframe tab, beside Track a subject): keys
  every 0.5 s over the whole clip at the playhead, one undo step. Like Deep Track's, the follow view and the keys take
  the path's direction and keep the view's field of view, roll and projection; the framer's own field of view (subject
  height to 45 %, 60–110°: 60° on the forward view) is unused. Nothing changes without it. The framer, fixed after a
  first measurement that framed the rider's legs for a whole ride and never showed the forward view (contact sheets
  `out/*.sheet-before.jpg`): the camera's carrier is never a subject (`Framer::carriers`, on the camera body's frame
  `Framer::body`: the stitch rotations). A seed is fixed to the body (median over its detections of the rms angle of
  its body-frame directions within a second, 25° at most) for 2 s and 40° below the horizon (median: the hand on the
  stick, the scooter's parts, −42° to −58° at 1–9°) or, mostly seen travelling, 40° tall (median: the rider's legs and
  torso, 46–97° at 1–22°) or 20° tall where people are seen in the body frame (within 15°) in 20 % of the ride's
  travelling keys (the rider's head: 26–60 %; the cyclist at a crossing and a walking companion: 10–15 %); standing,
  someone close is not told from the carrier. A close tracklet, or a travelling fixed one 20° tall, within 30° of a
  person's seed is the carrier too (the rider's detections hop between legs, hands, torso and head), and so is a close
  detection there (a cyclist's tracklet that hopped onto the rider). People filmed on purpose are 15–33° tall at −14°
  to −32°: none is dropped on v606, v627, v635 (dropped there: the selfie operator, low, its hands, a bench, an arm
  taken for a refrigerator). While travelling (30 % of the travel estimates within 2 s moving, `stab::fwd::moving`) a
  challenger earns the view over the direction of travel only as a person or animal 20° tall (about 5 m away) at every
  detection for 1.5 s; the subject stays while 10° tall; once the forward view is back the next one earns it afresh;
  standing, as before. The view turns at most 40 °/s from the switch on (a still scene crosses a 100° view in 2.5 s):
  a hand-over of 90° arrives 2.3 s after the switch, of 180° 4.5 s (before: about 1 s, the σ 0.6 s smoothing alone, at
  up to 84–110 °/s). Measured (Radeon 8060S, release; `target/wind-scratch/framing`: `logs/measure-release-*.log`
  before, `logs/measure-final.log` after, sheets `out/*.sheet.jpg`, a frame of the flat view at 100° every 1/20 of the
  clip, FORWARD / SUBJECT marked, the old ones of the first five beside as `*.sheet-before.jpg`; the measuring test in
  `scripts/framing_measure.rs`, the dumps and the lab the criterion came from in `dump/`, `scripts/framing_dump.rs`,
  `scripts/track_tests_with_lab.rs`): cost 50–52 s a minute of 30 fps footage (v606, v627, v635: the travel 25–27, the
  detections 24–25) and 59–64 s at 60 fps (rides 717, 670, 738, 692 of 2026-04-13, 03-04, 04-16, 04-29: 30–35 and
  29–31; the decode runs every frame, the detector every 10th, so twice at 60 fps), the models' first load 4–9 s more,
  the carrier test nothing measurable; the test profile (opt-level 0) is 3.7–6.2 times slower in the travel. Before →
  after, forward share, hand-overs, mean / p95 / max turn (°/s): v635 0 → 0 %, 1 → 1,
  6.4 / 9.4 / 9.4 → 5.6 / 9.4 / 9.4; v627 0 → 0 %, 3 → 2, 17.2 / 71.5 / 110 → 12.5 / 39.0 / 39.1; v606 0 → 33 %,
  4 → 5, 21.6 / 61.4 / 84.4 → 15.7 / 36.1 / 39.3; 717 0 → 100 %, 8 → 0, 7.3 / 32.0 / 43.4 → 0.1 / 1.2 / 1.7; 670
  0 → 62 %, 11 → 2, 11.9 / 40.4 / 84.4 → 5.8 / 37.6 / 39.7; 738 0 → 100 %, 6 → 0, 7.0 / 31.8 / 39.8 → 0.1 / 0.0 / 1.7;
  692 0 → 100 %, 10 → 0, 10.0 / 40.4 / 92.3 → 0.4 / 2.1 / 6.5; jitter (rms of the angular velocity off its 1 s mean,
  1 s clear of hand-overs) 0.04–0.96 °/s. Who wins: v606 the people walking with the camera (0–15.7 s), the forward
  view while walking on (15.7–26 s, where a woman a few degrees tall far off won before), people far off standing at
  the end as before; v627 the three people standing about, no longer the operator crouching at the end; v635 as
  before; 670 the forward view but for the cyclist waiting at the crossing (16–38.9 s, 37° tall, the rider standing
  too); 717, 738, 692 the forward view all ride (no one else near: checked on lens frames). Tests:
  `track::tests::framing` (expectations changed where the cap is meant: its turns at most 40 °/s, were under 80; the
  person 114° away for 3 s is neared within 7.3°, framed before; the A-to-B scene runs 12 s so B is reached),
  `track::tests::framing_ride` (a ride: the rider and its bag fixed to the body never win, the view looks ahead; a
  person 8° tall far off does not earn the view, one 30° tall passing does from 1.5 s after it came until a second
  after it went; standing the far one wins; without the body frame the rider wins), `ai::tests::framing_v635` (the
  path covers the clip, at most 9.4 °/s there; keys span it), `export::tests::follow_framing` (the followed export's
  first frame is the path's view's to the bit, 79.7 off the unframed one per channel), `export::tests::player_spec`,
  `editor::tests::auto_framing` (13 keys over 0–5.81 s on the path), `export::agent::tests::auto_framing_run` (refused
  on a lens pair and without burn; the pass on the lane, 30 frames, frame 0 77.7 off the fixed view's), mcp's argument
  tests; auto edit's (`ai::tests::auto_v635`, `auto_v635_music`) pass unchanged. Open: a companion riding alongside at
  one place for a fifth of a ride would be taken for the carrier, and standing a person posing close is kept (both
  look like the carrier in the body frame); standing, tiny far people still win (v606's end, as before); a hand-over
  pans over the rider when it sits between (670 at 39 s); a subject 90° or more away for a few seconds is barely
  reached at 40 °/s; the editor analyses the clip's whole file, not its span; the player's forward view and auto
  framing run the travel twice when started together (`export::travel` keeps a finished one only); the GUI paths (the
  tick, the button, the labels) are untested by hand.
- [x] **Pick by click.** The detector's people and animals boxed on the paused frame, a click starting Deep Track
  (the vendor's Identify Tracking Targets). With Deep Track armed (the toggle now in every view; drawing a box stays
  the perspective view's) and the player paused, `ai::pick` finds the frame's people and animals (`track::alive`)
  over `track::sphere()` on a thread of its own, as the privacy preview does, keyed by frame and stabilisation; they
  show as yellow boxes in any projection (`ai::outline`: each side a great circle through the preview's projection,
  `export::proj` / `px`, cut where it leaves the view, on both sides of the equirect's edge; Deep Track's own box now
  shows in every view too). A click (`export::ray`, the preview's ray, moved out of the export tests) takes the
  smallest box that holds it (`track::at`, `Ai::pick`'s rule, now people and animals only) and starts the run
  through `Player::track` as a drawn box does (its class known, the run skips the class detection); off the boxes
  `Ai::pick` searches crops of 90° and 30° around the click. A boxed subject is taken as shown, not searched again:
  on v627's frame 300 `Ai::pick` at the box of the woman close by returned a 65° box over her and the camera's
  carrier down to the nadir (its 90° crop reaches into the zone the sphere keeps out of), IoU 0.40 against her box;
  elsewhere it refines the box (0.62–0.83, the sphere's box of a close person often cut at the hips).
  `player::tests::pick` (v606 frame 0; `X5_PICK` for more): a person or more found, a click on the biggest person's
  box centre starts a run from that frame whose first box is that box (IoU 1.000), `Ai::pick`'s there printed; the
  views with the boxes into `X5_FIXTURES`. Run on v606 0 and 450, v627 300, v635 90, the ride 2026-03-04 `_670`
  1500, `_631` 3000 and `_605` 900 (Radeon 8060S, test profile): all pass in 28 s, the boxes in 0.4–1.4 s after the
  seek (6.3 s with the models' load), sitting on their subjects in perspective, Panini, equirect (also across its
  edge), tiny planet and a view on the box nearest a pole (`target/forward-scratch/pick/fx`). Open: people below
  about −63° (the companion beside the camera, the carrier) are never boxed, as the sphere leaves the nadir out; the
  carrier is boxed like anyone (the ride's rider, and at 0.51 what looks like the rider's arm and gear); the click
  itself and the drawing through egui are untested by hand.
- [x] **Seam flow while playing.** The player's RAFT seam (a checkbox by Proxy, on by default; paused frames and exports
  unchanged) draws played frames from the lenses through a flow stitcher's flow (`player::Seam`, `Preview::seam`): RAFT
  runs on a worker of its own on the belt that a new frame's draw records whenever the worker is idle, the flow held
  until the next run's lands, the decline rule judging each run. A late or dropped frame doubles the least frames
  between runs (64 on time take one off); past a second's frames RAFT stops for that source (template seam, a note). On
  the Radeon 8060S a run takes 75–90 ms, mostly burn recording on the worker's CPU, so it runs every 3.2 frames at 30
  fps and every 5 at 60 fps and lands about as many frames after its frame. That takes burn submitting at most 4 kernels
  a command buffer (`stitch::attach`; cubecl's default 32): beside a run, a 1920 × 1080 draw waited up to 30 ms (16
  kernels: 22, 8: 14, 4: 9, 1: 3.9 ms) and the run took 74.7 ms (75.0, 75.4, 80.1, 87.7). Headless playback
  (`player::verify::seam_cost`, ignored: frames taken on the wall clock at the clip's rate, drawn at 1920 × 1080, due to
  GPU done, 15 s), v606 (3840², 30 fps): template p50 0.7 ms, max 2.1; RAFT with 32 kernels 223 of 420 frames over 16.7
  ms (p50 17), a run every 10th frame still one each; with 4: p50 3.3, max 6.1 ms. Rides 2026-04-13 `_717` and 03-04
  `_670` (2880², 60 fps): 32 kernels 551 of 840 frames over 16.7 ms (`_717`), 4 kernels max 5.9 and 6.1 ms. No frame
  dropped, none over 8.3 ms with 4. A RAFT export (v635 5760 Main 10) 12.70 fps with 32 kernels, 12.32 and 12.42 with 4
  (8: 11.94: run to run noise), its video the same (MD5). Quality (`seam_score`'s measures on the flows the player
  draws, 60 frames after 20 of warm-up, two windows a clip; rides at 60 fps; `seam_hold` in
  `target/forward-scratch/play/harness`), seam: template / RAFT every frame / every 3rd held, landing 3 frames late (the
  player at 30 fps) / every 5th landing 6 late (60 fps), and wobble of the last three: v606 (walking, 1.2° a frame) 9.09
  / 5.77 / 6.16 / 6.39, 0.378 / 0.178 / 0.131; v635 12.95 / 8.11 / 8.20 / 8.31, 0.402 / 0.167 / 0.111; v627 (night) 5.16
  / 3.85 / 3.97 / 3.99, 0.214 / 0.091 / 0.057; `_717` 73.2 / 65.0 / 65.5 / 66.7, 1.83 / 0.84 / 0.51; `_670` 20.5 / 17.1
  / 17.3 / 17.3, 1.30 / 0.51 / 0.30. Held at the player's pace the seam stays within 0.7–6.7 % of RAFT on every frame
  (60 fps: 1.3–10.9 %) and 11–37 % under the template (9–36 %), every 10th frame still 6–35 % under; the wobble halves
  (a step every 3rd frame, none between). Also tried at every 3rd landing 3 late: carried by the gyro's turn, smoothed
  over time as per frame (1 − 0.7ⁿ) and extrapolated from the last two runs came within 2 % of holding on the seam at
  1.3–2.6× its wobble; interpolated, one run late, lost on both. Tests: `player::tests::seam_static` (v635's frame 0
  again and again: the flow drawn at every frame is a fresh run's), `seam_option` (off: a played frame is the template
  preview's picture, on: the flow's), stitch `seam_held` (v606: within 10 % of RAFT on every frame, 25 % under the
  template, less wobble). Not verified: playback in the window (vsync, egui's paint, the compositor); a GPU slower than
  the 8060S, where the pacing should give up.
- [x] **Frame grab denoise.** Stills from video (the player's and the Export screen's grabs, the MCP `render_still`)
  get non-local means in crate fx (`fx::Nlm`; `export::Enc::Jpeg`'s strength) on the stitched frame, before the
  effects and the JPEG encoder; videos and photos are not filtered (the camera processes photos itself; `Keep::photo`
  could take it the same way). RGB averaged with weights from 7 × 7 luma patches over a 15 × 15 search, h scaled by
  the frame's own noise: g = the 10th percentile of a 5 × 5 patch's mean squared luma difference 4 px away (mid-tone
  patches every 4 px), h² = (4·strength)²·max(g² − (1.5/255)², 0). Strength 0.5 by default: the Export screen's Grabs
  row (the player's grab takes it too), the MCP's `noise_reduction` (0–1; catalog: still.noise_reduction). Scored
  against the temporal mean of 6–27 frames from other GOPs of a standing stretch, each aligned onto the grabbed frame
  per 320² tile, on static pixels (night: 670 at a red light 17–33 s, v627 handheld; day: 689 standing 41–59 s; indoor
  v635), 5760 and 7680 wide: night +0.9…1.7 dB PSNR at 0.5 (at 1: +1.5…2.2), the reference's edge gradient kept
  92–98 % (1: 70–92 %), texture 81–93 % (1: 50–78 %); day and indoor bit-identical at any strength (g 0.8–1.4/255
  under the knee, night 2.7–4.9; without the knee day texture lost 23 % at the default). 7 × 7 patches and a 15 × 15
  search beat 3, 5, 9 and 9, 21 at the same detail kept, a search every 2nd pixel lost 13 % of the edges; a separate,
  stronger chroma filter and a two-scale filter came within 0.2 dB. Night noise is blotchy and correlated over
  several pixels, mostly luma (σ 2.5 against chroma's 0.9/255 on flat night pixels); all our footage is 8-bit HEVC,
  no 10-bit clip to try. Cost on the Radeon 8060S: 99 ms at 5760 × 2880, 175 at 7680 × 3840, 12 at 1920 × 1080, 51
  at 3840 × 2160; 4–7 ms under the knee. Contact sheets (100 % crops: input, 0.5, 1 or on day the filter without
  the knee, the temporal mean): `target/forward-scratch/grab/out/{night_n670_5760,night_v627_7680,day_d689_5760}.jpg`.
  Not compared with the vendor's own still denoise.
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
