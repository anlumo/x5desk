# Todo

Merged on 2026-11-10 from the comparison with golblick, `target/plates-train/PLAN.md`, the open risks of the desktop
app plan (`insta360/PLAN.md` §5, 2026-10-22) and the feature inventory's open questions
(`insta360/re/analysis/app_features.md` §c). Items already done in the code were left out.

## Stitch: takeaways from golblick

From comparing our stitch with golblick's notes
(<https://github.com/AkshayRao27/golblick/blob/main/docs/formats/insta360-agent-notes.md#stitching-the-seam>;
comparison page <https://claude.ai/artifact/WzKEEXqGvqmLj1ojXMq9p8>). Their figures are theirs, on their clips.

- [ ] **Two bands on the template path.** Without flow (live preview `fs`, exports with flow off) `stitch.wgsl`
  `shade` fades over one 6° ramp; golblick saw a single 6° fade double a glove and a zip. Their blend: a narrow
  fade, min(3°, max(1.5°, 4 rows)), plus the lens difference blurred over about 3° (a mip level) handed over across
  6°. Done when flow-off seam error and ghosting on 635 / 606 / 627 beat the single ramp.
- [ ] **Lens guards.** With guards on, the mask at 200° (`offset.rs` `pack`, q[16]) blends the guard's rim into the
  overlap. golblick reads a stored guard factor of 1.0225 (field of view about 2.25 % narrower) and picks it per
  photo by scoring the overlap at both angles. Find the field, shrink the image circle when set; needs a clip shot
  with guards.
- [ ] **Score colour matching against the app's export.** golblick found lens balancing worse than none against the
  vendor player (error 4.0 with, 2.4 without; near content moved 10–13 levels). Ours (`Opts::colour`, on by default:
  `app/src/export.rs` `colour: true`) was never scored that way. Error against an app export of the same clip, on
  vs off; turn the default off if off wins.
- [ ] **Decline rule for the flow.** golblick keeps the straight seam unless the alternative cuts the disagreement
  by 10 % (and the straight seam disagrees by 2 grey levels or more). Per frame, fall back to the template where
  RAFT doesn't pay (flat scenes, night) to drop the wobble there. Score with the belt dump (seam ±3°, wobble).
- [ ] **Levelling check without a rig.** Correlate the stitch with the camera's own stitched preview (trailer
  thumbnail) to check the IMU axes and the Mahony gains independently. golblick: FFT correlation, global 256 × 128
  then 512 × 256, 47 of 48 X5 stills aligned.

## Plate detector (branch `plates-yolox`, worktree `target/plates-wt`)

Training and evaluation tools in `target/plates-train` (`PLAN.md` there). Held out for good: the 2026-03-04 night
ride; held out of the flag training: 2026-04-16 and the sample clips v606 / v627.

- [ ] **Hard-negative fine-tune (in progress).** The user's "wrong blur" flags I confirmed as no plate
  (`flags/junk.json`) become negatives. Crop dump (`scripts/dump_flags.sh`) → `crops-flags` (`make_flags.py`) →
  train `plates_nano_flags` (`train_flags.sh`, from `plates_nano_two`) → ONNX (opset 11, onnxslim) → embed →
  `vs_onnxruntime` → `plate_views` → score with `scripts/rule_eval.py` against both galleries.
- [ ] **The 669 false positive.** The gap between a scooter rider and the scooter (2026-03-04, held out): check
  whether the fine-tune drops it without having seen it.
- [ ] **Re-pick `Hide`'s plate rule.** The second gallery's 10 missed-plate marks must stay covered. The 30 px width
  gate (`WIDE`) drops plates the user can read (22–25 px wide): drop or lower it and let score and repetition
  decide; a single detection at a high score (689 frame 807) has to count.
- [ ] **`flags_fx.py` and the fixtures.** Label regions by the views that show them whole instead of "wrong in either
  view", and add the second gallery's flags and marks.
- [ ] **The van plate nothing detects** (717 frame 2638, about 50 px): find out why.
- [ ] **Plates outside every vehicle crop are never searched.** The vehicle detector misses close-ups (in the
  downloaded stills: 1080 of 1747 greenpack, 455 of 915 Tgmt) and cars passing within metres (692 frame 56: plate
  tracked at frames 27–36 and 60–78, car lost between, region 4° off). Add a crop where a live plate track is
  predicted.
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

- [ ] **Audit against the app's AI features** (inventory §9, §11.3, §11.6). Open models in use: tracking
  (VitTrack), faces (YuNet), plates (YOLOX), seam flow (RAFT-small). List what is still missing.

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
