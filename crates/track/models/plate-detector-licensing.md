# Licence plate detector: licensing

`yolo-v9-t-256-license-plates.onnx` in this directory is the licence plate detector behind the "Blur license plates"
option (`src/plates.rs`). It comes from [ankandrew/open-image-models][oim], release `assets`, as
`yolo-v9-t-256-license-plates-end2end.onnx`, and is cut before its NMS. Everything below was checked on the primary
sources on 2026-11-03.

The app uses it under the user's decision of 2026-11-03 that non-commercially licensed dependencies are fine for
x5desk. The points below are open questions, not settled answers. They matter mainly if the app or the model is ever
distributed or used commercially. This is not legal advice.

## 1. The weights have no licence of their own

- The repository is MIT, "Copyright (c) 2024 ankandrew" ([LICENSE][oim-lic]; the copy here is `LICENSE-yolov9-plates`).
  The weights are release assets, and no licence file names them. A user asked about exactly this gap in
  [fast-alpr #47][fa47].
- The author says that from his side everything is MIT and may be used commercially. He adds that for the YOLOv9
  models it depends on whether the YOLOv9 licence applies to the exported weights, which he cannot judge
  ([#20][oim20], 2026-06-08; similarly [#14][oim14] and [#22][oim22]).

## 2. The training code is GPL-3.0

- The author trained and exported the YOLOv9 plate models with his fork [ankandrew/yolov9][fork], which is derived
  from [WongKinYiu/yolov9][yolov9] ([#22][oim22], [fast-alpr #47][fa47]). Both are GPL-3.0.
- Whether the GPL reaches weights a GPL program has trained is unsettled. The author says he does not know
  ([#14][oim14], [#20][oim20], [#22][oim22]).
- In [#14][oim14] (2025-12) he also wrote that he was not using any GPL code. His later answers name the GPL-3.0 fork
  as what he trained and exported with ([#22][oim22], 2026-06; [fast-alpr #47][fa47], 2026-07).

## 3. The training data is unpublished, and part of it is copyleft

- The training set is about 11,000 images scraped from the web, mostly from free and open datasets ([#14][oim14]).
  The author cannot upload it ([#9][oim9]).
- He points to the training list of his earlier detector [LocalizadorPatentes][lp] as most of what he used
  ([#9][oim9]; asked about the 640 px model, the same family as the 256 px one used here). That list ("Entrenamiento")
  contains:
  - [Open Images][oi]: Google licenses the annotations under CC BY 4.0. The images are "listed as having" CC BY 2.0,
    without warranty, and Google asks users to verify each image's licence themselves.
  - The [Romanian licence plate dataset][ro]: MIT.
  - The [OpenALPR benchmarks][alpr]: AGPL-3.0.
  - CCPD appears in the list unchecked, i.e. not used.
- LocalizadorPatentes itself has no licence file. The part of the ~11,000 images outside that list is unknown.
- UFPR-ALPR (research only) was not used ([#24][oim24]).
- Whether a training set's licence binds a model trained on it is unsettled as well.

## 4. No published checksum

GitHub shows no digest for the asset. We recorded our own sha256 values:

- the download: `938a34ce868da69432fd5a872f24337b86e3dc8753426e16ae18163f615c2a2c`;
- the cut file in this directory: `2104d7433b65dc49de4a215e652b6271c27371a0d2b96ff7843994769359bcf1`.

## What this means for x5desk

- **Personal, non-commercial use:** fine under the user's decision.
- **If the app is ever distributed:**
  - the MIT notice has to ship with it. It does: `LICENSE-yolov9-plates`, and the About page.
  - the GPL question about the weights (section 2) and the copyleft parts of the training data (section 3) would have
    to be cleared first.

## Options if this has to be cleared later

1. **Retrain.** Train a plate detector with an MIT-licensed trainer such as [MultimediaTechLab/YOLO][mityolo] (MIT),
   on data whose licences are documented. The author himself suggests retraining as the way around his weights
   ([fast-alpr #47][fa47]).
2. **EgoBlur.** Meta's [EgoBlur][ego] licence plate model: the repository is Apache-2.0. Its README does not state
   the training data. It is a TorchScript Faster R-CNN, probably not importable into burn, and not evaluated here.
3. **LPD-YuNet.** From the OpenCV zoo, Apache-2.0, trained on Chinese plates. It did poorly on European plates in our
   samples; see its "Considered" note in `README`.

[oim]: https://github.com/ankandrew/open-image-models
[oim-lic]: https://github.com/ankandrew/open-image-models/blob/main/LICENSE
[oim9]: https://github.com/ankandrew/open-image-models/issues/9
[oim14]: https://github.com/ankandrew/open-image-models/issues/14
[oim20]: https://github.com/ankandrew/open-image-models/issues/20
[oim22]: https://github.com/ankandrew/open-image-models/issues/22
[oim24]: https://github.com/ankandrew/open-image-models/issues/24
[fa47]: https://github.com/ankandrew/fast-alpr/issues/47
[fork]: https://github.com/ankandrew/yolov9
[yolov9]: https://github.com/WongKinYiu/yolov9
[lp]: https://github.com/ankandrew/LocalizadorPatentes
[oi]: https://storage.googleapis.com/openimages/web/factsfigures_v7.html
[ro]: https://github.com/RobertLucian/license-plate-dataset
[alpr]: https://github.com/openalpr/benchmarks
[mityolo]: https://github.com/MultimediaTechLab/YOLO
[ego]: https://github.com/facebookresearch/EgoBlur
