//! Library search: TinyCLIP ViT-39M/16 Text-19M (Microsoft, image-text embedding, MIT; `models/README`), its image and
//! text towers compiled in by burn-onnx (`build.rs`) and run by burn on the shared device. A frame is embedded from
//! windows of the equirect
//! ([`PICS`]), which `crop.wgsl` writes straight into the image tower's input; text goes through CLIP's byte-level BPE
//! (OpenAI CLIP's merges, `models/clip_merges.txt`) into the text tower. Both give unit vectors of [`DIM`]: a picture
//! matches a text, or another picture, by their dot product.

use crate::{Ai, Cam, Crops, In, Job, Norm, Pano, read};
use burn::tensor::{Int, Tensor, TensorData};
use glam::{Vec2, Vec3};
use std::{collections::HashMap, f32::consts::FRAC_PI_2, sync::OnceLock};

#[allow(clippy::all, unused)]
mod image {
    include!(concat!(env!("OUT_DIR"), "/model/tinyclip_39m_image.rs"));
}

#[allow(clippy::all, unused)]
mod text {
    include!(concat!(env!("OUT_DIR"), "/model/tinyclip_39m_text.rs"));
}

/// Embedding size.
pub const DIM: usize = 512;
/// Image tower input side, pixels.
pub const SIDE: u32 = 224;
/// CLIP's input normalisation, (colour − mean) / std.
const NORM: Norm = Norm {
    a: [1.0 / 0.26862954, 1.0 / 0.261_302_6, 1.0 / 0.275_777_1],
    b: [
        -0.48145466 / 0.26862954,
        -0.4578275 / 0.261_302_6,
        -0.40821073 / 0.275_777_1,
    ],
    bgr: false,
};
/// Text tower input: start, tokens, end, zero padding (the position table is this long); the attention mask is all
/// ones, as the tower's causal mask keeps the padding from the end token it pools at.
pub const CTX: usize = 77;
/// The pictures a frame is embedded from: equirect windows of all latitudes (centre longitude from the stitch's
/// centre column, half width; degrees), each squeezed into the square input. The two halves of a levelled frame
/// retrieved better than perspective views around the horizon (`models/README`): the sky, a ceiling, the ground and
/// whoever holds the camera say much of where it was.
pub const PICS: [[f32; 2]; 2] = [[-90.0, 90.0], [90.0, 90.0]];

/// Both towers and the image tower's input, made on first use.
pub(crate) struct Nets {
    pub img: image::Model,
    pub txt: text::Model,
    pub inp: In,
    pub c: Crops,
}

impl Ai {
    fn search_nets(&mut self) -> Result<Box<Nets>, String> {
        if let Some(n) = self.se.take() {
            return Ok(n);
        }
        let (s, v) = (SIDE as usize, PICS.len());
        Ok(Box::new(Nets {
            img: image::Model::from_embedded(&self.dev),
            txt: text::Model::from_embedded(&self.dev),
            inp: In::new(&self.dev, [v, 3, s, s])?,
            c: Crops::new(&self.wd, &self.g[1], v, v * 3 * s * s),
        }))
    }

    /// The [`PICS`] of an equirect (levelled) as unit embeddings, one after the other.
    pub fn pics(&mut self, p: &Pano) -> Result<Vec<f32>, String> {
        let n = self.search_nets()?;
        let s = (3 * SIDE * SIDE) as usize;
        let jobs: Vec<Job> = (PICS.iter().enumerate())
            .map(|(i, &[x, w])| {
                // a window: the crop's centre longitude and half extents in radians (as tangents for a view)
                let (x, w) = (x.to_radians(), w.to_radians());
                let c = Cam {
                    f: Vec3::X * x,
                    r: Vec3::ZERO,
                    u: Vec3::ZERO,
                    ext: Vec2::new(w, FRAC_PI_2),
                };
                let mut j = Job::new(&c, [SIDE; 2], i * s, &NORM, p.w);
                j.o[3] = 1;
                j
            })
            .collect();
        let d = self.crop(&n.c, p, &jobs, &n.inp);
        self.spent(0, d);
        let e = read(n.img.forward(n.inp.t.clone()));
        self.se = Some(n);
        Ok(unit(e))
    }

    /// Texts as unit embeddings, one after the other.
    pub fn text(&mut self, ts: &[&str]) -> Result<Vec<f32>, String> {
        let n = self.search_nets()?;
        let ids: Vec<i64> = ts.iter().flat_map(|t| tokens(t)).collect();
        let x = Tensor::<2, Int>::from_data(TensorData::new(ids, [ts.len(), CTX]), &self.dev);
        let e = read(n.txt.forward(x.clone(), x.ones_like()));
        self.se = Some(n);
        Ok(unit(e))
    }
}

/// Each [`DIM`] run scaled to length 1.
fn unit(mut v: Vec<f32>) -> Vec<f32> {
    for r in v.chunks_mut(DIM) {
        let l = r.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        r.iter_mut().for_each(|x| *x /= l);
    }
    v
}

/// CLIP's tokenizer (open_clip's `SimpleTokenizer` without its ftfy and HTML clean-up): lower case; split into
/// contractions, letter runs, single digits and runs of other symbols (whitespace between them dropped); each piece's
/// UTF-8 bytes as GPT-2's printable stand-ins, merged by byte-level BPE, the last symbol marked as the word's end.
/// Ids: the 256 bytes, the 256 bytes ending a word, then one per merge, then start and end.
struct Bpe {
    /// merge rank by its line, "a b"
    rank: HashMap<&'static str, usize>,
    /// each byte's stand-in and its id
    ch: [(char, i64); 256],
}

const START: i64 = 49406;
const END: i64 = 49407;

fn bpe() -> &'static Bpe {
    static B: OnceLock<Bpe> = OnceLock::new();
    B.get_or_init(|| {
        let mut ch = [('\0', 0); 256];
        // the printable bytes stand for themselves, the rest for 256 + n, in id order
        let mut bs: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
        bs.extend((0..256).filter(|b| !bs.contains(b)).collect::<Vec<_>>());
        for (i, &b) in bs.iter().enumerate() {
            let c = if i < 188 { b } else { 68 + i as u32 };
            ch[b as usize] = (char::from_u32(c).unwrap(), i as i64);
        }
        let rank = (include_str!("../models/clip_merges.txt")
            .lines()
            .enumerate())
        .map(|(r, l)| (l, r))
        .collect();
        Bpe { rank, ch }
    })
}

/// `t` as the text tower's input: start, its tokens (cut after 75), end, zeros.
pub fn tokens(t: &str) -> [i64; CTX] {
    let b = bpe();
    let c: Vec<char> = t.to_lowercase().chars().collect();
    let mut o = vec![START];
    let mut i = 0;
    while i < c.len() {
        let k = |j: usize| c.get(i + j).copied().unwrap_or(' ');
        let kind = |x: char| match x {
            x if x.is_whitespace() => 0,
            x if x.is_alphabetic() => 1,
            x if x.is_numeric() => 2,
            _ => 3,
        };
        let n = match (k(0), k(1), k(2)) {
            ('\'', 's' | 't' | 'm' | 'd', _) => 2,
            ('\'', 'r' | 'v', 'e') | ('\'', 'l', 'l') => 3,
            (x, ..) => match kind(x) {
                0 => {
                    i += 1;
                    continue;
                }
                2 => 1,
                q => c[i..].iter().take_while(|&&y| kind(y) == q).count(),
            },
        };
        // symbols with their ids; merging the best-ranked pair everywhere until none is left
        let mut w: Vec<(String, i64)> = (c[i..i + n].iter().collect::<String>().bytes())
            .map(|x| (b.ch[x as usize].0.to_string(), b.ch[x as usize].1))
            .collect();
        if let Some(l) = w.last_mut() {
            l.0 += "</w>";
            l.1 += 256;
        }
        while let Some((r, j)) = (1..w.len())
            .filter_map(|j| {
                Some((
                    *b.rank.get(format!("{} {}", w[j - 1].0, w[j].0).as_str())?,
                    j,
                ))
            })
            .min()
        {
            let (x, y) = (w[j - 1].0.clone(), w[j].0.clone());
            let mut m = Vec::with_capacity(w.len());
            let mut j = 0;
            while j < w.len() {
                if j + 1 < w.len() && w[j].0 == x && w[j + 1].0 == y {
                    m.push((x.clone() + &y, 512 + r as i64));
                    j += 2;
                } else {
                    m.push(w[j].clone());
                    j += 1;
                }
            }
            w = m;
        }
        o.extend(w.iter().map(|s| s.1));
        i += n;
    }
    o.truncate(CTX - 1);
    o.push(END);
    o.resize(CTX, 0);
    o.try_into().unwrap()
}
