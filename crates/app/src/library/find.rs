//! Library search on crate track's image-text model (TinyCLIP, `track::search`): each recording of the lists is
//! embedded from a few of its frames (`K` sync frames at the middles of equal parts, decoded and stitched level on the
//! shared device, their halves the pictures) by a worker thread, which keeps them on disk per file
//! (`<cache>/x5desk/search`). Text ranks the clips by how well the mean of their pictures matches it; tags are fixed
//! prompts scored the same way against a plain "a photo"; two clips are near-duplicates when every frame of each finds
//! a close one in the other.

use crate::{BURN, GATE, ai::MODELS};
use eframe::egui;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    thread,
    time::UNIX_EPOCH,
};
use track::search::{DIM, PICS};
use video::Gpu;

/// Frames a clip is embedded from, and the equirect width they are stitched at (the crop averages 4 × 4 pixels into
/// each of a half's 256 × 256).
const K: usize = 8;
const W: u32 = 2048;
/// A change of model, pictures or frames makes new cache entries.
const KIND: &str = "tinyclip_39m halves 8";
/// Tags: name, prompt, and how much better than `BASE` the clip's mean picture has to match the prompt. Prompts and
/// margins chosen on 46 clips labelled by eye (`models/README`): none of them tagged wrongly.
pub const TAGS: [(&str, &str, f32); 9] = [
    ("night", "a photo taken at night", -0.015),
    ("ride", "a photo taken while riding a scooter", 0.015),
    ("road", "asphalt", -0.025),
    ("city", "a photo of a city street", -0.033),
    ("forest", "a photo of a forest", 0.0),
    ("water", "a pond", -0.01),
    ("snow", "a photo of snow", 0.0),
    ("indoor", "an indoor scene", -0.034),
    ("people", "a crowd", -0.012),
];
const BASE: &str = "a photo";
/// Near-duplicates: every frame of either clip has one in the other this alike (mean of its pictures' dot products).
/// A clip standing still scores 0.95 to 0.98 against itself at other moments, one on the move 0.73 to 0.92, the next
/// minute of a ride 0.89 at most.
const DUP: f32 = 0.94;

enum Req {
    Clip(PathBuf),
    Text(String),
}

enum Up {
    Clip(PathBuf, Result<Vec<f32>, String>),
    Text(String, Result<Vec<f32>, String>),
    Tags(Vec<f32>),
}

pub struct Find {
    tx: mpsc::Sender<Req>,
    rx: mpsc::Receiver<Up>,
    /// per entry of the lists: its pictures' unit embeddings (frames × `PICS` × `DIM`), or why it has none; their
    /// mean, scaled to length 1
    pub emb: HashMap<PathBuf, Result<Vec<f32>, String>>,
    mean: HashMap<PathBuf, Vec<f32>>,
    /// files asked for and not answered yet
    pub wait: HashSet<PathBuf>,
    /// the query asked for, and the one embedded
    q: String,
    pub qe: Option<(String, Vec<f32>)>,
    /// the prompts of `TAGS` and `BASE`, embedded
    tags: Option<Vec<f32>>,
    /// per file: its tags (bits of `TAGS`) and the file it is a near-duplicate of
    pub tag: HashMap<PathBuf, u32>,
    pub dup: HashMap<PathBuf, PathBuf>,
}

impl Find {
    /// The worker on the shared device; `dir`: the cache (None: none).
    pub fn new(g: &Arc<Gpu>, cx: &stitch::Ctx, ctx: &egui::Context, dir: Option<PathBuf>) -> Find {
        let (tx, rq) = mpsc::channel();
        let (ut, rx) = mpsc::channel();
        let (g, cx, c) = (g.clone(), cx.clone(), ctx.clone());
        thread::Builder::new()
            .name("search".into())
            .spawn(move || {
                let send = |u| {
                    let ok = ut.send(u).is_ok();
                    c.request_repaint();
                    ok
                };
                let mut p: Vec<&str> = TAGS.iter().map(|t| t.1).collect();
                p.push(BASE);
                match MODELS.with(&g, false, Some(&c), |ai| {
                    let _b = BURN.lock();
                    ai.text(&p)
                }) {
                    Ok(t) => drop(send(Up::Tags(t))),
                    Err(e) => eprintln!("search tags: {e}"),
                }
                let (mut todo, mut text) = (VecDeque::new(), None);
                loop {
                    let mut rs: Vec<Req> = rq.try_iter().collect();
                    if rs.is_empty() && todo.is_empty() && text.is_none() {
                        let Ok(r) = rq.recv() else { return };
                        rs.push(r);
                    }
                    for r in rs {
                        match r {
                            Req::Clip(p) => todo.push_back(p),
                            Req::Text(t) => text = Some(t),
                        }
                    }
                    // the query first: typing waits for one clip at most
                    let u = if let Some(t) = text.take() {
                        let e = MODELS.with(&g, false, Some(&c), |ai| {
                            let _b = BURN.lock();
                            ai.text(&[&t])
                        });
                        Up::Text(t, e)
                    } else if let Some(p) = todo.pop_front() {
                        // a list's entry is its recording's main file unless only the proxy was there
                        let m = super::pair(&p)
                            .main
                            .first()
                            .cloned()
                            .unwrap_or_else(|| p.clone());
                        let f = dir.as_deref().and_then(|d| entry(d, &m));
                        let r = match f.as_deref().and_then(load) {
                            Some(v) => Ok(v),
                            None => {
                                let r = crate::export::scoped(&g.device, || clip(&g, &cx, &m));
                                if let (Ok(v), Some(f)) = (&r, &f) {
                                    save(f, v)
                                }
                                r
                            }
                        };
                        Up::Clip(p, r)
                    } else {
                        continue;
                    };
                    if !send(u) {
                        return;
                    }
                }
            })
            .expect("search thread");
        Find {
            tx,
            rx,
            emb: HashMap::new(),
            mean: HashMap::new(),
            wait: HashSet::new(),
            q: String::new(),
            qe: None,
            tags: None,
            tag: HashMap::new(),
            dup: HashMap::new(),
        }
    }

    /// Asks for `p`'s embeddings once.
    pub fn ask(&mut self, p: &Path) {
        if !self.emb.contains_key(p) && self.wait.insert(p.into()) {
            let _ = self.tx.send(Req::Clip(p.into()));
        }
    }

    /// Asks for the query's embedding when it changed.
    pub fn text(&mut self, q: &str) {
        let q = q.trim();
        if q != self.q {
            self.q = q.into();
            if !q.is_empty() {
                let _ = self.tx.send(Req::Text(q.into()));
            }
        }
    }

    /// Takes what the worker finished; a new clip's tags and near-duplicates with it.
    pub fn tick(&mut self) {
        while let Ok(u) = self.rx.try_recv() {
            match u {
                Up::Clip(p, r) => {
                    self.wait.remove(&p);
                    match &r {
                        Ok(e) => {
                            let mut m = vec![0.0; DIM];
                            for x in e.chunks(DIM) {
                                m.iter_mut().zip(x).for_each(|(a, b)| *a += b);
                            }
                            let l = dot(&m, &m).sqrt().max(1e-12);
                            m.iter_mut().for_each(|x| *x /= l);
                            if let Some(t) = &self.tags {
                                self.tag.insert(p.clone(), tags(&m, t));
                            }
                            for (q, b) in &self.emb {
                                if let Ok(b) = b
                                    && alike(e, b).min(alike(b, e)) > DUP
                                {
                                    self.dup.insert(p.clone(), q.clone());
                                    self.dup.insert(q.clone(), p.clone());
                                }
                            }
                            self.mean.insert(p.clone(), m);
                        }
                        Err(e) => eprintln!("search {}: {e}", p.display()),
                    }
                    self.emb.insert(p, r);
                }
                Up::Text(t, Ok(e)) => self.qe = Some((t, e)),
                Up::Text(t, Err(e)) => eprintln!("search \"{t}\": {e}"),
                Up::Tags(t) => {
                    self.tag = (self.mean.iter())
                        .map(|(p, m)| (p.clone(), tags(m, &t)))
                        .collect();
                    self.tags = Some(t);
                }
            }
        }
    }

    /// How well `p`'s pictures match the query (None: no query embedded yet, or no pictures of `p`).
    pub fn score(&self, p: &Path) -> Option<f32> {
        let (_, q) = self.qe.as_ref().filter(|(t, _)| *t == self.q)?;
        Some(dot(self.mean.get(p)?, q))
    }
}

/// The tags (bits of `TAGS`) of a clip's mean picture `m`; `t`: the prompts of `TAGS` and `BASE`, embedded.
fn tags(m: &[f32], t: &[f32]) -> u32 {
    let base = dot(m, &t[TAGS.len() * DIM..]);
    (0..TAGS.len())
        .filter(|&i| dot(m, &t[i * DIM..][..DIM]) - base > TAGS[i].2)
        .fold(0, |b, i| b | 1 << i)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// How well every frame of `a` finds itself in `b`: the least over `a`'s frames of the best frame of `b`, frames
/// compared picture by picture.
fn alike(a: &[f32], b: &[f32]) -> f32 {
    let f = PICS.len() * DIM;
    (a.chunks(f))
        .map(|x| {
            (b.chunks(f))
                .map(|y| dot(x, y) / PICS.len() as f32)
                .fold(f32::MIN, f32::max)
        })
        .fold(f32::MAX, f32::min)
}

/// `p`'s pictures: `K` sync frames at the middles of equal parts (fewer in a short clip), each stitched level (the
/// horizon lock without smoothing), `PICS` of each.
fn clip(g: &Arc<Gpu>, cx: &stitch::Ctx, p: &Path) -> Result<Vec<f32>, String> {
    let c = effects::Clip::open(p)?;
    let ks = c.keys();
    if ks.is_empty() {
        return Err(format!("{}: no frame decodes alone", p.display()));
    }
    let mut pick: Vec<usize> = (0..K)
        .map(|i| ks[(2 * i + 1) * ks.len() / (2 * K)])
        .collect();
    pick.dedup();
    let ps = c.poses(stab::Lock::Horizon, 0.0);
    let mut e = vec![];
    for k in pick {
        // a frame at a time: others' work on the models goes in between
        e.extend(MODELS.with(g, false, None, |ai| {
            let (mut gate, mut v) = (Some(GATE.read()), vec![]);
            c.frames(
                g,
                cx,
                W,
                false,
                &[k],
                |k| c.pose(k, ps.as_deref()),
                |t, _| {
                    gate.take();
                    let _b = BURN.lock();
                    v = ai.pics(&ai.pano(&t.create_view(&Default::default())))?;
                    Ok(())
                },
            )?;
            Ok(v)
        })?);
    }
    Ok(e)
}

/// `<dir>/<FNV-1a of KIND, path, size, modification time>`: a file replaced under its name gets a new entry.
fn entry(dir: &Path, p: &Path) -> Option<PathBuf> {
    let m = std::fs::metadata(p).ok()?;
    let t = m
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let mut x = 0xcbf2_9ce4_8422_2325u64;
    for b in format!("{KIND}\0{}\0{}\0{t}", p.display(), m.len()).bytes() {
        x = (x ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    Some(dir.join(format!("{x:016x}")))
}

/// An entry's embeddings: f32 LE, whole frames of `PICS` (anything else is made again).
fn load(f: &Path) -> Option<Vec<f32>> {
    let b = std::fs::read(f).ok()?;
    let n = 4 * PICS.len() * DIM;
    (!b.is_empty() && b.len() % n == 0).then(|| {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect()
    })
}

fn save(f: &Path, v: &[f32]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    // written whole, then renamed: a reader never sees half a file
    let t = f.with_extension("new");
    let r = (f.parent())
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|_| std::fs::write(&t, b))
        .and_then(|_| std::fs::rename(&t, f));
    if let Err(e) = r {
        eprintln!("{}: {e}", f.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::tests::{gpu, samples};
    use std::time::Instant;

    /// The sample clips through the worker as the Library asks it: words rank the clip they describe first (v606: the
    /// Brandenburg Gate's square on a sunny afternoon, people about; v627: the same square at night; v635: a zoo's
    /// hall, a pool under a glass roof among plants), the tags follow what the clips show, none is a near-duplicate
    /// of another but a copy of v635 is one of it. A second worker answers from the cache with the same embeddings.
    #[test]
    fn samples_ranked() {
        let Some(d) = samples() else { return };
        let (_l, g, cx) = gpu();
        let c = std::env::var_os("X5_FIXTURES")
            .map_or_else(std::env::temp_dir, PathBuf::from)
            .join(format!("x5desk-search-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&c);
        let ctx = egui::Context::default();
        let vs = ["v606", "v627", "v635"].map(|v| d.join(format!("{v}.insv")));
        let until = |f: &mut Find, done: &dyn Fn(&Find) -> bool| {
            let t = Instant::now();
            while !done(f) {
                assert!(t.elapsed().as_secs() < 600, "search worker");
                thread::sleep(std::time::Duration::from_millis(20));
                f.tick();
            }
            t.elapsed().as_secs_f64()
        };
        let mut f = Find::new(&g, &cx, &ctx, Some(c.clone()));
        vs.iter().for_each(|p| f.ask(p));
        let s = until(&mut f, &|f| f.wait.is_empty() && f.tags.is_some());
        println!("3 clips embedded in {s:.1} s (models loaded, kernels compiled)");
        for (q, want) in [
            ("night", 1),
            ("a city square at night with lights", 1),
            ("a sunny day with blue sky", 0),
            ("people walking on a square in the afternoon sun", 0),
            ("a pool of water in a greenhouse", 2),
            ("plants indoors", 2),
            ("a zoo", 2),
        ] {
            f.text(q);
            let s = until(&mut f, &|f| f.qe.as_ref().is_some_and(|e| e.0 == q));
            let sc: Vec<f32> = vs.iter().map(|p| f.score(p).unwrap()).collect();
            println!("\"{q}\": {sc:.3?} ({:.0} ms)", s * 1e3);
            let top = (0..3).max_by(|&a, &b| sc[a].total_cmp(&sc[b])).unwrap();
            assert_eq!(top, want, "\"{q}\"");
        }
        let t = |p: &Path| -> Vec<&str> {
            let m = f.tag[p];
            (0..TAGS.len())
                .filter(|i| m >> i & 1 == 1)
                .map(|i| TAGS[i].0)
                .collect()
        };
        println!("tags {:?}", vs.iter().map(|p| t(p)).collect::<Vec<_>>());
        assert!(t(&vs[1]).contains(&"night") && !t(&vs[0]).contains(&"night"));
        assert!(t(&vs[2]).contains(&"indoor") && !t(&vs[1]).contains(&"indoor"));
        assert!(t(&vs[2]).contains(&"water") && !t(&vs[0]).contains(&"water"));
        assert!(f.dup.is_empty(), "{:?}", f.dup);
        let cp = c.join("copy").join("v635.insv");
        std::fs::create_dir_all(cp.parent().unwrap()).unwrap();
        std::fs::copy(&vs[2], &cp).unwrap();
        f.ask(&cp);
        until(&mut f, &|f| f.wait.is_empty());
        assert_eq!((f.dup.len(), f.dup.get(&cp)), (2, Some(&vs[2])));
        let mut f2 = Find::new(&g, &cx, &ctx, Some(c.clone()));
        vs.iter().for_each(|p| f2.ask(p));
        let s = until(&mut f2, &|f| f.wait.is_empty());
        println!("again from the cache in {:.0} ms", s * 1e3);
        for p in &vs {
            assert_eq!(f.emb[p], f2.emb[p]);
        }
        std::fs::remove_dir_all(&c).unwrap();
    }
}
