//! Edits with undo. Every change is a splice of the clip list or a swap of the timeline-wide settings; the log
//! keeps only what each change removed, so undo and redo move clips back and forth without copying them.

use crate::{Clip, Ease, Error, File, GAP, Key, On, Out, Rate, Result, Timeline, View, bad, view};

enum Op {
    /// `n` clips at `at` are in the timeline; flipping puts `put` there instead and keeps the removed ones.
    Clips { at: usize, n: usize, put: Vec<Clip> },
    /// Flipping swaps it with the timeline's.
    Out(Out),
}

struct Entry {
    ops: Vec<Op>,
    /// Drag group (0: none).
    g: u64,
}

/// A timeline with its edit log. Edits validate their input and leave the timeline unchanged on error.
pub struct Project {
    tl: Timeline,
    undo: Vec<Entry>,
    redo: Vec<Entry>,
    /// Open drag group and the last one handed out.
    g: u64,
    gn: u64,
    /// The key move under way in drag group g: (g, track, its keys before the move's first step, the key's index
    /// there, its index now). Any other edit ends it.
    mv: Option<(u64, On, Vec<Key>, usize, usize)>,
    /// The redo list the open drag group's first change cleared, while that change's entry is on top: back if the
    /// drag ends where it started.
    stash: Option<Vec<Entry>>,
}

impl Project {
    pub fn new(fps: Rate) -> Result<Project> {
        Project::with(Timeline::new(fps))
    }

    /// Takes over `tl` (checked) with an empty log.
    pub fn with(mut tl: Timeline) -> Result<Project> {
        tl.check()?;
        tl.fix();
        Ok(Project {
            tl,
            undo: vec![],
            redo: vec![],
            g: 0,
            gn: 0,
            mv: None,
            stash: None,
        })
    }

    pub fn tl(&self) -> &Timeline {
        &self.tl
    }

    pub fn save(&self) -> String {
        self.tl.save()
    }

    pub fn load(s: &str) -> Result<Project> {
        Project::with(Timeline::load(s)?)
    }

    /// The id of source file `path` (added if new; a known path takes the new duration unless a clip, also one in the
    /// undo log, uses the file past it). Not an undo step: unused entries are harmless.
    pub fn file(&mut self, path: &str, dur: f64) -> Result<u32> {
        if !(dur > 0.0 && dur <= crate::MAX_FILE) {
            return bad(format!("duration {dur}"));
        }
        let fs = &mut self.tl.files;
        let i = fs.iter().position(|f| f.path == path).unwrap_or_else(|| {
            fs.push(File {
                path: path.into(),
                dur,
            });
            fs.len() - 1
        });
        let log = self
            .undo
            .iter()
            .chain(&self.redo)
            .chain(self.stash.iter().flatten());
        let logged = log.flat_map(|e| &e.ops).flat_map(|op| match op {
            Op::Clips { put, .. } => &put[..],
            Op::Out(_) => &[],
        });
        if dur < fs[i].dur
            && self
                .tl
                .clips
                .iter()
                .chain(logged)
                .any(|c| c.file == i as u32 && c.b.max(c.a) > dur)
        {
            return bad("clips use the file past that duration");
        }
        self.tl.files[i].dur = dur;
        Ok(i as u32)
    }

    /// Starts a drag: the edits until [`Project::end`] make one undo step, consecutive edits of the same clip range
    /// or of the settings collapse into one change.
    pub fn begin(&mut self) {
        self.gn += 1;
        self.g = self.gn;
    }

    /// Ends a drag; a drag that ended where it started leaves no undo step.
    pub fn end(&mut self) {
        let stash = self.stash.take();
        if let Some(e) = self.undo.last().filter(|e| self.g != 0 && e.g == self.g) {
            let same = match &e.ops[..] {
                [Op::Clips { at, n, put }] => self.tl.clips[*at..*at + *n] == put[..],
                [Op::Out(o)] => *o == self.tl.out,
                _ => false,
            };
            if same {
                self.undo.pop();
                // nothing else touched redo since the group's first change emptied it
                self.redo = stash.unwrap_or_default();
            }
        }
        (self.g, self.mv) = (0, None);
    }

    /// Undo and redo depths.
    pub fn history(&self) -> (usize, usize) {
        (self.undo.len(), self.redo.len())
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo(&mut self) -> bool {
        let Some(mut e) = self.undo.pop() else {
            return false;
        };
        e.ops.iter_mut().rev().for_each(|op| self.flip(op));
        self.tl.fix();
        self.redo.push(e);
        (self.g, self.stash) = (0, None);
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(mut e) = self.redo.pop() else {
            return false;
        };
        e.ops.iter_mut().for_each(|op| self.flip(op));
        self.tl.fix();
        self.undo.push(e);
        (self.g, self.stash) = (0, None);
        true
    }

    fn flip(&mut self, op: &mut Op) {
        match op {
            Op::Clips { at, n, put } => {
                let m = put.len();
                let old = self.tl.clips.splice(*at..*at + *n, put.drain(..)).collect();
                (*put, *n) = (old, m);
            }
            Op::Out(o) => std::mem::swap(o, &mut self.tl.out),
        }
    }

    fn splice(&mut self, ops: &mut Vec<Op>, at: usize, n: usize, new: Vec<Clip>) {
        let m = new.len();
        let put = self.tl.clips.splice(at..at + n, new).collect();
        ops.push(Op::Clips { at, n: m, put });
    }

    fn commit(&mut self, ops: Vec<Op>) {
        self.tl.fix();
        let redo = std::mem::take(&mut self.redo);
        self.mv = None;
        match self.undo.last_mut() {
            Some(e) if self.g != 0 && e.g == self.g => {
                for op in ops {
                    // a change of what the previous one changed: keep the older original
                    match (e.ops.last_mut(), &op) {
                        (Some(Op::Clips { at, n, .. }), Op::Clips { at: a, n: m, put })
                            if *at == *a && *n == put.len() =>
                        {
                            *n = *m
                        }
                        (Some(Op::Out(_)), Op::Out(_)) => {}
                        _ => e.ops.push(op),
                    }
                }
            }
            _ => {
                self.stash = (self.g != 0).then_some(redo);
                self.undo.push(Entry { ops, g: self.g })
            }
        }
    }

    /// Replaces clips `at..at + n` with `new` as one undo step (none if nothing changes).
    fn set(&mut self, at: usize, n: usize, new: Vec<Clip>) -> Result<()> {
        for c in &new {
            self.tl.ok(c)?;
        }
        if self.tl.clips[at..at + n] == new[..] {
            return Ok(());
        }
        let mut ops = vec![];
        self.splice(&mut ops, at, n, new);
        self.commit(ops);
        Ok(())
    }

    fn get(&self, k: usize) -> Result<&Clip> {
        self.tl
            .clips
            .get(k)
            .ok_or_else(|| Error(format!("no clip {k}")))
    }

    /// Inserts `c` before clip k (k = len: appends).
    pub fn insert(&mut self, k: usize, c: Clip) -> Result<()> {
        if k > self.tl.clips.len() {
            return bad(format!("no clip {k}"));
        }
        self.set(k, 0, vec![c])
    }

    /// Sets clip k's source in and out, s.
    pub fn trim(&mut self, k: usize, a: f64, b: f64) -> Result<()> {
        let c = self.get(k)?;
        if c.hold > 0.0 {
            return bad("a freeze frame has no in and out");
        }
        let c = Clip { a, b, ..c.clone() };
        self.set(k, 1, vec![c])
    }

    pub fn delete(&mut self, k: usize) -> Result<()> {
        self.get(k)?;
        self.set(k, 1, vec![])
    }

    /// Inserts a copy of clip k after it.
    pub fn dup(&mut self, k: usize) -> Result<()> {
        let c = self.get(k)?.clone();
        self.set(k + 1, 0, vec![c])
    }

    /// Moves clip `from` to index `to` (of the list without it).
    pub fn mv(&mut self, from: usize, to: usize) -> Result<()> {
        let c = self.get(from)?.clone();
        if to >= self.tl.clips.len() {
            return bad(format!("no clip {to}"));
        }
        // moving among equal clips changes nothing: no undo step
        let r = &self.tl.clips[from.min(to)..=from.max(to)];
        if r.iter().any(|x| *x != c) {
            let mut ops = vec![];
            self.splice(&mut ops, from, 1, vec![]);
            self.splice(&mut ops, to, 0, vec![c]);
            self.commit(ops);
        }
        Ok(())
    }

    /// Sets clip k's speed keys (source s, speed); one key: constant speed.
    pub fn speed(&mut self, k: usize, keys: Vec<(f64, f64)>) -> Result<()> {
        if self.get(k)?.hold > 0.0 {
            return bad("a freeze frame has no speed");
        }
        self.edit(k, |c| c.speed = keys)
    }

    /// Clip k cut before its local frame j: both parts, which show the same frames as the whole.
    fn cut(&self, k: usize, j: u64) -> Result<(Clip, Clip)> {
        let (c, r, fps) = (&self.tl.clips[k], self.tl.frames(k), self.tl.out.fps);
        if j == 0 || j >= r.end - r.start {
            return bad("the playhead is on a cut");
        }
        let (mut p, mut q, tau) = (c.clone(), c.clone(), fps.t(j));
        if c.hold > 0.0 {
            (p.hold, q.hold) = (tau, c.hold - tau);
            q.view.iter_mut().for_each(|k| k.t -= tau);
            q.vol.iter_mut().for_each(|k| k.0 -= tau);
        } else if c.rev {
            // the source boundary between the two output frames
            let x = c.src(c.dur() - tau);
            (p.a, q.b) = (x, x);
        } else {
            let x = c.src(tau);
            (p.b, q.a) = (x, x);
        }
        Ok((p, q))
    }

    /// Splits the clip under the playhead `t` (output s) before the frame showing there, which starts the second
    /// part; returns the second part's index.
    pub fn split(&mut self, t: f64) -> Result<usize> {
        let n = self
            .tl
            .frame_at(t)
            .ok_or_else(|| Error("the playhead is outside the timeline".into()))?;
        let k = self.tl.find(n).unwrap();
        let (p, q) = self.cut(k, n - self.tl.frames(k).start)?;
        self.set(k, 1, vec![p, q])?;
        Ok(k + 1)
    }

    /// Freezes the frame under the playhead for `hold` s: the clip is split there and the freeze frame inserted, so
    /// the picture stops and then plays on. Returns the freeze frame's index.
    pub fn freeze(&mut self, t: f64, hold: f64) -> Result<usize> {
        let tl = &self.tl;
        let n = tl
            .frame_at(t)
            .ok_or_else(|| Error("the playhead is outside the timeline".into()))?;
        let k = tl.find(n).unwrap();
        let (c, j) = (&tl.clips[k], n - tl.frames(k).start);
        if c.hold > 0.0 {
            return bad("already a freeze frame");
        }
        let s = tl.content(k, tl.out.fps.t(j));
        let f = Clip {
            look: c.look.clone(),
            amt: c.amt,
            adj: c.adj,
            // the clip's own keys move its view: start the freeze where they are
            view: view::at(&c.view, s, c.rev)
                .map(|v| {
                    vec![Key {
                        t: 0.0,
                        v,
                        e: Ease::Smooth,
                    }]
                })
                .unwrap_or_default(),
            ..Clip::still(c.file, s, hold)
        };
        let new = if j == 0 {
            vec![f, c.clone()]
        } else {
            let (p, q) = self.cut(k, j)?;
            vec![p, f, q]
        };
        self.set(k, 1, new)?;
        Ok(k + (j > 0) as usize)
    }

    /// Changes clip k with `f` as one undo step (look, volume, mute, reverse, view keys, ...). Rejected, with the
    /// clip unchanged, if the result is invalid.
    pub fn edit<R>(&mut self, k: usize, f: impl FnOnce(&mut Clip) -> R) -> Result<R> {
        let mut c = self.get(k)?.clone();
        let r = f(&mut c);
        self.set(k, 1, vec![c])?;
        Ok(r)
    }

    /// Changes the timeline-wide settings with `f` as one undo step (frame rate, output view keys, music).
    pub fn edit_out<R>(&mut self, f: impl FnOnce(&mut Out) -> R) -> Result<R> {
        let mut o = self.tl.out.clone();
        let r = f(&mut o);
        if o == self.tl.out {
            return Ok(r);
        }
        let old = std::mem::replace(&mut self.tl.out, o);
        if let Err(e) = self.tl.check() {
            self.tl.out = old;
            return Err(e);
        }
        self.commit(vec![Op::Out(old)]);
        Ok(r)
    }

    fn edit_track<R>(&mut self, on: On, f: impl FnOnce(&mut Vec<Key>) -> R) -> Result<R> {
        match on {
            On::Out => self.edit_out(|o| f(&mut o.view)),
            On::Clip(k) => self.edit(k, |c| f(&mut c.view)),
        }
    }

    /// Keyframe at the playhead `t` (output s) on track `on` (a clip track: the playhead must be on that clip).
    /// A key within half a frame takes the new view and ease. Returns the key's index.
    pub fn key(&mut self, on: On, t: f64, v: View, e: Ease) -> Result<usize> {
        let x = self
            .tl
            .to_track(on, t)
            .ok_or_else(|| Error("the playhead is not on that clip".into()))?;
        let near = self.tl.near(on, t);
        self.edit_track(on, |ks| match near {
            Some(i) => {
                (ks[i].v, ks[i].e) = (v, e);
                i
            }
            None => view::put(ks, Key { t: x, v, e }),
        })
    }

    pub fn unkey(&mut self, on: On, i: usize) -> Result<()> {
        if i >= self.tl.track(on).map_or(0, |ks| ks.len()) {
            return bad(format!("no key {i}"));
        }
        self.edit_track(on, |ks| {
            ks.remove(i);
        })
    }

    /// Moves key i of track `on` to output time `t`; a key within half a frame of there is replaced. Returns its
    /// new index. In a drag ([`Project::begin`]) the moves of one key in a row, each given the index the last one
    /// returned, all start from the track as it was before the first: a key the dragged one passes over is back once
    /// it moves on, and the drag ends as one move from its start to its end would. Any other edit in between ends
    /// that run; the next move starts from the track as it is then.
    pub fn mv_key(&mut self, on: On, i: usize, t: f64) -> Result<usize> {
        let x = self
            .tl
            .to_track(on, t)
            .ok_or_else(|| Error("outside the clip".into()))?;
        let (ks, j) = match &self.mv {
            Some((g, o, ks, j, now)) if *g == self.g && *o == on && *now == i => (ks.clone(), *j),
            _ => (self.tl.track(on).unwrap_or_default().to_vec(), i),
        };
        let Some(&k) = ks.get(j) else {
            return bad(format!("no key {i}"));
        };
        let tol = 0.5 * self.tl.out.fps.t(1);
        let hit = (0..ks.len()).find(|&m| {
            m != j
                && self
                    .tl
                    .from_track(on, ks[m].t)
                    .is_some_and(|y| (y - t).abs() <= tol)
        });
        let mut new: Vec<Key> = (ks.iter().enumerate())
            .filter(|&(m, _)| m != j && Some(m) != hit)
            .map(|(_, k)| *k)
            .collect();
        let r = view::put(&mut new, Key { t: x, ..k });
        self.edit_track(on, |ks| *ks = new)?;
        if self.g != 0 {
            self.mv = Some((self.g, on, ks, j, r));
        }
        Ok(r)
    }

    /// Places preset keys (times from 0, output s) at the playhead `t` on track `on`, replacing the keys in the
    /// range they cover. On a clip the preset must fit inside it. Speed ramps under a clip track bend the timing
    /// (its keys sit on source time).
    pub fn preset(&mut self, on: On, t: f64, keys: &[Key]) -> Result<()> {
        let mut ks = keys
            .iter()
            .map(|k| {
                let x = self
                    .tl
                    .to_track(on, t + k.t)
                    .ok_or_else(|| Error("the preset runs past the clip".into()))?;
                Ok(Key { t: x, ..*k })
            })
            .collect::<Result<Vec<Key>>>()?;
        // a reversed clip's track runs backwards; eases stay on their keys (they shape the motion on screen)
        if ks.len() > 1 && ks[0].t > ks[ks.len() - 1].t {
            ks.reverse();
        }
        ks.dedup_by(|b, a| b.t - a.t < GAP);
        let (lo, hi) = match (ks.first(), ks.last()) {
            (Some(a), Some(b)) => (a.t, b.t),
            _ => return Ok(()),
        };
        self.edit_track(on, |track| {
            track.retain(|k| k.t < lo || k.t > hi);
            for k in ks {
                view::put(track, k);
            }
        })
    }
}
