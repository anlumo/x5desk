//! The capture tools (the camera tier) and the camera's file writes (favourites; deletes, the delete tier). Each
//! checks what it can before anything goes to the camera, against the capture worker's state of the camera's session:
//! the mode and setting `/CONF` lists, the values it allows now, whether the camera captures, and whether x5desk ran
//! it on an X5 (`cam::conf`'s lists, 360, a sub-mode the protocol knows; else only with the `untested` tier). The
//! camera-writing tools take turns ([`Shared::act`]); a capture command goes as asked (numbered, with the camera as
//! it was checked: mode, lens, sub-mode, which the worker holds it to) and its own outcome comes back. A delete keeps
//! every other write off the camera until its note. A recording this host started is stopped when the host ends,
//! and by the host after `seconds` (the host's timer, `host::timer`).

use super::*;
use crate::host::{Rec, capturing};
use cam::{
    capture::{Cmd as Cap, Out, Seen, St},
    conf::{self, Block, CHECKED, SHOT, SWITCHED, Shot},
};
use link::pb;

/// A setting's value as given: as `/CONF` spells it, or a number or a boolean spelt so.
#[derive(Deserialize, JsonSchema)]
#[serde(untagged)]
pub(super) enum Val {
    Text(String),
    Number(f64),
    Bool(bool),
}

impl Val {
    fn spelt(&self) -> String {
        match self {
            Val::Text(s) => s.trim().to_string(),
            Val::Number(x) => format!("{x}"),
            Val::Bool(b) => b.to_string(),
        }
    }
}

/// `capture_mode` arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ModeArgs {
    /// a mode's name (VIDEO_NORMAL) or label (Video), as capture_status lists them
    mode: String,
}

/// `capture_set` arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct SetArgs {
    /// a setting's key from capture_status (white_balance)
    key: String,
    /// one of its allowed values from capture_status, as listed ("4000", "true", "5760_2880_30"); a number or
    /// boolean is taken as spelt so
    value: Val,
}

/// `capture_photo` arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct PhotoArgs {
    /// count the mode's self-timer down first, as the camera's app does (default true; false: at once)
    self_timer: Option<bool>,
}

#[derive(Deserialize, JsonSchema, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(super) enum RecAction {
    Start,
    Stop,
    Wait,
}

/// `capture_record` arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct RecordArgs {
    /// start, stop, or wait (until the camera stops capturing, wait_s at most)
    action: RecAction,
    /// start: the host stops it after this many seconds (1 to 600); up to 45 the call waits for the clip
    seconds: Option<f64>,
    /// wait: seconds to wait at most (default 60, at most 300)
    wait_s: Option<f64>,
}

/// `camera_favourite` arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct FavArgs {
    /// item keys from album_list
    keys: Vec<String>,
    /// set (true) or clear (false) the favourite flag
    on: bool,
}

/// `camera_delete` arguments.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct DeleteArgs {
    /// item keys from album_list
    keys: Vec<String>,
    /// delete only the items whose every file x5desk downloaded here and still has at its size; the others are kept
    /// and named (default false)
    downloaded_only: Option<bool>,
}

/// How long the worker may take with a command that takes a photo or starts or stops a capture (the link waits 30 s,
/// a status read before it 10 s).
const SHUTTER: f64 = 45.0;

/// How long a setting's or mode's write may take (the capture state read, the write, the values read again: five calls
/// of 10 s each at most).
const WRITE: f64 = 60.0;

/// What a tool needs of the capture state: the mode (a stop, a mark), `/CONF` too (the modes and settings), the mode's
/// values too (writes and the shutter).
#[derive(Clone, Copy, PartialEq, PartialOrd)]
enum Need {
    Mode,
    Conf,
    Vals,
}

/// How long a timed recording's call waits for its clip.
const TIMED: f64 = 45.0;

/// A refusal while a recording this host started, or its start, is on the camera.
const REC: &str = "a recording this host started (or its start) is on the camera: capture_record stop stops it, \
                   capture_record wait waits for its end";

/// The longest self-timer counted down.
const TIMER: u32 = 30;

/// What a mode's shutter does: photo, record, timelapse, live.
fn shot_kind(m: &str) -> &'static str {
    match conf::shot(m) {
        Shot::Photo(_) => "photo",
        Shot::Rec(_) => "record",
        Shot::Lapse(_) => "timelapse",
        Shot::None => "live",
    }
}

fn block(st: &St) -> Option<&Block> {
    st.conf.as_ref()?.block(st.both)
}

/// The modes of a lens block, as the camera lists them: photo, video, live.
fn modes(b: &Block) -> impl Iterator<Item = &String> {
    b.photo.iter().chain(&b.video).chain(&b.live)
}

/// Whether the camera's mode is the one its name says: 360, a sub-mode the protocol knows (a value it does not reads
/// as VIDEO_NORMAL or PHOTO_SINGLE), listed in `/CONF`.
fn trusted(st: &St) -> bool {
    let listed = |m: &String| block(st).is_some_and(|b| modes(b).any(|x| x == m));
    st.both && conf::known(&st.o) && st.mode.as_ref().is_some_and(listed)
}

/// The current mode's settings x5desk can show: key, slot and label of those in `conf::KEYS` the mode has.
fn keys(md: &conf::Mode) -> impl Iterator<Item = (&conf::Key, conf::Slot, &'static str)> {
    md.keys.iter().filter_map(|k| {
        let (_, sl, l) = conf::KEYS.iter().find(|x| x.0 == k.name)?;
        Some((k, *sl, *l))
    })
}

/// The same value on the wire.
fn same(sl: conf::Slot, a: &str, b: &str) -> bool {
    match (conf::val(sl, a), conf::val(sl, b)) {
        (Some(x), Some(y)) => conf::same(x, y),
        _ => a == b,
    }
}

/// An outcome that is no success, in words.
fn no(o: Out) -> String {
    match o {
        Out::No(t) => t,
        Out::Failed(e) => format!("the camera: {e}"),
        Out::Sent(_) => "sent".into(),
    }
}

impl Conn {
    /// Why camera writes cannot go now: a delete runs, or a capture command asked earlier has no outcome yet.
    pub(super) fn free(&self) -> Result<(), String> {
        let s = self.sh().st.lock().unwrap();
        if s.deleting.is_some() {
            return Err(
                "a delete still runs on the camera; camera_status shows when it is done".into(),
            );
        }
        let (n, t) = s.asked;
        if n > 0 && !s.outs.iter().any(|x| x.0 == n) {
            return Err(format!(
                "the camera still runs an earlier {t}; try again in a moment"
            ));
        }
        Ok(())
    }

    /// Why a check failed for lack of the `untested` tier: enabling it, with the host's tiers (`enable` replaces the
    /// set), or this client's `--tiers` leaving it out.
    fn untested(&self, what: String) -> String {
        let all = self.host.tiers();
        match all.has(Tier::Untested) {
            true => format!(
                "{what} has not been run on an X5 by x5desk's tests: it needs the untested tier, which this client's \
                 --tiers leaves out"
            ),
            false => format!(
                "{what} has not been run on an X5 by x5desk's tests, so agents may not use it. The user allows \
                 untested modes and settings with `x5desk mcp enable --tiers {all},untested`"
            ),
        }
    }

    fn untested_ok(&self) -> bool {
        self.tiers().has(Tier::Untested)
    }

    /// The capture state of the camera's session as `need`s it: waited for up to 15 s after a connect.
    async fn cap(&self, need: Need) -> Result<St, String> {
        let sh = self.sh();
        const OFF: &str = "the camera is not connected: camera_connect connects it (camera tier)";
        if sh.st.lock().unwrap().info.conn != Link::On {
            return Err(OFF.into());
        }
        let ok = |s: &State| {
            let c = &s.cap;
            s.csid == s.tsid
                && c.on
                && c.mode.is_some()
                && (need < Need::Conf || c.conf.is_some())
                && (need < Need::Vals || c.vals == c.mode)
        };
        if let Some(st) = sh.until(15.0, |s| ok(s).then(|| s.cap.clone())).await {
            return Ok(st);
        }
        let s = sh.st.lock().unwrap();
        if s.info.conn != Link::On {
            return Err(OFF.into());
        }
        let why = (s.csid == s.tsid)
            .then(|| s.cap.note.clone())
            .flatten()
            .filter(|n| n.1);
        Err(match why {
            Some((w, _)) => format!("the camera's capture state could not be read: {w}"),
            None => "the camera's capture state was not read within 15 s; camera_status shows the camera's notes"
                .into(),
        })
    }

    /// Capture command `c` asked (`seen`: the camera as it was checked), its outcome waited for up to `secs`.
    async fn asked(
        &self,
        tool: &'static str,
        c: Cap,
        seen: Option<Seen>,
        secs: f64,
    ) -> Result<Out, String> {
        let sh = self.sh();
        let n = sh.ask(tool, c, seen);
        let o = sh.until(secs, |s| {
            s.outs.iter().find(|x| x.0 == n).map(|x| x.1.clone())
        });
        o.await.ok_or_else(|| {
            format!(
                "no answer from the camera within {secs} s (it may still come: capture_status shows the camera's \
                 state)"
            )
        })
    }

    /// Waits until `f` finds what it wants (looking again every report and every second), with a progress
    /// notification per second when the call asked for them (`say`: its words). Err on the call's cancel or the
    /// host's end; None after `secs`.
    async fn watch<T>(
        &self,
        ctx: &RequestContext<RoleServer>,
        secs: f64,
        say: impl Fn(&State, f64) -> String,
        mut f: impl FnMut(&State) -> Option<T>,
    ) -> Result<Option<T>, String> {
        let sh = self.sh();
        let (t0, tok) = (Instant::now(), ctx.meta.get_progress_token());
        loop {
            let left = secs - t0.elapsed().as_secs_f64();
            if left <= 0.0 {
                return Ok(sh.until(0.0, &mut f).await);
            }
            tokio::select! {
                r = sh.until(left.min(1.0), &mut f) => if let Some(t) = r {
                    return Ok(Some(t))
                },
                _ = ctx.ct.cancelled() => return Err("cancelled".into()),
            }
            if sh.ct.is_cancelled() {
                return Err("the host is shutting down".into());
            }
            if let Some(t) = &tok {
                let e = t0.elapsed().as_secs_f64();
                let m = say(&sh.st.lock().unwrap(), e);
                let n = ProgressNotificationParam::new(t.clone(), e).with_message(m);
                let _ = ctx.peer.notify_progress(n).await;
            }
        }
    }

    /// Waits up to `secs` for the camera's album to list item `key` (a new photo or clip), asking for a listing once
    /// when it does not after 3 s.
    async fn listed(&self, key: &str, secs: f64) -> bool {
        let sh = self.sh();
        let f = |s: &State| {
            (s.album_cam && s.album.as_ref()?.iter().any(|i| i.key == key)).then_some(())
        };
        if sh.until(3.0_f64.min(secs), f).await.is_some() {
            return true;
        }
        sh.cmd(Cmd::Refresh);
        sh.until((secs - 3.0).max(0.0), f).await.is_some()
    }

    pub(super) async fn capture_status(&self) -> Result<CallToolResult, String> {
        let st = self.cap(Need::Conf).await?;
        let s = self.sh().st.lock().unwrap();
        let i = &s.info;
        let m = st.mode.clone().unwrap_or_default();
        let b = block(&st);
        let md = b.and_then(|b| b.mode(&m));
        let vals = st.vals.as_ref() == Some(&m);
        let ms: Vec<Value> = (b.into_iter().flat_map(modes))
            .map(|x| {
                json!({
                    "name": x,
                    "label": conf::label(x),
                    "kind": shot_kind(x),
                    "tested": st.both && SWITCHED.contains(&x.as_str()),
                    "current": *x == m,
                })
            })
            .collect();
        let writable = conf::func(&m).is_some();
        let sets: Vec<Value> = (md.into_iter().flat_map(keys))
            .map(|(k, sl, l)| {
                let al = st.allowed(&k.name);
                let cur = vals.then(|| st.cur(md.unwrap(), &k.name)).flatten();
                // one value allowed now: in effect, whatever the camera keeps for later (as the screen shows it)
                let fixed = al.len() == 1;
                let v = if fixed { al.first().cloned() } else { cur };
                let alv: Vec<Value> = (al.iter())
                    .map(|a| json!({ "value": a, "shown": conf::show(sl, a) }))
                    .collect();
                json!({
                    "key": k.name,
                    "label": l,
                    "value": v,
                    "shown": v.as_deref().map(|v| conf::show(sl, v)),
                    "allowed": alv,
                    "fixed_by": fixed.then(|| k.dep.clone()),
                    "settable": conf::wired(sl) && writable && al.len() > 1,
                    "tested": trusted(&st) && CHECKED.contains(&(m.as_str(), k.name.as_str())),
                })
            })
            .collect();
        let other: Vec<&str> = (md.into_iter())
            .flat_map(|md| md.keys.iter())
            .filter(|k| conf::slot(&k.name).is_none())
            .map(|k| k.name.as_str())
            .collect();
        let cap = capturing(&s);
        let b = i.battery.as_ref();
        let notes: Vec<Value> = (s.notes.iter())
            .filter_map(|(_, t, e)| Some((t.strip_prefix("capture: ")?, e)))
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .take(5)
            .rev()
            .map(|(t, e)| json!({ "text": t, "error": e }))
            .collect();
        let state = pb::CameraCaptureState::try_from(st.cap).map(|c| c.as_str_name().to_string());
        let next = match (cap, shot_kind(&m)) {
            (true, _) => "capture_record stop stops it; capture_record wait waits for its end",
            (false, "photo") => "capture_photo takes a photo; capture_mode switches the mode",
            (false, "live") => "capture_mode switches to a photo or video mode",
            _ => "capture_record start records; capture_mode switches the mode",
        };
        ok(json!({
            "mode": json!({
                "name": m,
                "label": conf::label(&m),
                "kind": shot_kind(&m),
                "tested": trusted(&st) && SHOT.contains(&m.as_str()),
                "known_to_the_protocol": conf::known(&st.o),
            }),
            "lens": if st.both { "360" } else { "single lens" },
            "capturing": cap,
            "capture_state": state.ok(),
            "capturing_for_s": st.since.filter(|_| cap).map(|t| t.elapsed().as_secs()),
            "started_by_this_host": s.rec.is_some(),
            "stops_in_s": s.rec.as_ref().and_then(|r| r.until).map(|t| {
                t.saturating_duration_since(Instant::now()).as_secs()
            }),
            "battery_percent": b.map(|b| cc::level(b).0),
            "charging": b.map(|b| cc::level(b).1),
            "storage_free_bytes": i.storage.as_ref().map(|s| s.free_space),
            "remaining_photos": st.o.remaining_pictures,
            "remaining_video_s": st.o.remaining_capture_time.max(st.po.remaining_time),
            "temperature": st.temp.map(|t| json!({
                "state": t.temp_state().as_str_name(),
                "value": t.temp_value,
            })),
            "modes": ms,
            "settings": sets,
            "values_read": vals,
            "other_settings": other,
            "untested_allowed": self.untested_ok(),
            "notes": notes,
            "next": next,
        }))
    }

    pub(super) async fn capture_mode(&self, a: ModeArgs) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let _a = sh.act.lock().await;
        self.free()?;
        let st = self.cap(Need::Conf).await?;
        let b = block(&st).ok_or("the camera's /CONF has no modes for its lens")?;
        let w = a.mode.trim();
        let Some(m) = modes(b)
            .find(|x| x.eq_ignore_ascii_case(w) || conf::label(x).eq_ignore_ascii_case(w))
            .cloned()
        else {
            let all: Vec<String> = modes(b)
                .map(|x| format!("{x} ({})", conf::label(x)))
                .collect();
            return Err(format!("no mode {w:?}; the camera has {}", all.join(", ")));
        };
        if st.mode.as_ref() == Some(&m) && trusted(&st) {
            return ok(json!({ "mode": m, "label": conf::label(&m), "note": "in it already" }));
        }
        if conf::shot(&m) == Shot::None {
            return Err(format!(
                "{m} is a live mode: x5desk's Live screen streams it; agents cannot use it yet"
            ));
        }
        {
            let s = sh.st.lock().unwrap();
            if capturing(&s) {
                return Err(
                    "the camera captures: no mode change until it stops (capture_record stop)"
                        .into(),
                );
            }
            if s.rec.is_some() {
                return Err(REC.into());
            }
        }
        if !(st.both && SWITCHED.contains(&m.as_str())) && !self.untested_ok() {
            return Err(self.untested(format!("switching to {m} ({})", conf::label(&m))));
        }
        if conf::sub(&m).is_none() {
            return Err(format!("{m}: x5desk does not know how to switch to it"));
        }
        let o = self
            .asked("capture_mode", Cap::Mode(m.clone()), Seen::of(&st), WRITE)
            .await?;
        if !matches!(o, Out::Sent(_)) {
            return Err(no(o));
        }
        // the camera may report the new mode a moment later
        let r = sh.until(15.0, |s| {
            (s.csid == s.tsid && s.cap.mode.as_ref() == Some(&m)).then_some(())
        });
        match r.await {
            Some(()) => ok(json!({ "mode": m, "label": conf::label(&m) })),
            None => Err(format!(
                "the switch to {m} went out, but the camera did not report it within 15 s; capture_status shows its \
                 mode"
            )),
        }
    }

    pub(super) async fn capture_set(&self, a: SetArgs) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let _a = sh.act.lock().await;
        self.free()?;
        let st = self.cap(Need::Vals).await?;
        let m = st.mode.clone().unwrap_or_default();
        let md = block(&st)
            .and_then(|b| b.mode(&m))
            .ok_or_else(|| format!("{m}: not in the camera's /CONF"))?;
        let k = a.key.trim();
        let Some((_, sl, l)) = keys(md).find(|x| x.0.name == k) else {
            let have: Vec<&str> = keys(md).map(|x| x.0.name.as_str()).collect();
            return Err(match md.keys.iter().any(|x| x.name == k) {
                true => format!("{k}: x5desk does not know how to write it"),
                false => format!(
                    "{m} has no setting {k:?}; its settings: {}",
                    have.join(", ")
                ),
            });
        };
        if !conf::wired(sl) {
            return Err(format!(
                "{k} ({l}) is read only here: its encoding is not known for sure"
            ));
        }
        let v = a.value.spelt();
        let al = st.allowed(k);
        let shown = |v: &str| conf::show(sl, v);
        if al.len() == 1 {
            let dep = md
                .keys
                .iter()
                .find(|x| x.name == k)
                .map(|x| x.dep.join(", "));
            return match same(sl, &al[0], &v) {
                true => ok(
                    json!({ "key": k, "value": al[0], "shown": shown(&al[0]), "note": "in effect already" }),
                ),
                false => Err(format!(
                    "{k} is fixed at {} now by {}: change that first",
                    al[0],
                    dep.unwrap_or_default()
                )),
            };
        }
        let Some(v) = al.iter().find(|x| same(sl, x, &v)).cloned() else {
            return Err(format!(
                "{k} = {v}: not allowed in {m} now (allowed: {})",
                al.join(", ")
            ));
        };
        if st.cur(md, k).is_some_and(|c| same(sl, &c, &v)) {
            return ok(json!({ "key": k, "value": v, "shown": shown(&v), "note": "set already" }));
        }
        {
            let s = sh.st.lock().unwrap();
            if capturing(&s) {
                return Err("the camera captures: no setting change until it stops".into());
            }
            if s.rec.is_some() {
                return Err(REC.into());
            }
        }
        if !(trusted(&st) && CHECKED.contains(&(m.as_str(), k))) && !self.untested_ok() {
            return Err(self.untested(format!("writing {k} ({l}) in {m}")));
        }
        let fm = conf::func(&m).ok_or_else(|| format!("{m}: x5desk cannot write its settings"))?;
        if conf::write(sl, &v, &st.po, fm, conf::photo(&m)).is_none() {
            return Err(format!("{k} = {v}: x5desk cannot write it"));
        }
        let o = self
            .asked(
                "capture_set",
                Cap::Set(k.into(), v.clone()),
                Seen::of(&st),
                WRITE,
            )
            .await?;
        if !matches!(o, Out::Sent(_)) {
            return Err(no(o));
        }
        let back = |s: &State| {
            let c = &s.cap;
            let md = c.conf.as_ref()?.block(c.both)?.mode(&m)?;
            (s.csid == s.tsid && c.mode.as_ref() == Some(&m) && c.vals.as_ref() == Some(&m))
                .then(|| c.cur(md, k))?
        };
        let r = sh
            .until(10.0, |s| back(s).filter(|c| same(sl, c, &v)))
            .await;
        match r {
            Some(_) => ok(json!({ "key": k, "value": v, "shown": shown(&v) })),
            None => {
                let now = back(&sh.st.lock().unwrap());
                Err(format!(
                    "{k} = {v} went out, but the camera reports {}",
                    now.map_or("no value".into(), |c| format!("{k} = {c}"))
                ))
            }
        }
    }

    pub(super) async fn capture_photo(
        &self,
        a: PhotoArgs,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<CallToolResult, String> {
        let sh = self.sh();
        // a cancel before the shutter takes none, also while it waits its turn or for the capture state
        const OFF: &str = "cancelled before the shutter: no photo taken";
        let _a = tokio::select! {
            a = sh.act.lock() => a,
            _ = ctx.ct.cancelled() => return Err(OFF.into()),
        };
        self.free()?;
        let st = tokio::select! {
            st = self.cap(Need::Vals) => st?,
            _ = ctx.ct.cancelled() => return Err(OFF.into()),
        };
        let m = st.mode.clone().unwrap_or_default();
        if !matches!(conf::shot(&m), Shot::Photo(_)) {
            return Err(format!(
                "{m} ({}) is a {} mode: capture_photo needs a photo mode (capture_mode PHOTO_SINGLE)",
                conf::label(&m),
                shot_kind(&m)
            ));
        }
        if capturing(&sh.st.lock().unwrap()) {
            return Err("the camera captures: no photo until it stops".into());
        }
        if !(trusted(&st) && SHOT.contains(&m.as_str())) && !self.untested_ok() {
            return Err(self.untested(format!("the shutter of {m} ({})", conf::label(&m))));
        }
        // the X5 leaves the self-timer to the app
        let t = match a.self_timer.unwrap_or(true) {
            true => st.po.photography_self_timer.min(TIMER),
            false => 0,
        };
        if t > 0 {
            let say = |_: &State, e: f64| format!("self-timer: {:.0} s", (t as f64 - e).max(0.0));
            let done = self.watch(ctx, t as f64, say, |_| None::<()>).await;
            if let Err(e) = done {
                sh.log
                    .line(&format!("{}: the self-timer {e}: no photo", self.client));
                return Err(format!("{e} during the self-timer: no photo taken"));
            }
        }
        if ctx.ct.is_cancelled() {
            sh.log.line(&format!(
                "{}: cancelled before the shutter: no photo",
                self.client
            ));
            return Err(OFF.into());
        }
        // the camera as it was checked: a mode, lens or sub-mode changed during the self-timer refuses it
        let o = self
            .asked("capture_photo", Cap::Shoot, Seen::of(&st), SHUTTER)
            .await?;
        let Out::Sent(u) = o else { return Err(no(o)) };
        let Some(u) = u else {
            return ok(
                json!({ "uri": null, "note": "the camera took it but named no file: album_list lists it" }),
            );
        };
        let key = album::key_of(&u);
        let listed = self.listed(&key, 10.0).await;
        ok(json!({
            "uri": u,
            "file": album::name(&u),
            "key": key,
            "in_album": listed,
            "self_timer_s": t,
            "next": "album_download downloads it by its key",
        }))
    }

    /// Stops the camera's capture (this host's or not); the clip's URI, none when the camera named none; Ok(None) with
    /// `already` when it did not capture.
    async fn stop(&self) -> Result<(Option<String>, bool), String> {
        let sh = self.sh();
        self.cap(Need::Mode).await?;
        // a stop of this host's still on its way (one that gave up waiting): its outcome waited for, not a second stop
        let pending = {
            let s = sh.st.lock().unwrap();
            (s.rec.as_ref().and_then(|r| r.stop)).filter(|k| !s.outs.iter().any(|x| x.0 == *k))
        };
        // a recording this host started: this stop's outcome ends it (the capture thread takes it)
        let n = pending.unwrap_or_else(|| {
            sh.ask_with("capture_record", Cap::Stop, None, |s, n| {
                if let Some(r) = &mut s.rec {
                    r.stop = Some(n)
                }
            })
        });
        let o = sh.until(SHUTTER, |s| {
            s.outs.iter().find(|x| x.0 == n).map(|x| x.1.clone())
        });
        let o = o.await.ok_or(
            "no answer to the stop within 45 s; capture_status shows whether the camera records",
        )?;
        let r = match o {
            Out::Sent(u) => (u, false),
            Out::No(t) if t.contains("nothing to stop") => (None, true),
            o => return Err(no(o)),
        };
        // the camera says it stopped a moment later: the tools after this one see it idle
        sh.until(5.0, |s| (!capturing(s)).then_some(())).await;
        Ok(r)
    }

    /// The stop of recording `n` this host started with `seconds`, now due (the host's timer, `host::timer`, again
    /// while it fails).
    pub(crate) async fn timed_stop(&self, n: u64) -> Result<(), String> {
        let sh = self.sh();
        let _a = sh.act.lock().await;
        if !sh.st.lock().unwrap().rec.as_ref().is_some_and(|r| r.n == n) {
            return Ok(());
        }
        self.stop().await?;
        // a start the camera has not shown capturing yet stays this host's: stopped once it does
        if sh.st.lock().unwrap().rec.as_ref().is_some_and(|r| r.n == n) {
            return Err("the camera has not begun capturing yet".into());
        }
        sh.log.line(&format!("timed stop of capture {n}"));
        Ok(())
    }

    /// A clip's answer: URI, file, key, whether listed.
    async fn clip(&self, u: Option<String>) -> Value {
        match u {
            Some(u) => {
                let key = album::key_of(&u);
                let listed = self.listed(&key, 10.0).await;
                json!({
                    "uri": u,
                    "file": album::name(&u),
                    "key": key,
                    "in_album": listed,
                    "next": "album_download downloads it by its key",
                })
            }
            None => json!({ "uri": null, "note": "the camera named no file: album_list lists it" }),
        }
    }

    pub(super) async fn capture_record(
        &self,
        a: RecordArgs,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<CallToolResult, String> {
        let sh = self.sh();
        match a.action {
            RecAction::Stop => {
                let _a = sh.act.lock().await;
                self.free()?;
                let (u, already) = self.stop().await?;
                if already {
                    let s = sh.st.lock().unwrap();
                    // its start went out moments ago: the host stops it once the camera shows it capturing
                    if s.rec.is_some() {
                        return ok(json!({
                            "capturing": false,
                            "note": "the start went out moments ago and the camera has not begun: the host stops it \
                                     as soon as it does",
                        }));
                    }
                    let why = s.ended.clone().map(|e| match e.1 {
                        Ok(_) => "stopped by this host".to_string(),
                        Err(w) => w,
                    });
                    return ok(json!({
                        "capturing": false,
                        "note": "the camera did not capture",
                        "ended": why,
                        "next": "album_list lists the newest clip",
                    }));
                }
                let mut v = self.clip(u).await;
                v["capturing"] = false.into();
                ok(v)
            }
            RecAction::Wait => {
                let w = a.wait_s.unwrap_or(60.0).clamp(0.0, 300.0);
                let say = |s: &State, e: f64| match capturing(s) {
                    true => format!("capturing, waited {e:.0} s"),
                    false => "stopped".into(),
                };
                let t0 = Instant::now();
                let r = self
                    .watch(ctx, w, say, |s| (!capturing(s)).then_some(()))
                    .await?;
                let s = sh.st.lock().unwrap();
                ok(json!({
                    "capturing": r.is_none(),
                    "waited_s": (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
                    "started_by_this_host": s.rec.is_some(),
                }))
            }
            RecAction::Start => self.start(a.seconds, ctx).await,
        }
    }

    async fn start(
        &self,
        seconds: Option<f64>,
        ctx: &RequestContext<RoleServer>,
    ) -> Result<CallToolResult, String> {
        let sh = self.sh();
        if let Some(x) = seconds
            && !(1.0..=600.0).contains(&x)
        {
            return Err(format!("seconds {x}: 1 to 600"));
        }
        let a = sh.act.lock().await;
        self.free()?;
        if sh.st.lock().unwrap().rec.is_some() {
            return Err(REC.into());
        }
        let st = self.cap(Need::Vals).await?;
        let m = st.mode.clone().unwrap_or_default();
        if !matches!(conf::shot(&m), Shot::Rec(_) | Shot::Lapse(_)) {
            return Err(format!(
                "{m} ({}) is a {} mode: capture_record needs a video or timelapse mode (capture_mode VIDEO_NORMAL)",
                conf::label(&m),
                shot_kind(&m)
            ));
        }
        if capturing(&sh.st.lock().unwrap()) {
            return Err("the camera captures already: capture_record stop stops it".into());
        }
        if !(trusted(&st) && SHOT.contains(&m.as_str())) && !self.untested_ok() {
            return Err(self.untested(format!("recording in {m} ({})", conf::label(&m))));
        }
        if sh
            .st
            .lock()
            .unwrap()
            .info
            .battery
            .as_ref()
            .is_some_and(cc::empty)
        {
            return Err("battery at 5 % or less: charge the camera to record".into());
        }
        // this host's from the moment it is asked: a host ending meanwhile stops it once it went out
        let at = Instant::now();
        let n = sh.ask_with("capture_record", Cap::Start, Seen::of(&st), |s, n| {
            s.rec = Some(Rec {
                serial: s.info.id.as_ref().map(|i| i.serial.clone()),
                n,
                sent: false,
                seen: false,
                mode: m.clone(),
                at,
                until: seconds.map(|x| at + Duration::from_secs_f64(x)),
                stop: None,
                sent_at: None,
            })
        });
        let o = sh
            .until(SHUTTER, |s| {
                s.outs.iter().find(|x| x.0 == n).map(|x| x.1.clone())
            })
            .await;
        let o = o.ok_or(
            "no answer to the start within 45 s; capture_status shows whether the camera records",
        )?;
        if !matches!(o, Out::Sent(_)) {
            return Err(no(o));
        }
        // the camera says it records a moment later
        let seen = sh
            .until(5.0, |s| capturing(s).then_some(()))
            .await
            .is_some();
        // with `seconds` the host's timer stops it (host::timer), also when this call is gone
        drop(a);
        let mut v = json!({
            "capturing": seen,
            "mode": m,
            "stops_in_s": seconds,
            "note": match seen {
                true => "recording; it stops when this host ends unless stopped before",
                false => "the start went out, but the camera has not said it records yet: capture_status shows it",
            },
        });
        // a short one: its clip waited for
        if let Some(x) = seconds.filter(|x| *x <= TIMED) {
            let say = |_: &State, e: f64| format!("recording: {e:.0} of {x:.0} s");
            let r = self
                .watch(ctx, x + SHUTTER, say, |s| {
                    s.ended.clone().filter(|e| e.0 == n).map(|e| e.1)
                })
                .await?;
            return match r {
                Some(Ok(u)) => {
                    let mut c = self.clip(u).await;
                    c["capturing"] = false.into();
                    c["seconds"] = x.into();
                    ok(c)
                }
                Some(Err(w)) => {
                    let c = capturing(&sh.st.lock().unwrap());
                    ok(json!({ "capturing": c, "ended": w, "next": "album_list lists the clip" }))
                }
                None => {
                    v["capturing"] = capturing(&sh.st.lock().unwrap()).into();
                    v["note"] = "its stop has not come yet (the host tries again): capture_record wait waits for \
                                 it"
                    .into();
                    ok(v)
                }
            };
        }
        ok(v)
    }

    pub(super) async fn capture_mark(&self) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let _a = sh.act.lock().await;
        self.free()?;
        let st = self.cap(Need::Mode).await?;
        let m = st.mode.clone().unwrap_or_default();
        if !capturing(&sh.st.lock().unwrap()) {
            return Err("the camera does not record: a highlight marks a running recording".into());
        }
        if !conf::marks(&m) {
            return Err(format!(
                "{m} ({}) takes no highlight marks",
                conf::label(&m)
            ));
        }
        if !(trusted(&st) && m == "VIDEO_NORMAL") && !self.untested_ok() {
            return Err(self.untested(format!("highlight marks in {m} ({})", conf::label(&m))));
        }
        let o = self
            .asked("capture_mark", Cap::Mark, Seen::of(&st), 15.0)
            .await?;
        if !matches!(o, Out::Sent(_)) {
            return Err(no(o));
        }
        let at = sh
            .st
            .lock()
            .unwrap()
            .cap
            .since
            .map(|t| t.elapsed().as_secs_f64());
        ok(json!({ "marked_at_s": at.map(|x| (x * 10.0).round() / 10.0) }))
    }

    /// The camera's album items of `keys` (deduplicated), with the keys it does not list; refused when the album
    /// shows a card or the camera is not connected.
    async fn camera_items(&self, keys: &[String]) -> Result<(Vec<Item>, Vec<String>), String> {
        if keys.is_empty() {
            return Err("give the keys of the items (album_list lists them)".into());
        }
        let (items, src) = self.album().await?;
        if let Some(r) = src {
            return Err(format!(
                "the album shows the card {}: card_close shows the camera's files again",
                r.display()
            ));
        }
        if self.sh().st.lock().unwrap().info.conn != Link::On {
            return Err(
                "the camera is not connected: camera_connect connects it (camera tier)".into(),
            );
        }
        let mut seen = HashSet::new();
        let (mut have, mut bad) = (vec![], vec![]);
        for k in keys.iter().filter(|k| seen.insert(k.as_str())) {
            match items.iter().find(|i| i.key == *k) {
                Some(i) => have.push(i.clone()),
                None => bad.push(k.clone()),
            }
        }
        Ok((have, bad))
    }

    /// Waits for the camera worker's note after number `seq` that starts with one of `w` (or says it was not
    /// connected), up to `secs`; then for the camera's listing after it (10 s). The note and whether it is an error.
    async fn noted(&self, seq: u64, w: &[&str], secs: f64) -> Option<(u64, String, bool)> {
        let sh = self.sh();
        let n = sh.until(secs, |s| {
            s.since(seq)
                .find(|n| n.1 == "not connected" || w.iter().any(|w| n.1.starts_with(w)))
                .cloned()
        });
        let n = n.await?;
        sh.until(10.0, |s| {
            (s.album_seq >= n.0 && s.album_cam && s.album.is_some()).then_some(())
        })
        .await;
        Some(n)
    }

    pub(super) async fn camera_favourite(&self, a: FavArgs) -> Result<CallToolResult, String> {
        if !self.untested_ok() {
            return Err(self.untested(
                "Favourites (the X5 on firmware v1.13.21 refused every form tried)".into(),
            ));
        }
        let sh = self.sh();
        let _a = sh.act.lock().await;
        self.free()?;
        if capturing(&sh.st.lock().unwrap()) {
            return Err("the camera captures: its album changes once it stops".into());
        }
        let (items, bad) = self.camera_items(&a.keys).await?;
        if !bad.is_empty() {
            return Err(format!(
                "no items {bad:?} in the camera's album: album_list lists the keys"
            ));
        }
        let seq = sh.st.lock().unwrap().seq;
        sh.cmd(Cmd::Fav(
            items.iter().map(|i| (i.flag().to_string(), a.on)).collect(),
        ));
        let n = self
            .noted(seq, &["favourite flag set", "favourites:"], 40.0)
            .await
            .ok_or("no answer from the camera within 40 s; camera_status shows its notes")?;
        if n.2 {
            return Err(format!("the camera refused: {}", n.1));
        }
        let s = sh.st.lock().unwrap();
        let now = s.album.as_deref().unwrap_or_default();
        let v: Vec<Value> = (items.iter())
            .map(|i| {
                let f = now.iter().find(|x| x.key == i.key).map(|x| x.fav);
                json!({ "key": i.key, "favourite": f })
            })
            .collect();
        ok(json!({ "items": v, "note": n.1 }))
    }

    /// How the item's files are kept here: each one a finished download of that very URI whose file has the size the
    /// camera sent for it, as a job of this host's or an earlier one's recorded it (`verified`); each one there at
    /// least by name (the agent folder, the app's download folders on a real host: `by_name`); or not every one
    /// (`none`).
    fn kept(
        &self,
        it: &Item,
        jobs: &crate::job::Jobs,
        q: &dl::Queue,
        dirs: &[PathBuf],
    ) -> &'static str {
        let mut verified = true;
        for u in &it.uris {
            let size = |d: &dl::Dl| {
                (jobs.v.iter().flat_map(|j| &j.files))
                    .find(|f| f.uri == *u && f.dst == d.dst && f.size.is_some())
                    .and_then(|f| f.size)
            };
            let ok = q.v.iter().any(|d| {
                d.st == dl::St::Done
                    && d.uri == *u
                    && size(d).is_some_and(|n| {
                        std::fs::metadata(&d.dst).is_ok_and(|m| m.is_file() && m.len() == n)
                    })
            });
            if ok {
                continue;
            }
            verified = false;
            let n = album::name(u);
            if !album::safe(n) || !dirs.iter().any(|d| d.join(n).is_file()) {
                return "none";
            }
        }
        if verified { "verified" } else { "by_name" }
    }

    pub(super) async fn camera_delete(&self, a: DeleteArgs) -> Result<CallToolResult, String> {
        let sh = self.sh();
        let _a = sh.act.lock().await;
        self.free()?;
        {
            let s = sh.st.lock().unwrap();
            if s.info.conn != Link::On && !self.tiers().has(Tier::Camera) {
                return Err(
                    "the camera is not connected, and connecting it needs the camera tier: the user enables it \
                     with `x5desk mcp enable --tiers <tiers>,camera,delete`"
                        .into(),
                );
            }
            if capturing(&s) {
                return Err("the camera captures: its files can be deleted once it stops".into());
            }
            if s.rec.is_some() {
                return Err("a recording this host started (or its start) is on the camera: capture_record stop \
                            first"
                    .into());
            }
        }
        let (items, bad) = self.camera_items(&a.keys).await?;
        let (gone, bad): (Vec<String>, Vec<String>) = {
            let s = sh.st.lock().unwrap();
            bad.into_iter().partition(|k| s.deleted.contains(k))
        };
        if !bad.is_empty() {
            return Err(format!(
                "no items {bad:?} in the camera's album (nothing deleted): album_list lists the keys"
            ));
        }
        let dirs: Vec<PathBuf> = [sh.agent.clone()]
            .into_iter()
            .chain(if sh.downloads {
                paths::downloads()
            } else {
                vec![]
            })
            .collect();
        let (mut del, mut keep) = (vec![], vec![]);
        {
            let (jobs, q) = (sh.jobs.lock().unwrap(), sh.q.lock().unwrap());
            let busy: Vec<&str> = (items.iter())
                .filter(|i| {
                    i.uris
                        .iter()
                        .any(|u| (q.v.iter()).any(|d| d.uri == *u && wanted(&q, &d.dst)))
                })
                .map(|i| i.key.as_str())
                .collect();
            if !busy.is_empty() {
                let js: Vec<&str> = (jobs.v.iter())
                    .filter(|j| j.keys.iter().any(|k| busy.contains(&k.as_str())))
                    .map(|j| j.h.as_str())
                    .collect();
                return Err(format!(
                    "{busy:?} are being downloaded (jobs {js:?}): wait for them (job_wait) or cancel them \
                     (job_cancel) first; nothing deleted"
                ));
            }
            for i in items {
                let k = self.kept(&i, &jobs, &q, &dirs);
                match a.downloaded_only.unwrap_or(false) && k != "verified" {
                    true => keep.push(json!({ "key": i.key, "local": k })),
                    false => del.push((i, k)),
                }
            }
        }
        if del.is_empty() {
            return ok(json!({
                "deleted": [],
                "already_deleted": gone,
                "kept_not_downloaded": keep,
            }));
        }
        let mut uris: Vec<String> = vec![];
        for (i, _) in &del {
            for u in &i.uris {
                if !uris.contains(u) {
                    uris.push(u.clone())
                }
            }
        }
        let seq = {
            let mut s = sh.st.lock().unwrap();
            s.deleting = Some(s.seq);
            s.seq
        };
        sh.log
            .line(&format!("{} deletes {} files", self.client, uris.len()));
        // the worker's passes: a batch, then one file at a time, 30 s each at most, five passes
        let secs = (60.0 + uris.len() as f64 * 0.25).min(900.0);
        sh.cmd(Cmd::Delete(uris));
        let Some(n) = self.noted(seq, &["deleted ", "delete:"], secs).await else {
            return Err(format!(
                "the delete still runs after {secs:.0} s; camera_status shows its outcome, and the camera's other \
                 writes wait for it"
            ));
        };
        let s = sh.st.lock().unwrap();
        let fresh = s.album_seq >= n.0 && s.album_cam;
        let now: HashSet<&String> = match fresh {
            true => s.album.iter().flatten().flat_map(|i| &i.uris).collect(),
            false => HashSet::new(),
        };
        let (mut done, mut part, mut left, mut never) = (vec![], vec![], vec![], vec![]);
        let mut dk = vec![];
        for (i, k) in &del {
            let (g, l): (Vec<&String>, Vec<&String>) =
                i.uris.iter().partition(|u| !now.contains(u));
            let names = |v: &[&String]| -> Vec<String> {
                v.iter().map(|u| album::name(u).to_string()).collect()
            };
            match (fresh, g.is_empty(), l.is_empty()) {
                (false, ..) => {}
                (true, _, true) => {
                    done.push(json!({ "key": i.key, "local": k }));
                    dk.push(i.key.clone());
                    if *k == "none" {
                        never.push(i.key.clone())
                    }
                }
                (true, true, _) => left.push(i.key.clone()),
                (true, false, false) => part.push(
                    json!({ "key": i.key, "gone": names(&g), "left": names(&l), "local": k }),
                ),
            }
        }
        drop(s);
        sh.st.lock().unwrap().deleted.extend(dk);
        let mut v = json!({
            "deleted": done,
            "partly_deleted": part,
            "not_deleted": left,
            "already_deleted": gone,
            "kept_not_downloaded": keep,
            "never_downloaded": never,
            "note": n.1,
        });
        if !fresh {
            v["note"] = format!(
                "{}; the album was not listed after it, so what is gone is not known",
                n.1
            )
            .into();
        }
        match n.2 {
            true => Err(format!("the delete failed: {}\n{v}", n.1)),
            false => ok(v),
        }
    }
}
