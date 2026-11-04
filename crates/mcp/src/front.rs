//! The front: every `x5desk mcp` speaks MCP on its own stdio and passes whole JSON-RPC lines to a host, the one in
//! this process (over a duplex: hosting and proxying look alike here) or another's over the endpoint ([`Elect`]). It
//! keeps the client's `initialize` and `notifications/initialized` (clients on 2025-11-25 handshake once; a service
//! that never saw `initialize` refuses them) and replays them to every later host, under an id of its own whose answer
//! the client never sees; 2026-07-28 clients send none. When the host goes, each request it had not answered gets an
//! error (-32603, [`ENDED`]) and the front elects again: it binds the endpoint and hosts, or connects to whoever did;
//! the switch goes to stderr. At stdin's end the front sends nothing more and passes on what the host still answers,
//! until every open request (not one the client cancelled: rmcp drops an answer not sent yet) has its answer, the
//! link ends or
//! [`DRAIN`] passes. A host also serves the endpoint's
//! connections, and ends with its own client: after stdin's end (or at a signal) it stops accepting, then releases the
//! camera, downloads keeping their `.part`. `--tiers` narrows this process's client only: here, whichever host it
//! has (a call outside them, of a tool it does not know or a line that is no JSON object never reaches the host), and
//! in a host of this process its `stdio` connection; the host serves the endpoint's with the tiers enabled.

use crate::{
    Host, Opts, Tiers,
    endpoint::{self, Read, Role, Write},
    server::{TOOLS, left_out},
};
use serde_json::{Value, json};
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::mpsc::{self, UnboundedReceiver, UnboundedSender},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

/// A line longer than this is dropped, as rmcp drops one.
pub const MAX: usize = 16 << 20;

/// What a request gets (code -32603) when its host went before answering it.
pub const ENDED: &str = "host ended; try again";

/// How long the answers to open requests are passed on after stdin's end (as rmcp's own drain).
pub const DRAIN: Duration = Duration::from_secs(5);

/// A UTF-8 byte order mark, which rmcp skips at a line's start: so does the front.
const BOM: &[u8] = b"\xEF\xBB\xBF";

/// Bytes cut into whole lines, each ending in `\n`; blank lines are left out, one over [`MAX`] bytes is dropped.
#[derive(Default)]
pub struct Lines {
    buf: Vec<u8>,
    /// the line under way is over [`MAX`]: dropped up to its end
    skip: bool,
}

impl Lines {
    /// The lines `b` completes, in order.
    pub fn push(&mut self, mut b: &[u8]) -> Vec<Vec<u8>> {
        let mut v = vec![];
        while !b.is_empty() {
            let (part, end) = match b.iter().position(|&c| c == b'\n') {
                Some(i) => (&b[..=i], true),
                None => (b, false),
            };
            b = &b[part.len()..];
            if !self.skip {
                match self.buf.len() + part.len() > MAX {
                    true => (self.buf, self.skip) = (vec![], true),
                    false => self.buf.extend_from_slice(part),
                }
            }
            if end {
                let l = std::mem::take(&mut self.buf);
                if !std::mem::take(&mut self.skip) && l.iter().any(|c| !c.is_ascii_whitespace()) {
                    v.push(l)
                }
            }
        }
        v
    }
}

/// What a line is, for the front's books.
#[derive(Debug, PartialEq)]
enum Msg {
    Request { id: Value, method: String },
    Note(String),
    Answer(Value),
    Other,
}

fn msg(v: &Value) -> Msg {
    let id = v.get("id").filter(|i| !i.is_null()).cloned();
    match (v.get("method").and_then(Value::as_str), id) {
        (Some(m), Some(id)) => Msg::Request {
            id,
            method: m.into(),
        },
        (Some(m), None) => Msg::Note(m.into()),
        (None, Some(id)) if v.get("result").is_some() || v.get("error").is_some() => {
            Msg::Answer(id)
        }
        _ => Msg::Other,
    }
}

fn line(v: &Value) -> Vec<u8> {
    let mut b = serde_json::to_vec(v).unwrap_or_default();
    b.push(b'\n');
    b
}

/// A link to a host: its two directions, and what the front says of it on stderr.
pub(crate) struct Link {
    /// what the host writes
    pub r: Read,
    /// what the host reads
    pub w: Write,
    /// `hosting as …`, `proxy to the host at …`
    pub name: String,
}

/// Where a front finds its hosts.
pub(crate) trait Attach {
    /// A link to a host, this process's or another's; an error ends the front (no host can be had).
    async fn attach(&mut self) -> Result<Link, String>;
    /// The front ends: a host of this process stops.
    async fn end(&mut self);
}

/// How a pump ended.
enum End {
    /// stdin's end (its answers passed on), stdout gone, or a signal
    Client,
    /// the link to the host dropped
    Host,
}

/// The client's side of the lines.
pub(crate) struct Front<A> {
    at: A,
    narrow: Option<Tiers>,
    /// the client's `initialize` request and `notifications/initialized`, replayed to each new host
    init: Option<Value>,
    inited: Option<Vec<u8>>,
    /// requests forwarded and neither answered nor cancelled by the client: id (as JSON text) → id, method
    open: HashMap<String, (Value, String)>,
    /// the replayed `initialize`'s id (as JSON text), whose answer the client never sees
    swallow: Option<String>,
    replays: u64,
}

/// The link's halves as channels: a task writes what is sent, one reads its lines; each ends with the link.
fn pipe(mut r: Read, mut w: Write) -> (UnboundedSender<Vec<u8>>, UnboundedReceiver<Vec<u8>>) {
    let (tx, mut to) = mpsc::unbounded_channel::<Vec<u8>>();
    let (from, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(l) = to.recv().await {
            if w.write_all(&l).await.is_err() || w.flush().await.is_err() {
                return;
            }
        }
        let _ = w.shutdown().await;
    });
    tokio::spawn(async move {
        let (mut ls, mut b) = (Lines::default(), vec![0; 1 << 16]);
        while let Ok(n @ 1..) = r.read(&mut b).await {
            for l in ls.push(&b[..n]) {
                if from.send(l).is_err() {
                    return;
                }
            }
        }
    });
    (tx, rx)
}

impl<A: Attach> Front<A> {
    /// A front finding hosts through `at`; `narrow`: the command line's `--tiers`.
    pub fn new(at: A, narrow: Option<Tiers>) -> Front<A> {
        Front {
            at,
            narrow,
            init: None,
            inited: None,
            open: HashMap::new(),
            swallow: None,
            replays: 0,
        }
    }

    /// Pumps lines between the client (`stdin`, `out`) and its hosts until the client goes (at stdin's end once the
    /// open requests are answered, the link ends or [`DRAIN`] passes) or `stop` is cancelled. Err: no host could be
    /// had.
    pub async fn run(
        &mut self,
        stdin: &mut UnboundedReceiver<Vec<u8>>,
        out: &std::sync::mpsc::Sender<Vec<u8>>,
        stop: &CancellationToken,
    ) -> Result<(), String> {
        let mut back = Duration::ZERO;
        loop {
            let k = self.at.attach().await?;
            eprintln!("x5desk mcp: {}", k.name);
            let (tx, mut rx) = pipe(k.r, k.w);
            self.replay(&tx);
            let (mut tx, mut heard) = (Some(tx), false);
            // stdin ended: until then the host's answers are passed on (it has seen the end too)
            let mut drain = None;
            let end = loop {
                tokio::select! {
                    l = stdin.recv(), if tx.is_some() => match (l, &tx) {
                        (Some(l), Some(t)) => if let Err(e) = self.up(l, t, out) { break e },
                        _ => {
                            tx = None;
                            if self.open.is_empty() {
                                break End::Client;
                            }
                            drain = Some(tokio::time::Instant::now() + DRAIN);
                        }
                    },
                    l = rx.recv() => match l {
                        Some(l) => {
                            heard = true;
                            if let Err(e) = self.down(l, out) { break e }
                            if tx.is_none() && self.open.is_empty() {
                                break End::Client;
                            }
                        }
                        None if tx.is_none() => break End::Client,
                        None => break End::Host,
                    },
                    _ = tokio::time::sleep_until(drain.unwrap_or_else(tokio::time::Instant::now)),
                        if drain.is_some() => break End::Client,
                    _ = stop.cancelled() => break End::Client,
                }
            };
            if let End::Client = end {
                return Ok(());
            }
            let n = self.open.len();
            for (_, (id, _)) in self.open.drain() {
                let e = json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32603, "message": ENDED } });
                let _ = out.send(line(&e));
            }
            self.swallow = None;
            eprintln!(
                "x5desk mcp: the host went away, {n} open requests answered with an error; electing again"
            );
            // a host that drops us before a word (one dying, its socket still there): not at once again
            back = match heard {
                true => Duration::ZERO,
                false => (back * 2).clamp(Duration::from_millis(50), Duration::from_secs(2)),
            };
            tokio::select! {
                _ = tokio::time::sleep(back) => {}
                _ = stop.cancelled() => return Ok(()),
            }
        }
    }

    /// The host of this process stops (camera released, `.part` kept).
    pub async fn end(&mut self) {
        self.at.end().await
    }

    /// The client's handshake to a new host, its answer swallowed.
    fn replay(&mut self, tx: &UnboundedSender<Vec<u8>>) {
        let Some(mut i) = self.init.clone() else {
            return;
        };
        self.replays += 1;
        let id = json!(format!("x5desk-replay-{}", self.replays));
        self.swallow = Some(id.to_string());
        i["id"] = id;
        let _ = tx.send(line(&i));
        if let Some(n) = &self.inited {
            let _ = tx.send(n.clone());
        }
    }

    /// What `--tiers` answers instead of the host (the answer without `jsonrpc` and `id`): a call of a tool in a tier
    /// left out (the tool's error), or of a name that is none of [`TOOLS`] (a protocol error, as the host's).
    fn refused(&self, method: &str, v: &Value) -> Option<Value> {
        let t = self.narrow.filter(|_| method == "tools/call")?;
        let name = v.pointer("/params/name").and_then(Value::as_str);
        let Some(d) = name.and_then(|n| TOOLS.iter().find(|d| d.name == n)) else {
            let m = name.map_or("tools/call names no tool".into(), |n| {
                format!("unknown tool {n:?}")
            });
            return Some(json!({ "error": { "code": -32602, "message": m } }));
        };
        (!t.has(d.tier)).then(|| {
            let e = left_out(d.name, d.tier, t);
            json!({ "result": { "content": [{ "type": "text", "text": e }], "isError": true } })
        })
    }

    /// A client's line (a leading BOM cut off): to the host, or answered here. Under `--tiers` only JSON objects
    /// pass (rmcp takes no batches): a batch's requests are answered -32600 here, anything else dropped.
    fn up(
        &mut self,
        mut l: Vec<u8>,
        tx: &UnboundedSender<Vec<u8>>,
        out: &std::sync::mpsc::Sender<Vec<u8>>,
    ) -> Result<(), End> {
        if l.starts_with(BOM) {
            l.drain(..BOM.len());
        }
        let v = serde_json::from_slice::<Value>(&l).ok();
        match v {
            Some(v) if v.is_object() => match msg(&v) {
                Msg::Request { id, method } => {
                    if let Some(mut r) = self.refused(&method, &v) {
                        r["jsonrpc"] = json!("2.0");
                        r["id"] = id;
                        return out.send(line(&r)).map_err(|_| End::Client);
                    }
                    if method == "initialize" {
                        (self.init, self.inited) = (Some(v.clone()), None)
                    }
                    self.open.insert(id.to_string(), (id, method));
                }
                Msg::Note(m) if m == "notifications/initialized" => self.inited = Some(l.clone()),
                // rmcp drops the answer to a request the client cancelled unless sent already: not waited for at
                // stdin's end (one that comes is still passed on, narrowed)
                Msg::Note(m) if m == "notifications/cancelled" => {
                    if let Some(id) = v.pointer("/params/requestId") {
                        self.open.remove(&id.to_string());
                    }
                }
                _ => {}
            },
            v if self.narrow.is_some() => {
                let errs: Vec<Value> = (v.as_ref().and_then(Value::as_array).into_iter().flatten())
                    .filter_map(|x| x.get("id").filter(|i| !i.is_null()))
                    .map(|id| {
                        json!({ "jsonrpc": "2.0", "id": id,
                            "error": { "code": -32600, "message": "batches are not supported" } })
                    })
                    .collect();
                return match errs.is_empty() {
                    true => Ok(()),
                    false => out.send(line(&Value::Array(errs))).map_err(|_| End::Client),
                };
            }
            _ => {}
        }
        tx.send(l).map_err(|_| End::Host)
    }

    /// A host's line: to the client, unless it answers the replay; a tool list narrowed by `--tiers`.
    fn down(&mut self, mut l: Vec<u8>, out: &std::sync::mpsc::Sender<Vec<u8>>) -> Result<(), End> {
        if let Ok(mut v) = serde_json::from_slice::<Value>(&l)
            && let Msg::Answer(id) = msg(&v)
        {
            let k = id.to_string();
            if self.swallow.as_ref() == Some(&k) {
                self.swallow = None;
                if let Some(e) = v.get("error") {
                    eprintln!("x5desk mcp: the new host refused the client's initialize: {e}");
                }
                return Ok(());
            }
            self.open.remove(&k);
            // any tool list, also the answer to one the client cancelled (open no more, yet it may come)
            if let Some(t) = self.narrow
                && let Some(a) = v.pointer_mut("/result/tools").and_then(Value::as_array_mut)
            {
                a.retain(|x| (TOOLS.iter()).any(|d| x["name"] == d.name && t.has(d.tier)));
                l = line(&v);
            }
        }
        out.send(l).map_err(|_| End::Client)
    }
}

/// A host of this process: the endpoint's accept loop, stopped before the host shuts down.
struct Hosting {
    host: Arc<Host>,
    stop: CancellationToken,
    accept: Option<JoinHandle<()>>,
}

/// The [`Attach`] of a process: hosts when it wins the endpoint (or has none), else proxies to the endpoint's host.
pub(crate) struct Elect {
    opts: Opts,
    ep: Option<PathBuf>,
    hosting: Option<Hosting>,
}

/// Serves the endpoint's connections on `host` until `stop`, then closes the listener (the socket goes).
async fn accept(host: Arc<Host>, mut l: endpoint::Listener, stop: CancellationToken) {
    let mut n = 0u64;
    loop {
        let r = tokio::select! {
            r = l.accept() => r,
            _ = stop.cancelled() => break,
            _ = host.closed() => break,
        };
        match r {
            Ok((r, w)) => {
                n += 1;
                let (h, c) = (host.clone(), format!("endpoint {n}"));
                tokio::spawn(async move {
                    if let Err(e) = h.serve(r, w, &c, None).await {
                        h.log().line(&e)
                    }
                });
            }
            Err(e) => {
                host.log().line(&format!("endpoint: {e}"));
                tokio::time::sleep(Duration::from_millis(100)).await
            }
        }
    }
}

impl Elect {
    /// Hosts with `opts` (its `narrow`, the command line's `--tiers`, for this process's own client only); `ep`: the
    /// endpoint, none to host alone.
    pub fn new(opts: Opts, ep: Option<PathBuf>) -> Elect {
        Elect {
            opts,
            ep,
            hosting: None,
        }
    }

    /// Hosts with the tiers enabled: this process's `--tiers` narrow only its own client ([`Elect::local`]).
    async fn host(&mut self, l: Option<endpoint::Listener>) -> Result<(), String> {
        let o = Opts {
            narrow: None,
            ..self.opts.clone()
        };
        let host = Host::new(o).await?;
        if host.tiers().is_empty() {
            eprintln!(
                "x5desk mcp: no tools enabled; the user enables them with `x5desk mcp enable --tiers read,local`"
            );
        }
        let stop = CancellationToken::new();
        let accept = l.map(|l| tokio::spawn(accept(host.clone(), l, stop.clone())));
        self.hosting = Some(Hosting { host, stop, accept });
        Ok(())
    }

    /// A link to the host of this process, served as `stdio` narrowed by this process's `--tiers`.
    fn local(&self) -> Link {
        let h = self.hosting.as_ref().unwrap();
        let (a, b) = tokio::io::duplex(1 << 20);
        let (hr, hw) = tokio::io::split(b);
        let (host, narrow) = (h.host.clone(), self.opts.narrow);
        tokio::spawn(async move {
            if let Err(e) = host.serve(hr, hw, "stdio", narrow).await {
                host.log().line(&e)
            }
        });
        let at = match (&self.ep, &h.accept) {
            (Some(p), Some(_)) => format!("endpoint {}", p.display()),
            _ => "no endpoint".into(),
        };
        let mine = narrow.map_or(String::new(), |n| format!(", this client's --tiers {n}"));
        let (r, w) = tokio::io::split(a);
        Link {
            r: Box::new(r),
            w: Box::new(w),
            name: format!(
                "hosting as {} (tiers {}{mine}, {at})",
                h.host.id(),
                h.host.tiers()
            ),
        }
    }
}

impl Attach for Elect {
    async fn attach(&mut self) -> Result<Link, String> {
        if self.hosting.as_ref().is_some_and(|h| !h.host.is_closed()) {
            return Ok(self.local());
        }
        self.end().await;
        match &self.ep {
            Some(ep) => match endpoint::elect(ep).await {
                Ok(Role::Proxy(r, w)) => {
                    if !self.opts.allow.is_empty() {
                        eprintln!("x5desk mcp: --allow-dir counts only while this process hosts");
                    }
                    let name = format!("proxy to the host at {}", ep.display());
                    return Ok(Link { r, w, name });
                }
                Ok(Role::Host(l)) => self.host(Some(l)).await?,
                Err(e) => {
                    eprintln!(
                        "x5desk mcp: endpoint {}: {e}; hosting without it",
                        ep.display()
                    );
                    self.host(None).await?
                }
            },
            None => self.host(None).await?,
        }
        Ok(self.local())
    }

    /// Stops accepting (the socket goes first, so a proxy's election finds it stale), then shuts the host down.
    async fn end(&mut self) {
        if let Some(h) = self.hosting.take() {
            h.stop.cancel();
            if let Some(a) = h.accept {
                let _ = a.await;
            }
            h.host.shutdown().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use tokio::io::{AsyncBufReadExt, BufReader, DuplexStream, ReadHalf, WriteHalf};

    #[test]
    fn framing() {
        let mut l = Lines::default();
        assert!(l.push(b"{\"a\":").is_empty());
        assert_eq!(
            l.push(b"1}\n{\"b\":2}\r\n\n  \n{\"c\""),
            [b"{\"a\":1}\n".to_vec(), b"{\"b\":2}\r\n".to_vec()]
        );
        assert_eq!(l.push(b":3}\n"), [b"{\"c\":3}\n".to_vec()]);
        // too long: dropped to its end, the next one whole
        let big = vec![b'x'; MAX];
        assert!(l.push(&big).is_empty());
        assert!(l.push(b"xx").is_empty());
        assert_eq!(l.push(b"\n{}\n"), [b"{}\n".to_vec()]);
        assert_eq!(
            l.push(&[&big[..MAX - 1], b"\n"].concat()).len(),
            1,
            "MAX with its newline fits"
        );
        assert_eq!(
            msg(&json!({"jsonrpc": "2.0", "id": 0, "method": "ping"})),
            Msg::Request {
                id: json!(0),
                method: "ping".into()
            }
        );
        assert_eq!(
            msg(&json!({"method": "notifications/initialized"})),
            Msg::Note("notifications/initialized".into())
        );
        assert_eq!(
            msg(&json!({"id": "a", "result": {}})),
            Msg::Answer(json!("a"))
        );
        assert_eq!(msg(&json!({"id": null, "error": {}})), Msg::Other);
    }

    /// Hosts the test plays: each attach hands out the next link.
    struct Fake(VecDeque<Link>);

    impl Attach for Fake {
        async fn attach(&mut self) -> Result<Link, String> {
            self.0.pop_front().ok_or_else(|| "no host".to_string())
        }
        async fn end(&mut self) {}
    }

    /// A host's end of a link: its lines in, its writer.
    type Far = (
        tokio::io::Lines<BufReader<ReadHalf<DuplexStream>>>,
        WriteHalf<DuplexStream>,
    );

    fn link() -> (Link, Far) {
        let (a, b) = tokio::io::duplex(1 << 16);
        let (r, w) = tokio::io::split(a);
        let (hr, hw) = tokio::io::split(b);
        let l = Link {
            r: Box::new(r),
            w: Box::new(w),
            name: "test host".into(),
        };
        (l, (BufReader::new(hr).lines(), hw))
    }

    async fn next(f: &mut Far) -> Value {
        let l = tokio::time::timeout(Duration::from_secs(5), f.0.next_line())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        serde_json::from_str(&l).unwrap()
    }

    async fn say(f: &mut Far, v: Value) {
        f.1.write_all(&line(&v)).await.unwrap()
    }

    /// The handshake passed on, a request the host never answered given an error, the handshake replayed to the
    /// next host and its answer swallowed, calls `--tiers` leaves out answered here, a tool list narrowed.
    #[tokio::test(flavor = "multi_thread")]
    async fn replay() {
        let ((l1, mut h1), (l2, mut h2)) = (link(), link());
        let narrow = Tiers::parse("read").ok();
        let mut f = Front::new(Fake(VecDeque::from([l1, l2])), narrow);
        let (itx, mut irx) = mpsc::unbounded_channel();
        let (otx, orx) = std::sync::mpsc::channel::<Vec<u8>>();
        let stop = CancellationToken::new();
        let s2 = stop.clone();
        let t = tokio::spawn(async move {
            let r = f.run(&mut irx, &otx, &s2).await;
            (r, f.replays)
        });
        let init = json!({
            "jsonrpc": "2.0", "id": 0, "method": "initialize", "params": { "protocolVersion": "2025-11-25" },
        });
        let send = |v: Value| itx.send(line(&v)).unwrap();
        let got = || -> Value {
            let b = orx.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(b.last(), Some(&b'\n'));
            serde_json::from_slice(&b).unwrap()
        };
        send(init.clone());
        assert_eq!(next(&mut h1).await, init);
        say(
            &mut h1,
            json!({"jsonrpc": "2.0", "id": 0, "result": {"protocolVersion": "2025-11-25"}}),
        )
        .await;
        assert_eq!(got()["result"]["protocolVersion"], "2025-11-25");
        send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        assert_eq!(next(&mut h1).await["method"], "notifications/initialized");
        // left out by --tiers: answered here, never sent
        send(
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "album_download"}}),
        );
        let r = got();
        assert_eq!(
            (r["id"].clone(), r["result"]["isError"].clone()),
            (json!(1), json!(true))
        );
        assert!(
            r["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("local tier"),
            "{r}"
        );
        send(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}));
        assert_eq!(next(&mut h1).await["id"], 2);
        let tools = json!([{ "name": "status" }, { "name": "job_cancel" }]);
        say(
            &mut h1,
            json!({ "jsonrpc": "2.0", "id": 2, "result": { "tools": tools } }),
        )
        .await;
        assert_eq!(got()["result"]["tools"], json!([{"name": "status"}]));
        // in flight when the host goes
        send(
            json!({"jsonrpc": "2.0", "id": "w", "method": "tools/call", "params": {"name": "job_wait"}}),
        );
        assert_eq!(next(&mut h1).await["id"], "w");
        drop(h1);
        let e = got();
        assert_eq!(
            (e["id"].clone(), e["error"]["code"].clone()),
            (json!("w"), json!(-32603))
        );
        assert_eq!(e["error"]["message"], ENDED);
        // the next host gets the handshake under the front's id; its answer never reaches the client
        let r = next(&mut h2).await;
        assert_eq!(
            (r["method"].clone(), r["id"].clone()),
            (json!("initialize"), json!("x5desk-replay-1"))
        );
        assert_eq!(r["params"], init["params"]);
        assert_eq!(next(&mut h2).await["method"], "notifications/initialized");
        say(
            &mut h2,
            json!({"jsonrpc": "2.0", "id": "x5desk-replay-1", "result": {}}),
        )
        .await;
        send(
            json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "status"}}),
        );
        assert_eq!(next(&mut h2).await["id"], 3);
        // split across writes
        h2.1.write_all(b"{\"jsonrpc\":\"2.0\",\"id\":3,")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        h2.1.write_all(b"\"result\":{\"ok\":1}}\n").await.unwrap();
        let r = got();
        assert_eq!(
            (r["id"].clone(), r["result"]["ok"].clone()),
            (json!(3), json!(1)),
            "not the replay's answer"
        );
        stop.cancel();
        let (r, n) = t.await.unwrap();
        assert_eq!((r, n), (Ok(()), 1));
        assert!(orx.try_recv().is_err(), "nothing more");
    }

    /// Under `--tiers` a BOM gets no call past the front, a batch's requests are answered here, a line that is no
    /// JSON is dropped, a call of a tool the front does not know (or of none) is refused, none of them sent; a
    /// BOM-prefixed `tools/list` is passed on without it and its answer narrowed.
    #[tokio::test(flavor = "multi_thread")]
    async fn narrowed() {
        let (l1, mut h1) = link();
        let mut f = Front::new(Fake(VecDeque::from([l1])), Tiers::parse("read").ok());
        let (itx, mut irx) = mpsc::unbounded_channel();
        let (otx, orx) = std::sync::mpsc::channel::<Vec<u8>>();
        let stop = CancellationToken::new();
        let s2 = stop.clone();
        let t = tokio::spawn(async move { f.run(&mut irx, &otx, &s2).await });
        let got = || -> Value {
            serde_json::from_slice(&orx.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap()
        };
        let call = |id: u64, name: Value| {
            line(
                &json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": name}}),
            )
        };
        itx.send([BOM, &call(1, json!("camera_connect"))].concat())
            .unwrap();
        let r = got();
        assert_eq!(
            (&r["id"], &r["result"]["isError"]),
            (&json!(1), &json!(true)),
            "{r}"
        );
        assert!(
            r["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("camera tier"),
            "{r}"
        );
        let batch = json!([
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "camera_connect"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ]);
        itx.send(line(&batch)).unwrap();
        let r = got();
        assert_eq!(r.as_array().map(Vec::len), Some(1), "{r}");
        assert_eq!(
            (&r[0]["id"], &r[0]["error"]["code"]),
            (&json!(2), &json!(-32600)),
            "{r}"
        );
        itx.send(b"{\"jsonrpc\": \"2.0\", \"id\": 9, nope\n".to_vec())
            .unwrap();
        itx.send(call(3, json!("album_delete"))).unwrap();
        let r = got();
        assert_eq!(
            (&r["id"], &r["error"]["code"]),
            (&json!(3), &json!(-32602)),
            "{r}"
        );
        assert!(
            r["error"]["message"]
                .as_str()
                .unwrap()
                .contains("unknown tool"),
            "{r}"
        );
        itx.send(call(4, json!(7))).unwrap();
        assert_eq!(got()["error"]["code"], -32602);
        itx.send(line(&json!({"jsonrpc": "2.0", "id": 5, "method": "ping"})))
            .unwrap();
        assert_eq!(
            next(&mut h1).await["id"],
            5,
            "nothing before it reached the host"
        );
        itx.send(
            [
                BOM,
                &line(&json!({"jsonrpc": "2.0", "id": 6, "method": "tools/list"})),
            ]
            .concat(),
        )
        .unwrap();
        assert_eq!(next(&mut h1).await["id"], 6);
        let tools = json!([{ "name": "status" }, { "name": "camera_connect" }]);
        say(
            &mut h1,
            json!({ "jsonrpc": "2.0", "id": 6, "result": { "tools": tools } }),
        )
        .await;
        assert_eq!(got()["result"]["tools"], json!([{"name": "status"}]));
        stop.cancel();
        assert_eq!(t.await.unwrap(), Ok(()));
        assert!(orx.try_recv().is_err(), "nothing more");
    }

    /// At stdin's end the host sees the end too and its answers to the open requests still come through (a BOM cut
    /// off without `--tiers` as well); a host that goes meanwhile ends the front without a new election.
    #[tokio::test(flavor = "multi_thread")]
    async fn drain() {
        let (l1, mut h1) = link();
        let mut f = Front::new(Fake(VecDeque::from([l1])), None);
        let (itx, mut irx) = mpsc::unbounded_channel();
        let (otx, orx) = std::sync::mpsc::channel::<Vec<u8>>();
        let stop = CancellationToken::new();
        let t = tokio::spawn(async move { f.run(&mut irx, &otx, &stop).await });
        let ping = json!({"jsonrpc": "2.0", "id": 1, "method": "ping"});
        itx.send([BOM, &line(&ping)].concat()).unwrap();
        itx.send(line(
            &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        ))
        .unwrap();
        drop(itx);
        assert_eq!(next(&mut h1).await, ping, "the BOM cut off");
        assert_eq!(next(&mut h1).await["id"], 2);
        let end = tokio::time::timeout(Duration::from_secs(5), h1.0.next_line())
            .await
            .unwrap();
        assert_eq!(end.unwrap(), None, "the host sees stdin's end");
        say(&mut h1, json!({"jsonrpc": "2.0", "id": 1, "result": {}})).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!t.is_finished(), "a request still open");
        say(
            &mut h1,
            json!({"jsonrpc": "2.0", "id": 2, "result": {"tools": []}}),
        )
        .await;
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), t)
                .await
                .unwrap()
                .unwrap(),
            Ok(())
        );
        let ids: Vec<Value> = orx
            .try_iter()
            .map(|b| serde_json::from_slice::<Value>(&b).unwrap()["id"].clone())
            .collect();
        assert_eq!(ids, [json!(1), json!(2)]);

        let ((l1, mut h1), (l2, _h2)) = (link(), link());
        let mut f = Front::new(Fake(VecDeque::from([l1, l2])), None);
        let (itx, mut irx) = mpsc::unbounded_channel();
        let (otx, orx) = std::sync::mpsc::channel::<Vec<u8>>();
        let stop = CancellationToken::new();
        let t = tokio::spawn(async move {
            let r = f.run(&mut irx, &otx, &stop).await;
            (r, f.at.0.len())
        });
        itx.send(line(&json!({"jsonrpc": "2.0", "id": 3, "method": "ping"})))
            .unwrap();
        drop(itx);
        assert_eq!(next(&mut h1).await["id"], 3);
        drop(h1);
        let r = tokio::time::timeout(Duration::from_secs(5), t)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(r, (Ok(()), 1), "no new election");
        assert!(orx.try_recv().is_err());
    }

    /// A request the client cancelled is not waited for at stdin's end (rmcp drops an answer not sent yet); the cancel
    /// is passed on, and an answer the host sent all the same still reaches the client, a tool list narrowed.
    #[tokio::test(flavor = "multi_thread")]
    async fn cancelled() {
        let (l1, mut h1) = link();
        let mut f = Front::new(
            Fake(VecDeque::from([l1])),
            Some(Tiers::parse("read").unwrap()),
        );
        let (itx, mut irx) = mpsc::unbounded_channel();
        let (otx, orx) = std::sync::mpsc::channel::<Vec<u8>>();
        let stop = CancellationToken::new();
        let t = tokio::spawn(async move { f.run(&mut irx, &otx, &stop).await });
        let wait = |id: Value| {
            let p = json!({ "name": "job_wait" });
            json!({ "jsonrpc": "2.0", "id": id, "method": "tools/call", "params": p })
        };
        let cancel = |id: Value| {
            let p = json!({ "requestId": id });
            json!({ "jsonrpc": "2.0", "method": "notifications/cancelled", "params": p })
        };
        for v in [wait(json!(1)), wait(json!("w")), cancel(json!(1))] {
            itx.send(line(&v)).unwrap();
        }
        for id in [json!(1), json!("w")] {
            assert_eq!(next(&mut h1).await["id"], id);
        }
        assert_eq!(next(&mut h1).await, cancel(json!(1)), "passed on");
        // answered all the same (sent before the cancel arrived)
        say(&mut h1, json!({"jsonrpc": "2.0", "id": 1, "result": {}})).await;
        assert_eq!(
            serde_json::from_slice::<Value>(&orx.recv_timeout(Duration::from_secs(5)).unwrap())
                .unwrap()["id"],
            1
        );
        itx.send(line(&cancel(json!("w")))).unwrap();
        assert_eq!(next(&mut h1).await, cancel(json!("w")));
        // a tool list cancelled after the host answered it: still narrowed
        let list = json!({ "jsonrpc": "2.0", "id": 2, "method": "tools/list" });
        itx.send(line(&list)).unwrap();
        assert_eq!(next(&mut h1).await, list);
        itx.send(line(&cancel(json!(2)))).unwrap();
        assert_eq!(next(&mut h1).await, cancel(json!(2)));
        let all: Vec<Value> = TOOLS.iter().map(|d| json!({ "name": d.name })).collect();
        say(
            &mut h1,
            json!({"jsonrpc": "2.0", "id": 2, "result": { "tools": all }}),
        )
        .await;
        let v: Value =
            serde_json::from_slice(&orx.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap();
        let read = |n: &Value| {
            TOOLS
                .iter()
                .any(|d| *n == d.name && d.tier == crate::Tier::Read)
        };
        let names: Vec<&Value> = v["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| &x["name"])
            .collect();
        assert!(!names.is_empty() && names.iter().all(|n| read(n)), "{v}");
        drop(itx);
        let t0 = std::time::Instant::now();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), t)
                .await
                .unwrap()
                .unwrap(),
            Ok(())
        );
        assert!(t0.elapsed() < DRAIN / 2, "not drained: {:?}", t0.elapsed());
        assert!(orx.try_recv().is_err());
    }
}
