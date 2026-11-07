//! The one GPU setup of a process, the window's and the headless MCP server's: the device (`video::Gpu`), burn's
//! cache dir before burn attaches (else cubecl keeps it under its Target root: an MCP client's project's `target/`),
//! burn attached, the device's error policy, the export threads' locks. The window panics on an uncaptured error, as
//! wgpu does (`crate::errors`: a bug); a headless host logs and counts it instead ([`Policy::Log`]): the handler runs
//! on whatever thread made the failing call, a lane's or a tokio worker's, in the process that holds the camera. wgpu
//! hands a lost device to no error handler, and cubecl only logs it (its device-lost callback stays: replacing it
//! blinds burn), so a headless host also installs a `log` logger, which notes cubecl's loss line ([`Watch`]).

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    sync::{
        Arc, Mutex, OnceLock, PoisonError,
        atomic::{AtomicBool, AtomicU32, Ordering::Relaxed},
    },
    time::{Duration, Instant},
};

/// What `open(.., attach: false)` says instead of attaching.
pub const NOATTACH: &str = "attach turned off (X5DESK_TEST=mcp-noattach)";
/// Log lines a minute at most (warn and up).
const LINES: u32 = 20;

/// What a headless device reports to: its log lines (the newest host's sink), the uncaptured errors so far, whether a
/// probe is due (an out-of-memory or internal error outside a scope, cubecl's loss line).
pub struct Watch {
    pub sink: Mutex<Arc<dyn Fn(String) + Send + Sync>>,
    pub errors: AtomicU32,
    pub suspect: AtomicBool,
}

impl Default for Watch {
    fn default() -> Watch {
        Watch {
            sink: Mutex::new(Arc::new(|_| {})),
            errors: AtomicU32::new(0),
            suspect: AtomicBool::new(false),
        }
    }
}

impl Watch {
    /// `l` into the sink, with no lock held while it writes.
    pub fn line(&self, l: String) {
        let f = self
            .sink
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        f(l)
    }
}

/// The device's error policy: the window's panic, or a headless host's log.
pub enum Policy {
    Panic,
    Log(Arc<Watch>),
}

/// Once per process: the device; `cache` for burn (None: memory only); burn attached (`attach`; a headless host's
/// panic in it is an error, the window's ends the app), first, as cubecl installs its own error handler then; the
/// error policy; `GATE` and `BURN` for the export threads. Err: no device. Ok: it, and whether burn attached (RAFT
/// off when not).
pub fn open(
    cache: Option<&Path>,
    p: Policy,
    attach: bool,
) -> Result<(Arc<video::Gpu>, Result<(), String>), String> {
    let g = Arc::new(video::Gpu::new().map_err(|e| e.to_string())?);
    stitch::cache(cache);
    let at = || stitch::attach(&g.instance, &g.adapter, &g.device, &g.queue).map(drop);
    let a = match (attach, &p) {
        (false, _) => Err(NOATTACH.into()),
        (true, Policy::Panic) => at(),
        (true, Policy::Log(_)) => catch_unwind(AssertUnwindSafe(at))
            .map_err(|_| "burn's attach panicked".to_string())
            .and_then(|r| r),
    };
    match p {
        Policy::Panic => crate::errors(&g.device),
        Policy::Log(w) => logged(&g.device, w),
    }
    ::export::locks(&crate::GATE, &crate::BURN);
    Ok((g, a))
}

/// A headless device's policy: each uncaptured error logged and counted, out-of-memory and internal ones make the
/// GPU suspect; never a panic, never a probe on the failing call's thread. And the `log` logger.
fn logged(dev: &wgpu::Device, w: Arc<Watch>) {
    let v = w.clone();
    dev.on_uncaptured_error(Arc::new(move |e| {
        v.errors.fetch_add(1, Relaxed);
        if matches!(
            e,
            wgpu::Error::OutOfMemory { .. } | wgpu::Error::Internal { .. }
        ) {
            v.suspect.store(true, Relaxed);
        }
        v.line(format!("uncaptured wgpu error: {e}"))
    }));
    if WATCH.set(w).is_ok() && log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(log::LevelFilter::Warn)
    }
}

/// The headless logger's watch.
static WATCH: OnceLock<Arc<Watch>> = OnceLock::new();
static LOGGER: Logger = Logger {
    rate: Mutex::new(None),
};

/// Warn and up into the watch's sink, [`LINES`] a minute at most (the lines past them counted in the next minute's
/// first); cubecl's "the wgpu device was lost" makes the GPU suspect.
struct Logger {
    /// this minute's start, its lines, the lines dropped
    rate: Mutex<Option<(Instant, u32, u32)>>,
}

impl log::Log for Logger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Warn
    }

    fn log(&self, r: &log::Record) {
        let Some(w) = WATCH.get().filter(|_| self.enabled(r.metadata())) else {
            return;
        };
        let m = r.args().to_string();
        if m.contains("device was lost") {
            w.suspect.store(true, Relaxed);
        }
        let dropped = {
            let mut g = self.rate.lock().unwrap_or_else(PoisonError::into_inner);
            let now = Instant::now();
            let (t, n, d) = g.get_or_insert((now, 0, 0));
            if now.duration_since(*t) >= Duration::from_secs(60) {
                (*t, *n) = (now, 0);
            }
            if *n >= LINES {
                *d += 1;
                return;
            }
            *n += 1;
            std::mem::take(d)
        };
        let more = match dropped {
            0 => String::new(),
            d => format!(" ({d} lines dropped before)"),
        };
        w.line(format!("{} {}: {m}{more}", r.level(), r.target()))
    }

    fn flush(&self) {}
}
