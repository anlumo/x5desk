//! `--sim`: the host on cam's simulated X5 (`cam::sim`: link's simulated camera on a fake card, its real session,
//! framing and tunnel) instead of USB, and a fake mounted card, all under a temp dir of its own (agent folder, card,
//! log, the kept queue and jobs) that goes when the host does; tests across hosts give a dir that stays. Nothing of the
//! user's config, cache, videos or USB is touched. The scrub tests look for two secrets in everything a client receives
//! and in the log: [`MARKER`], which the host knows from its start (they pass it in arguments before a connect), and
//! the camera's access-point password [`AP`], which it learns only from the camera, as a real host does.

use cam::{
    camera::Connect,
    sim::{self, Fake},
};
use link::Camera;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering::Relaxed},
    },
};

/// A secret a `--sim` host knows from its start (as if learnt): it must never reach a client or the log.
pub const MARKER: &str = "x5sim-AP-pw-4f7c1e9a";

/// The simulated camera's access-point password: a secret the host learns only from the camera's options, which must
/// never reach a client or the log either.
pub const AP: &str = "x5sim-camera-only-pw-b2d9e5a7";

/// The Wi-Fi client id under `--sim` (a secret as the real one is).
pub const ID: &str = "00000000-0000-4000-8000-00000000c11d";

/// The fake card's files, under `DCIM/Camera01`.
pub const CARD: [&str; 3] = [
    "VID_20260201_101010_00_700.insv",
    "LRV_20260201_101010_01_700.lrv",
    "IMG_20260201_111111_00_701.jpg",
];

/// How the simulated camera serves its files.
#[derive(Clone, Debug)]
pub struct Opts {
    /// bytes/s per tunnel connection (a download uses 4)
    pub rate: u64,
    /// bytes of each video and proxy file
    pub size: usize,
    /// the dir to use instead of a temp dir of its own: kept as it is, at the start and at the end, so a later host
    /// finds the agent folder, the queue and the jobs there (tests across hosts)
    pub dir: Option<PathBuf>,
}

impl Default for Opts {
    fn default() -> Opts {
        Opts {
            rate: 8_000_000,
            size: 1_000_000,
            dir: None,
        }
    }
}

/// The simulated camera and the temp dirs of a `--sim` host.
pub struct Sim {
    /// cam's fake card behind the simulated camera: its files, bytes, requests
    pub fake: Fake,
    /// bytes/s per tunnel connection
    pub rate: u64,
    /// the temp dir everything below is in
    pub dir: PathBuf,
    /// the fake card's mount point (a `DCIM` inside)
    pub card: PathBuf,
    /// the agent folder
    pub agent: PathBuf,
    /// the call log
    pub log: PathBuf,
    /// another process holds the camera's USB interface: a connect fails as nusb's does on macOS
    pub held: Arc<AtomicBool>,
    /// the dir is the caller's ([`Opts::dir`]): it stays
    keep: bool,
}

/// Bytes of a file that tell files and offsets apart.
pub fn bytes(name: &str, n: usize) -> Vec<u8> {
    let s = name
        .bytes()
        .fold(7u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
    (0..n as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(s) >> 13) as u8)
        .collect()
}

impl Sim {
    /// Makes the temp dirs, the fake card and the simulated camera's files.
    pub fn new(o: &Opts) -> std::io::Result<Sim> {
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = o.dir.clone().unwrap_or_else(|| {
            std::env::temp_dir().join(format!(
                "x5desk-mcp-sim-{}-{}",
                std::process::id(),
                N.fetch_add(1, Relaxed)
            ))
        });
        if o.dir.is_none() {
            let _ = std::fs::remove_dir_all(&dir);
        }
        let (card, agent) = (dir.join("card"), dir.join("agent"));
        let dcim = card.join("DCIM").join("Camera01");
        std::fs::create_dir_all(&dcim)?;
        for n in CARD {
            std::fs::write(dcim.join(n), bytes(n, 50_000))?;
        }
        let fake = sim::card(&sim::NAMES);
        {
            let mut c = fake.lock().unwrap();
            c.ap = Some(("X5 SIM001.OSC".into(), AP.into()));
            sim::capture(&mut c);
            for u in c.uris.clone() {
                let n = cam::album::name(&u);
                let k = match n.rsplit_once('.').map(|e| e.1) {
                    Some("insv" | "lrv") => o.size,
                    _ => 60_000,
                };
                c.data.insert(u.clone(), Arc::new(bytes(n, k)));
            }
        }
        Ok(Sim {
            fake,
            rate: o.rate,
            log: dir.join("mcp.log"),
            dir,
            card,
            agent,
            held: Arc::default(),
            keep: o.dir.is_some(),
        })
    }

    /// The worker's connections: to the fake card at the rate.
    pub fn connect(&self) -> Connect {
        let (f, rate, held) = (self.fake.clone(), self.rate, self.held.clone());
        Box::new(move || {
            if held.load(Relaxed) {
                let e = std::io::Error::other(
                    "could not open interface for exclusive access (0xe00002c5)",
                );
                return Err(link::Error::Io(e)).into();
            }
            let ff = f.clone();
            let files = move |u: &str| {
                let mut c = ff.lock().unwrap();
                *c.opens.entry(u.into()).or_default() += 1;
                c.data.get(u).cloned()
            };
            // the session's control kept: capture's notifications go through it
            Camera::sim(sim::handler(f.clone()), files, rate)
                .map(|(c, ev, s)| {
                    f.lock().unwrap().link = Some(s);
                    (c, ev)
                })
                .into()
        })
    }
}

impl Drop for Sim {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}
