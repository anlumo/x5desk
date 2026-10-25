//! Clients that stop reading: each holds at most `HOLD` frames, stays connected, and resumes with
//! the newest frame, also when this user's fds in flight exceed `RLIMIT_NOFILE` (`ETOOMANYREFS`).
//! Neither a full listen backlog nor an app that stopped reading blocks a client. Its own binary:
//! it lowers the process's fd limit.
#![cfg(target_os = "linux")]

use live::obs::*;
use std::{
    fs::File,
    io::{ErrorKind, Write},
    os::fd::{AsFd, FromRawFd, OwnedFd},
    time::Duration,
};

fn memfd(n: usize) -> File {
    let fd = unsafe { libc::memfd_create(c"plane".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0);
    let mut f = File::from(unsafe { OwnedFd::from_raw_fd(fd) });
    f.write_all(&vec![0; n]).unwrap();
    f
}

fn open_fds() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}

/// The frames waiting for `c`, each handed back; their timestamps.
fn drain(c: &mut Client) -> Vec<u64> {
    let mut v = vec![];
    loop {
        match c.recv(Some(Duration::from_millis(20))) {
            Ok(Msg::Frame { src, f, .. }) => {
                c.done(src, f.buf).unwrap();
                v.push(f.ts);
            }
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::TimedOut => return v,
            Err(e) => panic!("{e}"),
        }
    }
}

/// Removes the test's socket directory when dropped, also when the test fails.
struct Rm(std::path::PathBuf);

impl Drop for Rm {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn frame(i: u64) -> Frame {
    Frame {
        ts: i,
        w: 64,
        h: 32,
        buf: (i % 8) as u32,
        ..Default::default()
    }
}

#[test]
fn stall() {
    let base = open_fds();
    let dir = std::env::var_os("XDG_RUNTIME_DIR").unwrap_or("/tmp".into());
    let d = Rm(std::path::Path::new(&dir).join(format!("x5desk-stall-{}", std::process::id())));
    let p = d.0.join("obs.sock");
    let mut s = Server::bind(&p).unwrap();
    s.add(1, "360");
    s.add(2, "flat");
    let (buf, sync) = (memfd(4096), memfd(4));
    // 4 planes + sync: 5 fds per frame
    let pl = [buf.as_fd(); 4];

    // a keeps up, b never reads
    let mut a = Client::connect(&p, 1, &[]).unwrap();
    let mut b = Client::connect(&p, 1, &[]).unwrap();
    s.poll();
    let w = Some(Duration::from_millis(500));
    assert!(matches!(a.recv(w).unwrap(), Msg::Hello(_)));
    for i in 0..1000 {
        let n = s.frame(1, &frame(i), &pl, Some(sync.as_fd()));
        assert_eq!(n, if i < HOLD as u64 { 2 } else { 1 }, "frame {i}");
        let Msg::Frame { f, .. } = a.recv(w).unwrap() else {
            panic!("no frame {i}")
        };
        assert_eq!(f.ts, i);
        a.done(1, f.buf).unwrap();
        s.poll();
    }
    assert!(s.watched(1));
    assert!((0..HOLD as u32).all(|k| s.busy(1, k)) && !s.busy(1, HOLD as u32));
    // b gets only what it held, then the newest frame once it hands them back
    assert_eq!(drain(&mut b), (0..HOLD as u64).collect::<Vec<_>>());
    s.poll();
    assert!(!s.busy(1, 0));
    assert_eq!(s.frame(1, &frame(1000), &pl, Some(sync.as_fd())), 2);
    assert_eq!(drain(&mut b), [1000]);
    assert_eq!(drain(&mut a), [1000]);
    println!(
        "stalled client: held {HOLD} frames, missed {}, resumed at 1000",
        1000 - HOLD
    );

    // 6 stalled clients could hold 6 × HOLD × 5 = 120 fds in flight; allow 64
    let mut c: Vec<_> = (0..6)
        .map(|_| Client::connect(&p, 2, &[]).unwrap())
        .collect();
    s.poll();
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    assert_eq!(unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) }, 0);
    let low = libc::rlimit {
        rlim_cur: 64,
        ..lim
    };
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &low) }, 0);
    let sent: usize = (0..100)
        .map(|i| s.frame(2, &frame(i), &pl, Some(sync.as_fd())))
        .sum();
    assert_eq!(unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &lim) }, 0);
    assert!(
        0 < sent && sent < 6 * HOLD,
        "the fd limit didn't bite: {sent} sent"
    );
    assert!(s.watched(2));
    let held: Vec<_> = c.iter_mut().map(|x| drain(x).len()).collect();
    assert_eq!(held.iter().sum::<usize>(), sent);
    s.poll();
    // nobody was dropped
    assert_eq!(s.frame(2, &frame(100), &pl, Some(sync.as_fd())), 6);
    assert!(c.iter_mut().all(|x| drain(x) == [100]));
    println!(
        "at RLIMIT_NOFILE 64: {sent} of 600 frames delivered, per client {held:?}; all 6 kept"
    );

    // an app that stops reading (its UI thread stalls): the client's sends fail instead of blocking, and go through
    // again once it reads
    let x = Client::connect(&p, 1, &[]).unwrap();
    s.poll();
    let t = std::time::Instant::now();
    let mut n = 0;
    let e = loop {
        match x.done(1, 7) {
            Ok(()) => n += 1,
            Err(e) => break e,
        }
        assert!(n < 100_000);
    };
    assert_eq!(e.kind(), ErrorKind::WouldBlock);
    assert_eq!(x.select(2).unwrap_err().kind(), ErrorKind::WouldBlock);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    assert!(ms < 1000.0);
    s.poll();
    x.select(2).unwrap();
    x.done(2, 7).unwrap();
    println!("stalled app: {n} dones queued, then WouldBlock after {ms:.1} ms");
    drop(x);
    s.poll();

    // a server that never accepts: the backlog fills and connecting fails instead of blocking
    let p2 = d.0.join("hung.sock");
    let s2 = Server::bind(&p2).unwrap();
    let mut q = vec![];
    let e = loop {
        match Client::connect(&p2, 0, &[]) {
            Ok(x) => q.push(x),
            Err(e) => break e,
        }
        assert!(q.len() < 100);
    };
    assert_eq!(e.kind(), ErrorKind::WouldBlock);
    assert_eq!(
        Server::bind(&p2).err().map(|e| e.kind()),
        Some(ErrorKind::AddrInUse)
    );
    println!("hung server: {} queued, then WouldBlock", q.len());

    drop((q, s2, s, a, b, c, buf, sync));
    std::fs::remove_dir(&d.0).unwrap();
    assert_eq!(open_fds(), base, "fd leak");
}
