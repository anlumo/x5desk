//! Loopback: one server, two sources, four clients; memfds stand in for the DMA-BUFs and the
//! sync_file.
#![cfg(target_os = "linux")]

use live::obs::*;
use std::{
    fs::File,
    io::{ErrorKind, Write},
    os::{
        fd::{AsFd, FromRawFd, OwnedFd},
        unix::fs::{FileExt, MetadataExt},
    },
    time::{Duration, Instant},
};

fn memfd(data: &[u8]) -> File {
    let fd = unsafe { libc::memfd_create(c"plane".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0);
    let mut f = File::from(unsafe { OwnedFd::from_raw_fd(fd) });
    f.write_all(data).unwrap();
    f
}

fn open_fds() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}

const W: Duration = Duration::from_millis(500);

/// Removes the test's socket directory when dropped, also when the test fails.
struct Rm(std::path::PathBuf);

impl Drop for Rm {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn loopback() {
    let base = open_fds();
    let dir = std::env::var_os("XDG_RUNTIME_DIR").unwrap_or("/tmp".into());
    let d = Rm(std::path::Path::new(&dir).join(format!("x5desk-test-{}", std::process::id())));
    let p = d.0.join("obs.sock");
    let mut s = Server::bind(&p).unwrap();
    assert_eq!(
        Server::bind(&p).err().map(|e| e.kind()),
        Some(ErrorKind::AddrInUse)
    );
    s.add(1, "360");
    s.add(2, "flat");
    let fmt = Format {
        w: 2880,
        h: 1440,
        fourcc: fourcc(*b"NV12"),
        fps: [30, 1],
        color: BT601 | FULL,
        rate: 48000,
        ch: 2,
        ..Default::default()
    };
    s.format(1, fmt);

    let mut a = Client::connect(&p, 1, &[]).unwrap();
    let mut b = Client::connect(&p, 1, &[]).unwrap();
    let mut c = Client::connect(&p, 2, &[]).unwrap();
    let mut d = Client::connect(&p, 0, &[]).unwrap();
    s.poll();
    let list = vec![(1, "360".to_string()), (2, "flat".to_string())];
    for x in [&mut a, &mut b, &mut c, &mut d] {
        assert!(matches!(x.recv(Some(W)).unwrap(), Msg::Hello(v) if v == list));
    }
    for x in [&mut a, &mut b] {
        assert!(matches!(x.recv(Some(W)).unwrap(), Msg::Format(1, f) if f == fmt));
    }
    assert!(s.watched(1) && s.watched(2) && !s.watched(3));

    // NV12 in one buffer at two offsets, as Vulkan exports it, plus a sync_file stand-in
    let y: Vec<u8> = (0..4096u32).map(|i| (i * 7) as u8).collect();
    let (buf, sync) = (memfd(&y), memfd(b"sync"));
    let f = Frame {
        ts: 123_456_789,
        w: 64,
        h: 32,
        fourcc: fmt.fourcc,
        buf: 2,
        off: [0, 2048, 0, 0],
        stride: [64, 64, 0, 0],
        ..Default::default()
    };
    assert_eq!(
        s.frame(1, &f, &[buf.as_fd(), buf.as_fd()], Some(sync.as_fd())),
        2
    );
    for x in [&mut a, &mut b] {
        let Msg::Frame {
            src: 1,
            f: g,
            planes,
            sync: Some(sy),
        } = x.recv(Some(W)).unwrap()
        else {
            panic!("no frame")
        };
        assert_eq!(
            g,
            Frame {
                planes: 2,
                sync: 1,
                ..f
            }
        );
        for pl in planes.iter().take(2) {
            let pl = File::from(pl.as_ref().unwrap().try_clone().unwrap());
            assert_eq!(pl.metadata().unwrap().ino(), buf.metadata().unwrap().ino());
            let mut r = vec![0; 2048];
            pl.read_exact_at(&mut r, g.off[1] as u64).unwrap();
            assert_eq!(r, y[2048..]);
        }
        assert!(planes[2].is_none());
        assert_eq!(
            File::from(sy).metadata().unwrap().ino(),
            sync.metadata().unwrap().ino()
        );
    }
    // both hold buffer 2 until they hand it back
    assert!(s.busy(1, 2) && !s.busy(1, 1) && !s.busy(2, 2));
    // re-selecting the source it has keeps what a holds
    a.select(1).unwrap();
    b.done(1, 2).unwrap();
    s.poll();
    assert!(matches!(a.recv(Some(W)).unwrap(), Msg::Hello(_)));
    assert!(matches!(a.recv(Some(W)).unwrap(), Msg::Format(1, _)));
    assert!(s.busy(1, 2));
    a.done(1, 2).unwrap();
    s.poll();
    assert!(!s.busy(1, 2));
    // source 2: one plane, no sync; only c gets it
    assert_eq!(s.frame(2, &Frame { ts: 5, ..f }, &[buf.as_fd()], None), 1);
    assert!(matches!(
        c.recv(Some(W)).unwrap(),
        Msg::Frame {
            src: 2,
            f: Frame {
                planes: 1,
                sync: 0,
                ts: 5,
                ..
            },
            sync: None,
            ..
        }
    ));
    for x in [&mut a, &mut b, &mut d] {
        assert_eq!(
            x.recv(Some(Duration::from_millis(20)))
                .err()
                .map(|e| e.kind()),
            Some(ErrorKind::TimedOut)
        );
    }
    // a timeout under a millisecond still waits
    let t = Instant::now();
    assert!(d.recv(Some(Duration::from_micros(500))).is_err());
    assert!(
        t.elapsed() >= Duration::from_micros(500),
        "{:?}",
        t.elapsed()
    );

    // 0.25 s of stereo (96 KB) is split into several messages with continuous timestamps
    let pcm: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.001).sin()).collect();
    // one sample per channel wouldn't fit a message
    assert_eq!(s.audio(1, 0, 48000, 20_000, &pcm), 0);
    assert_eq!(s.audio(1, 1_000_000_000, 48000, 2, &pcm), 2);
    let (mut got, mut parts) = (vec![], 0);
    while got.len() < pcm.len() {
        let Msg::Audio {
            src: 1,
            ts,
            rate: 48000,
            ch: 2,
            pcm: p,
        } = a.recv(Some(W)).unwrap()
        else {
            panic!("no audio")
        };
        assert_eq!(
            ts,
            1_000_000_000 + got.len() as u64 / 2 * 1_000_000_000 / 48000
        );
        got.extend_from_slice(p);
        parts += 1;
    }
    assert_eq!(got, pcm);
    assert!(parts > 1);
    println!("audio: {} samples in {parts} messages", pcm.len());

    // a client leaves; another switches source; a source ends
    drop(b);
    c.select(1).unwrap();
    s.poll();
    assert!(matches!(c.recv(Some(W)).unwrap(), Msg::Hello(_)));
    assert!(matches!(c.recv(Some(W)).unwrap(), Msg::Format(1, _)));
    // selecting another source drops what c held of source 2
    assert!(!s.watched(2) && !s.busy(2, 2));
    assert_eq!(s.frame(1, &f, &[buf.as_fd()], None), 2);
    s.add(3, "lens");
    d.select(3).unwrap();
    s.poll();
    assert!(s.watched(3));
    s.remove(3);
    assert!(!s.watched(3));
    for _ in 0..2 {
        assert!(matches!(d.recv(Some(W)).unwrap(), Msg::Hello(v) if v.len() == 3));
    }
    assert!(matches!(d.recv(Some(W)).unwrap(), Msg::Bye(3)));
    assert!(matches!(d.recv(Some(W)).unwrap(), Msg::Hello(v) if v == list));
    drop(s);
    assert!(!p.exists());
    for x in [&mut a, &mut c] {
        while !matches!(x.recv(Some(W)).unwrap(), Msg::Bye(0)) {}
        assert_eq!(
            x.recv(Some(W)).err().map(|e| e.kind()),
            Some(ErrorKind::UnexpectedEof)
        );
    }
    assert!(matches!(d.recv(Some(W)).unwrap(), Msg::Bye(0)));
    drop((a, c, d, buf, sync));
    std::fs::remove_dir(p.parent().unwrap()).unwrap();
    assert_eq!(open_fds(), base, "fd leak");
}
