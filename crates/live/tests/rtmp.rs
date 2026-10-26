//! FLV and RTMP against FFmpeg (a test tool only): H.264 + AAC from `v635.lrv` ($X5_SAMPLES) and an
//! x264 re-encode with B-frames, written as FLV and published to `ffmpeg -listen 1`, then checked
//! with ffprobe. Outputs go to $X5_OUT (default: a temp dir): the fixtures under fixed names, each
//! renamed into place whole (concurrent runs share them); what a listener received is named per
//! process and removed with it.

use live::{flv, rtmp};
use std::{
    env, fs,
    io::{BufWriter, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::OnceLock,
    thread,
    time::{Duration, Instant},
};

fn samples() -> Option<PathBuf> {
    let d = env::var_os("X5_SAMPLES").map(PathBuf::from);
    if d.is_none() {
        println!("X5_SAMPLES is not set; skipped");
    }
    d
}

fn have_ffmpeg() -> bool {
    let ok = ["ffmpeg", "ffprobe"].iter().all(|t| {
        Command::new(t)
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
    });
    if !ok {
        println!("ffmpeg / ffprobe not found; skipped");
    }
    ok
}

fn dir() -> PathBuf {
    let d =
        env::var_os("X5_OUT").map_or_else(|| env::temp_dir().join("x5desk-live"), PathBuf::from);
    fs::create_dir_all(&d).unwrap();
    d
}

/// A file of this process's: concurrent runs share the dir.
fn out(name: &str) -> PathBuf {
    dir().join(format!("{}-{name}", std::process::id()))
}

/// A file of this process's: dropping it removes it, also when the test fails.
struct Rm(PathBuf);

impl Drop for Rm {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Fixture `name`, shared by concurrent runs and made again by each: `w` writes it under a name of
/// this process's, and it is renamed into place whole, so no run reads it half written.
fn fixture(name: &str, w: impl FnOnce(&Path)) -> PathBuf {
    let (p, t) = (dir().join(name), Rm(out(name)));
    w(&t.0);
    fs::rename(&t.0, &p).unwrap();
    p
}

// ---- a tiny MP4 reader, enough for these files ----

fn be32(b: &[u8], i: usize) -> u32 {
    u32::from_be_bytes(b[i..i + 4].try_into().unwrap())
}

fn kids(b: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> {
    let mut i = 0;
    std::iter::from_fn(move || {
        if i + 8 > b.len() {
            return None;
        }
        let (mut n, mut h) = (be32(b, i) as usize, 8);
        if n == 1 {
            n = u64::from_be_bytes(b[i + 8..i + 16].try_into().unwrap()) as usize;
            h = 16;
        } else if n == 0 {
            n = b.len() - i;
        }
        let r = (&b[i + 4..i + 8], &b[i + h..i + n]);
        i += n;
        Some(r)
    })
}

fn kid<'a>(b: &'a [u8], t: &[u8]) -> Option<&'a [u8]> {
    kids(b).find(|k| k.0 == t).map(|k| k.1)
}

fn desc(b: &[u8]) -> &[u8] {
    let (mut i, mut n) = (1, 0);
    loop {
        n = n << 7 | (b[i] & 0x7F) as usize;
        i += 1;
        if b[i - 1] & 0x80 == 0 {
            break &b[i..i + n];
        }
    }
}

/// A sample in ms, interleaved by dts.
#[derive(Clone, Copy, Debug)]
struct P {
    video: bool,
    dts: i64,
    cts: i32,
    key: bool,
    off: usize,
    len: usize,
}

struct Src {
    f: Vec<u8>,
    avcc: Vec<u8>,
    asc: Vec<u8>,
    w: u32,
    h: u32,
    p: Vec<P>,
    /// The first dts in the file's presentation timeline; `p` starts at 0.
    t0: i64,
}

fn mp4(path: &Path) -> Src {
    let f = fs::read(path).unwrap();
    let mut s = Src {
        f: vec![],
        avcc: vec![],
        asc: vec![],
        w: 0,
        h: 0,
        p: vec![],
        t0: 0,
    };
    let moov = kid(&f, b"moov").unwrap();
    let mvhd = kid(moov, b"mvhd").unwrap();
    let mscale = be32(mvhd, if mvhd[0] == 1 { 20 } else { 12 }) as i64;
    for (_, trak) in kids(moov).filter(|k| k.0 == b"trak") {
        let mdia = kid(trak, b"mdia").unwrap();
        let video = &kid(mdia, b"hdlr").unwrap()[8..12] == b"vide";
        let mdhd = kid(mdia, b"mdhd").unwrap();
        let scale = be32(mdhd, if mdhd[0] == 1 { 20 } else { 12 }) as i64;
        let stbl = kid(kid(mdia, b"minf").unwrap(), b"stbl").unwrap();
        let tab = |t: &[u8], w: usize| -> Vec<u32> {
            kid(stbl, t).map_or(vec![], |b| {
                (0..be32(b, 4) as usize * w)
                    .map(|i| be32(b, 8 + 4 * i))
                    .collect()
            })
        };
        let e = &kid(stbl, b"stsd").unwrap()[8..];
        if video {
            let e = &e[8..];
            (s.w, s.h) = (
                u16::from_be_bytes([e[24], e[25]]) as u32,
                u16::from_be_bytes([e[26], e[27]]) as u32,
            );
            s.avcc = kid(&e[78..], b"avcC").unwrap().to_vec();
        } else {
            let es = desc(&kid(&e[8 + 28..], b"esds").unwrap()[4..]);
            let mut j = 3;
            j += if es[2] & 0x80 != 0 { 2 } else { 0 } + if es[2] & 0x20 != 0 { 2 } else { 0 };
            if es[2] & 0x40 != 0 {
                j += 1 + es[j] as usize;
            }
            s.asc = desc(&desc(&es[j..])[13..]).to_vec();
        }
        let z = kid(stbl, b"stsz").unwrap();
        let n = be32(z, 8) as usize;
        let size: Vec<u32> = (0..n)
            .map(|i| {
                if be32(z, 4) != 0 {
                    be32(z, 4)
                } else {
                    be32(z, 12 + 4 * i)
                }
            })
            .collect();
        let (stts, ctts, stsc, stss) = (
            tab(b"stts", 2),
            tab(b"ctts", 2),
            tab(b"stsc", 3),
            tab(b"stss", 1),
        );
        let co: Vec<u64> = match kid(stbl, b"co64") {
            Some(b) => (0..be32(b, 4) as usize)
                .map(|i| u64::from_be_bytes(b[8 + 8 * i..16 + 8 * i].try_into().unwrap()))
                .collect(),
            None => tab(b"stco", 1).into_iter().map(u64::from).collect(),
        };
        // the edit list in media ticks: empty edits delay the track, a media time skips into it
        let mut sh = 0;
        if let Some(el) = kid(trak, b"edts").and_then(|e| kid(e, b"elst")) {
            for i in 0..be32(el, 4) as usize {
                let (d, m) = if el[0] == 1 {
                    let o = 8 + 20 * i;
                    let x = |o: usize| i64::from_be_bytes(el[o..o + 8].try_into().unwrap());
                    (x(o), x(o + 8))
                } else {
                    let o = 8 + 12 * i;
                    (be32(el, o) as i64, be32(el, o + 4) as i32 as i64)
                };
                if m != -1 {
                    sh -= m;
                    break;
                }
                sh += d * scale / mscale;
            }
        }
        let (mut dts, mut ofs, mut t) = (vec![], vec![], sh);
        for e in stts.chunks(2) {
            for _ in 0..e[0] {
                dts.push(t);
                t += e[1] as i64;
            }
        }
        for e in ctts.chunks(2) {
            ofs.extend((0..e[0]).map(|_| e[1] as i32 as i64));
        }
        let mut k = 0;
        for (c, &o) in co.iter().enumerate() {
            let spc = stsc.chunks(3).rfind(|e| e[0] as usize <= c + 1).unwrap()[1];
            let mut o = o as usize;
            for _ in 0..spc {
                let ms = |t: i64| (t * 1000 + scale / 2).div_euclid(scale);
                let (d, p) = (ms(dts[k]), ms(dts[k] + ofs.get(k).copied().unwrap_or(0)));
                let key = !video || stss.is_empty() || stss.contains(&(k as u32 + 1));
                s.p.push(P {
                    video,
                    dts: d,
                    cts: (p - d) as i32,
                    key,
                    off: o,
                    len: size[k] as usize,
                });
                o += size[k] as usize;
                k += 1;
            }
        }
    }
    s.p.sort_by_key(|p| (p.dts, !p.video));
    s.t0 = s.p[0].dts;
    s.p.iter_mut().for_each(|p| p.dts -= s.t0);
    s.f = f;
    s
}

fn avcc_nals(mut b: &[u8]) -> impl Iterator<Item = &[u8]> {
    std::iter::from_fn(move || {
        let n = u32::from_be_bytes(b.get(..4)?.try_into().unwrap()) as usize;
        let x = &b[4..4 + n];
        b = &b[4 + n..];
        Some(x)
    })
}

fn cfg(s: &Src, lag: u32) -> rtmp::Cfg {
    let meta = flv::Meta {
        w: s.w,
        h: s.h,
        fps: 29.97,
        rate: 48000,
        ch: 2,
        ..Default::default()
    };
    rtmp::Cfg {
        meta,
        avcc: s.avcc.clone(),
        asc: s.asc.clone(),
        lag,
    }
}

// ---- validation with ffprobe ----

/// (video, pts, dts, key, size) per packet, times in ms rounded like [`mp4`].
fn probe(p: &Path) -> Vec<(bool, i64, i64, bool, usize)> {
    let o = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "packet=codec_type,pts,dts,flags,size:stream=codec_type,time_base",
            "-of",
            "compact",
        ])
        .arg(p)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let o = String::from_utf8(o.stdout).unwrap();
    let f = |l: &str, k: &str| {
        l.split('|')
            .find_map(|x| x.strip_prefix(k).and_then(|x| x.strip_prefix('=')))
            .unwrap()
            .to_string()
    };
    let tb = |v: bool| {
        let l = o
            .lines()
            .find(|l| l.starts_with("stream|") && (f(l, "codec_type") == "video") == v)
            .unwrap();
        let t = f(l, "time_base");
        let (n, d) = t.split_once('/').unwrap();
        (n.parse::<i64>().unwrap(), d.parse::<i64>().unwrap())
    };
    let tb = [tb(false), tb(true)];
    o.lines()
        .filter(|l| l.starts_with("packet|"))
        .map(|l| {
            let v = f(l, "codec_type") == "video";
            let (n, d) = tb[v as usize];
            let ms = |k| (f(l, k).parse::<i64>().unwrap() * n * 1000 + d / 2).div_euclid(d);
            (
                v,
                ms("pts"),
                ms("dts"),
                f(l, "flags").starts_with('K'),
                f(l, "size").parse().unwrap(),
            )
        })
        .collect()
}

fn decode_errors(p: &Path) -> String {
    let o = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(p)
        .args(["-f", "null", "-"])
        .output()
        .unwrap();
    assert!(o.status.success());
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// The file holds exactly `want` shifted by `t0` ms, per stream and in order, and decodes without
/// errors.
fn check(file: &Path, want: &[P], t0: i64) {
    let got = probe(file);
    for v in [true, false] {
        let g: Vec<_> = got
            .iter()
            .filter(|p| p.0 == v)
            .map(|p| (p.1, p.2, p.3, p.4))
            .collect();
        let w: Vec<_> = want
            .iter()
            .filter(|p| p.video == v)
            .map(|p| (t0 + p.dts + p.cts as i64, t0 + p.dts, p.key, p.len))
            .collect();
        assert_eq!(
            g.len(),
            w.len(),
            "{} packet count",
            if v { "video" } else { "audio" }
        );
        assert_eq!(g, w);
        assert!(g.windows(2).all(|x| x[0].1 <= x[1].1), "dts not monotonic");
    }
    let e = decode_errors(file);
    assert!(e.is_empty(), "decode errors: {e}");
    let n = |v: bool| want.iter().filter(|p| p.video == v).count();
    println!(
        "{}: {} video + {} audio packets, timestamps and sizes match, decodes cleanly",
        file.display(),
        n(true),
        n(false)
    );
}

/// An x264 encode of the first 3 s with B-frames, for composition offsets. Made once per run, so
/// that no test reads it half-written.
fn bframes(d: &Path) -> PathBuf {
    static B: OnceLock<PathBuf> = OnceLock::new();
    B.get_or_init(|| {
        fixture("bframes.mp4", |o| {
            let s = Command::new("ffmpeg")
                .args(["-v", "error", "-y", "-i"])
                .arg(d.join("v635.lrv"))
                .args([
                    "-t", "3", "-c:v", "libx264", "-preset", "veryfast", "-bf", "3", "-g", "30",
                    "-c:a", "copy",
                ])
                .arg(o)
                .status()
                .unwrap();
            assert!(s.success());
        })
    })
    .clone()
}

#[test]
fn flv_file() {
    let Some(d) = samples() else { return };
    let s = mp4(&d.join("v635.lrv"));
    assert_eq!(
        (s.w, s.h, s.p.iter().filter(|p| p.video).count(), s.p.len()),
        (1664, 832, 173, 173 + 268)
    );
    // the camera's avcC is exactly what we build from its SPS and PPS
    let sps = &s.avcc[8..8 + u16::from_be_bytes([s.avcc[6], s.avcc[7]]) as usize];
    let pps = &s.avcc[8 + sps.len() + 3..];
    assert_eq!(flv::avcc(sps, pps), s.avcc);
    if !have_ffmpeg() {
        return;
    }
    let bf = bframes(&d);
    let b = mp4(&bf);
    // the reader agrees with FFmpeg on the sources, edit list included
    check(&d.join("v635.lrv"), &s.p, s.t0);
    check(&bf, &b.p, b.t0);
    assert_eq!((s.t0, b.t0), (0, -67));
    for (name, s) in [("lrv.flv", s), ("bframes.flv", b)] {
        let path = fixture(name, |f| {
            let mut w =
                flv::Writer::new(BufWriter::new(fs::File::create(f).unwrap()), true, true).unwrap();
            let mut b = vec![];
            flv::meta(&mut b, &cfg(&s, 0).meta);
            w.tag(flv::DATA, 0, &b).unwrap();
            b.clear();
            flv::avc_seq(&mut b, &s.avcc);
            w.tag(flv::VIDEO, 0, &b).unwrap();
            b.clear();
            flv::aac_seq(&mut b, &s.asc);
            w.tag(flv::AUDIO, 0, &b).unwrap();
            let mut ab = vec![];
            for p in &s.p {
                b.clear();
                let x = &s.f[p.off..p.off + p.len];
                if p.video {
                    // through Annex-B, the form the Vulkan encoder emits
                    ab.clear();
                    avcc_nals(x)
                        .for_each(|n| ab.extend_from_slice(&[&[0, 0, 0, 1][..], n].concat()));
                    flv::avc(&mut b, p.key, p.cts, flv::nals(&ab));
                    assert_eq!(&b[5..], x);
                } else {
                    flv::aac(&mut b, x);
                }
                w.tag(
                    if p.video { flv::VIDEO } else { flv::AUDIO },
                    p.dts as u32,
                    &b,
                )
                .unwrap();
            }
            w.w.flush().unwrap();
        });
        if name == "bframes.flv" {
            assert!(
                s.p.iter().any(|p| p.cts > 0),
                "no composition offsets in the B-frame fixture"
            );
        }
        check(&path, &s.p, 0);
    }
}

/// A test tool's process and the file it writes: dropping it kills and reaps the one, then removes the other, also
/// when the test fails.
struct Kid(Child, PathBuf);

impl Kid {
    /// Spawns `c`. The child also gets SIGKILL when this thread ends, so a test process that dies without unwinding
    /// (killed, aborted) leaves no listener behind.
    fn new(c: &mut Command, f: &Path) -> Kid {
        use std::os::unix::process::CommandExt;
        let pp = std::process::id();
        // only async-signal-safe calls between fork and exec
        unsafe {
            c.pre_exec(move || {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
                // the parent died before the prctl
                if libc::getppid() as u32 != pp {
                    libc::_exit(1);
                }
                Ok(())
            })
        };
        Kid(c.spawn().unwrap(), f.into())
    }
}

impl Drop for Kid {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
        let _ = fs::remove_file(&self.1);
    }
}

/// `ffmpeg -listen` on `port` into `file`; None when it ends before it listens (the port is taken).
fn listen(port: u16, file: &Path) -> Option<Kid> {
    let mut c = Kid::new(
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-listen",
                "1",
                "-i",
                &format!("rtmp://127.0.0.1:{port}/live/test"),
                "-c",
                "copy",
            ])
            .arg(file)
            .stdin(Stdio::null()),
        file,
    );
    // wait until it listens, on a socket of its own: another run's listener may hold the port. A
    // probe connection would be taken as the publisher.
    let (hex, pid) = (format!("0100007F:{port:04X}"), c.0.id());
    let own = |i: &str| {
        let s = format!("socket:[{i}]");
        fs::read_dir(format!("/proc/{pid}/fd"))
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| fs::read_link(e.path()).is_ok_and(|l| l.as_os_str() == &*s))
    };
    let t = Instant::now();
    while !fs::read_to_string("/proc/net/tcp")
        .unwrap()
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>())
        .any(|l| l.len() > 9 && l[1] == hex && l[3] == "0A" && own(l[9]))
    {
        if c.0.try_wait().unwrap().is_some() {
            return None;
        }
        assert!(
            t.elapsed() < Duration::from_secs(10),
            "ffmpeg isn't listening"
        );
        thread::sleep(Duration::from_millis(20));
    }
    Some(c)
}

/// A listener into `file` on a port the kernel found free, and that port; should another process
/// take it before ffmpeg does, on another.
fn serve(file: &Path) -> (Kid, u16) {
    (0..5)
        .find_map(|_| {
            let p = TcpListener::bind("127.0.0.1:0")
                .unwrap()
                .local_addr()
                .unwrap()
                .port();
            listen(p, file).map(|k| (k, p))
        })
        .expect("no port ffmpeg could listen on")
}

/// A socket already listening on the port is not taken for ffmpeg's: `listen` waits for ffmpeg to
/// give up.
#[test]
fn port_taken() {
    if !have_ffmpeg() {
        return;
    }
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    assert!(listen(l.local_addr().unwrap().port(), &out("taken.flv")).is_none());
}

fn wait(c: &mut Child) {
    let t = Instant::now();
    loop {
        if let Some(s) = c.try_wait().unwrap() {
            assert!(s.success(), "ffmpeg failed: {s}");
            return;
        }
        if t.elapsed() > Duration::from_secs(30) {
            panic!("ffmpeg didn't finish");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn send(r: &rtmp::Rtmp, s: &Src, p: &P, shift: u32) {
    let (x, ts) = (&s.f[p.off..p.off + p.len], p.dts as u32 + shift);
    if p.video {
        r.video(ts, p.cts, p.key, avcc_nals(x));
    } else {
        r.audio(ts, x);
    }
}

fn until_sent(r: &rtmp::Rtmp, n: usize, t: Instant) {
    while r.stats().sent.iter().sum::<u64>() < n as u64 {
        assert!(t.elapsed() < Duration::from_secs(20), "{:?}", r.stats());
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn rtmp_ffmpeg() {
    let Some(d) = samples() else { return };
    if !have_ffmpeg() {
        return;
    }
    for (src, name) in [
        (d.join("v635.lrv"), "rtmp_out.flv"),
        (bframes(&d), "rtmp_bframes.flv"),
    ] {
        let s = mp4(&src);
        let file = out(name);
        let (mut ff, port) = serve(&file);
        let t = Instant::now();
        let r = rtmp::Rtmp::start(
            &format!("rtmp://127.0.0.1:{port}/live"),
            "test",
            cfg(&s, 60_000),
        )
        .unwrap();
        s.p.iter().for_each(|p| send(&r, &s, p, 0));
        until_sent(&r, s.p.len(), t);
        let (st, e) = (r.stats(), t.elapsed());
        let t = Instant::now();
        drop(r);
        println!("{name}: {st:?} in {e:?}, stopped in {:?}", t.elapsed());
        assert_eq!(
            (st.state, st.drop, st.reconnects),
            (rtmp::State::Live, [0, 0], 0)
        );
        wait(&mut ff.0);
        check(&file, &s.p, 0);
    }
}

/// Live with nothing queued, the first audio arrives before the first key frame though it's newer,
/// as when video takes longer to encode: the stream still starts at that key frame, losing nothing.
#[test]
fn rtmp_late_key() {
    let Some(d) = samples() else { return };
    if !have_ffmpeg() {
        return;
    }
    let s = mp4(&bframes(&d));
    let file = out("rtmp_late_key.flv");
    let (mut ff, port) = serve(&file);
    let u = format!("rtmp://127.0.0.1:{port}/live/test");
    let r = rtmp::Rtmp::start(&u, "", cfg(&s, 60_000)).unwrap();
    let t = Instant::now();
    while r.stats().state != rtmp::State::Live {
        assert!(t.elapsed() < Duration::from_secs(10), "{:?}", r.stats());
        thread::sleep(Duration::from_millis(5));
    }
    let a = s.p.iter().position(|p| !p.video).unwrap();
    assert!(s.p[0].key && s.p[0].dts < s.p[a].dts);
    send(&r, &s, &s.p[a], 0);
    // ample time for the writer to take it, were it free to
    thread::sleep(Duration::from_millis(100));
    assert_eq!(
        r.stats().sent,
        [0, 0],
        "audio went out before the first key frame"
    );
    for (i, p) in s.p.iter().enumerate().filter(|x| x.0 != a) {
        send(&r, &s, p, 0);
        // the writer takes each frame as it comes, as in a live pipeline
        if i < 8 {
            thread::sleep(Duration::from_millis(20));
        }
    }
    until_sent(&r, s.p.len(), t);
    let st = r.stats();
    drop(r);
    println!(
        "first audio {} ms ahead of the key frame: {st:?}",
        s.p[a].dts - s.p[0].dts
    );
    wait(&mut ff.0);
    assert_eq!(st.drop, [0, 0]);
    check(&file, &s.p, 0);
}

/// Past 0xFFFFFF ms (4.6 h) timestamps take the extended field: a key frame at 0, then the clip
/// from 2 s before that mark.
#[test]
fn rtmp_ext_ts() {
    let Some(d) = samples() else { return };
    if !have_ffmpeg() {
        return;
    }
    let s = mp4(&d.join("v635.lrv"));
    let file = out("rtmp_ext.flv");
    let (mut ff, port) = serve(&file);
    let u = format!("rtmp://127.0.0.1:{port}/live/test");
    let r = rtmp::Rtmp::start(&u, "", cfg(&s, 60_000)).unwrap();
    let want: Vec<P> = s.p[..1]
        .iter()
        .copied()
        .chain(s.p.iter().map(|p| P {
            dts: p.dts + 0xFF_FFFF - 2000,
            ..*p
        }))
        .collect();
    assert!(want[0].video && want[0].key);
    // 4.6 h behind the queue's head the rest would be dropped, so the first frame goes alone
    let t = Instant::now();
    send(&r, &s, &want[0], 0);
    until_sent(&r, 1, t);
    want[1..].iter().for_each(|p| send(&r, &s, p, 0));
    until_sent(&r, want.len(), t);
    drop(r);
    wait(&mut ff.0);
    check(&file, &want, 0);
    let x = want.iter().filter(|p| p.dts > 0xFF_FFFF).count();
    println!("{x} packets past 0xFFFFFF ms");
}

/// Dropping a live publisher whose receiver froze takes at most `STOP`.
#[test]
fn rtmp_stop() {
    let Some(d) = samples() else { return };
    if !have_ffmpeg() {
        return;
    }
    let s = mp4(&d.join("v635.lrv"));
    let (ff, port) = serve(&out("rtmp_frozen.flv"));
    let u = format!("rtmp://127.0.0.1:{port}/live/test");
    let r = rtmp::Rtmp::start(&u, "", cfg(&s, 600_000)).unwrap();
    let t = Instant::now();
    s.p.iter().for_each(|p| send(&r, &s, p, 0));
    until_sent(&r, s.p.len(), t);
    unsafe { libc::kill(ff.0.id() as i32, libc::SIGSTOP) };
    // 19 more loops (about 57 MB), far beyond the socket buffers
    let len = s.p.iter().map(|p| p.dts).max().unwrap() as u32 + 40;
    for i in 1..20 {
        s.p.iter().for_each(|p| send(&r, &s, p, i * len));
    }
    thread::sleep(Duration::from_millis(500));
    let st = r.stats();
    let t = Instant::now();
    drop(r);
    let e = t.elapsed();
    drop(ff);
    println!("frozen receiver: {st:?}, stopped in {e:?}");
    assert!(st.sent[0] < 20 * 173, "the writer never blocked");
    assert!(e < rtmp::STOP + Duration::from_millis(500), "{e:?}");
}

#[test]
fn rtmp_backpressure() {
    let Some(d) = samples() else { return };
    if !have_ffmpeg() {
        return;
    }
    let s = mp4(&d.join("v635.lrv"));
    let file = out("rtmp_stall.flv");
    let (mut ff, port) = serve(&file);
    let pid = ff.0.id() as i32;
    let u = format!("rtmp://127.0.0.1:{port}/live/test");
    let r = rtmp::Rtmp::start(&u, "", cfg(&s, 500)).unwrap();
    // freeze the receiver for 3 s while publishing 4 loops of the clip at 4× real time
    let stall = thread::spawn(move || {
        thread::sleep(Duration::from_secs(1));
        unsafe { libc::kill(pid, libc::SIGSTOP) };
        thread::sleep(Duration::from_secs(3));
        unsafe { libc::kill(pid, libc::SIGCONT) };
    });
    let len = s.p.iter().map(|p| p.dts).max().unwrap() as u32 + 40;
    let t = Instant::now();
    for i in 0..4 {
        for p in &s.p {
            let ts = p.dts as u32 + i * len;
            if let Some(w) = Duration::from_millis(ts as u64 / 4).checked_sub(t.elapsed()) {
                thread::sleep(w);
            }
            send(&r, &s, p, i * len);
        }
    }
    stall.join().unwrap();
    let st = r.stats();
    drop(r);
    println!("stalled: {st:?}");
    wait(&mut ff.0);
    assert!(st.drop[0] > 0, "the stall dropped no video: {st:?}");
    let got = probe(&file);
    let n = |v: bool| got.iter().filter(|p| p.0 == v).count() as u64;
    assert_eq!(n(true) + st.drop[0], 4 * 173);
    assert_eq!(n(false) + st.drop[1], 4 * 268);
    for v in [true, false] {
        assert!(
            got.iter()
                .filter(|p| p.0 == v)
                .collect::<Vec<_>>()
                .windows(2)
                .all(|x| x[0].2 < x[1].2),
            "dts not increasing"
        );
    }
    let e = decode_errors(&file);
    assert!(e.is_empty(), "decode errors after drops: {e}");
    println!(
        "received {} video + {} audio packets, {:?} dropped, decodes cleanly",
        n(true),
        n(false),
        st.drop
    );
}

#[test]
fn rtmp_reconnect() {
    let Some(d) = samples() else { return };
    if !have_ffmpeg() {
        return;
    }
    let s = mp4(&d.join("v635.lrv"));
    let (f1, f2) = (out("rtmp_cut.flv"), out("rtmp_resumed.flv"));
    let (mut ff, port) = serve(&f1);
    let u = format!("rtmp://127.0.0.1:{port}/live/test");
    let r = rtmp::Rtmp::start(&u, "", cfg(&s, 2000)).unwrap();
    // publish 3 loops at 4× real time; the server dies after 1 s and a new one takes the port (on this thread: a
    // listener dies with the thread that spawned it)
    let len = s.p.iter().map(|p| p.dts).max().unwrap() as u32 + 40;
    let (t, mut cut) = (Instant::now(), false);
    for i in 0..3 {
        for p in &s.p {
            if let Some(w) =
                Duration::from_millis((p.dts as u32 + i * len) as u64 / 4).checked_sub(t.elapsed())
            {
                thread::sleep(w);
            }
            if !cut && t.elapsed() > Duration::from_secs(1) {
                drop(ff);
                ff = listen(port, &f2).expect("the port was taken");
                cut = true;
            }
            send(&r, &s, p, i * len);
        }
    }
    // every frame is either sent or dropped
    while {
        let x = r.stats();
        x.sent.iter().chain(&x.drop).sum::<u64>() < 3 * (173 + 268)
    } {
        assert!(t.elapsed() < Duration::from_secs(20), "{:?}", r.stats());
        thread::sleep(Duration::from_millis(10));
    }
    let st = r.stats();
    drop(r);
    println!("reconnected: {st:?}");
    wait(&mut ff.0);
    assert_eq!(st.reconnects, 1);
    assert_eq!(st.sent[0] + st.drop[0], 3 * 173);
    assert_eq!(st.sent[1] + st.drop[1], 3 * 268);
    let got = probe(&f2);
    let v: Vec<_> = got.iter().filter(|p| p.0).collect();
    // the new session starts at a key frame, at timestamp 0
    assert!(v[0].3 && v[0].2 == 0, "{:?}", v[0]);
    let a: Vec<_> = got.iter().filter(|p| !p.0).collect();
    for x in [&v, &a] {
        assert!(x.windows(2).all(|x| x[0].2 < x[1].2), "dts not increasing");
    }
    let e = decode_errors(&f2);
    assert!(e.is_empty(), "decode errors after the reconnect: {e}");
    println!(
        "second server: {} video + {} audio packets, decodes cleanly",
        v.len(),
        got.len() - v.len()
    );
}
