//! The per-user endpoint between x5desk processes: the first `x5desk mcp` binds it and hosts, later ones connect to
//! it and proxy (`front`). Unix: a socket in an owner-only dir (Linux `$XDG_RUNTIME_DIR/x5desk/mcp.sock`, without it
//! `<cache>/x5desk/mcp.sock`; macOS `<config>/x5desk/mcp.sock`; a path too long for `sun_path`
//! `/tmp/x5desk-<uid>/mcp.sock`). The election is serialised by an `flock` on `<socket>.lock`: connect, else remove a
//! stale socket and bind. A client connects only to a socket file of its own user and checks the peer too; a host
//! closes connections of other users. A host's socket goes when it does, under the lock and only if it is still its.
//! Windows: `\\.\pipe\x5desk-mcp-<user SID>`, the first instance made exclusively (`first_pipe_instance`, so the
//! election needs no lock), local clients only, a DACL of the current user alone, and the client checking that the
//! server process runs as its user. `X5DESK_MCP_ENDPOINT` names another endpoint (tests); `--sim` and `X5DESK_TEST`
//! use none without it.

use std::{io, path::PathBuf};
use tokio::io::{AsyncRead, AsyncWrite};

/// A connection's read half.
pub type Read = Box<dyn AsyncRead + Send + Unpin>;

/// A connection's write half.
pub type Write = Box<dyn AsyncWrite + Send + Unpin>;

/// What an election made of this process.
pub enum Role {
    /// it bound the endpoint: it hosts and serves the connections the listener accepts
    Host(Listener),
    /// another process hosts: a connection to it
    Proxy(Read, Write),
}

/// The endpoint to use: `X5DESK_MCP_ENDPOINT` if set, else none for tests (`--sim`, `X5DESK_TEST`), else the
/// user's (its dir made owner-only). An error when there is no home to put it in or its dir is not the user's.
pub fn path(test: bool) -> io::Result<Option<PathBuf>> {
    if let Some(p) = std::env::var_os("X5DESK_MCP_ENDPOINT").filter(|p| !p.is_empty()) {
        return Ok(Some(p.into()));
    }
    if test {
        return Ok(None);
    }
    default().map(Some)
}

#[cfg(unix)]
pub use unix::*;
#[cfg(windows)]
pub use win::*;

#[cfg(unix)]
mod unix {
    use super::{Read, Role, Write};
    use std::{
        fs::{File, OpenOptions, TryLockError},
        io,
        os::unix::{
            fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt},
            net,
        },
        path::{Path, PathBuf},
    };
    use tokio::net::{UnixListener, UnixStream};

    /// `sun_path`'s size: a socket path has fewer bytes (std wants room for the NUL).
    #[cfg(target_os = "linux")]
    pub const SUN: usize = 108;
    /// `sun_path`'s size: a socket path has fewer bytes (std wants room for the NUL).
    #[cfg(not(target_os = "linux"))]
    pub const SUN: usize = 104;

    fn euid() -> u32 {
        // SAFETY: geteuid has no preconditions and cannot fail.
        unsafe { libc::geteuid() }
    }

    fn denied(s: String) -> io::Error {
        io::Error::new(io::ErrorKind::PermissionDenied, s)
    }

    /// `p` if it fits `sun_path`, else `/tmp/x5desk-<uid>/mcp.sock`.
    pub fn fit(p: PathBuf, uid: u32) -> PathBuf {
        match p.as_os_str().len() < SUN {
            true => p,
            false => PathBuf::from(format!("/tmp/x5desk-{uid}/mcp.sock")),
        }
    }

    /// The user's socket, its dir made: owner-only (`cam::dirs::private`), or in `/tmp` one of the user's that no
    /// other can write to (`cam::dirs::mine`).
    pub(super) fn default() -> io::Result<PathBuf> {
        let base = match cfg!(target_os = "macos") {
            true => cam::dirs::xdg("XDG_CONFIG_HOME", ".config"),
            false => (std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from))
                .filter(|d| d.is_absolute())
                .or_else(|| cam::dirs::xdg("XDG_CACHE_HOME", ".cache")),
        };
        let d = base
            .ok_or_else(|| io::Error::other("no home for the endpoint (HOME not set)"))?
            .join("x5desk");
        let p = fit(d.join("mcp.sock"), euid());
        match p.starts_with(&d) {
            true => cam::dirs::private(&d)?,
            false => {
                let t = p.parent().unwrap().to_path_buf();
                cam::dirs::mine(t.clone()).ok_or_else(|| {
                    denied(format!("{}: not a dir of this user's alone", t.display()))
                })?;
            }
        }
        Ok(p)
    }

    /// The election's lock, `<socket>.lock`, held while the file lives.
    fn lock(sock: &Path, wait: bool) -> io::Result<Option<File>> {
        let mut p = sock.as_os_str().to_owned();
        p.push(".lock");
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&p)?;
        if !wait {
            return match f.try_lock() {
                Ok(()) => Ok(Some(f)),
                Err(TryLockError::WouldBlock) => Ok(None),
                Err(TryLockError::Error(e)) => Err(e),
            };
        }
        loop {
            match f.lock() {
                Ok(()) => return Ok(Some(f)),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
    }

    enum Got {
        Host(net::UnixListener, (u64, u64)),
        Proxy(net::UnixStream),
    }

    /// Under the lock: connect to a socket of this user's, else remove a stale one (nobody listens) and bind.
    fn elect_now(p: &Path) -> io::Result<Got> {
        let _l = lock(p, true)?;
        match std::fs::symlink_metadata(p) {
            Ok(m) if !m.file_type().is_socket() || m.uid() != euid() => {
                return Err(denied(format!(
                    "{}: not a socket of this user's (owner uid {})",
                    p.display(),
                    m.uid()
                )));
            }
            Ok(_) => match net::UnixStream::connect(p) {
                Ok(s) => return Ok(Got::Proxy(s)),
                Err(e) if e.kind() == io::ErrorKind::ConnectionRefused => std::fs::remove_file(p)?,
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        let l = net::UnixListener::bind(p)?;
        std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600))?;
        let m = std::fs::symlink_metadata(p)?;
        Ok(Got::Host(l, (m.dev(), m.ino())))
    }

    /// Hosts at `p` or connects to its host (the flock blocks: off the runtime's threads).
    pub async fn elect(p: &Path) -> io::Result<Role> {
        let q = p.to_path_buf();
        let g = tokio::task::spawn_blocking(move || elect_now(&q))
            .await
            .map_err(io::Error::other)??;
        match g {
            Got::Proxy(s) => {
                s.set_nonblocking(true)?;
                let s = UnixStream::from_std(s)?;
                let u = s.peer_cred()?.uid();
                if u != euid() {
                    return Err(denied(format!("{}: its host runs as uid {u}", p.display())));
                }
                let (r, w) = s.into_split();
                Ok(Role::Proxy(Box::new(r), Box::new(w)))
            }
            Got::Host(l, id) => {
                l.set_nonblocking(true)?;
                Ok(Role::Host(Listener {
                    l: Some(UnixListener::from_std(l)?),
                    path: p.to_path_buf(),
                    id,
                }))
            }
        }
    }

    /// The bound endpoint.
    pub struct Listener {
        l: Option<UnixListener>,
        path: PathBuf,
        /// the socket file's device and inode
        id: (u64, u64),
    }

    impl Listener {
        /// The next connection; one of another user's is closed and an error.
        pub async fn accept(&mut self) -> io::Result<(Read, Write)> {
            let (s, _) = self.l.as_ref().unwrap().accept().await?;
            let u = s.peer_cred()?.uid();
            if u != euid() {
                return Err(denied(format!("refused a client of uid {u}")));
            }
            let (r, w) = s.into_split();
            Ok((Box::new(r), Box::new(w)))
        }

        /// Where it listens.
        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for Listener {
        /// Closes the socket, then removes its file if it is still this one: under the lock, not waiting for it (an
        /// election under way finds the socket stale and replaces it).
        fn drop(&mut self) {
            drop(self.l.take());
            if let Ok(Some(_l)) = lock(&self.path, false)
                && std::fs::symlink_metadata(&self.path)
                    .is_ok_and(|m| (m.dev(), m.ino()) == self.id)
            {
                let _ = std::fs::remove_file(&self.path);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A dir of a test's, removed when it drops (also when the test fails): in the temp dir, or `/tmp` when a
        /// socket there would not fit `sun_path`.
        struct Rm(PathBuf);

        impl Rm {
            fn new(name: &str) -> Rm {
                let n = format!("x5mcp-{name}-{}", std::process::id());
                let mut d = std::env::temp_dir().join(&n);
                if d.join("mcp.sock").as_os_str().len() >= SUN {
                    d = Path::new("/tmp").join(n)
                }
                let _ = std::fs::remove_dir_all(&d);
                std::fs::create_dir_all(&d).unwrap();
                Rm(d)
            }
        }

        impl Drop for Rm {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        #[test]
        fn fallback() {
            let short = PathBuf::from("/Users/a/Library/Application Support/x5desk/mcp.sock");
            assert_eq!(fit(short.clone(), 501), short);
            let at = |n: usize| PathBuf::from(format!("/{}", "d".repeat(n - 1)));
            assert_eq!(fit(at(SUN - 1), 501), at(SUN - 1), "fits with its NUL");
            for n in [SUN, 300] {
                assert_eq!(
                    fit(at(n), 501),
                    Path::new("/tmp/x5desk-501/mcp.sock"),
                    "{n}"
                );
            }
            assert!(fit(at(SUN), u32::MAX).as_os_str().len() < SUN);
        }

        /// The election on a temp path: host, then a proxy of it, the stale socket of a host gone replaced, a file
        /// that is not a socket refused, the socket removed when its host ends.
        #[tokio::test(flavor = "multi_thread")]
        async fn election() {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let d = Rm::new("ep");
            let p = d.0.join("mcp.sock");
            let Role::Host(mut l) = elect(&p).await.unwrap() else {
                panic!("first: host")
            };
            let m = std::fs::metadata(&p).unwrap();
            assert_eq!(m.permissions().mode() & 0o777, 0o600);
            let Role::Proxy(mut r, mut w) = elect(&p).await.unwrap() else {
                panic!("second: proxy")
            };
            let (mut hr, mut hw) = l.accept().await.unwrap();
            w.write_all(b"ping\n").await.unwrap();
            let mut b = [0; 5];
            hr.read_exact(&mut b).await.unwrap();
            assert_eq!(&b, b"ping\n");
            hw.write_all(b"pong\n").await.unwrap();
            r.read_exact(&mut b).await.unwrap();
            assert_eq!(&b, b"pong\n");
            // a host that died: its socket closed, the file left
            drop(l.l.take());
            std::mem::forget(l);
            assert!(p.exists(), "a stale socket");
            let Role::Host(l) = elect(&p).await.unwrap() else {
                panic!("stale: host")
            };
            drop(l);
            assert!(!p.exists(), "removed with its host");
            std::fs::write(&p, b"").unwrap();
            let e = elect(&p).await.err().unwrap();
            assert!(e.to_string().contains("not a socket"), "{e}");
        }
    }
}

#[cfg(windows)]
mod win {
    //! Every unsafe call's contract is noted at the call. Handles from the system are wrapped in `OwnedHandle` (closed
    //! once) right away; buffers the system fills are sized by its first answer and aligned for the struct read.
    use super::{Read, Role, Write};
    use std::{
        ffi::{OsStr, OsString, c_void},
        io,
        os::windows::{
            ffi::OsStrExt,
            io::{AsRawHandle, FromRawHandle, OwnedHandle},
        },
        path::{Path, PathBuf},
        time::Duration,
    };
    use tokio::net::windows::named_pipe::{
        ClientOptions, NamedPipeClient, NamedPipeServer, ServerOptions,
    };
    use windows_sys::Win32::{
        Foundation::{ERROR_ACCESS_DENIED, ERROR_PIPE_BUSY, HANDLE, LocalFree},
        Security::{
            Authorization::{
                ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
                SDDL_REVISION_1,
            },
            EqualSid, GetTokenInformation, PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES, TOKEN_QUERY,
            TOKEN_USER, TokenUser,
        },
        System::{
            Pipes::GetNamedPipeServerProcessId,
            Threading::{
                GetCurrentProcess, OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION,
            },
        },
    };

    fn denied(s: String) -> io::Error {
        io::Error::new(io::ErrorKind::PermissionDenied, s)
    }

    /// A process's `TOKEN_USER`, in a buffer aligned for it (the SID it points to is in the same buffer).
    fn token_user(process: HANDLE) -> io::Result<Vec<u64>> {
        let mut t: HANDLE = std::ptr::null_mut();
        // SAFETY: `process` is a valid process handle (the pseudo handle or one opened with query rights); `t` is a
        // valid out pointer that receives a token handle we own.
        if unsafe { OpenProcessToken(process, TOKEN_QUERY, &mut t) } == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `t` was just returned to us open and is closed only by this OwnedHandle.
        let t = unsafe { OwnedHandle::from_raw_handle(t) };
        let mut n = 0u32;
        // SAFETY: a size query: a null buffer of length 0, `n` a valid out pointer; it fails with the size in `n`.
        unsafe {
            GetTokenInformation(
                t.as_raw_handle(),
                TokenUser,
                std::ptr::null_mut(),
                0,
                &mut n,
            )
        };
        if n == 0 {
            return Err(io::Error::last_os_error());
        }
        let mut b = vec![0u64; (n as usize).div_ceil(8)];
        // SAFETY: `b` has at least `n` writable bytes, aligned for TOKEN_USER (8); `n` a valid out pointer.
        if unsafe {
            GetTokenInformation(
                t.as_raw_handle(),
                TokenUser,
                b.as_mut_ptr().cast(),
                n,
                &mut n,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        Ok(b)
    }

    /// The SID inside a [`token_user`] buffer.
    fn sid(b: &[u64]) -> *mut c_void {
        // SAFETY: `b` holds a TOKEN_USER that GetTokenInformation wrote at its start, aligned for it.
        unsafe { (*(b.as_ptr() as *const TOKEN_USER)).User.Sid }
    }

    /// This process's user.
    fn me() -> io::Result<Vec<u64>> {
        // SAFETY: GetCurrentProcess returns a pseudo handle that needs no closing and is always valid.
        token_user(unsafe { GetCurrentProcess() })
    }

    /// This process's user's SID as text (`S-1-5-21-…`).
    fn sid_text() -> io::Result<String> {
        let u = me()?;
        let mut s: *mut u16 = std::ptr::null_mut();
        // SAFETY: the SID points into `u`, alive for the call; `s` receives a LocalAlloc'd string we free below.
        if unsafe { ConvertSidToStringSidW(sid(&u), &mut s) } == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `s` is a NUL-terminated UTF-16 string the call allocated; read up to the NUL, then freed once.
        let t = unsafe {
            let n = (0..).take_while(|&i| *s.add(i) != 0).count();
            let t = String::from_utf16_lossy(std::slice::from_raw_parts(s, n));
            LocalFree(s.cast());
            t
        };
        Ok(t)
    }

    /// `\\.\pipe\x5desk-mcp-<user SID>`.
    pub(super) fn default() -> io::Result<PathBuf> {
        Ok(PathBuf::from(format!(
            r"\\.\pipe\x5desk-mcp-{}",
            sid_text()?
        )))
    }

    /// A pipe instance at `name` that only this user may open, local clients only; `first`: fails if one exists.
    fn create(name: &OsStr, first: bool) -> io::Result<NamedPipeServer> {
        let sddl: Vec<u16> = OsString::from(format!("D:P(A;;GA;;;{})", sid_text()?))
            .encode_wide()
            .chain([0])
            .collect();
        let mut sd: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
        // SAFETY: `sddl` is NUL-terminated UTF-16 alive for the call; `sd` receives a LocalAlloc'd descriptor freed
        // below; the size out pointer may be null.
        if unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl.as_ptr(),
                SDDL_REVISION_1,
                &mut sd,
                std::ptr::null_mut(),
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }
        let mut sa = SECURITY_ATTRIBUTES {
            nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: sd,
            bInheritHandle: 0,
        };
        // SAFETY: `sa` is a valid SECURITY_ATTRIBUTES whose descriptor stays alive until after the call.
        let r = unsafe {
            ServerOptions::new()
                .first_pipe_instance(first)
                .reject_remote_clients(true)
                .create_with_security_attributes_raw(
                    name,
                    (&mut sa as *mut SECURITY_ATTRIBUTES).cast(),
                )
        };
        // SAFETY: `sd` was allocated by the conversion above and is not used after this.
        unsafe { LocalFree(sd) };
        r
    }

    /// Whether the process serving `c` runs as this user.
    fn ours(c: &NamedPipeClient) -> io::Result<bool> {
        let mut pid = 0u32;
        // SAFETY: `c` holds a connected pipe client handle for the call; `pid` is a valid out pointer.
        if unsafe { GetNamedPipeServerProcessId(c.as_raw_handle(), &mut pid) } == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: no pointers; the handle returned, if any, is ours to close.
        let h = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if h.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `h` was just opened for us and is closed only by this OwnedHandle.
        let h = unsafe { OwnedHandle::from_raw_handle(h) };
        let (them, us) = (token_user(h.as_raw_handle())?, me()?);
        // SAFETY: both SIDs point into buffers alive for the call.
        Ok(unsafe { EqualSid(sid(&them), sid(&us)) } != 0)
    }

    fn halves<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Send + 'static>(
        s: S,
    ) -> (Read, Write) {
        let (r, w) = tokio::io::split(s);
        (Box::new(r), Box::new(w))
    }

    /// Hosts at `name` or connects to its host. The first instance is made exclusively, so of two starters one
    /// hosts and the other connects.
    pub async fn elect(name: &Path) -> io::Result<Role> {
        let n = name.as_os_str();
        for _ in 0..100 {
            match ClientOptions::new().open(n) {
                Ok(c) => {
                    if !ours(&c)? {
                        return Err(denied(format!(
                            "{}: its host runs as another user",
                            name.display()
                        )));
                    }
                    let (r, w) = halves(c);
                    return Ok(Role::Proxy(r, w));
                }
                Err(e) if e.raw_os_error() == Some(ERROR_PIPE_BUSY as i32) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => match create(n, true) {
                    Ok(s) => {
                        return Ok(Role::Host(Listener {
                            name: name.to_path_buf(),
                            next: s,
                        }));
                    }
                    // another starter made the first instance: connect to it
                    Err(e) if e.raw_os_error() == Some(ERROR_ACCESS_DENIED as i32) => {}
                    Err(e) => return Err(e),
                },
                Err(e) => return Err(e),
            }
            tokio::time::sleep(Duration::from_millis(50)).await
        }
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            format!("{}: busy", name.display()),
        ))
    }

    /// The bound endpoint: the instance the next client connects to.
    pub struct Listener {
        name: PathBuf,
        next: NamedPipeServer,
    }

    impl Listener {
        /// The next connection (the pipe's DACL admits this user only).
        pub async fn accept(&mut self) -> io::Result<(Read, Write)> {
            self.next.connect().await?;
            // the next instance before this one is handed off, so a client always finds one
            let n = create(self.name.as_os_str(), false)?;
            Ok(halves(std::mem::replace(&mut self.next, n)))
        }

        /// Its pipe name.
        pub fn path(&self) -> &Path {
            &self.name
        }
    }
}
