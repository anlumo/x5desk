//! stdout belongs to the protocol. [`guard`] takes a copy of stdout for the protocol's writer and points stdout
//! itself at stderr, so whatever else prints (the app's hooks, dependencies) cannot corrupt the stream. Unix: a dup of
//! fd 1, then fd 2 onto fd 1 (`/dev/null` when there is no stderr). Windows: a duplicate of the stdout handle, then
//! `SetStdHandle(STD_OUTPUT_HANDLE, <stderr>)`: std looks the handle up on every write (`GetStdHandle` in
//! `sys/stdio/windows.rs` `write`), so `println!` follows; the C runtime's own fd 1 keeps the old handle.

use std::{fs::File, io};

/// The protocol's writer: what stdout was. From now on stdout is stderr.
#[cfg(unix)]
pub fn guard() -> io::Result<File> {
    use std::os::fd::FromRawFd;
    // SAFETY: fcntl on fd 1 with F_DUPFD_CLOEXEC only reads it; a new fd ≥ 3 is returned that we own.
    let fd = unsafe { libc::fcntl(1, libc::F_DUPFD_CLOEXEC, 3) };
    if fd < 0 {
        return Err(io::Error::other(format!(
            "stdout: {}",
            io::Error::last_os_error()
        )));
    }
    // SAFETY: `fd` was just returned to us open and is owned by this File alone.
    let f = unsafe { File::from_raw_fd(fd) };
    // SAFETY: dup2 onto fd 1, which std's stdout writes to; fd 2 is only read.
    if unsafe { libc::dup2(2, 1) } < 0 {
        let n = std::fs::OpenOptions::new().write(true).open("/dev/null")?;
        // SAFETY: as above; `n` stays open for the call and fd 1 keeps its own copy.
        if unsafe { libc::dup2(std::os::fd::AsRawFd::as_raw_fd(&n), 1) } < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(f)
}

/// The protocol's writer: what stdout was. From now on stdout is stderr.
#[cfg(windows)]
pub fn guard() -> io::Result<File> {
    use std::os::windows::io::FromRawHandle;
    use windows_sys::Win32::{
        Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE, INVALID_HANDLE_VALUE},
        System::{
            Console::{GetStdHandle, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE, SetStdHandle},
            Threading::GetCurrentProcess,
        },
    };
    // SAFETY: GetStdHandle has no preconditions; it returns the process's slot value, which we do not close.
    let out = unsafe { GetStdHandle(STD_OUTPUT_HANDLE) };
    if out.is_null() || out == INVALID_HANDLE_VALUE {
        return Err(io::Error::other("no stdout"));
    }
    let mut dup: HANDLE = std::ptr::null_mut();
    // SAFETY: the pseudo handle of this process (never closed) on both sides, `out` a valid handle, `dup` a valid
    // out pointer that receives a handle we own.
    let me = unsafe { GetCurrentProcess() };
    if unsafe { DuplicateHandle(me, out, me, &mut dup, 0, 0, DUPLICATE_SAME_ACCESS) } == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `dup` was just returned to us open and is owned by this File alone.
    let f = unsafe { File::from_raw_handle(dup) };
    // SAFETY: stderr's slot value (maybe null: std then drops stdout's writes as it does with no console) goes into
    // stdout's slot; neither handle is closed.
    if unsafe { SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE)) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(f)
}
