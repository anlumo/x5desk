//! A DRM timeline syncobj: OBS exports the fence of its draws into its points (`gs_sync_export_syncobj_timeline_point`)
//! and the receive thread checks them without blocking. Layouts and ioctl numbers follow `<drm/drm.h>` (checked by
//! `tests::layout`).
use std::{
    fs::OpenOptions,
    io,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
};

#[repr(C)]
pub(crate) struct Cap {
    pub cap: u64,
    pub val: u64,
}

#[repr(C)]
pub(crate) struct Create {
    pub handle: u32,
    pub flags: u32,
}

#[repr(C)]
pub(crate) struct Handle {
    pub handle: u32,
    pub flags: u32,
    pub fd: i32,
    pub pad: u32,
    pub point: u64,
}

#[repr(C)]
pub(crate) struct Wait {
    pub handles: u64,
    pub points: u64,
    pub timeout_nsec: i64,
    pub count_handles: u32,
    pub flags: u32,
    pub first_signaled: u32,
    pub pad: u32,
    pub deadline_nsec: u64,
}

pub(crate) const GET_CAP: u64 = 0xc010_640c;
pub(crate) const CREATE: u64 = 0xc008_64bf;
pub(crate) const DESTROY: u64 = 0xc008_64c0;
pub(crate) const HANDLE_TO_FD: u64 = 0xc018_64c1;
pub(crate) const TIMELINE_WAIT: u64 = 0xc030_64ca;
pub(crate) const CAP_SYNCOBJ_TIMELINE: u64 = 0x14;

fn ioctl<T>(fd: &OwnedFd, req: u64, a: &mut T) -> io::Result<()> {
    match unsafe { libc::ioctl(fd.as_raw_fd(), req as _, a as *mut T) } {
        0 => Ok(()),
        _ => Err(io::Error::last_os_error()),
    }
}

pub struct Obj {
    dev: OwnedFd,
    h: u32,
    /// what OBS imports
    pub fd: OwnedFd,
}

impl Obj {
    /// On the first render node with timeline syncobjs; any node works, fences are device-agnostic.
    pub fn new() -> Option<Obj> {
        (128..192).find_map(|i| {
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(format!("/dev/dri/renderD{i}"))
                .ok()?;
            let dev = OwnedFd::from(f);
            let mut c = Cap {
                cap: CAP_SYNCOBJ_TIMELINE,
                val: 0,
            };
            ioctl(&dev, GET_CAP, &mut c).ok().filter(|_| c.val != 0)?;
            let mut cr = Create {
                handle: 0,
                flags: 0,
            };
            ioctl(&dev, CREATE, &mut cr).ok()?;
            let mut h = Handle {
                handle: cr.handle,
                flags: 0,
                fd: -1,
                pad: 0,
                point: 0,
            };
            if ioctl(&dev, HANDLE_TO_FD, &mut h).is_err() {
                let _ = ioctl(&dev, DESTROY, &mut cr);
                return None;
            }
            let fd = unsafe { OwnedFd::from_raw_fd(h.fd) };
            Some(Obj {
                dev,
                h: cr.handle,
                fd,
            })
        })
    }

    /// Whether `point`, which has a fence, has signalled. Any error but ETIME counts as signalled: a point that
    /// can't be waited for must not pin the sender's buffer.
    pub fn signaled(&self, point: u64) -> bool {
        let mut w = Wait {
            handles: &self.h as *const u32 as u64,
            points: &point as *const u64 as u64,
            timeout_nsec: 0,
            count_handles: 1,
            flags: 0,
            first_signaled: 0,
            pad: 0,
            deadline_nsec: 0,
        };
        ioctl(&self.dev, TIMELINE_WAIT, &mut w)
            .err()
            .is_none_or(|e| e.raw_os_error() != Some(libc::ETIME))
    }
}

impl Drop for Obj {
    fn drop(&mut self) {
        let _ = ioctl(
            &self.dev,
            DESTROY,
            &mut Create {
                handle: self.h,
                flags: 0,
            },
        );
    }
}
