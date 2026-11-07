//! The app's camera code without its UI, so the app's screens and a headless host run the same session code: the
//! camera worker (`camera`: the session over USB or Wi-Fi, identity and options, the clock, the album and its
//! thumbnails, the download queue, the camera commands, mounted cards) and the capture worker (`capture`: mode,
//! settings and shutter on that session, from the camera's `/CONF`). Under them: the album's grouping and thumbnail
//! cache (`album`), the download queue (`dl`), mounted cards (`disk`), `/CONF` (`conf`), a recording's files
//! (`work`), the config, cache and video folders (`dirs`), App Nap and idle sleep (`awake`). The commands that change
//! the camera check what they may do in the workers (no delete while recording, `/CONF`'s allowed values, the 5 %
//! battery rule, an explicit start or stop), so every host gets the same guards. Linux and macOS; Windows type-checks
//! (no cards found, the camera clock left alone).

pub mod album;
pub mod awake;
pub mod camera;
pub mod capture;
pub mod conf;
pub mod dirs;
pub mod disk;
pub mod dl;
#[cfg(any(test, feature = "sim"))]
pub mod sim;
pub mod work;

/// CLOCK_MONOTONIC in ns, the clock the live preview's pictures, the camera clock's offset and the live outputs
/// (PipeWire, OBS) go by.
#[cfg(unix)]
pub fn mono() -> u64 {
    let mut t: libc::timespec = unsafe { std::mem::zeroed() };
    // SAFETY: clock_gettime only writes the timespec it is given
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut t) };
    t.tv_sec as u64 * 1_000_000_000 + t.tv_nsec as u64
}

/// A monotonic clock in ns (no CLOCK_MONOTONIC here): from a day before the first call, so a time taken before it
/// still subtracts.
#[cfg(not(unix))]
pub fn mono() -> u64 {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64 + 86_400_000_000_000
}
