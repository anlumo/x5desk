//! App Nap (macOS): once every window is hidden or covered and no sound plays, the system coalesces the process's
//! timers and lowers its threads' priority. Long work holds a [`Hold`] while it runs (exports and frame grabs, Shot
//! Lab and AI passes, live outputs, downloads and card imports): the first begins an NSProcessInfo activity (user
//! initiated, idle system sleep allowed), the last to go ends it. Elsewhere a hold does nothing.

/// App Nap stays off while one lives.
#[must_use = "App Nap is off only while the hold lives"]
pub struct Hold(());

pub fn hold() -> Hold {
    #[cfg(target_os = "macos")]
    mac::take();
    Hold(())
}

impl Drop for Hold {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        mac::give()
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use objc2::{
        rc::{Retained, autoreleasepool},
        runtime::{NSObjectProtocol, ProtocolObject},
    };
    use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSString};
    use std::sync::{Mutex, MutexGuard};

    /// The activity NSProcessInfo hands out, ended with it again.
    pub struct Token(pub Retained<ProtocolObject<dyn NSObjectProtocol>>);

    // SAFETY: the token is only handed back to NSProcessInfo, which is thread-safe, and released
    unsafe impl Send for Token {}

    /// The holds, and the activity while there are any.
    static ON: Mutex<(usize, Option<Token>)> = Mutex::new((0, None));

    pub fn on() -> MutexGuard<'static, (usize, Option<Token>)> {
        ON.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn take() {
        let mut g = on();
        g.0 += 1;
        if g.1.is_none() {
            let r = NSString::from_str("x5desk: an export, analysis, live output or download runs");
            let o = NSActivityOptions::UserInitiatedAllowingIdleSystemSleep;
            // a thread of ours has no pool for what the calls autorelease
            g.1 = Some(Token(autoreleasepool(|_| {
                NSProcessInfo::processInfo().beginActivityWithOptions_reason(o, &r)
            })));
        }
    }

    pub fn give() {
        let mut g = on();
        g.0 -= 1;
        if g.0 == 0
            && let Some(t) = g.1.take()
        {
            // SAFETY: the activity beginActivity returned
            autoreleasepool(|_| unsafe { NSProcessInfo::processInfo().endActivity(&t.0) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "macos")]
    use objc2::rc::Retained;

    /// Holds share one activity, which lives while any does (other tests take holds meanwhile: only what holds while
    /// this one lives is certain).
    #[test]
    fn holds() {
        let a = hold();
        #[cfg(target_os = "macos")]
        let t = {
            let g = mac::on();
            assert!(g.0 >= 1);
            Retained::as_ptr(&g.1.as_ref().expect("an activity").0)
        };
        let b = hold();
        drop(a);
        #[cfg(target_os = "macos")]
        {
            let g = mac::on();
            let u = Retained::as_ptr(&g.1.as_ref().expect("still the activity").0);
            assert_eq!(u, t, "a second activity");
        }
        drop(b);
    }
}
