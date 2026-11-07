//! App Nap (macOS): once every window is hidden or covered and no sound plays, the system coalesces the process's
//! timers and lowers its threads' priority. Long work holds a [`Hold`] while it runs (exports and frame grabs, Shot
//! Lab and AI passes, live outputs, downloads and card imports): the first begins an NSProcessInfo activity (user
//! initiated, idle system sleep allowed), the last to go ends it. An agent's renders hold a [`Keep`] instead, often
//! with no user at the keyboard: an activity of its own that also keeps idle system sleep off (the display may
//! sleep), so no hold's activity can weaken it. Elsewhere both do nothing (no sleep lock on Linux yet).

/// App Nap stays off while one lives.
#[must_use = "App Nap is off only while the hold lives"]
pub struct Hold(());

pub fn hold() -> Hold {
    #[cfg(target_os = "macos")]
    mac::HOLD.take();
    Hold(())
}

impl Drop for Hold {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        mac::HOLD.give()
    }
}

/// App Nap and idle system sleep stay off while one lives.
#[must_use = "idle sleep is off only while the keep lives"]
pub struct Keep(());

pub fn keep() -> Keep {
    #[cfg(target_os = "macos")]
    mac::KEEP.take();
    Keep(())
}

impl Drop for Keep {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        mac::KEEP.give()
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

    /// One kind of activity: its holders, and the activity while there are any.
    pub struct Act {
        on: Mutex<(usize, Option<Token>)>,
        why: &'static str,
        opts: NSActivityOptions,
    }

    pub static HOLD: Act = Act {
        on: Mutex::new((0, None)),
        why: "x5desk: an export, analysis, live output or download runs",
        opts: NSActivityOptions::UserInitiatedAllowingIdleSystemSleep,
    };

    pub static KEEP: Act = Act {
        on: Mutex::new((0, None)),
        why: "x5desk: an agent's render runs",
        opts: NSActivityOptions::UserInitiated,
    };

    impl Act {
        pub fn on(&self) -> MutexGuard<'_, (usize, Option<Token>)> {
            self.on.lock().unwrap_or_else(|e| e.into_inner())
        }

        pub fn take(&self) {
            let mut g = self.on();
            g.0 += 1;
            if g.1.is_none() {
                let r = NSString::from_str(self.why);
                // a thread of ours has no pool for what the calls autorelease
                g.1 = Some(Token(autoreleasepool(|_| {
                    NSProcessInfo::processInfo().beginActivityWithOptions_reason(self.opts, &r)
                })));
            }
        }

        pub fn give(&self) {
            let mut g = self.on();
            g.0 -= 1;
            if g.0 == 0
                && let Some(t) = g.1.take()
            {
                // SAFETY: the activity beginActivity returned
                autoreleasepool(|_| unsafe { NSProcessInfo::processInfo().endActivity(&t.0) });
            }
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
            let g = mac::HOLD.on();
            assert!(g.0 >= 1);
            Retained::as_ptr(&g.1.as_ref().expect("an activity").0)
        };
        let b = hold();
        drop(a);
        #[cfg(target_os = "macos")]
        {
            let g = mac::HOLD.on();
            let u = Retained::as_ptr(&g.1.as_ref().expect("still the activity").0);
            assert_eq!(u, t, "a second activity");
        }
        drop(b);
    }

    /// Keeps have an activity and a count of their own, apart from the holds' (only this test keeps).
    #[test]
    fn keeps() {
        let h = hold();
        let k = keep();
        #[cfg(target_os = "macos")]
        {
            let (g, n) = (mac::HOLD.on(), mac::KEEP.on());
            assert_eq!(n.0, 1);
            let (a, b) = (
                &g.1.as_ref().expect("a hold's").0,
                &n.1.as_ref().expect("a keep's").0,
            );
            assert_ne!(
                Retained::as_ptr(a),
                Retained::as_ptr(b),
                "one activity for both"
            );
        }
        let l = keep();
        drop(h);
        drop(k);
        #[cfg(target_os = "macos")]
        assert!(mac::KEEP.on().1.is_some(), "a hold's end ended the keep");
        drop(l);
        #[cfg(target_os = "macos")]
        {
            let n = mac::KEEP.on();
            assert_eq!(n.0, 0);
            assert!(n.1.is_none(), "the keep outlived its holders");
        }
    }
}
