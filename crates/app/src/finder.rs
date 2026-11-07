//! Files macOS asks the app to open (macOS only): a double-click or Open With in Finder, a drop on the Dock icon,
//! `open <file>`. They come as kAEOpenDocuments Apple events, which winit 0.30 leaves to AppKit's own handler, and that
//! has nobody to give them to (winit's application delegate has no `application:openURLs:`, the app no NSDocument).
//! [`watch`], from main() before eframe runs, puts a handler of ours in its place at the one point that works:
//! `-[NSApplication finishLaunching]` installs AppKit's handlers, posts NSApplicationWillFinishLaunchingNotification,
//! dispatches the launch's event (`oapp`, or `odoc` with the files) and only then sends `applicationDidFinishLaunching:`,
//! where winit starts and eframe calls the app creator. One installed before is replaced by AppKit's, one installed in
//! the app creator misses the launch's files (both tried). The handler queues the paths and wakes the UI, which takes
//! them each frame with the dropped files ([`take`]); the launch's wait for its first frame.
//!
//! Safe: Apple events arrive on the main thread from the event loop, between winit's callbacks, not inside an egui
//! pass; the handler takes the queue's lock only to push, then asks egui's context (thread safe) for a repaint, and
//! cannot panic. Only `odoc` is replaced: launching, reopening (a click on the Dock icon) and quitting stay AppKit's,
//! and AppKit still sends `applicationDidFinishLaunching:` after a launch `odoc` of ours (tried). NSAppleEventManager's
//! `setEventHandler:…` is the documented way to override an AppKit handler (from `applicationWillFinishLaunching:`,
//! which winit's delegate lacks: its notification is the same point), and winit's own class stays untouched.

use eframe::egui;
use objc2::{
    MainThreadMarker, MainThreadOnly, define_class, msg_send, rc::Retained, runtime::NSObject, sel,
};
use objc2_app_kit::NSApplicationWillFinishLaunchingNotification;
use objc2_foundation::{
    NSAppleEventDescriptor, NSAppleEventManager, NSNotification, NSNotificationCenter,
};
use std::{
    path::PathBuf,
    sync::{Mutex, MutexGuard, PoisonError},
};

/// The files asked for that the UI has not taken yet, and its context once it is up.
static Q: Mutex<(Vec<PathBuf>, Option<egui::Context>)> = Mutex::new((Vec::new(), None));

fn q() -> MutexGuard<'static, (Vec<PathBuf>, Option<egui::Context>)> {
    Q.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A FourCharCode (event class, event id, keyword).
const fn code(c: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*c)
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements, and Handler implements no Drop
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "X5deskOpenDocuments"]
    struct Handler;

    impl Handler {
        /// NSApplicationWillFinishLaunchingNotification: AppKit's handlers are in, the launch's event comes next.
        #[unsafe(method(launching:))]
        fn launching(&self, _: &NSNotification) {
            let m = NSAppleEventManager::sharedAppleEventManager();
            // SAFETY: `open:reply:` is ours, with the arguments the manager passes; the handler lives for good
            let _: () = unsafe {
                msg_send![&m, setEventHandler: self, andSelector: sel!(open:reply:),
                    forEventClass: code(b"aevt"), andEventID: code(b"odoc")]
            };
        }

        /// kAEOpenDocuments
        #[unsafe(method(open:reply:))]
        fn open(&self, ev: &NSAppleEventDescriptor, _reply: &NSAppleEventDescriptor) {
            push(files(ev))
        }
    }
);

/// Watches for the open-documents requests; main thread, before eframe runs.
pub fn watch() {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    // SAFETY: NSObject's init
    let h: Retained<Handler> = unsafe { msg_send![Handler::alloc(mtm), init] };
    // SAFETY: `launching:` is the handler's, taking the notification; the name is AppKit's constant
    unsafe {
        NSNotificationCenter::defaultCenter().addObserver_selector_name_object(
            &h,
            sel!(launching:),
            Some(NSApplicationWillFinishLaunchingNotification),
            None,
        )
    };
    // neither the center nor the event manager keeps it
    std::mem::forget(h);
}

/// The UI's context, woken when a request comes.
pub fn wake(ctx: &egui::Context) {
    q().1 = Some(ctx.clone());
}

/// The files asked for since the last call.
pub fn take() -> Vec<PathBuf> {
    std::mem::take(&mut q().0)
}

/// The event's direct object: a list of file URLs, or one.
fn files(ev: &NSAppleEventDescriptor) -> Vec<PathBuf> {
    // SAFETY: paramDescriptorForKeyword: takes an AEKeyword, returns a descriptor or nil
    let d: Option<Retained<NSAppleEventDescriptor>> =
        unsafe { msg_send![ev, paramDescriptorForKeyword: code(b"----")] };
    let Some(d) = d else { return vec![] };
    let v = match d.numberOfItems() {
        // not a list
        0 => vec![d],
        n => (1..=n).filter_map(|i| d.descriptorAtIndex(i)).collect(),
    };
    // the URL's path as sent (its file system representation decomposes "ä"; the Library and the editor canonicalise
    // either way, which gives the name as stored)
    v.iter()
        .filter_map(|d| Some(PathBuf::from(d.fileURLValue()?.path()?.to_string())))
        .collect()
}

fn push(v: Vec<PathBuf>) {
    if v.is_empty() {
        return;
    }
    let ctx = {
        let mut g = q();
        g.0.extend(v);
        g.1.clone()
    };
    if let Some(c) = ctx {
        // a minimised window runs no UI to take them: back, as a document app shows what it opens
        c.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
        c.request_repaint();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2::ClassType;
    use objc2_foundation::NSURL;
    use std::path::Path;

    /// An `odoc` event with that direct object.
    fn odoc(d: &NSAppleEventDescriptor) -> Retained<NSAppleEventDescriptor> {
        let none: Option<&NSAppleEventDescriptor> = None;
        // SAFETY: the class method and setter with their argument types (AEEventClass, AEEventID, the target,
        // AEReturnID, AETransactionID; the descriptor and its AEKeyword)
        unsafe {
            let e: Retained<NSAppleEventDescriptor> = msg_send![
                NSAppleEventDescriptor::class(),
                appleEventWithEventClass: code(b"aevt"), eventID: code(b"odoc"),
                targetDescriptor: none, returnID: -1i16, transactionID: 0i32
            ];
            let _: () = msg_send![&e, setParamDescriptor: d, forKeyword: code(b"----")];
            e
        }
    }

    fn url(p: &str) -> Retained<NSAppleEventDescriptor> {
        NSAppleEventDescriptor::descriptorWithFileURL(&NSURL::from_file_path(p).unwrap())
    }

    /// A list of files (LaunchServices), one file (AppleScript's `open`), nothing that is a file; queued until taken.
    #[test]
    fn odoc_files() {
        let l = NSAppleEventDescriptor::listDescriptor();
        for p in ["/v/VID_1_00_001.insv", "/v/ä b.mp4"] {
            l.insertDescriptor_atIndex(&url(p), 0);
        }
        let two = files(&odoc(&l));
        assert_eq!(
            two,
            [Path::new("/v/VID_1_00_001.insv"), Path::new("/v/ä b.mp4")]
        );
        assert_eq!(
            files(&odoc(&url("/v/IMG.insp"))),
            [Path::new("/v/IMG.insp")]
        );
        assert!(files(&odoc(&NSAppleEventDescriptor::nullDescriptor())).is_empty());
        push(two.clone());
        assert_eq!(take(), two);
        assert!(take().is_empty());
    }
}
