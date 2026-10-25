//! `Gpu::new` and drop on many threads at once, as parallel tests do (the app itself opens one `Gpu` at startup).
//! `X5_GPU_ITERS` sets the iterations per thread.
use video::Gpu;

#[test]
fn concurrent() {
    let n = std::env::var("X5_GPU_ITERS").map_or(32, |s| s.parse().unwrap());
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| (0..n).for_each(|_| drop(Gpu::new().unwrap())));
        }
    });
}
