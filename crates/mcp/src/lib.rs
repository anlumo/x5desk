//! x5desk as an MCP server (rmcp, stdio framing): agents read the camera's state and album, download into an agent
//! folder, and connect or release the camera, in tiers the user enables once (`config`). The server's core: the
//! headless host (`host`: the camera worker of `cam` without hotplug connects or clock, the download queue and the jobs
//! of `job`, kept for the next host), its tools over rmcp (`server`), the agent folder and read roots (`paths`), the
//! scrubber every output passes (`scrub`), the call log (`log`) and the simulated camera of `--sim` (`sim`). The GPU
//! tools (stills and exports) go through a backend the app hands in (`render`: plain data, no GPU here), their records
//! kept beside the jobs (`renders`). The process side builds on [`Host`]: the command line ([`main`], `cli`), the
//! stdout guard (`out`), the per-user endpoint (`endpoint`) and the front that pumps stdio's lines to the host of this
//! process or another's (`front`). One host per user, [`Host::serve`] once per client connection (stdio, an endpoint
//! connection), [`Host::shutdown`] at the end. Linux and macOS; Windows type-checks.
//!
//! ```no_run
//! # async fn run() -> Result<(), String> {
//! let host = mcp::Host::new(mcp::Opts { sim: Some(Default::default()), ..Default::default() }).await?;
//! // a client's two directions: stdin and the stdout guard's writer, or a socket's halves
//! let (_client, ours) = tokio::io::duplex(1 << 16);
//! let (r, w) = tokio::io::split(ours);
//! host.serve(r, w, "stdio", None).await?;
//! host.shutdown().await;
//! # Ok(()) }
//! ```

pub mod cli;
pub mod config;
pub mod endpoint;
pub mod front;
pub mod host;
pub mod job;
pub mod log;
pub mod out;
pub mod paths;
pub mod render;
pub mod renders;
pub mod scrub;
pub mod server;
pub mod sim;

pub use cli::main;
pub use config::{Tier, Tiers};
pub use host::{Host, Opts};
