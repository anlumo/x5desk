#!/bin/bash
# The Linux release build, run as root in a debian:13 container with the checkout at /src (release.yml; README, "Build
# from source", has the same command for a local run). Debian 13 has glibc 2.41 and PipeWire 1.4, so the binary runs
# where glibc is 2.41 or newer and PipeWire 1.2 or newer: the PipeWire camera calls `pw_stream_return_buffer`, which
# older libraries lack (Ubuntu 24.04's 1.0 fails at the link).
# In: $RUST (toolchain), $DIR (the archive's name), $CARGO_HOME; out: /src/$DIR.tar.gz and its .sha256.
set -euo pipefail
cd /src
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q --no-install-recommends ca-certificates curl python3 \
  build-essential pkg-config clang libclang-dev libasound2-dev libdbus-1-dev libpipewire-0.3-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "$RUST"
export PATH="$CARGO_HOME/bin:$PATH"

cargo build --release --locked -p app
bin=${CARGO_TARGET_DIR:-target}/release/x5desk
if ldd "$bin" | grep "not found"; then exit 1; fi
"$bin" mcp --help > /tmp/usage.txt
grep -q "usage: x5desk mcp" /tmp/usage.txt
ls -l "$bin"

# The licence texts of the bundled models and of every crate the binary links, as each crate ships them; INDEX.txt
# lists them all with licence and source, also those whose package holds no licence file.
rm -rf "$DIR"
mkdir -p "$DIR/licenses/models" "$DIR/licenses/crates"
cp "$bin" README.md LICENSE "$DIR/"
cp crates/*/models/LICENSE-* crates/media/src/aacenc/NOTICE-vo-aacenc "$DIR/licenses/models/"
python3 - "$DIR/licenses/crates" <<'EOF'
import json, pathlib, shutil, subprocess, sys

out = pathlib.Path(sys.argv[1])
m = json.loads(subprocess.check_output(
    ["cargo", "metadata", "--locked", "--format-version", "1", "--filter-platform", "x86_64-unknown-linux-gnu"]))
pk = {p["id"]: p for p in m["packages"]}
deps = {n["id"]: n["deps"] for n in m["resolve"]["nodes"]}
seen, todo = set(), [next(p["id"] for p in m["packages"] if p["name"] == "app" and p["source"] is None)]
while todo:
    i = todo.pop()
    if i not in seen:
        seen.add(i)
        todo += [d["pkg"] for d in deps[i] if any(k["kind"] != "dev" for k in d["dep_kinds"])]
rows = []
for p in sorted((pk[i] for i in seen if pk[i]["source"]), key=lambda p: (p["name"], p["version"])):
    fs = [f for f in pathlib.Path(p["manifest_path"]).parent.iterdir() if f.is_file()
          and f.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "NOTICE", "UNLICENSE"))]
    if fs:
        d = out / f'{p["name"]}-{p["version"]}'
        d.mkdir(parents=True)
        for f in fs:
            shutil.copy(f, d / f.name)
    rows.append(f'{p["name"]} {p["version"]} | {p["license"] or "see its files"} | {p["repository"] or ""} | {len(fs)} file(s)')
(out / "INDEX.txt").write_text("\n".join(rows) + "\n")
EOF
tar -czf "$DIR.tar.gz" "$DIR"
sha256sum "$DIR.tar.gz" | tee "$DIR.tar.gz.sha256"
ls -l "$DIR.tar.gz"
