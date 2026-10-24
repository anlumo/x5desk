// Small passes after `stitch.wgsl`'s statistics and after RAFT, and the YUV conversion for encoders; nothing here
// samples the lenses.

// Colour matching: per lens and channel c' = g·c + o so both lenses reach the band's mean and the geometric mean of
// their spreads; fits are smoothed over time (U.y = weight of the new fit, the first fit is taken as is).
@group(0) @binding(0) var<storage, read> S: array<vec4f>;          // per group: Σc0|n, Σc0², Σc1, Σc1²
@group(0) @binding(1) var<storage, read_write> col: array<vec4f>;  // gain 0, offset 0, gain 1, offset 1, fits
@group(0) @binding(2) var<uniform> U: vec4f;                        // x: groups, y: smoothing

@compute @workgroup_size(1)
fn fold() {
    var s = array<vec4f, 4>();
    for (var g = 0u; g < u32(U.x); g++) {
        for (var k = 0u; k < 4u; k++) { s[k] += S[4u * g + k]; }
    }
    let n = s[0].w;
    if n < 256.0 { return; }  // too little usable overlap (dark, clipped): keep the last fit
    let m0 = s[0].xyz / n;
    let m1 = s[2].xyz / n;
    let d0 = sqrt(max(s[1].xyz / n - m0 * m0, vec3f(1e-6)));
    let d1 = sqrt(max(s[3].xyz / n - m1 * m1, vec3f(1e-6)));
    let m = (m0 + m1) / 2.0;
    let g0 = clamp(sqrt(d0 * d1) / d0, vec3f(0.5), vec3f(2.0));
    let g1 = clamp(sqrt(d0 * d1) / d1, vec3f(0.5), vec3f(2.0));
    let fit = array(g0, m - g0 * m0, g1, m - g1 * m1);
    let t = select(U.y, 1.0, col[4].x == 0.0);
    for (var k = 0u; k < 4u; k++) { col[k] = vec4f(mix(col[k].xyz, fit[k], t), 0.0); }
    col[4].x += 1.0;
}

// RAFT region flows -> belt flows: extended over the whole belt (columns outside the region repeat its edge
// columns, so the warp zone never runs off the belt), nadir wrap cross-faded (blendFlow 2°), smoothed over time.
struct Fin { st: vec4u, cols: u32, rows: u32, c0: u32, w: u32, ema: f32, first: u32 }

@group(0) @binding(3) var<storage, read> R: array<f32>;            // [2 (L2R, R2L), 2 (x, y), H, W], strides st
@group(0) @binding(4) var<storage, read_write> Fl: array<vec2f>;   // belt flows: L2R, then R2L
@group(0) @binding(5) var<uniform> V: Fin;

fn reg(k: u32, y: u32, c: u32) -> vec2f {
    let i = k * V.st.x + y * V.st.z + (clamp(c, V.c0, V.c0 + V.w - 1u) - V.c0) * V.st.w;
    return vec2f(R[i], R[i + V.st.y]);
}

@compute @workgroup_size(8, 8)
fn fin(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= V.cols || id.y >= V.rows { return; }
    let k = id.z;
    let d = 400.0 / f32(V.rows - 1u);
    let span = u32(round(360.0 / d));
    // rows at a ∈ [−181°, −179°] and their copies 360° further along the 400° belt agree
    var v = reg(k, id.y, id.x);
    let a = -200.0 + f32(id.y) * d;
    if a >= -181.0 && a <= -179.0 && id.y + span < V.rows {
        v = mix(reg(k, id.y + span, id.x), v, clamp((a + 181.0) / 2.0, 0.0, 1.0));
    } else if id.y >= span {
        let b = a - f32(span) * d;
        if b >= -181.0 && b <= -179.0 {
            v = mix(v, reg(k, id.y - span, id.x), clamp((b + 181.0) / 2.0, 0.0, 1.0));
        }
    }
    let i = (k * V.rows + id.y) * V.cols + id.x;
    Fl[i] = select(mix(Fl[i], v, V.ema), v, V.first != 0u);
}

// R'G'B' picture -> YUV 4:2:0 (lib.rs `Fmt`): BT.709 limited range, NV12 (bytes) or P010 (u16, 10 bits at the top);
// Y rows, then interleaved CbCr rows, `w` rounded up to 4 samples per row (the columns past `w` repeat or wrap the
// edge; the encoder copies `w`). Chroma = R'G'B' filtered at its site, then converted
// (Cb, Cr are linear in R'G'B'): horizontally [¼ ½ ¼] around the even column (left, top-left) or [½ ½] (centre),
// vertically [½ ½] between the rows (left, centre) or [¼ ½ ¼] around the even row (top-left). One invocation per
// 4 × 2 pixels.
struct Cv { w: u32, h: u32, deep: u32, siting: u32, wrap: u32 }

@group(0) @binding(6) var src: texture_2d<f32>;
@group(0) @binding(7) var<storage, read_write> Y: array<u32>;
@group(0) @binding(8) var<uniform> C: Cv;

fn px(x: i32, y: i32) -> vec3f {
    let w = i32(C.w);
    let c = select(clamp(x, 0, w - 1), (x + w) % w, C.wrap != 0u);
    return clamp(textureLoad(src, vec2i(c, clamp(y, 0, i32(C.h) - 1)), 0).rgb, vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(8, 8)
fn yuv(@builtin(global_invocation_id) id: vec3u) {
    let o = id.xy * vec2u(4u, 2u);
    if any(o >= vec2u(C.w, C.h)) { return; }
    let k = vec3f(0.2126, 0.7152, 0.0722);
    let d = C.deep != 0u;
    // code = offset + scale · value; P010 keeps the 10 bits at the top of each u16
    let s = select(vec3f(16.0, 219.0, 224.0), vec3f(64.0, 876.0, 896.0), d);
    let c0 = select(128.0, 512.0, d);
    let q = vec2i(o);
    var y = array<vec4u, 2>();
    for (var r = 0; r < 2; r++) {
        for (var c = 0; c < 4; c++) {
            y[r][c] = u32(s.x + s.y * dot(px(q.x + c, q.y + r), k) + 0.5);
        }
    }
    let hw = select(vec3f(0.25, 0.5, 0.25), vec3f(0.0, 0.5, 0.5), C.siting == 1u);
    let vw = select(vec3f(0.0, 0.5, 0.5), vec3f(0.25, 0.5, 0.25), C.siting == 2u);
    var uv = vec4u();
    for (var i = 0; i < 2; i++) {
        var v = vec3f(0.0);
        for (var j = 0; j < 9; j++) {
            v += hw[j % 3] * vw[j / 3] * px(q.x + 2 * i + j % 3 - 1, q.y + j / 3 - 1);
        }
        let l = dot(v, k);
        uv[2 * i] = u32(c0 + s.z * (v.b - l) / 1.8556 + 0.5);
        uv[2 * i + 1] = u32(c0 + s.z * (v.r - l) / 1.5748 + 0.5);
    }
    // rows of 4 bytes (NV12) or 2 u16 (P010) per word; the chroma rows follow the h luma rows at the same pitch
    let n4 = (C.w + 3u) / 4u;
    if d {
        let n = 2u * n4;
        for (var r = 0u; r < 2u; r++) {
            let i = (o.y + r) * n + 2u * id.x;
            Y[i] = (y[r].x << 6u) | (y[r].y << 22u);
            Y[i + 1u] = (y[r].z << 6u) | (y[r].w << 22u);
        }
        let i = (C.h + id.y) * n + 2u * id.x;
        Y[i] = (uv.x << 6u) | (uv.y << 22u);
        Y[i + 1u] = (uv.z << 6u) | (uv.w << 22u);
    } else {
        for (var r = 0u; r < 2u; r++) {
            Y[(o.y + r) * n4 + id.x] = y[r].x | (y[r].y << 8u) | (y[r].z << 16u) | (y[r].w << 24u);
        }
        Y[(C.h + id.y) * n4 + id.x] = uv.x | (uv.y << 8u) | (uv.z << 16u) | (uv.w << 24u);
    }
}
