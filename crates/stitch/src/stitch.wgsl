// Dual fisheye from two NV12 / P010 lens pictures (whole textures or rects in them): equirect stitch, seam belt for
// RAFT, colour statistics, preview; preview from an equirect.
// P (offset.rs `pack`, lib.rs): [0..2] out size, [2] half blend angle (deg), [3] half blend angle near the nadir,
//   [4] flow on, [5] belt cols (across the seam), [6] belt rows (along), [7] A0 (deg), [8] deg/row,
//   [9] belt px/rad across, [10] belt alpha half-width (deg), [11] flow-region oversampling,
//   [12] flow-region first belt column, [13] flow-region width (belt columns),
//   [14] seam low band block side (lens px), [15..17] its blocks across, down per lens (0: none, with flow)
//   per lens at 32 + 32k: [0..9] world->camera rotation, [9] model (0 poly, 1 omni radtan, 2 omni radtan v6),
//   [10..12] principal point (lens px), [12..14] focal (lens px), [14] xi, [15] cos FOV/2 - 0.01,
//   [16] image-circle radius at FOV/2 (lens px), [17] mask feather per px, [18..31] distortion coefficients
// Frames: output ray -> pose -> camera body `w` (calibration world: Z up, lens 0 axis -X) -> lens camera.
// The seam belt lives in the template frame S = (-w.x, -w.y, w.z) (lens 0 on +X_S).

// lens k: NV12 / P010 planes; r: its rect in them (x, y, w, h luma px; a side-by-side half or a decoder's display
// window); k: full-range Y'CbCr = k.x · plane value, Cb Cr less k.y (lib.rs `scale`)
struct Lens { r: vec4u, k: vec4f }
@group(0) @binding(0) var y0: texture_2d<f32>;
@group(0) @binding(1) var u0: texture_2d<f32>;
@group(0) @binding(2) var<uniform> r0: Lens;
@group(0) @binding(3) var s0: sampler;
@group(1) @binding(0) var y1: texture_2d<f32>;
@group(1) @binding(1) var u1: texture_2d<f32>;
@group(1) @binding(2) var<uniform> r1: Lens;
@group(1) @binding(3) var s1: sampler;
// preview `fe`: an RGBA equirect, sampled with s0 (it repeats across longitude)
@group(0) @binding(4) var pano: texture_2d<f32>;

// r: output world -> body; rs: rolling shutter per lens, body rotation vector over the readout, w = on
struct Pose { r: mat3x3f, rs0: vec4f, rs1: vec4f }

@group(2) @binding(0) var<storage, read> P: array<f32>;
@group(2) @binding(1) var<uniform> pose: Pose;
@group(2) @binding(2) var<storage, read> col: array<vec4f>;  // gain 0, offset 0, gain 1, offset 1 (xyz), fits
@group(2) @binding(3) var<storage, read> lo: array<vec4f>;   // seam low band (`low`): lens 0's blocks, then lens 1's

@group(3) @binding(0) var dst: texture_storage_2d<rgba8unorm, write>;  // or rgba16float (lib.rs `Ctx::new`)
@group(3) @binding(2) var<storage, read> F: array<vec2f>;        // belt flows: L2R, then R2L
@group(3) @binding(3) var<storage, read_write> B: array<f32>;    // RAFT inputs: lens 0 [3, H, W], then lens 1
@group(3) @binding(4) var<storage, read_write> S: array<vec4f>;  // colour sums per workgroup: Σc0|n, Σc0², Σc1, Σc1²
@group(3) @binding(6) var<storage, read_write> Lo: array<vec4f>; // `lo` as `low` writes it

// r: view camera (x right, y up, z forward) -> output world; rect: viewport px; f: plane half-width; d: Panini
// distance or planet eye distance (0 = rectilinear)
struct View { r: mat3x3f, rect: vec4f, mode: u32, f: f32, d: f32, srgb: u32 }
@group(3) @binding(5) var<uniform> view: View;

const PI = 3.14159265358979;
const L0 = 32u;

fn m3(b: u32) -> mat3x3f {  // row-major in the buffer
    return transpose(mat3x3f(P[b], P[b + 1], P[b + 2], P[b + 3], P[b + 4], P[b + 5], P[b + 6], P[b + 7], P[b + 8]));
}

// OmniProjection validity: Z/|P| >= cos(FOV/2) - 0.01
fn inside(b: u32, p: vec3f) -> bool { return p.z / length(p) >= P[b + 15]; }

// camera ray -> (lens px x, y, mask weight)
fn project(b: u32, p: vec3f) -> vec3f {
    let n = length(p);
    if !inside(b, p) { return vec3f(0.0); }
    let c = vec2f(P[b + 10], P[b + 11]);
    var xy: vec2f;
    if P[b + 9] == 0.0 {  // v1/v2: r = θ·poly(θ)
        let th = atan2(length(p.xy), p.z);
        let r = th * (P[b + 18] + th * (P[b + 19] + th * (P[b + 20] + th * P[b + 21])));
        xy = c + P[b + 12] * r * normalize(p.xy);
    } else {  // v3/v6: Mei unified model + radial-tangential distortion
        let m = p.xy / (p.z + P[b + 14] * n);
        let r2 = dot(m, m);
        var d: vec2f;
        if P[b + 9] == 1.0 {  // k1 k2 k3 p1 p2
            let rad = 1.0 + r2 * (P[b + 18] + r2 * (P[b + 19] + r2 * P[b + 20]));
            let p1 = P[b + 21];
            let p2 = P[b + 22];
            d = m * rad + vec2f(2.0 * p1 * m.x * m.y + p2 * (r2 + 2.0 * m.x * m.x),
                                p1 * (r2 + 2.0 * m.y * m.y) + 2.0 * p2 * m.x * m.y);
        } else {  // v6, 13 coefficients
            let rad = 1.0 + r2 * (P[b + 18] + r2 * (P[b + 19] + r2 * (P[b + 20] + r2 * (P[b + 21] + r2 * P[b + 22]))));
            let ta = P[b + 23] + P[b + 25] * r2;
            let tb = P[b + 24] + P[b + 26] * r2;
            d = m * rad + vec2f((2.0 * m.x * m.x + r2) * ta + 2.0 * m.x * m.y * tb + P[b + 27] * r2 + P[b + 29] * r2 * r2,
                                (2.0 * m.y * m.y + r2) * tb + 2.0 * m.x * m.y * ta + P[b + 28] * r2 + P[b + 30] * r2 * r2);
        }
        xy = c + vec2f(P[b + 12], P[b + 13]) * d;
    }
    // calcFisheyeMaskv: linear feather inward from the image-circle edge
    return vec3f(xy, clamp((P[b + 16] - distance(xy, c)) * P[b + 17], 0.0, 1.0));
}

// Rodrigues: v rotated by the rotation vector t
fn rot(t: vec3f, v: vec3f) -> vec3f {
    let a = length(t);
    if a < 1e-9 { return v; }
    let k = t / a;
    return v * cos(a) + cross(k, v) * sin(a) + k * dot(k, v) * (1.0 - cos(a));
}

// body ray -> lens k (lens px, mask). Rolling shutter: the row found first was exposed (v - 0.5)·readout from the
// frame time, when the body had turned by (v - 0.5)·rs; one fixed-point step re-projects the ray seen then.
fn look(k: u32, w: vec3f) -> vec3f {
    let b = L0 + 32u * k;
    let m = m3(b);
    let q = project(b, m * w);
    let rs = select(pose.rs0, pose.rs1, k == 1u);
    if rs.w == 0.0 || q.z == 0.0 { return q; }
    let h = f32(select(r0.r.w, r1.r.w, k == 1u));
    return project(b, m * rot(-(q.y / h - 0.5) * rs.xyz, w));
}

// Bilinear at px q (texel centres at k + 0.5) of the n-texel rect at o, edges clamped, so the rect samples like a
// texture of its own: the 2 × 2 footprint, picked in integers, as the texture uv of its centre (where textureGather
// can't round to a neighbour), and the weights. A rect and a separate texture with its pixels give identical results.
fn foot(t: texture_2d<f32>, o: vec2u, n: vec2u, q: vec2f) -> vec4f {
    let p = q - 0.5;
    let i = clamp(floor(p), vec2f(0.0), vec2f(n) - 2.0);
    return vec4f((i + vec2f(o) + 1.0) / vec2f(textureDimensions(t)), clamp(p - i, vec2f(0.0), vec2f(1.0)));
}

// gathered texels (x: (0, 1), y: (1, 1), z: (1, 0), w: (0, 0)) at weights f
fn lerp4(g: vec4f, f: vec2f) -> f32 { return mix(mix(g.w, g.z, f.x), mix(g.x, g.y, f.x), f.y); }

// full-range BT.709 (X5: yuvj420p, bt709, chroma sited left) at lens px q of lens rect n.r
fn yuv(y: texture_2d<f32>, u: texture_2d<f32>, s: sampler, n: Lens, q: vec2f) -> vec3f {
    let r = n.r;
    let a = foot(y, r.xy, r.zw, q);
    let l = n.k.x * lerp4(textureGather(0, y, s, a.xy), a.zw);
    // left siting: chroma sample i sits on luma column 2i, half a luma texel left of a centred one
    let b = foot(u, r.xy / 2u, r.zw / 2u, (q + vec2f(0.5, 0.0)) / 2.0);
    let c = n.k.x * vec2f(lerp4(textureGather(0, u, s, b.xy), b.zw), lerp4(textureGather(1, u, s, b.xy), b.zw))
        - n.k.y;
    return clamp(vec3f(l + 1.5748 * c.y, l - 0.187324 * c.x - 0.468124 * c.y, l + 1.8556 * c.x), vec3f(0.0), vec3f(1.0));
}

// lens k colour at lens px q, before (raw) and after colour matching
fn raw(k: u32, q: vec3f) -> vec3f {
    if k == 0u { return yuv(y0, u0, s0, r0, q.xy); }
    return yuv(y1, u1, s1, r1, q.xy);
}
fn tex(k: u32, q: vec3f) -> vec3f { return clamp(col[2u * k].xyz * raw(k, q) + col[2u * k + 1u].xyz, vec3f(0.0), vec3f(1.0)); }

// lens k's colour at lens px q blurred over about 3° (`low`'s blocks, bilinear), colour-matched; w = 0 without data
fn soft(k: u32, q: vec3f) -> vec4f {
    let n = vec2u(u32(P[15]), u32(P[16]));
    let p = clamp(q.xy / P[14] - 0.5, vec2f(0.0), vec2f(n - 1u));
    let i = min(vec2u(p), n - 2u);
    let f = p - vec2f(i);
    let o = k * n.x * n.y + i.y * n.x + i.x;
    let v = mix(mix(lo[o], lo[o + 1u], f.x), mix(lo[o + n.x], lo[o + n.x + 1u], f.x), f.y);
    if v.w < 0.05 { return vec4f(0.0); }
    let s = select(r0.k, r1.k, k == 1u);
    let l = s.x * v.x / v.w;
    let c = s.x * v.yz / v.w - s.y;
    let e = clamp(vec3f(l + 1.5748 * c.y, l - 0.187324 * c.x - 0.468124 * c.y, l + 1.8556 * c.x), vec3f(0.0), vec3f(1.0));
    return vec4f(clamp(col[2u * k].xyz * e + col[2u * k + 1u].xyz, vec3f(0.0), vec3f(1.0)), 1.0);
}

// belt (col, row) -> S direction: rows along the seam a = A0 + row·Δ, columns gnomonic across it
fn belt_dir(b: vec2f) -> vec3f {
    let a = radians(P[7] + b.y * P[8]);
    return normalize(vec3f(((P[5] - 1.0) / 2.0 - b.x) / P[9], -sin(a), cos(a)));
}

// S direction -> belt (col, row), rows of the non-duplicated range a ∈ [-180, 180)
fn sphere_belt(s: vec3f) -> vec2f {
    var a = degrees(atan2(-s.y, s.z));
    if a >= 180.0 { a -= 360.0; }
    return vec2f((P[5] - 1.0) / 2.0 - P[9] * s.x / length(s.yz), (a - P[7]) / P[8]);
}

// bilinear, clamped lookup of flow k (0 = L2R, 1 = R2L) at belt position b
fn flow(k: u32, b: vec2f) -> vec2f {
    let size = vec2u(u32(P[5]), u32(P[6]));
    let c = clamp(b, vec2f(0.0), vec2f(size - 1u));
    let i = vec2u(floor(c));
    let j = min(i + 1u, size - 1u);
    let f = c - floor(c);
    let o = k * size.x * size.y;
    let v0 = mix(F[o + i.y * size.x + i.x], F[o + i.y * size.x + j.x], f.x);
    let v1 = mix(F[o + j.y * size.x + i.x], F[o + j.y * size.x + j.x], f.x);
    return mix(v0, v1, f.y);
}

fn body(s: vec3f) -> vec3f { return vec3f(-s.x, -s.y, s.z); }

// seam flow: the body rays lens 0 and lens 1 sample for body ray w
fn warp(w: vec3f) -> array<vec3f, 2> {
    // inside the belt zone lens 0 follows (1-α)·R2L, lens 1 follows α·L2R
    var r = array(w, w);
    let s = body(w);
    let al = clamp(0.5 + degrees(asin(clamp(s.x, -1.0, 1.0))) / (2.0 * P[10]), 0.0, 1.0);
    if al > 0.0 && al < 1.0 {
        let b = sphere_belt(s);
        let hi = vec2f(P[5] - 1.0, P[6] - 1.0);
        let q0 = body(belt_dir(clamp(b + (1.0 - al) * flow(1u, b), vec2f(0.0), hi)));
        let q1 = body(belt_dir(clamp(b + al * flow(0u, b), vec2f(0.0), hi)));
        if inside(L0, m3(L0) * q0) { r[0] = q0; }  // invalid warped position: keep the template lookup
        if inside(L0 + 32u, m3(L0 + 32u) * q1) { r[1] = q1; }
    }
    return r;
}

// stitched colour of body ray w (unit), lens k sampling along wk
fn shade(w: vec3f, w0: vec3f, w1: vec3f) -> vec3f {
    let a = look(0u, w0);
    let b = look(1u, w1);
    let c0 = tex(0u, a);
    let c1 = tex(1u, b);
    // blend: linear ramp ±h around 90° from lens 0's nominal axis,
    // h widening from P[2] to P[3] between polar angles 155° and 170°
    let h = mix(P[2], P[3], clamp((-0.9063 - w.z) / (0.9848 - 0.9063), 0.0, 1.0));
    let t = degrees(acos(clamp(-w.x, -1.0, 1.0)));
    let a0 = select(f32(t <= 90.0), clamp((90.0 + h - t) / (2.0 * h), 0.0, 1.0), h > 0.0);
    // StitchProgram: α = a0·m0 / (a0·m0 + (1−a0)·m1); out = mix(lens 1, lens 0, α)
    let ws = a0 * a.z + (1.0 - a0) * b.z;
    return mix(c1, c0, select(a0, a0 * a.z / ws, ws > 0.0));
}

// `shade`'s α for a ramp ±h
fn share(h: f32, t: f32, m: vec2f) -> f32 {
    let a0 = select(f32(t <= 90.0), clamp((90.0 + h - t) / (2.0 * h), 0.0, 1.0), h > 0.0);
    let ws = a0 * m.x + (1.0 - a0) * m.y;
    return select(a0, a0 * m.x / ws, ws > 0.0);
}

// `shade` without flow, in two bands (golblick): detail across a narrow fade of half-width n (deg), so near content is
// cut instead of doubled; the lenses' blurred difference handed over across `shade`'s ramp, so a brightness step
// between them still fades out. Apart from `shade` so the flow path compiles exactly as before (folding this into
// `shade` moved single pixels of the flow stitch by a level on RADV).
fn seam(w: vec3f, n: f32) -> vec3f {
    let a = look(0u, w);
    let b = look(1u, w);
    let c0 = tex(0u, a);
    let c1 = tex(1u, b);
    let h = mix(P[2], P[3], clamp((-0.9063 - w.z) / (0.9848 - 0.9063), 0.0, 1.0));
    let t = degrees(acos(clamp(-w.x, -1.0, 1.0)));
    let m = vec2f(a.z, b.z);
    let aw = share(h, t, m);
    let an = share(min(n, h), t, m);
    let o = mix(c1, c0, aw);
    if an == aw { return o; }
    let l0 = soft(0u, a);
    let l1 = soft(1u, b);
    if l0.w * l1.w == 0.0 { return o; }
    return clamp(mix(c1, c0, an) + (aw - an) * (l0.xyz - l1.xyz), vec3f(0.0), vec3f(1.0));
}

// Equirect, GL pano convention: row 0 = zenith, centre = lens 0 axis.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = vec2u(u32(P[0]), u32(P[1]));
    if any(id.xy >= size) { return; }
    let th = 2.0 * PI * f32(id.x) / f32(size.x);
    let ph = PI * f32(id.y) / f32(size.y - 1u);
    let g = vec3f(-sin(ph) * sin(th), cos(ph), sin(ph) * cos(th));
    let w = pose.r * vec3f(g.z, g.x, g.y);
    var r = array(w, w);
    if P[4] > 0.0 { r = warp(w); }
    textureStore(dst, id.xy, vec4f(shade(w, r[0], r[1]), 1.0));
}

// `main` without flow (`seam`); the detail fade spans 4 rows, at least 1.5° and at most 3° (golblick)
@compute @workgroup_size(8, 8)
fn tpl(@builtin(global_invocation_id) id: vec3u) {
    let size = vec2u(u32(P[0]), u32(P[1]));
    if any(id.xy >= size) { return; }
    let th = 2.0 * PI * f32(id.x) / f32(size.x);
    let ph = PI * f32(id.y) / f32(size.y - 1u);
    let g = vec3f(-sin(ph) * sin(th), cos(ph), sin(ph) * cos(th));
    textureStore(dst, id.xy, vec4f(seam(pose.r * vec3f(g.z, g.x, g.y), clamp(360.0 / (P[1] - 1.0), 0.75, 1.5)), 1.0));
}

// 8-bit luma as the RGBA belts were (0 outside the FOV)
fn luma(k: u32, q: vec3f) -> f32 {
    if q.z <= 0.0 { return 0.0; }
    return floor(dot(tex(k, q), vec3f(0.2126, 0.7152, 0.0722)) * 255.0 + 0.5);
}

// RAFT inputs (raft.rs) over the belt's flow region: s × s sub-samples per pixel at belt coordinates
// first column + (col·s + i)/s, no half-pixel shift, box-averaged (point-sampled belts alias and
// the flow jitters); a lens's pixels that lack a valid sub-sample take the other lens's value (RAFT has no mask
// input, and black areas pull the flow); scaled to [-1, 1], grey in 3 channels.
@compute @workgroup_size(8, 8)
fn belt(@builtin(global_invocation_id) id: vec3u) {
    let size = vec2u(u32(P[13]), u32(P[6]));
    if any(id.xy >= size) { return; }
    let s = u32(P[11]);
    var l = vec2f(0.0);
    var ok = vec2<bool>(true);
    for (var j = 0u; j < s * s; j++) {
        let w = body(belt_dir(vec2f(P[12], 0.0) + vec2f(id.xy * s + vec2u(j % s, j / s)) / P[11]));
        let a = look(0u, w);
        let b = look(1u, w);
        l += vec2f(luma(0u, a), luma(1u, b));
        ok &= vec2(a.z > 0.0, b.z > 0.0);
    }
    let v = select(l, l.yx, !ok & ok.yx) / (f32(s * s) * 127.5) - 1.0;
    let n = size.x * size.y;
    let k = id.y * size.x + id.x;
    for (var c = 0u; c < 3u; c++) {
        B[c * n + k] = v.x;
        B[(3u + c) * n + k] = v.y;
    }
}

// Colour statistics over the overlap band: S frame, a along the seam (0 = zenith) over ±150° (the nadir holds the
// camera's own body and the stick), e across over ±5°; 64 groups × 256 samples. Samples count where both lenses are
// fully inside their masks and neither is clipped or black.
var<workgroup> acc: array<array<vec4f, 4>, 256>;

@compute @workgroup_size(256)
fn stats(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32,
         @builtin(workgroup_id) wg: vec3u) {
    let a = radians(300.0 * ((f32(id.x / 16u) + 0.5) / 1024.0) - 150.0);
    let e = radians(10.0 * ((f32(id.x % 16u) + 0.5) / 16.0) - 5.0);
    let w = body(vec3f(sin(e), -sin(a) * cos(e), cos(a) * cos(e)));
    let p = look(0u, w);
    let q = look(1u, w);
    let c0 = raw(0u, p);
    let c1 = raw(1u, q);
    let n = f32(p.z >= 1.0 && q.z >= 1.0 && all(max(c0, c1) < vec3f(0.98)) && all(min(c0, c1) > vec3f(0.02)));
    acc[li] = array(vec4f(c0 * n, n), vec4f(c0 * c0 * n, 0.0), vec4f(c1 * n, 0.0), vec4f(c1 * c1 * n, 0.0));
    for (var k = 128u; k > 0u; k >>= 1u) {
        workgroupBarrier();
        if li < k {
            for (var i = 0u; i < 4u; i++) { acc[li][i] += acc[li + k][i]; }
        }
    }
    workgroupBarrier();
    if li < 4u { S[4u * wg.x + li] = acc[0][li]; }
}

// Seam low band of the template seam, per lens: blocks of P[14]² px, about 3° at the seam (the mip level a decoder's
// NV12 / P010 picture can't have), each the mask-weighted mean Y'CbCr of its pixels and the mean mask, so `soft`
// divides the rim's black back out (normalized convolution). A workgroup per block, over 2 × 2 luma quads.
var<workgroup> lacc: array<vec4f, 64>;

@compute @workgroup_size(64)
fn low(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u) {
    let k = wg.z;
    let b = L0 + 32u * k;
    let r = select(r0.r, r1.r, k == 1u);
    let f = u32(P[14]);
    let h = f / 2u;
    var a = vec4f(0.0);
    for (var j = li; j < h * h; j += 64u) {
        let p = wg.xy * f + 2u * vec2u(j % h, j / h);
        if any(p >= r.zw) { continue; }
        let m = clamp((P[b + 16] - distance(vec2f(p) + 1.0, vec2f(P[b + 10], P[b + 11]))) * P[b + 17], 0.0, 1.0);
        a += m * quad(k, r.xy + p);
    }
    lacc[li] = a;
    for (var s = 32u; s > 0u; s >>= 1u) {
        workgroupBarrier();
        if li < s { lacc[li] += lacc[li + s]; }
    }
    let n = vec2u(u32(P[15]), u32(P[16]));
    if li == 0u { Lo[k * n.x * n.y + wg.y * n.x + wg.x] = lacc[0] / f32(h * h); }
}

// lens k's plane values at the 2 × 2 luma texels from p (even): mean Y, their Cb Cr, 1
fn quad(k: u32, p: vec2u) -> vec4f {
    if k == 0u {
        let y = textureGather(0, y0, s0, vec2f(p + 1u) / vec2f(textureDimensions(y0)));
        return vec4f(dot(y, vec4f(0.25)), textureLoad(u0, p / 2u, 0).rg, 1.0);
    }
    let y = textureGather(0, y1, s1, vec2f(p + 1u) / vec2f(textureDimensions(y1)));
    return vec4f(dot(y, vec4f(0.25)), textureLoad(u1, p / 2u, 0).rg, 1.0);
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    return vec4f(vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0, 0.0, 1.0);
}

// Preview: the body ray of viewport pixel `pos`; 0 perspective / Panini, 1 equirect, 2 planet (general perspective
// from eye distance d: 1 stereographic, 0 rectilinear)
fn ray(pos: vec2f) -> vec3f {
    let u = (pos - view.rect.xy) / view.rect.zw;
    let x = vec2f(2.0 * u.x - 1.0, (1.0 - 2.0 * u.y) * view.rect.w / view.rect.z) * view.f;
    var r: vec3f;
    if view.mode == 1u {
        let lon = (2.0 * u.x - 1.0) * PI;
        let lat = (0.5 - u.y) * PI;
        r = vec3f(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
    } else if view.mode == 2u {
        // ρ = (d + 1)·sin θ / (d + cos θ) solved for θ: with k = ρ² / (d + 1)², d + cos θ = (d + s) / (k + 1),
        // s = √(k(1 − d²) + 1) (no cancellation); sin θ = ρ (d + cos θ) / (d + 1)
        let d = view.d;
        let p = length(x);
        let k = p * p / ((d + 1.0) * (d + 1.0));
        let e = (d + sqrt(k * (1.0 - d * d) + 1.0)) / (k + 1.0);
        r = vec3f(p * e / (d + 1.0) * normalize(x + vec2f(1e-12, 0.0)), e - d);
    } else {
        // general Panini, inverse (d = 0: rectilinear)
        let d = view.d;
        let k = x.x * x.x / ((d + 1.0) * (d + 1.0));
        let cl = (-k * d + sqrt(k * k * d * d - (k + 1.0) * (k * d * d - 1.0))) / (k + 1.0);
        let s = (d + 1.0) / (d + cl);
        let lon = atan2(x.x, s * cl);
        let lat = atan2(x.y, s);
        r = vec3f(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
    }
    return normalize(pose.r * (view.r * r));
}

// sRGB target: the hardware encodes, so hand it linear values
fn emit(c: vec3f) -> vec4f {
    let lin = select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
    return vec4f(select(c, lin, view.srgb != 0u), 1.0);
}

// Preview from the lenses, template seam; the detail fade spans 4 view pixels, at least 1.5° and at most 3°
@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let w = ray(pos.xy);
    return emit(seam(w, clamp(2.0 * degrees(max(length(dpdx(w)), length(dpdy(w)))), 0.75, 1.5)));
}

// Preview from the lenses with the stitcher's seam flow (`Preview::seam`)
@fragment
fn ff(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let w = ray(pos.xy);
    var r = array(w, w);
    if P[4] > 0.0 { r = warp(w); }
    return emit(shade(w, r[0], r[1]));
}

// Preview from an equirect in the stitcher's layout (column x at θ = 2πx/W, centre = −X; row y at φ = πy/(H − 1)),
// taken as body frame: `pose` turns it as it turns the lenses
@fragment
fn fe(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let w = ray(pos.xy);
    let n = vec2f(textureDimensions(pano));
    let t = vec2f(atan2(-w.y, w.x) / (2.0 * PI) * n.x, acos(clamp(w.z, -1.0, 1.0)) / PI * (n.y - 1.0)) + 0.5;
    return emit(textureSampleLevel(pano, s0, t / n, 0.0).rgb);
}
