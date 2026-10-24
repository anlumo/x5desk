// Colour effects on an RGBA frame (formulas: lib.rs `Params`): a luma pyramid for clarity (`down0`, `down`, `up`),
// frame statistics and the automatic corrections' estimate (`stat`, `solve`), and one fused pass per pixel (`main`).
// Values are display-encoded (sRGB transfer); `lin` / `enc` convert for the steps that work in linear light. `OUTFMT`
// is replaced per output format.

struct Par {
    m: array<vec4f, 3>,  // white balance · 2^exposure on linear RGB, rows
    a: vec4f,            // contrast, 0.25·whites, 0.15·blacks, highlights
    b: vec4f,            // 1.25·shadows, saturation, vibrance, look mix
    c: vec4f,            // 0.8·clarity, 2·sharpen, purple, green
    imin: vec4f,         // input 1D LUT: domain min, w: size (0 = off)
    imax: vec4f,         //   domain max, w: first entry in T
    lmin: vec4f,         // look 1D LUT, the same
    lmax: vec4f,
    dmin: vec4f,         // look 3D LUT: domain min, w: size (0 = off)
    dmax: vec4f,
    e: vec4f,            // blue, brightness exponent, underwater strength, auto strength
    g: vec4f,            // x: weight of this run's estimate in the smoothed one
    f: vec4u,            // width, height, flags (1 wrap x, 2 clarity, 4 defringe, 8 sharpen, 16 underwater, 32 auto),
                         // defringe radius
    s: vec4u,            // statistics grid: step (px), columns, rows, workgroups across
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(0) var<uniform> U: Par;
@group(1) @binding(1) var l3: texture_3d<f32>;
@group(1) @binding(2) var<storage, read> T: array<vec4f>;  // 1D LUTs: input, then look
@group(1) @binding(3) var lay: texture_2d<f32>;            // overlay layer: linear, premultiplied (an sRGB texture)
// pyramid passes: level in / out; main: pin = the pyramid base (half size), dst, the automatic corrections
@group(2) @binding(0) var pin: texture_2d<f32>;
@group(2) @binding(1) var pout: texture_storage_2d<r32float, write>;
@group(2) @binding(2) var dst: texture_storage_2d<OUTFMT, write>;
@group(2) @binding(3) var<storage, read> K: array<f32>;
// statistics: per-workgroup sums, the state (layout: lib.rs `Fx::kb`)
@group(2) @binding(4) var<storage, read_write> P: array<u32>;
@group(2) @binding(5) var<storage, read_write> KW: array<f32>;

override OV = false;  // main also composites the overlay layer

const W = vec3f(0.2126, 0.7152, 0.0722);
const PI = 3.14159265358979;
const TS = 40;       // tile stride: 16 + 2·12, the largest defringe radius
const NV = 6u;       // statistics per luma bin: Σw, Σw·r, Σw·g, Σw·b, Σw·r·g, Σw·s
const NS = 384u;     // 64 bins · NV
const Q = 262143.0;  // value scale of the sums: 256 samples · weight 64 · Q < 2³²

fn lin(c: vec3f) -> vec3f {
    let x = max(c, vec3f(0.0));
    return select(pow((x + 0.055) / 1.055, vec3f(2.4)), x / 12.92, x <= vec3f(0.04045));
}

fn enc(l: vec3f) -> vec3f {
    let x = max(l, vec3f(0.0));
    return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, 12.92 * x, x <= vec3f(0.0031308));
}

// 1D LUT, linear between entries; lo / hi: domain, w = size / first entry in T
fn tab(c: vec3f, lo: vec4f, hi: vec4f) -> vec3f {
    let u = clamp((c - lo.xyz) / (hi.xyz - lo.xyz), vec3f(0.0), vec3f(1.0)) * (lo.w - 1.0);
    let i = min(floor(u), vec3f(lo.w - 2.0));
    let f = u - i;
    let k = vec3u(i) + u32(hi.w);
    return vec3f(mix(T[k.x].x, T[k.x + 1u].x, f.x), mix(T[k.y].y, T[k.y + 1u].y, f.y), mix(T[k.z].z, T[k.z + 1u].z, f.z));
}

// 3D LUT, tetrahedral: c0 → a → b → c1 along the cell's tetrahedron that holds f (as ffmpeg's lut3d)
fn cube(c: vec3f) -> vec3f {
    let n = U.dmin.w;
    let u = clamp((c - U.dmin.xyz) / (U.dmax.xyz - U.dmin.xyz), vec3f(0.0), vec3f(1.0)) * (n - 1.0);
    let i = min(floor(u), vec3f(n - 2.0));
    let f = u - i;
    let p = vec3i(i);
    var a: vec3i;
    var b: vec3i;
    var w: vec4f;
    if f.r > f.g {
        if f.g > f.b {
            a = vec3i(1, 0, 0); b = vec3i(1, 1, 0); w = vec4f(1.0 - f.r, f.r - f.g, f.g - f.b, f.b);
        } else if f.r > f.b {
            a = vec3i(1, 0, 0); b = vec3i(1, 0, 1); w = vec4f(1.0 - f.r, f.r - f.b, f.b - f.g, f.g);
        } else {
            a = vec3i(0, 0, 1); b = vec3i(1, 0, 1); w = vec4f(1.0 - f.b, f.b - f.r, f.r - f.g, f.g);
        }
    } else {
        if f.b > f.g {
            a = vec3i(0, 0, 1); b = vec3i(0, 1, 1); w = vec4f(1.0 - f.b, f.b - f.g, f.g - f.r, f.r);
        } else if f.b > f.r {
            a = vec3i(0, 1, 0); b = vec3i(0, 1, 1); w = vec4f(1.0 - f.g, f.g - f.b, f.b - f.r, f.r);
        } else {
            a = vec3i(0, 1, 0); b = vec3i(1, 1, 0); w = vec4f(1.0 - f.g, f.g - f.r, f.r - f.b, f.b);
        }
    }
    return w.x * textureLoad(l3, p, 0).rgb + w.y * textureLoad(l3, p + a, 0).rgb + w.z * textureLoad(l3, p + b, 0).rgb
        + w.w * textureLoad(l3, p + vec3i(1), 0).rgb;
}

// x wraps on equirects (longitude), else clamps
fn ix(x: i32, w: i32) -> i32 { return select(clamp(x, 0, w - 1), (x % w + w) % w, (U.f.z & 1u) != 0u); }

// source texel after the input transform, luma in w
fn at(x: i32, y: i32) -> vec4f {
    var c = textureLoad(src, vec2i(ix(x, i32(U.f.x)), clamp(y, 0, i32(U.f.y) - 1)), 0).rgb;
    if U.imin.w > 0.0 { c = tab(c, U.imin, U.imax); }
    return vec4f(c, dot(c, W));
}

fn ld(x: i32, y: i32) -> f32 {
    let s = vec2i(textureDimensions(pin));
    return textureLoad(pin, vec2i(ix(x, s.x), clamp(y, 0, s.y - 1)), 0).x;
}

// pin bilinearly upsampled 2× at fine pixel p: fine 2i, 2i + 1 sit ¼ before / after coarse i
fn up2(p: vec2i) -> f32 {
    let i = p / 2;
    let j = select(i - 1, i + 1, (p & vec2i(1)) == vec2i(1));
    return 0.5625 * ld(i.x, i.y) + 0.1875 * (ld(j.x, i.y) + ld(i.x, j.y)) + 0.0625 * ld(j.x, j.y);
}

@compute @workgroup_size(16, 16)
fn down0(@builtin(global_invocation_id) id: vec3u) {
    let s = textureDimensions(pout);
    if id.x >= s.x || id.y >= s.y { return; }
    let p = vec2i(id.xy) * 2;
    let v = at(p.x, p.y).w + at(p.x + 1, p.y).w + at(p.x, p.y + 1).w + at(p.x + 1, p.y + 1).w;
    textureStore(pout, id.xy, vec4f(v / 4.0));
}

@compute @workgroup_size(16, 16)
fn down(@builtin(global_invocation_id) id: vec3u) {
    let s = textureDimensions(pout);
    if id.x >= s.x || id.y >= s.y { return; }
    let p = vec2i(id.xy) * 2 - 1;
    var k = array<f32, 4>(1.0, 3.0, 3.0, 1.0);
    var v = 0.0;
    for (var j = 0; j < 4; j++) {
        var r = 0.0;
        for (var i = 0; i < 4; i++) { r += k[i] * ld(p.x + i, p.y + j); }
        v += k[j] * r;
    }
    textureStore(pout, id.xy, vec4f(v / 64.0));
}

@compute @workgroup_size(16, 16)
fn up(@builtin(global_invocation_id) id: vec3u) {
    let s = textureDimensions(pout);
    if id.x >= s.x || id.y >= s.y { return; }
    textureStore(pout, id.xy, vec4f(up2(vec2i(id.xy))));
}

// Statistics on a grid of samples (one per step² pixels, at the cell centre) of the source after the input
// transform, binned by luma into 64 bins. Weights: 64, on equirects 64·cos(latitude) (solid angle). Integer sums
// (exact, order-free), one set per workgroup.
var<workgroup> HB: array<atomic<u32>, 384>;

@compute @workgroup_size(16, 16)
fn stat(@builtin(workgroup_id) g: vec3u, @builtin(local_invocation_id) l: vec3u,
        @builtin(local_invocation_index) li: u32) {
    for (var i = li; i < NS; i += 256u) { atomicStore(&HB[i], 0u); }
    workgroupBarrier();
    let q = g.xy * 16u + l.xy;
    if q.x < U.s.y && q.y < U.s.z {
        let p = min(q * U.s.x + U.s.x / 2u, U.f.xy - 1u);
        let c = at(i32(p.x), i32(p.y));
        let v = clamp(lin(c.rgb), vec3f(0.0), vec3f(1.0));
        let e = clamp(c.rgb, vec3f(0.0), vec3f(1.0));
        let mx = max(e.r, max(e.g, e.b));
        let s = (mx - min(e.r, min(e.g, e.b))) / max(mx, 1e-6);
        var w = 64u;
        if (U.f.z & 1u) != 0u { w = u32(round(64.0 * sin(PI * (f32(p.y) + 0.5) / f32(U.f.y)))); }
        let k = min(u32(clamp(c.w, 0.0, 1.0) * 64.0), 63u) * NV;
        atomicAdd(&HB[k], w);
        atomicAdd(&HB[k + 1u], w * u32(round(v.r * Q)));
        atomicAdd(&HB[k + 2u], w * u32(round(v.g * Q)));
        atomicAdd(&HB[k + 3u], w * u32(round(v.b * Q)));
        atomicAdd(&HB[k + 4u], w * u32(round(v.r * v.g * Q)));
        atomicAdd(&HB[k + 5u], w * u32(round(s * Q)));
    }
    workgroupBarrier();
    let o = (g.y * U.s.w + g.x) * NS;
    for (var i = li; i < NS; i += 256u) { P[o + i] = atomicLoad(&HB[i]); }
}

var<workgroup> Z: array<f32, 384>;
// per bin: weight, mean linear colour (after the corrections so far), luminance, mean r·g
var<private> BN: array<f32, 64>;
var<private> BM: array<vec3f, 64>;
var<private> BY: array<f32, 64>;
var<private> BQ: array<f32, 64>;

// mean colour of the darkest (`up`) or brightest share f of the weight n, bins 0…62 (63: clipped); `d` if none
fn share(f: f32, up: bool, n: f32, d: vec3f) -> vec3f {
    var s = vec3f(0.0);
    var w = 0.0;
    for (var i = 0u; i < 63u; i++) {
        let r = f * n - w;
        if r <= 0.0 { break; }
        let b = select(62u - i, i, up);
        let x = min(BN[b], r);
        s += x * BM[b];
        w += x;
    }
    return select(d, s / w, w > 0.0);
}

// BY of the bin holding the share f of the weight n, from the dark end
fn pct(f: f32, n: f32) -> f32 {
    var w = 0.0;
    for (var b = 0u; b < 64u; b++) {
        w += BN[b];
        if w >= f * n && BN[b] > 0.0 { return BY[b]; }
    }
    return BY[63];
}

// The automatic corrections' estimate from the statistics, smoothed over time, and the coefficients `main` applies
// (formulas: lib.rs `Params::water`, `Params::auto`). One workgroup.
@compute @workgroup_size(256)
fn solve(@builtin(local_invocation_index) li: u32) {
    let ng = U.s.w * ((U.s.z + 15u) / 16u);
    for (var j = li; j < NS; j += 256u) {
        var t = 0.0;
        for (var k = 0u; k < ng; k++) { t += f32(P[k * NS + j]); }
        Z[j] = t;
        KW[32u + j] = t;
    }
    workgroupBarrier();
    if li != 0u { return; }
    var n = 0.0;
    var m = vec3f(0.0);
    var sat = 0.0;
    for (var b = 0u; b < 64u; b++) {
        let k = b * NV;
        let w = Z[k];
        let s = vec3f(Z[k + 1u], Z[k + 2u], Z[k + 3u]) / Q;
        BN[b] = w;
        BM[b] = s / max(w, 1e-9);
        BQ[b] = Z[k + 4u] / (Q * max(w, 1e-9));
        n += w;
        m += s;
        sat += Z[k + 5u] / Q;
    }
    n = max(n, 1e-9);
    m /= n;
    var e = array<f32, 15>(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if (U.f.z & 16u) != 0u {
        let k = 0.5 * max(m.g - m.r, 0.0);
        var mc = vec3f(0.0);
        for (var b = 0u; b < 64u; b++) {
            BM[b].r += k * (BM[b].g - BQ[b]);
            mc += BN[b] * BM[b];
        }
        mc /= n;
        let v = share(0.01, true, n, mc);
        let a = max(mc - v, vec3f(1e-4));
        let h = max(share(0.01, false, n, mc) - v, vec3f(1e-4));
        let r = 1.0 / (pow(a, vec3f(0.25)) * pow(h, vec3f(0.75)));
        let gw = min(r / min(r.r, min(r.g, r.b)), vec3f(8.0));
        for (var b = 0u; b < 64u; b++) { BM[b] = max((BM[b] - v) * gw, vec3f(0.0)); }
        e[0] = k; e[1] = v.r; e[2] = v.g; e[3] = v.b; e[4] = gw.r; e[5] = gw.g; e[6] = gw.b;
    }
    if (U.f.z & 32u) != 0u {
        var mg = vec3f(0.0);
        for (var b = 0u; b < 64u; b++) { mg += BN[b] * BM[b]; }
        mg /= n;
        let il = sqrt(max(mg, vec3f(1e-5)) * max(share(0.02, false, n, mg), vec3f(1e-5)));
        let yi = dot(W, il);
        var ga = clamp(sqrt(yi / il), vec3f(0.8), vec3f(1.25));
        ga *= yi / dot(W, ga * il);
        for (var b = 0u; b < 64u; b++) { BY[b] = dot(W, ga * BM[b]); }
        let bk = 0.5 * pct(0.005, n);
        let wt = pct(0.995, n);
        var lk = 0.0;
        for (var b = 0u; b < 64u; b++) { lk += BN[b] * log2(max(BY[b] - bk, 1e-4)); }
        // up to 0.74 EV past what keeps the 99.5 % luminance from clipping more, given back there by highlights −1
        let cl = max(log2(max(wt, 1.0) / max(wt - bk, 1e-4)), 0.0);
        let ev = min(clamp(0.5 * (log2(0.18) - lk / n), -1.0, 1.0), cl + 0.74);
        let hl = -clamp((ev - cl) / 0.74, 0.0, 1.0);
        var l1 = 0.0;
        var l2 = 0.0;
        for (var b = 0u; b < 64u; b++) {
            let y = max(BY[b] - bk, 0.0) * exp2(ev);
            let t = (log2(max(y, 1e-6) / 0.18) - 1.5) / 1.25;
            let l = enc(vec3f(y * exp2(hl * exp(-0.5 * t * t)))).x;
            l1 += BN[b] * l;
            l2 += BN[b] * l * l;
        }
        l1 /= n;
        let sd = sqrt(max(l2 / n - l1 * l1, 0.0));
        e[7] = ga.r; e[8] = ga.g; e[9] = ga.b; e[10] = bk; e[11] = ev;
        e[12] = clamp(2.0 * (0.25 - sd), 0.0, 0.3);
        e[13] = clamp(2.0 * (0.35 - sat / n), 0.0, 0.5);
        e[14] = hl;
    }
    // gains are > 0 once written: a state never written starts from this frame
    let a = select(U.g.x, 1.0, KW[4] <= 0.0 || KW[7] <= 0.0);
    for (var i = 0u; i < 15u; i++) { KW[i] = mix(KW[i], e[i], a); }
    let sw = U.e.z;
    let sa = U.e.w;
    let gw = pow(vec3f(KW[4], KW[5], KW[6]), vec3f(sw));
    let g2 = pow(vec3f(KW[7], KW[8], KW[9]), vec3f(sa)) * exp2(sa * KW[11]);
    let gg = gw * g2;
    let o = (sw * vec3f(KW[1], KW[2], KW[3]) * gw + sa * KW[10]) * g2;
    KW[16] = sw * KW[0];
    KW[17] = gg.r; KW[18] = gg.g; KW[19] = gg.b;
    KW[20] = o.r; KW[21] = o.g; KW[22] = o.b;
    KW[23] = sa * KW[12];
    KW[24] = sa * KW[13];
    KW[25] = sa * KW[14];
}

// 1 inside ±i degrees of hue c, 0 beyond ±o
fn band(h: f32, c: f32, i: f32, o: f32) -> f32 {
    return 1.0 - smoothstep(i, o, abs(fract((h - c) / 360.0 + 0.5) - 0.5) * 360.0);
}

var<workgroup> S: array<f32, 1600>;  // luma of the tile around the group's 16² (apron r), stride TS
var<workgroup> HL: array<f32, 640>;  // defringe: min / max of S over the 2R + 1 window of each tile row, 16 per row
var<workgroup> HH: array<f32, 640>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) g: vec3u, @builtin(local_invocation_id) l: vec3u,
        @builtin(local_invocation_index) li: u32) {
    let p = vec2i(g.xy * 16u + l.xy);
    let fl = U.f.z;
    let s0 = at(p.x, p.y);
    var c = s0.rgb;
    let y = s0.w;
    var dl = 0.0;
    if (fl & 12u) != 0u {
        let r = select(1, i32(U.f.w), (fl & 4u) != 0u);
        let tw = 16 + 2 * r;
        let o = vec2i(g.xy * 16u) - r;
        for (var i = i32(li); i < tw * tw; i += 256) { S[i / tw * TS + i % tw] = at(o.x + i % tw, o.y + i / tw).w; }
        workgroupBarrier();
        if (fl & 4u) != 0u {
            // defringe: hue-limited desaturation toward the luma, near high-contrast edges with a bright side
            for (var i = i32(li); i < tw * 16; i += 256) {
                let b = i / 16 * TS + i % 16;
                var lo = S[b];
                var hi = lo;
                for (var j = 1; j <= 2 * r; j++) {
                    lo = min(lo, S[b + j]);
                    hi = max(hi, S[b + j]);
                }
                HL[i] = lo;
                HH[i] = hi;
            }
            workgroupBarrier();
            let b = i32(l.y) * 16 + i32(l.x);
            var lo = HL[b];
            var hi = HH[b];
            for (var j = 1; j <= 2 * r; j++) {
                lo = min(lo, HL[b + j * 16]);
                hi = max(hi, HH[b + j * 16]);
            }
            let e = smoothstep(0.25, 0.6, hi - lo);
            let mx = max(c.r, max(c.g, c.b));
            let d = mx - min(c.r, min(c.g, c.b));
            var h = 0.0;  // HSV hue in degrees
            if d > 0.0 {
                if mx == c.r { h = (c.g - c.b) / d; } else if mx == c.g { h = (c.b - c.r) / d + 2.0; } else { h = (c.r - c.g) / d + 4.0; }
                h = fract(h / 6.0) * 360.0;
            }
            // blue only next to near-clipped highlights: blue sky borders bright stone and statues too
            let a = min(e * (smoothstep(0.6, 0.9, hi) * (U.c.z * band(h, 285.0, 30.0, 50.0)
                + U.c.w * band(h, 120.0, 25.0, 45.0)) + smoothstep(0.85, 0.97, hi) * U.e.x * band(h, 220.0, 15.0, 30.0)), 1.0);
            c = y + (c - y) * (1.0 - a);
        }
        if (fl & 8u) != 0u {
            let k = (i32(l.y) + r) * TS + i32(l.x) + r;
            let b = (4.0 * y + 2.0 * (S[k - 1] + S[k + 1] + S[k - TS] + S[k + TS])
                + S[k - TS - 1] + S[k - TS + 1] + S[k + TS - 1] + S[k + TS + 1]) / 16.0;
            dl = U.c.y * (y - b);
        }
    }
    if p.x >= i32(U.f.x) || p.y >= i32(U.f.y) { return; }
    if (fl & 2u) != 0u {
        let t = clamp(y, 0.0, 1.0);
        dl += U.c.x * (y - up2(p)) * 4.0 * t * (1.0 - t);
    }
    c = max(c + dl, vec3f(0.0));
    var v = lin(c);
    let au = (fl & 48u) != 0u;
    if au {
        v.r += K[16] * max(1.0 - v.r, 0.0) * v.g;
        v = max(v * vec3f(K[17], K[18], K[19]) - vec3f(K[20], K[21], K[22]), vec3f(0.0));
    }
    v = max(vec3f(dot(U.m[0].xyz, v), dot(U.m[1].xyz, v), dot(U.m[2].xyz, v)), vec3f(0.0));
    let hl = clamp(U.a.w + select(0.0, K[25], au), -1.0, 1.0);
    if hl != 0.0 || U.b.x != 0.0 {
        let ev = log2(max(dot(v, W), 1e-6) / 0.18);
        let s = (ev + 2.25) / 1.5;
        let t = (ev - 1.5) / 1.25;
        v *= exp2(hl * exp(-0.5 * t * t) + U.b.x * exp(-0.5 * s * s));
    }
    var e = enc(v);
    let ct = clamp(U.a.x + select(0.0, K[23], au), -1.0, 1.0);
    if ct != 0.0 || U.a.y != 0.0 || U.a.z != 0.0 {
        var t = clamp(e, vec3f(0.0), vec3f(1.0));
        e -= ct * t * (1.0 - t) * (1.0 - 2.0 * t);
        t = clamp(e, vec3f(0.0), vec3f(1.0));
        e += U.a.y * t * t * t * t;
        t = 1.0 - clamp(e, vec3f(0.0), vec3f(1.0));
        e += U.a.z * t * t * t * t;
    }
    if U.e.y != 1.0 { e = pow(max(e, vec3f(0.0)), vec3f(U.e.y)); }
    let vb = clamp(U.b.z + select(0.0, K[24], au), -1.0, 1.0);
    if U.b.y != 0.0 || vb != 0.0 {
        v = lin(e);
        let yl = dot(v, W);
        let mx = max(v.r, max(v.g, v.b));
        let s = (mx - min(v.r, min(v.g, v.b))) / max(mx, 1e-6);
        e = enc(yl + (v - yl) * (1.0 + U.b.y) * (1.0 + vb * (1.0 - s)));
    }
    var lk = e;
    if U.lmin.w > 0.0 { lk = tab(lk, U.lmin, U.lmax); }
    if U.dmin.w > 0.0 { lk = cube(lk); }
    var o = vec4f(max(mix(e, lk, U.b.w), vec3f(0.0)), textureLoad(src, p, 0).a);
    if OV {
        let q = textureLoad(lay, p, 0);
        o = vec4f(enc(lin(o.rgb) * (1.0 - q.a) + q.rgb), q.a + o.a * (1.0 - q.a));
    }
    textureStore(dst, p, o);
}
