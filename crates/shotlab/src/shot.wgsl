// Shot Lab passes on equirect frames in stitch's layout: pixel (x, y) at azimuth 2πx/W, polar angle πy/(H − 1),
// direction (sin φ cos θ, −sin φ sin θ, cos φ). Frames hold sRGB-encoded values; light is added and stacked in linear
// light. OUTFMT is replaced by the destination's storage format. Each entry point has bindings of its own (auto layout).

const PI = 3.14159265358979;
const TAU = 6.28318530717959;
const LUMA = vec3f(0.2126, 0.7152, 0.0722);

fn lin(c: vec3f) -> vec3f {
    let a = abs(c);
    return sign(c) * select(pow((a + 0.055) / 1.055, vec3f(2.4)), a / 12.92, a <= vec3f(0.04045));
}

fn enc(l: vec3f) -> vec3f {
    let a = max(l, vec3f(0.0));
    return select(1.055 * pow(a, vec3f(1.0 / 2.4)) - 0.055, 12.92 * a, a <= vec3f(0.0031308));
}

fn ray(p: vec2f, d: vec2u) -> vec3f {
    let th = TAU * p.x / f32(d.x);
    let ph = PI * p.y / f32(d.y - 1u);
    return vec3f(sin(ph) * cos(th), -sin(ph) * sin(th), cos(ph));
}

// ---- median: per channel, by bisection over the value grid (exact on 8-bit frames: 8 halvings of 0..255)

struct Med {
    n: u32,
    // grid steps per unit (255, or 4095 for 16-bit float frames), halvings
    s: f32,
    bits: u32,
    p: u32,
}

@group(0) @binding(0) var mring: texture_2d_array<f32>;
@group(0) @binding(1) var<uniform> mu: Med;
@group(0) @binding(2) var mout: texture_storage_2d<OUTFMT, write>;

@compute @workgroup_size(8, 8)
fn med(@builtin(global_invocation_id) id: vec3u) {
    let d = textureDimensions(mout);
    if id.x >= d.x || id.y >= d.y {
        return;
    }
    // the lower median: the r-th smallest
    let r = vec3u((mu.n + 1u) / 2u);
    var lo = vec3f(0.0);
    var hi = vec3f(mu.s);
    for (var b = 0u; b < mu.bits; b++) {
        let m = floor((lo + hi) * 0.5);
        var c = vec3u(0u);
        for (var j = 0u; j < mu.n; j++) {
            let v = round(clamp(textureLoad(mring, id.xy, j, 0).rgb, vec3f(0.0), vec3f(1.0)) * mu.s);
            c += select(vec3u(0u), vec3u(1u), v <= m);
        }
        let ok = c >= r;
        hi = select(hi, m, ok);
        lo = select(m + 1.0, lo, ok);
    }
    textureStore(mout, id.xy, vec4f(lo / mu.s, 1.0));
}

// ---- light trails: decaying per-channel max of the bright parts, in linear light

struct Tr {
    thr: f32,
    soft: f32,
    decay: f32,
    gain: f32,
    add: u32,
    on: u32,
    w: u32,
    p: u32,
}

@group(0) @binding(3) var tsrc: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> tacc: array<vec2u>;
@group(0) @binding(5) var<uniform> tu: Tr;
@group(0) @binding(6) var tdst: texture_storage_2d<OUTFMT, write>;

@compute @workgroup_size(8, 8)
fn trail(@builtin(global_invocation_id) id: vec3u) {
    let d = textureDimensions(tdst);
    if id.x >= d.x || id.y >= d.y {
        return;
    }
    let c = textureLoad(tsrc, id.xy, 0);
    let l = lin(c.rgb);
    let i = id.y * tu.w + id.x;
    let p = tacc[i];
    var a = vec3f(unpack2x16float(p.x), unpack2x16float(p.y).x) * tu.decay;
    if tu.on != 0u {
        a = max(a, l * (tu.gain * smoothstep(tu.thr, tu.thr + tu.soft, dot(l, LUMA))));
    }
    // f16 range
    a = min(a, vec3f(60000.0));
    tacc[i] = vec2u(pack2x16float(a.rg), pack2x16float(vec2f(a.b, 0.0)));
    textureStore(tdst, id.xy, vec4f(enc(select(max(l, a), l + a, tu.add != 0u)), c.a));
}

// ---- frame delay: each pixel shows the frame s·span frames back, s from its direction, between ring frames linearly

struct De {
    // direction; w: 0 longitude from its azimuth, 1 angle from it, 2 columns, 3 rows
    dir: vec4f,
    // frames held, ring size, slot of the newest, reversed
    k: u32,
    n: u32,
    h: u32,
    rev: u32,
    span: f32,
    p0: f32,
    p1: f32,
    p2: f32,
}

@group(0) @binding(7) var dring: texture_2d_array<f32>;
@group(0) @binding(8) var<uniform> du: De;
@group(0) @binding(9) var ddst: texture_storage_2d<OUTFMT, write>;

fn slot(t: u32) -> u32 {
    return (du.h + du.n - t % du.n) % du.n;
}

@compute @workgroup_size(8, 8)
fn delay(@builtin(global_invocation_id) id: vec3u) {
    let d = textureDimensions(ddst);
    if id.x >= d.x || id.y >= d.y {
        return;
    }
    let r = ray(vec2f(id.xy), d);
    var s: f32;
    switch u32(du.dir.w) {
        case 0u: {
            let a = atan2(-r.y, r.x) - atan2(-du.dir.y, du.dir.x);
            s = abs(atan2(sin(a), cos(a))) / PI;
        }
        case 1u: {
            s = acos(clamp(dot(r, du.dir.xyz), -1.0, 1.0)) / PI;
        }
        case 2u: {
            s = f32(id.x) / f32(max(d.x, 2u) - 1u);
        }
        default: {
            s = f32(id.y) / f32(max(d.y, 2u) - 1u);
        }
    }
    if du.rev != 0u {
        s = 1.0 - s;
    }
    let last = f32(min(du.k, du.n) - 1u);
    let t = min(s * du.span, last);
    let t0 = floor(t);
    let a = textureLoad(dring, id.xy, slot(u32(t0)), 0);
    let b = textureLoad(dring, id.xy, slot(u32(min(t0 + 1.0, last))), 0);
    textureStore(ddst, id.xy, mix(a, b, t - t0));
}

// ---- star stacking: the frame resampled along the sky's rotation (the ground stays), summed or maxed in linear light

struct Sk {
    // reference -> frame direction (columns)
    m0: vec4f,
    m1: vec4f,
    m2: vec4f,
    // sin of the elevations where the ground ends and the sky starts; z: horizon on
    hor: vec4f,
    w: u32,
    h: u32,
    max: u32,
    p: u32,
}

@group(0) @binding(10) var ssrc: texture_2d<f32>;
@group(0) @binding(11) var ssmp: sampler;
@group(0) @binding(12) var<uniform> su: Sk;
@group(0) @binding(13) var<storage, read_write> ssum: array<vec4f>;

// texture coordinates of direction v (u repeats)
fn uv(v: vec3f, d: vec2f) -> vec2f {
    let th = atan2(-v.y, v.x);
    let ph = acos(clamp(v.z, -1.0, 1.0));
    return vec2f(th / TAU * d.x + 0.5, ph / PI * (d.y - 1.0) + 0.5) / d;
}

@compute @workgroup_size(8, 8)
fn stack(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= su.w || id.y >= su.h {
        return;
    }
    let r = ray(vec2f(id.xy), vec2u(su.w, su.h));
    let q = mat3x3f(su.m0.xyz, su.m1.xyz, su.m2.xyz) * r;
    let d = vec2f(textureDimensions(ssrc));
    var c = textureSampleLevel(ssrc, ssmp, uv(q, d), 0.0).rgb;
    if su.hor.z != 0.0 {
        let g = 1.0 - smoothstep(su.hor.x, su.hor.y, r.z);
        c = mix(c, textureSampleLevel(ssrc, ssmp, uv(r, d), 0.0).rgb, g);
    }
    let i = id.y * su.w + id.x;
    let l = vec4f(lin(c), 1.0);
    ssum[i] = select(ssum[i] + l, max(ssum[i], l), su.max != 0u);
}

@group(0) @binding(14) var<storage, read> fsum: array<vec4f>;
@group(0) @binding(15) var fout: texture_storage_2d<OUTFMT, write>;

@compute @workgroup_size(8, 8)
fn fin(@builtin(global_invocation_id) id: vec3u) {
    let d = textureDimensions(fout);
    if id.x >= d.x || id.y >= d.y {
        return;
    }
    let s = fsum[id.y * d.x + id.x];
    textureStore(fout, id.xy, vec4f(enc(s.rgb / max(s.w, 1e-6)), 1.0));
}

// ---- clones: a subject's perspective crop (the patch, the network input) and its cut-out composited back

struct Cam {
    // forward, right, up (unit); e.xy: tangent-plane half extents of the crop, e.zw: of the subject's box
    f: vec4f,
    r: vec4f,
    u: vec4f,
    e: vec4f,
}

struct Cu {
    // slot, side of the target, taps per axis, mask elements per slot
    slot: u32,
    side: u32,
    k: u32,
    p: u32,
}

@group(0) @binding(16) var csrc: texture_2d<f32>;
@group(0) @binding(17) var csmp: sampler;
@group(0) @binding(18) var<uniform> cu: Cu;
@group(0) @binding(19) var<storage, read> cams: array<Cam>;
@group(0) @binding(20) var cpat: texture_storage_2d_array<OUTFMT, write>;
@group(0) @binding(21) var<storage, read_write> cnet: array<f32>;

// the crop at target pixel q (side s): a k × k box over the pixel (a crop that shrinks the frame would alias)
fn crop(c: Cam, q: vec2f, s: f32, k: u32) -> vec3f {
    let d = vec2f(textureDimensions(csrc));
    var a = vec3f(0.0);
    for (var t = 0u; t < k * k; t++) {
        let p = (q + (vec2f(f32(t % k), f32(t / k)) + 0.5) / f32(k)) / s * 2.0 - 1.0;
        let w = normalize(c.f.xyz + p.x * c.e.x * c.r.xyz - p.y * c.e.y * c.u.xyz);
        a += textureSampleLevel(csrc, csmp, uv(w, d), 0.0).rgb;
    }
    return a / f32(k * k);
}

@compute @workgroup_size(8, 8)
fn pat(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= cu.side || id.y >= cu.side {
        return;
    }
    textureStore(cpat, id.xy, cu.slot, vec4f(crop(cams[cu.slot], vec2f(id.xy), f32(cu.side), cu.k), 1.0));
}

// network input: planar RGB, (c − ½) / ½
@compute @workgroup_size(8, 8)
fn netin(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= cu.side || id.y >= cu.side {
        return;
    }
    let c = clamp(crop(cams[cu.slot], vec2f(id.xy), f32(cu.side), cu.k), vec3f(0.0), vec3f(1.0)) * 2.0 - 1.0;
    let m = cu.side * cu.side;
    let i = id.y * cu.side + id.x;
    cnet[i] = c.r;
    cnet[i + m] = c.g;
    cnet[i + 2u * m] = c.b;
}

struct Pu {
    // mask strides (elements: channel, row, column), side; slots listed, alpha contrast
    st: vec4u,
    n: u32,
    msz: u32,
    gain: f32,
    p: u32,
    slots: array<vec4u, 16>,
}

@group(0) @binding(22) var pbg: texture_2d<f32>;
@group(0) @binding(23) var ppat: texture_2d_array<f32>;
@group(0) @binding(24) var psmp: sampler;
@group(0) @binding(25) var<storage, read> pcams: array<Cam>;
@group(0) @binding(26) var<storage, read> pmask: array<f32>;
@group(0) @binding(27) var<uniform> pu: Pu;
@group(0) @binding(28) var pdst: texture_storage_2d<OUTFMT, write>;

fn mk(b: u32, x: i32, y: i32) -> f32 {
    let s = i32(pu.st.w) - 1;
    return pmask[b + pu.st.x + pu.st.y * u32(clamp(y, 0, s)) + pu.st.z * u32(clamp(x, 0, s))];
}

@compute @workgroup_size(8, 8)
fn paste(@builtin(global_invocation_id) id: vec3u) {
    let d = textureDimensions(pdst);
    if id.x >= d.x || id.y >= d.y {
        return;
    }
    let w = ray(vec2f(id.xy), d);
    var o = textureLoad(pbg, id.xy, 0);
    for (var i = 0u; i < pu.n; i++) {
        let s = pu.slots[i / 4u][i % 4u];
        let c = pcams[s];
        let z = dot(w, c.f.xyz);
        if z <= 1e-4 {
            continue;
        }
        let p = vec2f(dot(w, c.r.xyz) / (z * c.e.x), -dot(w, c.u.xyz) / (z * c.e.y));
        // the subject's box, a little larger: other people in the crop stay out
        let g = c.e.zw / c.e.xy;
        let gate = (1.0 - smoothstep(g.x * 1.1, g.x * 1.3, abs(p.x))) * (1.0 - smoothstep(g.y * 1.05, g.y * 1.2, abs(p.y)));
        if gate <= 0.0 || any(abs(p) >= vec2f(1.0)) {
            continue;
        }
        // mask bilinear at the crop position (pixel centres at (i + ½) / side)
        let m = (p * 0.5 + 0.5) * f32(pu.st.w) - 0.5;
        let m0 = floor(m);
        let f = m - m0;
        let b = s * pu.msz;
        let x = i32(m0.x);
        let y = i32(m0.y);
        let v = mix(mix(mk(b, x, y), mk(b, x + 1, y), f.x), mix(mk(b, x, y + 1), mk(b, x + 1, y + 1), f.x), f.y);
        let a = clamp((v - 0.5) * pu.gain + 0.5, 0.0, 1.0) * gate;
        let q = textureSampleLevel(ppat, psmp, p * vec2f(0.5, 0.5) + 0.5, s, 0.0).rgb;
        o = vec4f(mix(o.rgb, q, a), o.a);
    }
    textureStore(pdst, id.xy, o);
}

// ---- sky swap: the network's tiles and the mask fixed to the world (an equirect in stitch's layout), the frame's
// directions turned into it by m

struct Sy {
    // world from frame (columns)
    m0: vec4f,
    m1: vec4f,
    m2: vec4f,
    // probability strides (tile, row, column), the network output's side
    st: vec4u,
    // weight of the new probability, of the new gain; spare; how far the panorama's colour moves to the old sky's
    w: f32,
    gw: f32,
    p0: f32,
    tone: f32,
    // taps per axis into a tile, into a mask pixel; tiles, the network input's side
    k: u32,
    kg: u32,
    n: u32,
    side: u32,
}

@group(0) @binding(29) var<uniform> sy: Sy;
@group(0) @binding(30) var<storage, read> stile: array<Cam>;
@group(0) @binding(31) var<storage, read_write> snet: array<f32>;

fn world(d: vec3f) -> vec3f {
    return mat3x3f(sy.m0.xyz, sy.m1.xyz, sy.m2.xyz) * d;
}

fn framed(w: vec3f) -> vec3f {
    return transpose(mat3x3f(sy.m0.xyz, sy.m1.xyz, sy.m2.xyz)) * w;
}

// network input: planar RGB, ImageNet's normalisation; tile id.z, read from the frame (csrc) along the frame's rays
@compute @workgroup_size(8, 8)
fn skyin(@builtin(global_invocation_id) id: vec3u) {
    let s = sy.side;
    if id.x >= s || id.y >= s {
        return;
    }
    let t = stile[id.z];
    let c = Cam(vec4f(framed(t.f.xyz), 0.0), vec4f(framed(t.r.xyz), 0.0), vec4f(framed(t.u.xyz), 0.0), t.e);
    let v = clamp(crop(c, vec2f(id.xy), f32(s), sy.k), vec3f(0.0), vec3f(1.0));
    let n = (v - vec3f(0.485, 0.456, 0.406)) / vec3f(0.229, 0.224, 0.225);
    let m = s * s;
    let i = id.z * 3u * m + id.y * s + id.x;
    snet[i] = n.r;
    snet[i + m] = n.g;
    snet[i + 2u * m] = n.b;
}

@group(0) @binding(32) var<storage, read> sprob: array<f32>;
@group(0) @binding(33) var sprev: texture_2d<f32>;
@group(0) @binding(34) var snext: texture_storage_2d<rgba16float, write>;

fn sp(t: u32, x: i32, y: i32) -> f32 {
    let s = i32(sy.st.w) - 1;
    return sprob[sy.st.x * t + sy.st.y * u32(clamp(y, 0, s)) + sy.st.z * u32(clamp(x, 0, s))];
}

// the world mask: the tiles' probability (weights falling to each tile's edge: they overlap widely, no seam shows),
// none below them, mixed into the last; the frame's colour over the pixel, the guided filter's guide
@compute @workgroup_size(8, 8)
fn skym(@builtin(global_invocation_id) id: vec3u) {
    let d = textureDimensions(snext);
    if id.x >= d.x || id.y >= d.y {
        return;
    }
    let w = ray(vec2f(id.xy), d);
    var p = 0.0;
    var sw = 0.0;
    for (var t = 0u; t < sy.n; t++) {
        let c = stile[t];
        let z = dot(w, c.f.xyz);
        if z <= 1e-4 {
            continue;
        }
        let q = vec2f(dot(w, c.r.xyz) / (z * c.e.x), -dot(w, c.u.xyz) / (z * c.e.y));
        let g = 1.0 - max(abs(q.x), abs(q.y));
        if g <= 0.0 {
            continue;
        }
        let m = (q * 0.5 + 0.5) * f32(sy.st.w) - 0.5;
        let m0 = floor(m);
        let f = m - m0;
        let x = i32(m0.x);
        let y = i32(m0.y);
        p += g * mix(mix(sp(t, x, y), sp(t, x + 1, y), f.x), mix(sp(t, x, y + 1), sp(t, x + 1, y + 1), f.x), f.y);
        sw += g;
    }
    p = select(0.0, p / sw, sw > 0.0);
    let k = sy.kg;
    let dd = vec2f(textureDimensions(csrc));
    var c = vec3f(0.0);
    for (var i = 0u; i < k * k; i++) {
        let o = (vec2f(f32(i % k), f32(i / k)) + 0.5) / f32(k) - 0.5;
        c += textureSampleLevel(csrc, csmp, uv(framed(ray(vec2f(id.xy) + o, d)), dd), 0.0).rgb;
    }
    textureStore(snext, id.xy, vec4f(mix(textureLoad(sprev, id.xy, 0).x, p, sy.w), c / f32(k * k)));
}

// guided filter (colour guide, He et al.: a white pole against a blue sky differs in colour, hardly in luma), radius GR
// mask pixels (3.5°: the network's edge is off by a degree or two, and a wider window leans more on the picture),
// regularised by GEPS; across longitude it wraps. The network's probability is soft across its edge (its output is a
// quarter of its input): steepened about ½ by GH first, so that the filter puts the edge where the picture has it
// instead of carrying the softness onto what stands against the sky
const GR = 10;
const GEPS = 0.002;
const GH = 3.0;

@group(0) @binding(35) var amask: texture_2d<f32>;
@group(0) @binding(36) var aout: texture_storage_2d<rgba32float, write>;

// coefficients a (per channel) and b of the window: a = (Σ + GEPS)⁻¹ cov(colour, p), b = mean p − a · mean colour
@compute @workgroup_size(8, 8)
fn skya(@builtin(global_invocation_id) id: vec3u) {
    let d = vec2i(textureDimensions(aout));
    if i32(id.x) >= d.x || i32(id.y) >= d.y {
        return;
    }
    var mc = vec3f(0.0);
    var mp = 0.0;
    var cp = vec3f(0.0);
    var dg = vec3f(0.0);
    var od = vec3f(0.0);
    for (var j = -GR; j <= GR; j++) {
        for (var i = -GR; i <= GR; i++) {
            let v = textureLoad(amask, vec2i((i32(id.x) + i + d.x) % d.x, clamp(i32(id.y) + j, 0, d.y - 1)), 0);
            let p = clamp((v.x - 0.5) * GH + 0.5, 0.0, 1.0);
            let c = v.yzw;
            mc += c;
            mp += p;
            cp += c * p;
            dg += c * c;
            od += c.xxy * c.yzz;
        }
    }
    let n = f32((2 * GR + 1) * (2 * GR + 1));
    mc /= n;
    mp /= n;
    let cv = cp / n - mc * mp;
    let dd = dg / n - mc * mc + GEPS;
    let oo = od / n - mc.xxy * mc.yzz;
    // Σ's columns; its inverse's rows are their cross products over the determinant
    let s0 = vec3f(dd.x, oo.x, oo.y);
    let s1 = vec3f(oo.x, dd.y, oo.z);
    let s2 = vec3f(oo.y, oo.z, dd.z);
    let r0 = cross(s1, s2);
    let a = vec3f(dot(r0, cv), dot(cross(s2, s0), cv), dot(cross(s0, s1), cv)) / dot(s0, r0);
    textureStore(aout, id.xy, vec4f(a, mp - dot(a, mc)));
}

@group(0) @binding(37) var bin: texture_2d<f32>;
@group(0) @binding(38) var bout: texture_storage_2d<rgba32float, write>;

// the coefficients' mean over the same box
@compute @workgroup_size(8, 8)
fn skyb(@builtin(global_invocation_id) id: vec3u) {
    let d = vec2i(textureDimensions(bout));
    if i32(id.x) >= d.x || i32(id.y) >= d.y {
        return;
    }
    var s = vec4f(0.0);
    for (var j = -GR; j <= GR; j++) {
        for (var i = -GR; i <= GR; i++) {
            s += textureLoad(bin, vec2i((i32(id.x) + i + d.x) % d.x, clamp(i32(id.y) + j, 0, d.y - 1)), 0);
        }
    }
    textureStore(bout, id.xy, s / f32((2 * GR + 1) * (2 * GR + 1)));
}

@group(0) @binding(39) var gmask: texture_2d<f32>;
@group(0) @binding(40) var gpano: texture_2d<f32>;
@group(0) @binding(41) var gsmp: sampler;
@group(0) @binding(42) var<storage, read_write> ggain: vec4f;

var<workgroup> wo: array<vec4f, 256>;
var<workgroup> wn: array<vec4f, 256>;

// the colour match: per channel the old sky's mean over the new one's where the mask has sky (by solid angle, every
// 2nd mask pixel either way, linear light), to the power tone, mixed into the last (w: 0 until a sky was measured)
@compute @workgroup_size(256)
fn skyg(@builtin(local_invocation_index) li: u32) {
    let d = textureDimensions(gmask);
    let pd = vec2f(textureDimensions(gpano));
    let cols = d.x / 2u;
    let cnt = cols * (d.y / 2u);
    var o = vec4f(0.0);
    var n = vec4f(0.0);
    for (var i = li; i < cnt; i += 256u) {
        let q = vec2u(i % cols, i / cols) * 2u;
        let v = textureLoad(gmask, q, 0);
        let w = ray(vec2f(q), d);
        let a = v.x * sqrt(max(1.0 - w.z * w.z, 0.0));
        o += vec4f(lin(v.yzw), 1.0) * a;
        n += vec4f(lin(textureSampleLevel(gpano, gsmp, uv(w, pd), 0.0).rgb), 0.0) * a;
    }
    wo[li] = o;
    wn[li] = n;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if li < s {
            wo[li] += wo[li + s];
            wn[li] += wn[li + s];
        }
        workgroupBarrier();
    }
    // less than 0.2 % of the samples' solid angle sky: nothing to measure, the gain stays
    if li == 0u && wo[0].w > 0.002 * f32(cnt) {
        let now = pow(max(wo[0].xyz, vec3f(1e-7)) / max(wn[0].xyz, vec3f(1e-7)), vec3f(sy.tone));
        let g = select(now, mix(ggain.xyz, now, sy.gw), ggain.w > 0.0);
        ggain = vec4f(clamp(g, vec3f(1.0 / 64.0), vec3f(16.0)), 1.0);
    }
}

@group(0) @binding(43) var obg: texture_2d<f32>;
@group(0) @binding(44) var oab: texture_2d<f32>;
@group(0) @binding(45) var opano: texture_2d<f32>;
@group(0) @binding(46) var osmp: sampler;
@group(0) @binding(47) var<storage, read> ogain: vec4f;
@group(0) @binding(48) var odst: texture_storage_2d<OUTFMT, write>;

// the composite: alpha from the coefficients at the pixel's world direction and the pixel's own colour, the panorama at
// that direction in its place (linear light)
fn ab(x: i32, y: i32) -> vec4f {
    let d = vec2i(textureDimensions(oab));
    return textureLoad(oab, vec2i((x % d.x + d.x) % d.x, clamp(y, 0, d.y - 1)), 0);
}

@compute @workgroup_size(8, 8)
fn skyo(@builtin(global_invocation_id) id: vec3u) {
    let d = textureDimensions(odst);
    if id.x >= d.x || id.y >= d.y {
        return;
    }
    let c = textureLoad(obg, id.xy, 0);
    let w = world(ray(vec2f(id.xy), d));
    // the coefficients bilinear (32-bit floats: no filtering sampler)
    let ad = vec2f(textureDimensions(oab));
    let t = uv(w, ad) * ad - 0.5;
    let t0 = floor(t);
    let fr = t - t0;
    let x = i32(t0.x);
    let y = i32(t0.y);
    let f = mix(mix(ab(x, y), ab(x + 1, y), fr.x), mix(ab(x, y + 1), ab(x + 1, y + 1), fr.x), fr.y);
    let a = clamp(dot(f.xyz, c.rgb) + f.w, 0.0, 1.0);
    let g = select(vec3f(1.0), ogain.xyz, ogain.w > 0.0);
    let s = lin(textureSampleLevel(opano, osmp, uv(w, vec2f(textureDimensions(opano))), 0.0).rgb) * g;
    // the ground stays as it was, to the bit
    textureStore(odst, id.xy, select(vec4f(enc(mix(lin(c.rgb), s, a)), c.a), c, a <= 0.0));
}
