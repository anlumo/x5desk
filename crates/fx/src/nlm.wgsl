// Non-local means on a still (lib.rs `Nlm`): `stat` histograms the frame's noise statistic, `solve` turns it into h²,
// `main` filters RGB with weights from luma patches.

struct Par {
    s: vec4u,  // width, height, 1: equirect (x wraps, the statistic skips the poles)
    k: vec4f,  // k², g0² (lib.rs `Nlm`)
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
// 0…255: the statistic's histogram, 256: h² (f32 bits)
@group(0) @binding(2) var<storage, read_write> HG: array<atomic<u32>, 257>;
@group(0) @binding(3) var<uniform> U: Par;

const W = vec3f(0.2126, 0.7152, 0.0722);
const S = 7;    // search radius
const P = 3;    // patch radius
const A = 10;   // tile apron, S + P
const TW = 36;  // tile side, 16 + 2A
const EW = 22;  // patch centres' side, 16 + 2P
// histogram: log2 of the statistic from L0, NB bins per unit
const L0 = -26.0;
const NB = 256.0 / 24.0;

fn ix(x: i32) -> i32 {
    let w = i32(U.s.x);
    return select(clamp(x, 0, w - 1), (x % w + w) % w, U.s.z != 0u);
}

fn lum(x: i32, y: i32) -> f32 {
    return dot(textureLoad(src, vec2i(ix(x), clamp(y, 0, i32(U.s.y) - 1)), 0).rgb, W);
}

var<workgroup> HL: array<atomic<u32>, 256>;

// The statistic at a grid every 4 px: the mean over a 5×5 patch of the squared luma differences 4 px right and 4 px
// down, where the patch's mean luma is 0.15…0.85 (deep shadows are smoothed by the camera, highlights clip)
@compute @workgroup_size(16, 16)
fn stat(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) li: u32) {
    atomicStore(&HL[li], 0u);
    workgroupBarrier();
    let s = vec2i(U.s.xy);
    let p = vec2i(id.xy) * 4 + 2;
    // equirect: 30…150° polar, the poles' rows are resampled from few lens pixels
    let r = select(vec2i(2, s.y - 6), vec2i(max(s.y / 6, 2), min(s.y - s.y / 6, s.y - 6)), U.s.z != 0u);
    if p.x < s.x - 6 && p.y >= r.x && p.y < r.y {
        var m = 0.0;
        var d = 0.0;
        for (var j = -2; j <= 2; j++) {
            for (var i = -2; i <= 2; i++) {
                let a = lum(p.x + i, p.y + j);
                let bx = a - lum(p.x + i + 4, p.y + j);
                let by = a - lum(p.x + i, p.y + j + 4);
                m += a;
                d += bx * bx + by * by;
            }
        }
        m /= 25.0;
        if m > 0.15 && m < 0.85 {
            atomicAdd(&HL[u32(clamp((log2(max(d / 50.0, 1e-12)) - L0) * NB, 0.0, 255.0))], 1u);
        }
    }
    workgroupBarrier();
    let v = atomicLoad(&HL[li]);
    if v > 0u { atomicAdd(&HG[li], v); }
}

// h² = k²·max(g² − g0², 0), g² = the statistic's 10th percentile; 0 without samples
@compute @workgroup_size(1)
fn solve() {
    var n = 0u;
    for (var i = 0u; i < 256u; i++) { n += atomicLoad(&HG[i]); }
    var c = 0u;
    var b = 0u;
    for (; b < 255u; b++) {
        c += atomicLoad(&HG[b]);
        if 10u * c >= n { break; }
    }
    let g2 = exp2((f32(b) + 0.5) / NB + L0);
    atomicStore(&HG[256], bitcast<u32>(select(0.0, U.k.x * max(g2 - U.k.y, 0.0), n > 0u)));
}

var<workgroup> TY: array<f32, 1296>;  // luma of the tile around the group's 16², stride TW
var<workgroup> TC: array<u32, 1296>;  // its RGB, packed (8-bit frames)
var<workgroup> E: array<f32, 484>;    // squared luma differences at the current offset, patch centres' rows
var<workgroup> R: array<f32, 352>;    // their sums over 2P + 1 columns, 16 per row
var<workgroup> H2: f32;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) g: vec3u, @builtin(local_invocation_id) l: vec3u,
        @builtin(local_invocation_index) li: u32) {
    if li == 0u { H2 = bitcast<f32>(atomicLoad(&HG[256])); }
    let o = vec2i(g.xy * 16u) - A;
    for (var i = i32(li); i < TW * TW; i += 256) {
        let c = textureLoad(src, vec2i(ix(o.x + i % TW), clamp(o.y + i / TW, 0, i32(U.s.y) - 1)), 0).rgb;
        TY[i] = dot(c, W);
        TC[i] = pack4x8unorm(vec4f(c, 0.0));
    }
    let h2 = workgroupUniformLoad(&H2);
    let c0 = (i32(l.y) + A) * TW + i32(l.x) + A;
    var acc = vec3f(0.0);
    var ws = 0.0;
    // the pixel's own weight: the largest other one (its own patch distance, 0, would outweigh the rest)
    var wm = 1e-6;
    if h2 > 0.0 {
        for (var dy = -S; dy <= S; dy++) {
            for (var dx = -S; dx <= S; dx++) {
                if dx == 0 && dy == 0 { continue; }
                for (var i = i32(li); i < EW * EW; i += 256) {
                    let a = (i / EW + S) * TW + i % EW + S;
                    let d = TY[a] - TY[a + dy * TW + dx];
                    E[i] = d * d;
                }
                workgroupBarrier();
                for (var i = i32(li); i < EW * 16; i += 256) {
                    let b = i / 16 * EW + i % 16;
                    R[i] = E[b] + E[b + 1] + E[b + 2] + E[b + 3] + E[b + 4] + E[b + 5] + E[b + 6];
                }
                workgroupBarrier();
                let k = i32(l.y) * 16 + i32(l.x);
                let d = (R[k] + R[k + 16] + R[k + 32] + R[k + 48] + R[k + 64] + R[k + 80] + R[k + 96]) / 49.0;
                let w = exp(-d / h2);
                acc += w * unpack4x8unorm(TC[c0 + dy * TW + dx]).rgb;
                ws += w;
                wm = max(wm, w);
                workgroupBarrier();
            }
        }
    }
    let p = vec2i(g.xy * 16u + l.xy);
    if p.x >= i32(U.s.x) || p.y >= i32(U.s.y) { return; }
    let c = textureLoad(src, p, 0);
    textureStore(dst, p, vec4f((acc + wm * c.rgb) / (ws + wm), c.a));
}
