// Perspective crops of an equirect in the stitcher's layout (column x at θ = 2πx/W, centre = −X; row y at
// φ = πy/(H − 1); Z up), written as planar (NCHW) f32 network input. One invocation per output pixel, z = job.

struct Job {
    // camera forward, right, up (unit, equirect frame)
    f: vec4f,
    r: vec4f,
    u: vec4f,
    // tangent-plane half extents x, y; output width, height
    e: vec4f,
    // channel c = colour[c] · a[c] + b[c], colour in network order
    a: vec4f,
    b: vec4f,
    // first element in dst, supersampling taps per axis, BGR order
    o: vec4u,
}

@group(0) @binding(0) var pano: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(1) @binding(0) var<storage, read> jobs: array<Job>;
@group(1) @binding(1) var<storage, read_write> dst: array<f32>;

const PI = 3.14159265358979;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let j = jobs[id.z];
    let s = vec2u(j.e.zw);
    if any(id.xy >= s) { return; }
    let n = vec2f(textureDimensions(pano));
    let k = j.o.y;
    var c = vec3f(0.0);
    // a k × k box over the output pixel: a crop that shrinks the equirect would alias with one tap
    for (var t = 0u; t < k * k; t++) {
        let p = (vec2f(id.xy) + (vec2f(f32(t % k), f32(t / k)) + 0.5) / f32(k)) / vec2f(s) * 2.0 - 1.0;
        let w = normalize(j.f.xyz + p.x * j.e.x * j.r.xyz - p.y * j.e.y * j.u.xyz);
        let q = vec2f(atan2(-w.y, w.x) / (2.0 * PI) * n.x, acos(clamp(w.z, -1.0, 1.0)) / PI * (n.y - 1.0)) + 0.5;
        c += textureSampleLevel(pano, smp, q / n, 0.0).rgb;
    }
    c /= f32(k * k);
    if j.o.z != 0u { c = c.bgr; }
    let m = s.x * s.y;
    let i = j.o.x + id.y * s.x + id.x;
    dst[i] = c.x * j.a.x + j.b.x;
    dst[i + m] = c.y * j.a.y + j.b.y;
    dst[i + 2u * m] = c.z * j.a.z + j.b.z;
}
