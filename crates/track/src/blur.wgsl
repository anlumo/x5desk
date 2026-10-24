// Privacy blur on an equirect in the stitcher's layout (column x at θ = 2πx/W, centre −X; row y at φ = πy/(H − 1);
// Z up). Draw 0: a full-screen copy. Draw 1: one quad per region (its bound in the equirect, split at the seam), whose
// fragments blend a disc blur over the region's ellipse, feathered at its edge.

struct Reg {
    // region forward, right, up (unit, equirect frame)
    f: vec4f,
    r: vec4f,
    u: vec4f,
    // ellipse half extents in the tangent plane, blur radius (tangent units)
    e: vec4f,
    // the quad: x0, y0, x1, y1 in pixels
    q: vec4f,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<storage, read> regs: array<Reg>;

const PI = 3.14159265358979;

struct Out {
    @builtin(position) p: vec4f,
    @location(0) @interpolate(flat) i: u32,
}

@vertex
fn full(@builtin(vertex_index) v: u32) -> @builtin(position) vec4f {
    return vec4f(f32(v / 2u) * 4.0 - 1.0, f32(v % 2u) * 4.0 - 1.0, 0.0, 1.0);
}

@fragment
fn copy(@builtin(position) p: vec4f) -> @location(0) vec4f {
    return textureLoad(src, vec2i(p.xy), 0);
}

@vertex
fn quad(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Out {
    let n = vec2f(textureDimensions(src));
    let q = regs[i].q;
    let c = vec2f(select(q.x, q.z, (v & 1u) != 0u), select(q.y, q.w, v >= 2u)) / n;
    return Out(vec4f(c.x * 2.0 - 1.0, 1.0 - c.y * 2.0, 0.0, 1.0), i);
}

fn dir(p: vec2f, n: vec2f) -> vec3f {
    let t = 2.0 * PI * (p.x - 0.5) / n.x;
    let f = PI * clamp((p.y - 0.5) / (n.y - 1.0), 0.0, 1.0);
    return vec3f(sin(f) * cos(t), -sin(f) * sin(t), cos(f));
}

fn uv(w: vec3f, n: vec2f) -> vec2f {
    return (vec2f(atan2(-w.y, w.x) / (2.0 * PI) * n.x, acos(clamp(w.z, -1.0, 1.0)) / PI * (n.y - 1.0)) + 0.5) / n;
}

@fragment
fn blur(o: Out) -> @location(0) vec4f {
    let g = regs[o.i];
    let n = vec2f(textureDimensions(src));
    let w = dir(o.p.xy, n);
    let z = dot(w, g.f.xyz);
    if z <= 0.0 { discard; }
    let d = length(vec2f(dot(w, g.r.xyz), dot(w, g.u.xyz)) / z / g.e.xy);
    let a = 1.0 - smoothstep(1.0, 1.25, d);
    if a <= 0.0 { discard; }
    // 12 × 12 taps over the disc, the grid turned per pixel (interleaved gradient noise): a fixed grid over a large
    // face leaves shifted copies of its features, a turned one only grain
    let h = fract(52.9829189 * fract(dot(o.p.xy, vec2f(0.06711056, 0.00583715)))) * 2.0 * PI;
    let m = mat2x2f(cos(h), sin(h), -sin(h), cos(h));
    var c = vec4f(0.0);
    for (var k = 0u; k < 144u; k++) {
        let t = (vec2f(f32(k % 12u), f32(k / 12u)) + 0.5) / 6.0 - 1.0;
        if dot(t, t) > 1.0 { continue; }
        let s = m * t * g.e.z;
        let v = normalize(w + s.x * g.r.xyz + s.y * g.u.xyz);
        c += vec4f(textureSampleLevel(src, smp, uv(v, n), 0.0).rgb, 1.0);
    }
    return vec4f(c.rgb / c.w * a, a);
}
