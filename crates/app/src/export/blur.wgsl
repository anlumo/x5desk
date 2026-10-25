// Privacy blur on a flat view: crate track's `blur.wgsl` (an equirect) with the preview's projection (crate stitch,
// `ray` in `stitch.wgsl`), so regions found on the sphere blur where the view shows them. Draw 0: a full-screen copy.
// Draw 1: one quad per region (its bound in the view), whose fragments blend a disc blur over the region's ellipse,
// feathered at its edge.

struct Reg {
    // region forward, right, up (unit, world frame)
    f: vec4f,
    r: vec4f,
    u: vec4f,
    // ellipse half extents in the tangent plane, blur radius (tangent units)
    e: vec4f,
    // the quad: x0, y0, x1, y1 in pixels
    q: vec4f,
}

// r: view camera (x right, y up, z forward) -> world; target px; 0 perspective / Panini, 1 equirect, 2 planet; plane
// half-width f; Panini or planet distance d
struct View { r: mat3x3f, size: vec2f, mode: u32, f: f32, d: f32 }

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<storage, read> regs: array<Reg>;
@group(0) @binding(3) var<uniform> view: View;

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
    let q = regs[i].q;
    let c = vec2f(select(q.x, q.z, (v & 1u) != 0u), select(q.y, q.w, v >= 2u)) / view.size;
    return Out(vec4f(c.x * 2.0 - 1.0, 1.0 - c.y * 2.0, 0.0, 1.0), i);
}

// the world ray of pixel p (stitch's `ray` without the pose: regions are in the output world frame)
fn ray(p: vec2f) -> vec3f {
    let u = p / view.size;
    let x = vec2f(2.0 * u.x - 1.0, (1.0 - 2.0 * u.y) * view.size.y / view.size.x) * view.f;
    var r: vec3f;
    if view.mode == 1u {
        let lon = (2.0 * u.x - 1.0) * PI;
        let lat = (0.5 - u.y) * PI;
        r = vec3f(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
    } else if view.mode == 2u {
        let d = view.d;
        let p = length(x);
        let k = p * p / ((d + 1.0) * (d + 1.0));
        let e = (d + sqrt(k * (1.0 - d * d) + 1.0)) / (k + 1.0);
        r = vec3f(p * e / (d + 1.0) * normalize(x + vec2f(1e-12, 0.0)), e - d);
    } else {
        let d = view.d;
        let k = x.x * x.x / ((d + 1.0) * (d + 1.0));
        let cl = (-k * d + sqrt(k * k * d * d - (k + 1.0) * (k * d * d - 1.0))) / (k + 1.0);
        let s = (d + 1.0) / (d + cl);
        let lon = atan2(x.x, s * cl);
        let lat = atan2(x.y, s);
        r = vec3f(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
    }
    return normalize(view.r * r);
}

// texture coordinates of world direction w: `ray` inverted
fn uv(w: vec3f) -> vec2f {
    let c = transpose(view.r) * w;
    if view.mode == 1u {
        return vec2f(atan2(c.x, c.z) / (2.0 * PI) + 0.5, 0.5 - asin(clamp(c.y, -1.0, 1.0)) / PI);
    }
    let d = view.d;
    var x: vec2f;
    if view.mode == 2u {
        let t = acos(clamp(c.z, -1.0, 1.0));
        x = (d + 1.0) * sin(t) / (d + cos(t)) * normalize(c.xy + vec2f(1e-12, 0.0));
    } else {
        let lon = atan2(c.x, c.z);
        let s = (d + 1.0) / (d + cos(lon));
        x = vec2f(s * sin(lon), s * c.y / max(length(c.xz), 1e-6));
    }
    return vec2f(x.x / view.f + 1.0, 1.0 - x.y * view.size.x / (view.f * view.size.y)) * 0.5;
}

@fragment
fn blur(o: Out) -> @location(0) vec4f {
    let g = regs[o.i];
    let w = ray(o.p.xy);
    let z = dot(w, g.f.xyz);
    if z <= 0.0 { discard; }
    let d = length(vec2f(dot(w, g.r.xyz), dot(w, g.u.xyz)) / z / g.e.xy);
    let a = 1.0 - smoothstep(1.0, 1.25, d);
    if a <= 0.0 { discard; }
    // 12 × 12 taps over the disc, the grid turned per pixel (interleaved gradient noise), as crate track's blur
    let h = fract(52.9829189 * fract(dot(o.p.xy, vec2f(0.06711056, 0.00583715)))) * 2.0 * PI;
    let m = mat2x2f(cos(h), sin(h), -sin(h), cos(h));
    var c = vec4f(0.0);
    for (var k = 0u; k < 144u; k++) {
        let t = (vec2f(f32(k % 12u), f32(k / 12u)) + 0.5) / 6.0 - 1.0;
        if dot(t, t) > 1.0 { continue; }
        let s = m * t * g.e.z;
        let v = normalize(w + s.x * g.r.xyz + s.y * g.u.xyz);
        c += vec4f(textureSampleLevel(src, smp, uv(v), 0.0).rgb, 1.0);
    }
    return vec4f(c.rgb / c.w * a, a);
}
