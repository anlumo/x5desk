// Overlays: a premultiplied RGBA image in linear light (an sRGB texture, mipmapped) drawn into the overlay layer
// (`Fx::run`), placed flat (frame pixels, rotated) or on the sphere (a tangent plane at a direction, drawn into an
// equirect: column x at yaw 360°·(x / W − ½), row y at pitch 90° − 180°·y / (H − 1), as the stitcher lays it out).

struct O {
    a: vec4f,  // flat: centre (px), cos, sin of the rotation; sphere: forward (xyz)
    b: vec4f,  // flat: half size (px); sphere: right / (2·tan(half width)) (xyz)
    c: vec4f,  // sphere: up / (2·tan(half height)) (xyz)
    d: vec4f,  // frame width, height, opacity, mode (0 flat, 1 sphere)
}

@group(0) @binding(0) var img: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> o: O;

const PI = 3.14159265358979;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    return vec4f(vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) q: vec4f) -> @location(0) vec4f {
    var uv: vec2f;
    var ok = true;
    if o.d.w == 0.0 {
        let d = q.xy - o.a.xy;
        uv = vec2f(dot(d, o.a.zw), dot(d, vec2f(-o.a.w, o.a.z))) / (2.0 * o.b.xy) + 0.5;
    } else {
        let lon = 2.0 * PI * ((q.x - 0.5) / o.d.x - 0.5);
        let lat = PI / 2.0 - PI * (q.y - 0.5) / (o.d.y - 1.0);
        let r = vec3f(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
        let z = dot(r, o.a.xyz);
        ok = z > 1e-4;
        uv = select(vec2f(-1.0), vec2f(dot(r, o.b.xyz), -dot(r, o.c.xyz)) / z + 0.5, ok);
    }
    // antialiased border: distance to the image edge in frame pixels. Far outside (and behind the tangent plane)
    // fwidth can be huge where uv jumps, so only the band next to the image counts.
    let s = vec2f(textureDimensions(img));
    let t = uv * s;
    let e = min(t, s - t) / max(fwidth(t), vec2f(1e-6));
    let a = clamp(min(e.x, e.y) + 0.5, 0.0, 1.0) * f32(ok && all(abs(uv - 0.5) < vec2f(1.0)));
    return textureSample(img, smp, uv) * (a * o.d.z);
}
