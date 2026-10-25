// A decoded NV12 / P010 picture to RGBA: the display window (x, y in the picture) through BT.709 limited range, as
// crate video encodes. Chroma by its 2 × 2 site (nearest).

@group(0) @binding(0) var y: texture_2d<f32>;
@group(0) @binding(1) var uv: texture_2d<f32>;
// window x, y; z: 1 for P010, whose R16 / RG16 plane views read the 10-bit code d as 64·d / 65535
@group(0) @binding(2) var<uniform> o: vec4u;

@vertex
fn vs(@builtin(vertex_index) v: u32) -> @builtin(position) vec4f {
    return vec4f(f32(v / 2u) * 4.0 - 1.0, f32(v % 2u) * 4.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
    let q = vec2u(p.xy) + o.xy;
    // the code in 8-bit units: d / 4 for 10 bits
    let m = select(255.0, 65535.0 / 256.0, o.z != 0u);
    let l = (textureLoad(y, q, 0).r * m - 16.0) / 219.0;
    let c = (textureLoad(uv, q / 2u, 0).rg * m - 128.0) / 224.0;
    return vec4f(l + 1.5748 * c.y, l - 0.1873 * c.x - 0.4681 * c.y, l + 1.8556 * c.x, 1.0);
}
