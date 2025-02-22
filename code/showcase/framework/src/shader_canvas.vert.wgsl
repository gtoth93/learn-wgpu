@vertex
fn main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((((in_vertex_index + 2) / 3) % 2));
    let y = f32((((in_vertex_index + 1) / 3) % 2));
    return vec4<f32>(-1.0 + x * 2.0, -1.0 + y * 2.0, 0.0, 1.0);
}
