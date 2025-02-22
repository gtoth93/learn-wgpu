struct SimulationData {
    clear_color: vec4<f32>,
    canvas_size: vec2<f32>,
    mouse_pos: vec2<f32>,
    time: vec2<f32>,
}

@group(0) @binding(0) var<uniform> simulation_data: SimulationData;

@fragment
fn main(@builtin(position) clip_position: vec4<f32>) -> @location(0) vec4<f32> {
    let t = simulation_data.time.x;
    let uv = ((clip_position.xyz * 0.5) + vec3(0.5));
    return mix(vec4<f32>(uv, 1.0), simulation_data.clear_color, vec4(sin(t)));
}
