use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupLayout, Buffer, BufferUsages, Device, PipelineLayoutDescriptor,
    PrimitiveTopology, RenderPass, RenderPipeline, TextureFormat, VertexBufferLayout,
    VertexStepMode,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PositionColor {
    position: [f32; 3],
    color: [f32; 3],
}

const AXIS_COLORS: &[PositionColor] = &[
    // X
    PositionColor {
        position: [0.0, 0.0, 0.0],
        color: [0.5, 0.0, 0.0],
    },
    PositionColor {
        position: [1.0, 0.0, 0.0],
        color: [1.0, 0.0, 0.0],
    },
    // Y
    PositionColor {
        position: [0.0, 0.0, 0.0],
        color: [0.0, 0.5, 0.0],
    },
    PositionColor {
        position: [0.0, 1.0, 0.0],
        color: [0.0, 1.0, 0.0],
    },
    // Z
    PositionColor {
        position: [0.0, 0.0, 0.0],
        color: [0.0, 0.0, 0.5],
    },
    PositionColor {
        position: [0.0, 0.0, 1.0],
        color: [0.0, 0.0, 1.0],
    },
];

const POSITION_COLOR_LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
    array_stride: size_of::<PositionColor>() as _,
    step_mode: VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
    ],
};

pub struct Debug {
    color_lines: RenderPipeline,
    axis: Buffer,
}

impl Debug {
    pub fn new(
        device: &Device,
        camera_layout: &BindGroupLayout,
        color_format: TextureFormat,
    ) -> Self {
        let axis_desc = &BufferInitDescriptor {
            label: Some("Debug::axis"),
            contents: bytemuck::cast_slice(AXIS_COLORS),
            usage: BufferUsages::COPY_DST | BufferUsages::VERTEX,
        };
        let axis = device.create_buffer_init(axis_desc);

        let shader = wgpu::include_wgsl!("debug.wgsl");
        let layout_desc = &PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[camera_layout],
            push_constant_ranges: &[],
        };
        let layout = device.create_pipeline_layout(layout_desc);
        let color_lines = crate::create_render_pipeline(
            device,
            &layout,
            color_format,
            None,
            &[POSITION_COLOR_LAYOUT],
            PrimitiveTopology::LineList,
            shader,
        );

        Self { color_lines, axis }
    }

    pub fn draw_axis<'a: 'b, 'b>(&'a self, pass: &'b mut RenderPass<'a>, camera: &'a BindGroup) {
        pass.set_pipeline(&self.color_lines);
        pass.set_bind_group(0, camera, &[]);
        pass.set_vertex_buffer(0, self.axis.slice(..));
        pass.draw(0..u32::try_from(AXIS_COLORS.len()).unwrap(), 0..1);
    }
}
