//! Features
//! - [ ] Support fullscreen drawing
//! - [ ] Data struct for basic uniforms (time, mousePos, etc.)
//! - [ ] Lambda support for other bind groups
//! - [ ] Drawing to texture (maybe have the render pass decide this?)
//! - [ ] Saving to file

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use web_time::Instant;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendState, Buffer, BufferBindingType,
    BufferUsages, ColorTargetState, ColorWrites, CommandEncoder, Device, Face, FragmentState,
    FrontFace, LoadOp, MultisampleState, Operations, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderStages, StoreOp, SurfaceConfiguration, TextureFormat,
    TextureView, VertexState,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct SimulationData {
    clear_color: [f32; 4],
    canvas_size: [f32; 2],
    mouse_pos: [f32; 2],
    time: f32,
    delta_time: f32,
}

#[derive(Error, Debug)]
pub enum ShaderBuildError {
    #[error("Please supply a valid vertex shader")]
    InvalidVertexShader,
    #[error("Please supply a valid fragment shader")]
    InvalidFragmentShader,
    #[error("Please supply a valid display format")]
    InvalidDisplayFormat,
}

pub struct ShaderCanvas {
    pipeline: RenderPipeline,
    start_time: Option<Instant>,
    last_time: Option<Instant>,
    simulation_data: SimulationData,
    simulation_data_buffer: Buffer,
    simulation_bind_group: BindGroup,
}

impl ShaderCanvas {
    pub fn input(&mut self, mouse_x: f32, mouse_y: f32) {
        self.simulation_data.mouse_pos[0] = mouse_x;
        self.simulation_data.mouse_pos[1] = mouse_y;
    }

    pub fn delta_input(&mut self, dx: f32, dy: f32) {
        self.simulation_data.mouse_pos[0] += dx;
        self.simulation_data.mouse_pos[1] += dy;
    }

    pub fn render(
        &mut self,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        frame: &TextureView,
        width: f32,
        height: f32,
    ) {
        let current_time = Instant::now();
        let start_time = if let Some(t) = self.start_time {
            t
        } else {
            let t = current_time;
            self.start_time = Some(t);
            t
        };
        let last_time = self.last_time.unwrap_or(current_time);
        self.last_time = Some(current_time);
        self.simulation_data.time = (current_time - start_time).as_secs_f32();
        self.simulation_data.delta_time = (current_time - last_time).as_secs_f32();
        self.simulation_data.canvas_size[0] = width;
        self.simulation_data.canvas_size[1] = height;
        queue.write_buffer(
            &self.simulation_data_buffer,
            0,
            bytemuck::cast_slice(&[self.simulation_data]),
        );

        let color_attachment = RenderPassColorAttachment {
            view: frame,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
        };
        let render_pass_desc = &RenderPassDescriptor {
            label: Some("Shader Canvas Render Pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        };
        let mut pass = encoder.begin_render_pass(render_pass_desc);
        pass.set_bind_group(0, &self.simulation_bind_group, &[]);
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..6, 0..1);
    }
}

pub struct ShaderCanvasBuilder<'a> {
    canvas_size: [f32; 2],
    clear_color: [f32; 4],
    label: Option<&'a str>,
    display_format: Option<TextureFormat>,
    frag_code: Option<ShaderModuleDescriptor<'a>>,
    vert_code: Option<ShaderModuleDescriptor<'a>>,
}

impl<'a> Default for ShaderCanvasBuilder<'a> {
    fn default() -> Self {
        Self {
            canvas_size: [256.0; 2],
            clear_color: [0.0, 0.0, 0.0, 1.0],
            label: None,
            display_format: None,
            frag_code: Some(wgpu::include_wgsl!("shader_canvas.frag.wgsl")),
            vert_code: Some(wgpu::include_wgsl!("shader_canvas.vert.wgsl")),
        }
    }
}

impl<'a> ShaderCanvasBuilder<'a> {
    pub fn canvas_size(&mut self, width: f32, height: f32) -> &mut Self {
        self.canvas_size = [width, height];
        self
    }

    pub fn display_format(&mut self, format: TextureFormat) -> &mut Self {
        self.display_format = Some(format);
        self
    }

    pub fn use_swap_chain_desc(&mut self, config: &SurfaceConfiguration) -> &mut Self {
        self.display_format(config.format);
        self.canvas_size(config.width as f32, config.height as f32)
    }

    pub fn fragment_shader(&mut self, code: ShaderModuleDescriptor<'a>) -> &mut Self {
        self.frag_code = Some(code);
        self
    }

    pub fn vertex_shader(&mut self, code: ShaderModuleDescriptor<'a>) -> &mut Self {
        self.vert_code = Some(code);
        self
    }

    /// # Errors
    ///
    /// Will return `Err` if there is no display format or if there is no fragment or vertex shader.
    #[allow(clippy::too_many_lines)]
    pub fn build(&mut self, device: &Device) -> Result<ShaderCanvas, ShaderBuildError> {
        let display_format = self
            .display_format
            .ok_or(ShaderBuildError::InvalidDisplayFormat)?;
        let frag_code = self
            .frag_code
            .take()
            .ok_or(ShaderBuildError::InvalidFragmentShader)?;
        let vert_code = self
            .vert_code
            .take()
            .ok_or(ShaderBuildError::InvalidVertexShader)?;

        let simulation_data = SimulationData {
            time: 0.0,
            delta_time: 0.0,
            mouse_pos: [0.0; 2],
            canvas_size: self.canvas_size,
            clear_color: self.clear_color,
        };
        let simulation_data_slice = &[simulation_data];
        let simulation_data_buffer_desc = &BufferInitDescriptor {
            label: self.label,
            contents: bytemuck::cast_slice(simulation_data_slice),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        };
        let simulation_data_buffer = device.create_buffer_init(simulation_data_buffer_desc);

        let simulation_bind_group_layout_desc = &BindGroupLayoutDescriptor {
            label: self.label,
            entries: &[
                // SimulationData
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    count: None,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ],
        };
        let simulation_bind_group_layout =
            device.create_bind_group_layout(simulation_bind_group_layout_desc);
        let bind_group_entry = BindGroupEntry {
            binding: 0,
            resource: simulation_data_buffer.as_entire_binding(),
        };
        let simulation_bind_group_desc = &BindGroupDescriptor {
            layout: &simulation_bind_group_layout,
            label: self.label,
            entries: &[bind_group_entry],
        };
        let simulation_bind_group = device.create_bind_group(simulation_bind_group_desc);

        let vert_module = device.create_shader_module(vert_code);
        let frag_module = device.create_shader_module(frag_code);

        let pipeline_layout_desc = &PipelineLayoutDescriptor {
            label: self.label,
            bind_group_layouts: &[&simulation_bind_group_layout],
            push_constant_ranges: &[],
        };
        let pipeline_layout = device.create_pipeline_layout(pipeline_layout_desc);
        let color_target = ColorTargetState {
            format: display_format,
            blend: Some(BlendState::REPLACE),
            write_mask: ColorWrites::ALL,
        };
        let fragment = FragmentState {
            entry_point: Some("main"),
            module: &frag_module,
            targets: &[Some(color_target)],
            compilation_options: PipelineCompilationOptions::default(),
        };
        let pipeline_desc = &RenderPipelineDescriptor {
            label: self.label,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                entry_point: Some("main"),
                module: &vert_module,
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(fragment),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
            // Useful for optimizing shader compilation on Android
            cache: None,
        };
        let pipeline = device.create_render_pipeline(pipeline_desc);

        let shader_canvas = ShaderCanvas {
            pipeline,
            start_time: None,
            last_time: None,
            simulation_data,
            simulation_data_buffer,
            simulation_bind_group,
        };
        Ok(shader_canvas)
    }
}
