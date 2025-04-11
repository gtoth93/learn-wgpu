use crate::model::Vertex;
use std::num::NonZeroU32;
use thiserror::Error;
use wgpu::{
    ColorTargetState, ColorWrites, CompareFunction, DepthBiasState, DepthStencilState, Device,
    Face, FragmentState, FrontFace, IndexFormat, MultisampleState, PipelineCompilationOptions,
    PipelineLayout, PolygonMode, PrimitiveState, PrimitiveTopology, RenderPipeline,
    RenderPipelineDescriptor, ShaderModule, ShaderModuleDescriptor, StencilState, TextureFormat,
    VertexBufferLayout, VertexState,
};

#[derive(Debug, Error)]
#[allow(clippy::enum_variant_names)]
pub enum Error {
    #[error("No pipeline layout supplied!")]
    NoPipelineLayout,
    #[error("No vertex shader supplied!")]
    NoVertexShader,
    #[error("No fragment shader supplied!")]
    NoFragmentShader,
}

pub struct RenderPipelineBuilder<'a> {
    layout: Option<&'a PipelineLayout>,
    vertex_shader: Option<ShaderModuleDescriptor<'a>>,
    fragment_shader: Option<ShaderModuleDescriptor<'a>>,
    front_face: FrontFace,
    cull_mode: Option<Face>,
    depth_bias: i32,
    depth_bias_slope_scale: f32,
    depth_bias_clamp: f32,
    primitive_topology: PrimitiveTopology,
    color_states: Vec<Option<ColorTargetState>>,
    depth_stencil: Option<DepthStencilState>,
    index_format: IndexFormat,
    vertex_buffers: Vec<VertexBufferLayout<'a>>,
    sample_count: u32,
    sample_mask: u64,
    alpha_to_coverage_enabled: bool,
    multiview: Option<NonZeroU32>,
}

impl Default for RenderPipelineBuilder<'_> {
    fn default() -> Self {
        Self {
            layout: None,
            vertex_shader: None,
            fragment_shader: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
            primitive_topology: PrimitiveTopology::TriangleList,
            color_states: Vec::new(),
            depth_stencil: None,
            index_format: IndexFormat::Uint32,
            vertex_buffers: Vec::new(),
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
            multiview: None,
        }
    }
}

impl<'a> RenderPipelineBuilder<'a> {
    pub fn layout(&mut self, layout: &'a PipelineLayout) -> &mut Self {
        self.layout = Some(layout);
        self
    }

    pub fn vertex_shader(&mut self, src: ShaderModuleDescriptor<'a>) -> &mut Self {
        self.vertex_shader = Some(src);
        self
    }

    pub fn fragment_shader(&mut self, src: ShaderModuleDescriptor<'a>) -> &mut Self {
        self.fragment_shader = Some(src);
        self
    }

    #[allow(dead_code)]
    pub fn front_face(&mut self, ff: FrontFace) -> &mut Self {
        self.front_face = ff;
        self
    }

    #[allow(dead_code)]
    pub fn cull_mode(&mut self, cm: Option<Face>) -> &mut Self {
        self.cull_mode = cm;
        self
    }

    #[allow(dead_code)]
    pub fn depth_bias(&mut self, db: i32) -> &mut Self {
        self.depth_bias = db;
        self
    }

    #[allow(dead_code)]
    pub fn depth_bias_slope_scale(&mut self, dbss: f32) -> &mut Self {
        self.depth_bias_slope_scale = dbss;
        self
    }

    #[allow(dead_code)]
    pub fn depth_bias_clamp(&mut self, dbc: f32) -> &mut Self {
        self.depth_bias_clamp = dbc;
        self
    }

    #[allow(dead_code)]
    pub fn primitive_topology(&mut self, pt: PrimitiveTopology) -> &mut Self {
        self.primitive_topology = pt;
        self
    }

    pub fn color_state(&mut self, cs: ColorTargetState) -> &mut Self {
        self.color_states.push(Some(cs));
        self
    }

    /// Helper method for [`RenderPipelineBuilder::color_state`]
    pub fn color_solid(&mut self, format: TextureFormat) -> &mut Self {
        let color_state = ColorTargetState {
            format,
            blend: None,
            write_mask: ColorWrites::ALL,
        };
        self.color_state(color_state)
    }

    pub fn depth_stencil(&mut self, dss: DepthStencilState) -> &mut Self {
        self.depth_stencil = Some(dss);
        self
    }

    /// Helper method for [`RenderPipelineBuilder::depth_stencil`]
    pub fn depth_no_stencil(
        &mut self,
        format: TextureFormat,
        depth_write_enabled: bool,
        depth_compare: CompareFunction,
    ) -> &mut Self {
        let depth_stencil = DepthStencilState {
            format,
            depth_write_enabled,
            depth_compare,
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        };
        self.depth_stencil(depth_stencil)
    }

    /// Helper method for [`RenderPipelineBuilder::depth_no_stencil`]
    pub fn depth_format(&mut self, format: TextureFormat) -> &mut Self {
        self.depth_no_stencil(format, true, CompareFunction::Less)
    }

    #[allow(dead_code)]
    pub fn index_format(&mut self, ifmt: IndexFormat) -> &mut Self {
        self.index_format = ifmt;
        self
    }

    pub fn vertex_buffer<V: Vertex>(&mut self) -> &mut Self {
        self.vertex_buffers.push(V::desc());
        self
    }

    pub fn vertex_buffer_desc(&mut self, vb: VertexBufferLayout<'a>) -> &mut Self {
        self.vertex_buffers.push(vb);
        self
    }

    #[allow(dead_code)]
    pub fn sample_count(&mut self, sc: u32) -> &mut Self {
        self.sample_count = sc;
        self
    }

    #[allow(dead_code)]
    pub fn sample_mask(&mut self, sm: u64) -> &mut Self {
        self.sample_mask = sm;
        self
    }

    #[allow(dead_code)]
    pub fn alpha_to_coverage_enabled(&mut self, atce: bool) -> &mut Self {
        self.alpha_to_coverage_enabled = atce;
        self
    }

    pub fn multiview(&mut self, value: Option<NonZeroU32>) -> &mut Self {
        self.multiview = value;
        self
    }

    /// # Errors
    ///
    /// Will return `Err` if there is no pipeline layout or if there is no vertex shader or fragment shader.
    pub fn build(&mut self, device: &Device) -> Result<RenderPipeline, Error> {
        // We need a layout
        let layout = self.layout.ok_or(Error::NoPipelineLayout)?;

        // Render pipelines always have a vertex shader, but due
        // to the way the builder pattern works, we can't
        // guarantee that the user will specify one, so we'll
        // just return an error if they forgot.
        //
        // We could supply a default one, but a "default" vertex
        // could take on many forms. An error is much more
        // explicit.
        let vs = create_shader_module(
            device,
            self.vertex_shader.take().ok_or(Error::NoVertexShader)?,
        );

        // The fragment shader is optional (IDK why, but it is).
        // Having the shader be optional is giving me issues with
        // the borrow checker so I'm going to use a default shader
        // if the user doesn't supply one.
        let fs = create_shader_module(
            device,
            self.fragment_shader.take().ok_or(Error::NoFragmentShader)?,
        );

        let fragment = FragmentState {
            module: &fs,
            entry_point: Some("main"),
            targets: &self.color_states,
            compilation_options: PipelineCompilationOptions::default(),
        };
        let pipeline_desc = &RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(layout),
            vertex: VertexState {
                module: &vs,
                entry_point: Some("main"),
                buffers: &self.vertex_buffers,
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(fragment),
            primitive: PrimitiveState {
                topology: self.primitive_topology,
                front_face: self.front_face,
                cull_mode: self.cull_mode,
                strip_index_format: None,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: self.depth_stencil.clone(),
            multisample: MultisampleState {
                count: self.sample_count,
                mask: self.sample_mask,
                alpha_to_coverage_enabled: self.alpha_to_coverage_enabled,
            },
            multiview: self.multiview,
            cache: None,
        };
        let pipeline = device.create_render_pipeline(pipeline_desc);
        Ok(pipeline)
    }
}

fn create_shader_module(device: &Device, desc: ShaderModuleDescriptor) -> ShaderModule {
    device.create_shader_module(desc)
}
