use crate::texture::Texture;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, CommandEncoder, Device, FilterMode, LoadOp,
    Operations, PipelineLayoutDescriptor, PrimitiveTopology, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, SamplerBindingType, ShaderStages, StoreOp,
    SurfaceConfiguration, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDimension,
};

/// Owns the render texture and controls tonemapping
#[allow(clippy::module_name_repetitions)]
pub struct HdrPipeline {
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    texture: Texture,
    width: u32,
    height: u32,
    format: TextureFormat,
    layout: BindGroupLayout,
}

impl HdrPipeline {
    pub fn new(device: &Device, config: &SurfaceConfiguration) -> Self {
        let width = config.width;
        let height = config.height;

        // We could use `Rgba32Float`, but that requires some extra
        // features to be enabled.
        let format = TextureFormat::Rgba16Float;

        let texture = Texture::create_2d_texture(
            device,
            width,
            height,
            format,
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            FilterMode::Nearest,
            Some("Hdr::texture"),
        );

        let layout_desc = &BindGroupLayoutDescriptor {
            label: Some("Hdr::layout"),
            entries: &[
                // This is the HDR texture
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        // The Rgba16Float format cannot be filtered
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        };
        let layout = device.create_bind_group_layout(layout_desc);
        let bind_group_desc = &BindGroupDescriptor {
            label: Some("Hdr::bind_group"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&texture.view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&texture.sampler),
                },
            ],
        };
        let bind_group = device.create_bind_group(bind_group_desc);

        // We'll cover the shader next
        let shader = wgpu::include_wgsl!("hdr.wgsl");
        let pipeline_layout_desc = &PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        };
        let pipeline_layout = device.create_pipeline_layout(pipeline_layout_desc);

        let pipeline = crate::create_render_pipeline(
            device,
            &pipeline_layout,
            config.format.add_srgb_suffix(),
            None,
            // We'll use some math to generate the vertex data in
            // the shader, so we don't need any vertex buffers
            &[],
            PrimitiveTopology::TriangleList,
            shader,
        );

        Self {
            pipeline,
            bind_group,
            texture,
            width,
            height,
            format,
            layout,
        }
    }

    /// Resize the HDR texture
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        self.texture = Texture::create_2d_texture(
            device,
            width,
            height,
            TextureFormat::Rgba16Float,
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            FilterMode::Nearest,
            Some("Hdr::texture"),
        );
        let bind_group_desc = &BindGroupDescriptor {
            label: Some("Hdr::bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.texture.view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.texture.sampler),
                },
            ],
        };
        self.bind_group = device.create_bind_group(bind_group_desc);
        self.width = width;
        self.height = height;
    }

    /// Exposes the HDR texture
    pub fn view(&self) -> &TextureView {
        &self.texture.view
    }

    /// The format of the HDR texture
    pub fn format(&self) -> TextureFormat {
        self.format
    }

    /// This renders the internal HDR texture to the [`TextureView`]
    /// supplied as parameter.
    pub fn process(&self, encoder: &mut CommandEncoder, output: &TextureView) {
        let color_attachment = RenderPassColorAttachment {
            view: output,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            },
        };
        let pass_desc = &RenderPassDescriptor {
            label: Some("Hdr::process"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        };
        let mut pass = encoder.begin_render_pass(pass_desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
