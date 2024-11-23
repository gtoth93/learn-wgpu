#![warn(clippy::pedantic)]

use image::{ImageBuffer, Rgba};
use wgpu::{
    Backends, BlendState, BufferAddress, BufferDescriptor, BufferUsages, Color, ColorTargetState,
    ColorWrites, CommandEncoderDescriptor, DeviceDescriptor, Extent3d, Face, FragmentState,
    FrontFace, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, Instance, InstanceDescriptor,
    LoadOp, Maintain, MapMode, MultisampleState, Operations, Origin3d, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PolygonMode, PowerPreference, PrimitiveState, PrimitiveTopology,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
    RequestAdapterOptions, StoreOp, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureViewDescriptor, VertexState,
};

#[allow(clippy::too_many_lines)]
async fn run() {
    let instance_desc = InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    };
    let instance = Instance::new(instance_desc);

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&DeviceDescriptor::default(), None)
        .await
        .unwrap();

    let texture_size = 256u32;
    let texture_desc = TextureDescriptor {
        size: Extent3d {
            width: texture_size,
            height: texture_size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT,
        label: None,
        view_formats: &[],
    };
    let texture = device.create_texture(&texture_desc);
    let texture_view = texture.create_view(&TextureViewDescriptor::default());

    // we need to store this for later
    let u32_size = u32::try_from(size_of::<u32>()).unwrap();

    let output_buffer_size = BufferAddress::from(u32_size * texture_size * texture_size);
    let output_buffer_desc = BufferDescriptor {
        size: output_buffer_size,
        usage: BufferUsages::COPY_DST
            // this tells wpgu that we want to read this buffer from the cpu
            | BufferUsages::MAP_READ,
        label: None,
        mapped_at_creation: false,
    };
    let output_buffer = device.create_buffer(&output_buffer_desc);

    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    let render_pipeline_layout_desc = &PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    };
    let render_pipeline_layout = device.create_pipeline_layout(render_pipeline_layout_desc);

    let color_target = ColorTargetState {
        format: texture_desc.format,
        blend: Some(BlendState::REPLACE),
        write_mask: ColorWrites::ALL,
    };
    let fragment = FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        targets: &[Some(color_target)],
        compilation_options: PipelineCompilationOptions::default(),
    };
    let render_pipeline_desc = &RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: PipelineCompilationOptions::default(),
        },
        fragment: Some(fragment),
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
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
        cache: None,
    };
    let render_pipeline = device.create_render_pipeline(render_pipeline_desc);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });

    {
        let clear_color = Color {
            r: 0.1,
            g: 0.2,
            b: 0.3,
            a: 1.0,
        };
        let color_attachment = RenderPassColorAttachment {
            view: &texture_view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(clear_color),
                store: StoreOp::Store,
            },
        };
        let render_pass_desc = RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        };
        let mut render_pass = encoder.begin_render_pass(&render_pass_desc);

        render_pass.set_pipeline(&render_pipeline);
        render_pass.draw(0..3, 0..1);
    }

    encoder.copy_texture_to_buffer(
        ImageCopyTexture {
            aspect: TextureAspect::All,
            texture: &texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
        },
        ImageCopyBuffer {
            buffer: &output_buffer,
            layout: ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(u32_size * texture_size),
                rows_per_image: Some(texture_size),
            },
        },
        texture_desc.size,
    );

    queue.submit(Some(encoder.finish()));

    // We need to scope the mapping variables so that we can
    // unmap the buffer
    {
        let buffer_slice = output_buffer.slice(..);

        // NOTE: We have to create the mapping THEN device.poll() before await
        // the future. Otherwise the application will freeze.

        let (tx, rx) = oneshot::channel();
        let callback = move |result| {
            tx.send(result).unwrap();
        };
        buffer_slice.map_async(MapMode::Read, callback);
        device.poll(Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();

        let buffer =
            ImageBuffer::<Rgba<u8>, _>::from_raw(texture_size, texture_size, data).unwrap();
        buffer.save("image.png").unwrap();
    }
    output_buffer.unmap();
}

fn main() {
    pollster::block_on(run());
}
