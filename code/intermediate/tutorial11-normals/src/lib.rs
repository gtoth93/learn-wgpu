#![warn(clippy::pedantic)]

mod model;
mod resources;
mod texture;

use crate::{
    model::{DrawLight, DrawModel, Material, Model, ModelVertex, Vertex},
    texture::Texture,
};
use anyhow::Result;
use glam::{Mat3, Mat4, Quat, Vec3, Vec3A, Vec4};
use std::sync::Arc;
use tracing::Level;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendState, Buffer, BufferAddress, BufferBindingType,
    BufferUsages, Color, ColorTargetState, ColorWrites, CommandEncoderDescriptor, CompareFunction,
    DepthBiasState, DepthStencilState, Device, DeviceDescriptor, Face, Features, FragmentState,
    FrontFace, InstanceDescriptor, Limits, LoadOp, MemoryHints, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, PolygonMode,
    PowerPreference, PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline,
    RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType, ShaderModuleDescriptor,
    ShaderStages, StencilState, StoreOp, Surface, SurfaceConfiguration, SurfaceError,
    TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension,
    VertexAttribute, VertexBufferLayout, VertexState, VertexStepMode,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const fn degrees_to_radians(degrees: f32) -> f32 {
    degrees * std::f32::consts::PI / 180.0
}

const NUM_INSTANCES_PER_ROW: u16 = 10;
const SPACE_BETWEEN: f32 = 3.0;

#[derive(Debug)]
struct Camera {
    eye: Vec3A,
    target: Vec3A,
    up: Vec3A,
    aspect: f32,
    fov_y: f32,
    z_near: f32,
    z_far: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye.into(), self.target.into(), self.up.into());
        let proj = Mat4::perspective_rh(self.fov_y, self.aspect, self.z_near, self.z_far);
        proj * view
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_position: Vec4,
    view_proj: Mat4,
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: Vec4::ZERO,
            view_proj: Mat4::IDENTITY,
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        // We're using Vec4 because of the uniforms' 16-byte spacing requirement
        self.view_position = camera.eye.extend(1.0);
        self.view_proj = camera.build_view_projection_matrix();
    }
}

#[allow(clippy::struct_excessive_bools)]
struct CameraController {
    speed: f32,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_up_pressed: false,
            is_down_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::Space => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    KeyCode::ShiftLeft => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyW | KeyCode::ArrowUp => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.length();

        // Prevents glitching when the camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the forward/backward is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.length();

        if self.is_right_pressed {
            // Rescale the distance between the target and the eye so
            // that it doesn't change. The eye, therefore, still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}

struct Instance {
    position: Vec3A,
    rotation: Quat,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: Mat4::from_rotation_translation(self.rotation, self.position.into())
                .to_cols_array_2d(),
            normal: Mat3::from_quat(self.rotation).to_cols_array_2d(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

impl InstanceRaw {
    // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
    // for each vec4. We'll have to reassemble the mat4 in the shader.
    // While our vertex shader only uses locations 0, and 1 now, in later tutorials, we'll
    // be using 2, 3, and 4, for Vertex. We'll start at slot 5, not conflict with them later
    const ATTRIBS: &'static [VertexAttribute] = &wgpu::vertex_attr_array![
        5 => Float32x4,
        6 => Float32x4,
        7 => Float32x4,
        8 => Float32x4,
        9 => Float32x3,
        10 => Float32x3,
        11 => Float32x3,
    ];

    fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: size_of::<InstanceRaw>() as BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: VertexStepMode::Instance,
            attributes: Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: Vec3,
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding: u32,
    color: Vec3,
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding2: u32,
}

fn create_render_pipeline(
    device: &Device,
    layout: &PipelineLayout,
    color_format: TextureFormat,
    depth_format: Option<TextureFormat>,
    vertex_layouts: &[VertexBufferLayout],
    shader: ShaderModuleDescriptor,
) -> RenderPipeline {
    let shader = device.create_shader_module(shader);
    let color_target = ColorTargetState {
        format: color_format,
        blend: Some(BlendState::REPLACE),
        write_mask: ColorWrites::ALL,
    };
    let fragment = FragmentState {
        module: &shader,
        entry_point: Some("fs_main"),
        targets: &[Some(color_target)],
        compilation_options: PipelineCompilationOptions::default(),
    };
    let depth_format_map_fn = |format| DepthStencilState {
        format,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Less,
        stencil: StencilState::default(),
        bias: DepthBiasState::default(),
    };
    let render_pipeline_desc = RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: vertex_layouts,
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
        depth_stencil: depth_format.map(depth_format_map_fn),
        multisample: MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        // If the pipeline is used with a multiview render pass, this
        // indicates how many array layers the attachments will have.
        multiview: None,
        // Useful for optimizing shader compilation on Android
        cache: None,
    };
    device.create_render_pipeline(&render_pipeline_desc)
}

struct State {
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    size: PhysicalSize<u32>,
    window: Arc<Window>,
    surface_configured: bool,
    render_pipeline: RenderPipeline,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: Buffer,
    camera_bind_group: BindGroup,
    camera_controller: CameraController,
    instances: Vec<Instance>,
    instance_buffer: Buffer,
    depth_texture: Texture,
    obj_model: Model,
    light_uniform: LightUniform,
    light_buffer: Buffer,
    light_bind_group: BindGroup,
    light_render_pipeline: RenderPipeline,
    #[allow(dead_code)]
    debug_material: Material, // NEW!
}

impl State {
    #[allow(clippy::too_many_lines)]
    async fn new(window: Arc<Window>) -> State {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        tracing::warn!("WGPU setup");
        let instance_desc = InstanceDescriptor {
            backends: if cfg!(not(target_arch = "wasm32")) {
                Backends::PRIMARY
            } else {
                Backends::GL
            },
            ..Default::default()
        };
        let instance = wgpu::Instance::new(instance_desc);

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        tracing::warn!("device and queue");
        let device_desc = DeviceDescriptor {
            label: None,
            required_features: Features::empty(),
            // WebGL doesn't support all of wgpu's features, so if
            // we're building for the web, we'll have to disable some.
            required_limits: if cfg!(target_arch = "wasm32") {
                Limits::downlevel_webgl2_defaults()
            } else {
                Limits::default()
            },
            memory_hints: MemoryHints::default(),
        };
        let (device, queue) = adapter.request_device(&device_desc, None).await.unwrap();

        tracing::warn!("Surface");
        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an Srgb surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // Srgb surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(TextureFormat::is_srgb)
            .unwrap_or(surface_caps.formats[0]);
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };

        let surface_configured = if cfg!(not(target_arch = "wasm32")) {
            surface.configure(&device, &config);
            true
        } else {
            false
        };

        let texture_bind_group_layout_desc = BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2,
                        sample_type: TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    // This should match the filterable field of the
                    // corresponding Texture entry above.
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // NEW!
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("texture_bind_group_layout"),
        };
        let texture_bind_group_layout =
            device.create_bind_group_layout(&texture_bind_group_layout_desc);

        let camera = Camera {
            // Camera position is 5 units up and 10 units back
            // +z is out of the screen
            eye: Vec3A::new(0.0, 5.0, 10.0),
            // Camera looks at the origin
            target: Vec3A::ZERO,
            // Unit vector which points upwards
            up: Vec3A::Y,
            #[allow(clippy::cast_precision_loss)]
            aspect: config.width as f32 / config.height as f32,
            fov_y: degrees_to_radians(45.0),
            z_near: 0.1,
            z_far: 100.0,
        };
        let camera_controller = CameraController::new(0.0125);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_uniform_slice = &[camera_uniform];
        let camera_buffer_desc = BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(camera_uniform_slice),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        };
        let camera_buffer = device.create_buffer_init(&camera_buffer_desc);

        let instances = (0..NUM_INSTANCES_PER_ROW)
            .flat_map(|z| {
                let instance_fn = move |x| {
                    let x = SPACE_BETWEEN * (f32::from(x) - f32::from(NUM_INSTANCES_PER_ROW) / 2.0);
                    let z = SPACE_BETWEEN * (f32::from(z) - f32::from(NUM_INSTANCES_PER_ROW) / 2.0);

                    let position = Vec3A::new(x, 0.0, z);

                    let rotation = if position == Vec3A::ZERO {
                        // this is needed so that an object at (0, 0, 0) won't get scaled to zero
                        // as Quaternions can affect scale if they're not created correctly
                        Quat::from_axis_angle(Vec3A::Z.into(), 0.0)
                    } else {
                        Quat::from_axis_angle(position.normalize().into(), degrees_to_radians(45.0))
                    };

                    Instance { position, rotation }
                };
                (0..NUM_INSTANCES_PER_ROW).map(instance_fn)
            })
            .collect::<Vec<_>>();

        let instance_data = instances.iter().map(Instance::to_raw).collect::<Vec<_>>();
        let instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: BufferUsages::VERTEX,
        });

        let camera_bind_group_layout_entry = BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let camera_bind_group_layout_desc = BindGroupLayoutDescriptor {
            entries: &[camera_bind_group_layout_entry],
            label: Some("camera_bind_group_layout"),
        };
        let camera_bind_group_layout =
            device.create_bind_group_layout(&camera_bind_group_layout_desc);

        let camera_bind_group_entry = BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        };
        let camera_bind_group_desc = BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[camera_bind_group_entry],
            label: Some("camera_bind_group"),
        };
        let camera_bind_group = device.create_bind_group(&camera_bind_group_desc);

        tracing::warn!("Load model");
        let obj_model =
            resources::load_model("cube.obj", &device, &queue, &texture_bind_group_layout)
                .await
                .unwrap();

        let light_uniform = LightUniform {
            position: Vec3::new(2.0, 2.0, 2.0),
            _padding: 0,
            color: Vec3::new(1.0, 1.0, 1.0),
            _padding2: 0,
        };

        // We'll want to update our light's position, so we use COPY_DST
        let light_uniform_slice = &[light_uniform];
        let light_buffer_desc = BufferInitDescriptor {
            label: Some("Light VB"),
            contents: bytemuck::cast_slice(light_uniform_slice),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        };
        let light_buffer = device.create_buffer_init(&light_buffer_desc);

        let light_bind_group_layout_entry = BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let light_bind_group_layout_desc = BindGroupLayoutDescriptor {
            entries: &[light_bind_group_layout_entry],
            label: None,
        };
        let light_bind_group_layout =
            device.create_bind_group_layout(&light_bind_group_layout_desc);

        let light_bind_group_entry = BindGroupEntry {
            binding: 0,
            resource: light_buffer.as_entire_binding(),
        };
        let light_bind_group_desc = BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[light_bind_group_entry],
            label: None,
        };
        let light_bind_group = device.create_bind_group(&light_bind_group_desc);

        let depth_texture = Texture::create_depth_texture(&device, &config, "depth_texture");

        let pipeline_layout_desc = PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &camera_bind_group_layout,
                &light_bind_group_layout,
            ],
            push_constant_ranges: &[],
        };
        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_desc);

        let render_pipeline = create_render_pipeline(
            &device,
            &pipeline_layout,
            config.format,
            Some(Texture::DEPTH_FORMAT),
            &[ModelVertex::desc(), InstanceRaw::desc()],
            wgpu::include_wgsl!("shader.wgsl"),
        );

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(Texture::DEPTH_FORMAT),
                &[ModelVertex::desc()],
                wgpu::include_wgsl!("light.wgsl"),
            )
        };

        // NEW!
        let debug_material = {
            let diffuse_bytes = include_bytes!("../res/cobble-diffuse.png");
            let normal_bytes = include_bytes!("../res/cobble-normal.png");

            let diffuse_texture =
                Texture::from_bytes(&device, &queue, diffuse_bytes, "res/alt-diffuse.png", false)
                    .unwrap();
            let normal_texture =
                Texture::from_bytes(&device, &queue, normal_bytes, "res/alt-normal.png", true)
                    .unwrap();

            Material::new(
                &device,
                "alt-material",
                diffuse_texture,
                normal_texture,
                &texture_bind_group_layout,
            )
        };

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            surface_configured,
            render_pipeline,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,
            instances,
            instance_buffer,
            depth_texture,
            obj_model,
            light_uniform,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
            debug_material, // NEW!
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            #[allow(clippy::cast_precision_loss)]
            {
                self.camera.aspect = self.config.width as f32 / self.config.height as f32;
            }
            self.depth_texture =
                Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        // tracing::info!("{:?}", self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        // tracing::info!("{:?}", self.camera_uniform);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update the light
        let old_position: Vec3A = self.light_uniform.position.into();
        self.light_uniform.position =
            (Quat::from_axis_angle(Vec3::Y, degrees_to_radians(0.0625)) * old_position).into();
        self.queue.write_buffer(
            &self.light_buffer,
            0,
            bytemuck::cast_slice(&[self.light_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let clear_color = Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            };
            // This is what @location(0) in the fragment shader targets
            let color_attachment = RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(clear_color),
                    store: StoreOp::Store,
                },
            };
            let depth_ops = Operations {
                load: LoadOp::Clear(1.0),
                store: StoreOp::Store,
            };
            let depth_stencil_attachment = RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(depth_ops),
                stencil_ops: None,
            };
            let render_pass_desc = RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: Some(depth_stencil_attachment),
                occlusion_query_set: None,
                timestamp_writes: None,
            };
            let mut render_pass = encoder.begin_render_pass(&render_pass_desc);

            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(&self.light_render_pipeline);

            render_pass.draw_light_model(
                &self.obj_model,
                &self.camera_bind_group,
                &self.light_bind_group,
            );
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.obj_model,
                0..self.instances.len().try_into().unwrap(),
                &self.camera_bind_group,
                &self.light_bind_group,
            );
            // NEW!
            // render_pass.draw_model_instanced_with_material(
            //     &self.obj_model,
            //     &self.debug_material,
            //     0..self.instances.len().try_into().unwrap(),
            //     &self.camera_bind_group,
            //     &self.light_bind_group,
            // );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

enum UserEvent {
    StateReady(State),
}

struct App {
    state: Option<State>,
    event_loop_proxy: EventLoopProxy<UserEvent>,
}

impl App {
    fn new(event_loop: &EventLoop<UserEvent>) -> Self {
        Self {
            state: None,
            event_loop_proxy: event_loop.create_proxy(),
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        tracing::info!("Resumed");
        let window_attrs = Window::default_attributes();
        let window = event_loop
            .create_window(window_attrs)
            .expect("Couldn't create window.");

        #[cfg(target_arch = "wasm32")]
        {
            use web_sys::Element;
            use winit::platform::web::WindowExtWebSys;

            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| {
                    let dst = doc.get_element_by_id("wasm-example")?;
                    let canvas = Element::from(window.canvas()?);
                    dst.append_child(&canvas).ok()?;
                    Some(())
                })
                .expect("Couldn't append canvas to document body.");

            // Winit prevents sizing with CSS, so we have to set
            // the size manually when on a web target.
            let _ = window.request_inner_size(PhysicalSize::new(450, 400));

            let state_future = State::new(Arc::new(window));
            let event_loop_proxy = self.event_loop_proxy.clone();
            let future = async move {
                let state = state_future.await;
                assert!(event_loop_proxy
                    .send_event(UserEvent::StateReady(state))
                    .is_ok());
            };
            wasm_bindgen_futures::spawn_local(future)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let state = pollster::block_on(State::new(Arc::new(window)));
            assert!(self
                .event_loop_proxy
                .send_event(UserEvent::StateReady(state))
                .is_ok());
        }
    }

    fn user_event(&mut self, _: &ActiveEventLoop, event: UserEvent) {
        let UserEvent::StateReady(state) = event;
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(ref mut state) = self.state else {
            return;
        };

        if window_id != state.window.id() {
            return;
        }

        if state.input(&event) {
            return;
        }

        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                tracing::info!("Exited!");
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                tracing::info!("physical_size: {physical_size:?}");
                state.surface_configured = true;
                state.resize(physical_size);
            }
            WindowEvent::RedrawRequested => {
                if !state.surface_configured {
                    return;
                }
                state.update();
                match state.render() {
                    Ok(()) => {}
                    // Reconfigure the surface if it's lost or outdated
                    Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                        state.resize(state.size);
                    }
                    // The system is out of memory, we should probably quit
                    Err(SurfaceError::OutOfMemory) => {
                        tracing::error!("OutOfMemory");
                        event_loop.exit();
                    }

                    // This happens when a frame takes too long to present
                    Err(SurfaceError::Timeout) => {
                        tracing::warn!("Surface timeout");
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(ref mut state) = self.state {
            if !state.surface_configured {
                let size = state.window.inner_size();
                if size.width > 0 && size.height > 0 {
                    state.surface_configured = true;
                    state.resize(size);
                }
            }
            // This tells winit that we want another frame
            state.window.request_redraw();
        };
    }
}

fn init_tracing_subscriber() -> Result<()> {
    #[allow(unused_mut)]
    let mut env_filter = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy()
        .add_directive("wgpu_core::device::resource=warn".parse()?);

    #[cfg(debug_assertions)]
    {
        env_filter = env_filter.add_directive("wgpu_hal::auxil::dxgi::exception=off".parse()?);
    }

    let subscriber = tracing_subscriber::registry().with(env_filter);
    #[cfg(target_arch = "wasm32")]
    {
        use tracing_wasm::{WASMLayer, WASMLayerConfig};

        console_error_panic_hook::set_once();
        let wasm_layer = WASMLayer::new(WASMLayerConfig::default());

        subscriber.with(wasm_layer).init();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use tracing_subscriber::fmt::Layer;

        let fmt_layer = Layer::default();
        subscriber.with(fmt_layer).init();
    }
    Ok(())
}

/// Runs the application.
///
/// # Errors
/// The event loop can return an error if the event loop creation fails or if the application has
/// exited with an error status. These errors are propagated to the caller.
#[allow(unused_mut)]
pub fn run() -> Result<()> {
    init_tracing_subscriber()?;

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let mut app = App::new(&event_loop);

    #[cfg(not(target_arch = "wasm32"))]
    {
        event_loop.run_app(&mut app)?;
    }
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
    Ok(())
}
