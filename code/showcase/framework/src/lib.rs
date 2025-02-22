#![warn(clippy::pedantic)]

mod buffer;
mod camera;
mod light;
mod model;
mod pipeline;
mod shader_canvas;
mod texture;

pub use buffer::*;
pub use camera::*;
pub use light::*;
pub use model::*;
pub use pipeline::*;
pub use shader_canvas::*;
pub use texture::*;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};
use std::sync::Arc;
use thiserror::Error;
use tracing::Level;
use tracing_subscriber::{
    filter::ParseError, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter,
};
use web_time::{Duration, Instant};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType, BufferUsages,
    CommandEncoder, CreateSurfaceError, Device, DeviceDescriptor, Features, Instance,
    InstanceDescriptor, Limits, MemoryHints, PowerPreference, Queue, RequestAdapterOptions,
    RequestDeviceError, ShaderStages, Surface, SurfaceConfiguration, TextureFormat, TextureUsages,
};
use winit::{
    application::ApplicationHandler,
    error::EventLoopError,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0:?}")]
    CreateSurfaceError(#[from] CreateSurfaceError),
    #[error("{0:?}")]
    EventLoopError(#[from] EventLoopError),
    #[error("No adapter found when requested")]
    NoAdapter,
    #[error("{0:?}")]
    RequestDeviceError(#[from] RequestDeviceError),
    #[error("{0:?}")]
    TracingDirectiveParseError(#[from] ParseError),
}

pub struct Display {
    surface: Surface<'static>,
    surface_configured: bool,
    pub window: Arc<Window>,
    pub config: SurfaceConfiguration,
    pub device: Device,
    pub queue: Queue,
}

impl Display {
    /// # Errors
    ///
    /// Will return `Err` if the surface cannot be created or there is no adapter or device when requested.
    pub async fn new(window: Arc<Window>) -> Result<Self, Error> {
        let size = window.inner_size();

        tracing::warn!("WGPU setup");
        let instance_desc = &InstanceDescriptor {
            backends: if cfg!(not(target_arch = "wasm32")) {
                Backends::PRIMARY
            } else {
                Backends::GL
            },
            ..Default::default()
        };
        let instance = Instance::new(instance_desc);
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or(Error::NoAdapter)?;

        tracing::warn!("device and queue");
        let device_desc = &DeviceDescriptor {
            label: None,
            required_features: Features::empty(),
            // WebGL doesn't support all of wgpu's features, so if
            // we're building for the web we'll have to disable some.
            required_limits: if cfg!(target_arch = "wasm32") {
                Limits::downlevel_webgl2_defaults()
            } else {
                Limits::default()
            },
            memory_hints: MemoryHints::default(),
        };
        let (device, queue) = adapter.request_device(device_desc, None).await?;

        tracing::warn!("Surface");
        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an Srgb surface texture. Using a different
        // one will result all the colors comming out darker. If you want to support non
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
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let surface_configured = if cfg!(not(target_arch = "wasm32")) {
            surface.configure(&device, &config);
            true
        } else {
            false
        };

        let display = Self {
            surface,
            surface_configured,
            window,
            config,
            device,
            queue,
        };
        Ok(display)
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.surface_configured = true;
        }
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }
}

/**
 * Holds the camera data to be passed to wgpu.
 */
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct UniformData {
    view_position: Vec4,
    view_proj: Mat4,
}

pub struct CameraUniform {
    data: UniformData,
    buffer: wgpu::Buffer,
}

impl CameraUniform {
    #[must_use]
    pub fn new(device: &Device) -> Self {
        let data = UniformData {
            view_position: Vec4::ZERO,
            view_proj: Mat4::IDENTITY,
        };
        let data_slice = &[data];
        let buffer_desc = &BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(data_slice),
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
        };
        let buffer = device.create_buffer_init(buffer_desc);

        Self { data, buffer }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        self.data.view_position = camera.position.extend(1.0);
        self.data.view_proj = projection.calc_matrix() * camera.calc_matrix();
    }

    pub fn update_buffer(&self, device: &Device, encoder: &mut CommandEncoder) {
        let data_slice = &[self.data];
        let buffer_desc = &BufferInitDescriptor {
            label: Some("Camera Update Buffer"),
            contents: bytemuck::cast_slice(data_slice),
            usage: BufferUsages::COPY_SRC,
        };
        let staging_buffer = device.create_buffer_init(buffer_desc);
        encoder.copy_buffer_to_buffer(
            &staging_buffer,
            0,
            &self.buffer,
            0,
            size_of::<UniformData>() as _,
        );
    }
}

/**
 * Holds the `BindGroupLayout` and one `BindGroup` for the
 * just the `CameraUniform` struct.
 */
pub struct UniformBinding {
    pub layout: BindGroupLayout,
    pub bind_group: BindGroup,
}

impl UniformBinding {
    #[must_use]
    pub fn new(device: &Device, camera_uniform: &CameraUniform) -> Self {
        let bind_group_layout_entry = BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout_desc = &BindGroupLayoutDescriptor {
            entries: &[bind_group_layout_entry],
            label: Some("CameraBinding::layout"),
        };
        let layout = device.create_bind_group_layout(bind_group_layout_desc);
        let bind_group_entry = BindGroupEntry {
            binding: 0,
            resource: camera_uniform.buffer.as_entire_binding(),
        };
        let bind_group_desc = &BindGroupDescriptor {
            layout: &layout,
            entries: &[bind_group_entry],
            label: Some("CameraBinding::bind_group"),
        };
        let bind_group = device.create_bind_group(bind_group_desc);

        Self { layout, bind_group }
    }

    pub fn rebind(&mut self, device: &Device, camera_uniform: &CameraUniform) {
        let bind_group_entry = BindGroupEntry {
            binding: 0,
            resource: camera_uniform.buffer.as_entire_binding(),
        };
        let bind_group_desc = &BindGroupDescriptor {
            layout: &self.layout,
            entries: &[bind_group_entry],
            label: Some("CameraBinding::bind_group"),
        };
        self.bind_group = device.create_bind_group(bind_group_desc);
    }
}

pub trait Demo: 'static + Sized {
    fn init(display: &Display) -> Self;
    fn process_mouse(&mut self, dx: f64, dy: f64);
    fn process_keyboard(&mut self, key: KeyCode, pressed: bool);
    fn resize(&mut self, display: &Display);
    fn update(&mut self, display: &Display, dt: Duration);
    fn render(&mut self, display: &mut Display);
}

enum UserEvent {
    DisplayReady(Display),
}

struct App<D: Demo> {
    is_resumed: bool,
    is_focused: bool,
    last_update: Instant,
    event_loop_proxy: EventLoopProxy<UserEvent>,
    display_demo: Option<(Display, D)>,
}

impl<D: Demo> App<D> {
    fn new(event_loop: &EventLoop<UserEvent>) -> Self {
        Self {
            is_resumed: true,
            is_focused: true,
            last_update: Instant::now(),
            event_loop_proxy: event_loop.create_proxy(),
            display_demo: None,
        }
    }
}

impl<D: Demo> ApplicationHandler<UserEvent> for App<D> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attr = WindowAttributes::default().with_title(env!("CARGO_PKG_NAME"));
        let window = event_loop.create_window(window_attr).unwrap();

        #[cfg(target_arch = "wasm32")]
        {
            use web_sys::Element;
            use winit::{dpi::PhysicalSize, platform::web::WindowExtWebSys};

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

            let state_future = Display::new(Arc::new(window));
            let event_loop_proxy = self.event_loop_proxy.clone();
            let future = async move {
                if let Ok(display) = state_future.await {
                    assert!(event_loop_proxy
                        .send_event(UserEvent::DisplayReady(display))
                        .is_ok());
                }
            };
            wasm_bindgen_futures::spawn_local(future)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(display) = pollster::block_on(Display::new(Arc::new(window))) {
                assert!(self
                    .event_loop_proxy
                    .send_event(UserEvent::DisplayReady(display))
                    .is_ok());
            }
        }
    }

    fn user_event(&mut self, _: &ActiveEventLoop, event: UserEvent) {
        let UserEvent::DisplayReady(display) = event;
        let demo = D::init(&display);
        self.display_demo = Some((display, demo));
        self.is_resumed = true;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some((display, demo)) = &mut self.display_demo {
            if window_id == display.window.id() {
                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::Focused(f) => self.is_focused = f,
                    WindowEvent::Resized(new_inner_size) => {
                        display.resize(new_inner_size.width, new_inner_size.height);
                        demo.resize(display);
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(key),
                                state,
                                ..
                            },
                        ..
                    } => {
                        demo.process_keyboard(key, state == ElementState::Pressed);
                    }
                    WindowEvent::RedrawRequested => {
                        if !display.surface_configured {
                            return;
                        }
                        let now = Instant::now();
                        let dt = now - self.last_update;
                        self.last_update = now;

                        demo.update(display, dt);
                        demo.render(display);
                    }
                    _ => {}
                }
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some((display, demo)) = &mut self.display_demo {
            if !display.surface_configured {
                let size = display.window().inner_size();
                if size.width > 0 && size.height > 0 {
                    display.resize(size.width, size.height);
                    demo.resize(display);
                }
            }
            if self.is_focused && self.is_resumed {
                // This tells winit that we want another frame
                display.window().request_redraw();
            } else {
                // Freeze time while the demo is not in the foreground
                self.last_update = Instant::now();
            }
        };
    }
}

fn init_tracing_subscriber() -> Result<(), Error> {
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

/// # Errors
///
/// Will return `Err` if the event loop cannot be created.
pub fn run<D: Demo>() -> Result<(), Error> {
    init_tracing_subscriber()?;

    let event_loop = EventLoop::<UserEvent>::with_user_event().build()?;
    let mut app = App::<D>::new(&event_loop);
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
