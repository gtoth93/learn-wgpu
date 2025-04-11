use anyhow::Result;
use std::sync::Arc;
use tracing::Level;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use wgpu::{
    Backends, Color, CommandEncoderDescriptor, Device, DeviceDescriptor, Features, Instance,
    InstanceDescriptor, Limits, LoadOp, MemoryHints, Operations, PowerPreference, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RequestAdapterOptions, StoreOp, Surface,
    SurfaceConfiguration, SurfaceError, TextureFormat, TextureUsages, TextureViewDescriptor, Trace,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

struct State {
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    size: PhysicalSize<u32>,
    window: Arc<Window>,
    surface_configured: bool,
}

impl State {
    async fn new(window: Arc<Window>) -> State {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
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
            trace: Trace::Off,
        };
        let (device, queue) = adapter.request_device(&device_desc).await.unwrap();

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

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            surface_configured,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    #[allow(clippy::unused_self)]
    fn input(&mut self, _: &WindowEvent) -> bool {
        false
    }

    #[allow(clippy::unused_self)]
    fn update(&mut self) {}

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
            let color_attachment = RenderPassColorAttachment {
                view: &view,
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
            let _render_pass = encoder.begin_render_pass(&render_pass_desc);
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
                assert!(
                    event_loop_proxy
                        .send_event(UserEvent::StateReady(state))
                        .is_ok()
                );
            };
            wasm_bindgen_futures::spawn_local(future)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let state = pollster::block_on(State::new(Arc::new(window)));
            assert!(
                self.event_loop_proxy
                    .send_event(UserEvent::StateReady(state))
                    .is_ok()
            );
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
                    Err(SurfaceError::OutOfMemory | SurfaceError::Other) => {
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
        }
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
