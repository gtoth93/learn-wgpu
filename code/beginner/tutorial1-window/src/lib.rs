use anyhow::Result;
use tracing::Level;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        tracing::debug!("Resumed");
        let window = event_loop
            .create_window(Window::default_attributes())
            .expect("Couldn't create window.");

        #[cfg(target_arch = "wasm32")]
        {
            use winit::{dpi::PhysicalSize, platform::web::WindowExtWebSys};

            let _ = window.request_inner_size(PhysicalSize::new(450, 400));
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| {
                    let dst = doc.get_element_by_id("wasm-example")?;
                    let canvas = web_sys::Element::from(window.canvas()?);
                    dst.append_child(&canvas).ok()?;
                    Some(())
                })
                .expect("Couldn't append canvas to document body.");
        }

        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(ref window) = self.window else {
            return;
        };
        if window_id != window.id() {
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
            } => event_loop.exit(),
            _ => {}
        }
    }
}

pub fn run() -> Result<()> {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            use tracing_subscriber::{
                fmt::writer::MakeWriterExt, layer::SubscriberExt, util::SubscriberInitExt,
            };

            console_error_panic_hook::set_once();
            let layer = tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .without_time()
                .with_writer(tracing_web::MakeWebConsoleWriter::new().with_max_level(Level::DEBUG));
            tracing_subscriber::registry().with(layer).init();
        } else {
            use tracing_subscriber::EnvFilter;

            let filter = EnvFilter::builder()
                .with_default_directive(Level::DEBUG.into())
                .from_env_lossy();
            tracing_subscriber::fmt().with_env_filter(filter).init();
        }
    }

    let event_loop = EventLoop::new()?;
    let mut app = App::default();

    event_loop.run_app(&mut app)?;
    Ok(())
}
