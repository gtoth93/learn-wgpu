# Dependencies and the window

## Boring, I know

Some of you reading this are very experienced with opening up windows in Rust and probably have your 
favorite windowing library, but this guide is designed for everybody, so it's something that we need to 
cover. Luckily, you don't need to read this if you know what you're doing. One thing that you do need to 
know is that whatever windowing solution you use needs to support the 
[raw-window-handle](https://github.com/rust-windowing/raw-window-handle) crate.

## What crates are we using?

For the beginner stuff, we're going to keep things very simple. We'll add things as we go, but I've listed 
the relevant `Cargo.toml` bits below.

```toml
[dependencies]
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
winit = { version = "0.30", features = ["android-native-activity"] }
```
* The [winit](https://docs.rs/winit) crate is what we'll be using for opening windows and handling 
various events. 
* The [anyhow](https://docs.rs/anyhow) crate is for simplifying error handling.
* The [tracing](https://docs.rs/tracing) crate is for creating log messages. 
* The [tracing-subscriber](https://docs.rs/tracing-subscriber) crate is for outputting these logs to 
stdout. It can also output the logs created by dependencies like winit and wgpu, so this crate is quite 
useful for troubleshooting.

## Using Rust's new resolver

As of version 0.10, wgpu requires Cargo's [newest feature resolver](https://doc.rust-lang.org/cargo/reference/resolver.html#feature-resolver-version-2), which is the default in the 2021 edition (any new project started with Rust version 1.56.0 or newer). However, if you are still using the 2018 edition, you must include `resolver = "2"` in either the `[package]` section of `Cargo.toml` if you are working on a single crate or the `[workspace]` section of the root `Cargo.toml` in a workspace.

## Create a new project

run ```cargo new project_name``` where project_name is the name of the project.\
(In the example below, I have used 'tutorial1_window')

## The code

Let's begin with a `run()` function that sets up logging and starts up an event loop. Paste this into your 
`lib.rs` or equivalent.

```rust
use anyhow::Result;
use tracing::Level;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use winit::event_loop::EventLoop;

pub fn run() -> Result<()> {
    // Logging setup (with a little bit of filtering)
    let env_filter = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy()
        .add_directive("wgpu_core::device::resource=warn".parse()?);
    let subscriber = tracing_subscriber::registry().with(env_filter);
    let fmt_layer = tracing_subscriber::fmt::Layer::default();
    subscriber.with(fmt_layer).init();

    // Starting up an event loop
    let event_loop = EventLoop::new()?;
    let mut app = App::default();

    event_loop.run_app(&mut app)?;
    Ok(())
}

```

Next, we'll need a `main.rs` to run the code. It's quite simple. It just imports `run()` and, well, runs 
it!

```rust
use anyhow::Result;

use tutorial1_window::run;

fn main() -> Result<()> {
    run()
}
```

(Where 'tutorial1_window' is the name of the project you created with Cargo earlier)

You might notice that this won't compile! winit 0.30 changed the way you start up your application, going 
for a trait based approach instead of the old closure based one. First, we'll need a struct:

```rust
use winit::window::Window;

#[derive(Default)]
struct App {
    window: Option<Window>,
}
```

This struct will store all of our state, which is only the window for now. The window is wrapped in an 
`Option` since we can only create it after starting up the event loop. Next, we'll need to implement the 
`ApplicationHandler` trait for our struct:

```rust
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::WindowId,
};

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
    }
}
```

All methods in this trait are callbacks for various events. Two of these we have to implement, the rest are 
optional.

The `resumed` function is called when an application resumes from suspension. It is recommended by the 
maintainers of winit that we create our window here. Some platforms cannot suspend or resume their 
applications so for compatibility reasons, `resumed` is called at least once for all platforms after the 
startup of the event loop. There's not much going on here yet, all this does is create a window and store 
it in the `App` struct.

```rust
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        tracing::info!("Resumed");
        let window_attrs = Window::default_attributes();
        let window = event_loop
            .create_window(window_attrs)
            .expect("Couldn't create window.");

        self.window = Some(window);
    }
}
```

The `window_event` function is called when the OS sends an event to the window. For now the only events 
we're handling are the user closing the window and the user pressing Escape, both of them causing the 
application to exit.

```rust
use winit::{
    event::KeyEvent,
    keyboard::{KeyCode, PhysicalKey},
};

impl ApplicationHandler for App {
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
            } => {
                tracing::info!("Exited!");
                event_loop.exit()
            }
            _ => {}
        }
    }
}
```

If you only want to support desktops, that's all you have to do! In the next tutorial, we'll start using 
wgpu!

## Added support for the web

If I go through this tutorial about WebGPU and never talk about using it on the web, then I'd hardly call 
this tutorial complete. Fortunately, getting a wgpu application running in a browser is not too difficult 
once you get things set up.

Let's start with the changes we need to make to our `Cargo.toml`:

```toml
[lib]
crate-type = ["cdylib", "rlib"]
```

These lines tell Cargo that we want to allow our crate to build a native Rust static library (rlib) and 
a C/C++ compatible library (cdylib). We need rlib if we want to run wgpu in a desktop environment. We 
need cdylib to create the Web Assembly that the browser will run.

## Web Assembly

Web Assembly, i.e. WASM, is a binary format supported by most modern browsers that allows lower-level 
languages such as Rust to run on a web page. This allows us to write the bulk of our application in Rust 
and use a few lines of Javascript to get it running in a web browser.

Now, all we need are some more dependencies that are specific to running in WASM:

```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
console_error_panic_hook = "0.1"
tracing-wasm = "0.2"
wasm-bindgen = "0.2"
web-sys = { version = "0.3", features = ["Document", "Element", "Window"] }
```

The `[target.'cfg(target_arch = "wasm32")'.dependencies]` line tells Cargo to only include these 
dependencies if we are targeting the `wasm32` architecture. The next few dependencies just make 
interfacing with JavaScript a lot easier.

* [console_error_panic_hook](https://docs.rs/console_error_panic_hook) configures the `panic!` macro to 
send errors to the javascript console. Without this, when you encounter panics, you'll be left in the 
dark about what caused them.
* [tracing-wasm](https://docs.rs/console_log) implements a tracing_subscriber `Layer` which sends all logs to the javascript 
console, which is great for debugging.
* [wasm-bindgen](https://docs.rs/wasm-bindgen) is the most important dependency in this list. It's responsible for generating the 
boilerplate code that will tell the browser how to use our crate. It also allows us to expose methods in 
Rust that can be used in JavaScript and vice-versa. I won't get into the specifics of wasm-bindgen, so if 
you need a primer (or just a refresher), check out [this](https://rustwasm.github.io/wasm-bindgen/).
* [web-sys](https://docs.rs/web-sys) is a crate with many methods and structures available in a normal javascript application: 
`get_element_by_id`, `append_child`. The features listed are only the bare minimum of what we need 
currently.

## More code

First, we need to import `wasm-bindgen` in `main.rs`:

```rust
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
```

Next, we need to tell wasm-bindgen to run our `main()` function when the WASM is loaded:

```rust
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(main))]
fn main() -> Result<()> {
    // same as above for now...
}
```

Then, we need to toggle what logging layer we are using based on whether we are in WASM land or not. Add 
the following to the top of the run function, replacing the original logging setup:

```rust
pub fn run() -> Result<()> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(Level::INFO.into())
        .from_env_lossy()
        .add_directive("wgpu_core::device::resource=warn".parse()?);
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
        let fmt_layer = tracing_subscriber::fmt::Layer::default();
        subscriber.with(fmt_layer).init();
    }
    
    // Starting up the event loop
}
```

This will set up `console_error_panic_hook` and add a `WASMLayer` to the tracing subscriber in a web 
build and will add a `fmt::Layer` to the tracing subscriber in a normal build.

Next, after we create our window in the `resumed` function, we need to add a canvas to the HTML document 
that we will host our application:

```rust
impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        tracing::info!("Resumed");
        let window_attrs = Window::default_attributes();
        let window = event_loop
            .create_window(window_attrs)
            .expect("Couldn't create window.");
    
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
            // the size manually when on the web.
            let _ = window.request_inner_size(PhysicalSize::new(450, 400));
        }
    
        self.window = Some(window);
    }
}
```

The `"wasm-example"` id is specific to my project (aka. this tutorial). You can substitute this for 
whatever id you're using in your HTML. Alternatively, you could add the canvas directly to the `<body>` 
as they do in the wgpu repo. This part is ultimately up to you.

That's all the web-specific code we need for now. The next thing we need to do is build the Web Assembly 
itself.

## Wasm Pack

Now you can build a wgpu application with just wasm-bindgen, but I ran into some issues doing that. For 
one, you need to install wasm-bindgen on your computer as well as include it as a dependency. The version 
you install as a dependency **needs** to exactly match the version you installed. Otherwise, your build 
will fail.

To get around this shortcoming and to make the lives of everyone reading this easier, I opted to add 
[wasm-pack](https://rustwasm.github.io/docs/wasm-pack/) to the mix. Wasm-pack handles installing the correct version of wasm-bindgen for you, 
and it supports building for different types of web targets as well: browser, NodeJS, and bundlers such 
as webpack.

To use wasm-pack, first, you need to [install it](https://rustwasm.github.io/wasm-pack/installer/).

Once you've done that, we can use it to build our crate. If you only have one crate in your project, you 
can just use `wasm-pack build`. If you're using a workspace, you'll have to specify what crate you want to 
build. Imagine your crate is a directory called `game`. You would then use:

```bash
wasm-pack build game
```

Once wasm-pack is done building, you'll have a `pkg` directory in the same directory as your crate. This 
has all the javascript code needed to run the WASM code. You'd then import the WASM module in javascript:

```js
const init = await import('./pkg/game.js');
init().then(() => console.log("WASM Loaded"));
```

This site uses [Vuepress](https://vuepress.vuejs.org/), so I load the WASM in a Vue component. How you handle your WASM will 
depend on what you want to do. If you want to check out how I'm doing things, take a look at [this](https://github.com/sotrh/learn-wgpu/blob/master/docs/.vuepress/components/WasmExample.vue).

<div class="note">

If you intend to use your WASM module in a plain HTML website, you'll need to tell wasm-pack to target 
the web:

```bash
wasm-pack build --target web
```

You'll then need to run the WASM code in an ES6 Module:

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Learn WGPU</title>
    <style>
        canvas {
            background-color: black;
        }
    </style>
</head>

<body id="wasm-example">
  <script type="module">
      import init from "./pkg/pong.js";
      init().then(() => {
          console.log("WASM Loaded");
      });
  </script>
</body>

</html>
```

</div>

Press the button below, and you will see the code running!

<WasmExample example="tutorial1_window"></WasmExample>

<AutoGithubLink/>
