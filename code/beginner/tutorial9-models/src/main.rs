#![warn(clippy::pedantic)]

use anyhow::Result;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen(main))]
fn main() -> Result<()> {
    tutorial9_models::run()
}
