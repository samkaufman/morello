use std::cell::RefCell;

use clap::ValueEnum;

thread_local! {
    /// Singleton for the color mode.
    static COLOR_MODE: RefCell<ColorMode> = RefCell::new(ColorMode::Auto);
}

#[derive(Copy, Clone, PartialEq, ValueEnum)]
pub enum ColorMode {
    Never,
    Auto,
    Always,
}

pub fn set_color_mode(color: ColorMode) {
    COLOR_MODE.with(|c| *c.borrow_mut() = color);
}

pub fn do_color() -> bool {
    COLOR_MODE.with(|c| {
        *c.borrow() == ColorMode::Always
            || (*c.borrow() == ColorMode::Auto && atty::is(atty::Stream::Stdout))
    })
}
