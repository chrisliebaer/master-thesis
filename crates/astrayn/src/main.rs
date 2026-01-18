extern crate alloc;
extern crate core;

/// The application module deals with the wrapping state around the game and sits between the operating system and
/// internal systems.
mod application;
mod debug;
mod ecs;
mod game;
mod renderer;
mod scheduler;
mod thesis;

use lazy_static::lazy_static;
use miette::{
	IntoDiagnostic,
	Result,
	WrapErr,
};
use tracing::info;

lazy_static! {
	pub static ref APP_VERSION: semver::Version = semver::Version::parse(env!("CARGO_PKG_VERSION"))
		.into_diagnostic()
		.wrap_err("failed to parse version")
		.unwrap();
	pub static ref APP_NAME: String = env!("CARGO_PKG_NAME").into();
}

fn main() -> Result<()> {
	tracing_subscriber::fmt::init();
	info!(version = %*APP_VERSION, "Starting {}...", *APP_NAME);

	let app = application::Application::new()?;

	app.run()
}
