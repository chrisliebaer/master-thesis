use egui_winit_vulkano::Gui;
use miette::Result;
use winit::event_loop::EventLoop;

pub mod vulkan;

pub trait RenderBackend {
	/// Creates an instance of egui.
	fn create_egui(&self, event_loop: &EventLoop<()>) -> Gui;

	/// This method is used to notify the render backend that the window has been resized.
	fn notify_resize(&mut self);

	/// Called when the application is ready to render a new frame.
	fn render_frame(&mut self, egui: &mut Gui) -> Result<()>;

	/// Demo compute shader.
	fn demo_compute(&self) -> Result<()>;
}
