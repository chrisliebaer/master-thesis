use std::{
	ops::{
		Deref,
		DerefMut,
	},
	sync::Arc,
};

use bevy_ecs::prelude::{
	Commands,
	NonSendMut,
	Resource,
	World,
};
use egui_winit_vulkano::{
	Gui,
	GuiConfig,
};
use vulkano::{
	format::Format,
	swapchain::Surface,
};
use winit::event_loop::EventLoop;

use crate::{
	application::{
		Application,
		Plugin,
	},
	ecs::AnyRes,
	renderer::vulkan::renderer::GraphicsQueue,
	scheduler::{
		MainSchedule,
		SetupSchedule,
	},
};

/// Contains child view that is owned and can easily be passed around.
#[derive(Resource)]
pub struct DebugUi(egui::Context);

impl DebugUi {
	pub fn new(context: egui::Context) -> Self {
		Self(context)
	}
}

impl Deref for DebugUi {
	type Target = egui::Context;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl DerefMut for DebugUi {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}

/// Creates the egui instance and adds it to the world.
fn create_egui(world: &mut World) {
	let event_loop = world.non_send_resource::<EventLoop<()>>();
	let surface = world.resource::<AnyRes<Arc<Surface>>>();
	let graphics_queue = &world.resource::<GraphicsQueue>().0;

	let gui_config = GuiConfig {
		is_overlay: true,
		allow_srgb_render_target: true, // TODO: we need to render into unorm and blip later
		..Default::default()
	};

	let egui = Gui::new(
		event_loop,
		surface.deref().clone(),
		graphics_queue.clone(),
		Format::B8G8R8A8_SRGB, // TODO: rendering into unorm and bliping will make this a unorm
		gui_config,
	);

	world.insert_non_send_resource(egui);
}

fn prepare_ui(mut commands: Commands, mut egui: NonSendMut<Gui>) {
	egui.begin_frame();
	let context = egui.context();

	egui::Window::new("egui Settings").default_open(false).show(&context, |ui| {
		context.settings_ui(ui);
	});

	egui::Window::new("egui Textures").default_open(false).show(&context, |ui| {
		context.texture_ui(ui);
	});

	// TODO: since we are currently not clearing the framebuffer, insert a full screen panel
	egui::CentralPanel::default().show(&context, |ui| {
		ui.label("This is a full screen panel, you should probably replace it");
	});

	commands.insert_resource(DebugUi::new(context));
}

// TODO: this
fn commit_draw(mut egui: NonSendMut<Gui>) {
	todo!("for now we render directly inside the main render system, but we should eventually prepare the buffer here")
}

pub fn hack_handle_window_event(world: &mut World, event: &winit::event::WindowEvent) -> bool {
	let mut egui = world.non_send_resource_mut::<Gui>();
	egui.update(event)
}

/// Extension trait for `egui::Ui` to add some helper methods.
pub trait EguiDebugHelper {
	/// Sets up a collapsing header with the given text and content from an iterator.
	///
	/// # Arguments
	///
	/// - `id`: The unique identifier for the collapsible content.
	/// - `text`: The text to display in the header.
	/// - `it`: An iterator over the content items.
	/// - `f`: A closure that takes a mutable `egui::Ui` reference, the collapsible content `id`, and an item from the
	///   iterator. The closure will be called for each item in the iterator to create the content.
	fn collapsing_header_from_iter<I>(
		&mut self,
		id: egui::Id,
		text: &str,
		it: impl Iterator<Item = I>,
		f: impl Fn(&mut egui::Ui, egui::Id, I),
	);
}

impl EguiDebugHelper for egui::Ui {
	fn collapsing_header_from_iter<I>(
		&mut self,
		id: egui::Id,
		text: &str,
		it: impl IntoIterator<Item = I>,
		f: impl Fn(&mut egui::Ui, egui::Id, I),
	) {
		let elements = it.into_iter().collect::<Vec<_>>();
		if elements.is_empty() {
			egui::CollapsingHeader::new(format!("{} (empty)", text))
				.id_source(id)
				.enabled(false)
				.show(self, |_ui| {});
		} else {
			egui::CollapsingHeader::new(format!("{} ({})", text, elements.len()))
				.id_source(id)
				.show(self, |ui| {
					for element in elements {
						f(ui, id, element)
					}
				});
		};
	}
}

pub struct EguiDebugPlugin;

impl Plugin for EguiDebugPlugin {
	fn build(&self, app: &mut Application) {
		app
			.add_systems(SetupSchedule::Egui, create_egui)
			.add_systems(MainSchedule::First, prepare_ui);
	}
}
