use std::{
	ops::Deref,
	sync::Arc,
};

use bevy_ecs::{
	prelude::{
		Event as BevyEvent,
		Events as BevyEvents,
		FromWorld,
		IntoSystemSetConfigs,
		Resource,
		Schedule,
		Schedules,
		World,
	},
	schedule::{
		IntoSystemConfigs,
		ScheduleLabel,
	},
};
use miette::{
	IntoDiagnostic,
	Result,
	WrapErr,
};
use winit::{
	event::{
		Event as WinitEvent,
		WindowEvent,
	},
	event_loop::EventLoop,
	window::Window,
};

use crate::{
	debug::{
		hack_handle_window_event,
		EguiDebugPlugin,
	},
	ecs::AnyRes,
	renderer::vulkan::renderer::{
		VulkanBackendCommands,
		VulkanBackendPlugin,
	},
	scheduler::{
		MainSchedule::First,
		MainScheduler,
		MainSchedulerPlugin,
		SetupSchedule,
		SetupScheduler,
	},
	thesis::ThesisPlugin,
	APP_NAME,
};

pub struct Application {
	world: World,
}

const PLUGINS: &[&dyn Plugin] = &[
	&MainSchedulerPlugin,
	&ApplicationPlugin,
	&VulkanBackendPlugin,
	&EguiDebugPlugin,
	&ThesisPlugin,
];

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WinitInputScheduler;

// TODO: this class currently also handles window creation and os event handling, this should be moved to a separate
// TODO: class, so that this class only works headless. also remove use alias for bevy event / events
impl Application {
	pub fn new() -> Result<Self> {
		let mut world = World::new();
		world.insert_resource(Schedules::new());

		let mut app = Self {
			world,
		};

		for plugin in PLUGINS {
			app.add_plugin(*plugin);
		}

		Ok(app)
	}

	pub fn add_plugin(&mut self, plugin: &dyn Plugin) -> &mut Self {
		plugin.build(self);
		self
	}

	pub fn add_systems<M>(&mut self, schedule: impl ScheduleLabel, systems: impl IntoSystemConfigs<M>) -> &mut Self {
		let schedule = schedule.intern();
		let mut schedules = self.world.resource_mut::<Schedules>();

		match schedules.get_mut(schedule) {
			Some(schedule) => {
				schedule.add_systems(systems);
			},
			None => {
				let mut schedule = Schedule::new(schedule);
				schedule.add_systems(systems);
				schedules.insert(schedule);
			},
		}

		self
	}

	pub fn add_schedule(&mut self, schedule: Schedule) -> &mut Self {
		let mut schedules = self.world.resource_mut::<Schedules>();
		schedules.insert(schedule);

		self
	}

	pub fn configure_sets(&mut self, schedule: impl ScheduleLabel, sets: impl IntoSystemSetConfigs) -> &mut Self {
		let schedule = schedule.intern();
		let mut schedules = self.world.resource_mut::<Schedules>();
		if let Some(schedule) = schedules.get_mut(schedule) {
			schedule.configure_sets(sets);
		} else {
			let mut new_schedule = Schedule::new(schedule);
			new_schedule.configure_sets(sets);
			schedules.insert(new_schedule);
		}
		self
	}

	pub fn insert_resource<R: Resource>(&mut self, resource: R) -> &mut Self {
		self.world.insert_resource(resource);
		self
	}

	pub fn insert_non_send_resource<R: 'static>(&mut self, resource: R) -> &mut Self {
		self.world.insert_non_send_resource(resource);
		self
	}

	pub fn add_event<T>(&mut self) -> &mut Self
	where T: BevyEvent {
		if !self.world.contains_resource::<BevyEvents<T>>() {
			self.init_resource::<BevyEvents<T>>().add_systems(
				First,
				bevy_ecs::event::event_update_system::<T>.run_if(bevy_ecs::event::event_update_condition::<T>),
			);
		}
		self
	}

	pub fn init_resource<R: Resource + FromWorld>(&mut self) -> &mut Self {
		self.world.init_resource::<R>();
		self
	}

	/// This call gives control to the application loop, which will run the application until it is closed.
	/// This call will not return and the calling thread will become the event loop thread.
	/// Some plattforms may require that this call is made from the main thread.
	pub fn run(mut self) -> ! {
		// initialize app by running setup scheduler
		self.world.run_schedule(SetupScheduler);

		// removing event loop from world, as we will now use this thread to drive it
		let event_loop = self
			.world
			.remove_non_send_resource::<EventLoop<()>>()
			.expect("event loop not found");

		let window = self.world.resource::<AnyRes<Arc<Window>>>().deref().clone();

		event_loop.run(move |event, _, control_flow| {
			if let WinitEvent::WindowEvent {
				event, ..
			} = &event
			{
				// TODO: horrible hack since event can't be cloned or passed outside of the closure
				// TODO: should somehow be converted by our own input system asap
				hack_handle_window_event(&mut self.world, event);
			}

			match event {
				WinitEvent::WindowEvent {
					event: window_event, ..
				} => match window_event {
					WindowEvent::Resized(_) => self.world.send_event(VulkanBackendCommands::NotifyResize),
					WindowEvent::CloseRequested => {
						control_flow.set_exit();
					},
					_ => {},
				},
				WinitEvent::RedrawRequested(_) => {
					self.world.run_schedule(MainScheduler);
				},
				WinitEvent::RedrawEventsCleared => {
					// set redraw flag, so we are called again, in case nothing else has already requested a redraw
					window.request_redraw();
				},
				WinitEvent::LoopDestroyed => {},
				_ => {},
			}
		});
	}
}

pub trait Plugin {
	fn build(&self, app: &mut Application);
}

struct ApplicationPlugin;

impl Plugin for ApplicationPlugin {
	fn build(&self, app: &mut Application) {
		app.add_systems(SetupSchedule::Window, ApplicationPlugin::create_window);
	}
}

impl ApplicationPlugin {
	fn create_window(world: &mut World) {
		// window lives in main thread, so we run exclusive on main thread
		let event_loop = EventLoop::new();

		let window = Arc::new(
			winit::window::WindowBuilder::new()
				.with_title(&*APP_NAME)
				.build(&event_loop)
				.into_diagnostic()
				.wrap_err("failed to create window")
				.unwrap(),
		);
		let window = AnyRes::new(window);

		world.insert_non_send_resource(event_loop);
		world.insert_resource(window);
	}
}
