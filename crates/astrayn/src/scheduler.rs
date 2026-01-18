use std::{
	fmt::Debug,
	hash::Hash,
};

use bevy_ecs::{
	prelude::{
		Schedule,
		World,
	},
	schedule::{
		ExecutorKind,
		ScheduleLabel,
	},
};
use enum_iterator::{
	all,
	Sequence,
};

use crate::application::{
	Application,
	Plugin,
};
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SetupScheduler;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MainScheduler;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FixedScheduler;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GraphicsScheduler;

#[derive(ScheduleLabel, Sequence, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SetupSchedule {
	/// Setting up the window.
	Window,

	/// Setting up the graphics backend.
	GraphicsBackend,

	/// Creating egui.
	Egui,

	/// For setups that require the graphics backend to be fully initialized.
	AfterGraphicsBackend,

	/// Setting up the input system.
	EventLoop,
}

#[derive(ScheduleLabel, Sequence, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MainSchedule {
	/// Mainly used for internal logic of the ECS scheduler.
	First,

	/// Processing raw input events.
	RawInput,

	/// Processing semantic input events, such as concrete actions.
	SemanticInput,

	/// Reading and integrating new network data into local state.
	NetworkReceive,

	/// General stage for things that need to be done every tick.
	Main,

	/// Branches into the fixed update schedule, if enough time has accumulated.
	Fixed,

	/// Branches into the graphics schedule, if a redraw is requested.
	Graphics,

	/// Collecting state that needs to be sent to the network.
	NetworkAcc,

	/// Potentially compressing state and handling
	NetworkFlush,

	/// Mainly used for internal logic of the ECS scheduler.
	Last,
}

#[derive(ScheduleLabel, Sequence, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Fixed {
	// TODO: a lot of stuff missing here
	Physics,
}

#[derive(ScheduleLabel, Sequence, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GraphicsSchedule {
	/// Exclusive for egui management. Swaps out old child ui with new one.
	Egui,

	/// For uploading using the async transfer queue
	TransferAsync,

	/// For uploading using the graphics queue, only for data that is needed immediately, as it will block the graphics
	/// queue.
	Transfer,

	/// Collecting various command buffers.
	PrepareFrame,

	/// Submitting command buffers to the graphics queue and presenting the frame.
	CommitFrame,

	/// Cleanup after the frame has been passed to the GPU.
	PostFrame,
}

fn run_schedule<S>(world: &mut World)
where S: ScheduleLabel + Sequence {
	for labels in all::<S>() {
		let _ = world.try_run_schedule(labels);
	}
}

impl SetupScheduler {
	pub fn run_setup(world: &mut World) {
		run_schedule::<SetupSchedule>(world);
	}
}

impl MainScheduler {
	pub fn run_main(world: &mut World) {
		run_schedule::<MainSchedule>(world);
	}
}

impl FixedScheduler {
	pub fn run_fixed(world: &mut World) {
		run_schedule::<Fixed>(world);
	}
}

impl GraphicsScheduler {
	pub fn run_graphics(world: &mut World) {
		run_schedule::<GraphicsSchedule>(world);
	}
}

/// The main scheduler plugin handles the entire application lifecycle.
///
/// The core runs a total 4 schedules:
/// Setup: This is run only once on startup and allows setup of basic resources.
///
/// The Core loop (driven by the OS event loop when running in a windowed environment, or a precise timer when running
/// in headless mode):
/// 1. Main Loop: This is the main loop of the application. It is run on every run of the event loop. It drives both the
///    Fixed and Graphics schedules. System can also be added to this schedule if they need to react to every event loop
///    run.
/// 2. Fixed: This is run at a fixed rate, and is used for driving the game state. It will only run if enough time has
///    been accumulated.
/// 3. Graphics: This is run whenever a redraw is requested. It is where the majority of the rendering work is done.
pub struct MainSchedulerPlugin;

impl Plugin for MainSchedulerPlugin {
	fn build(&self, app: &mut Application) {
		let setup_schedule = Schedule::new(SetupScheduler);
		app
			.add_schedule(setup_schedule)
			.add_systems(SetupScheduler, SetupScheduler::run_setup);

		// schedule wrappers are only driving sub schedules and only have one system that runs in sequence

		let mut main_schedule = Schedule::new(MainScheduler);
		main_schedule.set_executor_kind(ExecutorKind::SingleThreaded);
		app
			.add_schedule(main_schedule)
			.add_systems(MainScheduler, MainScheduler::run_main);

		let mut fixed_schedule = Schedule::new(FixedScheduler);
		fixed_schedule.set_executor_kind(ExecutorKind::SingleThreaded);
		app
			.add_schedule(fixed_schedule)
			// TODO: this is where we insert the fixed time run condition
			.add_systems(MainSchedule::Fixed, FixedScheduler::run_fixed);

		let mut graphics_schedule = Schedule::new(GraphicsScheduler);
		graphics_schedule.set_executor_kind(ExecutorKind::SingleThreaded);
		app
			.add_schedule(graphics_schedule)
			.add_systems(MainSchedule::Graphics, GraphicsScheduler::run_graphics);
	}
}
