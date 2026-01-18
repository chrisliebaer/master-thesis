mod swapchain;

use std::sync::Arc;

use bevy_ecs::prelude::{
	apply_deferred,
	Commands,
	Event,
	EventReader,
	IntoSystemConfigs,
	NonSend,
	NonSendMut,
	Res,
	ResMut,
	Resource,
	World,
};
use egui_winit_vulkano::Gui;
use lazy_static::lazy_static;
use miette::{
	IntoDiagnostic,
	WrapErr,
};
use tracing::debug;
use vulkano::{
	command_buffer::allocator::StandardCommandBufferAllocator,
	descriptor_set::allocator::StandardDescriptorSetAllocator,
	device::{
		Device,
		DeviceExtensions,
		Queue,
	},
	format::Format,
	memory::allocator::StandardMemoryAllocator,
	swapchain::{
		Surface,
		SwapchainPresentInfo,
	},
	sync,
	sync::GpuFuture,
	Validated,
	VulkanError,
	VulkanLibrary,
};
use winit::{
	event_loop::EventLoop,
	window::Window,
};

use crate::{
	application::{
		Application,
		Plugin,
	},
	ecs::ResWrap,
	renderer::vulkan::{
		renderer::swapchain::{
			AquireResult,
			ImagePair,
			SwapchainContext,
			WaitForPresent,
		},
		shaders::{
			create_shader_registry,
			debug_ui::debugui_shader_registry,
		},
	},
	scheduler::{
		GraphicsSchedule,
		MainSchedule,
		SetupSchedule,
	},
};

lazy_static! {
	/// The Vulkan library.
	static ref VULKAN_LIBRARY: Arc<VulkanLibrary> = VulkanLibrary::new()
		.into_diagnostic()
		.wrap_err("failed to load vulkan library")
		.unwrap();
}

const REQUIRED_DEVICE_EXTENSIONS: &DeviceExtensions = &DeviceExtensions {
	khr_swapchain: true,
	khr_dynamic_rendering: true,
	khr_storage_buffer_storage_class: true,
	..DeviceExtensions::empty()
};

#[derive(Resource, Clone)]
#[repr(transparent)]
pub struct GraphicsQueue(pub Arc<Queue>);

#[derive(Resource)]
#[repr(transparent)]
pub struct TransferQueue(pub Arc<Queue>);

#[derive(Resource, Clone)]
pub struct Allocators {
	pub memory: Arc<StandardMemoryAllocator>,
	pub descriptor_set: Arc<StandardDescriptorSetAllocator>,
	pub command_buffer: Arc<StandardCommandBufferAllocator>,
}

impl Allocators {
	pub fn new(device: &Arc<Device>) -> Self {
		Self {
			memory: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
			descriptor_set: Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default())),
			command_buffer: Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default())),
		}
	}
}

/// The maximum number of frames that can be in flight at the same time.
const MAX_FRAMES_IN_FLIGHT: usize = 3;

// most common formats, we can still work with
const SWAPCHAIN_IMAGE_FORMATS: &[Format] = &[Format::B8G8R8A8_SRGB, Format::A8B8G8R8_SRGB_PACK32, Format::R8G8B8A8_SRGB];

#[derive(Event)]
pub enum VulkanBackendCommands {
	NotifyResize,
}

pub struct VulkanBackend;

#[derive(Resource, Default)]
struct SwapchainRecreation(bool);

impl SwapchainRecreation {
	fn queue_recreate(&mut self) {
		self.0 = true;
	}
}

impl VulkanBackend {
	fn window_resized(mut reader: EventReader<VulkanBackendCommands>, mut swapchain_recreation: ResMut<SwapchainRecreation>) {
		for event in reader.read() {
			if let VulkanBackendCommands::NotifyResize = event {
				debug!("received resize event, marking swapchain for recreation");
				*swapchain_recreation = SwapchainRecreation(true);
			}
		}
	}

	fn recreate_swapchain(
		device: ResWrap<Arc<Device>>,
		window: ResWrap<Arc<Window>>,
		surface: ResWrap<Arc<Surface>>,
		mut swapchain_context: ResMut<SwapchainContext>,
		mut swapchain_recreation: ResMut<SwapchainRecreation>,
	) {
		*swapchain_recreation = {
			let SwapchainRecreation(swapchain_recreation) = *swapchain_recreation;
			if swapchain_recreation {
				debug!("recreating swapchain");
				swapchain_context.recreate_swapchain(&device, &window, &surface).unwrap();

				// recreated, clear the flag, otherwise early return will prevent write back to the resource
				SwapchainRecreation(false)
			} else {
				return;
			}
		};
	}

	fn render_frame(
		device: ResWrap<Arc<Device>>,
		queue_graphics: Res<GraphicsQueue>,
		swapchain_context: Res<SwapchainContext>,
		mut present_futures: NonSendMut<WaitForPresent>,
		mut swapchain_recreation: ResMut<SwapchainRecreation>,
		mut egui: NonSendMut<Gui>,
	) {
		let queue_graphics = &queue_graphics.0;

		// prevent too many frames in flight and wait for the oldest one to finish
		present_futures.wait().wrap_err("failed to wait for present future").unwrap();

		let AquireResult {
			suboptimal,
			future,
			image_index,
			pair: ImagePair {
				image: _image,
				view,
			},
		} = match swapchain_context.acquire_next_image(None) {
			Ok(r) => r,
			Err(Validated::Error(VulkanError::OutOfDate)) => {
				swapchain_recreation.queue_recreate();
				return;
			},
			Err(err) => panic!("failed to acquire next image: {}", err),
		};

		// at this point we have already acquired the image, so we will continue and recreate the swapchain in the next frame
		if suboptimal {
			swapchain_recreation.queue_recreate();
		}

		let egui_future = egui.draw_on_image(future, view);

		let mut present = sync::now(device.clone())
			.join(egui_future)
			.then_swapchain_present(
				queue_graphics.clone(),
				SwapchainPresentInfo::swapchain_image_index(swapchain_context.get_swapchain().clone(), image_index),
			)
			.then_signal_fence_and_flush()
			.into_diagnostic()
			.wrap_err("failed to present frame")
			.unwrap();

		present_futures.add(Box::new(move || {
			present.wait(None).into_diagnostic().map(|_| present.cleanup_finished())
		}));
	}
}

mod builder {
	use std::{
		mem::forget,
		sync::Arc,
	};

	use bevy_ecs::prelude::{
		Commands,
		NonSend,
	};
	use miette::{
		miette,
		IntoDiagnostic,
		Result,
		WrapErr,
	};
	use tracing::{
		debug,
		error,
		info,
		trace,
		warn,
	};
	use vulkano::{
		device::{
			physical::{
				PhysicalDevice,
				PhysicalDeviceType,
			},
			Device,
			DeviceCreateInfo,
			Features,
			Queue,
			QueueCreateInfo,
			QueueFlags,
		},
		instance::{
			debug::{
				DebugUtilsMessageSeverity,
				DebugUtilsMessageType,
				DebugUtilsMessenger,
				DebugUtilsMessengerCallback,
				DebugUtilsMessengerCreateInfo,
			},
			Instance,
			InstanceCreateInfo,
			InstanceExtensions,
		},
		memory::MemoryPropertyFlags,
		swapchain::{
			ColorSpace,
			Surface,
		},
	};
	use winit::{
		event_loop::EventLoop,
		window::Window,
	};

	use crate::{
		ecs::{
			AnyRes,
			ResWrap,
		},
		renderer::vulkan::renderer::{
			Allocators,
			GraphicsQueue,
			TransferQueue,
			REQUIRED_DEVICE_EXTENSIONS,
			SWAPCHAIN_IMAGE_FORMATS,
			VULKAN_LIBRARY,
		},
		APP_NAME,
		APP_VERSION,
	};

	pub fn create_context(mut commands: Commands, window: ResWrap<Arc<Window>>, event_loop: NonSend<EventLoop<()>>) -> Result<()> {
		let instance = build_instance(&event_loop)?;
		register_debug_callback(&instance)?;

		let surface = Surface::from_window(instance.clone(), window.clone())
			.into_diagnostic()
			.wrap_err("failed to create surface")?;

		let physical_device = select_device(&instance, &surface)?;
		let (device, queue_graphics, queue_transfer) = create_device_and_queues(&physical_device)?;

		commands.insert_resource(Allocators::new(&device));
		commands.insert_resource(AnyRes::new(instance));
		commands.insert_resource(AnyRes::new(surface));
		commands.insert_resource(AnyRes::new(physical_device));
		commands.insert_resource(AnyRes::new(device));
		commands.insert_resource(GraphicsQueue(queue_graphics));

		if let Some(queue_transfer) = queue_transfer {
			commands.insert_resource(TransferQueue(queue_transfer));
		}

		Ok(())
	}

	fn build_instance(event_loop: &EventLoop<()>) -> Result<Arc<Instance>> {
		let library = VULKAN_LIBRARY.clone();
		let version = &*APP_VERSION;

		// vulkano uses a different semver version
		let vulkano_version = vulkano::Version {
			major: version.major as u32,
			minor: version.minor as u32,
			patch: version.patch as u32,
		};

		let enabled_instance_extensions = {
			let mut extensions = InstanceExtensions::empty();
			extensions = extensions.union(&Surface::required_extensions(event_loop));
			extensions.ext_debug_utils = true;

			extensions
		};

		let create_info = InstanceCreateInfo {
			application_name: Some(APP_NAME.to_owned()),
			application_version: vulkano_version,
			engine_name: Some(APP_NAME.to_owned()),
			engine_version: vulkano_version,
			enabled_extensions: enabled_instance_extensions,
			..Default::default()
		};

		Instance::new(library, create_info)
			.into_diagnostic()
			.wrap_err("failed to create instance")
	}

	fn select_device(instance: &Arc<Instance>, surface: &Surface) -> Result<Arc<PhysicalDevice>> {
		// filter usable devices and score them
		let physical_device = {
			instance
				.enumerate_physical_devices()
				.into_diagnostic()
				.wrap_err("failed to enumerate physical devices")?
				.map(|device| {
					let surface_formats = device
						.surface_formats(surface, Default::default())
						.into_diagnostic()
						.wrap_err("failed to get surface formats")?;
					Ok((device, surface_formats))
				})
				// if we encounter an error, we want to terminate the chain and return the error
				.collect::<Result<Vec<_>>>()?
				.into_iter()
				.filter(|(device, surface_formats)| {
					// check if device has graphics queue
					let has_graphics_queue = device
						.queue_family_properties()
						.iter()
						.any(|family| family.queue_flags.contains(QueueFlags::GRAPHICS));

					let has_required_extensions = device.supported_extensions().contains(REQUIRED_DEVICE_EXTENSIONS);

					let has_required_formats = surface_formats
						.iter()
						.any(|(format, color_space)| SWAPCHAIN_IMAGE_FORMATS.contains(format) && color_space == &ColorSpace::SrgbNonLinear);

					has_graphics_queue && has_required_extensions && has_required_formats
				})
				.map(|(device, _)| device)
				.fold(None, |acc, device| {
					// rate device and keep the best one
					let score = rate_physical_device(&device);
					if let Some((_, best_score)) = acc {
						if score > best_score { Some((device, score)) } else { acc }
					} else {
						Some((device, score))
					}
				})
				.map(|(device, _)| device)
				.ok_or(miette!("no suitable physical device found"))?
		};

		// log device details for debugging
		let properties = physical_device.properties();
		debug!(
			driver_version = properties.driver_version,
			queue_count = physical_device.queue_family_properties().len(),
			"Found device: {}",
			properties.device_name
		);
		for (i, family) in physical_device.queue_family_properties().iter().enumerate() {
			trace!("\t - queue {} supports {:?} operations", i, family.queue_flags);
		}

		Ok(physical_device)
	}

	// come on, that's a very simple type
	#[allow(clippy::type_complexity)]
	fn create_device_and_queues(physical_device: &Arc<PhysicalDevice>) -> Result<(Arc<Device>, Arc<Queue>, Option<Arc<Queue>>)> {
		let (idx_graphics, idx_transfer) = {
			let mut idx_graphics = None;
			let mut idx_transfer = None;

			for (i, family) in physical_device.queue_family_properties().iter().enumerate() {
				// a transfer queue is a queue without graphics support, a graphics queue will always support transfer
				if family.queue_flags.contains(QueueFlags::GRAPHICS) {
					idx_graphics = Some(i);
					trace!("found graphics queue at index {}", i)
				} else if family.queue_flags.contains(QueueFlags::TRANSFER) {
					idx_transfer = Some(i);
				}
			}

			if idx_transfer.is_some() {
				trace!("found transfer queue at index {}", idx_transfer.unwrap());
			} else {
				trace!("no transfer queue found, using graphics queue for transfer");
			}

			let idx_graphics = idx_graphics.ok_or(miette!("no graphics queue found"))?;
			(idx_graphics, idx_transfer)
		};

		// create device and queues (transfer queue is optional and will only be requested if available)
		let queue_create_infos = vec![
			Some(QueueCreateInfo {
				queue_family_index: idx_graphics as u32,
				..Default::default()
			}),
			idx_transfer.map(|transfer_index| QueueCreateInfo {
				queue_family_index: transfer_index as u32,
				..Default::default()
			}),
		]
		.into_iter()
		.flatten()
		.collect();

		let (device, mut queues) = Device::new(physical_device.clone(), DeviceCreateInfo {
			queue_create_infos,
			enabled_extensions: REQUIRED_DEVICE_EXTENSIONS.to_owned(),
			enabled_features: Features {
				dynamic_rendering: true,
				shader_int64: true,
				shader_int16: true,
				shader_int8: true,
				shader_float64: true,
				vulkan_memory_model: true,
				variable_pointers: true,
				variable_pointers_storage_buffer: true,
				..Default::default()
			},
			..Default::default()
		})
		.into_diagnostic()
		.wrap_err("failed to create device and queues")?;

		Ok((device, queues.next().unwrap(), queues.next()))
	}

	fn rate_physical_device(physical_device: &PhysicalDevice) -> u64 {
		let mut score = 0;

		// device type is always dominating factor
		score += match physical_device.properties().device_type {
			PhysicalDeviceType::DiscreteGpu => 100,
			PhysicalDeviceType::IntegratedGpu => 50,
			PhysicalDeviceType::VirtualGpu => 25,
			PhysicalDeviceType::Cpu => 10,
			_ => 0,
		};

		score += {
			let props = physical_device.memory_properties();
			let heaps = &props.memory_heaps;
			let types = &props.memory_types;

			// we only consider device local memory
			let total = types.iter().fold(0u64, |acc, ty| {
				if ty.property_flags.contains(MemoryPropertyFlags::DEVICE_LOCAL) {
					acc + heaps[ty.heap_index as usize].size
				} else {
					acc
				}
			});
			// device local memory in GB is used as tiebreaker
			total / 1024 / 1024 / 1024
		};

		score
	}

	fn register_debug_callback(instance: &Arc<Instance>) -> Result<()> {
		let callback = unsafe {
			DebugUtilsMessenger::new(instance.clone(), DebugUtilsMessengerCreateInfo {
				message_severity: DebugUtilsMessageSeverity::ERROR
					| DebugUtilsMessageSeverity::WARNING
					| DebugUtilsMessageSeverity::INFO
					| DebugUtilsMessageSeverity::VERBOSE,
				message_type: DebugUtilsMessageType::GENERAL | DebugUtilsMessageType::VALIDATION | DebugUtilsMessageType::PERFORMANCE,
				..DebugUtilsMessengerCreateInfo::user_callback(DebugUtilsMessengerCallback::new(|severity, message_type, data| {
					// callback handler will ignore panics, so we make sure to implement our own panic handler
					let ty = match message_type {
						DebugUtilsMessageType::GENERAL => "general",
						DebugUtilsMessageType::VALIDATION => "validation",
						DebugUtilsMessageType::PERFORMANCE => "performance",
						_ => "unknown",
					};

					let id_hex = format!("{:x}", data.message_id_number);
					let queue_labels = data.queue_labels.map(|label| label.label_name).collect::<Vec<_>>();
					let command_buffer_labels = data.cmd_buf_labels.map(|label| label.label_name).collect::<Vec<_>>();
					let objects = data
						.objects
						.map(|object| {
							// handle is u64, so we need to convert it to hex
							format!(
								"{:?}({:?})@0x{:x}",
								object.object_type, object.object_name, object.object_handle
							)
						})
						.collect::<Vec<_>>();

					// we deliberately make explicit calls to the logging macros here, to avoid another layer of match statements
					match severity {
						DebugUtilsMessageSeverity::ERROR => {
							error!(
								ty,
								name = data.message_id_name,
								id = id_hex,
								queue_labels = ?queue_labels,
								cmd_buf_labels = ?command_buffer_labels,
								objects = ?objects,
								message = data.message
							);
						},
						DebugUtilsMessageSeverity::WARNING => {
							warn!(
								ty,
								name = data.message_id_name,
								id = id_hex,
								queue_labels = ?queue_labels,
								cmd_buf_labels = ?command_buffer_labels,
								objects = ?objects,
								message = data.message
							);
						},
						DebugUtilsMessageSeverity::INFO => {
							info!(
								ty,
								name = data.message_id_name,
								id = id_hex,
								queue_labels = ?queue_labels,
								cmd_buf_labels = ?command_buffer_labels,
								objects = ?objects,
								message = data.message
							);
						},
						DebugUtilsMessageSeverity::VERBOSE => {
							debug!(
								ty,
								name = data.message_id_name,
								id = id_hex,
								queue_labels = ?queue_labels,
								cmd_buf_labels = ?command_buffer_labels,
								objects = ?objects,
								message = data.message
							);
						},
						_ => panic!("unknown debug severity"),
					};
				}))
			})
			.into_diagnostic()
			.wrap_err("failed to register debug callback")?
		};
		forget(callback);
		Ok(())
	}
}

pub struct VulkanBackendPlugin;

impl VulkanBackendPlugin {
	fn create_conext(commands: Commands, window: ResWrap<Arc<Window>>, event_loop: NonSend<EventLoop<()>>) {
		builder::create_context(commands, window, event_loop).unwrap();
	}

	fn create_swapchain(
		mut commands: Commands,
		device: ResWrap<Arc<Device>>,
		window: ResWrap<Arc<Window>>,
		surface: ResWrap<Arc<Surface>>,
	) {
		let swapchain_context = SwapchainContext::new(&device, &window, &surface).unwrap();
		commands.insert_resource(swapchain_context);
	}

	fn create_wait_for_present(world: &mut World) {
		world.insert_non_send_resource(WaitForPresent::default());
	}
}

impl Plugin for VulkanBackendPlugin {
	fn build(&self, app: &mut Application) {
		app
			.add_systems(
				SetupSchedule::GraphicsBackend,
				(
					(VulkanBackendPlugin::create_conext, apply_deferred).chain(),
					(
						create_shader_registry,
						VulkanBackendPlugin::create_swapchain,
						VulkanBackendPlugin::create_wait_for_present,
					),
				)
					.chain(),
			)
			.add_systems(
				GraphicsSchedule::PrepareFrame,
				(VulkanBackend::window_resized, VulkanBackend::recreate_swapchain).chain(),
			)
			.add_systems(MainSchedule::Fixed, debugui_shader_registry)
			.add_systems(GraphicsSchedule::CommitFrame, VulkanBackend::render_frame)
			.add_event::<VulkanBackendCommands>()
			.insert_resource(SwapchainRecreation::default());
	}
}
