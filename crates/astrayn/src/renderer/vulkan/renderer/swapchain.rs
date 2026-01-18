use std::{
	cmp::max,
	sync::Arc,
	time::Duration,
};

use bevy_ecs::prelude::Resource;
use miette::{
	IntoDiagnostic,
	Result,
	WrapErr,
};
use vulkano::{
	device::{
		physical::PhysicalDevice,
		Device,
	},
	format::Format,
	image::{
		view::ImageView,
		Image,
		ImageUsage,
	},
	swapchain::{
		ColorSpace,
		Surface,
		Swapchain,
		SwapchainAcquireFuture,
		SwapchainCreateInfo,
	},
	Validated,
	VulkanError,
};
use winit::window::Window;

use crate::renderer::vulkan::renderer::MAX_FRAMES_IN_FLIGHT;

/// Struct `WaitForPresent` is used to manage a collection of futures that are executed in a round-robin fashion.
/// It contains the current index and an array of `InnerWaitForPresent` futures.
pub(super) struct WaitForPresent {
	current: usize,
	futures: [InnerWaitForPresent; MAX_FRAMES_IN_FLIGHT],
}

impl WaitForPresent {
	/// Creates a new `WaitForPresent` with the current index set to 0 and the futures array initialized to its default
	/// state.
	pub fn new() -> Self {
		Self {
			current: 0,
			futures: Default::default(),
		}
	}

	/// Executes the future at the current index.
	pub fn wait(&mut self) -> Result<()> {
		let fun = &mut self.futures[self.current].future;
		fun()
	}

	/// Adds a new future to the array at the current index and increments the current index.
	/// If the current index reaches the maximum number of frames in flight, it wraps around to 0.
	pub fn add(&mut self, future: Box<dyn FnMut() -> Result<()>>) {
		self.futures[self.current] = InnerWaitForPresent {
			future,
		};
		self.current = (self.current + 1) % MAX_FRAMES_IN_FLIGHT;
	}
}

impl Default for WaitForPresent {
	/// Provides a default instance of `WaitForPresent` by calling the `new` function.
	fn default() -> Self {
		Self::new()
	}
}

/// Struct `InnerWaitForPresent` wraps a future in a box.
/// The future is a function that takes no arguments and returns no value.
struct InnerWaitForPresent {
	future: Box<dyn FnMut() -> Result<()>>,
}

impl Default for InnerWaitForPresent {
	/// Provides a default instance of `InnerWaitForPresent` with an empty future.
	fn default() -> Self {
		Self {
			future: Box::new(|| Ok(())),
		}
	}
}

pub(super) struct AquireResult {
	pub suboptimal: bool,
	pub future: SwapchainAcquireFuture,
	pub image_index: u32,
	pub pair: ImagePair,
}

pub(super) struct ImagePair {
	pub image: Arc<Image>,
	pub view: Arc<ImageView>,
}

impl ImagePair {
	fn new(image: Arc<Image>) -> Self {
		Self {
			view: ImageView::new_default(image.clone())
				.into_diagnostic()
				.wrap_err("failed to create image view")
				.unwrap(),
			image,
		}
	}
}

/// Manages the swapchain and presentation. Also allows recreation of the swapchain.
#[derive(Resource)]
pub(super) struct SwapchainContext {
	/// The Vulkan swapchain.
	swapchain: Arc<Swapchain>,

	/// The swapchain images as combined image and image view pairs.
	image_pairs: Vec<ImagePair>,
}

impl SwapchainContext {
	pub fn new(device: &Arc<Device>, window: &Arc<Window>, surface: &Arc<Surface>) -> Result<Self> {
		let (swapchain, images) = Self::create_or_recreate_swapchain(device.physical_device(), device, window, surface, None)?;

		Ok(Self {
			swapchain,
			image_pairs: images.iter().map(|image| ImagePair::new(image.clone())).collect(),
		})
	}

	pub fn acquire_next_image(&self, duration: Option<Duration>) -> std::result::Result<AquireResult, Validated<VulkanError>> {
		vulkano::swapchain::acquire_next_image(self.swapchain.clone(), duration).map(|(image_index, suboptimal, future)| {
			AquireResult {
				suboptimal,
				future,
				image_index,
				pair: ImagePair::new(self.image_pairs[image_index as usize].image.clone()),
			}
		})
	}

	pub fn recreate_swapchain(&mut self, device: &Arc<Device>, window: &Arc<Window>, surface: &Arc<Surface>) -> Result<()> {
		let (swapchain, images) = Self::create_or_recreate_swapchain(
			device.physical_device(),
			device,
			window,
			surface,
			Some(self.swapchain.clone()),
		)?;

		self.swapchain = swapchain;
		self.image_pairs = images.iter().map(|image| ImagePair::new(image.clone())).collect();

		Ok(())
	}

	fn create_or_recreate_swapchain(
		physical_device: &Arc<PhysicalDevice>,
		device: &Arc<Device>,
		window: &Arc<Window>,
		surface: &Arc<Surface>,
		swapchain: Option<Arc<Swapchain>>,
	) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>)> {
		let caps = physical_device
			.surface_capabilities(surface, Default::default())
			.into_diagnostic()
			.wrap_err("failed to get surface capabilities")?;

		let swapchain_create_info = SwapchainCreateInfo {
			// we ideally want at least MAX_FRAMES_IN_FLIGHT images in the swapchain, but stay within the bounds of the provided caps
			min_image_count: max(MAX_FRAMES_IN_FLIGHT as u32, caps.min_image_count),
			image_format: Format::B8G8R8A8_SRGB,
			image_color_space: ColorSpace::SrgbNonLinear,
			image_extent: window.inner_size().into(),
			image_usage: ImageUsage::COLOR_ATTACHMENT,
			clipped: true,
			win32_monitor: None,
			..Default::default()
		};

		// if swapchain is already present, we need to recreate it, which is a slightly different call
		let (swapchain, images) = if let Some(swapchain) = swapchain {
			swapchain
				.recreate(swapchain_create_info)
				.into_diagnostic()
				.wrap_err("failed to recreate swapchain")?
		} else {
			Swapchain::new(device.clone(), surface.clone(), swapchain_create_info)
				.into_diagnostic()
				.wrap_err("failed to create swapchain")?
		};

		// old swapchain will be dropped after reassignment of swapchain

		Ok((swapchain, images))
	}

	pub fn get_swapchain(&self) -> &Arc<Swapchain> {
		&self.swapchain
	}
}
