use std::sync::Arc;

use bytemuck::Pod;
use glam::{
	UVec2,
	Vec2,
};
use kahan::KahanSum;
use pcg32::Pcg32;
use shader::thesis::{
	entrypoints::{
		debug_image_pp::ProcessingOptions,
		debug_shader::DebugPushConstants,
		sample_shader::{
			PerThreadState,
			SamplerPushConstants,
		},
	},
	math,
	mixture::{
		Mixture,
		MortonSliceSelection,
		SliceSelectionMethod,
	},
};
use tracing::info;
use vulkano::{
	buffer::{
		Buffer,
		BufferContents,
		BufferCreateInfo,
		BufferUsage,
		Subbuffer,
	},
	command_buffer::{
		AutoCommandBufferBuilder,
		CommandBufferUsage,
	},
	descriptor_set::{
		PersistentDescriptorSet,
		WriteDescriptorSet,
	},
	device::Device,
	image::view::ImageView,
	memory::allocator::{
		AllocationCreateInfo,
		MemoryAllocator,
		MemoryTypeFilter,
	},
	pipeline::{
		compute::ComputePipelineCreateInfo,
		layout::PipelineDescriptorSetLayoutCreateInfo,
		ComputePipeline,
		Pipeline,
		PipelineBindPoint,
		PipelineLayout,
		PipelineShaderStageCreateInfo,
	},
	shader::EntryPoint,
	sync,
	sync::GpuFuture,
};

use crate::{
	renderer::vulkan::{
		renderer::{
			Allocators,
			GraphicsQueue,
		},
		shaders::ShaderRegistry,
	},
	thesis::{
		env_map::EnvironmentMap,
		SamplingConfigUiState,
	},
};

#[derive(Debug)]
struct InvocationTile {
	offset: [u32; 3],
	size: [u32; 3],
}

#[derive(Debug)]
#[repr(transparent)]
struct InvocationKernelSize([u32; 3]);

impl InvocationKernelSize {
	pub fn adjust(&self, target_size: [u32; 3]) -> [u32; 3] {
		// ensure that the group size is a multiple of the kernel size
		if target_size[0] % self.0[0] != 0 || target_size[1] % self.0[1] != 0 || target_size[2] % self.0[2] != 0 {
			panic!("image dimensions are not a multiple of the kernel size");
		}

		let group_size = [
			target_size[0] / self.0[0],
			target_size[1] / self.0[1],
			target_size[2] / self.0[2],
		];

		// ensure that the group size is at least 1
		if group_size[0] == 0 || group_size[1] == 0 || group_size[2] == 0 {
			panic!("image dimensions are too small for the kernel size");
		}

		group_size
	}

	/// Partitions the input into a tiled grid of invocations.
	pub fn tile(&self, whole: [u32; 3]) -> Vec<InvocationTile> {
		let group_size = self.adjust(whole);
		let mut tiles = Vec::new();
		for x in 0..group_size[0] {
			for y in 0..group_size[1] {
				for z in 0..group_size[2] {
					let offset = [x * self.0[0], y * self.0[1], z * self.0[2]];
					tiles.push(InvocationTile {
						offset,
						size: self.0,
					});
				}
			}
		}
		tiles
	}
}

/// Helper function for creating a buffer and writing data to it.
///
/// This is a shortcut for creating a buffer, binding it, and writing data to it.
fn create_buffer_and_write<T: BufferContents + Sized + Pod>(allocator: Arc<dyn MemoryAllocator>, data: &T) -> Subbuffer<T> {
	let buffer = create_storage_buffer::<T>(allocator);
	let mut write = buffer.write().unwrap();
	*write = *data;
	drop(write);
	buffer
}

/// Helper function for creating device-local storage buffers used by most compute shaders to pass memory mapped
/// structs.
fn create_storage_buffer<T: BufferContents + Sized>(allocator: Arc<dyn MemoryAllocator>) -> Subbuffer<T> {
	Buffer::new_sized::<T>(
		allocator.clone(),
		BufferCreateInfo {
			usage: BufferUsage::STORAGE_BUFFER,
			..Default::default()
		},
		AllocationCreateInfo {
			memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
			..Default::default()
		},
	)
	.expect("failed to create buffer")
}

fn create_storage_from_iter<T: BufferContents + Sized>(
	allocator: Arc<dyn MemoryAllocator>,
	data: impl Iterator<Item = T> + ExactSizeIterator,
) -> Subbuffer<[T]> {
	Buffer::from_iter(
		allocator,
		BufferCreateInfo {
			usage: BufferUsage::STORAGE_BUFFER,
			..Default::default()
		},
		AllocationCreateInfo {
			memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
			..Default::default()
		},
		data,
	)
	.expect("failed to create buffer")
}

fn create_environment_map_buffer(allocator: Arc<dyn MemoryAllocator>, environment_map: &[f64]) -> Subbuffer<[f32]> {
	// TODO: remove once env map is fixed to use f32
	// build f32 buffer from f64 data on heap
	let dataf32 = environment_map.iter().map(|&x| x as f32).collect::<Vec<f32>>();

	create_storage_from_iter(allocator, dataf32.into_iter())
}

/// Creates a new compute pipeline from a given entry point.
///
/// # Arguments
///
/// * `entry_point` - An EntryPoint object representing the shader entry point.
/// * `device` - A reference to the Arc<Device> object representing the Vulkan device.
///
/// # Returns
///
/// * An Arc<ComputePipeline> object representing the created compute pipeline.
fn pipeline_from_entry_point(entry_point: EntryPoint, device: &Arc<Device>) -> Arc<ComputePipeline> {
	let stage = PipelineShaderStageCreateInfo::new(entry_point);
	let pipeline_layout = PipelineLayout::new(
		device.clone(),
		PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
			.into_pipeline_layout_create_info(device.clone())
			.expect("failed to create pipeline layout create info"),
	)
	.expect("failed to create pipeline layout");

	ComputePipeline::new(
		device.clone(),
		None,
		ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout),
	)
	.expect("failed to create compute pipeline")
}

fn run_compute_pipeline<C: BufferContents>(
	pipeline: &Arc<ComputePipeline>,
	descriptor_set: &Arc<PersistentDescriptorSet>,
	context: &Context,
	group_counts: [u32; 3],
	push_constants: Option<C>,
) {
	let mut builder = AutoCommandBufferBuilder::primary(
		&context.allocators.command_buffer,
		context.queue.0.queue_family_index(),
		CommandBufferUsage::OneTimeSubmit,
	)
	.expect("failed to create command buffer builder");

	builder
		.bind_pipeline_compute(pipeline.clone())
		.expect("failed to bind pipeline")
		.bind_descriptor_sets(
			PipelineBindPoint::Compute,
			pipeline.layout().clone(),
			0,
			descriptor_set.clone(),
		)
		.expect("failed to bind descriptor set");

	if let Some(push_constants) = push_constants {
		builder
			.push_constants(pipeline.layout().clone(), 0, push_constants)
			.expect("failed to push constants");
	}

	builder.dispatch(group_counts).expect("failed to dispatch");

	let command_buffer = builder.build().expect("failed to build command buffer");
	let future = sync::now(context.device.clone())
		.then_execute(context.queue.0.clone(), command_buffer)
		.expect("failed to execute command buffer")
		.then_signal_fence_and_flush()
		.expect("failed to signal fence and flush");

	// since this is a one-shot action, we wait for completion
	future.wait(None).expect("failed to wait for future");
}

#[derive(Clone)]
pub struct Context {
	pub device: Arc<Device>,
	pub queue: GraphicsQueue,
	pub allocators: Allocators,
}

impl Context {
	pub fn new(device: Arc<Device>, queue: GraphicsQueue, allocators: Allocators) -> Self {
		Self {
			device,
			queue,
			allocators,
		}
	}
}

pub struct MixtureDebugRenderer {
	pipeline: Arc<ComputePipeline>,
	context: Context,
}

impl MixtureDebugRenderer {
	const INVOCATION_GROUP_SIZE: InvocationKernelSize = InvocationKernelSize([4, 4, 1]);

	pub fn new(shader_registry: &ShaderRegistry, context: Context, entry_point: &str) -> Self {
		let pipeline = {
			let entry_point = shader_registry
				.entry_point_by_name(entry_point)
				.expect("failed to find entry point");
			pipeline_from_entry_point(entry_point, &context.device)
		};

		Self {
			pipeline,
			context,
		}
	}

	#[allow(clippy::too_many_arguments)]
	pub fn draw(
		&self,
		image_view: &Arc<ImageView>,
		environment_map: &EnvironmentMap,
		mixture: &Mixture,
		grid: &[u32],
		mouse_pos: Option<Vec2>,
		slize_size: u32,
		quantile: f32,
		morton_debug: u32,
		slice_expansion: u32,
		expansion_threshold: f32,
	) {
		let buffer_mixture = create_buffer_and_write(self.context.allocators.memory.clone(), mixture);
		let buffer_grid = create_storage_from_iter(self.context.allocators.memory.clone(), grid.iter().copied());
		let buffer_enviroment_map =
			create_environment_map_buffer(self.context.allocators.memory.clone(), environment_map.data.as_ref());

		let layout = &self.pipeline.layout().set_layouts()[0];
		let descriptor_set = PersistentDescriptorSet::new(
			&self.context.allocators.descriptor_set,
			layout.clone(),
			[
				WriteDescriptorSet::image_view(0, image_view.clone()),
				WriteDescriptorSet::buffer(1, buffer_mixture.clone()),
				WriteDescriptorSet::buffer(2, buffer_enviroment_map.clone()),
				WriteDescriptorSet::buffer(3, buffer_grid.clone()),
			],
			[],
		)
		.expect("failed to create descriptor set");

		// dispatching the entire compute shader at once will cause a timeout on the driver
		// instead we perform a checkboard pattern to avoid this and split the work into multiple submits
		let tile_size = InvocationKernelSize([128, 128, 1]);
		for tile in tile_size.tile(image_view.image().extent()) {
			let push_constants = DebugPushConstants {
				offset: UVec2::new(tile.offset[0], tile.offset[1]),
				cursor: if let Some(mouse_pos) = mouse_pos {
					mouse_pos
				} else {
					Vec2::splat(f32::NAN)
				},
				slize_size,
				ground_truth: environment_map.ground_truth as f32,
				env_map_resolution: environment_map.dim,
				quantile,
				morton_debug,
				slice_expansion,
				expansion_threshold,
			};

			// adjust the group size to kernel size
			let groups = Self::INVOCATION_GROUP_SIZE.adjust(tile.size);
			run_compute_pipeline(&self.pipeline, &descriptor_set, &self.context, groups, Some(push_constants));
		}
	}
}

pub struct PostProcessShader {
	pipeline: Arc<ComputePipeline>,
	context: Context,
}

impl PostProcessShader {
	const INVOCATION_GROUP_SIZE: InvocationKernelSize = InvocationKernelSize([4, 4, 1]);

	pub fn new(shader_registry: &ShaderRegistry, context: Context) -> Self {
		let pipeline = {
			let entry_point = shader_registry
				.entry_point_by_name("thesis::entrypoints::debug_image_pp::post_process")
				.expect("failed to find entry point");
			pipeline_from_entry_point(entry_point, &context.device)
		};

		Self {
			pipeline,
			context,
		}
	}

	pub fn draw(
		&self,
		input: &Arc<ImageView>,
		output: &Arc<ImageView>,
		opts: &ProcessingOptions,
		environment_map: &EnvironmentMap,
	) {
		let layout = &self.pipeline.layout().set_layouts()[0];
		let descriptor_set = PersistentDescriptorSet::new(
			&self.context.allocators.descriptor_set,
			layout.clone(),
			[
				WriteDescriptorSet::image_view(0, input.clone()),
				WriteDescriptorSet::image_view(1, output.clone()),
				WriteDescriptorSet::buffer(
					2,
					create_environment_map_buffer(self.context.allocators.memory.clone(), environment_map.data.as_ref()),
				),
			],
			[],
		)
		.expect("failed to create descriptor set");

		// duplicate opts to inject dimensions
		let opts = ProcessingOptions {
			env_map_resolution: environment_map.dim,
			..*opts
		};

		let groups = Self::INVOCATION_GROUP_SIZE.adjust(output.image().extent());
		run_compute_pipeline(&self.pipeline, &descriptor_set, &self.context, groups, Some(opts));
	}
}

#[derive(Debug)]
pub struct MixtureSamplerResult {
	pub estimate: f64,
	pub variance_sum: f64,
	pub slice_miss_rate: f64,
	pub sample_count: u64,
}

pub struct MixtureSamplerShader {
	context: Context,

	current_pipeline: usize,
	pipelines: Box<[(&'static str, Arc<ComputePipeline>)]>,
}

impl MixtureSamplerShader {
	const COMMAND_BUFFER_SUBMITS: u32 = 4;
	/// The kernel size of the shader.
	const INVOCATION_GROUP_SIZE: InvocationKernelSize = InvocationKernelSize([64, 1, 1]);
	/// Number of times the shader is invoked.
	const ITERATIONS: u32 = 1;
	/// Number of samples that each thread will pull.
	const SAMPLES_PER_THREAD: u32 = 1;
	const SAMPLING_ENTRYPOINTS: [&'static str; 6] = [
		"mixture_all",
		"mixture_morton",
		"mixture_knn",
		"mixture_tree_random_full",
		"mixture_tree_towards_pdf_full",
		"mixture_tree_weighted_random_full",
	];
	/// Number of threads per iteration.
	///
	/// Not the number of invocation groups.
	const THREADS_PER_ITERATION: usize = 1024 * 1024;

	pub fn new(shader_registry: &ShaderRegistry, context: Context) -> Self {
		// create a pipeline for each entry point
		let pipelines = Self::SAMPLING_ENTRYPOINTS
			.into_iter()
			.map(|name| {
				let full_name = format!("thesis::entrypoints::sample_shader::{}", name);
				let entry_point = shader_registry
					.entry_point_by_name(&full_name)
					.expect("failed to find entry point");
				(name, pipeline_from_entry_point(entry_point, &context.device))
			})
			.collect::<Vec<_>>();

		Self {
			context,
			pipelines: pipelines.into_boxed_slice(),
			current_pipeline: 0,
		}
	}

	pub fn draw_ui(&mut self, ui: &mut egui::Ui) {
		let mut current_pipeline = self.current_pipeline;
		egui::ComboBox::from_label("Sampling method")
			.selected_text(Self::SAMPLING_ENTRYPOINTS[current_pipeline])
			.show_ui(ui, |ui| {
				for (i, name) in Self::SAMPLING_ENTRYPOINTS.iter().enumerate() {
					if ui.selectable_label(current_pipeline == i, *name).clicked() {
						current_pipeline = i;
					}
				}
			});

		self.current_pipeline = current_pipeline;
	}

	pub fn sample(
		&self,
		environment_map: &EnvironmentMap,
		mixture: &Mixture,
		grid: &[u32],
		config: &SamplingConfigUiState,
	) -> MixtureSamplerResult {
		let context = &self.context;
		let thread_count = Self::THREADS_PER_ITERATION;

		let pipeline = self.pipelines[self.current_pipeline].1.clone();
		let layout = &pipeline.layout().set_layouts()[0];

		// build push constant from config
		let push_constants = SamplerPushConstants {
			ground_truth: environment_map.ground_truth,
			slice_size: config.slice_size,
			slice_expansion: config.slice_expansion,
			expansion_threshold: config.expansion_threshold,
			sample_count: Self::SAMPLES_PER_THREAD,
			thread_count: thread_count as u32,
			env_map_resolution: environment_map.dim,
			_padding: [0; 1],
		};

		// allocate buffers
		let buffer_mixture = create_buffer_and_write(context.allocators.memory.clone(), mixture);
		let buffer_grid = create_storage_from_iter(context.allocators.memory.clone(), grid.iter().copied());
		let buffer_enviroment_map = create_environment_map_buffer(context.allocators.memory.clone(), environment_map.data.as_ref());
		let buffer_thread_state = {
			let mut pcg = Pcg32::new(config.seed, 0);
			let thread_state = (0..thread_count)
				.map(|_| PerThreadState::new(pcg.gen() as u64))
				.collect::<Vec<_>>();
			create_storage_from_iter(context.allocators.memory.clone(), thread_state.into_iter())
		};

		let descriptor_set = PersistentDescriptorSet::new(
			&self.context.allocators.descriptor_set,
			layout.clone(),
			[
				WriteDescriptorSet::buffer(1, buffer_mixture.clone()),
				WriteDescriptorSet::buffer(2, buffer_enviroment_map.clone()),
				WriteDescriptorSet::buffer(3, buffer_grid.clone()),
				WriteDescriptorSet::buffer(20, buffer_thread_state.clone()),
			],
			[],
		)
		.expect("failed to create descriptor set");

		let mut builder = AutoCommandBufferBuilder::primary(
			&context.allocators.command_buffer,
			context.queue.0.queue_family_index(),
			CommandBufferUsage::MultipleSubmit,
		)
		.expect("failed to create command buffer builder");

		// bind static stuff
		builder
			.bind_pipeline_compute(pipeline.clone())
			.expect("failed to bind pipeline")
			.bind_descriptor_sets(
				PipelineBindPoint::Compute,
				pipeline.layout().clone(),
				0,
				descriptor_set.clone(),
			)
			.expect("failed to bind descriptor set");

		builder
			.push_constants(pipeline.layout().clone(), 0, push_constants)
			.expect("failed to push constants");

		for _ in 0..Self::ITERATIONS {
			let adjusted = Self::INVOCATION_GROUP_SIZE.adjust([thread_count as u32, 1, 1]);
			builder.dispatch(adjusted).expect("failed to dispatch");
		}

		let command_buffer = builder.build().expect("failed to build command buffer");

		// driver timeout forces multiple submits
		// we also want more precicion in the results, so we collect the results from multiple submits
		let mut results = Vec::new();
		for i in 0..Self::COMMAND_BUFFER_SUBMITS {
			info!("submitting command buffer {}/{}", i + 1, Self::COMMAND_BUFFER_SUBMITS);
			sync::now(context.device.clone())
				.then_execute(context.queue.0.clone(), command_buffer.clone())
				.expect("failed to execute command buffer")
				.then_signal_fence_and_flush()
				.expect("failed to signal fence and flush")
				.wait(None)
				.expect("failed to wait for future");

			// read back intermediate thread state and compute statistics
			let result = {
				let buffer = buffer_thread_state.read().unwrap();
				let inner = buffer.as_ref();
				self.reduce_thread_state(inner, Self::ITERATIONS as u64, Self::SAMPLES_PER_THREAD as u64)
			};
			results.push(result);

			// reset thread state
			let mut buffer = buffer_thread_state.write().unwrap();
			for state in buffer.iter_mut() {
				state.clear();
				// *state = PerThreadState::default();
			}
		}

		// combine results with another kahan sum
		let final_result = {
			let mut estimate = KahanSum::<f64>::new();
			let mut variance_sum = KahanSum::<f64>::new();
			let mut slice_miss_rate = KahanSum::<f64>::new();
			let mut total_samples_drawn = 0;

			let n = results.len() as f64;
			for result in results.iter() {
				// all local results are already normalized
				estimate += result.estimate / n;
				variance_sum += result.variance_sum / n;
				slice_miss_rate += result.slice_miss_rate / n;
				total_samples_drawn += result.sample_count;
			}

			MixtureSamplerResult {
				estimate: estimate.sum(),
				variance_sum: variance_sum.sum(),
				slice_miss_rate: slice_miss_rate.sum(),
				sample_count: total_samples_drawn,
			}
		};

		let num = format_big_number(final_result.sample_count);

		info!("samples drawn: {}", num);
		info!("result: {:?}", final_result);
		info!("ground truth delta: {}", final_result.estimate - environment_map.ground_truth);

		final_result
	}

	/// Reduces the thread state to a single result.
	///
	/// # Arguments
	/// - `thread_state` - The thread state buffer.
	/// - `invocation_count` - The number of invocations of the shader.
	/// - `sample_count` - The number of samples drawn by a single thread in a single invocation.
	fn reduce_thread_state(
		&self,
		thread_state: &[PerThreadState],
		invocation_count: u64,
		sample_count: u64,
	) -> MixtureSamplerResult {
		// ensure that all threads managed to finish the expected number of invocations
		for (i, state) in thread_state.iter().enumerate() {
			assert_eq!(
				state.invocation_counter, invocation_count,
				"thread state {} has not been invoked the expected number of times",
				i
			);
		}

		let samples_drawn_per_thread = (sample_count * invocation_count) as f64;

		let mut estimate = KahanSum::<f64>::new();
		let mut variance_sum = KahanSum::<f64>::new();
		let mut slice_misses = 0;

		// first aggregate all thread states
		for state in thread_state.iter() {
			estimate += state.estimate / samples_drawn_per_thread;
			variance_sum += state.variance_sum / samples_drawn_per_thread;
			slice_misses += state.slice_misses;
		}
		// then normalize the results since we have mutliple threads
		let num_threads_u64 = thread_state.len() as u64;
		let num_threads_f64 = thread_state.len() as f64;
		let result = MixtureSamplerResult {
			// sum of average for each thread divided by the number of threads
			estimate: estimate.sum() / num_threads_f64,
			variance_sum: variance_sum.sum() / num_threads_f64,

			// total number of misses divided by attempts to draw a sample
			slice_miss_rate: slice_misses as f64 / (samples_drawn_per_thread * num_threads_f64),

			// total number of samples drawn by all threads
			sample_count: sample_count * invocation_count * num_threads_u64,
		};

		result
	}
}

// pretty format number of samples to make it easier to read
// https://stackoverflow.com/a/67834588/1834100
pub fn format_big_number(number: u64) -> String {
	number
		.to_string()
		.as_bytes()
		.rchunks(3)
		.rev()
		.map(std::str::from_utf8)
		.collect::<Result<Vec<&str>, _>>()
		.unwrap()
		.join("_")
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_invocation_kernel_size() {
		let kernel_size = InvocationKernelSize([4, 4, 1]);
		assert_eq!(kernel_size.adjust([4, 4, 1]), [1, 1, 1]);
		assert_eq!(kernel_size.adjust([8, 8, 1]), [2, 2, 1]);
		assert_eq!(kernel_size.adjust([16, 16, 1]), [4, 4, 1]);
		assert_eq!(kernel_size.adjust([32, 32, 1]), [8, 8, 1]);
		assert_eq!(kernel_size.adjust([64, 64, 1]), [16, 16, 1]);
	}

	#[test]
	fn test_invalid_kernel_size() {
		let kernel_size = InvocationKernelSize([4, 4, 1]);
		assert!(std::panic::catch_unwind(|| kernel_size.adjust([3, 3, 1])).is_err());
		assert!(std::panic::catch_unwind(|| kernel_size.adjust([1, 1, 1])).is_err());
	}

	#[test]
	fn test_tiling() {
		let whole = [1000, 1000, 1000];
		let parts = InvocationKernelSize([100, 100, 100]).tile(whole);

		// we expect 10 calles in each dimension, so a total of 1000
		assert_eq!(parts.len(), 1000);
	}
}
