use bytemuck::{
	Pod,
	Zeroable,
};
use glam::{
	UVec2,
	UVec3,
	Vec3,
};
use pcg32::Pcg32;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::spirv;
use static_assertions::const_assert;

use crate::thesis::{
	env_map::{
		env_map_get_dir,
		env_map_validate,
	},
	mixture::{
		FullSliceSampler,
		Mixture,
		RandomTraversal,
		TowardsPdfTraversal,
		WeightedRandomTraversal,
	},
	mixture_ext::{
		MixturePDF,
		MixtureVariance,
	},
};

const_assert!(core::mem::size_of::<SamplerPushConstants>() <= 128);
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct SamplerPushConstants {
	/// Size of the slice of lobes that we are going to sample.
	pub slice_size: u32,

	/// Size of additional slices that the initial selection can be expanded by.
	pub slice_expansion: u32,

	/// Minimum pdf required for lobes to be included in the expansion part.
	pub expansion_threshold: f32,

	/// Number of threads that are going to be used per invocation.
	pub thread_count: u32,

	/// The number of samples that each thread is going to sample
	pub sample_count: u32,

	pub _padding: [u32; 1],

	/// Resolution of the environment map.
	pub env_map_resolution: UVec2,

	/// Actual ground truth value that we are trying to estimate.
	pub ground_truth: f64,
}

#[derive(Copy, Clone, Pod, Zeroable, Debug)]
#[repr(C)]
pub struct PerThreadState {
	/// Number of times that the selected slice did not contain the origin lobe.
	pub slice_misses: u64,

	/// Current estimate of the integral.
	pub estimate: f64,

	/// Current sum of the variances of the samples.
	pub variance_sum: f64,

	/// Number of times that the sampler was invoked.
	///
	/// This is stored per thread to avoid synchronization.
	pub invocation_counter: u64,

	/// Seed for the random number generator.
	///
	/// Updated at the end of each invocation for the next invocation.
	pub seed: u64,

	/// Per thread allocated memory for the VMF selection.
	pub vmf_selection_memory: [VMFSelection; VMF_SELECTION_SIZE],
}

pub const VMF_SELECTION_SIZE: usize = 1024;

#[derive(Copy, Clone, Pod, Zeroable, Debug, Default, PartialEq)]
#[repr(C)]
pub struct VMFSelection(
	/// The pdf of the selected VMF.
	pub f32,
	/// The index of the selected VMF.
	pub u32,
);

impl PerThreadState {
	pub fn new(seed: u64) -> Self {
		Self {
			slice_misses: 0,
			estimate: 0.0,
			variance_sum: 0.0,
			invocation_counter: 0,
			seed,
			vmf_selection_memory: [VMFSelection(0.0, 0); VMF_SELECTION_SIZE],
		}
	}

	pub fn clear(&mut self) {
		self.slice_misses = 0;
		self.estimate = 0.0;
		self.variance_sum = 0.0;
		self.invocation_counter = 0;
	}
}

#[spirv(compute(threads(64)))]
pub fn mixture_all(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &SamplerPushConstants,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 20)] state: &mut [PerThreadState],
) {
	sample_internal::<FullSampling>(id, constants, mixture, grid, env_map, state);
}

#[spirv(compute(threads(64)))]
pub fn mixture_morton(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &SamplerPushConstants,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 20)] state: &mut [PerThreadState],
) {
	sample_internal::<MortonSampling>(id, constants, mixture, grid, env_map, state);
}

#[spirv(compute(threads(64)))]
pub fn mixture_knn(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &SamplerPushConstants,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 20)] state: &mut [PerThreadState],
) {
	sample_internal::<KNNSampling>(id, constants, mixture, grid, env_map, state);
}

#[spirv(compute(threads(64)))]
pub fn mixture_tree_random_full(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &SamplerPushConstants,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 20)] state: &mut [PerThreadState],
) {
	sample_internal::<TreeRandomFullStrategy>(id, constants, mixture, grid, env_map, state);
}

#[spirv(compute(threads(64)))]
pub fn mixture_tree_towards_pdf_full(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &SamplerPushConstants,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 20)] state: &mut [PerThreadState],
) {
	sample_internal::<TreeTowardsPdfStrategy>(id, constants, mixture, grid, env_map, state);
}

#[spirv(compute(threads(64)))]
pub fn mixture_tree_weighted_random_full(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &SamplerPushConstants,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 20)] state: &mut [PerThreadState],
) {
	sample_internal::<TreeWeightedRandomStrategy>(id, constants, mixture, grid, env_map, state);
}

// noinspection RsAssertEqual
fn sample_internal<Sampler: SamplingStrategy>(
	id: UVec3,
	constants: &SamplerPushConstants,
	mixture: &Mixture,
	grid: &[u32],
	env_map: &[f32],
	state: &mut [PerThreadState],
) {
	// force usage of grid
	assert!(grid[0] != grid[1]);

	env_map_validate(constants.env_map_resolution, env_map);

	let idx = id.x as usize;
	let state = &mut state[idx];

	// unpack variables and do type conversions. required since we need to use fixed size types
	let ground_truth = constants.ground_truth;
	let sample_count = constants.sample_count as usize;

	// add id to prevent threads from synchronizing
	let mut pcg = Pcg32::new(state.seed + idx as u64, 0);

	let mut estimate = state.estimate;
	let mut variance_sum = state.variance_sum;
	let mut slice_misses = state.slice_misses;

	for _ in 0..sample_count {
		let (dir, pdf, valid) = Sampler::sample(
			mixture,
			&mut pcg,
			constants.slice_size as usize,
			constants.slice_expansion as usize,
			constants.expansion_threshold,
			&mut state.vmf_selection_memory,
		);

		let contribution = if valid {
			// proper sample, we can use it
			let color = env_map_get_dir(env_map, constants.env_map_resolution, dir) as f64;
			pdf.estimate(color)
		} else {
			// sample is imposible, we set it to 0 to account for valid case
			slice_misses += 1;
			0.0
		};

		// update the estimate and variance sum
		estimate += contribution;
		variance_sum += contribution.variance(ground_truth);
	}

	// write back state into buffer
	state.estimate = estimate;
	state.variance_sum = variance_sum;
	state.slice_misses = slice_misses;
	state.invocation_counter += 1;

	state.seed = (pcg.gen() as u64) << 32 | (pcg.gen() as u64);
}

pub trait SamplingStrategy {
	/// Samples a direction from the mixture.
	///
	/// # Parameters
	/// - `mixture`: The mixture to sample from.
	/// - `pcg`: The random number generator to use.
	/// - `slice_size`: The size of the slice around the sample that we are going to sample.
	///
	/// # Returns
	/// A tuple containing the sampled direction, the pdf of the sample and a boolean indicating if the sample is valid.
	/// The sample is invalid if the slice does not contain the origin lobe.
	fn sample(
		mixture: &Mixture,
		pcg: &mut Pcg32,
		slice_size: usize,
		slice_expansion: usize,
		expansion_threshold: f32,
		vmf_selection_memory: &mut [VMFSelection; VMF_SELECTION_SIZE],
	) -> (Vec3, f32, bool);
}

// slice based sampling strategies
struct FullSampling;
impl SamplingStrategy for FullSampling {
	fn sample(
		mixture: &Mixture,
		pcg: &mut Pcg32,
		_slice_size: usize,
		_slice_expansion: usize,
		_expansion_threshold: f32,
		_vmf_selection_memory: &mut [VMFSelection; VMF_SELECTION_SIZE],
	) -> (Vec3, f32, bool) {
		mixture.sample_all(pcg)
	}
}

struct MortonSampling;
impl SamplingStrategy for MortonSampling {
	fn sample(
		mixture: &Mixture,
		pcg: &mut Pcg32,
		slice_size: usize,
		slice_expansion: usize,
		expansion_threshold: f32,
		_vmf_selection_memory: &mut [VMFSelection; VMF_SELECTION_SIZE],
	) -> (Vec3, f32, bool) {
		mixture.sample_morton(pcg, slice_size, slice_expansion, expansion_threshold)
	}
}

struct KNNSampling;
impl SamplingStrategy for KNNSampling {
	#[allow(clippy::needless_range_loop)]
	fn sample(
		mixture: &Mixture,
		pcg: &mut Pcg32,
		slice_size: usize,
		_slice_expansion: usize,
		_expansion_threshold: f32,
		vmf_selection_memory: &mut [VMFSelection; VMF_SELECTION_SIZE],
	) -> (Vec3, f32, bool) {
		// clear the memory before sampling
		for i in 0..vmf_selection_memory.len() {
			vmf_selection_memory[i] = VMFSelection(0.0, 0);
		}

		mixture.sample_knn(pcg, vmf_selection_memory, slice_size)
	}
}

// tree sampling strategies
trait TreeSamplingStrayegy {
	fn sample_tree(mixture: &Mixture, pcg: &mut Pcg32, slice_size: usize) -> (Vec3, f32, bool);
}

impl<T> SamplingStrategy for T
where T: TreeSamplingStrayegy
{
	fn sample(
		mixture: &Mixture,
		pcg: &mut Pcg32,
		slice_size: usize,
		_slice_expansion: usize,
		_expansion_threshold: f32,
		_vmf_selection_memory: &mut [VMFSelection; VMF_SELECTION_SIZE],
	) -> (Vec3, f32, bool) {
		T::sample_tree(mixture, pcg, slice_size)
	}
}

struct TreeRandomFullStrategy;
impl TreeSamplingStrayegy for TreeRandomFullStrategy {
	fn sample_tree(mixture: &Mixture, pcg: &mut Pcg32, slice_size: usize) -> (Vec3, f32, bool) {
		mixture.sample_hierarchy::<RandomTraversal, FullSliceSampler>(pcg, slice_size)
	}
}

struct TreeTowardsPdfStrategy;
impl TreeSamplingStrayegy for TreeTowardsPdfStrategy {
	fn sample_tree(mixture: &Mixture, pcg: &mut Pcg32, slice_size: usize) -> (Vec3, f32, bool) {
		mixture.sample_hierarchy::<TowardsPdfTraversal, FullSliceSampler>(pcg, slice_size)
	}
}

struct TreeWeightedRandomStrategy;
impl TreeSamplingStrayegy for TreeWeightedRandomStrategy {
	fn sample_tree(mixture: &Mixture, pcg: &mut Pcg32, slice_size: usize) -> (Vec3, f32, bool) {
		mixture.sample_hierarchy::<WeightedRandomTraversal, FullSliceSampler>(pcg, slice_size)
	}
}
