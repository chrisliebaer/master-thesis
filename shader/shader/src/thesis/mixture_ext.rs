use core::{
	cmp::min,
	ops::{
		Div,
		MulAssign,
	},
};

use glam::Vec3;
use spirv_std::float::Float;

use crate::thesis::{
	entrypoints::{
		sample_shader::VMFSelection,
		validate_scalar,
	},
	mixture::Mixture,
	vmf_grid::GridLayout,
};

impl Mixture {
	pub fn pdf_sum_all(&self, dir: Vec3) -> f32 {
		let mut pdf = 0.0;
		for i in 0..self.size() {
			let (component, weight) = self.get_component(i);
			pdf += component.pdf(dir) * weight;
		}
		pdf
	}

	pub fn pdf_morton_slice(&self, dir: Vec3, (start, end): (usize, usize)) -> f32 {
		let mut pdf = 0.0;
		for i in start..end {
			let (_, component, weight) = self.get_morton_component(i);
			pdf += component.pdf(dir) * weight;
		}
		pdf
	}

	pub fn pdf_full_and_morton(&self, dir: Vec3, (start, end): (usize, usize)) -> (f32, f32) {
		let mut pdf_full = 0.0;
		let mut pdf_morton = 0.0;

		// we can't directly construct a set of non-slice components, instead iterate over all components
		for i in 0..self.size() {
			let (_, component, weight) = self.get_morton_component(i);
			pdf_full += component.pdf(dir) * weight;
			if start <= i && i < end {
				pdf_morton += component.pdf(dir) * weight;
			}
		}

		(pdf_full, pdf_morton)
	}

	pub fn pdf_full_and_hierarchy(&self, dir: Vec3, (start, end): (usize, usize)) -> (f32, f32) {
		let mut pdf_full = 0.0;
		let mut pdf_hierarchy = 0.0;

		// we can't directly construct a set of non-slice components, instead iterate over all components
		for i in 0..self.size() {
			let (_, component, weight) = self.get_tree_slice_component(i);
			pdf_full += component.pdf(dir) * weight;
			if start <= i && i < end {
				pdf_hierarchy += component.pdf(dir) * weight;
			}
		}

		(pdf_full, pdf_hierarchy)
	}

	pub fn pdf_full_and_knn<const N: usize>(
		&self,
		dir: Vec3,
		selection_size: usize,
		count_leaf_components: bool,
	) -> (f32, f32, usize) {
		let mut vmf_storage = [VMFSelection(0.0, 0); N];
		let (pdf_knnm, _, visited_nodes) = self.pdf_knn_tree(dir, 0, &mut vmf_storage, selection_size, count_leaf_components);
		let pdf_full = self.pdf_sum_all(dir);
		(pdf_full, pdf_knnm, visited_nodes)
	}

	#[allow(clippy::needless_range_loop)]
	pub fn pdf_knn_tree<const N: usize>(
		&self,
		dir: Vec3,
		origin: usize,
		vmf_storage: &mut [VMFSelection; N],
		selection_size: usize,
		count_leaf_components: bool,
	) -> (f32, bool, usize) {
		assert!(selection_size <= vmf_storage.len());

		// the last node index we visited
		let mut last_index = -1;

		// the current node index we are visiting
		let mut current_index = 0i32;

		// our current lowest bound (we explore all paths that promise a better bound)
		let mut worst_bound = 0.0;

		// the number of visited nodes
		let mut visited_nodes = 0;

		// perform recursionless traversal (gpu doesn't support recursion)
		while current_index != -1 {
			// we only want to count descents into a new node, which is the case if the last index was smaller than the current
			if last_index < current_index {
				visited_nodes += 1;
			}

			// if node is leaf, we need to add number of components to visited nodes, since they will all be checked
			let node = self.get_hierarchy_node(current_index as usize);
			if count_leaf_components && node.is_leaf() {
				let (start, end) = node.get_slice();
				visited_nodes += end - start;
			}

			let old_index = current_index;
			(current_index, worst_bound) =
				self.advance_knn_search(dir, vmf_storage, selection_size, last_index, current_index, worst_bound);

			last_index = old_index;
		}

		// sum up all pdfs and check if origin lob is included
		let mut pdf_total = 0.0;
		let mut found = false;
		for i in 0..selection_size {
			let VMFSelection(pdf, mixture_index) = vmf_storage[i];

			pdf_total += pdf;

			if mixture_index as usize == origin {
				found = true;
			}
		}
		(pdf_total, found, visited_nodes)
	}

	pub fn pdf_grid(&self, dir: Vec3, origin: usize, index: usize, grid: &[u32], slice_size: usize) -> (f32, bool) {
		let mut pdf = 0.0;
		let mut found = false;

		let grid_layout = self.grid_layout();

		let slice_size = min(slice_size, grid_layout.bucket_size());

		for i in 0..slice_size {
			let component_index = grid[index * grid_layout.bucket_size() + i] as usize;
			let (component, weight) = self.get_component(component_index);
			pdf += component.pdf(dir) * weight;

			if component_index == origin {
				found = true;
			}
		}

		(pdf, found)
	}

	pub fn advance_knn_search<const N: usize>(
		&self,
		dir: Vec3,
		vmf_storage: &mut [VMFSelection; N],
		selection_size: usize,
		last_index: i32,
		current_index: i32,
		worst_bound: f32,
	) -> (i32, f32) {
		let node = self.get_hierarchy_node(current_index as usize);

		let (last_index, mut current_index, mut worst_bound) = (last_index, current_index, worst_bound);

		if node.is_leaf() {
			// we check the entire slice against our current selection
			let (start, end) = node.get_slice();
			for i in start..end {
				let (mixture_index, component, weight) = self.get_tree_slice_component(i);
				let pdf = component.pdf(dir) * weight;

				// only need to check entire slice if the worst bound is worse than this vmf pdf
				if pdf > worst_bound {
					worst_bound = replace_smallest(vmf_storage, selection_size, (mixture_index, pdf));
				}
			}

			// we reached a leaf and need to turn back
			current_index = node.get_parent();
		} else {
			let left_child = self.get_hierarchy_node((current_index + 1) as usize);
			let left_pdf = left_child.upper_bound(dir);
			let right_child = self.get_hierarchy_node(node.get_right_child() as usize);
			let right_pdf = right_child.upper_bound(dir);

			// abstract indices from direction
			let ((small_index, small_pdf), (big_index, big_pdf)) = if left_pdf < right_pdf {
				((current_index + 1, left_pdf), (node.get_right_child(), right_pdf))
			} else {
				((node.get_right_child(), right_pdf), (current_index + 1, left_pdf))
			};

			// there are 3 possible states
			let (target, upper_pdf) = if last_index < current_index {
				// we are entering the node the first time (move towards bigger pdf)
				(big_index, big_pdf)
			} else if last_index == big_index {
				// we ascended from the bigger index (we need to check if smaller pdf is still good)
				(small_index, small_pdf)
			} else {
				// we ascended from the smaller index (ascent back to parent and fake large pdf so always take this path)
				(node.get_parent(), 2.0)
			};

			// we know which path we are going to take, but we need to check if the path also promises a better pdf
			if upper_pdf > worst_bound {
				// target promises a better pdf, we continue
				current_index = target;
			} else {
				// target should not contain a better pdf, we ascend
				current_index = node.get_parent();
			}
		}

		(current_index, worst_bound)
	}
}

/// Replaces the smallest value in the slice with the given value.
///
/// This function assumes that any given parameter is actually replacing the smallest value in the slice and will panic,
/// if there is no smaller value.
///
/// # Arguments
/// * `vmf_storage` - The slice to replace the smallest value in.
/// * `slice_size` - The size of the slice (required since we can't use subslices and the slice might be larger than the
///   actual data).
/// * `idx` - The index of the lobe in the hierarchy indirection list.
///
/// # Returns
/// The updated smallest value. Note that this might not be the value that was replaced.
#[allow(clippy::manual_swap, clippy::needless_range_loop)]
fn replace_smallest<const N: usize>(vmf_storage: &mut [VMFSelection; N], slice_size: usize, (idx, pdf): (usize, f32)) -> f32 {
	// https://stackoverflow.com/questions/26335400/finding-the-second-minimum

	let mut smallest_idx = 0;
	let mut smallest = vmf_storage[0].0;
	let mut second_smallest = vmf_storage[1].0;

	if second_smallest < smallest {
		let tmp = smallest;
		smallest = second_smallest;
		second_smallest = tmp;
		smallest_idx = 1;
	}

	// find replacement index
	for i in 2..slice_size {
		let slot = vmf_storage[i].0;
		if slot < smallest {
			second_smallest = smallest;
			smallest = slot;
			smallest_idx = i;
		} else if slot < second_smallest {
			second_smallest = slot;
		}
	}

	assert!(pdf > smallest);

	// replace smallest value
	vmf_storage[smallest_idx] = VMFSelection(pdf, idx as u32);

	// second smallest could still be larger than the new smallest value
	if second_smallest < pdf { second_smallest } else { pdf }
}

pub trait MixturePDF {
	fn estimate<T>(&self, color: T) -> T
	where T: From<f32> + Div<Output = T>;
}

impl MixturePDF for f32 {
	fn estimate<T>(&self, color: T) -> T
	where T: From<f32> + Div<Output = T> {
		let pdf = T::from(*self);
		color / pdf
	}
}

pub trait MixtureVariance<T> {
	fn variance(&self, ground_truth: T) -> T
	where T: Float;
}

impl<T> MixtureVariance<T> for T {
	fn variance(&self, ground_truth: T) -> T
	where T: Float {
		(*self - ground_truth).powi(2)
	}
}

#[derive(Debug)]
pub struct AnalyticalMixtureStatistics<T>
where T: Float + From<f32> {
	estimate: T,
	mse: T,
	rmse: T,
	density: T,
	miss_rate: T,
	visited_nodes: T,
}

impl<T> AnalyticalMixtureStatistics<T>
where T: Float + From<f32> + MulAssign
{
	#[cfg(not(target_arch = "spirv"))]
	pub fn for_ideal(mixture: &Mixture, dir: Vec3, color: T, ground_truth: T, slice_size: usize) -> Self {
		// find mixtures with highest pdf for direction

		let mut pdfs = Vec::with_capacity(mixture.size());
		let mut pdf_full = 0.0;
		for i in 0..mixture.size() {
			let (component, weight) = mixture.get_component(i);
			let pdf = component.pdf(dir) * weight;
			pdf_full += pdf;
			pdfs.push((i, pdf));
		}

		// sort by pdf descending
		pdfs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
		let pdf_partial = pdfs.iter().take(slice_size).map(|(_, pdf)| *pdf).sum::<f32>();

		Self::for_internal(color, ground_truth, pdf_full, pdf_partial, mixture.size())
	}

	pub fn for_uniform(color: T, ground_truth: T, density: T) -> Self {
		let estimate = color / density;
		let mse = estimate.variance(ground_truth);

		Self {
			estimate,
			mse,
			rmse: mse.sqrt(),
			density,
			miss_rate: T::zero(),
			visited_nodes: T::zero(),
		}
	}

	pub fn for_full(mixture: &Mixture, dir: Vec3, color: T, ground_truth: T) -> Self {
		let pdf_full = mixture.pdf_sum_all(dir);
		let estimate = pdf_full.estimate(color);
		let pdf_full: T = pdf_full.into();
		let mse = estimate.variance(ground_truth);

		Self {
			estimate,
			mse,
			rmse: mse.sqrt(),
			density: pdf_full,
			miss_rate: T::zero(),
			visited_nodes: <T as From<f32>>::from(mixture.size() as f32),
		}
	}

	pub fn for_morton(mixture: &Mixture, dir: Vec3, color: T, ground_truth: T, (start, end): (usize, usize)) -> Self {
		let (pdf_full, pdf_morton) = mixture.pdf_full_and_morton(dir, (start, end));
		Self::for_internal(color, ground_truth, pdf_full, pdf_morton, end - start)
	}

	pub fn for_hierarchy(
		mixture: &Mixture,
		dir: Vec3,
		color: T,
		ground_truth: T,
		(start, end): (usize, usize),
		nodes_visited: usize,
	) -> Self {
		let (pdf_full, pdf_hierarchy) = mixture.pdf_full_and_hierarchy(dir, (start, end));
		Self::for_internal(color, ground_truth, pdf_full, pdf_hierarchy, nodes_visited)
	}

	pub fn for_knn<const N: usize>(
		mixture: &Mixture,
		dir: Vec3,
		color: T,
		ground_truth: T,
		selection_size: usize,
		count_leaf_components: bool,
	) -> Self {
		let (pdf_full, pdf_knnm, visited_nodes) = mixture.pdf_full_and_knn::<N>(dir, selection_size, count_leaf_components);
		Self::for_internal(color, ground_truth, pdf_full, pdf_knnm, visited_nodes)
	}

	pub fn for_grid(
		mixture: &Mixture,
		dir: Vec3,
		cell_index: usize,
		grid: &[u32],
		color: T,
		ground_truth: T,
		slice_size: usize,
	) -> Self {
		let (pdf_grid, _) = mixture.pdf_grid(dir, 0, cell_index, grid, slice_size);
		let pdf_full = mixture.pdf_sum_all(dir);

		Self::for_internal(color, ground_truth, pdf_full, pdf_grid, slice_size)
	}

	fn for_internal(color: T, ground_truth: T, pdf_full: f32, pdf_partial: f32, visited_nodes: usize) -> Self {
		// uniform density for a sphere
		const UNIFORM_DENSITY: f32 = 1.0 / (4.0 * core::f32::consts::PI);

		// todo: make this configurable
		let uniform_weight = 0.3;

		validate_scalar(pdf_full);
		validate_scalar(pdf_partial);

		// rescale mixture pdf to make room for a fake component with uniform density
		let pdf_full = pdf_full * (1.0 - uniform_weight) + uniform_weight * UNIFORM_DENSITY;
		let pdf_partial = pdf_partial * (1.0 - uniform_weight) + uniform_weight * UNIFORM_DENSITY;

		let estimate_morton = pdf_partial.estimate(color);

		// convert pdfs into T
		let pdf_full: T = pdf_full.into();
		let pdf_partial: T = pdf_partial.into();

		// we now need to consider that we can either use a slice lobe for generation, or not
		let (estimate, mse) = {
			// normalize pdf_morton since rmse fractions need to sum to 1
			let pdf_morton_norm = pdf_partial / pdf_full;

			let estimate = estimate_morton * pdf_morton_norm;
			let mse = estimate_morton.variance(ground_truth) * pdf_morton_norm
				+ T::zero().variance(ground_truth) * (T::one() - pdf_morton_norm);
			(estimate, mse)
		};

		Self {
			estimate,
			mse,
			rmse: mse.sqrt(),
			density: pdf_full,
			miss_rate: T::one() - pdf_partial / pdf_full,
			visited_nodes: <T as From<f32>>::from(visited_nodes as f32),
		}
	}

	pub fn reduce(s1: Self, s2: Self) -> Self {
		Self {
			estimate: s1.estimate + s2.estimate,
			mse: s1.mse + s2.mse,
			rmse: s1.rmse + s2.rmse,
			density: s1.density + s2.density,
			miss_rate: s1.miss_rate + s2.miss_rate,
			visited_nodes: s1.visited_nodes + s2.visited_nodes,
		}
	}

	pub fn estimate(&self) -> T {
		self.estimate
	}

	pub fn mse(&self) -> T {
		self.mse
	}

	pub fn rmse(&self) -> T {
		self.rmse
	}

	pub fn density(&self) -> T {
		self.density
	}

	pub fn miss_rate(&self) -> T {
		self.miss_rate
	}

	pub fn visited_nodes(&self) -> T {
		self.visited_nodes
	}
}
