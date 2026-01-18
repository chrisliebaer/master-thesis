use core::cmp::{
	max,
	min,
};

use bytemuck::{
	Pod,
	Zeroable,
};
use glam::{
	UVec2,
	Vec2,
	Vec3,
};
use pcg32::Pcg32;

use crate::thesis::{
	entrypoints::{
		sample_shader::VMFSelection,
		validate_vector3,
	},
	math::dir2pnt,
	mixture_tree::TreeNode,
	morton::pnt2morton,
	pcg32_ext::Pcg32Ext,
	vmf_distribution::VMFDistribution,
	vmf_grid::GridLayout,
	VMF_COUNT,
};

/// This function performs a binary search on a sorted array and returns the index of the first element that is greater
/// or equal to the given value.
fn binary_search_greater_than<T: PartialOrd, const N: usize>(arr: &[T; N], value: &T, len: usize) -> usize {
	if len == 0 {
		panic!("Array is empty");
	}

	let mut left = 0;
	let mut right = len - 1;

	while left < right {
		let mid = (left + right) / 2;

		if arr[mid] < *value {
			left = mid + 1;
		} else {
			right = mid;
		}
	}

	left
}

fn binary_search_closest<'a, T: PartialOrd, const N: usize>(arr: &[T; N], value: &T, size: usize) -> usize
where for<'b> &'b T: core::ops::Sub<Output = T> {
	let idx = binary_search_greater_than(arr, value, size);

	// we know that id is the first element that is greater than value
	if idx > 0 {
		let left = &arr[idx - 1];
		let right = &arr[idx];

		// two options: last value is smaller or larger than value
		if right < value {
			// right is smaller than value, so we return it
			// if we didn't we would cause an overflow when subtracting in the next step
			idx
		} else if value - left < right - value {
			// left is closer
			idx - 1
		} else {
			// right is closer
			idx
		}
	} else {
		// no next value, so we return the current idx
		idx
	}
}

#[derive(Copy, Clone, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct Mixture {
	/// Array of von Mises-Fisher distributions.
	vmf_distributions: [VMFDistribution; VMF_COUNT],

	/// Array of weights for the von Mises-Fisher distributions.
	/// The weights must sum to 1.0.
	/// One weight for each von Mises-Fisher distribution.
	weights: [f32; VMF_COUNT],

	/// Accumulated weights, allows for binary search.
	///
	/// The weights are accumulated, so that the first element is the weight of the first distribution,
	/// the second element is the sum of the first and second distribution, and so on.
	/// The last element is the sum of all weights.
	accumulated_weights: [f32; VMF_COUNT],

	/// Sorted arraya of Morton z-indices pointing to their VMF lobe.
	zindices: [u32; VMF_COUNT],

	/// Index of the z-indices in the original array.
	zindices_index: [u32; VMF_COUNT],

	/// Array of tree nodes that represent the hierarchy of the mixture model.
	hierarchy: [TreeNode; VMF_COUNT],

	/// Indirection array for the hierarchy.
	hierarchy_index: [u32; VMF_COUNT],

	/// Number of valid entries in the arrays.
	count: u32,

	/// Number of nodes in the hierarchy.
	tree_node_count: u32,

	_padding: [u32; 2],

	/// Layout of the accompanying grid.
	grid_layout: GridLayout,
}

impl Mixture {
	pub fn add_component(&mut self, vmf: VMFDistribution, weight: f32) {
		let count = self.count as usize;

		if count >= VMF_COUNT {
			panic!("Mixture is full");
		}

		self.vmf_distributions[count] = vmf;
		self.weights[count] = weight;
		self.accumulated_weights[count] = if count == 0 {
			weight
		} else {
			self.accumulated_weights[count - 1] + weight
		};

		let pnt = dir2pnt(vmf.mean());
		self.zindices[count] = pnt2morton(pnt);
		self.zindices_index[count] = count as u32;

		self.count += 1;
	}

	#[cfg(not(target_arch = "spirv"))]
	pub fn finalize<Builder: GpuTreeBuilder>(&mut self, tree_builder: &Builder, grid_layout: &GridLayout) {
		if self.count == 0 {
			panic!("Mixture is empty");
		}

		let sum = self.accumulated_weights[self.count as usize - 1];
		if (sum - 1.0).abs() > 1e-4 {
			panic!("Weights do not sum to 1.0 but {}", sum);
		}

		// TODO: self.optimize();

		// zip both arrays, sort them by the z-indices and write the sorted arrays back
		let mut zipped = self
			.zindices
			.iter()
			.copied()
			.zip(self.zindices_index.iter().copied())
			.take(self.count as usize)
			.collect::<Vec<_>>();
		zipped.sort_by_key(|&(a, _)| a);

		// check if zindices are in ascending order
		assert!(zipped.windows(2).all(|w| w[0].0 <= w[1].0));

		// confirm that each index properly points to the correct VMF lobe
		for (zindex, index) in zipped.iter() {
			let pnt = dir2pnt(self.vmf_distributions[*index as usize].mean());
			let zindex2 = pnt2morton(pnt);
			assert_eq!(*zindex, zindex2);
		}

		for (i, (zindex, index)) in zipped.iter().enumerate() {
			self.zindices[i] = *zindex;
			self.zindices_index[i] = *index;
		}

		let mut pcg = Pcg32::new(0, 0);
		let (indirection, hierarchy) = tree_builder.tree_from_mixture(self, &mut pcg);

		let first = self.get_component(0);
		println!(
			"Hierarchy has {} nodes and {} indirections (first component is {}:{})",
			hierarchy.len(),
			indirection.len(),
			first.0.mean(),
			first.0.kappa()
		);

		let size = self.count as usize;

		self.hierarchy[..hierarchy.len()].copy_from_slice(&hierarchy);

		assert_eq!(indirection.len(), size);
		self.hierarchy_index[..size].copy_from_slice(&indirection);

		self.tree_node_count = hierarchy.len() as u32;
		self.grid_layout = *grid_layout;
	}

	#[cfg(not(target_arch = "spirv"))]
	pub fn iter(&self) -> impl Iterator<Item = (&VMFDistribution, f32)> {
		(0..self.count as usize).map(move |i| self.get_component(i))
	}

	/// Updates the internal ordering of VMF lobes based on their weights to evenly distribute them.
	///
	/// This attempts to balance vastly different weights to ensure that binary search will roughly need the same amount
	/// of steps for each component. This is done by inserting all components in order of their weights in reverse binary
	/// search order.
	pub fn optimize(&mut self) {
		unimplemented!("kekw, not implemented yet")
	}

	pub fn size(&self) -> usize {
		self.count as usize
	}

	pub fn get_count(&self) -> usize {
		self.count as usize
	}

	pub fn grid_layout(&self) -> &GridLayout {
		&self.grid_layout
	}

	pub fn tree_node_count(&self) -> usize {
		self.tree_node_count as usize
	}

	/// This function returns a component and its weight based on an index.
	pub fn get_component(&self, idx: usize) -> (&VMFDistribution, f32) {
		if idx < self.count as usize {
			let vmf = &self.vmf_distributions[idx];
			let weight = self.weights[idx];
			(vmf, weight)
		} else {
			panic!("Index out of bounds");
		}
	}

	pub fn get_component_mut(&mut self, idx: usize) -> (&mut VMFDistribution, &mut f32) {
		if idx < self.count as usize {
			let vmf = &mut self.vmf_distributions[idx];
			let weight = &mut self.weights[idx];
			(vmf, weight)
		} else {
			panic!("Index out of bounds");
		}
	}

	/// This function works similar to `get_component`, but it uses the sorted z-indices to find the component.
	pub fn get_morton_component(&self, idx: usize) -> (usize, &VMFDistribution, f32) {
		if idx < self.count as usize {
			let index = self.zindices_index[idx] as usize;
			// include actual index to allow for easy comparison
			let (vmf, weight) = self.get_component(index);
			(index, vmf, weight)
		} else {
			panic!("Index out of bounds");
		}
	}

	/// This function works similar to `get_component`, but it uses the hierarchy to find the component.
	pub fn get_tree_slice_component(&self, idx: usize) -> (usize, &VMFDistribution, f32) {
		if idx < self.count as usize {
			let index = self.hierarchy_index[idx] as usize;
			// include actual index to allow for easy comparison
			let (vmf, weight) = self.get_component(index);
			(index, vmf, weight)
		} else {
			panic!("Index out of bounds");
		}
	}

	/// This function uses the accumulated weights to select a component based on a random number.
	/// It uses a binary search to find the correct component.
	fn select_component(&self, rng: f32) -> usize {
		if self.count == 0 {
			panic!("Mixture is empty");
		}

		binary_search_greater_than(&self.accumulated_weights, &rng, self.count as usize)
	}

	/// This function draws a single sample from the mixture model using all components.
	///
	/// It first creates a random number to select a component based on the weights.
	/// Then it samples from the selected component and calculates the pdf of that sample for all components.
	pub fn sample_all(&self, pcg: &mut Pcg32) -> (Vec3, f32, bool) {
		let (sample, _) = self.sample(pcg);

		// build sum of all pdfs for the sample
		let mut pdf = 0.0;
		for i in 0..self.vmf_distributions.len() {
			pdf += self.vmf_distributions[i].pdf(sample) * self.weights[i];
		}

		(sample, pdf, true)
	}

	/// Draw a single sample using morton indices to select a subset of components for pdf calculation.
	pub fn sample_morton(
		&self,
		pcg: &mut Pcg32,
		range: usize,
		slice_expansion: usize,
		expansion_threshold: f32,
	) -> (Vec3, f32, bool) {
		let (sample, origin) = self.sample(pcg);

		// multiply each axis with unsigned max value to get a 32 bit morton index
		let (start, end) = MortonSliceSelection::select_slice(self, sample, range);
		let (start, end) = if slice_expansion > 0 {
			self.expand_slice((start, end), range + slice_expansion, expansion_threshold, sample)
		} else {
			(start, end)
		};

		// compute pdf for the selected components (and track if the original component is in the range)
		let mut pdf = 0.0;
		let mut found = false;
		for i in start..end {
			let (index, vmf, weight) = self.get_morton_component(i);

			pdf += vmf.pdf(sample) * weight;

			if index == origin {
				found = true;
			}
		}

		if !found {
			// if the original component is not in the range, the sample is impossible
			pdf = 0.0;
		}

		(sample, pdf, found)
	}

	pub fn sample_hierarchy<Traversal: TreeTraversal, Sampler: TreeSliceSampler>(
		&self,
		pcg: &mut Pcg32,
		slice_size: usize,
	) -> (Vec3, f32, bool) {
		let (sample, origin) = self.sample(pcg);
		let (start, end, _) = Traversal::traverse(self, sample, pcg, slice_size);
		let (sample, pdf, found) = Sampler::sample_slice(self, sample, pcg, origin, start, end);

		(sample, pdf, found)
	}

	pub fn sample_knn<const N: usize>(
		&self,
		pcg: &mut Pcg32,
		selection_memory: &mut [VMFSelection; N],
		selection_size: usize,
	) -> (Vec3, f32, bool) {
		let (sample, origin) = self.sample(pcg);
		let (pdf, found, _) = self.pdf_knn_tree(sample, origin, selection_memory, selection_size, false);

		(sample, pdf, found)
	}

	/// This function selects a component and samples from it.
	///
	/// It returns the sampled direction and the index of the selected component.
	pub fn sample(&self, pcg: &mut Pcg32) -> (Vec3, usize) {
		let idx = self.select_component(pcg.gen_f32());
		let component = &self.vmf_distributions[idx];

		let sample = {
			let rnd = Vec2::new(pcg.gen_f32(), pcg.gen_f32());

			component.sample(rnd)
		};
		validate_vector3(sample);

		(sample, idx)
	}

	pub fn get_hierarchy_node(&self, idx: usize) -> &TreeNode {
		&self.hierarchy[idx]
	}
}

impl Default for Mixture {
	fn default() -> Self {
		Self {
			vmf_distributions: [VMFDistribution::new(Vec3::ZERO, 0.0); VMF_COUNT],
			weights: [0.0f32; VMF_COUNT],
			accumulated_weights: [0.0f32; VMF_COUNT],
			zindices: [0; VMF_COUNT],
			zindices_index: [0; VMF_COUNT],
			hierarchy: [TreeNode::default(); VMF_COUNT],
			hierarchy_index: [0; VMF_COUNT],
			count: 0,
			tree_node_count: 0,
			_padding: [0; 2],
			grid_layout: GridLayout::new(UVec2::ZERO, 0.0, 0),
		}
	}
}

#[cfg(not(target_arch = "spirv"))]
pub trait GpuTreeBuilder {
	fn tree_from_mixture(&self, mixture: &Mixture, pcg: &mut Pcg32) -> (Vec<u32>, Vec<TreeNode>);
}

pub trait SliceSelectionMethod {
	/// Selects a slice of the mixture model.
	fn select_slice(mixture: &Mixture, dir: Vec3, size: usize) -> (usize, usize);
}

pub struct AllSelection;
impl SliceSelectionMethod for AllSelection {
	fn select_slice(mixture: &Mixture, _dir: Vec3, _size: usize) -> (usize, usize) {
		(0, mixture.count as usize)
	}
}

pub struct MortonSliceSelection;
impl SliceSelectionMethod for MortonSliceSelection {
	fn select_slice(mixture: &Mixture, dir: Vec3, size: usize) -> (usize, usize) {
		let pnt = dir2pnt(dir);
		let zindex = pnt2morton(pnt);

		// offsets need to be computed in such a way that the slice width equals the requested size
		// for even sizes, this means up and down offsets are equal
		// for odd sizes, the up offset is one larger than the down offset
		let size = size as isize;
		let down = size / 2;
		let up = size - down;

		let mid = binary_search_closest(&mixture.zindices, &zindex, mixture.count as usize);
		let mid = mid as isize;

		let start = max(0, mid - down) as usize;
		let end = min(mixture.count as isize, mid + up) as usize;

		// slice must be size wide, unless we are at the start or end of the array
		if start != 0 && end != mixture.count as usize {
			assert!(end - start == size as usize);
		}
		assert!(end - start <= size as usize);

		(start, end)
	}
}

pub trait SliceExpansion {
	/// This function expands a slice to the left and right if the slice is smaller than the requested size.
	///
	/// It uses the given treshold to determine the expansion cutoff.
	/// If the pdf of the next component is below the treshold, expansion is stopped.
	///
	/// # Arguments
	/// * `mixture` - The mixture model.
	/// * `slice` - The current start and end index of the slice.
	/// * `size` - The requested size of the slice.
	/// * `treshold_pdf` - The treshold for the pdf to stop expanding.
	///
	/// # Returns
	/// A tuple containing the new start and end index of the slice, which is guaranteed to not exceed the requested size
	/// but may be smaller.
	fn expand_slice(&self, slice: (usize, usize), size: usize, treshold_pdf: f32, dir: Vec3) -> (usize, usize);
}

impl SliceExpansion for Mixture {
	#[allow(clippy::manual_range_contains)]
	fn expand_slice(&self, (start, end): (usize, usize), size: usize, treshold_pdf: f32, dir: Vec3) -> (usize, usize) {
		assert!(0.0 <= treshold_pdf && treshold_pdf <= 1.0);

		// while we expand the slice, we track the left and right component and always expand the one with the higher pdf
		let (mut start, mut end) = (start, end);

		while (end - start) < size {
			// pdf of next component to the left and right (will be -1.0 if out of bounds and therefore fail threshold)
			let left_pdf = Self::previous_pdf(self, start, dir);
			let right_pdf = Self::next_pdf(self, end - 1, dir);

			if left_pdf < right_pdf {
				// expand right

				if right_pdf < treshold_pdf {
					// stop expanding
					break;
				}
				end += 1;
			} else {
				// expand left

				if left_pdf < treshold_pdf {
					// stop expanding
					break;
				}
				start -= 1;
			}
		}

		(start, end)
	}
}

fn child_pdf(mixture: &Mixture, idx: usize, node: &TreeNode, dir: Vec3) -> (f32, f32) {
	// this function is only called for inner nodes
	assert!(!node.is_leaf());

	let left_pdf = {
		let left = &mixture.hierarchy[idx + 1];
		let (weight, vmf) = left.get_vmf();
		weight * vmf.pdf(dir)
	};

	let right_pdf = {
		let right = &mixture.hierarchy[node.get_right_child() as usize];
		let (weight, vmf) = right.get_vmf();
		weight * vmf.pdf(dir)
	};

	(left_pdf, right_pdf)
}

pub trait TreeTraversal {
	/// Traverses the tree and returns the start and end index of a slice.
	///
	/// # Arguments
	/// * `mixture` - The mixture model.
	/// * `dir` - The direction of the sample to base the traversal on.
	/// * `pcg` - Random number generator for non-deterministic traversal.
	fn traverse(mixture: &Mixture, dir: Vec3, pcg: &mut Pcg32, slice_size: usize) -> (usize, usize, usize);
}

/// This traversal method selects a random child node at each level regardless of the distribution or weight.
pub struct RandomTraversal;
impl TreeTraversal for RandomTraversal {
	fn traverse(mixture: &Mixture, _dir: Vec3, pcg: &mut Pcg32, slice_size: usize) -> (usize, usize, usize) {
		// current node (we start at the root)
		let mut idx = 0;
		let mut current = &mixture.hierarchy[idx];

		let mut nodes_visited = 0;

		while !current.is_leaf() {
			nodes_visited += 1;
			if pcg.gen_f32() < 0.5 {
				// select left child
				idx += 1;
			} else {
				// select right child
				idx = current.get_right_child() as usize;
			}

			current = &mixture.hierarchy[idx];
		}

		let (start, end) = current.get_slice();
		(start, end, nodes_visited)
	}
}

/// This traversal method selects the child node with the highest pdf in the direction of the sample.
pub struct TowardsPdfTraversal;
impl TreeTraversal for TowardsPdfTraversal {
	fn traverse(mixture: &Mixture, dir: Vec3, _pcg: &mut Pcg32, slice_size: usize) -> (usize, usize, usize) {
		let mut idx = 0;
		let mut current = &mixture.hierarchy[idx];

		let mut nodes_visited = 0;

		while !current.is_leaf() {
			nodes_visited += 1;

			// abort early if remaining tree slice is smaller than wanted slice (sample whole node)
			let (start, end) = current.get_slice();
			if end - start <= slice_size {
				break;
			}

			let (left_pdf, right_pdf) = child_pdf(mixture, idx, current, dir);

			if left_pdf > right_pdf {
				idx += 1;
			} else {
				idx = current.get_right_child() as usize;
			}

			current = &mixture.hierarchy[idx];
		}

		let (start, end) = current.get_slice();
		(start, end, nodes_visited)
	}
}

/// This traversal method chooses a random child node based on the relative probability for creating the given sample.
pub struct WeightedRandomTraversal;
impl TreeTraversal for WeightedRandomTraversal {
	fn traverse(mixture: &Mixture, dir: Vec3, pcg: &mut Pcg32, slice_size: usize) -> (usize, usize, usize) {
		let mut idx = 0;
		let mut current = &mixture.hierarchy[idx];

		let mut nodes_visited = 0;

		while !current.is_leaf() {
			nodes_visited += 1;
			let (left_pdf, right_pdf) = child_pdf(mixture, idx, current, dir);
			let total = left_pdf + right_pdf;
			let left_prob = left_pdf / total;

			if pcg.gen_f32() < left_prob {
				idx += 1;
			} else {
				idx = current.get_right_child() as usize;
			}

			current = &mixture.hierarchy[idx];
		}

		let (start, end) = current.get_slice();
		(start, end, nodes_visited)
	}
}

pub trait TreeSliceSampler {
	fn sample_slice(
		mixture: &Mixture,
		dir: Vec3,
		pcg: &mut Pcg32,
		origin_idx: usize,
		start: usize,
		end: usize,
	) -> (Vec3, f32, bool);
}

/// This slice sampler samples from all components in the slice exactly once.
pub struct FullSliceSampler;
impl TreeSliceSampler for FullSliceSampler {
	fn sample_slice(
		mixture: &Mixture,
		dir: Vec3,
		_pcg: &mut Pcg32,
		origin_idx: usize,
		start: usize,
		end: usize,
	) -> (Vec3, f32, bool) {
		let mut found = false;
		let mut pdf = 0.0;

		for i in start..end {
			let (index, vmf, weight) = mixture.get_tree_slice_component(i);
			pdf += vmf.pdf(dir) * weight;

			if index == origin_idx {
				found = true;
			}
		}

		(dir, pdf, found)
	}
}

trait SliceExpansionInternal {
	fn previous_pdf(&self, idx: usize, dir: Vec3) -> f32;
	fn next_pdf(&self, idx: usize, dir: Vec3) -> f32;
}

impl SliceExpansionInternal for Mixture {
	fn previous_pdf(&self, idx: usize, dir: Vec3) -> f32 {
		if idx > 0 {
			let (_, vmf, weight) = self.get_morton_component(idx - 1);
			vmf.pdf(dir) * weight // TODO: probably dont want to consider weight here?
		} else {
			-1.0
		}
	}

	fn next_pdf(&self, idx: usize, dir: Vec3) -> f32 {
		if idx < self.size() - 1 {
			let (_, vmf, weight) = self.get_morton_component(idx + 1);
			vmf.pdf(dir) * weight // TODO: probably dont want to consider weight here?
		} else {
			-1.0
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	const TEST_ARRAY_SORTED_FLOAT: [f32; 4] = [0.1, 0.3, 0.6, 1.0];

	fn dummy_hierarchy_builder() -> impl GpuTreeBuilder {
		struct DummyBuilder;
		impl GpuTreeBuilder for DummyBuilder {
			fn tree_from_mixture(&self, mixture: &Mixture, _pcg: &mut Pcg32) -> (Vec<u32>, Vec<TreeNode>) {
				(
					mixture.hierarchy_index[..mixture.size()].to_vec(),
					mixture.hierarchy[..mixture.size()].to_vec(),
				)
			}
		}

		DummyBuilder
	}

	#[test]
	fn test_binary_search_discrete_needle_max() {
		let array = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0];
		let active_len = 10;

		let idx = binary_search_closest(&array, &60, active_len);

		// we expect the last element to be the closest
		assert_eq!(idx, active_len - 1);
	}

	#[test]
	fn test_binary_search_discrete_needle_next_to_max() {
		let array = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 15, 0, 0, 0, 0, 0];
		let active_len = 10;

		let idx = binary_search_closest(&array, &10, active_len);

		// we expect the last element to be the closest
		assert_eq!(idx, active_len - 2);
	}

	#[test]
	fn test_binary_search_greater_than() {
		let array = TEST_ARRAY_SORTED_FLOAT;

		assert_eq!(binary_search_greater_than(&array, &0.00, array.len()), 0);
		assert_eq!(binary_search_greater_than(&array, &0.09, array.len()), 0);

		assert_eq!(binary_search_greater_than(&array, &0.11, array.len()), 1);
		assert_eq!(binary_search_greater_than(&array, &0.29, array.len()), 1);

		assert_eq!(binary_search_greater_than(&array, &0.31, array.len()), 2);
		assert_eq!(binary_search_greater_than(&array, &0.49, array.len()), 2);
		assert_eq!(binary_search_greater_than(&array, &0.51, array.len()), 2);

		assert_eq!(binary_search_greater_than(&array, &0.69, array.len()), 3);
		assert_eq!(binary_search_greater_than(&array, &0.71, array.len()), 3);
		assert_eq!(binary_search_greater_than(&array, &0.89, array.len()), 3);
		assert_eq!(binary_search_greater_than(&array, &0.91, array.len()), 3);
		assert_eq!(binary_search_greater_than(&array, &0.99, array.len()), 3);

		assert_eq!(binary_search_greater_than(&array, &1.01, array.len()), 3);

		let array = [0, 5, 15, 30, 60];
		assert_eq!(binary_search_greater_than(&array, &2, array.len()), 1);
	}

	#[test]
	fn test_binary_search_greater_than_single() {
		let array = [1.0];

		assert_eq!(binary_search_greater_than(&array, &0.1, array.len()), 0);
		assert_eq!(binary_search_greater_than(&array, &0.99, array.len()), 0);
		assert_eq!(binary_search_greater_than(&array, &1.00, array.len()), 0);
	}

	#[test]
	fn test_binary_search_closest() {
		let array = [0, 5, 15, 30, 60];

		assert_eq!(binary_search_closest(&array, &0, array.len()), 0);
		assert_eq!(binary_search_closest(&array, &2, array.len()), 0);

		assert_eq!(binary_search_closest(&array, &3, array.len()), 1);
		assert_eq!(binary_search_closest(&array, &5, array.len()), 1);
		assert_eq!(binary_search_closest(&array, &7, array.len()), 1);

		assert_eq!(binary_search_closest(&array, &10, array.len()), 2);
		assert_eq!(binary_search_closest(&array, &12, array.len()), 2);
		assert_eq!(binary_search_closest(&array, &15, array.len()), 2);
		assert_eq!(binary_search_closest(&array, &20, array.len()), 2);

		assert_eq!(binary_search_closest(&array, &25, array.len()), 3);
		assert_eq!(binary_search_closest(&array, &30, array.len()), 3);
		assert_eq!(binary_search_closest(&array, &35, array.len()), 3);
		assert_eq!(binary_search_closest(&array, &40, array.len()), 3);

		assert_eq!(binary_search_closest(&array, &50, array.len()), 4);
		assert_eq!(binary_search_closest(&array, &60, array.len()), 4);
		assert_eq!(binary_search_closest(&array, &70, array.len()), 4);
	}

	fn test_mixtures() -> Mixture {
		let mut mixture = Mixture::default();

		let dirs = [
			Vec3::new(1.0, 0.0, 0.0),
			Vec3::new(0.0, 1.0, 0.0),
			Vec3::new(0.0, 0.0, 1.0),
			Vec3::new(1.0, 1.0, 0.0),
			Vec3::new(0.0, 1.0, 1.0),
			Vec3::new(1.0, 0.0, 1.0),
		];

		for dir in dirs.iter() {
			mixture.add_component(VMFDistribution::new(*dir, 1.0), 1.0 / dirs.len() as f32);
		}

		mixture.finalize(&dummy_hierarchy_builder(), &GridLayout::new(UVec2::ZERO, 0.0, 8));

		mixture
	}

	#[test]
	fn test_morton_slice_width() {
		let mixture = test_mixtures();

		// walk from 0 to 1 in 0.05 steps along both x and y-axis
		// expect slice width to be 1 for all points
		for x in (0..20).map(|x| x as f32 * 0.05) {
			for y in (0..20).map(|y| y as f32 * 0.05) {
				let (start, end) = MortonSliceSelection::select_slice(&mixture, Vec3::new(x, y, 0.0), 1);
				assert_eq!(end - start, 1);
			}
		}
	}
}
