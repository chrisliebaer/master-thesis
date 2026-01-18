use core::borrow::Borrow;

use glam::{
	Vec2,
	Vec3,
};
use pcg32::Pcg32;
use rayon::{
	iter::{
		IndexedParallelIterator,
		ParallelIterator,
	},
	prelude::{
		IntoParallelIterator,
		IntoParallelRefIterator,
	},
};
use shader::thesis::{
	entrypoints::{
		validate_scalar,
		validate_vector3,
	},
	mixture::{
		GpuTreeBuilder,
		Mixture,
	},
	mixture_tree::TreeNode,
	pcg32_ext::Pcg32Ext,
	vmf_distribution::VMFDistribution,
};

/// This struct is used to store the sufficient statistics of a von Mises-Fisher distribution.
struct SufficentStatistics {
	dir_weights: Vec3,
	sum_weights: f32,
}

impl SufficentStatistics {
	/// Creates a new set of sufficient statistics.
	fn from_vmf(weight: f32, vmf: &VMFDistribution) -> Self {
		let kappa = vmf.kappa() as f64;

		let cosh = kappa.cosh();
		let sinh = kappa.sinh();
		let r_bottom = cosh / sinh - 1.0 / kappa;
		let r_bottom = r_bottom as f32;
		validate_scalar(r_bottom);
		let dir_weights = weight * r_bottom * vmf.mean();
		validate_vector3(dir_weights);

		let sum_weights = dir_weights.length() / r_bottom;

		SufficentStatistics {
			dir_weights,
			sum_weights,
		}
	}
}

/// Implement addition on two sufficient statistics.
impl std::ops::Add for SufficentStatistics {
	type Output = Self;

	fn add(self, other: Self) -> Self {
		SufficentStatistics {
			dir_weights: self.dir_weights + other.dir_weights,
			sum_weights: self.sum_weights + other.sum_weights,
		}
	}
}

impl std::ops::AddAssign for SufficentStatistics {
	fn add_assign(&mut self, other: Self) {
		self.dir_weights += other.dir_weights;
		self.sum_weights += other.sum_weights;
	}
}

trait VMFExt {
	fn combine(first: (f32, &Self), second: (f32, &Self)) -> (f32, Self);
	fn from_sufficient_statistics(stats: SufficentStatistics) -> Self;
}

impl VMFExt for VMFDistribution {
	fn combine((w1, vmf1): (f32, &Self), (w2, vmf2): (f32, &Self)) -> (f32, Self) {
		let stats1 = SufficentStatistics::from_vmf(w1, vmf1);
		let stats2 = SufficentStatistics::from_vmf(w2, vmf2);
		let combined_stats = stats1 + stats2;

		(w1 + w2, Self::from_sufficient_statistics(combined_stats))
	}

	fn from_sufficient_statistics(stats: SufficentStatistics) -> Self {
		let r = stats.dir_weights;
		validate_vector3(r);
		let rl = r.length();
		validate_scalar(rl);
		if rl == 0.0 {
			panic!("rl == 0");
		}
		let r_bottom = rl / stats.sum_weights;

		let mean = r / rl;
		let k = (3.0 * r_bottom - r_bottom * r_bottom * r_bottom) / (1.0 - r_bottom * r_bottom);

		Self::new(mean, k)
	}
}

enum VMFNode {
	/// A leaf node that contains a single von Mises-Fisher distribution.
	Leaf(f32, usize, VMFDistribution),

	/// A fat leaf node that contains multiple von Mises-Fisher distributions in a single node.
	FatLeaf(f32, Vec<usize>, VMFDistribution),

	/// A branch node that contains two child nodes.
	Branch(f32, VMFDistribution, Box<VMFNode>, Box<VMFNode>),
}

impl VMFNode {
	/// Extracts the VMF that describes of this node.
	fn vmf(&self) -> &VMFDistribution {
		match self {
			VMFNode::Leaf(_, _, vmf) | VMFNode::FatLeaf(_, _, vmf) | VMFNode::Branch(_, vmf, ..) => vmf,
		}
	}

	/// Extracts the weight of this node.
	fn weight(&self) -> f32 {
		match self {
			VMFNode::Leaf(w, ..) | VMFNode::FatLeaf(w, ..) | VMFNode::Branch(w, ..) => *w,
		}
	}

	/// Extracts the VMF distributions of this part of the tree.
	fn visit_vmfs(&self, vmfs: &mut Vec<usize>) {
		match self {
			VMFNode::Leaf(_, i, _) => {
				vmfs.push(*i);
			},
			VMFNode::FatLeaf(_, indices, _) => {
				for i in indices {
					vmfs.push(*i);
				}
			},
			VMFNode::Branch(_, _, left, right) => {
				left.visit_vmfs(vmfs);
				right.visit_vmfs(vmfs);
			},
		}
	}
}

pub enum TreeBuildMethod {
	BottomUpRandom,
	TopDownRecursive,
}

pub struct VMFHierarchyBuilder {
	/// The method to use for building the tree.
	method: TreeBuildMethod,

	/// Optional for trimming leaf nodes.
	trim_size: Option<usize>,

	/// Number of samples (per lobe) to draw for upper bound estimation.
	bound_sample_count: usize,
}

impl GpuTreeBuilder for VMFHierarchyBuilder {
	fn tree_from_mixture(&self, mixture: &Mixture, pcg: &mut Pcg32) -> (Vec<u32>, Vec<TreeNode>) {
		let (indirection, tree_structure) = self.build(mixture, pcg);
		(indirection.into_iter().map(|i| i as u32).collect(), tree_structure)
	}
}

impl VMFHierarchyBuilder {
	pub fn new(method: TreeBuildMethod, trim_size: Option<usize>, bound_sample_count: usize) -> Self {
		VMFHierarchyBuilder {
			method,
			trim_size,
			bound_sample_count,
		}
	}

	pub fn build(&self, mixture: &Mixture, mut pcg: &mut Pcg32) -> (Vec<usize>, Vec<TreeNode>) {
		let vmfs: Vec<_> = mixture.iter().map(|(vmf, w)| (w, *vmf)).collect();
		let mut tree = match self.method {
			TreeBuildMethod::BottomUpRandom => self.random_build(&vmfs, pcg),
			TreeBuildMethod::TopDownRecursive => self.recursive_build(),
		};

		if let Some(trim_size) = self.trim_size {
			self.leaf_trim(&mut tree, trim_size);
		}

		let (indirection, tree_structure) = stacker::grow(32 * 1024 * 1024, || self.build_gpu_tree(&tree, mixture, pcg));

		// perform quick traversal in original tree to check that node count matches
		fn quick_count(node: &VMFNode) -> usize {
			match node {
				VMFNode::Branch(_, _, left, right) => quick_count(right) + quick_count(left) + 1,
				VMFNode::Leaf(..) | VMFNode::FatLeaf(..) => 1,
			}
		}
		let quick_count = quick_count(&tree);
		assert_eq!(quick_count, tree_structure.len());

		// perform quick traversal in linear list and check that these also match
		let mut current = 0;
		let linear_count = loop {
			let node = &tree_structure[current];
			if node.is_leaf() {
				break current + 1;
			}

			// check if next index would result in out of bounds access
			let new_current = node.get_right_child() as usize;
			assert!(new_current < tree_structure.len());

			current = new_current;
		};
		assert_eq!(quick_count, linear_count);

		(indirection, tree_structure)
	}

	/// Builds a hierarchy of von Mises-Fisher distributions from the given mixture.
	///
	/// The hierarchy is built by randomly combining existing nodes with their closest neighbor.
	fn random_build(&self, vmfs: &[(f32, VMFDistribution)], pcg: &mut Pcg32) -> VMFNode {
		// convert all VMFs to leaf nodes and store them in a vector from which we will draw randomly
		let mut nodes: Vec<VMFNode> = vmfs
			.iter()
			.enumerate()
			.map(|(i, (w, vmf))| VMFNode::Leaf(*w, i, *vmf))
			.collect();

		while nodes.len() > 1 {
			let first_idx = self.first_node_largest_kappa(&mut nodes, pcg);
			let first_node = nodes.remove(first_idx);

			// find the best neighbor to the first node by brute force
			let second_idx = (0..nodes.len())
				.into_par_iter()
				.map(|i| {
					let second_node = &nodes[i];
					let metric = Self::combine_by_angle(&first_node, second_node);
					(i, metric)
				})
				// combination metric requires the best candidate to have the largest metric
				.reduce_with(|(i1, metric1), (i2, metric2)| if metric1 > metric2 { (i1, metric1) } else { (i2, metric2) })
				.map(|(i, _)| i)
				.expect("No nodes left to combine.");
			let second_node = nodes.remove(second_idx);

			let (w, vmf) = VMFDistribution::combine(
				(first_node.weight(), first_node.vmf()),
				(second_node.weight(), second_node.vmf()),
			);

			let combined_node = VMFNode::Branch(w, vmf, Box::new(first_node), Box::new(second_node));
			nodes.push(combined_node);
		}
		assert_eq!(nodes.len(), 1);
		nodes.pop().expect("No root node found.")
	}

	/// Samples an upper bound for the given node by drawing samples from the lower vmfs and scaling the upper vmfs
	/// accordingly.
	///
	/// Scaling is done by scaling the pdf up the upper vmf to match the pdf of the lower vmfs.
	/// The upper bound is then the maximum of the largest scalar.
	fn sample_upper_bound<T: Borrow<VMFDistribution>>(&self, upper: &VMFDistribution, lowers: &[(T, f32)], pcg: &mut Pcg32) -> f32 {
		let mut max_scalar = 0.0;
		for lower in lowers {
			let (lower, weight) = (lower.0.borrow(), lower.1);
			for _ in 0..self.bound_sample_count {
				// fix samples to 50% quantile
				let rnd = Vec2::new(pcg.gen_f32(), 0.5);
				let dir = lower.sample(rnd);
				let pdf_lower = lower.pdf(dir) * weight;
				let pdf_upper = upper.pdf(dir);

				// find scalar such that: pdf_upper * scalar = pdf_lower
				let scalar = pdf_lower / pdf_upper;

				max_scalar = f32::max(max_scalar, scalar);
			}
		}

		max_scalar
	}

	fn first_node_random(&self, nodes: &mut [VMFNode], pcg: &mut Pcg32) -> usize {
		pcg.gen_max(nodes.len() as f32) as usize
	}

	fn first_node_largest_kappa(&self, nodes: &mut [VMFNode], _pcg: &mut Pcg32) -> usize {
		nodes
			.par_iter()
			.enumerate()
			.map(|(i, node)| (i, node.vmf().kappa()))
			.max_by(|(_, kappa1), (_, kappa2)| kappa1.partial_cmp(kappa2).unwrap())
			.map(|(i, _)| i)
			.expect("No nodes left to combine.")
	}

	/// Metric that combines two nodes based on the angle between their means.
	fn combine_by_angle(first_node: &VMFNode, second_node: &VMFNode) -> f32 {
		first_node.vmf().mean().dot(second_node.vmf().mean())
	}

	/// Metric that combines two nodes based on the resulting kappa of the combined distribution.
	fn combine_by_kappa(first_node: &VMFNode, second_node: &VMFNode) -> f32 {
		// there is an edge cases where the means are exactly opposite, in this case the resulting kappa is 0 and the sufficient
		// statistics are invalid
		if first_node.vmf().mean().dot(-second_node.vmf().mean()) >= 1.0 - 1e-6 {
			return 0.0;
		}

		let (weight, vmf) = VMFDistribution::combine(
			(first_node.weight(), first_node.vmf()),
			(second_node.weight(), second_node.vmf()),
		);
		weight * vmf.kappa()
	}

	fn recursive_build(&self) -> ! {
		unimplemented!("VMFHierarchyBuilder::recursive_build")
	}

	/// Combines lower leaf nodes into a single fat leaf node to reduce the depth of the tree.
	fn leaf_trim(&self, root: &mut VMFNode, nodes_per_leaf: usize) {
		// recursively traverse the tree and combine leaf nodes into fat leaf nodes
		fn leaf_trim_recursive(node: &mut VMFNode, nodes_per_leaf: usize) -> usize {
			// first delegate to children (to make sure we are at the bottom of the tree)
			let vmf_count = match node {
				VMFNode::Branch(_, _, left, right) => {
					leaf_trim_recursive(left, nodes_per_leaf) + leaf_trim_recursive(right, nodes_per_leaf)
				},
				VMFNode::Leaf(..) => return 1,
				VMFNode::FatLeaf(_, vmfs, ..) => return vmfs.len(),
			};

			// if number of childres does not exceed the limit, combine this node into a fat leaf
			if vmf_count <= nodes_per_leaf {
				let mut vmfs = Vec::with_capacity(vmf_count);
				node.visit_vmfs(&mut vmfs);

				// rewrite this node as a fat leaf node (only the vector needs to be added, rest is already representing collection)
				*node = VMFNode::FatLeaf(node.weight(), vmfs, *node.vmf());
			}

			vmf_count
		}

		leaf_trim_recursive(root, nodes_per_leaf);
	}

	fn build_gpu_tree(&self, root: &VMFNode, mixture: &Mixture, pcg: &mut Pcg32) -> (Vec<usize>, Vec<TreeNode>) {
		// two dfs with two mut vectors: first contains tree structure, second indirection to vmfs
		let mut indirection: Vec<usize> = Vec::new();
		let mut tree_structure: Vec<TreeNode> = Vec::new();

		// recursive function to traverse the tree and build the GPU tree
		self.traverse(root, -1, &mut indirection, &mut tree_structure, mixture, pcg);

		(indirection, tree_structure)
	}

	fn traverse(
		&self,
		node: &VMFNode,
		parent_index: i32,
		indirection: &mut Vec<usize>,
		tree_structure: &mut Vec<TreeNode>,
		mixture: &Mixture,
		pcg: &mut Pcg32,
	) -> (usize, usize, usize) {
		match node {
			VMFNode::Branch(_, _, left_node, right_node) => {
				// insert raw node into tree structure and update it after all children have been processed
				let node_idx = tree_structure.len();

				tree_structure.push(TreeNode::new_raw_node(parent_index, *node.vmf(), node.weight()));

				let (left_idx, slice_start, _) = self.traverse(left_node, node_idx as i32, indirection, tree_structure, mixture, pcg);
				let (right_idx, _, slice_end) = self.traverse(right_node, node_idx as i32, indirection, tree_structure, mixture, pcg);

				// left id is implicitly +1
				assert_eq!(left_idx, node_idx + 1);

				// compute upper bound across all lower vmfs
				// TODO: check if it is okay to sample from the combined lobes of our two children
				assert!(slice_start < slice_end, "Slice start must be smaller than slice end");
				let lowers = indirection[slice_start..slice_end]
					.iter()
					.map(|i| mixture.get_component(*i))
					.collect::<Vec<_>>();
				let sample_upper_bound = self.sample_upper_bound(node.vmf(), &lowers, pcg);

				// finalize the node with the correct children and slice indices
				tree_structure[node_idx].finalize_node(right_idx as i32, slice_start as i32, slice_end as i32, sample_upper_bound);

				(node_idx, slice_start, slice_end)
			},
			VMFNode::FatLeaf(..) | VMFNode::Leaf(..) => {
				// insert all indices into the indirection vector
				let slice_start = indirection.len();
				node.visit_vmfs(indirection);
				let slice_end = indirection.len();

				// insert fat leaf node into tree structure
				let node_idx = tree_structure.len();

				let lowers = indirection[slice_start..slice_end]
					.iter()
					// we do not(!) use the get_tree_slice_component accessor, since the indirection array contains vmf indices without
					// indirection
					.map(|i| mixture.get_component(*i))
					.collect::<Vec<_>>();
				let sample_upper_bound = self.sample_upper_bound(node.vmf(), &lowers, pcg);
				tree_structure.push(TreeNode::new_leaf(
					parent_index,
					sample_upper_bound,
					*node.vmf(),
					node.weight(),
					slice_start as i32,
					slice_end as i32,
				));

				(node_idx, slice_start, slice_end)
			},
		}
	}

	pub fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
		let mut changed = false;

		// model trim_size as button
		let mut trim_size_enabled = self.trim_size.is_some();
		if ui.checkbox(&mut trim_size_enabled, "Enable Trim").changed() {
			if trim_size_enabled {
				self.trim_size = Some(8);
			} else {
				self.trim_size = None;
			}
			changed = true;
		}

		if let Some(mut trim_size) = self.trim_size {
			if ui.add(egui::Slider::new(&mut trim_size, 1..=100).text("Trim Size")).changed() {
				self.trim_size = Some(trim_size);
				changed = true;
			}
		}

		if ui
			.add(egui::Slider::new(&mut self.bound_sample_count, 1..=100).text("Bound Sample Count"))
			.changed()
		{
			changed = true;
		}

		changed
	}
}

#[cfg(test)]
mod tests {
	use std::collections::HashSet;

	use approx::assert_abs_diff_eq;
	use shader::thesis::entrypoints::sample_shader::VMFSelection;

	use super::*;
	use crate::thesis::{
		mixture_loader::MixtureLoader,
		util::simple_mixture,
	};

	fn default_builder() -> VMFHierarchyBuilder {
		VMFHierarchyBuilder::new(TreeBuildMethod::BottomUpRandom, Some(8), 8)
	}

	// perform depth-first traversal and validate weight sum of parent nodes
	fn validate_tree(node: &VMFNode) -> f32 {
		let w = match node {
			// leafs simply return their weight
			VMFNode::Leaf(w, ..) => *w,
			VMFNode::FatLeaf(w, ..) => *w,
			VMFNode::Branch(w, _, left, right) => {
				// collect weight from children
				let w1 = validate_tree(left);
				let w2 = validate_tree(right);

				// validate that the sum of the children is equal to the parent
				assert_abs_diff_eq!(w1 + w2, *w, epsilon = 1e-6);

				// return the parent weight to be more numerically stable
				*w
			},
		};

		// all weights must be positive and non-zero
		assert!(w > 0.0);
		w
	}

	fn run_with_multiple_mixture<F: FnMut(&Mixture)>(mut f: F) {
		let mut mixture_loader = MixtureLoader::new("../../../mixture_python/export/json".into());
		let mixtures = mixture_loader.get_all_for_testing();

		for (name, mixture) in mixtures.iter() {
			println!("Running mixture {}", name);
			f(mixture);
		}
	}

	#[test]
	fn test_combine_identity() {
		let mean = Vec3::new(1.3, 4.4, 3.2).normalize();
		let vmf = VMFDistribution::new(mean, 20.0);

		let (w_combined, vmf_combined) = VMFDistribution::combine((0.5, &vmf), (0.5, &vmf));

		assert_abs_diff_eq!(w_combined, 1.0, epsilon = 1e-6);
		assert_abs_diff_eq!(vmf.kappa(), vmf_combined.kappa(), epsilon = 1.0);

		let diff = vmf.mean() - vmf_combined.mean();
		assert_abs_diff_eq!(diff.length(), 0.0, epsilon = 1e-6);
	}

	#[test]
	fn test_vmf_ext() {
		run_with_multiple_mixture(|mixture| {
			for (vmf, w) in mixture.iter() {
				let stats = SufficentStatistics::from_vmf(w, vmf);
				let vmf2 = VMFDistribution::from_sufficient_statistics(stats);

				// allow small delta due to floating point inaccuracies
				let diff = (vmf.mean() - vmf2.mean()).length();
				assert_abs_diff_eq!(diff, 0.0, epsilon = 1e-6);
			}
		});
	}

	#[test]
	fn test_random_build() {
		run_with_multiple_mixture(|mixture| {
			let mut pcg = Pcg32::new(0, 0);

			let vmfs: Vec<_> = mixture.iter().map(|(vmf, w)| (w, *vmf)).collect();
			let builder = default_builder();
			let tree = builder.random_build(&vmfs, &mut pcg);
			validate_tree(&tree);
		});
	}

	#[test]
	fn test_leaf_trim() {
		run_with_multiple_mixture(|mixture| {
			let mut pcg = Pcg32::new(0, 0);

			let vmfs: Vec<_> = mixture.iter().map(|(vmf, w)| (w, *vmf)).collect();
			let builder = default_builder();
			let mut tree = builder.random_build(&vmfs, &mut pcg);
			builder.leaf_trim(&mut tree, 8);
			validate_tree(&tree);

			let mut children = Vec::new();
			tree.visit_vmfs(&mut children);
			assert_eq!(children.len(), mixture.size());

			// we expect each index to be present exactly once
			let mut unique = HashSet::new();
			for i in children {
				assert!(unique.insert(i));
			}
		});
	}

	#[test]
	fn test_gpu_structure_construct() {
		run_with_multiple_mixture(|mixture| {
			let mut pcg = Pcg32::new(0, 0);

			let vmfs: Vec<_> = mixture.iter().map(|(vmf, w)| (w, *vmf)).collect();
			let builder = default_builder();
			let mut tree = builder.random_build(&vmfs, &mut pcg);
			builder.leaf_trim(&mut tree, 8);
			validate_tree(&tree);

			let (indirection, tree_structure) = builder.build_gpu_tree(&tree, &mixture, &mut pcg);

			// validate that all indices are unique
			let mut unique = HashSet::new();
			for i in &indirection {
				assert!(unique.insert(*i));
			}

			// validate that all indices are in the correct range
			for i in &indirection {
				assert!(*i < mixture.size());
			}

			// travers gpu tree and validate that all indices are in the correct range and that all nodes are reachable
			let mut visited_nodes = vec![false; tree_structure.len()];
			let mut visited_vmfs = vec![false; mixture.size()];

			fn explore_subtree(
				parent: i32,
				node_idx: usize,
				tree: &[TreeNode],
				indirection: &[usize],
				visited_nodes: &mut [bool],
				visited_vmfs: &mut [bool],
			) {
				if visited_nodes[node_idx] {
					panic!("Node {} was visited twice", node_idx);
				}
				visited_nodes[node_idx] = true;

				let node = &tree[node_idx];

				// check that parent is correct
				if parent >= 0 {
					assert_eq!(node.get_parent(), parent);
				}

				// check that upper bound scalar is positive
				assert!(node.get_upper_bound_scaling() > 0.0);

				if node.is_leaf() {
					// visit all vmfs in the slice
					let (start, end) = node.get_slice();
					for i in indirection[start..end].iter() {
						if visited_vmfs[*i] {
							panic!("VMF {} was visited twice", i);
						}
						visited_vmfs[*i] = true;
					}
				} else {
					// node is branch, delegate to children and check afterward that all children vmfs were visited
					let left_idx = node_idx + 1;
					let right_idx = node.get_right_child() as usize;

					explore_subtree(node_idx as i32, left_idx, tree, indirection, visited_nodes, visited_vmfs);
					explore_subtree(node_idx as i32, right_idx, tree, indirection, visited_nodes, visited_vmfs);

					// check that lobe slice was visited
					let (start, end) = node.get_slice();
					for i in indirection[start..end].iter() {
						if !visited_vmfs[*i] {
							panic!("VMF {} was not visited", i);
						}
					}
				}
			}

			explore_subtree(-1, 0, &tree_structure, &indirection, &mut visited_nodes, &mut visited_vmfs);

			// check that all nodes were visited
			for (i, visited) in visited_nodes.iter().enumerate() {
				if !visited {
					panic!("Node {} was not visited", i);
				}
			}
		});
	}

	/// This test needs to be implemented in the CPU library since GPU code can't generate the hierarchy.
	#[test]
	#[allow(clippy::needless_range_loop)]
	pub fn test_stackless_traversal() {
		run_with_multiple_mixture(|mixture| {
			let dirs = vec![
				Vec3::new(1.0, 0.0, 0.0),
				Vec3::new(0.0, 1.0, 0.0),
				Vec3::new(0.0, 0.0, 1.0),
				Vec3::new(1.0, 1.0, 1.0),
				Vec3::new(1.0, 1.0, -1.0),
				Vec3::new(1.0, -1.0, 1.0),
				Vec3::new(1.0, -1.0, -1.0),
				Vec3::new(-1.0, 1.0, 1.0),
				Vec3::new(-1.0, 1.0, -1.0),
				Vec3::new(-1.0, -1.0, 1.0),
				Vec3::new(-1.0, -1.0, -1.0),
				Vec3::new(0.5, 0.0, 0.0),
				Vec3::new(1.0, 0.0, 0.0),
				Vec3::new(0.0, 0.5, 0.0),
				Vec3::new(0.0, 1.0, 0.0),
				Vec3::new(0.0, 0.0, 0.5),
				Vec3::new(0.0, 0.0, 1.0),
			];

			let traversed_count = {
				// lenght of node list is implicitly known by traversing the tree
				let mut idx = 0;
				loop {
					let node = mixture.get_hierarchy_node(idx);
					if node.is_leaf() {
						break idx + 1;
					}
					idx = node.get_right_child() as usize;
				}
			};

			assert_eq!(traversed_count, mixture.tree_node_count());

			for dir in dirs {
				let dir = dir.normalize();
				test_stackless_internal(&mixture, dir);
			}
		});
	}

	fn test_stackless_internal(mixture: &Mixture, dir: Vec3) {
		fn selection_to_sorted_pdf(selection: &[VMFSelection]) -> Vec<f32> {
			let mut pdfs = selection.iter().map(|VMFSelection(pdf, _)| *pdf).collect::<Vec<_>>();
			pdfs.sort_by(|a, b| b.partial_cmp(a).unwrap());
			pdfs
		}

		fn ensure_only_low_pdf_replaced(old_selection: &[VMFSelection], new_selection: &[VMFSelection]) {
			let old_pdfs = selection_to_sorted_pdf(old_selection);
			let new_pdfs = selection_to_sorted_pdf(new_selection);
			let smallest_new_pdf = *new_pdfs.last().unwrap();

			// selection may find multiple better pdfs, so we check that all removed pdfs were smaller than the smallest new
			for pdf in old_pdfs {
				if pdf < smallest_new_pdf {
					// pdf must not be in new selection
					assert!(!new_pdfs.contains(&pdf), "pdf {} was not replaced", pdf);
				}
			}
		}

		let mut visited = HashSet::new();
		let node_count = mixture.tree_node_count();

		// we leave a few sloots unused to check if they are not touched
		const TEST_SELECTION_SIZE: usize = 8;
		const TEST_SELECTION_SIZE_WITH_UNUSED: usize = 16;
		let selection = &mut [VMFSelection::default(); TEST_SELECTION_SIZE_WITH_UNUSED];
		let selection_size = TEST_SELECTION_SIZE;

		let (mut last_index, mut current_index, mut worst_bound) = (-1, 0, 0.0);
		while current_index != -1 {
			let action = if last_index < current_index {
				"descending"
			} else {
				"ascending"
			};
			println!(
				"{} (last: {}, current: {}, worst: {})",
				action, last_index, current_index, worst_bound
			);

			// last node can not be the same as the current node
			assert_ne!(last_index, current_index, "last index is the same as current index");

			// every node must only be visted once from an id that's smaller
			if last_index < current_index {
				assert!(!visited.contains(&current_index), "node was visited twice");
				visited.insert(current_index);
			} else {
				// we come from a child node, so we must have visited the parent
				assert!(visited.contains(&current_index), "parent was not visited");
			}

			// peek if next node is a leaf and if so, manipulate the selection
			if mixture.get_hierarchy_node(current_index as usize).is_leaf() {
				// we test proper replacement of all indices in the selection
				for i in 0..selection_size {
					let write_copy = &mut [VMFSelection(0.0, 0); TEST_SELECTION_SIZE_WITH_UNUSED];
					write_copy.copy_from_slice(selection);

					// initially the selection is all zeros, if that's the case, we need to fill the other slots with large values to force
					// update in the i-th slot
					for VMFSelection(pdf, _) in write_copy.iter_mut().take(selection_size) {
						if *pdf == 0.0 {
							*pdf = 1.0;
						}
					}

					// simulate missing values in the selection
					write_copy[i] = VMFSelection(0.0, 0);

					mixture.advance_knn_search(dir, write_copy, selection_size, last_index, current_index, 0.0);
					ensure_only_low_pdf_replaced(selection, write_copy.as_slice());
					assert!(write_copy[i].0 > 0.0, "pdf was not replaced");
				}
			}

			// copy selection and only pass copy to the traversal function
			let write_copy = &mut [VMFSelection(0.0, 0); TEST_SELECTION_SIZE_WITH_UNUSED];
			write_copy.copy_from_slice(selection);

			let (new_current_index, new_worst_bound) =
				mixture.advance_knn_search(dir, write_copy, selection_size, last_index, current_index, worst_bound);

			// if node was not a leaf, the selection should not have changed
			if !mixture.get_hierarchy_node(current_index as usize).is_leaf() {
				assert_eq!(write_copy, selection, "selection was changed from non-leaf node");
			}

			// if worst bound got changed, it should be larger and the smallest value in the selection should be replaced
			if new_worst_bound != worst_bound {
				assert!(new_worst_bound > worst_bound, "worst bound decreased");

				// find the smallest value in old select, which we expect to be replaced
				let smallest_idx = selection[0..selection_size]
					.iter()
					.enumerate()
					.min_by(|(_, VMFSelection(pdf, _)), (_, VMFSelection(pdf2, _))| pdf.partial_cmp(pdf2).unwrap())
					.unwrap()
					.0;

				assert!(
					write_copy[smallest_idx].0 > selection[smallest_idx].0,
					"smallest pdf was not replaced but worst_bound increased (did some other pdf get replaced wrongly?)"
				);

				ensure_only_low_pdf_replaced(selection, write_copy.as_slice());
			}

			// commit values
			last_index = current_index;
			current_index = new_current_index;
			worst_bound = new_worst_bound;
			selection.copy_from_slice(write_copy.as_slice());
		}

		// there are always enough nodes to fill selection, so check for full selection
		for i in 0..selection_size {
			let VMFSelection(pdf, _) = selection[i];
			assert!(pdf > 0.0, "pdf is not positive at index {}", i);
		}

		// the remaining slots should be untouched
		for i in selection_size..selection.len() {
			let VMFSelection(pdf, idx) = selection[i];
			assert_eq!(pdf, 0.0, "pdf is not zero at index {}", i);
			assert_eq!(idx, 0, "index is not zero at index {}", i);
		}

		println!("Number of visited nodes using proper traversal: {}", visited.len());

		// traversal with proper selection will not visit all nodes
		// so we perform a second pass where we fake an empty selection for each iteration
		let mut visited = HashSet::new();
		let (mut last_index, mut current_index) = (-1, 0);
		while current_index != -1 {
			visited.insert(current_index);

			let slice = &mut [VMFSelection(0.0, 0); 8];
			let (new_current_index, new_worst_bound) = mixture.advance_knn_search(dir, slice, 8, last_index, current_index, 0.0);

			let node = mixture.get_hierarchy_node(current_index as usize);
			if !node.is_leaf() {
				// if branch node, the worst bound should be zero
				assert_eq!(new_worst_bound, 0.0);

				// if knn wants to ascent from a branch, both children must have been visited
				if new_current_index == node.get_parent() {
					let left = current_index + 1;
					let right = node.get_right_child();
					assert!(visited.contains(&left), "left child was not visited");
					assert!(visited.contains(&right), "right child was not visited");
				}
			}

			// commit values
			last_index = current_index;
			current_index = new_current_index;
		}

		// all nodes must have been visited
		for i in 0..node_count {
			if !visited.contains(&(i as i32)) {
				let node = mixture.get_hierarchy_node(i);
				let parent = node.get_parent();
				panic!("Node {} was not visited (reachable from parent {})", i, parent);
			}
		}
		assert_eq!(visited.len(), node_count, "excess nodes were visited");
		println!("Number of visited nodes forcing full traversal: {}", visited.len());
	}
}
