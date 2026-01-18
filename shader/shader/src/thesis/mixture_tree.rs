use bytemuck::{
	Pod,
	Zeroable,
};
use glam::Vec3;
use spirv_std::num_traits::Float;

use crate::thesis::vmf_distribution::VMFDistribution;

#[derive(Copy, Clone, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct TreeNode {
	/// VMF distribution that represents the node.
	vmf: VMFDistribution,

	/// Weight of the node as sum of all weights of its children.
	weight: f32,

	/// Scaling for raw vmf for upper bound calculation.
	upper_bound_scaling: f32,

	/// Index of parent node in the tree. -1 if root.
	parent: i32,

	// left_child: u32, -- implicit +1
	right_child: i32,

	/// Start index in the vmf array (or +1 if inner node).
	slice_start: i32,

	/// End index in the vmf array (or -1 if inner node).
	slice_end: i32,

	_padding: [u32; 2],
}

impl Default for TreeNode {
	fn default() -> Self {
		TreeNode {
			parent: -1,
			vmf: VMFDistribution::new(Vec3::ZERO, -1.0),
			weight: f32::nan(),
			upper_bound_scaling: f32::nan(),
			right_child: -1,
			slice_start: -1,
			slice_end: -1,
			_padding: [0; 2],
		}
	}
}

impl TreeNode {
	pub fn new_raw_node(parent: i32, vmf: VMFDistribution, weight: f32) -> Self {
		TreeNode {
			parent,
			vmf,
			weight,
			upper_bound_scaling: f32::nan(),
			right_child: -1,
			slice_start: -1,
			slice_end: -1,
			_padding: [0; 2],
		}
	}

	pub fn new_leaf(
		parent: i32,
		upper_bound_scaling: f32,
		vmf: VMFDistribution,
		weight: f32,
		slice_start: i32,
		slice_end: i32,
	) -> Self {
		TreeNode {
			parent,
			vmf,
			weight,
			upper_bound_scaling,
			// leafs don't have children
			right_child: -1,
			slice_start,
			slice_end,
			_padding: [0; 2],
		}
	}

	pub fn finalize_node(&mut self, right_child: i32, slice_start: i32, slice_end: i32, upper_bound_scaling: f32) {
		// can only finalize raw nodes
		assert_eq!(self.slice_start, -1);
		assert_eq!(self.slice_end, -1);

		// inputs must not be negative
		assert!(right_child >= 0);
		assert!(slice_start >= 0);
		assert!(slice_end >= 0);
		assert!(upper_bound_scaling >= 0.0);

		self.right_child = right_child;
		self.slice_start = slice_start;
		self.slice_end = slice_end;
		self.upper_bound_scaling = upper_bound_scaling;
	}

	pub fn get_parent(&self) -> i32 {
		self.parent
	}

	pub fn is_leaf(&self) -> bool {
		self.right_child < 0
	}

	pub fn get_vmf(&self) -> (f32, &VMFDistribution) {
		(self.weight, &self.vmf)
	}

	pub fn get_right_child(&self) -> i32 {
		if self.right_child < 0 {
			panic!("Node is a leaf and has no children");
		}

		self.right_child
	}

	pub fn get_slice(&self) -> (usize, usize) {
		(self.slice_start as usize, self.slice_end as usize)
	}

	pub fn upper_bound(&self, dir: Vec3) -> f32 {
		self.vmf.pdf(dir) * self.upper_bound_scaling
	}

	pub fn get_upper_bound_scaling(&self) -> f32 {
		self.upper_bound_scaling
	}
}
