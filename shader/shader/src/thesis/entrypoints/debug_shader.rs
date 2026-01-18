use bytemuck::{
	Pod,
	Zeroable,
};
use glam::{
	UVec2,
	UVec3,
	Vec2,
	Vec2Swizzles,
	Vec3Swizzles,
	Vec4,
};
use pcg32::Pcg32;
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::{
	spirv,
	Image,
};
use static_assertions::const_assert;

use crate::thesis::{
	env_map::env_map_get_pnt,
	math::pnt2dir,
	mixture::{
		AllSelection,
		Mixture,
		MortonSliceSelection,
		SliceExpansion,
		SliceSelectionMethod,
		TowardsPdfTraversal,
		TreeTraversal,
	},
	mixture_ext::AnalyticalMixtureStatistics,
	morton::deinterleave_morton,
	vmf_grid::GridLayout,
};

const_assert!(core::mem::size_of::<DebugPushConstants>() <= 128);
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct DebugPushConstants {
	/// Offset for thread invocation, used for tiled execution.
	pub offset: UVec2,

	pub env_map_resolution: UVec2,

	/// The position of the cursor in the environment map. Coordinates are in the range [0, 1).
	pub cursor: Vec2,

	/// Size of the slice of lobes that we are going to sample.
	pub slize_size: u32,

	/// The quantile that we are going to use to draw the lobes.
	pub quantile: f32,

	/// Slider in UI used to debug morton curve
	pub morton_debug: u32,

	/// The ground truth value that we are trying to estimate.
	pub ground_truth: f32,

	pub slice_expansion: u32,

	pub expansion_threshold: f32,
}

// noinspection RsAssertEqual
/// Tints the environment map based on the pdf of the mixture at each pixel.
#[spirv(compute(threads(4, 4)))]
pub fn tint_pdf(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &DebugPushConstants,
	#[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba32f, sampled = false),
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
) {
	// env_map isn't used, but rust compiler will remove it completely if we don't use it
	assert!(env_map[0] >= 0.0);
	assert!(grid[0] != grid[1]);

	let id = constants.offset + id.xy();
	let pnt = id.xy().as_vec2() / constants.env_map_resolution.as_vec2();

	let color = mixture.pdf_morton_slice(
		pnt2dir(pnt),
		slice_from_cursor(mixture, constants.cursor, constants.slize_size),
	);
	let color = Vec4::new(color, color, color, 1.0);

	// write the color to the image
	let pnt = pnt * constants.env_map_resolution.as_vec2();
	let pnt = pnt.as_uvec2();
	unsafe {
		image.write(pnt, color);
	}
}

// noinspection RsAssertEqual
/// Tints the environment map based on the pdf of the mixture at each pixel.
#[spirv(compute(threads(4, 4)))]
pub fn tint_tree_pdf(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &DebugPushConstants,
	#[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba32f, sampled = false),
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
) {
	// env_map isn't used, but rust compiler will remove it completely if we don't use it
	assert!(env_map[0] >= 0.0);
	assert!(grid[0] != grid[1]);

	let id = constants.offset + id.xy();
	let mut pcg = Pcg32::new(0, 0);
	let pnt = id.xy().as_vec2() / constants.env_map_resolution.as_vec2();

	let (start, end) = tree_from_cursor::<TowardsPdfTraversal>(mixture, constants.cursor, constants.slize_size, &mut pcg);

	// full mixture for comparison
	let morton = mixture.pdf_morton_slice(
		pnt2dir(pnt),
		slice_from_cursor(mixture, constants.cursor, constants.slize_size),
	);

	// can be tree slice or entire slice, both will work with indirection
	let tree = {
		let mut pdf = 0.0;
		for i in start..end {
			let (_, component, weight) = mixture.get_tree_slice_component(i);
			pdf += weight * component.pdf(pnt2dir(pnt));
		}
		pdf
	};

	let color = Vec4::new(tree, morton, 0.0, 1.0);

	// write the color to the image
	let pnt = pnt * constants.env_map_resolution.as_vec2();
	let pnt = pnt.as_uvec2();
	unsafe {
		image.write(pnt, color);
	}
}

// noinspection RsAssertEqual
/// Draws visual representation of the mixtures.
///
/// This shader needs to be called for each pixel in the environment map texture.
#[spirv(compute(threads(4, 4)))]
pub fn draw_mixtures(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &DebugPushConstants,
	#[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba32f, sampled = false),
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
) {
	// env_map isn't used, but rust compiler will remove it completely if we don't use it
	assert!(env_map[0] >= 0.0);
	assert!(grid[0] != grid[1]);

	let id = constants.offset + id.xy();
	let pnt = id.xy().as_vec2() / constants.env_map_resolution.as_vec2();
	let mut color = Vec4::new(0.0, 0.0, 0.0, 1.0);
	let dir = pnt2dir(pnt);

	let (start, end) = slice_from_cursor(mixture, constants.cursor, constants.slize_size);

	let (start_ext, end_ext) = mixture.expand_slice(
		(start, end),
		(constants.slize_size + constants.slice_expansion) as usize,
		constants.expansion_threshold,
		dir,
	);

	for i in start_ext..end_ext {
		let (_, component, _) = mixture.get_morton_component(i);

		// plot the component if it is within the quantile
		if component.is_within_quantile(dir, constants.quantile) {
			// color depends on if the component is in the expanded slice or the original slice
			if start <= i && i < end {
				// inner
				color += Vec4::new(0.5, 0.0, 0.0, 1.0);
			} else if start_ext <= i && i < end_ext {
				// expanded
				color += Vec4::new(0.0, 0.5, 0.0, 1.0);
			}
		}

		// plot means by distance (center marker)
		let dist = (component.mean() - dir).length();
		if dist < 0.01 {
			color += Vec4::new(0.0, 0.5, 0.0, 1.0);
		}
	}

	// draw cursor circle
	if !constants.cursor.is_nan() {
		let cursor = constants.cursor;
		let dist = (pnt - cursor).length();
		if dist < 0.02 {
			color += Vec4::new(0.0, 0.2, 0.2, 1.0);
		}
	}

	{
		let morton_debug = constants.morton_debug;
		let (x, y) = deinterleave_morton(morton_debug);
		// x and y are u16 and need to map to env map resolution
		let x = x as f32 / u16::MAX as f32;
		let y = y as f32 / u16::MAX as f32;
		let morton_pnt = Vec2::new(x, y);
		let dist = (pnt - morton_pnt).length();
		if dist < 0.02 {
			// color += Vec4::new(0.2, 0.0, 0.2, 1.0);
		}
	}

	// write the color to the image
	let pnt = pnt * constants.env_map_resolution.as_vec2();
	let pnt = pnt.as_uvec2();
	unsafe {
		image.write(pnt, color);
	}
}

// noinspection RsAssertEqual
#[spirv(compute(threads(4, 4)))]
pub fn draw_rmse(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &DebugPushConstants,
	#[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba32f, sampled = false),
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
) {
	assert!(grid[0] != grid[1]);

	let id = constants.offset + id.xy();
	let pnt = id.xy().as_vec2() / constants.env_map_resolution.as_vec2();
	let dir = pnt2dir(pnt);

	// color of all samples that we would be reading if we were to sample from current direction
	let color = env_map_get_pnt(env_map, constants.env_map_resolution, pnt);
	let (start, end) = MortonSliceSelection::select_slice(mixture, dir, constants.slize_size as usize);
	let (start_ext, end_ext) = mixture.expand_slice(
		(start, end),
		(constants.slize_size + constants.slice_expansion) as usize,
		constants.expansion_threshold,
		dir,
	);

	let slice_diff = (end_ext - start_ext) - (end - start);
	let expansion_fraction = slice_diff as f32 / constants.slice_expansion as f32;

	let stats = AnalyticalMixtureStatistics::for_morton(mixture, dir, color, constants.ground_truth, (start, end));
	let rmse = stats.rmse() * stats.density();

	// write the color to the image
	let pnt = pnt * constants.env_map_resolution.as_vec2();
	let pnt = pnt.as_uvec2();
	unsafe {
		image.write(pnt, Vec4::new(rmse, stats.miss_rate(), expansion_fraction, 1.0));
	}
}

// noinspection RsAssertEqual
#[spirv(compute(threads(4, 4)))]
pub fn draw_stats_hierarchy(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &DebugPushConstants,
	#[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba32f, sampled = false),
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
) {
	assert!(grid[0] != grid[1]);

	let id = constants.offset + id.xy();
	let pnt = id.xy().as_vec2() / constants.env_map_resolution.as_vec2();
	let dir = pnt2dir(pnt);
	let color = env_map_get_pnt(env_map, constants.env_map_resolution, pnt);

	let (start, end, nodes_visited) =
		TowardsPdfTraversal::traverse(mixture, dir, &mut Pcg32::new(0, 0), constants.slize_size as usize);
	let stats =
		AnalyticalMixtureStatistics::for_hierarchy(mixture, dir, color, constants.ground_truth, (start, end), nodes_visited);
	let rmse = stats.rmse() * stats.density();

	let pnt = pnt * constants.env_map_resolution.as_vec2();
	let pnt = pnt.as_uvec2();
	unsafe {
		image.write(pnt, Vec4::new(rmse, stats.miss_rate(), 0.0, 1.0));
	}
}

// noinspection RsAssertEqual
#[spirv(compute(threads(4, 4)))]
pub fn draw_stats_knn(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &DebugPushConstants,
	#[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba32f, sampled = false),
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
) {
	assert!(grid[0] != grid[1]);

	let id = constants.offset + id.xy();
	let pnt = id.xy().as_vec2() / constants.env_map_resolution.as_vec2();
	let dir = pnt2dir(pnt);
	let color = env_map_get_pnt(env_map, constants.env_map_resolution, pnt);

	let stats = AnalyticalMixtureStatistics::for_knn::<1024>(
		mixture,
		dir,
		color,
		constants.ground_truth,
		constants.slize_size as usize,
		false,
	);
	let rmse = stats.rmse() * stats.density();
	let visited_ratio = stats.visited_nodes() / mixture.tree_node_count() as f32;

	let pnt = pnt * constants.env_map_resolution.as_vec2();
	let pnt = pnt.as_uvec2();
	unsafe {
		image.write(pnt, Vec4::new(rmse, stats.miss_rate(), visited_ratio, 1.0));
	}
}

// noinspection RsAssertEqual
#[spirv(compute(threads(4, 4)))]
pub fn draw_stats_grid(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &DebugPushConstants,
	#[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba32f, sampled = false),
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
) {
	assert!(grid[0] != grid[1]);

	let id = constants.offset + id.xy();
	let pnt = id.xy().as_vec2() / constants.env_map_resolution.as_vec2();
	let dir = pnt2dir(pnt);
	let color = env_map_get_pnt(env_map, constants.env_map_resolution, pnt);

	let cell_index = mixture.grid_layout().pnt2cell(pnt);

	let stats = AnalyticalMixtureStatistics::for_grid(
		mixture,
		dir,
		cell_index,
		grid,
		color,
		constants.ground_truth,
		constants.slize_size as usize,
	);
	let rmse = stats.rmse() * stats.density();

	// we use the last channel to visualize the post processed color mapping and write values from 0 to 1 from left to right
	// add some padding left and right to show out of bounds values
	let x_tint = (pnt.x * 1.1) - 0.05;

	let pnt = pnt * constants.env_map_resolution.as_vec2();
	let pnt = pnt.as_uvec2();
	unsafe {
		image.write(pnt, Vec4::new(rmse, stats.miss_rate(), x_tint, 1.0));
	}
}

// noinspection RsAssertEqual
#[spirv(compute(threads(4, 4)))]
pub fn draw_pdf_diff(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &DebugPushConstants,
	#[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba32f, sampled = false),
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
) {
	// env_map isn't used, but rust compiler will remove it completely if we don't use it
	assert!(env_map[0] >= 0.0);
	assert!(grid[0] != grid[1]);

	let id = constants.offset + id.xy();
	let pnt = id.xy().as_vec2() / constants.env_map_resolution.as_vec2();
	let dir = pnt2dir(pnt);

	let (pdf_full, pdf_morton) = {
		let (start, end) = MortonSliceSelection::select_slice(mixture, dir, constants.slize_size as usize);
		mixture.pdf_full_and_morton(dir, (start, end))
	};
	let diff_morton = (pdf_full - pdf_morton).abs();

	let pdf_tree = {
		let (start, end, _) = TowardsPdfTraversal::traverse(mixture, dir, &mut Pcg32::new(0, 0), constants.slize_size as usize);
		let mut pdf_tree = 0.0;
		for i in start..end {
			let (_, component, weight) = mixture.get_tree_slice_component(i);
			pdf_tree += component.pdf(dir) * weight;
		}
		pdf_tree
	};
	let diff_tree = (pdf_full - pdf_tree).abs();

	// difference between morton and tree pdf
	let diff_both = (pdf_morton - pdf_tree).abs();

	// write the color to the image
	let pnt = pnt * constants.env_map_resolution.as_vec2();
	let pnt = pnt.as_uvec2();

	let mut output = Vec4::new(diff_morton, diff_tree, diff_both, 1.0);

	// weight differences by how often they are sampled
	// output *= pdf_full;
	output.z = 1.0;

	unsafe {
		image.write(pnt, output);
	}
}

// noinspection RsAssertEqual
#[spirv(compute(threads(4, 4)))]
pub fn draw_grid_debug(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &DebugPushConstants,
	#[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba32f, sampled = false),
	#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mixture: &Mixture,
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
	#[spirv(storage_buffer, descriptor_set = 0, binding = 3)] grid: &[u32],
) {
	// force usage of stuff
	assert!(env_map[0] >= 0.0);
	assert!(mixture.tree_node_count() > 0);
	assert!(grid[0] != grid[200]);

	let id = constants.offset + id.xy();
	let pnt = id.xy().as_vec2() / constants.env_map_resolution.as_vec2();

	let mut color = Vec4::new(0.0, 0.0, 0.0, 1.0);

	let grid_layout = mixture.grid_layout();

	let this_grid = grid_layout.pnt2cell(pnt);
	let next_grid = {
		let next_id = id.xy() + UVec2::new(1, 1);
		let next_pnt = next_id.as_vec2() / constants.env_map_resolution.as_vec2();
		grid_layout.pnt2cell(next_pnt)
	};

	// if moving to next pixel leads to a different cell, color the pixel red
	if this_grid != next_grid {
		color.x += 0.5;
	};

	// if cell center is close to the pixel, color the pixel green
	let cell_center = grid_layout.cell_center_pnt(this_grid);
	let dist = (cell_center - pnt).length();
	if dist < 0.002 {
		color.y += 0.5;
	};

	if !constants.cursor.is_nan() {
		let mouse_index = grid_layout.pnt2cell(constants.cursor);
		if mouse_index == this_grid {
			color.z += 0.5;
		}
	}

	// fetch pdf from cell and add it to all channels
	let dir = pnt2dir(pnt);
	let pdf = mixture.pdf_grid(dir, 0, this_grid, grid, constants.slize_size as usize).0 * 0.25;

	color += Vec4::new(pdf, pdf, pdf, 1.0);

	let pnt = pnt * constants.env_map_resolution.as_vec2();
	let pnt = pnt.as_uvec2();
	unsafe {
		image.write(pnt, color);
	}
}

fn slice_from_cursor(mixture: &Mixture, cursor: Vec2, size: u32) -> (usize, usize) {
	let dir = pnt2dir(cursor);
	let size = size as usize;
	if cursor.is_nan() {
		AllSelection::select_slice(mixture, dir, size)
	} else {
		MortonSliceSelection::select_slice(mixture, dir, size)
	}
}

fn tree_from_cursor<Traversal: TreeTraversal>(mixture: &Mixture, cursor: Vec2, size: u32, pcg: &mut Pcg32) -> (usize, usize) {
	let dir = pnt2dir(cursor);
	let size = size as usize;
	if cursor.is_nan() {
		AllSelection::select_slice(mixture, dir, size)
	} else {
		let (start, end, _) = Traversal::traverse(mixture, dir, pcg, size);
		(start, end)
	}
}
