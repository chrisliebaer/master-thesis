use bytemuck::{
	Pod,
	Zeroable,
};
use glam::{
	UVec2,
	Vec2,
};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[derive(Copy, Clone, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct GridLayout {
	/// The number of cells in the x and y direction.
	extend: UVec2,

	/// The minimum distance between two cells. If the distance is smaller, the cells are collapsed.
	min_distance: f32,

	/// Size of individual buckets in the grid.
	bucket_size: u32,
}

impl GridLayout {
	pub fn new(extend: UVec2, min_distance: f32, bucket_size: u32) -> Self {
		GridLayout {
			extend,
			min_distance,
			bucket_size,
		}
	}

	pub fn extend(&self) -> UVec2 {
		self.extend
	}

	pub fn set_extend(&mut self, extend: UVec2) {
		self.extend = extend;
	}

	pub fn min_distance(&self) -> f32 {
		self.min_distance
	}

	pub fn set_min_distance(&mut self, min_distance: f32) {
		assert!(min_distance >= 0.0);
		self.min_distance = min_distance;
	}

	pub fn bucket_size(&self) -> usize {
		self.bucket_size as usize
	}

	pub fn set_bucket_size(&mut self, bucket_size: usize) {
		self.bucket_size = bucket_size as u32;
	}

	pub fn is_valid_index(&self, index: usize) -> bool {
		let y = index / self.extend.x as usize;
		let x = index % self.extend.x as usize;

		// check if x is within the bounds of the collapsed cells for the given y and y is within the bounds of the grid
		x < self.cells_per_latitude(y) && y < self.extend.y as usize
	}

	/// Convert a point in longitude/latitude space to a cell index.
	pub fn pnt2cell(&self, pnt: Vec2) -> usize {
		let y = (pnt.y * self.extend.y as f32) as usize;

		// readjust x to fit the circular arc
		let x = pnt.x * self.cells_per_latitude(y) as f32;

		// 2d array doesn't care about collapsed cells and keeps the original cell count with empty cells at the end
		y * self.extend.x as usize + x as usize
	}

	/// Convert a cell index to a point in longitude/latitude space.
	fn cell2pnt(&self, cell: usize) -> Vec2 {
		// extract y and x from the cell index
		let y = cell / self.extend.x as usize;
		let x = cell % self.extend.x as usize;

		// assert that original x is within the bounds of the collapsed cells for the given y
		assert!(x < self.cells_per_latitude(y));

		// x was adjusted to fit the circular arc, we need to reverse this
		let x = x as f32 * self.cell_width(y);

		Vec2::new(x, y as f32 / self.extend.y as f32)
	}

	pub fn cell_center_pnt(&self, cell: usize) -> Vec2 {
		let y = cell / self.extend.x as usize;
		let top_left = self.cell2pnt(cell);

		// cell center is half a cell down and half a cell right (collapsed cells are larger in the x direction)
		let cell_height = 1.0 / self.extend.y as f32;
		Vec2::new(top_left.x + self.cell_width(y) * 0.5, top_left.y + cell_height * 0.5)
	}

	fn cells_per_uniform_cells(&self, y: usize) -> usize {
		// shift down by half a cell to get the center of the cell and also avoid division by zero
		let frac = (y as f32 + 0.5) / self.extend.y as f32;
		let pi_frac = frac * core::f32::consts::PI;

		let circumference = f32::sin(pi_frac) * 2.0 * core::f32::consts::PI;

		// check how many cells fit into the circular arc
		let cpa = (self.min_distance / circumference) as usize;

		// there must be at least one cell
		cpa.max(1)
	}

	fn cells_per_latitude(&self, y: usize) -> usize {
		self.extend.x as usize / self.cells_per_uniform_cells(y)
	}

	fn cell_width(&self, y: usize) -> f32 {
		1.0 / self.cells_per_latitude(y) as f32
	}
}

#[cfg(test)]
mod tests {
	use approx::assert_abs_diff_eq;

	use super::*;

	#[test]
	fn test_bijective() {
		// min distance is 0.0, so the grid is not collapsed
		let grid = GridLayout::new(UVec2::new(100, 50), 0.0, 8);

		// iterate over exact coordinates of the grid where projection is bijective
		for i in 0..100 {
			for j in 0..50 {
				let input = Vec2::new(i as f32 / 100.0, j as f32 / 50.0);
				let index = grid.pnt2cell(input);
				let output = grid.cell2pnt(index);

				// mapping basically cuts off third decimal place, so we need to be a bit more lenient
				assert_abs_diff_eq!(input.x, output.x, epsilon = 1e-2);
				assert_abs_diff_eq!(input.y, output.y, epsilon = 1e-2);
			}
		}
	}

	#[test]
	fn test_cell_center_inside_cell() {
		let grid = GridLayout::new(UVec2::new(200, 200), 1.0, 8);

		// iterate over all valid indices, the cell center must be inside the cell
		for y in 0..200 {
			let x_size = grid.cells_per_latitude(y);
			for x in 0..x_size {
				let index = y * 200 + x;
				let center = grid.cell_center_pnt(index);
				let top_left = grid.cell2pnt(index);

				let index_center = grid.pnt2cell(center);

				// center must map back to the same cell
				assert_eq!(index, index_center);

				// center must be inside the cell
				assert!(center.x >= top_left.x);

				// very complicated way of expressing that the center is left of the right border (which may be an invalid cell, but the
				// math checks out anyway)
				assert!(center.x <= top_left.x + grid.cells_per_uniform_cells(y) as f32 / 200.0);

				// distance from along y between top left and center must be half a cell
				assert_abs_diff_eq!(center.y - top_left.y, 0.5 / 200.0, epsilon = 1e-6);
			}
		}
	}

	#[test]
	fn test_cells_per_latitude() {
		let grid = GridLayout::new(UVec2::new(100, 500), 1f32, 8);

		// the number of cells per latitude should be equal for mirrored longitudes
		for i in 0..500 {
			let frac = i as f32 / 1000f32;
			assert!(frac < 0.5);

			let cpa1_in = (frac * 500.0) as usize;
			let cpa1 = grid.cells_per_latitude(cpa1_in);
			let cpa2_in = (499.0 - frac * 500.0) as usize;
			let cpa2 = grid.cells_per_latitude(cpa2_in);

			// cpa2 is allowed to differ by one since the number of cells per latitude is rounded down
			if cpa1 != cpa2 {
				let cpa2 = grid.cells_per_latitude(cpa2_in + 1);
				println!("y: {}, cpa1: {}, cpa2: {} (y-1)", i, cpa1, cpa2);
				assert_eq!(cpa1, cpa2);
			} else {
				println!("y: {}, cpa1: {}, cpa2: {}", i, cpa1, cpa2);
				assert_eq!(cpa1, cpa2);
			}
		}
	}

	#[test]
	fn test_full_fill_check_empty() {
		const X_RES: usize = 500;
		const Y_RES: usize = 500;
		let grid = GridLayout::new(UVec2::new(X_RES as u32, Y_RES as u32), 1f32, 8);

		let mut grid_storage = vec![0; X_RES * Y_RES];

		// while we move towards the equator, the number of cells per latitude increases (and decreases when we move past it)
		let mut last_latitude_cell_count = 0;

		// iterate with higher resolution to provoke inter cell access
		for y in 0..Y_RES * 10 {
			let cells_per_latitude = grid.cells_per_latitude(y / 10);

			// before we move past the equator, the number of cells per latitude should increase
			if y < Y_RES * 5 {
				assert!(cells_per_latitude >= last_latitude_cell_count);
			} else {
				assert!(cells_per_latitude <= last_latitude_cell_count);
			}
			last_latitude_cell_count = cells_per_latitude;

			for x in 0..X_RES * 10 {
				let mut pnt = Vec2::new(x as f32 / (X_RES * 10) as f32, y as f32 / (Y_RES * 10) as f32);

				// add a small offset since floats might fall back to the previous cell due to rounding errors
				pnt.x += 1e-6;
				pnt.y += 1e-6;

				let cell = grid.pnt2cell(pnt);

				// write within y line must be within the cells_per_latitude count
				assert!(cell % X_RES < cells_per_latitude);

				grid_storage[cell] += 1;
			}

			// cells in the same latitude should only be filled up to the cells_per_latitude count
			let x_start = (y / 10) * X_RES;

			// all valid cells were written at least once
			grid_storage[x_start..x_start + cells_per_latitude]
				.iter()
				.for_each(|&x| assert!(x > 0));

			// while the remaining cells are empty
			grid_storage[x_start + cells_per_latitude..x_start + X_RES]
				.iter()
				.for_each(|&x| assert_eq!(x, 0));
		}
	}

	#[test]
	fn test_specific_center_not_middle() {
		let grid = GridLayout::new(UVec2::new(30, 50), 5.0, 8);

		// first row has only one cell and center must be exactly in the middle
		assert_eq!(grid.cells_per_latitude(0), 1);
		let center = grid.cell_center_pnt(0);
		assert_abs_diff_eq!(center.x, 0.5, epsilon = 1e-6);

		// second row (cell_idx: 2) has three cells and center must be exactly in the middle
		assert_eq!(grid.cells_per_latitude(1), 3);
		let center = grid.cell_center_pnt(30 + 1);
		assert_abs_diff_eq!(center.x, 0.5, epsilon = 1e-6);
	}
}
