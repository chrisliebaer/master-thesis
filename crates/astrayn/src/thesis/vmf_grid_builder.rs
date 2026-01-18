use rayon::{
	iter::ParallelIterator,
	prelude::IntoParallelIterator,
};
use shader::thesis::{
	math::pnt2dir,
	mixture::Mixture,
	vmf_grid::GridLayout,
};
use tracing::error;

pub struct VMFGridBuilder {
	grid: GridLayout,
}

impl VMFGridBuilder {
	pub fn new(grid: GridLayout) -> Self {
		VMFGridBuilder {
			grid,
		}
	}

	pub fn build(&self, mixture: &Mixture) -> Box<[u32]> {
		// iterate over all (valid) cells
		let y_rows = (0..self.grid.extend().y)
			.into_par_iter()
			.map(|y| {
				// note that some elements are already invalid and will not be written to
				let mut row = vec![0; self.grid.extend().x as usize * self.grid.bucket_size()];

				// iterate over all x values
				for x in 0..self.grid.extend().x {
					let cell_index = (y * self.grid.extend().x + x) as usize;

					// some indices are invalid due to collapsing cells
					if !self.grid.is_valid_index(cell_index) {
						continue;
					}

					let center = self.grid.cell_center_pnt(cell_index);
					let dir = pnt2dir(center);

					// sample all components of mixture and find the "bucket_size" best ones
					let mut pdfs = (0..mixture.size())
						.map(|i| {
							let (vmf, weight) = mixture.get_component(i);
							let pdf = vmf.pdf(dir) * weight;

							(i, pdf)
						})
						.collect::<Vec<(usize, f32)>>();

					// sort by pdf (descending)
					pdfs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

					// write the indices of the best "bucket_size" components to the row
					// note that cell_index is addressing a global array, but we are writing a row-local array
					let start_index = x as usize * self.grid.bucket_size();
					pdfs
						.iter()
						.take(self.grid.bucket_size())
						.enumerate()
						.for_each(|(i, (index, _))| {
							row[start_index + i] = *index as u32;
						});
				}

				row
			})
			.collect::<Vec<Vec<u32>>>();

		// collapse nested vectors into a single one
		let mut data = Vec::new();
		for row in y_rows {
			assert_eq!(row.len(), self.grid.extend().x as usize * self.grid.bucket_size());
			data.extend(row);
		}

		data.into_boxed_slice()
	}

	pub fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
		let mut changed = false;

		changed |= self.grid.draw_ui(ui);

		changed
	}

	pub fn grid_layout(&self) -> &GridLayout {
		&self.grid
	}
}

trait GridLayoutUi {
	fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool;
}
impl GridLayoutUi for GridLayout {
	fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
		let mut changed = false;
		let mut extend = self.extend();
		let mut min_distance = self.min_distance();

		ui.label("Grid Layout");
		ui.horizontal(|ui| {
			ui.label("Resolution");
			changed |= ui.add(egui::Slider::new(&mut extend.x, 1..=1000).text("X")).changed();
			changed |= ui.add(egui::Slider::new(&mut extend.y, 1..=1000).text("Y")).changed();
		});
		changed |= ui
			.add(egui::Slider::new(&mut min_distance, 0.00..=100.0).text("Min Distance"))
			.changed();

		ui.label("Bucket Size");
		let mut bucket_size = self.bucket_size() as u32;
		changed |= ui
			.add(egui::Slider::new(&mut bucket_size, 1..=100).text("Bucket Size"))
			.changed();

		if changed {
			self.set_bucket_size(bucket_size as usize);
			self.set_extend(extend);
			self.set_min_distance(min_distance);
		}

		changed
	}
}

#[cfg(test)]
mod tests {
	use glam::UVec2;

	use super::*;
	use crate::thesis::util::simple_mixture;

	#[test]
	fn test_build() {
		let grid = GridLayout::new(UVec2::new(100, 50), 3f32, 5);
		let builder = VMFGridBuilder::new(grid);

		let mixture = simple_mixture();
		builder.build(&mixture);
	}
}
