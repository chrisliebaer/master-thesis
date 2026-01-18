use std::{
	collections::HashMap,
	path::PathBuf,
};

use egui::Context;
use glam::{
	UVec2,
	Vec2,
	Vec3,
};
use pcg32::Pcg32;
use rayon::{
	iter::{
		IndexedParallelIterator,
		ParallelIterator,
	},
	prelude::IntoParallelIterator,
};
use serde::{
	Deserialize,
	Serialize,
};
use shader::thesis::{
	env_map::env_map_get_pnt,
	math::{
		jacobian,
		pnt2dir,
	},
	mixture::{
		Mixture,
		MortonSliceSelection,
		SliceExpansion,
		SliceSelectionMethod,
		TowardsPdfTraversal,
		TreeTraversal,
	},
	mixture_ext::AnalyticalMixtureStatistics,
	vmf_grid::GridLayout,
};
use tracing::{
	error,
	info,
};

use crate::thesis::{
	load_exr,
	util::integrate_sphere,
	vmf_grid_builder::VMFGridBuilder,
};

/// Encapsulates the current environment map.
pub struct EnvironmentMap {
	/// The path to the current environment map.
	pub path: String,

	/// Ground truth integral of the environment map.
	pub ground_truth: f64,

	/// The dimensions of the environment map.
	pub dim: UVec2,

	/// The environment map as a 2D texture.
	pub data: Box<[f64]>,
}

impl EnvironmentMap {
	/// Loads the environment map from the given path.
	///
	/// # Arguments
	/// * `path` - The path to the EXR file.
	/// * `divide` - The factor by which to divide the resolution of the image.
	pub fn load(path: &str, divide: usize) -> Self {
		let image = load_exr(path, divide);
		let pixels = image.layer_data.channel_data.pixels;
		let width = pixels[0].len();
		let height = pixels.len();

		let mut data = vec![0.0; width * height].into_boxed_slice();
		for i in 0..width {
			for j in 0..height {
				let pixel = pixels[j][i];

				data[j * width + i] = pixel;
			}
		}

		let dim = UVec2::new(width as u32, height as u32);
		let ground_truth = integrate_sphere(&data, dim);

		Self {
			path: path.to_string(),
			ground_truth,
			dim,
			data,
		}
	}

	pub fn name(&self) -> String {
		// remove the path and the file extension
		let mut name = self.path.clone();

		// remove backslashes if we are on Windows
		name = name.replace('\\', "/");

		// cut off the path
		if let Some(index) = name.rfind('/') {
			name = name.split_at(index + 1).1.to_string();
		}

		// cut off the file extension
		if let Some(index) = name.rfind('.') {
			name = name.split_at(index).0.to_string();
		}

		name
	}

	pub fn empty() -> Self {
		// a black image is a valid environment map

		let size = 1024;

		Self {
			data: vec![0.0; size * size].into_boxed_slice(),
			ground_truth: 0.0,
			dim: UVec2::new(size as u32, size as u32),
			path: "None".to_string(),
		}
	}
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PartialMixtureStistics {
	pub variance: f64,
	pub density: f64,
	pub contribution: f64,
	pub miss_rate: f64,
	pub visited_nodes: f64,
}

impl PartialMixtureStistics {
	fn from_mixture_stats(stats: AnalyticalMixtureStatistics<f64>, jacobian: f64) -> Self {
		PartialMixtureStistics {
			variance: stats.mse() * stats.density() * jacobian,
			density: stats.density() * jacobian,
			contribution: stats.estimate() * stats.density() * jacobian,
			miss_rate: stats.miss_rate() * stats.density() * jacobian,
			visited_nodes: stats.visited_nodes() * stats.density() * jacobian,
		}
	}

	/// Scales the statistics by the given factor for normalization.
	fn scale(&mut self, factor: f64) {
		self.variance *= factor;
		self.density *= factor;
		self.contribution *= factor;
		self.miss_rate *= factor;
		self.visited_nodes *= factor;
	}

	fn reduce(first: Self, second: Self) -> Self {
		Self {
			variance: first.variance + second.variance,
			density: first.density + second.density,
			contribution: first.contribution + second.contribution,
			miss_rate: first.miss_rate + second.miss_rate,
			visited_nodes: first.visited_nodes + second.visited_nodes,
		}
	}
}

pub struct SamplingCalculatorContainer {
	stats: Vec<(String, PartialMixtureStistics)>,
}

impl SamplingCalculatorContainer {
	pub fn new() -> Self {
		Self {
			stats: Vec::new(),
		}
	}

	#[allow(clippy::too_many_arguments)]
	pub fn draw_ui(
		&mut self,
		ui: &Context,
		mixture: &Mixture,
		grid: &[u32],
		env_map: &EnvironmentMap,
		slice_size: usize,
		slice_expansion: usize,
		expansion_threshold: f32,
	) {
		egui::Window::new("Analytical Statistics").show(ui, |ui| {
			if ui.button("Calculate Statistics").clicked() {
				let calculator = SamplingCalculator::new(env_map, mixture, grid);
				self.stats = calculator.statistics(slice_size, slice_expansion, expansion_threshold);
			}

			if self.stats.is_empty() {
				ui.label("No statistics calculated yet.");
				return;
			}

			for (name, stat) in &self.stats {
				ui.heading(name);
				ui.label(format!("Variance: {:.4}", stat.variance));
				ui.label(format!("Density: {:.4}", stat.density));
				ui.label(format!("Contribution: {:.4}", stat.contribution));
				ui.label(format!("Miss Rate: {:.4}", stat.miss_rate));
				ui.label(format!("Visited Nodes: {:.4}", stat.visited_nodes));
			}
		});
	}
}

/// This struct can explicitly calculate many different statistics about the sampling process without conducting the
/// sampling itself.
struct SamplingCalculator<'a> {
	environment_map: &'a EnvironmentMap,
	mixture: &'a Mixture,
	grid: &'a [u32],
}

impl SamplingCalculator<'_> {
	pub fn new<'a>(environment_map: &'a EnvironmentMap, mixture: &'a Mixture, grid: &'a [u32]) -> SamplingCalculator<'a> {
		SamplingCalculator {
			environment_map,
			mixture,
			grid,
		}
	}

	pub fn statistics(
		&self,
		slice_size: usize,
		slice_expansion: usize,
		expansion_threshold: f32,
	) -> Vec<(String, PartialMixtureStistics)> {
		let pixel_count = self.environment_map.data.len();

		let stats = (0..pixel_count)
			.into_par_iter()
			.map(|i| self.calculate_pixel(i, slice_size, slice_expansion, expansion_threshold))
			.reduce_with(|a, b| {
				a.into_iter()
					.zip(b)
					.map(|(a, b)| PartialMixtureStistics::reduce(a, b))
					.collect()
			})
			.expect("No pixels in the environment map.");

		// assign names to the different statistics
		let names = vec![
			"Uniform",
			"Full Mixture",
			"Ideal n-Sample",
			"Morton Sampling",
			"Hierarchy Traversal",
			"KNN",
			"Grid",
		];

		names
			.into_iter()
			.zip(stats)
			.map(|(name, stat)| (name.to_string(), stat))
			.collect()
	}

	fn calculate_pixel(
		&self,
		i: usize,
		slice_size: usize,
		slice_expansion: usize,
		expansion_threshold: f32,
	) -> Vec<PartialMixtureStistics> {
		let mut stats = Vec::new();

		let pixel_count = self.environment_map.data.len();

		let pnt = Vec2::new(
			(i % self.environment_map.dim.x as usize) as f32 / self.environment_map.dim.x as f32,
			(i / self.environment_map.dim.x as usize) as f32 / self.environment_map.dim.y as f32,
		);
		let dir = pnt2dir(pnt);
		let color = env_map_get_pnt(&self.environment_map.data, self.environment_map.dim, pnt);

		let mut uniform = self.calculate_uniform(pnt, color);
		uniform.scale(1.0 / pixel_count as f64);
		stats.push(uniform);

		let mut full = self.calculate_pixel_full(pnt, dir, color);
		full.scale(1.0 / pixel_count as f64);
		stats.push(full);

		let mut ideal = self.calculate_pixel_ideal(pnt, dir, color, slice_size);
		ideal.scale(1.0 / pixel_count as f64);
		stats.push(ideal);

		let mut morton = self.calculate_pixel_morton(pnt, dir, color, slice_size, slice_expansion, expansion_threshold);
		morton.scale(1.0 / pixel_count as f64);
		stats.push(morton);

		let mut hierarchy = self.calculate_pixel_hierarchy(pnt, dir, color, slice_size);
		hierarchy.scale(1.0 / pixel_count as f64);
		stats.push(hierarchy);

		let mut knn = self.calculate_pixel_knn(pnt, dir, color, slice_size);
		knn.scale(1.0 / pixel_count as f64);
		stats.push(knn);

		let mut grid = self.calculate_pixel_grid(pnt, dir, color, self.grid, self.mixture, slice_size);
		grid.scale(1.0 / pixel_count as f64);
		stats.push(grid);

		stats
	}

	fn calculate_uniform(&self, pnt: Vec2, color: f64) -> PartialMixtureStistics {
		// uniform has same density everywhere
		let density = 1.0 / (4.0 * std::f64::consts::PI);
		let stats = AnalyticalMixtureStatistics::for_uniform(color, self.environment_map.ground_truth, density);

		let jacobian = jacobian(pnt) as f64;
		PartialMixtureStistics::from_mixture_stats(stats, jacobian)
	}

	fn calculate_pixel_ideal(&self, pnt: Vec2, dir: Vec3, color: f64, slice_size: usize) -> PartialMixtureStistics {
		let stats = AnalyticalMixtureStatistics::for_ideal(self.mixture, dir, color, self.environment_map.ground_truth, slice_size);
		let jacobian = jacobian(pnt) as f64;
		PartialMixtureStistics::from_mixture_stats(stats, jacobian)
	}

	/// Uses full mixture model.
	fn calculate_pixel_full(&self, pnt: Vec2, dir: Vec3, color: f64) -> PartialMixtureStistics {
		let stats = AnalyticalMixtureStatistics::for_full(self.mixture, dir, color, self.environment_map.ground_truth);
		let jacobian = jacobian(pnt) as f64;
		PartialMixtureStistics::from_mixture_stats(stats, jacobian)
	}

	/// Uses Morton sampling.
	fn calculate_pixel_morton(
		&self,
		pnt: Vec2,
		dir: Vec3,
		color: f64,
		slice_size: usize,
		slice_expansion: usize,
		expansion_threshold: f32,
	) -> PartialMixtureStistics {
		let (start, end) = MortonSliceSelection::select_slice(self.mixture, dir, slice_size);
		let (start, end) = self
			.mixture
			.expand_slice((start, end), slice_expansion, expansion_threshold, dir);
		let stats =
			AnalyticalMixtureStatistics::for_morton(self.mixture, dir, color, self.environment_map.ground_truth, (start, end));
		let jacobian = jacobian(pnt) as f64;
		PartialMixtureStistics::from_mixture_stats(stats, jacobian)
	}

	fn calculate_pixel_hierarchy(&self, pnt: Vec2, dir: Vec3, color: f64, slice_size: usize) -> PartialMixtureStistics {
		let mut pcg_dummy = Pcg32::new(0, 0);
		let (start, end, nodes_visited) = TowardsPdfTraversal::traverse(self.mixture, dir, &mut pcg_dummy, slice_size);

		let stats = AnalyticalMixtureStatistics::for_hierarchy(
			self.mixture,
			dir,
			color,
			self.environment_map.ground_truth,
			(start, end),
			nodes_visited,
		);
		let jacobian = jacobian(pnt) as f64;
		PartialMixtureStistics::from_mixture_stats(stats, jacobian)
	}

	fn calculate_pixel_knn(&self, pnt: Vec2, dir: Vec3, color: f64, slice_size: usize) -> PartialMixtureStistics {
		let stats =
			AnalyticalMixtureStatistics::for_knn::<1024>(self.mixture, dir, color, self.environment_map.ground_truth, slice_size, true);
		let jacobian = jacobian(pnt) as f64;
		PartialMixtureStistics::from_mixture_stats(stats, jacobian)
	}

	fn calculate_pixel_grid(
		&self,
		pnt: Vec2,
		dir: Vec3,
		color: f64,
		grid: &[u32],
		mixture: &Mixture,
		slice_size: usize,
	) -> PartialMixtureStistics {
		let layout = mixture.grid_layout();
		let cell_index = layout.pnt2cell(pnt);

		let stats = AnalyticalMixtureStatistics::for_grid(
			mixture,
			dir,
			cell_index,
			grid,
			color,
			self.environment_map.ground_truth,
			slice_size,
		);
		let jacobian = jacobian(pnt) as f64;

		PartialMixtureStistics::from_mixture_stats(stats, jacobian)
	}
}

/// This struct is used for generating JSON outputs for the plots in the thesis.
/// It is given the current env map and mixture and will dynamically configure everything else to generate data for
/// plotting.
pub struct PlotExport {
	plot_path: PathBuf,
}

// noinspection DuplicatedCode
impl PlotExport {
	pub fn new() -> Self {
		Self {
			plot_path: PathBuf::from("plots"),
		}
	}

	fn write_json<Data>(&self, name: &str, data: Vec<Data>) -> Result<(), Box<dyn std::error::Error>>
	where Data: Serialize {
		let path = self.plot_path.join(format!("{}.json", name));
		let file = std::fs::File::create(path)?;
		serde_json::to_writer_pretty(file, &data)?;

		Ok(())
	}

	/// Performs a single calculate for the entire environment map using the provided function to calculate the
	/// statistics.
	fn do_single_calculate_entire<F>(
		&self,
		func: F,
		env_map: &EnvironmentMap,
		mixture: &Mixture,
		grid: &[u32],
	) -> PartialMixtureStistics
	where
		F: Fn(&SamplingCalculator, usize, Vec2, Vec3, f64) -> PartialMixtureStistics + Sync,
	{
		let calculator = SamplingCalculator::new(env_map, mixture, grid);

		// parallelize the calculation of the statistics for each pixel using rayon
		let pixel_count = env_map.data.len();
		let mut stats = (0..pixel_count)
			.into_par_iter()
			.map(|i| {
				// we want func to define the statistics calculation, but also precompute a few things that are always identical
				let pnt = Vec2::new(
					(i % env_map.dim.x as usize) as f32 / env_map.dim.x as f32,
					(i / env_map.dim.x as usize) as f32 / env_map.dim.y as f32,
				);
				let dir = pnt2dir(pnt);
				let color = env_map_get_pnt(&env_map.data, env_map.dim, pnt);

				func(&calculator, i, pnt, dir, color)
			})
			.reduce_with(PartialMixtureStistics::reduce)
			.expect("No pixels in the environment map.");
		stats.scale(1.0 / pixel_count as f64);
		stats
	}

	pub fn draw_ui(&mut self, ui: &Context, env_map: &EnvironmentMap, mixture: &Mixture, grid_builder: &VMFGridBuilder) {
		egui::Window::new("Plot Export").show(ui, |ui| {
			if ui.button("NBS").clicked() {
				// output: run NBS for each number of components
				self.stop_watch_run_wrapper(|self_| self_.run_nbs(env_map, mixture), "NBS");
			}
			if ui.button("Morton").clicked() {
				// output: run Morton for each number of components
				self.stop_watch_run_wrapper(|self_| self_.run_morton(env_map, mixture), "Morton");
			}
			if ui.button("Morton Expansion 16th0.0001").clicked() {
				// output: run Morton for each number of components
				self.stop_watch_run_wrapper(
					|self_| self_.run_morton_expansion(env_map, mixture, 16, 0.0001),
					"Morton Expansion 16th0.0001",
				);
			}
			if ui.button("TSS").clicked() {
				// output: run TSS for each number of components
				self.stop_watch_run_wrapper(|self_| self_.run_hierarchy(env_map, mixture), "TSS");
			}
			if ui.button("KNN").clicked() {
				// output: run KNN for each number of components
				self.stop_watch_run_wrapper(|self_| self_.run_knn(env_map, mixture), "KNN");
			}
			if ui.button("Grid").clicked() {
				// output: run Grid for each number of components
				let grid = grid_builder.build(mixture);
				self.stop_watch_run_wrapper(|self_| self_.run_grid(env_map, mixture, &grid), "Grid");
			}

			// run all in sequence
			if ui.button("All").clicked() {
				let duration = self.stop_watch_run_wrapper(
					|self_| {
						self_.run_nbs(env_map, mixture);
						self_.run_morton(env_map, mixture);
						// self_.run_morton_expansion(env_map, mixture, 16, 0.0001);
						// self_.run_hierarchy(env_map, mixture);
						self_.run_knn(env_map, mixture);
						let grid = grid_builder.build(mixture);
						self_.run_grid(env_map, mixture, &grid);
					},
					"All",
				);
				ui.label(format!("All took {} minutes.", duration));
			}
		});
	}

	fn stop_watch_run_wrapper<F>(&mut self, func: F, name: &str) -> f64
	where F: FnOnce(&mut Self) {
		let start = std::time::Instant::now();
		func(self);
		let duration = start.elapsed().as_secs_f64() / 60.0;
		info!("{} took {} minutes.", name, duration);
		duration
	}

	fn run_nbs(&mut self, env_map: &EnvironmentMap, mixture: &Mixture) {
		// for each number of components, perform one nbs run
		let mut plot = Vec::new();
		for slice_size in 1..=100 {
			info!("Running NBS for {} components.", slice_size);
			let stats = self.do_single_calculate_entire(
				|calculator, _i, pnt, dir, color| calculator.calculate_pixel_ideal(pnt, dir, color, slice_size),
				env_map,
				mixture,
				&[0],
			);
			plot.push((slice_size, stats));
		}

		let name = format!("nbs_{}", env_map.name());
		self.write_json(&name, plot).expect("Failed to write NBS data.");
	}

	fn run_morton(&mut self, env_map: &EnvironmentMap, mixture: &Mixture) {
		// for each number of components, perform one nbs run
		let mut plot = Vec::new();
		for slice_size in 1..=100 {
			info!("Running Morton for {} components.", slice_size);
			let stats = self.do_single_calculate_entire(
				|calculator, _i, pnt, dir, color| calculator.calculate_pixel_morton(pnt, dir, color, slice_size, 0, 0.0),
				env_map,
				mixture,
				&[0],
			);
			plot.push((slice_size, stats));
		}

		let name = format!("morton_{}", env_map.name());
		self.write_json(&name, plot).expect("Failed to write Morton data.");
	}

	fn run_hierarchy(&mut self, env_map: &EnvironmentMap, mixture: &Mixture) {
		// for each number of components, perform one nbs run
		let mut plot = Vec::new();
		for slice_size in 1..=100 {
			info!("Running Hierarchy for {} components.", slice_size);
			let stats = self.do_single_calculate_entire(
				|calculator, _i, pnt, dir, color| calculator.calculate_pixel_hierarchy(pnt, dir, color, slice_size),
				env_map,
				mixture,
				&[0],
			);
			plot.push((slice_size, stats));
		}

		let name = format!("hierarchy_{}", env_map.name());
		self.write_json(&name, plot).expect("Failed to write Hierarchy data.");
	}

	fn run_morton_expansion(
		&mut self,
		env_map: &EnvironmentMap,
		mixture: &Mixture,
		expansion_size: usize,
		expansion_threshold: f32,
	) {
		// for each number of components, perform one nbs run
		let mut plot = Vec::new();
		for slice_size in 1..=100 {
			info!("Running Morton for {} components.", slice_size);
			let stats = self.do_single_calculate_entire(
				|calculator, _i, pnt, dir, color| {
					calculator.calculate_pixel_morton(pnt, dir, color, slice_size, expansion_size, expansion_threshold)
				},
				env_map,
				mixture,
				&[0],
			);
			plot.push((slice_size, stats));
		}

		let name = format!(
			"morton_expansion_{}_{}th{}",
			env_map.name(),
			expansion_size,
			expansion_threshold
		);
		self.write_json(&name, plot).expect("Failed to write Morton data.");
	}

	fn run_knn(&mut self, env_map: &EnvironmentMap, mixture: &Mixture) {
		// for each number of components, perform one nbs run
		let mut plot = Vec::new();

		// reverse order to start with the largest slice size (it takes the longest)
		for slice_size in (1..=100).rev() {
			info!("Running KNN for {} components.", slice_size);
			let stats = self.do_single_calculate_entire(
				|calculator, _i, pnt, dir, color| calculator.calculate_pixel_knn(pnt, dir, color, slice_size),
				env_map,
				mixture,
				&[0],
			);
			plot.push((slice_size, stats));
		}

		let name = format!("knn_{}", env_map.name());
		self.write_json(&name, plot).expect("Failed to write KNN data.");
	}

	fn run_grid(&mut self, env_map: &EnvironmentMap, mixture: &Mixture, grid: &[u32]) {
		const SLICE_SIZE_TARGET: usize = 100;

		// ensure grid can satisfy wanted limit
		if (mixture.grid_layout().bucket_size()) < SLICE_SIZE_TARGET {
			error!("Grid bucket size is too small for target slice size.");
			return;
		}

		// for each number of components, perform one nbs run
		let mut plot = Vec::new();

		// reverse order to start with the largest slice size (it takes the longest)
		for slice_size in 1..=SLICE_SIZE_TARGET {
			info!("Running Grid for {} components.", slice_size);
			let stats = self.do_single_calculate_entire(
				|calculator, _i, pnt, dir, color| calculator.calculate_pixel_grid(pnt, dir, color, grid, mixture, slice_size),
				env_map,
				mixture,
				grid,
			);
			plot.push((slice_size, stats));
		}

		let extend = mixture.grid_layout().extend();
		let name = format!(
			"grid_{}x{}x{}_{}",
			extend.x,
			extend.y,
			mixture.grid_layout().min_distance(),
			env_map.name()
		);
		self.write_json(&name, plot).expect("Failed to write Grid data.");
	}
}
