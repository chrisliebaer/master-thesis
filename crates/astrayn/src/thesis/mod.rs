use std::sync::Arc;

use bevy_ecs::{
	change_detection::NonSendMut,
	prelude::{
		Commands,
		Res,
		ResMut,
		Resource,
	},
	system::SystemParam,
};
use debug_images::DebugImages;
use egui_plot::{
	PlotPoint,
	PlotPoints,
};
use egui_winit_vulkano::Gui;
use exr::{
	image::PixelImage,
	prelude::RgbaChannels,
};
use glam::Vec2;
use shader::thesis::entrypoints::debug_image_pp::ProcessingOptions;
use tracing::{
	debug,
	info,
};
use vulkano::{
	descriptor_set::{
		allocator::StandardDescriptorSetAllocator,
		layout::DescriptorSetLayout,
		PersistentDescriptorSet,
		WriteDescriptorSet,
	},
	device::Device,
	format::Format,
	image::{
		sampler::SamplerCreateInfo,
		view::ImageView,
		Image,
		ImageCreateInfo,
		ImageType,
		ImageUsage,
	},
	memory::allocator::{
		AllocationCreateInfo,
		MemoryTypeFilter,
		StandardMemoryAllocator,
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
	sync,
	DeviceSize,
};

use crate::{
	application::{
		Application,
		Plugin,
	},
	debug::DebugUi,
	ecs::ResWrap,
	renderer::vulkan::{
		renderer::{
			Allocators,
			GraphicsQueue,
		},
		shaders::ShaderRegistry,
	},
	scheduler::{
		MainSchedule,
		SetupSchedule,
	},
	thesis::{
		debug_images::{
			ImageNames,
			PostProcessingUi,
		},
		env_map::{
			EnvironmentMap,
			PlotExport,
			SamplingCalculatorContainer,
		},
		mixture_loader::MixtureLoader,
		shader_wrapper::{
			format_big_number,
			Context,
			MixtureDebugRenderer,
			MixtureSamplerResult,
			MixtureSamplerShader,
			PostProcessShader,
		},
	},
};

mod debug_images;
mod env_map;
mod mixture_loader;
mod shader_wrapper;
mod util;
mod vmf_grid_builder;
pub mod vmf_hierachy_builder;

#[derive(SystemParam)]
struct InitSystemParam<'w> {
	device: ResWrap<'w, Arc<Device>>,
	queue: Res<'w, GraphicsQueue>,
	allocators: Res<'w, Allocators>,
	shader_registry: Res<'w, ShaderRegistry>,
	egui: NonSendMut<'w, Gui>,
}

#[derive(SystemParam)]
struct OpSystemParams<'w> {
	state: ResMut<'w, SamplingState>,

	device: ResWrap<'w, Arc<Device>>,
	queue: Res<'w, GraphicsQueue>,
	ui: Res<'w, DebugUi>,
	allocators: Res<'w, Allocators>,

	egui: NonSendMut<'w, Gui>,
}

/// Stores the current state of the sampling configuration UI.
/// This state is then copied into the ShaderDataIn buffer upon execution.
struct SamplingConfigUiState {
	/// The seed for the random number generator.
	seed: u64,

	/// Whether the seed should be auto incremented after each run.
	auto_increment: bool,

	/// The size of the lobe slice.
	slice_size: u32,

	slice_expansion: u32,
	expansion_threshold: f32,

	/// Divisor for the resolution of the image.
	divide: usize,
}

/// Implementation of the sampling configuration UI.
impl SamplingConfigUiState {
	fn new() -> Self {
		Self {
			seed: 1337,
			auto_increment: true,
			slice_size: 16,
			slice_expansion: 16,
			expansion_threshold: 0.50,
			divide: 8,
		}
	}

	fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
		ui.heading("Sampling Configuration");

		let mut changed = false;

		ui.horizontal(|ui| {
			ui.label("Seed");
			changed |= ui.add(egui::DragValue::new(&mut self.seed).speed(1.0)).changed();

			changed |= ui.checkbox(&mut self.auto_increment, "Auto Increment").changed();
		});

		ui.horizontal(|ui| {
			ui.label("Slice Size");
			changed |= ui.add(egui::DragValue::new(&mut self.slice_size).speed(1.0)).changed();
		});

		ui.horizontal(|ui| {
			ui.label("Slice Expansion");
			changed |= ui.add(egui::DragValue::new(&mut self.slice_expansion).speed(1.0)).changed();
		});

		ui.horizontal(|ui| {
			ui.label("Expansion Threshold");
			changed |= ui
				.add(
					egui::DragValue::new(&mut self.expansion_threshold)
						.speed(0.0001)
						.clamp_range(0.0..=0.1),
				)
				.changed();
		});

		ui.horizontal(|ui| {
			ui.label("Divide");

			// must be a power of 2 in order to work with downscaler
			changed |= ui
				.add(egui::DragValue::new(&mut self.divide).speed(1.0).clamp_range(1..=16))
				.changed();

			if !self.divide.is_power_of_two() {
				self.divide = self.divide.next_power_of_two();
			}
		});

		changed
	}
}

/// Settings exclusively used for the debug view.
struct DebugViewSettings {
	/// The quantile render each lobe at.
	quantile: f32,
	morton_debug: u32,
}

impl DebugViewSettings {
	fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
		ui.heading("Debug View Settings");
		let mut changed = false;

		ui.horizontal(|ui| {
			ui.label("Quantile");
			changed |= ui
				.add(egui::DragValue::new(&mut self.quantile).speed(0.01).clamp_range(0.0..=1.0))
				.changed();
		});

		ui.horizontal(|ui| {
			ui.label("Morton Debug");
			changed |= ui.add(egui::DragValue::new(&mut self.morton_debug).speed(50000.0)).changed();
		});

		changed
	}
}

#[derive(Resource)]
struct SamplingState {
	/// Stores the sampling configuration between runs.
	sampling_config_ui_state: SamplingConfigUiState,

	/// Stores the debug view settings.
	debug_settings: DebugViewSettings,

	/// The current environment map.
	environment_map: EnvironmentMap,

	debug_views: DebugImages,

	/// Mixture loader
	mixture_loader: MixtureLoader,

	/// Debug post-processing renderer.
	post_processing_renderer: PostProcessShader,

	post_processing_options: ProcessingOptions,

	/// Debug renderer for the mixture distribution.
	mixture_debug_renderer: MixtureDebugRenderer,

	/// Debug for tinted pdf visualization.
	mixture_tint_renderer: MixtureDebugRenderer,

	/// Debug for tree tinted pdf visualization.
	mixture_tree_tint_renderer: MixtureDebugRenderer,

	/// Debug for RMSE visualization.
	mixture_rmse_renderer: MixtureDebugRenderer,

	/// Debug for RMSE hierarchy visualization.
	mixture_rmse_hierarchy_renderer: MixtureDebugRenderer,

	/// Debug for RMSE knn visualization.
	mixture_rmse_knn_renderer: MixtureDebugRenderer,

	/// Debug for RMSE grid visualization.
	mixture_rmse_grid_renderer: MixtureDebugRenderer,

	/// Debug for difference between pdfs.
	mixture_pdf_diff_renderer: MixtureDebugRenderer,

	/// Analytical statistics about sampling process.
	statisics: SamplingCalculatorContainer,

	/// The shader that handles the sampling.
	sampler_shader: MixtureSamplerShader,

	/// Debug for grid debug visualization.
	mixture_draw_grid_debug: MixtureDebugRenderer,

	/// List of available EXR files in the `../hdr_images` directory.
	exr_files: Vec<String>,
	/// The result of the last run, if there was one.
	result: Option<MixtureSamplerResult>,

	plot: PlotExport,
}

fn load_exr(path: &str, divide: usize) -> PixelImage<Vec<Vec<f64>>, RgbaChannels> {
	let image = exr::prelude::read_first_rgba_layer_from_file(
		path,
		move |resolution, _| {
			let default_pixel = 0.0f64;
			let width = resolution.width() / divide;
			let height = resolution.height() / divide;

			let empty_line = vec![default_pixel; width];
			vec![empty_line; height]
		},
		move |pixel_vector, position, (r, g, b, _a): (f32, f32, f32, f32)| {
			// load as grayscale

			let pixel = ((r + g + b) / 3.0) as f64;

			// consider downsampling averages across the divide x divide block
			let pixel = pixel / (divide * divide) as f64;

			let x = position.x() / divide;
			let y = position.y() / divide;

			pixel_vector[y][x] += pixel;
		},
	)
	.expect("failed to read exr file");

	debug!("loaded exr image: {:?}", image.layer_data.attributes);

	image
}

impl SamplingState {
	fn get_available_exr_files() -> Vec<String> {
		let mut files = std::fs::read_dir("../hdr_images")
			.expect("failed to read directory")
			.map(|entry| entry.unwrap().path())
			.filter(|path| path.extension().map_or(false, |ext| ext == "exr"))
			.map(|path| path.to_string_lossy().to_string())
			.collect::<Vec<_>>();

		files.sort_unstable();

		files
	}

	fn create_gpu_objects(mut commands: Commands, mut params: InitSystemParam) {
		let context = Context::new(params.device.clone(), params.queue.clone(), params.allocators.clone());
		commands.insert_resource(SamplingState {
			plot: PlotExport::new(),
			sampling_config_ui_state: SamplingConfigUiState::new(),
			debug_settings: DebugViewSettings {
				quantile: 0.05,
				morton_debug: 0,
			},
			environment_map: EnvironmentMap::empty(),
			debug_views: DebugImages::new(&params.allocators, &mut params.egui),
			post_processing_renderer: PostProcessShader::new(&params.shader_registry, context.clone()),
			post_processing_options: ProcessingOptions::default(),
			mixture_debug_renderer: MixtureDebugRenderer::new(
				&params.shader_registry,
				context.clone(),
				"thesis::entrypoints::debug_shader::draw_mixtures",
			),
			mixture_tint_renderer: MixtureDebugRenderer::new(
				&params.shader_registry,
				context.clone(),
				"thesis::entrypoints::debug_shader::tint_pdf",
			),
			mixture_tree_tint_renderer: MixtureDebugRenderer::new(
				&params.shader_registry,
				context.clone(),
				"thesis::entrypoints::debug_shader::tint_tree_pdf",
			),
			mixture_rmse_renderer: MixtureDebugRenderer::new(
				&params.shader_registry,
				context.clone(),
				"thesis::entrypoints::debug_shader::draw_rmse",
			),
			mixture_rmse_hierarchy_renderer: MixtureDebugRenderer::new(
				&params.shader_registry,
				context.clone(),
				"thesis::entrypoints::debug_shader::draw_stats_hierarchy",
			),
			mixture_rmse_knn_renderer: MixtureDebugRenderer::new(
				&params.shader_registry,
				context.clone(),
				"thesis::entrypoints::debug_shader::draw_stats_knn",
			),
			mixture_rmse_grid_renderer: MixtureDebugRenderer::new(
				&params.shader_registry,
				context.clone(),
				"thesis::entrypoints::debug_shader::draw_stats_grid",
			),
			mixture_pdf_diff_renderer: MixtureDebugRenderer::new(
				&params.shader_registry,
				context.clone(),
				"thesis::entrypoints::debug_shader::draw_pdf_diff",
			),
			mixture_draw_grid_debug: MixtureDebugRenderer::new(
				&params.shader_registry,
				context.clone(),
				"thesis::entrypoints::debug_shader::draw_grid_debug",
			),
			statisics: SamplingCalculatorContainer::new(),
			sampler_shader: MixtureSamplerShader::new(&params.shader_registry, context.clone()),

			mixture_loader: MixtureLoader::new("../mixture_python/export/json".into()),

			exr_files: Self::get_available_exr_files(),
			result: None,
		});
	}

	fn draw_exr_file_selection(&mut self, ui: &mut egui::Ui) -> bool {
		let mut changed = false;

		// exr file selection and reload button for refeshing the list of available files
		ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
			let mut current = self.environment_map.path.as_str();
			egui::ComboBox::from_label("Environment Map")
				.selected_text(current)
				.show_ui(ui, |ui| {
					ui.set_min_width(200.0);

					// default to "None" until user has selected a file
					if current == "None" {
						ui.selectable_value(&mut current, "None", "None");
					}

					for file in &self.exr_files {
						changed |= ui.selectable_value(&mut current, file, file).clicked();
					}
				});

			// if path changed, we need to load the new file and update aggregate values
			if changed {
				debug!("changed env map to: {}", current);

				let new_map = EnvironmentMap::load(current, self.sampling_config_ui_state.divide);
				self.environment_map = new_map;
			}

			if ui.button("Reload").clicked() {
				self.exr_files = Self::get_available_exr_files();
				info!("reloading exr files, available: {:?}", self.exr_files);
			}
		});

		changed
	}

	fn draw_enviroment_map_ui(&mut self, ui: &mut egui::Ui) {
		ui.heading("Environment Map");
		ui.label(format!("Ground Truth: {}", self.environment_map.ground_truth));
	}

	fn sample_run(&mut self) {
		let result = self.sampler_shader.sample(
			&self.environment_map,
			self.mixture_loader.get_mixture(),
			self.mixture_loader.get_grid(),
			&self.sampling_config_ui_state,
		);

		self.result = Some(result);
	}

	fn draw_debug_ui_system(mut params: OpSystemParams) {
		let state = &mut *params.state;
		let ui = &**params.ui;
		let mut changed = false;

		// check changed the bit in debug views from last frame
		changed |= state.debug_views.get_and_clear_changed();

		egui::Window::new("Mixture Loader")
			.collapsible(true)
			.resizable(true)
			.show(ui, |ui| {
				changed |= state.mixture_loader.draw_ui(ui);
			});

		egui::Window::new("Sampling")
			.collapsible(true)
			.resizable(true)
			.show(ui, |ui| {
				changed |= state.sampling_config_ui_state.draw_ui(ui);
				changed |= state.debug_settings.draw_ui(ui);
				state.draw_enviroment_map_ui(ui);
				changed |= state.draw_exr_file_selection(ui);

				state.sampler_shader.draw_ui(ui);

				// some changes require a re-run
				if changed {
					// recreate debug images

					if changed {
						state.debug_views.update_env_map(&mut params.egui, &state.environment_map);
					}

					// only update currently visible image
					if let Some(name) = state.debug_views.get_current() {
						let renderer = match name {
							ImageNames::MixtureDebug => &state.mixture_debug_renderer,
							ImageNames::MixtureTint => &state.mixture_tint_renderer,
							ImageNames::MixtureTreeTint => &state.mixture_tree_tint_renderer,
							ImageNames::RMSE => &state.mixture_rmse_renderer,
							ImageNames::RMSEHierarchy => &state.mixture_rmse_hierarchy_renderer,
							ImageNames::RMSEKNN => &state.mixture_rmse_knn_renderer,
							ImageNames::RMSEGrid => &state.mixture_rmse_grid_renderer,
							ImageNames::PDFDiff => &state.mixture_pdf_diff_renderer,
							ImageNames::GridDebug => &state.mixture_draw_grid_debug,
						};

						renderer.draw(
							&state.debug_views.get_image_view(name),
							&state.environment_map,
							state.mixture_loader.get_mixture(),
							state.mixture_loader.get_grid(),
							state.debug_views.get_cursor_pos(),
							state.sampling_config_ui_state.slice_size,
							state.debug_settings.quantile,
							state.debug_settings.morton_debug,
							state.sampling_config_ui_state.slice_expansion,
							state.sampling_config_ui_state.expansion_threshold,
						);
					}
				}

				// run configured sampling pass
				if ui.button("Sample").clicked() {
					state.sample_run();
					if state.sampling_config_ui_state.auto_increment {
						state.sampling_config_ui_state.seed += 1;
					}
				}
			});

		// we need to render this after debug update, since update might invalidate textures
		// but we need to carry over the changed flag
		egui::Window::new("Debug Images")
			.collapsible(true)
			.resizable(true)
			.show(ui, |ui| {
				state.debug_views.draw_ui(
					ui,
					Context::new(params.device.clone(), params.queue.clone(), params.allocators.clone()),
					&state.environment_map,
				);
			});

		if let Some(result) = &state.result {
			egui::Window::new("Sampling Result")
				.collapsible(true)
				.resizable(true)
				.show(ui, |ui| {
					ui.label(format!("Samples: {}", format_big_number(result.sample_count)));
					ui.label(format!("Variance: {}", result.variance_sum));
					ui.label(format!("Morton miss rate: {:.3}%", result.slice_miss_rate * 100.0));
					ui.label(format!("Estimate: {}", result.estimate));
					ui.label(format!("Delta: {}", result.estimate - state.environment_map.ground_truth));
					ui.heading("Variance Plot");
					// egui_plot::Plot::new("variance_plot").show(ui, |ui| {
					// 	ui.line(egui_plot::Line::new(PlotPoints::Owned(result.plot.clone())));
					// });
				});
		}

		// if changed or post-processing settings changed, we need to re-run the post-processing shader
		let pp_changed = state.post_processing_options.post_processing_ui(ui);
		if changed || pp_changed {
			// take currently selected image and apply post-processing into target image
			if let Some(name) = state.debug_views.get_current() {
				let source = state.debug_views.get_image_view(name);
				let target = state.debug_views.get_pp_image_view();
				state
					.post_processing_renderer
					.draw(&source, &target, &state.post_processing_options, &state.environment_map);
			}
		}

		state.statisics.draw_ui(
			ui,
			state.mixture_loader.get_mixture(),
			state.mixture_loader.get_grid(),
			&state.environment_map,
			state.sampling_config_ui_state.slice_size as usize,
			state.sampling_config_ui_state.slice_expansion as usize,
			state.sampling_config_ui_state.expansion_threshold,
		);

		state.plot.draw_ui(
			ui,
			&state.environment_map,
			state.mixture_loader.get_mixture(),
			state.mixture_loader.get_grid_builder(),
		);
	}
}

pub struct ThesisPlugin;

impl Plugin for ThesisPlugin {
	fn build(&self, app: &mut Application) {
		app
			.add_systems(SetupSchedule::AfterGraphicsBackend, SamplingState::create_gpu_objects)
			.add_systems(MainSchedule::Main, SamplingState::draw_debug_ui_system);
	}
}

#[cfg(test)]
mod tests {
	use shader::thesis::morton::{
		interleave_morton,
		interleave_morton_naive,
	};

	/// This test is horribly slow, so we run it in the main application where we have access to rayon.
	#[test]
	fn test_matches_naive() {
		use rayon::prelude::*;
		(0..u16::MAX).into_par_iter().for_each(|x| {
			(0..u16::MAX).into_par_iter().for_each(|y| {
				let z = interleave_morton(x, y);
				let z_naive = interleave_morton_naive(x, y);
				assert_eq!(z, z_naive);
			});
		});
	}
}
