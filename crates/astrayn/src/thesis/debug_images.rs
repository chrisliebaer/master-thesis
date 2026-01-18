use std::{
	collections::HashSet,
	path::PathBuf,
	sync::Arc,
};

use egui::{
	load::SizedTexture,
	ImageSource,
	TextureId,
	Widget,
	WidgetText,
};
use egui_winit_vulkano::Gui;
use exr::prelude::{
	Encoding,
	ImageAttributes,
	IntegerBounds,
	LayerAttributes,
	SpecificChannels,
	WritableImage,
};
use glam::{
	Vec2,
	Vec4,
};
use rfd::FileDialog;
use shader::thesis::entrypoints::debug_image_pp::{
	ProcessingOptions,
	FLAG_LOG_SCALE,
};
use tracing::info;
use vulkano::{
	buffer::{
		Buffer,
		BufferCreateInfo,
		BufferUsage,
	},
	command_buffer::{
		AutoCommandBufferBuilder,
		CommandBufferUsage,
		CopyImageToBufferInfo,
	},
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
	},
	sync,
	sync::GpuFuture,
};

use crate::{
	renderer::vulkan::renderer::Allocators,
	thesis::{
		env_map::EnvironmentMap,
		shader_wrapper::Context,
	},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageNames {
	MixtureDebug,
	MixtureTint,
	MixtureTreeTint,
	RMSE,
	PDFDiff,
	RMSEHierarchy,
	RMSEKNN,
	GridDebug,
	RMSEGrid,
}

type UpdateFn = Box<dyn Fn(&Allocators, &EnvironmentMap) -> Arc<Image> + Send + Sync>;
/// Describes a debug image used by the DebugImages struct.
///
/// This struct bridged the gap between Vulkan images and their egui representation.
/// The struct also encapsulates recreation of the image when the environment map changes and a new resolution is
/// needed.
struct DebugImage {
	name: ImageNames,

	/// The Vulkan image.
	image: Arc<Image>,

	/// The Vulkan image view.
	image_view: Arc<ImageView>,

	/// Function for updating the image.
	update: UpdateFn,
}

impl DebugImage {
	/// Creates a new debug image.
	fn new(name: ImageNames, allocators: &Allocators, environment_map: &EnvironmentMap, update: UpdateFn) -> Self {
		let image = update(allocators, environment_map);
		let image_view = Self::create_from_image(image.clone());
		Self {
			name,
			image,
			image_view,
			update,
		}
	}

	fn create_from_image(image: Arc<Image>) -> Arc<ImageView> {
		ImageView::new_default(image.clone()).expect("failed to create image view")
	}

	/// Updates the image.
	fn update(&mut self, allocators: &Allocators, environment_map: &EnvironmentMap) {
		let new_image = (self.update)(allocators, environment_map);
		let image_view = Self::create_from_image(new_image.clone());

		// update fields
		self.image = new_image;
		self.image_view = image_view;
	}
}

/// A struct that holds image views for various debugging purposes.
/// Using egui, the user can select which image to display.
pub struct DebugImages {
	current: Option<usize>,
	allocators: Allocators,
	images: Vec<DebugImage>,
	cursor_pos: Option<Vec2>,

	/// Has no influence on the rendering, only tracks position for the user.
	last_cursor_pos: Option<Vec2>,
	changed: bool,

	/// This is the final image with post-processing applied.
	pp_image: Arc<Image>,
	pp_image_view: Arc<ImageView>,
	pp_texture_id: TextureId,

	/// The last path where the user saved an image.
	last_save_path: Option<String>,

	/// Should write store the input image, or the final mix from post-processing?
	save_should_write_raw: bool,
}

impl DebugImages {
	pub fn new(allocators: &Allocators, egui: &mut Gui) -> Self {
		let dummy_env_map = EnvironmentMap::empty();
		let images = vec![
			DebugImage::new(ImageNames::MixtureDebug, allocators, &dummy_env_map, Box::new(rgbaf32)),
			DebugImage::new(ImageNames::MixtureTint, allocators, &dummy_env_map, Box::new(rgbaf32)),
			DebugImage::new(ImageNames::MixtureTreeTint, allocators, &dummy_env_map, Box::new(rgbaf32)),
			DebugImage::new(ImageNames::RMSE, allocators, &dummy_env_map, Box::new(rgbaf32)),
			DebugImage::new(ImageNames::RMSEHierarchy, allocators, &dummy_env_map, Box::new(rgbaf32)),
			DebugImage::new(ImageNames::RMSEKNN, allocators, &dummy_env_map, Box::new(rgbaf32)),
			DebugImage::new(ImageNames::RMSEGrid, allocators, &dummy_env_map, Box::new(rgbaf32)),
			DebugImage::new(ImageNames::PDFDiff, allocators, &dummy_env_map, Box::new(rgbaf32)),
			DebugImage::new(ImageNames::GridDebug, allocators, &dummy_env_map, Box::new(rgbaf32)),
		];

		// ensure that each image name is unique
		let names = images.iter().map(|image: &DebugImage| image.name).collect::<HashSet<_>>();
		if names.len() != images.len() {
			panic!("duplicate image names found");
		}

		let pp_image = rgbaf32(allocators, &dummy_env_map);
		let pp_image_view = DebugImage::create_from_image(pp_image.clone());
		let pp_texture_id = egui.register_user_image_view(pp_image_view.clone(), SamplerCreateInfo::default());

		Self {
			current: None,
			allocators: allocators.clone(),
			images,
			cursor_pos: None,
			last_cursor_pos: None,
			changed: false,

			pp_image,
			pp_image_view,
			pp_texture_id,

			last_save_path: None,
			save_should_write_raw: true,
		}
	}

	pub fn get_image_view(&self, name: ImageNames) -> Arc<ImageView> {
		self
			.images
			.iter()
			.find(|image| image.name == name)
			.expect("image not found")
			.image_view
			.clone()
	}

	/// Notifies the debug images that the environment map has been updated.
	pub fn update_env_map(&mut self, egui: &mut Gui, environment_map: &EnvironmentMap) {
		// update all images
		for image in &mut self.images {
			image.update(&self.allocators, environment_map);
		}

		egui.unregister_user_image(self.pp_texture_id);
		self.pp_image = rgbaf32(&self.allocators, environment_map);
		self.pp_image_view = DebugImage::create_from_image(self.pp_image.clone());
		self.pp_texture_id = egui.register_user_image_view(self.pp_image_view.clone(), SamplerCreateInfo::default());
	}

	pub(crate) fn draw_ui(&mut self, ui: &mut egui::Ui, context: Context, environment_map: &EnvironmentMap) {
		ui.heading("Select Debug Image");

		let current_str = if let Some(current) = self.current {
			format!("{:?}", self.images[current].name)
		} else {
			"None".to_string()
		};

		egui::ComboBox::from_label("Debug Image")
			.selected_text(current_str)
			.show_ui(ui, |ui| {
				// if no image is selected, we inject a "None" option
				if self.current.is_none() {
					ui.selectable_value(&mut self.current, None, "None");
				}

				for (i, image) in self.images.iter().enumerate() {
					let name = format!("{:?}", image.name);
					ui.selectable_value(&mut self.current, Some(i), &name);
				}
			});

		self.draw_save_button(ui, context, environment_map);

		// horizontal layout for the clear cursor button
		ui.horizontal(|ui| {
			// button for clearing the cursor position (will cause change bit to be set)
			if ui.button("Clear Cursor").clicked() {
				self.cursor_pos = None;
				self.changed = true;
			}

			if let Some(last_cursor_pos) = self.last_cursor_pos {
				ui.label(format!("Current: ({:.2}, {:.2})", last_cursor_pos.x, last_cursor_pos.y));
			}

			// if cursor is set, display the current position
			if let Some(cursor_pos) = self.cursor_pos {
				ui.label(format!("Clicked: ({:.2}, {:.2})", cursor_pos.x, cursor_pos.y));
			}
		});

		ui.separator();
		if self.current.is_some() {
			let image = &self.pp_image;
			let dim = {
				let extent = image.extent();
				[extent[0] as f32, extent[1] as f32]
			};

			let egui_image = egui::Image::new(ImageSource::Texture(SizedTexture::new(self.pp_texture_id, dim)))
				.sense(egui::Sense::click_and_drag())
				.shrink_to_fit()
				.ui(ui);

			// always capture position, so we can display it in the UI
			let pos = ui.input(|input| {
				input.pointer.latest_pos().unwrap_or(egui::Pos2 {
					x: 0.0,
					y: 0.0,
				})
			});
			// let pos = egui_image.interact_pointer_pos().unwrap();
			let pos = pos - egui_image.rect.left_top();

			// we need to consider the current scaling of the image widget
			let dims = egui_image.rect;
			let pos = egui::Pos2::new(pos.x / dims.width(), pos.y / dims.height());

			self.last_cursor_pos = Some(Vec2::new(pos.x, pos.y));

			if egui_image.dragged() {
				// drag operations can begin inside the image and move outwards, in this case, we ignore the drag temporarily
				if pos.x >= 0.0 && pos.x <= 1.0 && pos.y >= 0.0 && pos.y <= 1.0 {
					self.cursor_pos = Some(Vec2::new(pos.x, pos.y));
					self.changed = true;
				}
			}
		} else {
			ui.label("No image selected");
		}
	}

	pub fn get_and_clear_changed(&mut self) -> bool {
		let changed = self.changed;
		self.changed = false;
		changed
	}

	pub fn get_current(&self) -> Option<ImageNames> {
		self.current.map(|i| self.images[i].name)
	}

	pub fn get_cursor_pos(&self) -> Option<Vec2> {
		self.cursor_pos
	}

	pub fn get_pp_image_view(&self) -> Arc<ImageView> {
		self.pp_image_view.clone()
	}

	/// Draws a save button for the currently selected image.
	///
	/// Clicking the button will open a file dialog to save the image as an exr file.
	fn draw_save_button(&mut self, ui: &mut egui::Ui, context: Context, environment_map: &EnvironmentMap) {
		if let Some(current) = self.current {
			ui.horizontal(|ui| {
				// checkbox for toggling between raw and post-processed image
				ui.checkbox(&mut self.save_should_write_raw, "Save Raw Image");

				if ui.button("Save").clicked() {
					let raw_image = &self.images[current];
					let (image, name) = if self.save_should_write_raw {
						(&raw_image.image, format!("{:?}_raw", raw_image.name))
					} else {
						(&self.pp_image, format!("{:?}_pp", raw_image.name))
					};

					let extend = image.extent();
					let dim = (extend[0] as usize, extend[1] as usize);

					// ask user for file path before proceeding
					let default_file_name = format!("debug_{}_{}_{}x{}.exr", environment_map.name(), name, dim.0, dim.1);
					let file_path = FileDialog::new()
						.add_filter("EXR", &["exr"])
						.set_file_name(default_file_name)
						.set_directory(std::env::current_dir().unwrap())
						.save_file();

					if let Some(file_path) = file_path {
						let mut builder = AutoCommandBufferBuilder::primary(
							&context.allocators.command_buffer,
							context.queue.0.queue_family_index(),
							CommandBufferUsage::OneTimeSubmit,
						)
						.expect("failed to create command buffer builder");

						let size: u32 = extend.iter().product();
						let buffer = Buffer::new_slice::<Vec4>(
							context.allocators.memory.clone(),
							BufferCreateInfo {
								usage: BufferUsage::TRANSFER_DST,
								..Default::default()
							},
							AllocationCreateInfo {
								memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
								..Default::default()
							},
							size.into(),
						)
						.expect("failed to create buffer");

						builder
							.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image.clone(), buffer.as_bytes().clone()))
							.expect("failed to copy image to buffer");

						let command_buffer = builder.build().expect("failed to build command buffer");
						let future = sync::now(context.device.clone())
							.then_execute(context.queue.0.clone(), command_buffer)
							.expect("failed to execute command buffer")
							.then_signal_fence_and_flush()
							.expect("failed to signal fence and flush");

						future.wait(None).expect("failed to wait for future");

						let debug_layer_data = {
							let read = buffer.read().expect("failed to read buffer");
							read.iter().map(|b| b.to_owned()).collect::<Vec<Vec4>>()
						};

						// build exr layers, first layer is the debug layer
						let debug_layer = exr::prelude::Layer::new(
							dim,
							LayerAttributes::named(name.as_str()),
							Encoding::FAST_LOSSLESS,
							SpecificChannels::rgb::<f32, f32, f32>(|pos| {
								let exr::prelude::Vec2(x, y) = pos;
								let index = y * dim.0 + x;
								let vec: Vec4 = debug_layer_data[index];
								(vec.x, vec.y, vec.z)
							}),
						);

						let env_map_layer = exr::prelude::Layer::new(
							dim,
							LayerAttributes::named("Environment Map"),
							Encoding::FAST_LOSSLESS,
							SpecificChannels::build().with_channel("R").with_pixel_fn(|pos| {
								let exr::prelude::Vec2(x, y) = pos;
								let index = y * dim.0 + x;
								let vec = environment_map.data[index];
								(vec as f32,)
							}),
						);

						let image = exr::prelude::Image::empty(ImageAttributes::new(IntegerBounds::from_dimensions(dim)))
							.with_layer(debug_layer)
							.with_layer(env_map_layer);

						image
							.write()
							.to_file(file_path.clone())
							.expect("failed to write image to file");

						// save the path for the next time (remove the file name)
						let path = file_path.parent().unwrap().to_str().unwrap().to_string();
						self.last_save_path = Some(path);
					}
				}

				// button for opening cwd
				if ui.button("Open Last Directory").clicked() {
					let path = self
						.last_save_path
						.clone()
						.map(PathBuf::from)
						.unwrap_or_else(|| std::env::current_dir().unwrap().to_path_buf());

					// on windows, use explorer on linux, use xdg-open
					let binary = if cfg!(windows) { "explorer" } else { "xdg-open" };

					std::process::Command::new(binary)
						.arg(path)
						.output()
						.expect("failed to open explorer");
				}
			});
		}
	}
}

pub trait PostProcessingUi {
	fn post_processing_ui(&mut self, ui: &egui::Context) -> bool;
	fn draw_flag_ui(&mut self, ui: &mut egui::Ui, bit: u64, name: impl Into<WidgetText>) -> bool;
}

impl PostProcessingUi for ProcessingOptions {
	fn post_processing_ui(&mut self, ui: &egui::Context) -> bool {
		let mut changed = false;
		egui::Window::new("Post Processing").show(ui, |ui| {
			ui.heading("Post Processing");
			if ui.button("Reset").clicked() {
				*self = Default::default();
				changed |= true;
			}

			ui.horizontal(|ui| {
				ui.label("Ghost:");
				changed |= ui
					.add(egui::Slider::new(&mut self.env_map_ghost, 0.0..=1.0).step_by(0.01))
					.changed();
			});

			ui.horizontal(|ui| {
				ui.label("Scale:");
				changed |= ui.add(egui::Slider::new(&mut self.scale, 0.0..=5.0).step_by(0.001)).changed();
			});

			ui.horizontal(|ui| {
				ui.label("Offset:");
				changed |= ui
					.add(egui::Slider::new(&mut self.offset, -1.0..=1.0).step_by(0.01))
					.changed();
			});

			changed |= self.draw_flag_ui(ui, FLAG_LOG_SCALE, "Log Scale");

			// checkboxes for individual mask bits for channel R, G, B
			ui.horizontal(|ui| {
				ui.label("Mask:");
				let channels = ["R", "G", "B"];
				for i in 0..3 {
					let mask = 1 << i;
					let mut checked = self.mask & mask != 0;
					if ui.checkbox(&mut checked, channels[i]).changed() {
						if checked {
							self.mask |= mask;
						} else {
							self.mask &= !mask;
						}
						changed = true;
					}
				}
			});
		});

		changed
	}

	fn draw_flag_ui(&mut self, ui: &mut egui::Ui, bit: u64, name: impl Into<WidgetText>) -> bool {
		let mut checked = self.flags & bit != 0;
		if ui.checkbox(&mut checked, name).changed() {
			if checked {
				self.flags |= bit;
			} else {
				self.flags &= !bit;
			}
			true
		} else {
			false
		}
	}
}

fn rgbaf32(allocators: &Allocators, env_map: &EnvironmentMap) -> Arc<Image> {
	let dim = env_map.dim;
	Image::new(
		allocators.memory.clone(),
		ImageCreateInfo {
			image_type: ImageType::Dim2d,
			format: Format::R32G32B32A32_SFLOAT,
			extent: [dim.x, dim.y, 1],
			usage: ImageUsage::STORAGE | ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC,
			..Default::default()
		},
		AllocationCreateInfo {
			memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
			..Default::default()
		},
	)
	.expect("failed to create image")
}
