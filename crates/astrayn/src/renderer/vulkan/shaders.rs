use std::{
	path::Path,
	sync::Arc,
};

use bevy_ecs::{
	change_detection::Res,
	prelude::{
		Commands,
		Resource,
	},
};
use miette::{
	IntoDiagnostic,
	Result,
	WrapErr,
};
use tracing::info;
use vulkano::{
	device::Device,
	shader::{
		reflect::entry_points,
		spirv,
		spirv::{
			bytes_to_words,
			Instruction::Extension,
			Spirv,
		},
		EntryPoint,
		EntryPointInfo,
		ShaderModule,
		ShaderModuleCreateInfo,
	},
};

use crate::{
	ecs::AnyRes,
	renderer::vulkan::shaders::debug_ui::DebugUiState,
};

struct RegisteredShader {
	/// The runtime ID of the shader. This is used to identify the shader in the shader registry.
	runtime_id: u32,

	/// The source that we use for loading the shader code.
	source: ShaderSource,

	/// The Vulkan representation of the shader module.
	module: Arc<ShaderModule>,

	/// Additional information about the shader.
	info: ShaderInfo,
}

pub struct ShaderInfo {
	// TODO: not sure if we need that after extracting the shader info
	spirv: Spirv,
	extensions: Box<[String]>,
	entry_points: Box<[(spirv::Id, EntryPointInfo)]>,
}

impl ShaderInfo {
	pub fn new(words: &[u32]) -> Result<Self> {
		let spirv = Spirv::new(words).into_diagnostic().wrap_err("Failed to parse SPIR-V")?;
		let entry_points = entry_points(&spirv).collect();
		let extensions = spirv
			.iter_extension()
			.map(|ext| {
				if let Extension {
					name, ..
				} = ext
				{
					name.to_string()
				} else {
					unreachable!("Iterating over extensions should only yield extensions")
				}
			})
			.collect();

		// TODO: collect struct and struct member information to reconstruct layouts of shader types

		Ok(Self {
			spirv,
			extensions,
			entry_points,
		})
	}
}

#[derive(Resource)]
pub struct ShaderRegistry {
	/// The device on which this registry operates.
	device: Arc<Device>,

	/// The next runtime ID that will be assigned to a shader.
	next_runtime_id: u32,

	/// Shaders which have been registered with this registry and passed to Vulkan.
	shaders: Vec<RegisteredShader>,
}

impl ShaderRegistry {
	pub fn new(device: Arc<Device>) -> Self {
		Self {
			device: device.clone(),
			next_runtime_id: 0,
			shaders: Vec::new(),
		}
	}

	pub fn entry_point_by_name(&self, name: &str) -> Option<EntryPoint> {
		self
			.shaders
			.iter()
			.find(|shader| shader.info.entry_points.iter().any(|(_, info)| info.name == name))
			.map(|shader| shader.module.entry_point(name).expect("Failed to find entry point"))
	}

	pub fn load_shader(&mut self, source: ShaderSource) -> Result<()> {
		let shader_code = (source.loader)()?;
		let shader_code = bytes_to_words(&shader_code)
			.into_diagnostic()
			.wrap_err("Failed to convert shader code to SPIR-V")?;

		let module = unsafe {
			ShaderModule::new(self.device.clone(), ShaderModuleCreateInfo::new(&shader_code))
				.into_diagnostic()
				.wrap_err(format!("Failed to create shader module from source: {}", source.name))?
		};

		let info = ShaderInfo::new(&shader_code)?;

		self.shaders.push(RegisteredShader {
			runtime_id: self.next_runtime_id,
			source,
			module,
			info,
		});
		self.next_runtime_id += 1;

		Ok(())
	}

	/// Attempts to reload all shaders from their respective source.
	/// Errors if any shader fails to load, meaning that there is no fallback to the previous state.
	pub fn reload_shaders(&mut self) -> Result<()> {
		for shader in self.shaders.iter_mut() {
			// TODO: log error and continue with remaining shaders
			let shader_code = (shader.source.loader)()?;
		}

		Ok(())
	}
}

pub(super) fn load_shaders(shader_registry: &mut ShaderRegistry) {
	// TODO: remove once we have proper asset loading
	// iterate over all .spv files in the shader directory and add them to the registry
	for entry in std::fs::read_dir("shader/target/spirv-builder/spirv-unknown-vulkan1.2/release/deps/shader.spvs").unwrap() {
		let entry = entry.unwrap();
		let path = entry.path();
		if Some("spv") == path.extension().and_then(|x| x.to_str()) {
			info!("loading shader: {:?}", path);
			shader_registry.load_shader(ShaderSource::from_file(&path)).unwrap();
		}
	}
}

pub(super) fn create_shader_registry(mut commands: Commands, device: Res<AnyRes<Arc<Device>>>) {
	let mut registry = ShaderRegistry::new(device.clone());
	let debug_ui_state = DebugUiState;

	load_shaders(&mut registry);

	commands.insert_resource(registry);
	commands.insert_resource(debug_ui_state);
}

/// A shader source. This uniquely identifies a shader. Sources are use for the initial loading of shaders but also are
/// used to reload shaders.
pub struct ShaderSource {
	name: String,
	loader: Box<dyn Fn() -> Result<Box<[u8]>> + Send + Sync>,
}

impl ShaderSource {
	pub fn from_fn(source: String, f: impl Fn() -> Result<Box<[u8]>> + Send + Sync + 'static) -> Self {
		Self {
			name: source,
			loader: Box::new(f),
		}
	}

	pub fn from_file(path: &Path) -> Self {
		let path = path.to_owned();
		Self::from_fn(format!("file://{}", path.display()), move || {
			std::fs::read(&path).map(Into::into).into_diagnostic()
		})
	}
}

pub mod debug_ui {
	use std::ops::{
		Deref,
		DerefMut,
	};

	use bevy_ecs::prelude::{
		Res,
		ResMut,
		Resource,
	};
	use egui::ScrollArea;
	use miette::Result;
	use tracing::warn;

	use crate::{
		debug::{
			DebugUi,
			EguiDebugHelper,
		},
		renderer::vulkan::shaders::ShaderRegistry,
	};

	pub fn debugui_shader_registry(
		ui: Res<DebugUi>,
		shader_registry: Res<ShaderRegistry>,
		mut debug_ui_state: ResMut<DebugUiState>,
	) {
		DebugUiState::draw_debug_ui(
			debug_ui_state.deref_mut(),
			ui.deref().deref(),
			egui::Id::new("shader-registry"),
			&shader_registry,
		)
		.unwrap();
	}

	#[derive(Resource)]
	pub struct DebugUiState;

	impl DebugUiState {
		pub fn draw_debug_ui(&mut self, ui: &egui::Context, id: egui::Id, registry: &ShaderRegistry) -> Result<()> {
			let mut reload_shaders = false;

			egui::Window::new("Shader Registry").id(id).show(ui, |ui| {
				ui.small(
					"This window contains a list of all shaders that are currently loaded. It allows you basic inspection of the shaders.",
				);

				if ui.button("Reload Shaders").clicked() {
					reload_shaders = true;
				}

				ui.separator();

				ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
					ui.with_layout(egui::Layout::top_down(egui::Align::LEFT).with_cross_justify(true), |ui| {
						ui.collapsing_header_from_iter(
							"shader-registry-list".into(),
							"Shader Modules",
							registry.shaders.iter(),
							|ui, id, shader| {
								// make tree unique with runtime id
								let id = id.with(format!("-{}", shader.runtime_id));

								egui::CollapsingHeader::new(&shader.source.name).id_source(id).show(ui, |ui| {
									// shader extensions
									ui.collapsing_header_from_iter(
										id.with("-extensions"),
										"Extensions",
										shader.info.extensions.iter(),
										|ui, _, ext| {
											ui.label(ext);
										},
									);

									// shader entry points
									ui.collapsing_header_from_iter(
										id.with("-entry-points"),
										"Entry Points",
										shader.info.entry_points.iter(),
										|ui, id, (op_id, info)| {
											// entry point node
											egui::CollapsingHeader::new(format!("{}: ({:?})", info.name, info.execution_model))
												.id_source(id.with(format!("-{}", op_id)))
												.show(ui, |ui| {
													// bindings
													ui.collapsing_header_from_iter(
														id.with("-entry-point-bindings"),
														"Bindings",
														info.descriptor_binding_requirements.iter(),
														|ui, id, ((set, binding), req)| {
															egui::CollapsingHeader::new(format!("Set: {}, Location: {}", set, binding))
																.id_source(id.with(format!("-{}-{}", set, binding)))
																.show(ui, |ui| {
																	ui.collapsing_header_from_iter(
																		id.with("-types"),
																		"Types",
																		req.descriptor_types.iter(),
																		|ui, _, ty| {
																			ui.label(format!("{:?}", ty));
																		},
																	);

																	// TODO:
																	ui.label("TODO: add more information about descriptor");
																});
														},
													);

													// TODO:
													ui.label("TODO: add more information for push constants, shader in and out");
												});
										},
									);
								});
							},
						);
					});
				});
			});

			if reload_shaders {
				warn!("Shader reloading is not yet implemented!");
			}

			Ok(())
		}
	}
}
