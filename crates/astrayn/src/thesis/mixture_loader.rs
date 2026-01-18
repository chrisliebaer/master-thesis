use std::path::PathBuf;

use glam::{
	UVec2,
	Vec3,
};
use miette::{
	IntoDiagnostic,
	WrapErr,
};
use serde::{
	Deserialize,
	Serialize,
};
use shader::thesis::{
	mixture::Mixture,
	vmf_distribution::VMFDistribution,
	vmf_grid::GridLayout,
};
use tracing::{
	debug,
	warn,
};

use crate::thesis::{
	util::simple_mixture,
	vmf_grid_builder::VMFGridBuilder,
	vmf_hierachy_builder::{
		TreeBuildMethod,
		VMFHierarchyBuilder,
	},
};

struct MixtureEntry {
	name: String,
	mixture: Box<Mixture>,
}

/// This struct is initialed with a directory and will index all available mixtures in that directory.
/// It implements an egui debug ui to select a mixture.
pub struct MixtureLoader {
	/// The directory from which mixtures are loaded.
	path: PathBuf,

	/// The available mixtures.
	entries: Vec<MixtureEntry>,
	selected: usize,

	/// The current mixture.
	runtime_mixture: Box<Mixture>,
	runtime_grid: Box<[u32]>,

	hierarchy_builder: VMFHierarchyBuilder,
	grid_builder: VMFGridBuilder,

	kappa_limit: Option<f32>,
}

impl MixtureLoader {
	pub fn new(path: PathBuf) -> Self {
		let simple_mixture = simple_mixture();
		let hierarchy_builder = VMFHierarchyBuilder::new(TreeBuildMethod::BottomUpRandom, None, 64);
		let grid_builder = VMFGridBuilder::new(GridLayout::new(UVec2::new(64, 16), 10.0, 100));
		let simple_grid = grid_builder.build(&simple_mixture);
		let mut new = Self {
			path,
			entries: Vec::new(),
			selected: 0,

			runtime_mixture: simple_mixture,
			runtime_grid: simple_grid,

			hierarchy_builder,
			grid_builder,

			kappa_limit: None,
		};
		new.reload();
		new.rebuild();
		new
	}

	pub fn get_grid_builder(&self) -> &VMFGridBuilder {
		&self.grid_builder
	}

	/// Load all mixtures from the directory.
	pub fn reload(&mut self) {
		let mut entries = Vec::new();

		// always add the simple mixture
		entries.push(MixtureEntry {
			name: "Simple Mixture".to_string(),
			mixture: simple_mixture(),
		});

		for entry in std::fs::read_dir(&self.path).unwrap() {
			let entry = entry.unwrap();
			let path = entry.path();

			// only load json files, ignore directories and other files
			if let Some(ext) = path.extension() {
				if ext == "json" {
					let name = path.file_stem().unwrap().to_str().unwrap().to_string();
					let mixture = load_vmf(&path).unwrap();
					entries.push(MixtureEntry {
						name,
						mixture,
					});
				}
			}
		}

		self.entries = entries;
	}

	pub fn get_mixture(&self) -> &Mixture {
		&self.runtime_mixture
	}

	pub fn get_all_for_testing(&mut self) -> Vec<(String, Box<Mixture>)> {
		// store so we can restore the selected mixture afterwards
		let current = self.selected;

		// we need to build all mixtures for testing and return them with their name
		let mut all = Vec::new();
		for id in 0..self.entries.len() {
			self.selected = id;
			let name = self.entries[id].name.clone();
			self.rebuild();
			all.push((name, self.runtime_mixture.clone()));
		}

		// restore selected mixture
		self.selected = current;
		self.rebuild();

		all
	}

	pub fn get_grid(&self) -> &[u32] {
		&self.runtime_grid
	}

	pub fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
		let mut changed = false;
		ui.small(
			"Configuration changes require an explicit rebuild to apply. Switching the active mixture will automatically rebuild.",
		);

		let mut mixture_changed = false;
		egui::ComboBox::from_label("Mixture")
			.selected_text(self.entries[self.selected].name.clone())
			.show_ui(ui, |ui| {
				for (i, entry) in self.entries.iter().enumerate() {
					if ui.selectable_label(self.selected == i, entry.name.clone()).clicked() {
						changed = true;
						mixture_changed = true;
						self.selected = i;
					}
				}
			});

		// hierarchy builder
		egui::CollapsingHeader::new("Hierarchy Builder")
			.default_open(true)
			.show(ui, |ui| {
				changed |= self.hierarchy_builder.draw_ui(ui);
			});

		// grid builder
		egui::CollapsingHeader::new("Grid Builder").default_open(true).show(ui, |ui| {
			changed |= self.grid_builder.draw_ui(ui);
		});

		let mut kappa_limit_enabled = self.kappa_limit.is_some();
		if ui.checkbox(&mut kappa_limit_enabled, "Kappa Limit").changed() {
			self.kappa_limit = if kappa_limit_enabled { Some(50.0) } else { None };
			changed = true;
		}

		if let Some(mut kappa_limit) = self.kappa_limit {
			if ui
				.add(egui::Slider::new(&mut kappa_limit, 0.0..=1000.0).text("Kappa Limit"))
				.changed()
			{
				self.kappa_limit = Some(kappa_limit);
				changed = true;
			}
		}

		// explicit rebuild button
		if ui.button("Rebuild").clicked() || mixture_changed {
			self.rebuild();
			changed = true;
		}

		changed
	}

	fn rebuild(&mut self) {
		let template = &self.entries[self.selected].mixture;
		self.runtime_mixture = template.clone();

		if let Some(kappa_limit) = self.kappa_limit {
			for i in 0..self.runtime_mixture.size() {
				let (component, _) = self.runtime_mixture.get_component_mut(i);
				if component.kappa() > kappa_limit {
					component.set_kappa(kappa_limit);
				}
			}
		}

		let grid_layout = self.grid_builder.grid_layout();
		self.runtime_mixture.finalize(&self.hierarchy_builder, grid_layout);
		self.runtime_grid = self.grid_builder.build(&self.runtime_mixture);
	}
}

fn load_vmf(path: &PathBuf) -> miette::Result<Box<Mixture>> {
	debug!("loading vmf from {:?}", path);

	let file = std::fs::File::open(path).into_diagnostic().wrap_err("failed to open file")?;

	// file contains a json list of VMFs
	let serde_vmfs: Vec<SerdeVMF> = serde_json::from_reader(file)
		.into_diagnostic()
		.wrap_err("failed to parse json")?;

	let pi_sum = serde_vmfs.iter().map(|json| json.pi).sum::<f32>();

	// build mixture from json
	let mut mixture = Box::<Mixture>::default();
	for json in serde_vmfs {
		let vmf = VMFDistribution::new(Vec3::new(json.dir.x, json.dir.y, json.dir.z), json.k);
		mixture.add_component(vmf, json.pi / pi_sum);
	}

	// check all kappa values are greater than 0 (including an epsilon)
	let mut max_kappa = 0.0;
	for i in 0..mixture.get_count() {
		let (component, _) = mixture.get_component(i);
		let kappa = component.kappa();
		if kappa > max_kappa {
			max_kappa = kappa;
		}
		assert!(kappa > 0.0, "kappa value is not greater than 0 for component {}", i);
	}

	// check that largest kappa is still numerically stable
	let ex = f64::exp((-2.0 * max_kappa) as f64);
	if ex == 0.0 {
		warn!("mixture {} exceeds exp limit with a kappa of {}", path.display(), max_kappa);
	}

	Ok(mixture)
}

#[derive(Debug, Serialize, Deserialize)]
struct SerdeVMF {
	pi: f32,
	k: f32,
	dir: SerdeVec3,
}

#[derive(Debug, Serialize, Deserialize)]
struct SerdeVec3 {
	x: f32,
	y: f32,
	z: f32,
}
