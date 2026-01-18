use glam::{
	UVec2,
	Vec2,
	Vec3,
};
use kahan::KahanSum;
use shader::thesis::{
	math::jacobian,
	mixture::Mixture,
	vmf_distribution::VMFDistribution,
	vmf_grid::GridLayout,
};

use crate::thesis::vmf_hierachy_builder::{
	TreeBuildMethod,
	VMFHierarchyBuilder,
};

/// Creates a simple mixture of von Mises-Fisher distributions.
///
/// This mixture is created by placing rings of points on a sphere.
/// These are not equidistant or in any way fancy, but they serve as a good starting point.
pub fn simple_mixture() -> Box<Mixture> {
	let mut mixture = Box::<Mixture>::default();
	let kappa = 10f32;
	let count = 1024;
	let rings = count / 8;
	let points = count / rings;
	for i in 0..rings {
		for j in 0..points {
			let long = (i as f32 / rings as f32) * 2f32 * std::f32::consts::PI;
			let lat = (j as f32 / points as f32) * std::f32::consts::PI;
			let dir = Vec3::new(long.sin() * lat.cos(), long.sin() * lat.sin(), long.cos());
			mixture.add_component(VMFDistribution::new(dir, kappa), 1.0 / count as f32);
		}
	}

	// just use whatever hierarchy builder, this mixture is simply for testing
	let hierarchical_builder = VMFHierarchyBuilder::new(TreeBuildMethod::BottomUpRandom, Some(8), 8);
	mixture.finalize(&hierarchical_builder, &GridLayout::new(UVec2::ZERO, 0.0, 8));

	mixture
}

pub fn integrate_sphere(env_map: &[f64], dim: UVec2) -> f64 {
	let sup = 4;
	let width = dim.x * sup;
	let height = dim.y * sup;

	let mut total = KahanSum::new();
	for y in 0..height {
		for x in 0..width {
			// offset of 0.5, so we don't sample the 0 border, where jacobian(0) = 0
			let pnt = Vec2::new((x as f32 + 0.5) / width as f32, (y as f32 + 0.5) / height as f32);
			let idx = (y / sup) * dim.x + (x / sup);
			total += env_map[idx as usize] * jacobian(pnt) as f64 / (height * width) as f64
		}
	}

	total.sum()
}

#[cfg(test)]
mod tests {
	use approx::assert_abs_diff_eq;
	use shader::thesis::TEXTURE_SIZE;

	use super::*;

	#[test]
	fn integration_is_correct() {
		let env_map = vec![1.0; TEXTURE_SIZE * TEXTURE_SIZE];
		let integral = integrate_sphere(&env_map, UVec2::new(TEXTURE_SIZE as u32, TEXTURE_SIZE as u32));
		let pi = std::f64::consts::PI;
		assert_abs_diff_eq!(integral, 4.0 * pi, epsilon = 1e-1);
	}

	// TODO: integration über cos winkel bezüglich einer achse
	// oder merhere achsen, kostet ja nichts
}
