use glam::{
	UVec2,
	Vec2,
	Vec3,
};

use crate::thesis::math::dir2pnt;

pub fn env_map_validate(dim: UVec2, data: &[f32]) {
	if data.len() != (dim.x * dim.y) as usize {
		panic!("env_map_validate: data length does not match dimensions");
	}
}

pub fn env_map_get_pnt<T: Copy>(env_map: &[T], dim: UVec2, pnt: Vec2) -> T {
	// allow values to clip slightly into both ends
	let pnt = pnt.clamp_with_epsilon(Vec2::splat(0f32), Vec2::splat(1f32), 1e-6);

	// ensure that the coordinates are in the range [0, 1)
	if pnt.x < 0.0 || pnt.x > 1.0 || pnt.y < 0.0 || pnt.y > 1.0 {
		panic!("env_map_get_pnt: pnt out of range");
	}

	// get the coordinates in the range [0, dim)
	let x = (pnt.x * dim.x as f32) as usize;
	let y = (pnt.y * dim.y as f32) as usize;

	let idx = y * dim.x as usize + x;
	env_map[idx]
}

pub fn env_map_get_dir<T: Copy>(env_map: &[T], dim: UVec2, dir: Vec3) -> T {
	let pnt = dir2pnt(dir);
	env_map_get_pnt(env_map, dim, pnt)
}

trait ClampWithEpsilon {
	fn clamp_with_epsilon(self, min: Self, max: Self, epsilon: f32) -> Self;
}

impl ClampWithEpsilon for f32 {
	fn clamp_with_epsilon(self, min: f32, max: f32, epsilon: f32) -> f32 {
		if self < min && self > min - epsilon {
			min
		} else if self > max && self < max + epsilon {
			max
		} else {
			self.clamp(min, max)
		}
	}
}

impl ClampWithEpsilon for Vec2 {
	fn clamp_with_epsilon(self, min: Vec2, max: Vec2, epsilon: f32) -> Vec2 {
		Vec2::new(
			self.x.clamp_with_epsilon(min.x, max.x, epsilon),
			self.y.clamp_with_epsilon(min.y, max.y, epsilon),
		)
	}
}
