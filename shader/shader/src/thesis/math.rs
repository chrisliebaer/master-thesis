use glam::{
	Vec2,
	Vec3,
	Vec4,
};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::num_traits::FloatConst;

pub fn pnt2dir(pnt: Vec2) -> Vec3 {
	let theta = pnt.y * f32::PI();
	let phi = pnt.x * 2.0 * f32::PI();
	Vec3::new(phi.sin() * theta.sin(), theta.cos(), phi.cos() * theta.sin())
}

pub fn dir2pnt(dir: Vec3) -> Vec2 {
	let uv = Vec2::new(dir.x.atan2(dir.z) / (2.0 * f32::PI()), dir.y.acos() / f32::PI());
	uv - uv.floor()
}

pub fn jacobian(pnt: Vec2) -> f32 {
	let theta = pnt.y * f32::PI();
	theta.sin() * 2.0 * f32::PI() * f32::PI()
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
	let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
	t * t * (3.0 - 2.0 * t)
}

fn mix(a: Vec4, b: Vec4, t: f32) -> Vec4 {
	a * (1.0 - t) + b * t
}

fn step(edge: f32, x: f32) -> f32 {
	(x >= edge) as u32 as f32
}

// https://unpkg.com/browse/glsl-colormap@1.0.1/viridis.glsl
#[allow(clippy::excessive_precision)]
pub fn viridis(x: f32) -> Vec4 {
	const E0: f32 = 0.0;
	const V0: Vec4 = Vec4::new(0.26666666666666666, 0.00392156862745098, 0.32941176470588235, 1.0);
	const E1: f32 = 0.13;
	const V1: Vec4 = Vec4::new(0.2784313725490196, 0.17254901960784313, 0.47843137254901963, 1.0);
	const E2: f32 = 0.25;
	const V2: Vec4 = Vec4::new(0.23137254901960785, 0.3176470588235294, 0.5450980392156862, 1.0);
	const E3: f32 = 0.38;
	const V3: Vec4 = Vec4::new(0.17254901960784313, 0.44313725490196076, 0.5568627450980392, 1.0);
	const E4: f32 = 0.5;
	const V4: Vec4 = Vec4::new(0.12941176470588237, 0.5647058823529412, 0.5529411764705883, 1.0);
	const E5: f32 = 0.63;
	const V5: Vec4 = Vec4::new(0.15294117647058825, 0.6784313725490196, 0.5058823529411764, 1.0);
	const E6: f32 = 0.75;
	const V6: Vec4 = Vec4::new(0.3607843137254902, 0.7843137254901961, 0.38823529411764707, 1.0);
	const E7: f32 = 0.88;
	const V7: Vec4 = Vec4::new(0.6666666666666666, 0.8627450980392157, 0.19607843137254902, 1.0);
	const E8: f32 = 1.0;
	const V8: Vec4 = Vec4::new(0.9921568627450981, 0.9058823529411765, 0.1450980392156863, 1.0);

	let a0 = smoothstep(E0, E1, x);
	let a1 = smoothstep(E1, E2, x);
	let a2 = smoothstep(E2, E3, x);
	let a3 = smoothstep(E3, E4, x);
	let a4 = smoothstep(E4, E5, x);
	let a5 = smoothstep(E5, E6, x);
	let a6 = smoothstep(E6, E7, x);
	let a7 = smoothstep(E7, E8, x);

	Vec4::max(
		mix(V0, V1, a0) * step(E0, x) * step(x, E1),
		Vec4::max(
			mix(V1, V2, a1) * step(E1, x) * step(x, E2),
			Vec4::max(
				mix(V2, V3, a2) * step(E2, x) * step(x, E3),
				Vec4::max(
					mix(V3, V4, a3) * step(E3, x) * step(x, E4),
					Vec4::max(
						mix(V4, V5, a4) * step(E4, x) * step(x, E5),
						Vec4::max(
							mix(V5, V6, a5) * step(E5, x) * step(x, E6),
							Vec4::max(
								mix(V6, V7, a6) * step(E6, x) * step(x, E7),
								mix(V7, V8, a7) * step(E7, x) * step(x, E8),
							),
						),
					),
				),
			),
		),
	)
}

#[cfg(test)]
mod tests {
	use approx::assert_abs_diff_eq;

	use super::*;

	fn check_reciprocal(pnt: Vec2) {
		let dir = pnt2dir(pnt);
		let pnt2 = dir2pnt(dir);
		assert_abs_diff_eq!(pnt.x, pnt2.x, epsilon = 1e-6);
		assert_abs_diff_eq!(pnt.y, pnt2.y, epsilon = 1e-6);
	}

	#[test]
	fn pnt2dir_and_dir2pnt_are_reciprocal() {
		let vecs = [
			Vec2::new(0.0, 0.0),
			Vec2::new(0.5, 0.5),
			Vec2::new(0.0, 0.5),
			Vec2::new(0.5, 0.0),
			Vec2::new(0.99, 0.99),
		];

		for &pnt in vecs.iter() {
			check_reciprocal(pnt);
		}
	}

	#[test]
	fn jacobian_is_positive() {
		let vecs = [
			Vec2::new(0.01, 0.01),
			Vec2::new(0.5, 0.5),
			Vec2::new(0.1, 0.5),
			Vec2::new(0.5, 0.1),
			Vec2::new(0.99, 0.99),
		];

		for &pnt in vecs.iter() {
			assert!(jacobian(pnt) > 0.0);
		}
	}
}
