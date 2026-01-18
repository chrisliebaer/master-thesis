use bytemuck::{
	Pod,
	Zeroable,
};
use glam::{
	vec3,
	vec4,
	Mat3,
	Vec2,
	Vec3,
	Vec4,
	Vec4Swizzles,
};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::num_traits::FloatConst;

use crate::thesis::entrypoints::{
	validate_scalar,
	validate_vector3,
};

/// Represents a von Mises-Fisher distribution.
#[derive(Copy, Clone, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct VMFDistribution {
	/// This is a combination of the mean direction and the concentration of the distribution in the last component.
	param: Vec4,
}

impl VMFDistribution {
	/// Creates a new von Mises-Fisher distribution.
	pub fn new(mean: Vec3, kappa: f32) -> Self {
		let vmf = Self {
			param: vec4(mean.x, mean.y, mean.z, kappa),
		};
		vmf.validate();
		vmf
	}

	pub fn pdf(&self, dir: Vec3) -> f32 {
		let u = self.param.xyz();
		let k = self.param.w;
		let n = Self::norm_n(k);
		let pdf = n * f32::exp(k * (dir.dot(u) - 1.0));

		validate_scalar(pdf);

		pdf
	}

	pub fn sample(&self, rnd: Vec2) -> Vec3 {
		// cast all values to f64 to reduce precision errors in the exponent
		// if rnd.y is 0, the exp function can diminish the value to 0, causing log to return -inf
		let kappa = self.param.w as f64;
		let rndy = rnd.y as f64;

		let w = if rndy == 0.0 {
			let w = 1.0 + (-2.0 * kappa) / kappa;
			w as f32
		} else {
			// FIXME: rust-gpu can't generate proper f64 code
			// let w = 1.0 + f64::ln(rndy + (1.0 - rndy) * f64::exp(-2.0 * kappa)) / kappa;

			let rndy = rnd.y;
			let kappa = self.param.w;
			let w = 1.0 + f32::ln(rndy + (1.0 - rndy) * f32::exp(-2.0 * kappa)) / kappa;
			w
		};

		validate_scalar(w);

		let w2 = f32::sqrt(1.0 - w * w);
		validate_scalar(w2);

		// effectively turning rnd into point on unit circle
		let d = Vec3::new(
			w2 * f32::cos(2.0 * f32::PI() * rnd.x),
			w2 * f32::sin(2.0 * f32::PI() * rnd.x),
			w,
		);

		validate_vector3(d);

		let sample = Self::ortho_normal_basis(self.param.xyz()) * d;
		validate_vector3(sample);

		sample
	}

	pub fn mean(&self) -> Vec3 {
		self.param.xyz()
	}

	pub fn kappa(&self) -> f32 {
		self.param.w
	}

	pub fn set_kappa(&mut self, kappa: f32) {
		self.param.w = kappa;
	}

	/// Returns true if the given direction is within the given quantile of the distribution.
	pub fn is_within_quantile(&self, dir: Vec3, quantile: f32) -> bool {
		let quantile = 1.0 - quantile;
		let rng = Vec2::new(0.0, quantile);
		let sample = self.sample(rng);

		// take angle between mean and sample as threshold for quantile
		let threshold = self.angle_between(sample);
		let angle = self.angle_between(dir);

		angle <= threshold
	}

	fn angle_between(&self, other: Vec3) -> f32 {
		let dot = self.mean().dot(other);
		let dot = dot.clamp(-1.0, 1.0);
		let angle = f32::acos(dot);
		validate_scalar(angle);
		angle
	}

	fn norm_n(k: f32) -> f32 {
		// no ideal what pi is (is for mixture sampling)
		let pi = 1.0;
		if k > 0.0 {
			pi * k / (2.0 * f32::PI() * (1.0 - f32::exp(-2.0 * k)))
		} else {
			0.0
		}
	}

	// TODO: precompute? or check if compiler does it
	fn ortho_normal_basis(u: Vec3) -> Mat3 {
		let sign = u.z.signum();
		let a = -1.0 / (sign + u.z);
		let b = u.x * u.y * a;

		Mat3::from_cols(
			vec3(1.0 + sign * u.x * u.x * a, sign * b, -sign * u.x),
			vec3(b, sign + u.y * u.y * a, -u.y),
			vec3(u.x, u.y, u.z),
		)
	}

	fn validate(&self) {
		validate_vector3(self.param.xyz());
		validate_scalar(self.param.w);
	}
}

#[cfg(test)]
mod tests {
	use approx::assert_abs_diff_eq;
	use pcg32::Pcg32;

	use super::*;
	use crate::thesis::{
		math::{
			jacobian,
			pnt2dir,
		},
		pcg32_ext::Pcg32Ext,
	};

	const SUBDIVISIONS: usize = 1000;
	const SAMPLES: usize = 10000;

	fn distributions() -> Vec<VMFDistribution> {
		let kappas = [0.1, 1.0, 10.0, 100.0];
		let means = [
			Vec3::new(1.0, 0.0, 0.0),
			Vec3::new(0.0, 1.0, 0.0),
			Vec3::new(0.0, 0.0, 1.0),
			Vec3::new(1.0, 1.0, 1.0),
			Vec3::new(-1.0, 0.42f32, 1.0),
		];

		let mut distributions = Vec::new();
		for kappa in kappas.iter() {
			for mean in means.iter() {
				let mean = mean.normalize();
				distributions.push(VMFDistribution::new(mean, *kappa));
			}
		}
		distributions
	}

	// TODO: it works, but dividing by the number of subdivisions seems fishy, understand math *ugh*
	fn summation_helper(vmf: &VMFDistribution) -> f64 {
		let mut sum = 0.0f64;
		for i in 0..SUBDIVISIONS {
			for j in 0..SUBDIVISIONS {
				let x = (i as f32 + 0.5f32) / SUBDIVISIONS as f32;
				let y = (j as f32 + 0.5f32) / SUBDIVISIONS as f32;
				let pnt = Vec2::new(x, y);

				let dir = pnt2dir(pnt);
				sum += (vmf.pdf(dir) * jacobian(pnt)) as f64;
			}
		}
		sum / (SUBDIVISIONS * SUBDIVISIONS) as f64
	}

	#[test]
	fn test_summation() {
		// pdf accros the sphere should sum to 1 if divided by the number of subdivisions
		let distributions = distributions();

		for vmf in distributions.iter() {
			let sum = summation_helper(vmf);
			assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-2);
		}
	}

	#[test]
	fn test_sampling() {
		// only for very small kappa, otherwise we need WAY too many samples

		let vmf = VMFDistribution::new(Vec3::new(1.0, 0.0, 0.0), 0.1);
		// always use same seed for reproducibility
		let mut rng = Pcg32::new(0, 0);

		let mut sum = 0.0f64;
		for _ in 0..SAMPLES {
			let rnd = Vec2::new(rng.gen_f32(), rng.gen_f32());
			let dir = vmf.sample(rnd);
			let contribution = 1f64 / vmf.pdf(dir) as f64;
			sum += contribution;
		}
		let avg = sum / SAMPLES as f64;

		// should equal sphere surface area
		assert_abs_diff_eq!(avg, 4.0 * f64::PI(), epsilon = 1e-1);
	}

	#[test]
	fn test_quantile() {
		let distributions = distributions();

		// 0 to 1 with 0.05 steps
		let quantiles = (0..=20).map(|i| i as f32 * 0.05);

		for vmf in distributions.iter() {
			// q1 should always be true
			assert!(vmf.is_within_quantile(vmf.mean(), 0.99));
		}

		// just run everything else once, there isn't really anything to test, just to see if it runs
		for quantile in quantiles {
			for vmf in distributions.iter() {
				vmf.is_within_quantile(vmf.mean(), quantile);
			}
		}
	}

	#[test]
	fn test_ghost_mixture() {
		let mean_dir = pnt2dir(Vec2::new(0.97, 0.78));
		let ghost_dir = pnt2dir(Vec2::new(0.47, 0.22));
		let unrelated_dir = pnt2dir(Vec2::new(20.0, 25.0));

		let vmf = VMFDistribution::new(mean_dir, 20.0);

		// mean should be in 5% quantile
		assert!(vmf.is_within_quantile(mean_dir, 0.05));

		// unrelated direction should not be in 5% quantile
		assert!(!vmf.is_within_quantile(unrelated_dir, 0.05));

		// ghost should not be in 5% quantile
		assert!(!vmf.is_within_quantile(ghost_dir, 0.05));
	}
}
