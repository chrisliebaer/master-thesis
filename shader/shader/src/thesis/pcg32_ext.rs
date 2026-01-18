use pcg32::Pcg32;

/// Extension methods for PcG32.
pub trait Pcg32Ext {
	/// Generates a random number in the range [0, 1).
	fn gen_f32(&mut self) -> f32;

	/// Generates a random number in the range [0, max).
	fn gen_max<T>(&mut self, max: T) -> T
	where T: Into<f32> + From<f32> {
		(max.into() * self.gen_f32()).into()
	}
}

// https://github.com/wjakob/pcg32/blob/master/pcg32.h
impl Pcg32Ext for Pcg32 {
	fn gen_f32(&mut self) -> f32 {
		// draw from [1, 2) and subtract 1
		let u = self.gen() >> 9 | 0x3f800000;
		f32::from_bits(u) - 1.0
	}
}
