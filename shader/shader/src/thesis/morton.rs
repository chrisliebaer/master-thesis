// implementation from: https://github.com/ReinierMaas/morton

use glam::Vec2;

pub fn pnt2morton(pnt: Vec2) -> u32 {
	let scaled = pnt * u16::MAX as f32;
	let ivec = scaled.as_ivec2();
	interleave_morton(ivec.x as u16, ivec.y as u16)
}

pub fn interleave_morton(x: u16, y: u16) -> u32 {
	let x = x as u32;
	let x = (x | (x << 8)) & 0x00ff00ff;
	let x = (x | (x << 4)) & 0x0f0f0f0f;
	let x = (x | (x << 2)) & 0x33333333;
	let x = (x | (x << 1)) & 0x55555555;

	let y = y as u32;
	let y = (y | (y << 8)) & 0x00ff00ff;
	let y = (y | (y << 4)) & 0x0f0f0f0f;
	let y = (y | (y << 2)) & 0x33333333;
	let y = (y | (y << 1)) & 0x55555555;

	x | (y << 1)
}

/// This function is only included since it is part of the paper.
/// It is not used in the implementation, but we have a test that ensures that is actually matches the optimized
/// version. Clarity is the most important thing in this code.
pub fn interleave_morton_naive(x: u16, y: u16) -> u32 {
	let mut result: u32 = 0;
	let x = x as u32;
	let y = y as u32;
	for i in 0..16 {
		// get the i-th bit of x and y
		let x_bit = x & (1 << i);
		let y_bit = y & (1 << i);

		// fill result from low to high bit
		result |= (x_bit) << i;
		result |= (y_bit) << (i + 1);
	}
	result
}

/// Convert Morton z-order value to 2D spatial coordinates
///
/// Uses bithacks as described in:
/// http://stackoverflow.com/questions/4909263/how-to-efficiently-de-interleave-bits-inverse-morton
#[inline]
pub fn deinterleave_morton(z: u32) -> (u16, u16) {
	let x = z & 0x55555555;
	let x = (x | (x >> 1)) & 0x33333333;
	let x = (x | (x >> 2)) & 0x0f0f0f0f;
	let x = (x | (x >> 4)) & 0x00ff00ff;
	let x = ((x | (x >> 8)) & 0x0000ffff) as u16;

	let y = (z >> 1) & 0x55555555;
	let y = (y | (y >> 1)) & 0x33333333;
	let y = (y | (y >> 2)) & 0x0f0f0f0f;
	let y = (y | (y >> 4)) & 0x00ff00ff;
	let y = ((y | (y >> 8)) & 0x0000ffff) as u16;

	(x, y)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_morton() {
		let x = 0x1234;
		let y = 0x5678;
		let z = interleave_morton(x, y);
		let (x2, y2) = deinterleave_morton(z);
		assert_eq!(x, x2);
		assert_eq!(y, y2);
	}
}
