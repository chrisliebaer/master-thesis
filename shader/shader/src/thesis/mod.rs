pub mod entrypoints;
pub mod env_map;
pub mod math;
pub mod mixture;
pub mod mixture_ext;
pub mod mixture_tree;
pub mod morton;
pub mod pcg32_ext;
pub mod vmf_distribution;
pub mod vmf_grid;

/// Size of the texture used for the environment map.
pub const TEXTURE_SIZE: usize = 1024;

/// Maximum number of von Mises-Fisher distributions in a mixture.
pub const VMF_COUNT: usize = 4096;

#[cfg(test)]
mod tests {

	use pcg32::Pcg32;

	#[allow(unused_imports)]
	use super::*;
	use crate::thesis::pcg32_ext::Pcg32Ext;

	#[test]
	fn test_unique_random() {
		let mut pcg = Pcg32::new(0, 0);
		let mut pcg = Pcg32::new(pcg.gen() as u64, 0);
		let mut vec1 = Vec::new();
		for _ in 0..10 {
			let value = pcg.gen_f32();
			vec1.push(value);
		}

		let mut pcg = Pcg32::new(0, 1);
		let mut pcg = Pcg32::new(pcg.gen() as u64, 0);
		let mut vec2 = Vec::new();
		for _ in 0..10 {
			let value = pcg.gen_f32();
			vec2.push(value);
		}

		// print both vectors
		println!("Vec1 Vec2");
		for i in 0..10 {
			println!("{} {}", vec1[i], vec2[i]);
		}
		println!("Checking for duplicates...");

		// print interleaved
		for i in 0..10 {
			println!("{}: {} {}", i, vec1[i], vec2[i]);

			// ensure no vector contains the value of the other (regarless of order)
			assert!(!vec2.contains(&vec1[i]));
			assert!(!vec1.contains(&vec2[i]));
		}
	}
}
