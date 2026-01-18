use bytemuck::{
	Pod,
	Zeroable,
};
use glam::{
	UVec2,
	UVec3,
	Vec3Swizzles,
	Vec4,
	Vec4Swizzles,
};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;
use spirv_std::{
	spirv,
	Image,
};
use static_assertions::const_assert;

use crate::thesis::{
	env_map::env_map_get_pnt,
	math::viridis,
};

/// Toogles log scale.
pub const FLAG_LOG_SCALE: u64 = 1 << 0;

const_assert!(core::mem::size_of::<ProcessingOptions>() <= 128);
#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct ProcessingOptions {
	/// General purpose flags.
	pub flags: u64,
	pub scale: f32,
	pub offset: f32,
	pub env_map_resolution: UVec2,
	pub env_map_ghost: f32,
	pub mask: u32,
}

impl Default for ProcessingOptions {
	fn default() -> Self {
		Self {
			flags: 0,
			scale: 1.0,
			offset: 0.0,
			env_map_resolution: UVec2::new(1, 1),
			env_map_ghost: 0.0,
			mask: 0,
		}
	}
}

#[spirv(compute(threads(4, 4)))]
pub fn post_process(
	#[spirv(global_invocation_id)] id: UVec3,
	#[spirv(push_constant)] constants: &ProcessingOptions,
	#[spirv(descriptor_set = 0, binding = 0)] input: &Image!(2D, format = rgba32f, sampled = false),
	#[spirv(descriptor_set = 0, binding = 1)] output: &Image!(2D, format = rgba32f, sampled = false),
	#[spirv(storage_buffer, descriptor_set = 0, binding = 2)] env_map: &[f32],
) {
	let pnt = id.xy().as_vec2() / constants.env_map_resolution.as_vec2();
	let env = env_map_get_pnt(env_map, constants.env_map_resolution, pnt) * 0.25f32;

	let mut input_color = input.read(id.xy()).xyz();
	// if log is enabled, we apply scale to log output, otherwise we apply scale to linear output
	input_color = if constants.flags & FLAG_LOG_SCALE == 0 {
		// log disabled
		input_color * constants.scale + constants.offset
	} else {
		// transform is applied later
		input_color
	};

	// if mask has any bit set, use bit to select channels which are exclusively written to all channels
	let mut output_color = Vec4::new(input_color.x, input_color.y, input_color.z, 1.0);
	if constants.mask != 0 {
		let mut masked_color = 0.0;
		for i in 0..3 {
			if constants.mask & (1 << i) != 0 {
				masked_color += input_color[i];
				// masked_color = 2.0;
			}
		}

		// check log scale flag
		if constants.flags & FLAG_LOG_SCALE != 0 {
			// avoid log(0) by adding 1.0
			masked_color = masked_color.max(0.0) + 1.0;
			masked_color = masked_color.log(10.0);

			masked_color = masked_color * constants.scale + constants.offset;
		}

		// debug test: color from 0 to 1 from left to right (override everything else, we remove this lader TODO, FIXME)
		// let masked_color = pnt.x;
		let viridis_color = viridis(masked_color);

		output_color = Vec4::new(viridis_color.x, viridis_color.y, viridis_color.z, 1.0);
	}

	output_color += env * constants.env_map_ghost;

	unsafe {
		output.write(id.xy(), output_color);
	}
}
