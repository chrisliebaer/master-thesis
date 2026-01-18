use glam::{
	Vec2,
	Vec3,
	Vec4,
};

pub mod debug_image_pp;
pub mod debug_shader;
pub mod sample_shader;

#[allow(dead_code)]
pub fn validate_vector2(v: Vec2) {
	validate_scalar(v.x);
	validate_scalar(v.y);
}

pub fn validate_vector3(v: Vec3) {
	validate_scalar(v.x);
	validate_scalar(v.y);
	validate_scalar(v.z);
}

#[allow(dead_code)]
pub fn validate_vector4(v: Vec4) {
	validate_scalar(v.x);
	validate_scalar(v.y);
	validate_scalar(v.z);
	validate_scalar(v.w);
}

pub fn validate_scalar(s: f32) {
	if s.is_nan() {
		panic!("NaN scalar");
	}

	if s.is_infinite() {
		panic!("Infinite scalar");
	}
}
