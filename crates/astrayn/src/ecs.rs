use std::ops::{
	Deref,
	DerefMut,
};

use bevy_ecs::prelude::Res;

pub type ResWrap<'a, T> = Res<'a, AnyRes<T>>;

/// A wrapper struct that allows compatible types to be used as ECS resources. This is useful for foreign types where
/// deriving `Resource` is not possible. Care must be taken to ensure that types are still unique within the ECS world.
pub struct AnyRes<T>(T);
impl<T: Send + Sync + 'static> bevy_ecs::prelude::Resource for AnyRes<T> {}

impl<T> Deref for AnyRes<T> {
	type Target = T;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl<T> DerefMut for AnyRes<T> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}

impl<T> AnyRes<T> {
	/// Create a new `ResWrap` from the given value.
	pub fn new(value: T) -> Self {
		Self(value)
	}
}
