use spirv_builder::{
	Capability,
	MetadataPrintout,
	ShaderPanicStrategy,
	SpirvBuilder,
	SpirvMetadata,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	// luckily, the shader directory never contains any build files
	println!("cargo:rerun-if-changed=../shader");
	println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
	println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");

	SpirvBuilder::new("../shader", "spirv-unknown-vulkan1.2")
		.spirv_metadata(SpirvMetadata::None)
		//.extension("VK_EXT_shader_atomic_float"))
		//.extension("SPV_EXT_shader_atomic_float_add")
		.capability(Capability::Float64)
		.capability(Capability::Int64)
		.capability(Capability::Int16)
		.capability(Capability::Int8)
		.capability(Capability::VariablePointers)
		.release(true)
		//.deny_warnings(true)
		// if not set, vulkano might panic, if shader layout doesn't match
		.preserve_bindings(true)
		.shader_panic_strategy(ShaderPanicStrategy::DebugPrintfThenExit {
			print_inputs: true,
			print_backtrace: true,
		})
		// required since vulkano can't handle multiple entry points having different push constants
		// see: https://github.com/vulkano-rs/vulkano/pull/2405
		.multimodule(true)
		// full not allowed with multimodule
		.print_metadata(MetadataPrintout::None)
		.build()?;
	Ok(())
}
