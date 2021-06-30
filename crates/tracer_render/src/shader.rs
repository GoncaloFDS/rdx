use crate::resources::ShaderModule;
use erupt::{vk, DeviceLoader};

pub struct Shader {
    pub module: ShaderModule,
    pub entry: Box<str>,
    pub stage: vk::ShaderStageFlags,
}

pub struct ShaderModuleInfo {
    pub code: Box<u8>,
    pub language: ShaderLanguage,
}

pub enum ShaderLanguage {
    GLSL,
    SPIRV,
}

pub fn create_shader_module(device: &DeviceLoader, file: &str) -> vk::ShaderModule {
    use std::fs::File;
    use std::io::*;
    let mut shader_file = File::open(file).unwrap_or_else(|_| panic!("Failed to open {}", file));
    let mut bytes = Vec::new();
    shader_file.read_to_end(&mut bytes).unwrap();
    let spv = erupt::utils::decode_spv(&bytes).unwrap();
    let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&spv);
    unsafe { device.create_shader_module(&module_info, None) }.unwrap()
}
