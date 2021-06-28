use crate::resources::ShaderModule;
use erupt::vk;

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
