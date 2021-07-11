use crate::resources::ShaderModule;
use erupt::{vk, DeviceLoader};
use std::env;
use std::fs::File;
use std::io::*;
use std::path::Path;

#[derive(Clone)]
pub struct Shader {
    pub module: ShaderModule,
    pub entry: Box<str>,
    pub stage: vk::ShaderStageFlagBits,
}

impl Shader {
    pub fn new(module: ShaderModule, stage: vk::ShaderStageFlagBits) -> Self {
        Shader {
            module,
            entry: "main".into(),
            stage,
        }
    }
}

#[derive(Clone)]
pub struct ShaderModuleInfo {
    pub code: Box<[u8]>,
}

impl ShaderModuleInfo {
    pub fn new(file: &str) -> Self {
        let path = env::current_dir()
            .unwrap()
            .join("assets")
            .join("shaders")
            .join(file);
        tracing::debug!("reading shader {:?}", path);
        let mut shader_file =
            File::open(path).unwrap_or_else(|_| panic!("Failed to open {}", file));
        let mut bytes = Vec::new();
        shader_file.read_to_end(&mut bytes).unwrap();

        ShaderModuleInfo { code: bytes.into() }
    }
}
