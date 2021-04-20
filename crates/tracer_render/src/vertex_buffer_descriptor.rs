use std::{
    borrow::Cow,
    hash::{Hash, Hasher},
};

use bevy::reflect::{Reflect, ReflectDeserialize};
use serde::{Deserialize, Serialize};

use crate::vertex_format::VertexFormat;

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Reflect, Serialize, Deserialize)]
pub enum Attribute {
    Color,
    Normal,
    Tangent,
    Position,
    Uv,
    Multiple,
}

#[derive(Clone, Debug, Eq, PartialEq, Reflect, Serialize, Deserialize)]
#[reflect_value(Serialize, Deserialize, PartialEq)]
pub struct VertexBufferLayout {
    pub attribute: Attribute,
    pub stride: u64,
    pub step_mode: InputStepMode,
    pub attributes: Vec<VertexAttribute>,
}

impl VertexBufferLayout {
    pub fn new_from_attribute(
        vertex_attribute: VertexAttribute,
        step_mode: InputStepMode,
    ) -> VertexBufferLayout {
        VertexBufferLayout {
            attribute: vertex_attribute.attribute.clone(),
            stride: vertex_attribute.format.get_size(),
            step_mode,
            attributes: vec![vertex_attribute],
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum InputStepMode {
    Vertex = 0,
    Instance = 1,
}

impl Default for InputStepMode {
    fn default() -> Self {
        InputStepMode::Vertex
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VertexAttribute {
    pub attribute: Attribute,
    pub format: VertexFormat,
    pub offset: u64,
    pub shader_location: u32,
}
