use std::collections::{BTreeMap, HashMap, HashSet};
use std::collections::hash_map::RandomState;

use bevy::app::EventReader;
use bevy::asset::{AssetEvent, Assets, Handle};
use bevy::core::AsBytes;
use bevy::ecs::entity::Entity;
use bevy::ecs::system::{Local, Res};
use bevy::reflect::TypeUuid;
use bevy::utils::tracing::*;
use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::render_context::RenderContext;
use crate::vertex_buffer_descriptor::{Attribute, InputStepMode, VertexAttribute, VertexBufferLayout};
use crate::vertex_format::VertexFormat;

#[derive(Debug, TypeUuid, Clone)]
#[uuid = "8ecbac0f-f545-4473-ad43-e1f4243af51e"]
pub struct Mesh {
    primitive_topology: PrimitiveTopology,
    attributes: BTreeMap<Attribute, VertexAttributeValues>,
    indices: Option<Indices>,
}

impl Mesh {
    pub fn new(primitive_topology: PrimitiveTopology) -> Self {
        Mesh {
            primitive_topology,
            attributes: Default::default(),
            indices: None,
        }
    }

    pub fn primitive_topology(&self) -> PrimitiveTopology {
        self.primitive_topology
    }

    pub fn set_attribute(&mut self, attribute: Attribute, values: impl Into<VertexAttributeValues>) {
        let values: VertexAttributeValues = values.into();
        self.attributes.insert(attribute, values);
    }

    pub fn attribute(&self, attribute: Attribute) -> Option<&VertexAttributeValues> {
        self.attributes.get(&attribute)
    }

    pub fn attribute_mut(
        &mut self,
        attribute: Attribute,
    ) -> Option<&mut VertexAttributeValues> {
        self.attributes.get_mut(&attribute)
    }

    /// Indices describe how triangles are constructed out of the vertex attributes.
    /// They are only useful for the [`crate::pipeline::PrimitiveTopology`] variants that use
    /// triangles
    pub fn set_indices(&mut self, indices: Option<Indices>) {
        self.indices = indices;
    }

    pub fn indices(&self) -> Option<&Indices> {
        self.indices.as_ref()
    }

    pub fn indices_mut(&mut self) -> Option<&mut Indices> {
        self.indices.as_mut()
    }

    pub fn get_index_buffer_bytes(&self) -> Option<Vec<u8>> {
        self.indices.as_ref().map(|indices| match &indices {
            Indices::U16(indices) => indices.as_slice().as_bytes().to_vec(),
            Indices::U32(indices) => indices.as_slice().as_bytes().to_vec(),
        })
    }

    pub fn get_vertex_buffer_layout(&self) -> VertexBufferLayout {
        let mut attributes = Vec::new();
        let mut accumulated_offset = 0;
        for (attribute, attribute_values) in self.attributes.iter() {
            let vertex_format = VertexFormat::from(attribute_values);
            attributes.push(VertexAttribute {
                attribute: attribute.clone(),
                offset: accumulated_offset,
                format: vertex_format,
                shader_location: 0,
            });
            accumulated_offset += vertex_format.get_size();
        }

        VertexBufferLayout {
            attribute: Attribute::Multiple,
            stride: accumulated_offset,
            step_mode: InputStepMode::Vertex,
            attributes,
        }
    }

    pub fn count_vertices(&self) -> usize {
        let mut vertex_count: Option<usize> = None;
        for (attribute_name, attribute_data) in self.attributes.iter() {
            let attribute_len = attribute_data.len();
            if let Some(previous_vertex_count) = vertex_count {
                assert_eq!(previous_vertex_count, attribute_len,
                           "Attribute {:?} has a different vertex count ({}) than other attributes ({}) in this mesh.", attribute_name, attribute_len, previous_vertex_count);
            }
            vertex_count = Some(attribute_len);
        }

        vertex_count.unwrap_or(0)
    }

    pub fn get_vertex_buffer_data(&self) -> Vec<u8> {
        let mut vertex_size = 0;
        for attribute_values in self.attributes.values() {
            let vertex_format = VertexFormat::from(attribute_values);
            vertex_size += vertex_format.get_size() as usize;
        }

        let vertex_count = self.count_vertices();
        let mut attributes_interleaved_buffer = vec![0; vertex_count * vertex_size];
        // bundle into interleaved buffers
        let mut attribute_offset = 0;
        for attribute_values in self.attributes.values() {
            let vertex_format = VertexFormat::from(attribute_values);
            let attribute_size = vertex_format.get_size() as usize;
            let attributes_bytes = attribute_values.get_bytes();
            for (vertex_index, attribute_bytes) in
            attributes_bytes.chunks_exact(attribute_size).enumerate()
            {
                let offset = vertex_index * vertex_size + attribute_offset;
                attributes_interleaved_buffer[offset..offset + attribute_size]
                    .copy_from_slice(attribute_bytes);
            }

            attribute_offset += attribute_size;
        }

        attributes_interleaved_buffer
    }

    pub fn compute_flat_normals(&mut self) {
        if self.indices().is_some() {
            panic!("`compute_flat_normals` can't work on indexed geometry. Consider calling `Mesh::duplicate_vertices`.");
        }

        let positions = self
            .attribute(Attribute::Position)
            .unwrap()
            .as_float3()
            .expect("`Mesh::ATTRIBUTE_POSITION` vertex attributes should be of type `float3`");

        let normals: Vec<_> = positions
            .chunks_exact(3)
            .map(|p| face_normal(p[0], p[1], p[2]))
            .flat_map(|normal| std::array::IntoIter::new([normal, normal, normal]))
            .collect();

        self.set_attribute(Attribute::Normal, normals);
    }
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let (a, b, c) = (Vec3::from(a), Vec3::from(b), Vec3::from(c));
    (b - a).cross(c - a).normalize().into()
}

/// Describes how the VertexAttributes should be interpreted while rendering
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum PrimitiveTopology {
    PointList = 0,
    LineList = 1,
    LineStrip = 2,
    TriangleList = 3,
    TriangleStrip = 4,
}


/// An array where each entry describes a property of a single vertex.
#[derive(Clone, Debug)]
pub enum VertexAttributeValues {
    Float(Vec<f32>),
    Int(Vec<i32>),
    Uint(Vec<u32>),
    Float2(Vec<[f32; 2]>),
    Int2(Vec<[i32; 2]>),
    Uint2(Vec<[u32; 2]>),
    Float3(Vec<[f32; 3]>),
    Int3(Vec<[i32; 3]>),
    Uint3(Vec<[u32; 3]>),
    Float4(Vec<[f32; 4]>),
    Int4(Vec<[i32; 4]>),
    Uint4(Vec<[u32; 4]>),
    Short2(Vec<[i16; 2]>),
    Short2Norm(Vec<[i16; 2]>),
    Ushort2(Vec<[u16; 2]>),
    Ushort2Norm(Vec<[u16; 2]>),
    Short4(Vec<[i16; 4]>),
    Short4Norm(Vec<[i16; 4]>),
    Ushort4(Vec<[u16; 4]>),
    Ushort4Norm(Vec<[u16; 4]>),
    Char2(Vec<[i8; 2]>),
    Char2Norm(Vec<[i8; 2]>),
    Uchar2(Vec<[u8; 2]>),
    Uchar2Norm(Vec<[u8; 2]>),
    Char4(Vec<[i8; 4]>),
    Char4Norm(Vec<[i8; 4]>),
    Uchar4(Vec<[u8; 4]>),
    Uchar4Norm(Vec<[u8; 4]>),
}

impl VertexAttributeValues {
    /// Returns the number of vertices in this VertexAttribute. For a single
    /// mesh, all of the VertexAttributeValues must have the same length.
    pub fn len(&self) -> usize {
        match *self {
            VertexAttributeValues::Float(ref values) => values.len(),
            VertexAttributeValues::Int(ref values) => values.len(),
            VertexAttributeValues::Uint(ref values) => values.len(),
            VertexAttributeValues::Float2(ref values) => values.len(),
            VertexAttributeValues::Int2(ref values) => values.len(),
            VertexAttributeValues::Uint2(ref values) => values.len(),
            VertexAttributeValues::Float3(ref values) => values.len(),
            VertexAttributeValues::Int3(ref values) => values.len(),
            VertexAttributeValues::Uint3(ref values) => values.len(),
            VertexAttributeValues::Float4(ref values) => values.len(),
            VertexAttributeValues::Int4(ref values) => values.len(),
            VertexAttributeValues::Uint4(ref values) => values.len(),
            VertexAttributeValues::Short2(ref values) => values.len(),
            VertexAttributeValues::Short2Norm(ref values) => values.len(),
            VertexAttributeValues::Ushort2(ref values) => values.len(),
            VertexAttributeValues::Ushort2Norm(ref values) => values.len(),
            VertexAttributeValues::Short4(ref values) => values.len(),
            VertexAttributeValues::Short4Norm(ref values) => values.len(),
            VertexAttributeValues::Ushort4(ref values) => values.len(),
            VertexAttributeValues::Ushort4Norm(ref values) => values.len(),
            VertexAttributeValues::Char2(ref values) => values.len(),
            VertexAttributeValues::Char2Norm(ref values) => values.len(),
            VertexAttributeValues::Uchar2(ref values) => values.len(),
            VertexAttributeValues::Uchar2Norm(ref values) => values.len(),
            VertexAttributeValues::Char4(ref values) => values.len(),
            VertexAttributeValues::Char4Norm(ref values) => values.len(),
            VertexAttributeValues::Uchar4(ref values) => values.len(),
            VertexAttributeValues::Uchar4Norm(ref values) => values.len(),
        }
    }

    /// Returns `true` if there are no vertices in this VertexAttributeValue
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn as_float3(&self) -> Option<&[[f32; 3]]> {
        match self {
            VertexAttributeValues::Float3(values) => Some(values),
            _ => None,
        }
    }

    // TODO: add vertex format as parameter here and perform type conversions
    /// Flattens the VertexAttributeArray into a sequence of bytes. This is
    /// useful for serialization and sending to the GPU.
    pub fn get_bytes(&self) -> &[u8] {
        match self {
            VertexAttributeValues::Float(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Int(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Uint(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Float2(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Int2(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Uint2(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Float3(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Int3(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Uint3(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Float4(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Int4(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Uint4(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Short2(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Short2Norm(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Ushort2(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Ushort2Norm(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Short4(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Short4Norm(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Ushort4(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Ushort4Norm(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Char2(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Char2Norm(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Uchar2(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Uchar2Norm(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Char4(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Char4Norm(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Uchar4(values) => values.as_slice().as_bytes(),
            VertexAttributeValues::Uchar4Norm(values) => values.as_slice().as_bytes(),
        }
    }
}

impl From<Vec<f32>> for VertexAttributeValues {
    fn from(vec: Vec<f32>) -> Self {
        VertexAttributeValues::Float(vec)
    }
}

impl From<Vec<i32>> for VertexAttributeValues {
    fn from(vec: Vec<i32>) -> Self {
        VertexAttributeValues::Int(vec)
    }
}

impl From<Vec<u32>> for VertexAttributeValues {
    fn from(vec: Vec<u32>) -> Self {
        VertexAttributeValues::Uint(vec)
    }
}

impl From<Vec<[f32; 2]>> for VertexAttributeValues {
    fn from(vec: Vec<[f32; 2]>) -> Self {
        VertexAttributeValues::Float2(vec)
    }
}

impl From<Vec<[i32; 2]>> for VertexAttributeValues {
    fn from(vec: Vec<[i32; 2]>) -> Self {
        VertexAttributeValues::Int2(vec)
    }
}

impl From<Vec<[u32; 2]>> for VertexAttributeValues {
    fn from(vec: Vec<[u32; 2]>) -> Self {
        VertexAttributeValues::Uint2(vec)
    }
}

impl From<Vec<[f32; 3]>> for VertexAttributeValues {
    fn from(vec: Vec<[f32; 3]>) -> Self {
        VertexAttributeValues::Float3(vec)
    }
}

impl From<Vec<[i32; 3]>> for VertexAttributeValues {
    fn from(vec: Vec<[i32; 3]>) -> Self {
        VertexAttributeValues::Int3(vec)
    }
}

impl From<Vec<[u32; 3]>> for VertexAttributeValues {
    fn from(vec: Vec<[u32; 3]>) -> Self {
        VertexAttributeValues::Uint3(vec)
    }
}

impl From<Vec<[f32; 4]>> for VertexAttributeValues {
    fn from(vec: Vec<[f32; 4]>) -> Self {
        VertexAttributeValues::Float4(vec)
    }
}

impl From<Vec<[i32; 4]>> for VertexAttributeValues {
    fn from(vec: Vec<[i32; 4]>) -> Self {
        VertexAttributeValues::Int4(vec)
    }
}

impl From<Vec<[u32; 4]>> for VertexAttributeValues {
    fn from(vec: Vec<[u32; 4]>) -> Self {
        VertexAttributeValues::Uint4(vec)
    }
}

impl From<Vec<[u8; 4]>> for VertexAttributeValues {
    fn from(vec: Vec<[u8; 4]>) -> Self {
        VertexAttributeValues::Uchar4Norm(vec)
    }
}

impl From<&VertexAttributeValues> for VertexFormat {
    fn from(values: &VertexAttributeValues) -> Self {
        match values {
            VertexAttributeValues::Float(_) => VertexFormat::Float,
            VertexAttributeValues::Int(_) => VertexFormat::Int,
            VertexAttributeValues::Uint(_) => VertexFormat::Uint,
            VertexAttributeValues::Float2(_) => VertexFormat::Float2,
            VertexAttributeValues::Int2(_) => VertexFormat::Int2,
            VertexAttributeValues::Uint2(_) => VertexFormat::Uint2,
            VertexAttributeValues::Float3(_) => VertexFormat::Float3,
            VertexAttributeValues::Int3(_) => VertexFormat::Int3,
            VertexAttributeValues::Uint3(_) => VertexFormat::Uint3,
            VertexAttributeValues::Float4(_) => VertexFormat::Float4,
            VertexAttributeValues::Int4(_) => VertexFormat::Int4,
            VertexAttributeValues::Uint4(_) => VertexFormat::Uint4,
            VertexAttributeValues::Short2(_) => VertexFormat::Short2,
            VertexAttributeValues::Short2Norm(_) => VertexFormat::Short2Norm,
            VertexAttributeValues::Ushort2(_) => VertexFormat::Ushort2,
            VertexAttributeValues::Ushort2Norm(_) => VertexFormat::Ushort2Norm,
            VertexAttributeValues::Short4(_) => VertexFormat::Short4,
            VertexAttributeValues::Short4Norm(_) => VertexFormat::Short4Norm,
            VertexAttributeValues::Ushort4(_) => VertexFormat::Ushort4,
            VertexAttributeValues::Ushort4Norm(_) => VertexFormat::Ushort4Norm,
            VertexAttributeValues::Char2(_) => VertexFormat::Char2,
            VertexAttributeValues::Char2Norm(_) => VertexFormat::Char2Norm,
            VertexAttributeValues::Uchar2(_) => VertexFormat::Uchar2,
            VertexAttributeValues::Uchar2Norm(_) => VertexFormat::Uchar2Norm,
            VertexAttributeValues::Char4(_) => VertexFormat::Char4,
            VertexAttributeValues::Char4Norm(_) => VertexFormat::Char4Norm,
            VertexAttributeValues::Uchar4(_) => VertexFormat::Uchar4,
            VertexAttributeValues::Uchar4Norm(_) => VertexFormat::Uchar4Norm,
        }
    }
}

/// An array of indices into the VertexAttributeValues for a mesh.
///
/// It describes the order in which the vertex attributes should be joined into faces.
#[derive(Debug, Clone)]
pub enum Indices {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl Indices {
    fn iter(&self) -> impl Iterator<Item=usize> + '_ {
        match self {
            Indices::U16(vec) => IndicesIter::U16(vec.iter()),
            Indices::U32(vec) => IndicesIter::U32(vec.iter()),
        }
    }
}

enum IndicesIter<'a> {
    U16(std::slice::Iter<'a, u16>),
    U32(std::slice::Iter<'a, u32>),
}

impl Iterator for IndicesIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IndicesIter::U16(iter) => iter.next().map(|val| *val as usize),
            IndicesIter::U32(iter) => iter.next().map(|val| *val as usize),
        }
    }
}

#[derive(Default)]
pub struct MeshEntities {
    entities: HashSet<Entity>,
}

#[derive(Default)]
pub struct MeshResourceProviderState {
    entities: HashMap<Handle<Mesh>, MeshEntities>,
}

pub fn resource_provider_system(
    mut state: Local<MeshResourceProviderState>,
    meshes: Res<Assets<Mesh>>,
    mut mesh_events: EventReader<AssetEvent<Mesh>>,
    render_context: Res<RenderContext>,
) {
    let mut changed_meshes: HashSet<Handle<_>, RandomState> = HashSet::default();
    let render_context = &*render_context;
    for event in mesh_events.iter() {
        match event {
            AssetEvent::Created { ref handle } => {
                changed_meshes.insert(handle.clone_weak());
            }
            AssetEvent::Modified { ref handle } => {
                changed_meshes.insert(handle.clone_weak());
            }
            AssetEvent::Removed { ref handle } => {
                changed_meshes.remove(handle);
            }
        }
    }

    for (i, changed_mesh_handle) in changed_meshes.iter().enumerate() {
        if let Some(mesh) = meshes.get(changed_mesh_handle) {
            if let Some(data) = mesh.get_index_buffer_bytes() {
            }
        }
    }
}