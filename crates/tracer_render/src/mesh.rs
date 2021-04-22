use ash::vk;
use bevy::reflect::TypeUuid;
use glam::*;
use memoffset::*;

use crate::vk_types::AllocatedBuffer;

#[derive(TypeUuid)]
#[uuid = "8ecbac0f-f545-4473-ad43-e1f4243af51e"]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub vertex_buffer: AllocatedBuffer,
    pub index_buffer: AllocatedBuffer,
}

impl Mesh {
    pub fn load_from_obj(path: &str, allocator: &vk_mem::Allocator) -> Mesh {
        let (models, _) = tobj::load_obj(path, true).unwrap();

        let mesh = &models[0].mesh;
        let positions = mesh.positions.as_slice();
        let vertex_count = mesh.positions.len() / 3;

        let mut vertices = Vec::with_capacity(vertex_count);
        for i in 0..vertex_count {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            let vertex = Vertex {
                position: vec3(x, y, z),
                normal: Default::default(),
                color: Default::default(),
            };
            vertices.push(vertex);
        }

        let indices = mesh.indices.clone();

        let vertex_buffer = upload_data_to_buffer(allocator, &vertices, vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let index_buffer = upload_data_to_buffer(allocator, &indices, vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

        Mesh {
            vertices,
            indices,
            vertex_buffer,
            index_buffer,
        }
    }

    pub fn index_count(&self) -> u32 { self.indices.len() as u32 }
    pub fn vertex_count(&self) -> u32 { self.vertices.len() as u32 }
}

#[derive(Debug)]
pub struct Vertex {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub color: glam::Vec3,
}

impl Vertex {
    pub fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()]
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, position) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, normal) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, color) as u32)
                .build(),
        ]
    }

    pub fn stride() -> vk::DeviceSize {
        std::mem::size_of::<Self>() as vk::DeviceSize
    }
}

pub fn upload_data_to_buffer<T>(
    allocator: &vk_mem::Allocator,
    data: &[T],
    usage_flags: vk::BufferUsageFlags,
) -> AllocatedBuffer {
    let buffer_size = (data.len() * std::mem::size_of::<T>()) as vk::DeviceSize;
    let buffer = AllocatedBuffer::new(
        &allocator,
        buffer_size,
        usage_flags,
        vk_mem::MemoryUsage::CpuToGpu,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );
    let mapped_memory = allocator.map_memory(&buffer.allocation).unwrap() as *mut T;
    unsafe {
        mapped_memory.copy_from_nonoverlapping(data.as_ptr(), data.len());
    }
    allocator.unmap_memory(&buffer.allocation).unwrap();

    buffer
}

