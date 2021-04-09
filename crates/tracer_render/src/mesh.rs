use ash::vk;
use glam::vec3;
use memoffset::offset_of;

use crate::vk_types::AllocatedBuffer;

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub vertex_buffer: AllocatedBuffer,
}

impl Mesh {
    // pub fn load_from_obj(path: &str, allocator: &vk_mem::Allocator) -> Mesh {
    //     let (models, _materials) = tobj::load_obj(path, true).unwrap();
    //
    //     let mut vertices = vec![];
    //
    //     for m in models.iter() {
    //         let mesh = &m.mesh;
    //         for &i in mesh.indices.iter() {
    //             let i = i as usize;
    //             let p_x = mesh.positions[3 * i];
    //             let p_y = mesh.positions[3 * i + 1];
    //             let p_z = mesh.positions[3 * i + 2];
    //             let vertex = Vertex {
    //                 position: vec3(p_x, p_y, p_z),
    //                 normal: vec3(1.0, 0.0, 0.0),
    //                 color: vec3(0.0, 1.0, 0.0),
    //             };
    //             vertices.push(vertex);
    //         }
    //     }
    //
    //     let vertex_buffer = Self::upload_mesh_to_gpu(allocator, &vertices);
    //
    //     Mesh {
    //         vertices,
    //         vertex_buffer,
    //     }
    // }

    pub fn upload_mesh_to_gpu(
        allocator: &vk_mem::Allocator,
        vertices: &[Vertex],
    ) -> AllocatedBuffer {
        let buffer_size = (vertices.len() * std::mem::size_of::<Vertex>()) as vk::DeviceSize;
        let buffer = AllocatedBuffer::new(
            &allocator,
            buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk_mem::MemoryUsage::CpuToGpu,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let mapped_memory = allocator.map_memory(&buffer.allocation).unwrap() as *mut Vertex;
        unsafe {
            mapped_memory.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
        }
        allocator.unmap_memory(&buffer.allocation).unwrap();

        buffer
    }
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
}
