use crate::buffer_resource::BufferResource;
use crate::raytracing_builder::{AccelerationStructureInstance, BlasInput, RaytracingBuilder};
use crate::vertex::Vertex;
use erupt::{vk, DeviceLoader};
use glam::vec3;
use gpu_alloc::{GpuAllocator, UsageFlags};
use std::sync::{Arc, Mutex};

pub struct RaytracingContext {
    device: Arc<DeviceLoader>,
    allocator: Arc<Mutex<GpuAllocator<vk::DeviceMemory>>>,
    queue_index: u32,
    queue: vk::Queue,
    raytracing_builder: RaytracingBuilder,
}

impl RaytracingContext {
    pub fn new(
        device: Arc<DeviceLoader>,
        allocator: Arc<Mutex<GpuAllocator<vk::DeviceMemory>>>,
        queue_index: u32,
        queue: vk::Queue,
    ) -> Self {
        let raytracing_builder =
            RaytracingBuilder::new(device.clone(), allocator.clone(), queue_index, queue);
        RaytracingContext {
            device,
            allocator,
            queue_index,
            queue,
            raytracing_builder,
        }
    }

    pub fn create_bottom_level_as(&mut self) {
        let vertices = [
            Vertex {
                position: vec3(-0.5, -0.5, 0.0),
            },
            Vertex {
                position: vec3(0.0, 0.5, 0.0),
            },
            Vertex {
                position: vec3(0.5, -0.5, 0.0),
            },
        ];

        let vertex_count = vertices.len();
        let vertex_stride = std::mem::size_of::<Vertex>();
        let vertex_buffer_size = vertex_stride * vertex_count;
        let mut vertex_buffer = BufferResource::new(
            self.device.clone(),
            self.allocator.clone(),
            vertex_buffer_size as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            UsageFlags::DEVICE_ADDRESS | UsageFlags::HOST_ACCESS,
        );
        vertex_buffer.store(&vertices);

        let indices = [0u16, 1, 2];
        let index_count = indices.len();
        let index_buffer_size = std::mem::size_of::<u16>() * index_count;
        let mut index_buffer = BufferResource::new(
            self.device.clone(),
            self.allocator.clone(),
            index_buffer_size as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            UsageFlags::DEVICE_ADDRESS | UsageFlags::HOST_ACCESS,
        );
        index_buffer.store(&indices);

        let all_blas = vec![BlasInput::new(
            &vertices,
            &vertex_buffer,
            &indices,
            &index_buffer,
        )];

        self.raytracing_builder.build_blas(
            all_blas,
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD_KHR
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION_KHR,
        );
    }

    pub fn create_top_level_as(&mut self) {
        let mut instances = vec![];

        let instance = AccelerationStructureInstance {
            blas_id: 0,
            instance_custom_id: 0,
            hit_group_id: 0,
            visibility_mask: 0,
            flags: vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE_KHR,
            transform: Default::default(),
        };

        instances.push(instance);

        self.raytracing_builder.build_tlas(
            instances,
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE_KHR,
            false,
        )
    }

    pub fn destroy(&mut self) {
        self.raytracing_builder.destroy();
    }
}
