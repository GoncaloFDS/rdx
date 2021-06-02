use crate::buffer_resource::BufferResource;
use crate::commands::CommandPool;
use crate::vertex::Vertex;
use erupt::{vk, DeviceLoader};
use glam::Mat4;
use gpu_alloc::{GpuAllocator, UsageFlags};
use std::sync::{Arc, Mutex};

pub struct RaytracingBuilder {
    blas_container: Vec<BlasEntry>,
    tlas: Tlas,

    queue_index: u32,
    queue: vk::Queue,

    device: Arc<DeviceLoader>,
    allocator: Arc<Mutex<GpuAllocator<vk::DeviceMemory>>>,
}

impl RaytracingBuilder {
    pub fn new(
        device: Arc<DeviceLoader>,
        allocator: Arc<Mutex<GpuAllocator<vk::DeviceMemory>>>,
        queue_index: u32,
        queue: vk::Queue,
    ) -> Self {
        RaytracingBuilder {
            blas_container: vec![],
            tlas: Default::default(),
            queue_index,
            queue,
            device,
            allocator,
        }
    }

    pub fn build_blas(
        &mut self,
        inputs: Vec<BlasInput>,
        flags: vk::BuildAccelerationStructureFlagsKHR,
    ) {
        assert!(self.blas_container.is_empty());

        for input in inputs {
            self.blas_container.push(BlasEntry::new(input));
        }

        let mut build_infos: Vec<_> = self
            .blas_container
            .iter()
            .map(|entry| {
                let geometries: Vec<_> = entry
                    .input
                    .as_geometry
                    .iter()
                    .map(|geo| geo.into_builder())
                    .collect();
                vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
                    .flags(flags)
                    .geometries(&geometries)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD_KHR)
                    ._type(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL_KHR)
                    .build()
            })
            .collect();

        let mut max_scracth = 0;
        for (i, entry) in self.blas_container.iter_mut().enumerate() {
            let max_prim_count: Vec<_> = entry
                .input
                .as_build_offset_info
                .iter()
                .map(|info| info.primitive_count)
                .collect();

            let size_info = unsafe {
                self.device.get_acceleration_structure_build_sizes_khr(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE_KHR,
                    &build_infos[i],
                    &max_prim_count,
                    None,
                )
            };

            let blas_buffer = BufferResource::new(
                self.device.clone(),
                self.allocator.clone(),
                size_info.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                UsageFlags::FAST_DEVICE_ACCESS | UsageFlags::DEVICE_ADDRESS,
            );

            let create_info = vk::AccelerationStructureCreateInfoKHRBuilder::new()
                ._type(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL_KHR)
                .size(size_info.acceleration_structure_size)
                .buffer(blas_buffer.buffer);

            let acceleration_structure = unsafe {
                self.device
                    .create_acceleration_structure_khr(&create_info, None, None)
                    .unwrap()
            };
            entry.acceleration_structure = Some(AccelerationStructure {
                accel: acceleration_structure,
                buffer: blas_buffer,
            });
            build_infos[i].dst_acceleration_structure = acceleration_structure;
            max_scracth = size_info.build_scratch_size.max(max_scracth);
        }

        let scratch_buffer = BufferResource::new(
            self.device.clone(),
            self.allocator.clone(),
            max_scracth,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            UsageFlags::FAST_DEVICE_ACCESS | UsageFlags::DEVICE_ADDRESS,
        );

        let query_pool = unsafe {
            self.device
                .create_query_pool(
                    &vk::QueryPoolCreateInfoBuilder::new()
                        .query_count(self.blas_container.len() as _)
                        .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR),
                    None,
                    None,
                )
                .unwrap()
        };

        unsafe {
            self.device
                .reset_query_pool(query_pool, 0, self.blas_container.len() as u32);
        }

        let command_pool = CommandPool::new(
            self.device.clone(),
            self.queue,
            self.queue_index,
            vk::CommandPoolCreateFlags::TRANSIENT,
        );

        let command_buffers = command_pool.create_command_buffers(
            vk::CommandBufferLevel::PRIMARY,
            self.blas_container.len() as u32,
        );

        // Building the acceleration structures
        for (i, blas) in self.blas_container.iter().enumerate() {
            build_infos[i].scratch_data.device_address = scratch_buffer.get_device_address();

            unsafe {
                self.device
                    .begin_command_buffer(
                        command_buffers[i],
                        &vk::CommandBufferBeginInfoBuilder::new()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap();
            }

            let p_build_offset_info: Vec<_> = blas
                .input
                .as_build_offset_info
                .iter()
                .map(|offset| offset as *const vk::AccelerationStructureBuildRangeInfoKHR)
                .collect();

            unsafe {
                self.device.cmd_build_acceleration_structures_khr(
                    command_buffers[i],
                    &[build_infos[i].into_builder()],
                    &p_build_offset_info,
                )
            }

            let barrier = vk::MemoryBarrierBuilder::new()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);

            unsafe {
                self.device.cmd_pipeline_barrier(
                    command_buffers[i],
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                    None,
                    &[barrier],
                    &[],
                    &[],
                )
            }
        }
        command_pool.submit_and_wait(&command_buffers);
        command_pool.destroy();

        unsafe { self.device.destroy_query_pool(Some(query_pool), None) }
    }

    pub fn build_tlas(
        &mut self,
        instances: Vec<AccelerationStructureInstance>,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        update: bool,
    ) {
        assert!(self.tlas.acceleration_structure.is_none() || update);
        let command_pool = CommandPool::new(
            self.device.clone(),
            self.queue,
            self.queue_index,
            vk::CommandPoolCreateFlags::TRANSIENT,
        );

        let command_buffer =
            command_pool.create_command_buffers(vk::CommandBufferLevel::PRIMARY, 1)[0];
        unsafe {
            self.device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfoBuilder::new()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }

        self.tlas.flags = flags;

        let geometry_instances: Vec<_> = instances
            .iter()
            .map(|instance| instance.to_vulkan(&self.blas_container))
            .collect();

        let instance_desc_size =
            instances.len() * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>();

        if update {
            //delete instances buffer
            todo!();
        }

        let mut instances_buffer = BufferResource::new(
            self.device.clone(),
            self.allocator.clone(),
            instance_desc_size as _,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            UsageFlags::DEVICE_ADDRESS | UsageFlags::HOST_ACCESS,
        );
        instances_buffer.store(&geometry_instances);

        let memory_barrier = vk::MemoryBarrierBuilder::new()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR);
        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                None,
                &[memory_barrier],
                &[],
                &[],
            );
        }

        // ---

        let instances_vk = vk::AccelerationStructureGeometryInstancesDataKHRBuilder::new()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instances_buffer.get_device_address(),
            })
            .build();

        let tlas_geometry = vk::AccelerationStructureGeometryKHRBuilder::new()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES_KHR)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: instances_vk,
            });

        // find sizes
        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
            .flags(flags)
            .geometries(&[tlas_geometry])
            .mode(if update {
                vk::BuildAccelerationStructureModeKHR::UPDATE_KHR
            } else {
                vk::BuildAccelerationStructureModeKHR::BUILD_KHR
            })
            ._type(vk::AccelerationStructureTypeKHR::TOP_LEVEL_KHR)
            .src_acceleration_structure(vk::AccelerationStructureKHR::null())
            .build();

        let size_info = unsafe {
            self.device.get_acceleration_structure_build_sizes_khr(
                vk::AccelerationStructureBuildTypeKHR::DEVICE_KHR,
                &build_info,
                &[instances.len() as _],
                None,
            )
        };

        if !update {
            let tlas_buffer = BufferResource::new(
                self.device.clone(),
                self.allocator.clone(),
                size_info.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                UsageFlags::FAST_DEVICE_ACCESS | UsageFlags::DEVICE_ADDRESS,
            );

            let create_info = vk::AccelerationStructureCreateInfoKHRBuilder::new()
                ._type(vk::AccelerationStructureTypeKHR::TOP_LEVEL_KHR)
                .size(size_info.acceleration_structure_size)
                .buffer(tlas_buffer.buffer);

            let acceleration_structure = unsafe {
                self.device
                    .create_acceleration_structure_khr(&create_info, None, None)
                    .unwrap()
            };

            self.tlas.acceleration_structure = Some(AccelerationStructure {
                accel: acceleration_structure,
                buffer: tlas_buffer,
            })
        }

        // scratch memory
        let scratch_buffer = BufferResource::new(
            self.device.clone(),
            self.allocator.clone(),
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            UsageFlags::FAST_DEVICE_ACCESS | UsageFlags::DEVICE_ADDRESS,
        );

        build_info.src_acceleration_structure = if update {
            self.tlas.acceleration_structure.as_ref().unwrap().accel
        } else {
            vk::AccelerationStructureKHR::null()
        };
        build_info.dst_acceleration_structure =
            self.tlas.acceleration_structure.as_ref().unwrap().accel;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.get_device_address(),
        };

        let build_offset_info = vk::AccelerationStructureBuildRangeInfoKHRBuilder::new()
            .primitive_count(instances.len() as u32)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0)
            .build();

        unsafe {
            self.device.cmd_build_acceleration_structures_khr(
                command_buffer,
                &[build_info.into_builder()],
                &[&build_offset_info as *const vk::AccelerationStructureBuildRangeInfoKHR],
            )
        }
        command_pool.submit_and_wait(&[command_buffer]);
        command_pool.destroy();
    }

    pub fn destroy(&mut self) {
        tracing::info!("raytracing destroy not implemented");
    }
}

#[derive(Clone)]
pub struct BlasInput {
    pub as_geometry: Vec<vk::AccelerationStructureGeometryKHR>,
    pub as_build_offset_info: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
}

impl BlasInput {
    pub fn new(
        vertices: &[Vertex],
        vertex_buffer: &BufferResource,
        indices: &[u16],
        index_buffer: &BufferResource,
    ) -> Self {
        let max_primitive_count = indices.len() / 3;
        let vertex_stride = std::mem::size_of::<Vertex>();

        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHRBuilder::new()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vertex_buffer.get_device_address(),
            })
            .vertex_stride(vertex_stride as _)
            .index_type(vk::IndexType::UINT16)
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: index_buffer.get_device_address(),
            })
            .max_vertex(vertices.len() as _)
            .build();

        let as_geometry = vk::AccelerationStructureGeometryKHRBuilder::new()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES_KHR)
            .flags(vk::GeometryFlagsKHR::OPAQUE_KHR)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
            .build();

        let as_build_offset_info = vk::AccelerationStructureBuildRangeInfoKHRBuilder::new()
            .first_vertex(0)
            .primitive_count(max_primitive_count as _)
            .primitive_offset(0)
            .transform_offset(0)
            .build();

        BlasInput {
            as_geometry: vec![as_geometry],
            as_build_offset_info: vec![as_build_offset_info],
        }
    }
}

// safety: vk::AccelerationStructureGeometryKHR is a raw pointer
unsafe impl Send for BlasInput {}
unsafe impl Sync for BlasInput {}

pub struct BlasEntry {
    pub input: BlasInput,
    pub acceleration_structure: Option<AccelerationStructure>,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
}

impl BlasEntry {
    pub fn new(input: BlasInput) -> Self {
        BlasEntry {
            input,
            acceleration_structure: None,
            flags: Default::default(),
        }
    }
}

pub struct AccelerationStructure {
    accel: vk::AccelerationStructureKHR,
    buffer: BufferResource,
}

#[derive(Default)]
pub struct Tlas {
    pub acceleration_structure: Option<AccelerationStructure>,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
}

#[derive(Copy, Clone, Debug)]
pub struct AccelerationStructureInstance {
    pub blas_id: u32,
    pub instance_custom_id: u32,
    pub hit_group_id: u32,
    pub visibility_mask: u32,
    pub flags: vk::GeometryInstanceFlagsKHR,
    pub transform: Mat4,
}

impl AccelerationStructureInstance {
    pub fn to_vulkan(&self, blas_container: &[BlasEntry]) -> vk::AccelerationStructureInstanceKHR {
        let blas = &blas_container[self.blas_id as usize];
        let blas_address = blas
            .acceleration_structure
            .as_ref()
            .unwrap()
            .buffer
            .get_device_address();

        vk::AccelerationStructureInstanceKHRBuilder::new()
            .instance_custom_index(self.instance_custom_id)
            .mask(self.visibility_mask)
            .instance_shader_binding_table_record_offset(self.hit_group_id)
            .flags(self.flags)
            .acceleration_structure_reference(blas_address)
            // .transform()
            .build()
    }
}
