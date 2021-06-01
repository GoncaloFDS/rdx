use crate::buffer_resource::BufferResource;
use erupt::{vk, DeviceLoader};

pub struct BottomLevelAccelerationStructure {
    acceleration_structure: vk::AccelerationStructureKHR,
    build_geometry_info: vk::AccelerationStructureBuildGeometryInfoKHR,
    pub build_sizes_info: vk::AccelerationStructureBuildSizesInfoKHR,
}

impl BottomLevelAccelerationStructure {
    pub fn new(
        device: &DeviceLoader,
        geometries: &[vk::AccelerationStructureGeometryKHRBuilder],
        acc_build_range_info: &[vk::AccelerationStructureBuildRangeInfoKHRBuilder],
    ) -> Self {
        let build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
            ._type(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL_KHR)
            .geometries(geometries)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD_KHR)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD_KHR)
            .build();

        let max_prim_count: Vec<_> = acc_build_range_info
            .iter()
            .map(|build_offset_info| build_offset_info.primitive_count)
            .collect();

        let build_sizes_info = vk::AccelerationStructureBuildSizesInfoKHRBuilder::default().build();
        let build_sizes_info = unsafe {
            device.get_acceleration_structure_build_sizes_khr(
                vk::AccelerationStructureBuildTypeKHR::DEVICE_KHR,
                &build_geometry_info,
                max_prim_count.as_slice(),
                Some(build_sizes_info),
            )
        };

        BottomLevelAccelerationStructure {
            acceleration_structure: vk::AccelerationStructureKHR::null(),
            build_geometry_info,
            build_sizes_info,
        }
    }

    pub fn build(
        &mut self,
        device: &DeviceLoader,
        command_buffer: vk::CommandBuffer,
        buffer: BufferResource,
        buffer_offset: vk::DeviceSize,
        scratch_buffer: BufferResource,
        scratch_buffer_offset: vk::DeviceSize,
    ) {
        self.acceleration_structure = unsafe {
            device
                .create_acceleration_structure_khr(
                    &vk::AccelerationStructureCreateInfoKHRBuilder::new()
                        ._type(self.build_geometry_info._type)
                        .size(self.build_sizes_info.acceleration_structure_size)
                        .buffer(buffer.buffer)
                        .offset(buffer_offset),
                    None,
                    Some(self.acceleration_structure),
                )
                .unwrap()
        };

        // unsafe {
        //     device.cmd_build_acceleration_structures_khr(
        //         command_buffer,
        //         &[self.build_geometry_info.into_builder()],
        //         &[self.build_sizes_info],
        //     )
        // }
    }
}

pub fn get_memory_requirements(
    acceleration_structures: &[BottomLevelAccelerationStructure],
) -> vk::AccelerationStructureBuildSizesInfoKHR {
    let mut acceleration_structure_size = 0;
    let mut build_scratch_size = 0;
    let mut update_scrath_size = 0;
    for acceleration_structure in acceleration_structures {
        acceleration_structure_size += acceleration_structure
            .build_sizes_info
            .acceleration_structure_size;
        build_scratch_size += acceleration_structure.build_sizes_info.build_scratch_size;
        update_scrath_size += acceleration_structure.build_sizes_info.update_scratch_size;
    }
    vk::AccelerationStructureBuildSizesInfoKHRBuilder::new()
        .acceleration_structure_size(acceleration_structure_size)
        .build_scratch_size(build_scratch_size)
        .update_scratch_size(update_scrath_size)
        .build()
}
