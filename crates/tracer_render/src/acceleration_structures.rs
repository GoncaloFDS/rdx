use crate::buffer::{BufferRegion, DeviceAddress};
use crate::resources::AccelerationStructure;
use crevice::internal::bytemuck;
use erupt::vk;
use glam::Mat4;

#[derive(Clone)]
pub struct AccelerationStructureInfo {
    pub level: AccelerationStructureLevel,
    pub region: BufferRegion,
}

#[derive(Clone)]
pub struct AccelerationStructureBuildSizesInfo {
    pub acceleration_structure_size: u64,
    pub update_scratch_size: u64,
    pub build_scratch_size: u64,
}

#[derive(Clone)]
pub enum AccelerationStructureLevel {
    Bottom,
    Top,
}

#[derive(Clone)]
pub enum AccelerationStructureGeometryInfo {
    Triangles {
        max_primitive_count: u32,
        max_vertex_count: u32,
        vertex_format: vk::Format,
    },
}

#[derive(Clone)]
pub struct AccelerationStructureBuildGeometryInfo<'a> {
    pub src: Option<AccelerationStructure>,
    pub dst: AccelerationStructure,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    pub geometries: &'a [AccelerationStructureGeometry],
    pub scratch: DeviceAddress,
}

pub enum AccelerationStructureGeometry {
    Triangles {
        flags: vk::GeometryFlagsKHR,
        vertex_format: vk::Format,
        vertex_data: DeviceAddress,
        vertex_stride: u64,
        vertex_count: u32,
        first_vertex: u32,
        primitive_count: u32,
        index_data: Option<DeviceAddress>,
        transform_data: Option<DeviceAddress>,
    },
}

#[derive(Clone, Copy)]
#[repr(align(16))]
#[repr(C)]
pub struct AccelerationStructureInstance {
    pub transform: Mat4,
    pub acceleration_structure_reference: DeviceAddress,
}

unsafe impl bytemuck::Zeroable for AccelerationStructureInstance {}
unsafe impl bytemuck::Pod for AccelerationStructureInstance {}
