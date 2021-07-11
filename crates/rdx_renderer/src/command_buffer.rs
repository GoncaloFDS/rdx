use crate::acceleration_structures::{AccelerationStructureGeometry, AccelerationStructureLevel};
use crate::buffer::{BufferRegion, DeviceAddress};
use crate::device::Device;
use crate::encoder::Command;
use crate::render_pass::{ClearValue, DEFAULT_ATTACHMENT_COUNT};
use crate::util::ToErupt;
use erupt::vk;
use smallvec::SmallVec;

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    queue_family_index: u32,
    recording: bool,
}

impl CommandBuffer {
    pub fn new(handle: vk::CommandBuffer, queue_family_index: u32) -> Self {
        CommandBuffer {
            handle,
            queue_family_index,
            recording: false,
        }
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub fn write(&mut self, device: &Device, commands: &[Command<'_>]) {
        let device = device.handle();
        if !self.recording {
            unsafe {
                device
                    .begin_command_buffer(
                        self.handle,
                        &vk::CommandBufferBeginInfoBuilder::new()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap()
            }
            self.recording = true;
        }

        for command in commands {
            match *command {
                Command::BeginRenderPass {
                    render_pass,
                    framebuffer,
                    clears,
                } => unsafe {
                    let mut clears = clears.into_iter();
                    let clear_values = render_pass
                        .info()
                        .attachments
                        .iter()
                        .map(|attachment| {
                            let clear = clears.next().unwrap();
                            match *clear {
                                ClearValue::Color(r, g, b, a) => vk::ClearValue {
                                    color: vk::ClearColorValue {
                                        float32: [r, g, b, a],
                                    },
                                },
                                ClearValue::DepthStencil(depth, stencil) => vk::ClearValue {
                                    depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
                                },
                            }
                        })
                        .collect::<SmallVec<[_; DEFAULT_ATTACHMENT_COUNT]>>();

                    device.cmd_begin_render_pass(
                        self.handle,
                        &vk::RenderPassBeginInfoBuilder::new()
                            .render_pass(render_pass.handle())
                            .framebuffer(framebuffer.handle())
                            .render_area(vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent: framebuffer.info().extent,
                            })
                            .clear_values(&clear_values),
                        vk::SubpassContents::INLINE,
                    )
                },
                Command::EndRenderPass => unsafe { device.cmd_end_render_pass(self.handle) },
                Command::BindGraphicsPipeline { pipeline } => unsafe {
                    device.cmd_bind_pipeline(
                        self.handle,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.handle(),
                    )
                },
                Command::BindRayTracingPipeline { pipeline } => unsafe {
                    device.cmd_bind_pipeline(
                        self.handle,
                        vk::PipelineBindPoint::RAY_TRACING_KHR,
                        pipeline.handle(),
                    )
                },
                Command::BindDescriptorSets {
                    bind_point,
                    layout,
                    first_set,
                    descriptor_sets,
                    dynamic_offsets,
                } => unsafe {
                    device.cmd_bind_descriptor_sets(
                        self.handle,
                        bind_point,
                        layout.handle(),
                        first_set,
                        &descriptor_sets
                            .iter()
                            .map(|set| set.handle())
                            .collect::<Vec<_>>(),
                        dynamic_offsets,
                    )
                },
                Command::SetViewport { viewport } => unsafe {
                    device.cmd_set_viewport(self.handle, 0, &[viewport.into_builder()])
                },
                Command::SetScissor { scissor } => unsafe {
                    device.cmd_set_scissor(self.handle, 0, &[scissor.into_builder()])
                },
                Command::Draw {
                    ref vertices,
                    ref instances,
                } => unsafe {
                    device.cmd_draw(
                        self.handle,
                        vertices.end - vertices.start,
                        instances.end - instances.start,
                        vertices.start,
                        instances.start,
                    )
                },
                Command::DrawIndexed {
                    ref indices,
                    vertex_offset,
                    ref instances,
                } => unsafe {
                    device.cmd_draw_indexed(
                        self.handle,
                        indices.end - indices.start,
                        instances.end - indices.start,
                        indices.start,
                        vertex_offset,
                        instances.start,
                    )
                },
                Command::UpdateBuffer {
                    buffer,
                    offset,
                    data,
                } => unsafe {
                    device.cmd_update_buffer(
                        self.handle,
                        buffer.handle(),
                        offset,
                        data.len() as _,
                        data.as_ptr() as _,
                    )
                },
                Command::BindVertexBuffers { first, buffers } => unsafe {
                    let (buffers, offsets): (Vec<_>, Vec<_>) = buffers
                        .iter()
                        .map(|(buffer, offset)| (buffer.handle(), offset))
                        .unzip();
                    device.cmd_bind_vertex_buffers(self.handle, first, &buffers, &offsets)
                },
                Command::BindIndexBuffer {
                    buffer,
                    offset,
                    index_type,
                } => unsafe {
                    device.cmd_bind_index_buffer(self.handle, buffer.handle(), offset, index_type)
                },
                Command::BuildAccelerationStructure { infos } => unsafe {
                    tracing::debug!("building blas");
                    match infos[0].dst.info().level {
                        AccelerationStructureLevel::Bottom => {
                            tracing::debug!("building blas")
                        }
                        AccelerationStructureLevel::Top => tracing::debug!("building tlas"),
                    }
                    let mut geometries = vec![];
                    let mut offsets = vec![];

                    let ranges: Vec<_> = infos.iter().map(|info| {
                        let mut total_primitive_count = 0u64;
                        let offset = geometries.len();

                        for geometry in info.geometries {
                            match geometry {
                                AccelerationStructureGeometry::Triangles {
                                    flags,
                                    vertex_format,
                                    vertex_data,
                                    vertex_stride,
                                    vertex_count,
                                    first_vertex,
                                    primitive_count,
                                    index_data,
                                    transform_data,
                                } => {
                                    total_primitive_count += (*primitive_count) as u64;
                                    geometries.push(vk::AccelerationStructureGeometryKHRBuilder::new()
                                        .flags(flags.clone())
                                        .geometry_type(vk::GeometryTypeKHR::TRIANGLES_KHR)
                                        .geometry(vk::AccelerationStructureGeometryDataKHR {
                                            triangles: vk::AccelerationStructureGeometryTrianglesDataKHRBuilder::new()
                                                .vertex_format(vertex_format.clone())
                                                .vertex_data(vertex_data.to_erupt())
                                                .vertex_stride(*vertex_stride)
                                                .max_vertex(*vertex_count)
                                                .index_type(vk::IndexType::UINT16)
                                                .index_data(match index_data {
                                                    None => vk::DeviceOrHostAddressConstKHR::default(),
                                                    Some(address) => address.to_erupt(),
                                                })
                                                .transform_data(transform_data.as_ref().map(|device_address| device_address.to_erupt()).unwrap_or_default())
                                                .build()
                                        }));

                                    offsets.push(vk::AccelerationStructureBuildRangeInfoKHRBuilder::new()
                                        .primitive_count(*primitive_count)
                                        .first_vertex(*first_vertex)
                                    );
                                }
                                AccelerationStructureGeometry::Instances { flags, data, primitive_count } => {
                                    geometries.push(vk::AccelerationStructureGeometryKHRBuilder::new()
                                        .flags(flags.clone())
                                        .geometry_type(vk::GeometryTypeKHR::INSTANCES_KHR)
                                        .geometry(vk::AccelerationStructureGeometryDataKHR {
                                            instances: vk::AccelerationStructureGeometryInstancesDataKHRBuilder::new()
                                                .data(data.to_erupt())
                                                .build()
                                        }));

                                    offsets.push(vk::AccelerationStructureBuildRangeInfoKHRBuilder::new()
                                        .primitive_count(*primitive_count)
                                    );
                                }
                            }
                        }

                        offset .. geometries.len()
                    }).collect();

                    let build_infos: SmallVec<[_; 32]> = infos
                        .iter()
                        .zip(&ranges)
                        .map(|(info, range)| {
                            let src = info
                                .src
                                .as_ref()
                                .map(|src| src.handle())
                                .unwrap_or_default();

                            vk::AccelerationStructureBuildGeometryInfoKHRBuilder::new()
                                ._type(info.dst.info().level.to_erupt())
                                .flags(info.flags.clone())
                                .mode(if info.src.is_some() {
                                    vk::BuildAccelerationStructureModeKHR::UPDATE_KHR
                                } else {
                                    vk::BuildAccelerationStructureModeKHR::BUILD_KHR
                                })
                                .src_acceleration_structure(src)
                                .dst_acceleration_structure(info.dst.handle())
                                .scratch_data(info.scratch.to_erupt())
                                .geometries(&geometries[range.clone()])
                        })
                        .collect();

                    let build_offsets: SmallVec<[_; 32]> = ranges
                        .into_iter()
                        .map(|range| &*offsets[range][0] as *const _)
                        .collect();

                    unsafe {
                        device.cmd_build_acceleration_structures_khr(
                            self.handle,
                            &build_infos,
                            &build_offsets,
                        )
                    }
                },
                Command::TraceRays {
                    shader_binding_table,
                    extent,
                } => unsafe {
                    let to_erupt = |buffer_region: &BufferRegion| {
                        let device_address = buffer_region.buffer.device_address().unwrap().0.get();

                        vk::StridedDeviceAddressRegionKHRBuilder::new()
                            .device_address(device_address + buffer_region.offset)
                            .stride(buffer_region.stride.unwrap())
                            .size(buffer_region.size)
                            .build()
                    };
                    device.cmd_trace_rays_khr(
                        self.handle,
                        &shader_binding_table
                            .raygen
                            .as_ref()
                            .map_or(vk::StridedDeviceAddressRegionKHR::default(), to_erupt),
                        &shader_binding_table
                            .miss
                            .as_ref()
                            .map_or(vk::StridedDeviceAddressRegionKHR::default(), to_erupt),
                        &shader_binding_table
                            .hit
                            .as_ref()
                            .map_or(vk::StridedDeviceAddressRegionKHR::default(), to_erupt),
                        &shader_binding_table
                            .callable
                            .as_ref()
                            .map_or(vk::StridedDeviceAddressRegionKHR::default(), to_erupt),
                        extent.width,
                        extent.height,
                        1,
                    );
                },
                Command::PipelineBarrier {
                    src,
                    dst,
                    src_access_mask,
                    dst_access_mask,
                    image_barriers,
                } => unsafe {
                    device.cmd_pipeline_barrier(
                        self.handle,
                        src,
                        dst,
                        None,
                        &[vk::MemoryBarrierBuilder::new()
                            .src_access_mask(src_access_mask)
                            .dst_access_mask(dst_access_mask)],
                        &[],
                        &image_barriers
                            .iter()
                            .map(|image_barrier| {
                                vk::ImageMemoryBarrierBuilder::new()
                                    .image(image_barrier.image.handle())
                                    .src_access_mask(src_access_mask)
                                    .dst_access_mask(src_access_mask)
                                    .old_layout(
                                        image_barrier
                                            .old_layout
                                            .unwrap_or(vk::ImageLayout::UNDEFINED),
                                    )
                                    .new_layout(image_barrier.new_layout)
                                    .src_queue_family_index(
                                        image_barrier
                                            .family_transfer
                                            .as_ref()
                                            .map(|range| range.start)
                                            .unwrap_or(vk::QUEUE_FAMILY_IGNORED),
                                    )
                                    .dst_queue_family_index(
                                        image_barrier
                                            .family_transfer
                                            .as_ref()
                                            .map(|range| range.end)
                                            .unwrap_or(vk::QUEUE_FAMILY_IGNORED),
                                    )
                                    .subresource_range(image_barrier.subresource.to_erupt())
                            })
                            .collect::<Vec<_>>(),
                    )
                },
            }
        }

        unsafe {
            device.end_command_buffer(self.handle).unwrap();
        }
    }
}
