use crate::acceleration_structures::{
    AccelerationStructureBuildGeometryInfo, AccelerationStructureGeometry,
    AccelerationStructureGeometryInfo, AccelerationStructureInfo, AccelerationStructureInstance,
    AccelerationStructureLevel,
};
use crate::buffer::{BufferInfo, BufferRegion, DeviceAddress};
use crate::descriptor::{
    DescriptorSetInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutInfo, DescriptorType,
    Descriptors, WriteDescriptorSet,
};
use crate::image::{Image, ImageInfo, ImageMemoryBarrier, ImageViewInfo};
use crate::pipeline::{
    PipelineLayoutInfo, RayTracingPipelineInfo, RayTracingShaderGroupInfo, ShaderBindingTable,
    ShaderBindingTableInfo,
};
use crate::render_context::RenderContext;
use crate::renderer::Pass;
use crate::resources::{
    AccelerationStructure, Buffer, DescriptorSet, Fence, PipelineLayout, RayTracingPipeline,
    Semaphore,
};
use crate::shader::{Shader, ShaderModuleInfo};
use crate::util::align_up;
use bumpalo::Bump;
use erupt::vk;
use erupt::vk1_0::{Extent2D, PipelineStageFlags};
use glam::vec3;
use std::collections::HashMap;

const MAX_INSTANCE_COUNT: u32 = 2048;

pub struct RayTracingPass {
    pipeline_layout: PipelineLayout,
    pipeline: RayTracingPipeline,
    shader_binding_table: ShaderBindingTable,
    tlas: AccelerationStructure,
    scratch_buffer: Buffer,
    descriptor_set: DescriptorSet,
    instances_buffer: Buffer,
    output_image: Image,
}

pub struct Input<'a> {
    pub blases: &'a HashMap<u8, AccelerationStructure>,
}

pub struct Output {
    pub tlas: AccelerationStructure,
    pub output_image: Image,
}

impl<'a> Pass<'a> for RayTracingPass {
    type Input = Input<'a>;
    type Output = Output;

    fn draw(
        &mut self,
        input: Self::Input,
        frame: u64,
        wait: &[(PipelineStageFlags, Semaphore)],
        signal: &[Semaphore],
        fence: Option<&Fence>,
        render_context: &mut RenderContext,
        bump: &Bump,
    ) -> Self::Output {
        let mut encoder = render_context.queue.create_enconder();

        encoder.pipeline_barrier(
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
            &[],
        );

        let mut as_instances = vec![];

        let blas = input.blases.get(&0u8).unwrap();

        as_instances.push(AccelerationStructureInstance::new(blas.device_address()));

        let build_info = [AccelerationStructureBuildGeometryInfo {
            src: None,
            dst: self.tlas.clone(),
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD_KHR,
            geometries: &[AccelerationStructureGeometry::Instances {
                flags: vk::GeometryFlagsKHR::OPAQUE_KHR,
                data: self.instances_buffer.device_address().unwrap(),
                primitive_count: 1,
            }],
            scratch: self.scratch_buffer.device_address().unwrap(),
        }];

        encoder.build_acceleration_structure(&build_info);

        render_context.write_buffer(&mut self.instances_buffer, 0, &as_instances);

        let output_image_view = render_context.create_image_view(ImageViewInfo::new(
            self.output_image.clone(),
            vk::ImageAspectFlags::COLOR,
        ));

        let write_descriptor_sets = [WriteDescriptorSet {
            descriptor_set: &self.descriptor_set,
            binding: 1,
            element: 0,
            descriptors: Descriptors::StorageImage(&[(
                output_image_view.clone(),
                vk::ImageLayout::GENERAL,
            )]),
        }];
        render_context.update_descriptor_sets(&write_descriptor_sets, &[]);
        drop(write_descriptor_sets);

        encoder.bind_ray_tracing_pipeline(&self.pipeline);

        let descriptor_sets = [self.descriptor_set.clone()];
        encoder.bind_descriptor_sets(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            &self.pipeline_layout,
            0,
            &descriptor_sets,
            &[],
        );

        let image_barriers = [ImageMemoryBarrier::initialize_whole(
            &self.output_image,
            vk::ImageLayout::GENERAL,
        )];

        encoder.pipeline_barrier(
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags::MEMORY_WRITE,
            vk::AccessFlags::MEMORY_WRITE,
            &image_barriers,
        );

        encoder.trace_rays(&self.shader_binding_table, self.output_image.info().extent);

        let image_barriers = [ImageMemoryBarrier::transition_whole(
            &self.output_image,
            vk::ImageLayout::GENERAL..vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )];
        encoder.pipeline_barrier(
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::MEMORY_WRITE,
            vk::AccessFlags::MEMORY_WRITE,
            &image_barriers,
        );

        let command_buffer = encoder.finish(&render_context.device);

        render_context
            .queue
            .submit(command_buffer, wait, signal, fence);

        Output {
            tlas: self.tlas.clone(),
            output_image: self.output_image.clone(),
        }
    }
}

impl RayTracingPass {
    pub fn new(render_context: &RenderContext, extent: vk::Extent2D) -> Self {
        let descriptor_set_layout =
            render_context.create_descriptor_set_layout(DescriptorSetLayoutInfo {
                bindings: vec![
                    // TLAS
                    DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_type: DescriptorType::AccelerationStructure,
                        count: 1,
                        stages: vk::ShaderStageFlags::RAYGEN_KHR
                            | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                        flags: vk::DescriptorBindingFlags::empty(),
                    },
                    // Image
                    DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_type: DescriptorType::StorageImage,
                        count: 1,
                        stages: vk::ShaderStageFlags::RAYGEN_KHR,
                        flags: vk::DescriptorBindingFlags::empty(),
                    },
                ],
                flags: vk::DescriptorSetLayoutCreateFlags::empty(),
            });

        let pipeline_layout = render_context.create_pipeline_layout(PipelineLayoutInfo {
            sets: vec![descriptor_set_layout.clone()],
            push_constants: vec![],
        });

        let ray_gen_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("raytrace.rgen.spv")),
            vk::ShaderStageFlagBits::RAYGEN_KHR,
        );

        let miss_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("raytrace.rmiss.spv")),
            vk::ShaderStageFlagBits::MISS_KHR,
        );

        let closest_hit_shader = Shader::new(
            render_context.create_shader_module(ShaderModuleInfo::new("raytrace.rchit.spv")),
            vk::ShaderStageFlagBits::CLOSEST_HIT_KHR,
        );

        let pipeline = render_context.create_ray_tracing_pipeline(RayTracingPipelineInfo {
            shaders: vec![ray_gen_shader, miss_shader, closest_hit_shader],
            groups: vec![
                RayTracingShaderGroupInfo::Raygen { raygen: 0 },
                RayTracingShaderGroupInfo::Miss { miss: 1 },
                RayTracingShaderGroupInfo::Triangle {
                    any_hit: None,
                    closest_hit: Some(2),
                },
            ],
            max_recursion_depth: 2,
            layout: pipeline_layout.clone(),
        });

        let shader_binding_table = render_context.create_shader_binding_table(
            &pipeline,
            ShaderBindingTableInfo {
                raygen: Some(0),
                miss: &[1],
                hit: &[2],
                callable: &[],
            },
        );

        let tlas_build_sizes = render_context.get_acceleration_structure_build_sizes(
            AccelerationStructureLevel::Top,
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD_KHR,
            &[AccelerationStructureGeometryInfo::Instances {
                max_primitive_count: MAX_INSTANCE_COUNT,
            }],
        );

        let tlas_buffer = render_context.create_buffer(BufferInfo {
            align: 255,
            size: tlas_build_sizes.acceleration_structure_size,
            usage_flags: vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
            allocation_flags: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
        });

        let tlas = render_context.create_acceleration_structure(AccelerationStructureInfo {
            level: AccelerationStructureLevel::Top,
            region: BufferRegion::whole(tlas_buffer),
        });

        let scratch_buffer = render_context.create_buffer(BufferInfo {
            align: 255,
            size: tlas_build_sizes.build_scratch_size,
            usage_flags: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            allocation_flags: gpu_alloc::UsageFlags::DEVICE_ADDRESS,
        });

        let instances_buffer = render_context.create_buffer(BufferInfo {
            align: 255,
            size: std::mem::size_of::<[AccelerationStructureInstance; MAX_INSTANCE_COUNT as usize]>(
            ) as _,
            usage_flags: vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            allocation_flags: gpu_alloc::UsageFlags::DEVICE_ADDRESS
                | gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
                | gpu_alloc::UsageFlags::HOST_ACCESS,
        });

        let output_image = render_context.create_image(ImageInfo {
            extent,
            format: vk::Format::R32G32B32A32_SFLOAT,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlagBits::_1,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        });

        let output_image_view = render_context.create_image_view(ImageViewInfo::new(
            output_image.clone(),
            vk::ImageAspectFlags::COLOR,
        ));

        let descriptor_set = render_context.create_descriptor_set(DescriptorSetInfo {
            layout: descriptor_set_layout.clone(),
        });

        render_context.update_descriptor_sets(
            &[
                WriteDescriptorSet {
                    descriptor_set: &descriptor_set,
                    binding: 0,
                    element: 0,
                    descriptors: Descriptors::AccelerationStructure(std::slice::from_ref(&tlas)),
                },
                WriteDescriptorSet {
                    descriptor_set: &descriptor_set,
                    binding: 1,
                    element: 0,
                    descriptors: Descriptors::StorageImage(&[(
                        output_image_view,
                        vk::ImageLayout::GENERAL,
                    )]),
                },
            ],
            &[],
        );

        RayTracingPass {
            pipeline_layout,
            pipeline,
            shader_binding_table,
            tlas,
            scratch_buffer,
            descriptor_set,
            instances_buffer,
            output_image,
        }
    }
}
