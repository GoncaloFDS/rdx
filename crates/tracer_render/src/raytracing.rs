use crate::buffer_resource::{BufferResource, Texture};
use crate::raytracing_builder::{AccelerationStructureInstance, BlasInput, RaytracingBuilder};
use crate::renderer::create_shader_module;
use crate::vertex::Vertex;
use crevice::std430::{AsStd430, Std430};
use erupt::{vk, DeviceLoader, ExtendableFrom};
use glam::{vec3, Vec3, Vec4};
use gpu_alloc::{GpuAllocator, UsageFlags};
use gpu_alloc_erupt::EruptMemoryDevice;
use std::ffi::{CStr, CString};
use std::io::Read;
use std::sync::{Arc, Mutex};

// #[derive(Copy, Clone, AsStd430)]
pub struct PushConstants {
    clear_color: Vec4,
    light_position: Vec3,
    light_intensity: f32,
}

unsafe impl Send for RaytracingContext {}
unsafe impl Sync for RaytracingContext {}

pub struct RaytracingContext {
    device: Arc<DeviceLoader>,
    allocator: Arc<Mutex<GpuAllocator<vk::DeviceMemory>>>,
    queue_index: u32,
    queue: vk::Queue,
    raytracing_builder: RaytracingBuilder,
    descriptor_set_bindings: DescriptorSetBindings,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,

    offscreen_target: Texture,

    shader_groups: Vec<vk::RayTracingShaderGroupCreateInfoKHR>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    sbt_buffer: BufferResource,

    vertex_buffer: BufferResource,
    index_buffer: BufferResource,
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
            descriptor_set_bindings: Default::default(),
            descriptor_pool: Default::default(),
            descriptor_set_layout: Default::default(),
            descriptor_set: Default::default(),
            offscreen_target: Default::default(),
            shader_groups: vec![],
            pipeline_layout: Default::default(),
            pipeline: Default::default(),
            sbt_buffer: Default::default(),
            vertex_buffer: Default::default(),
            index_buffer: Default::default(),
        }
    }

    pub fn create_offscreen_render(&mut self) {
        let image_create_info = vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .extent(vk::Extent3D {
                width: 600,
                height: 800,
                depth: 1,
            })
            .samples(vk::SampleCountFlagBits::_1)
            .mip_levels(1)
            .array_layers(1)
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .tiling(vk::ImageTiling::OPTIMAL)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe {
            self.device
                .create_image(&image_create_info, None, None)
                .unwrap()
        };

        let image_requirements = unsafe { self.device.get_image_memory_requirements(image, None) };

        let memory_block = unsafe {
            self.allocator
                .lock()
                .unwrap()
                .alloc(
                    EruptMemoryDevice::wrap(&self.device),
                    gpu_alloc::Request {
                        size: image_requirements.size,
                        align_mask: image_requirements.alignment - 1,
                        usage: gpu_alloc::UsageFlags::empty(),
                        memory_types: image_requirements.memory_type_bits,
                    },
                )
                .expect("Failed to create Image memory block")
        };

        unsafe {
            self.device
                .bind_image_memory(image, *memory_block.memory(), memory_block.offset())
                .unwrap();
        }
        // create_image_view
        let view_create_info = vk::ImageViewCreateInfoBuilder::new()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(image_create_info.format)
            .subresource_range(
                vk::ImageSubresourceRangeBuilder::new()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );
        let image_view = unsafe {
            self.device
                .create_image_view(&view_create_info, None, None)
                .unwrap()
        };

        // create sampler
        let sampler_create_info = vk::SamplerCreateInfo::default();
        let sampler = unsafe {
            self.device
                .create_sampler(&sampler_create_info, None, None)
                .unwrap()
        };

        // create offscreen_target
        self.offscreen_target.image = image;
        self.offscreen_target.allocation = Some(memory_block);
        self.offscreen_target.descriptor.image_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        self.offscreen_target.descriptor.image_view = image_view;
        self.offscreen_target.descriptor.sampler = sampler;
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
        self.vertex_buffer = BufferResource::new(
            self.device.clone(),
            self.allocator.clone(),
            vertex_buffer_size as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            UsageFlags::DEVICE_ADDRESS | UsageFlags::HOST_ACCESS,
            "vertex",
        );
        self.vertex_buffer.store(&self.device, &vertices);

        let indices = [0u16, 1, 2];
        let index_count = indices.len();
        let index_buffer_size = std::mem::size_of::<u16>() * index_count;
        self.index_buffer = BufferResource::new(
            self.device.clone(),
            self.allocator.clone(),
            index_buffer_size as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            UsageFlags::DEVICE_ADDRESS | UsageFlags::HOST_ACCESS,
            "index",
        );
        self.index_buffer.store(&self.device, &indices);

        let all_blas = vec![BlasInput::new(
            &vertices,
            &self.vertex_buffer,
            &indices,
            &self.index_buffer,
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

    pub fn create_descriptor_set(&mut self) {
        self.descriptor_set_bindings.add_bindings(&[
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(0)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(
                    vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                )
                .build(),
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                .build(),
        ]);

        self.descriptor_pool = self.descriptor_set_bindings.create_pool(&self.device);
        self.descriptor_set_layout = self.descriptor_set_bindings.create_layout(&self.device);
        self.descriptor_set = unsafe {
            self.device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfoBuilder::new()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&[self.descriptor_set_layout]),
                )
                .unwrap()[0]
        };

        // make write as
        let acc_structures = [self.raytracing_builder.get_acceleration_structure()];
        let mut acc_write_desc = vk::WriteDescriptorSetBuilder::new()
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .dst_binding(0)
            .dst_set(self.descriptor_set)
            .dst_array_element(0);
        acc_write_desc.descriptor_count = 1;
        let mut acc_structure_write = vk::WriteDescriptorSetAccelerationStructureKHRBuilder::new()
            .acceleration_structures(&acc_structures);
        let acc_write_desc = acc_write_desc.extend_from(&mut *acc_structure_write);
        tracing::info!("{:#?}", acc_write_desc);

        // make write image

        let image_infos = [vk::DescriptorImageInfoBuilder::new()
            .image_view(self.offscreen_target.descriptor.image_view)
            .image_layout(vk::ImageLayout::GENERAL)
            .sampler(vk::Sampler::default())];

        let image_write_desc = vk::WriteDescriptorSetBuilder::new()
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(1)
            .dst_set(self.descriptor_set)
            .dst_array_element(0)
            .image_info(&image_infos);

        let writes = [acc_write_desc, image_write_desc];

        tracing::info!("{:#?}", writes);

        unsafe { self.device.update_descriptor_sets(&writes, &[]) }
    }

    pub fn create_raytracing_pipeline(&mut self) {
        let push_constants = [vk::PushConstantRangeBuilder::new()
            .offset(0)
            .size(std::mem::size_of::<PushConstants>() as u32)
            .stage_flags(
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::MISS_KHR,
            )];

        // only one set for now (raytracing)
        // Maybe add a different one that is shared with the rasterization pipeline
        let descriptor_set_layouts = [self.descriptor_set_layout];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .push_constant_ranges(&push_constants)
            .set_layouts(&descriptor_set_layouts);

        self.pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_create_info, None, None)
                .unwrap()
        };

        //
        let shader_entry_point = CString::new("main").unwrap();
        let pipeline_shader_stages = [
            vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::RAYGEN_KHR)
                .module(create_shader_module(
                    &self.device,
                    "assets/shaders/raytrace.rgen.spv",
                ))
                .name(&shader_entry_point),
            vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::MISS_KHR)
                .module(create_shader_module(
                    &self.device,
                    "assets/shaders/raytrace.rmiss.spv",
                ))
                .name(&shader_entry_point),
            vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::CLOSEST_HIT_KHR)
                .module(create_shader_module(
                    &self.device,
                    "assets/shaders/raytrace.rchit.spv",
                ))
                .name(&shader_entry_point),
        ];

        let shader_groups = [
            // Raygen
            vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
                ._type(vk::RayTracingShaderGroupTypeKHR::GENERAL_KHR)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // Miss
            vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
                ._type(vk::RayTracingShaderGroupTypeKHR::GENERAL_KHR)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // Hit Group = Closest Hit + Any Hit
            vk::RayTracingShaderGroupCreateInfoKHRBuilder::new()
                ._type(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP_KHR)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        ];

        let pipeline_create_info = vk::RayTracingPipelineCreateInfoKHRBuilder::new()
            .stages(&pipeline_shader_stages)
            .groups(&shader_groups)
            .max_pipeline_ray_recursion_depth(1)
            .layout(self.pipeline_layout);

        self.pipeline = unsafe {
            self.device
                .create_ray_tracing_pipelines_khr(None, None, &[pipeline_create_info], None)
                .unwrap()[0]
        };

        for stage in &pipeline_shader_stages {
            unsafe { self.device.destroy_shader_module(Some(stage.module), None) }
        }
    }

    pub fn create_shader_binding_table(&mut self) {}

    pub fn destroy(&mut self) {
        self.raytracing_builder.destroy();
    }
}

#[derive(Default)]
pub struct DescriptorSetBindings {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
    bindings_flags: Vec<vk::DescriptorBindingFlags>,
}

unsafe impl Send for DescriptorSetBindings {}
unsafe impl Sync for DescriptorSetBindings {}

impl DescriptorSetBindings {
    pub fn add_bindings(&mut self, binding: &[vk::DescriptorSetLayoutBinding]) {
        self.bindings.extend_from_slice(binding);
    }

    pub fn create_pool(&self, device: &DeviceLoader) -> vk::DescriptorPool {
        let pool_sizes: Vec<_> = self
            .bindings
            .iter()
            .map(|binding| {
                vk::DescriptorPoolSizeBuilder::new()
                    ._type(binding.descriptor_type)
                    .descriptor_count(binding.descriptor_count)
            })
            .collect();

        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .max_sets(1)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::empty())
            .build();

        unsafe {
            device
                .create_descriptor_pool(&create_info, None, None)
                .unwrap()
        }
    }

    pub fn create_layout(&self, device: &DeviceLoader) -> vk::DescriptorSetLayout {
        let bindings: Vec<_> = self
            .bindings
            .iter()
            .map(|binding| binding.into_builder())
            .collect();
        let create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::empty());

        unsafe {
            device
                .create_descriptor_set_layout(&create_info, None, None)
                .unwrap()
        }
    }
}
