use crate::resources::{
    AccelerationStructure, Buffer, DescriptorSet, DescriptorSetLayout, ImageView, Sampler,
};
use erupt::vk;

const DESCRIPTOR_TYPES_COUNT: usize = 12;

pub struct DescriptorSetInfo {
    pub layout: DescriptorSetLayout,
}

pub struct WriteDescriptorSet<'a> {
    pub set: &'a DescriptorSet,
    pub binding: u32,
    pub element: u32,
    pub descriptors: Descriptors<'a>,
}

pub enum Descriptors<'a> {
    Sampler(&'a [Sampler]),
    CombinedImageSampler(&'a [(ImageView, vk::ImageLayout, Sampler)]),
    SampledImage(&'a [(ImageView, vk::ImageLayout)]),
    StorageImage(&'a [(ImageView, vk::ImageLayout)]),
    UniformBuffer(&'a [(Buffer, u64, u64)]),
    StorageBuffer(&'a [(Buffer, u64, u64)]),
    UniformBufferDynamic(&'a [(Buffer, u64, u64)]),
    StorageBufferDynamic(&'a [(Buffer, u64, u64)]),
    InputAttachment(&'a [(ImageView, vk::ImageLayout)]),
    AccelerationStructure(&'a [AccelerationStructure]),
}

pub struct CopyDescriptorSet<'a> {
    pub src: &'a DescriptorSet,
    pub src_binding: u32,
    pub src_element: u32,
    pub dst: &'a DescriptorSet,
    pub dst_binding: u32,
    pub dst_element: u32,
    pub count: u32,
}

pub struct DescriptorSetLayoutInfo {
    pub bindings: Vec<DescriptorSetLayoutBinding>,
    pub flags: vk::DescriptorSetLayoutCreateFlags,
}

pub struct DescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: DescriptorType,
    pub count: u32,
    pub stages: vk::ShaderStageFlags,
    pub flags: vk::DescriptorBindingFlags,
}

#[derive(Copy, Clone)]
pub enum DescriptorType {
    Sampler,
    CombinedImageSampler,
    SampledImage,
    StorageImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    UniformBufferDynamic,
    StorageBufferDynamic,
    InputAttachment,
    AccelerationStructure,
}

fn descriptor_type_from_index(index: usize) -> vk::DescriptorType {
    debug_assert!(index < DESCRIPTOR_TYPES_COUNT);

    match index {
        0 => {
            debug_assert_eq!(DescriptorType::Sampler as usize, index);

            vk::DescriptorType::SAMPLER
        }
        1 => {
            debug_assert_eq!(DescriptorType::CombinedImageSampler as usize, index);

            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
        }
        2 => {
            debug_assert_eq!(DescriptorType::SampledImage as usize, index);

            vk::DescriptorType::SAMPLED_IMAGE
        }
        3 => {
            debug_assert_eq!(DescriptorType::StorageImage as usize, index);

            vk::DescriptorType::STORAGE_IMAGE
        }
        4 => {
            debug_assert_eq!(DescriptorType::UniformTexelBuffer as usize, index);

            vk::DescriptorType::UNIFORM_TEXEL_BUFFER
        }
        5 => {
            debug_assert_eq!(DescriptorType::StorageTexelBuffer as usize, index);

            vk::DescriptorType::STORAGE_TEXEL_BUFFER
        }
        6 => {
            debug_assert_eq!(DescriptorType::UniformBuffer as usize, index);

            vk::DescriptorType::UNIFORM_BUFFER
        }
        7 => {
            debug_assert_eq!(DescriptorType::StorageBuffer as usize, index);

            vk::DescriptorType::STORAGE_BUFFER
        }
        8 => {
            debug_assert_eq!(DescriptorType::UniformBufferDynamic as usize, index);

            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
        }
        9 => {
            debug_assert_eq!(DescriptorType::StorageBufferDynamic as usize, index);

            vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
        }
        10 => {
            debug_assert_eq!(DescriptorType::InputAttachment as usize, index);

            vk::DescriptorType::INPUT_ATTACHMENT
        }
        11 => {
            debug_assert_eq!(DescriptorType::AccelerationStructure as usize, index);

            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
        }
        _ => unreachable!(),
    }
}

#[derive(Clone, Debug)]
pub struct DescriptorSizesBuilder {
    sizes: [u32; DESCRIPTOR_TYPES_COUNT],
}

impl DescriptorSizesBuilder {
    pub fn zero() -> Self {
        DescriptorSizesBuilder {
            sizes: [0; DESCRIPTOR_TYPES_COUNT],
        }
    }

    pub fn add_binding(&mut self, binding: &DescriptorSetLayoutBinding) {
        self.sizes[binding.descriptor_type as usize] += binding.count;
    }

    pub fn from_bindings(bindings: &[DescriptorSetLayoutBinding]) -> Self {
        let mut ranges = Self::zero();

        for binding in bindings {
            ranges.add_binding(binding);
        }

        ranges
    }

    pub fn build(self) -> DescriptorSizes {
        let mut sizes = [vk::DescriptorPoolSizeBuilder::new()
            ._type(vk::DescriptorType::SAMPLER)
            .descriptor_count(0); DESCRIPTOR_TYPES_COUNT];

        let mut count = 0u8;

        for (i, size) in self.sizes.iter().copied().enumerate() {
            if size > 0 {
                sizes[count as usize]._type = descriptor_type_from_index(i);

                sizes[count as usize].descriptor_count = size;

                count += 1;
            }
        }

        DescriptorSizes { sizes, count }
    }
}

/// Number of descriptors per type.
#[derive(Clone, Debug)]
pub struct DescriptorSizes {
    sizes: [vk::DescriptorPoolSizeBuilder<'static>; DESCRIPTOR_TYPES_COUNT],
    count: u8,
}
