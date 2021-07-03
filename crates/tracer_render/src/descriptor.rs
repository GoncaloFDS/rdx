use crate::image::ImageView;
use crate::resources::{
    AccelerationStructure, Buffer, DescriptorSet, DescriptorSetLayout, Sampler,
};
use erupt::vk;
use std::ops::Deref;

const DESCRIPTOR_TYPES_COUNT: usize = 12;

#[derive(Clone)]
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

#[derive(Clone)]
pub struct DescriptorSetLayoutInfo {
    pub bindings: Vec<DescriptorSetLayoutBinding>,
    pub flags: vk::DescriptorSetLayoutCreateFlags,
}

#[derive(Clone)]
pub struct DescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub count: u32,
    pub stages: vk::ShaderStageFlags,
    pub flags: vk::DescriptorBindingFlags,
}

fn descriptor_type_from_index(index: usize) -> vk::DescriptorType {
    debug_assert!(index < DESCRIPTOR_TYPES_COUNT);

    match index {
        0 => vk::DescriptorType::SAMPLER,
        1 => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        2 => vk::DescriptorType::SAMPLED_IMAGE,
        3 => vk::DescriptorType::STORAGE_IMAGE,
        4 => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        5 => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        6 => vk::DescriptorType::UNIFORM_BUFFER,
        7 => vk::DescriptorType::STORAGE_BUFFER,
        8 => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        9 => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
        10 => vk::DescriptorType::INPUT_ATTACHMENT,
        11 => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
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
        self.sizes[binding.descriptor_type.0 as usize] += binding.count;
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

impl DescriptorSizes {
    pub fn as_slice(&self) -> &[vk::DescriptorPoolSizeBuilder<'static>] {
        &self.sizes[..self.count.into()]
    }

    pub fn from_bindings(bindings: &[DescriptorSetLayoutBinding]) -> Self {
        DescriptorSizesBuilder::from_bindings(bindings).build()
    }
}

impl Deref for DescriptorSizes {
    type Target = [vk::DescriptorPoolSizeBuilder<'static>];

    fn deref(&self) -> &[vk::DescriptorPoolSizeBuilder<'static>] {
        self.as_slice()
    }
}
