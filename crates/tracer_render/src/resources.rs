use crate::acceleration_structures::AccelerationStructureInfo;
use crate::buffer::BufferInfo;
use crate::descriptor::{DescriptorSetInfo, DescriptorSetLayoutInfo, DescriptorSizes};
use crate::framebuffer::FramebufferInfo;
use crate::image::{ImageInfo, ImageViewInfo};
use crate::pipeline::{GraphicsPipelineInfo, PipelineLayoutInfo, RayTracingPipelineInfo};
use crate::render_pass::RenderPassInfo;
use crate::shader::ShaderModuleInfo;
use erupt::vk;
use erupt::vk::DeviceAddress;
use gpu_alloc::{MemoryBlock, UsageFlags};
use std::cell::UnsafeCell;
use std::ops::Deref;
use std::sync::Arc;

struct BufferInner {
    info: BufferInfo,
    handle: vk::Buffer,
    device_address: Option<DeviceAddress>,
    index: usize,
    memory_handle: vk::DeviceMemory,
    memory_offset: u64,
    memory_size: u64,
    memory_block: UnsafeCell<MemoryBlock<vk::DeviceMemory>>,
}

#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}

impl Buffer {
    pub fn info(&self) -> &BufferInfo {
        &self.inner.info
    }
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

pub struct MappableBuffer {
    buffer: Buffer,
    allocation_flags: UsageFlags,
}

impl From<MappableBuffer> for Buffer {
    fn from(buffer: MappableBuffer) -> Self {
        buffer.buffer
    }
}

impl Deref for MappableBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Buffer {
        &self.buffer
    }
}

impl MappableBuffer {
    pub fn new(
        info: BufferInfo,
        handle: vk::Buffer,
        device_address: Option<DeviceAddress>,
        index: usize,
        memory_block: MemoryBlock<vk::DeviceMemory>,
        allocation_flags: UsageFlags,
    ) -> Self {
        MappableBuffer {
            buffer: Buffer {
                inner: Arc::new(BufferInner {
                    info,
                    handle,
                    device_address,
                    memory_handle: *memory_block.memory(),
                    memory_offset: memory_block.offset(),
                    memory_size: memory_block.size(),
                    memory_block: UnsafeCell::new(memory_block),
                    index,
                }),
            },
            allocation_flags,
        }
    }

    pub unsafe fn memory_block(&mut self) -> &mut MemoryBlock<vk::DeviceMemory> {
        &mut *self.inner.memory_block.get()
    }
}

struct ImageInner {
    info: ImageInfo,
    handle: vk::Image,
    memory_block: Option<MemoryBlock<vk::DeviceMemory>>,
}

pub struct Image {
    inner: Arc<ImageInner>,
}

impl Image {
    pub fn info(&self) -> &ImageInfo {
        &self.inner.info
    }
}

pub struct ImageView {
    info: ImageViewInfo,
    handle: vk::ImageView,
}

pub struct Fence {
    handle: vk::Fence,
}

pub struct Semaphore {
    pub handle: vk::Semaphore,
}

pub struct RenderPass {
    info: RenderPassInfo,
    handle: vk::RenderPass,
}

pub struct Sampler {
    handle: vk::Sampler,
}

pub struct Framebuffer {
    info: FramebufferInfo,
    handle: vk::Framebuffer,
}

pub struct ShaderModule {
    info: ShaderModuleInfo,
    handle: vk::ShaderModule,
}

pub struct DescriptorSetLayout {
    info: DescriptorSetLayoutInfo,
    handle: vk::DescriptorSetLayout,
    sizes: DescriptorSizes,
}

pub struct DescriptorSet {
    info: DescriptorSetInfo,
    handle: vk::DescriptorSet,
    pool: vk::DescriptorPool,
}

pub struct PipelineLayout {
    info: PipelineLayoutInfo,
    handle: vk::PipelineLayout,
}

pub struct GraphicsPipeline {
    info: GraphicsPipelineInfo,
    handle: vk::Pipeline,
}

pub struct AccelerationStructure {
    info: AccelerationStructureInfo,
    handle: vk::AccelerationStructureKHR,
    address: DeviceAddress,
}

pub struct RayTracingPipeline {
    info: RayTracingPipelineInfo,
    handle: vk::Pipeline,
    group_handlers: Arc<[u8]>,
}
