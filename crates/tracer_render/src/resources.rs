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
use std::hash::{Hash, Hasher};
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

#[derive(Clone)]
pub struct Image {
    inner: Arc<ImageInner>,
}

impl PartialEq for Image {
    fn eq(&self, rhs: &Self) -> bool {
        self.inner.handle == rhs.inner.handle
    }
}

impl Eq for Image {}

impl Hash for Image {
    fn hash<H>(&self, hasher: &mut H)
    where
        H: Hasher,
    {
        self.inner.handle.hash(hasher)
    }
}

impl Image {
    pub fn new(
        info: ImageInfo,
        handle: vk::Image,
        memory_block: Option<MemoryBlock<vk::DeviceMemory>>,
    ) -> Self {
        Image {
            inner: Arc::new(ImageInner {
                info,
                handle,
                memory_block,
            }),
        }
    }
    pub fn info(&self) -> &ImageInfo {
        &self.inner.info
    }

    pub fn handle(&self) -> vk::Image {
        self.inner.handle
    }
}

#[derive(Clone)]
pub struct ImageView {
    info: ImageViewInfo,
    handle: vk::ImageView,
}

#[derive(Clone)]
pub struct Fence {
    handle: vk::Fence,
}

impl Fence {
    pub fn new(handle: vk::Fence) -> Self {
        Fence { handle }
    }
}

#[derive(Clone)]
pub struct Semaphore {
    handle: vk::Semaphore,
}

impl Semaphore {
    pub fn new(handle: vk::Semaphore) -> Self {
        Semaphore { handle }
    }

    pub fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

#[derive(Clone)]
pub struct RenderPass {
    info: RenderPassInfo,
    handle: vk::RenderPass,
}

impl RenderPass {
    pub fn new(info: RenderPassInfo, handle: vk::RenderPass) -> Self {
        RenderPass { info, handle }
    }

    pub fn handle(&self) -> vk::RenderPass {
        self.handle
    }
}

#[derive(Clone)]
pub struct Sampler {
    handle: vk::Sampler,
}

#[derive(Clone)]
pub struct Framebuffer {
    info: FramebufferInfo,
    handle: vk::Framebuffer,
}

#[derive(Clone)]
pub struct ShaderModule {
    info: ShaderModuleInfo,
    handle: vk::ShaderModule,
}

impl ShaderModule {
    pub fn new(info: ShaderModuleInfo, handle: vk::ShaderModule) -> Self {
        ShaderModule { info, handle }
    }

    pub fn info(&self) -> &ShaderModuleInfo {
        &self.info
    }

    pub fn handle(&self) -> vk::ShaderModule {
        self.handle
    }
}

#[derive(Clone)]
pub struct DescriptorSetLayout {
    info: DescriptorSetLayoutInfo,
    handle: vk::DescriptorSetLayout,
    sizes: DescriptorSizes,
}

impl DescriptorSetLayout {
    pub fn new(
        info: DescriptorSetLayoutInfo,
        handle: vk::DescriptorSetLayout,
        sizes: DescriptorSizes,
    ) -> Self {
        DescriptorSetLayout {
            info,
            handle,
            sizes,
        }
    }

    pub fn info(&self) -> &DescriptorSetLayoutInfo {
        &self.info
    }

    pub fn handle(&self) -> vk::DescriptorSetLayout {
        self.handle
    }

    pub fn sizes(&self) -> &DescriptorSizes {
        &self.sizes
    }
}

#[derive(Clone)]
pub struct DescriptorSet {
    info: DescriptorSetInfo,
    handle: vk::DescriptorSet,
    pool: vk::DescriptorPool,
}

impl DescriptorSet {
    pub fn new(
        info: DescriptorSetInfo,
        handle: vk::DescriptorSet,
        pool: vk::DescriptorPool,
    ) -> Self {
        DescriptorSet { info, handle, pool }
    }
}

#[derive(Clone)]
pub struct PipelineLayout {
    info: PipelineLayoutInfo,
    handle: vk::PipelineLayout,
}

impl PipelineLayout {
    pub fn info(&self) -> &PipelineLayoutInfo {
        &self.info
    }

    pub fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }
}

impl PipelineLayout {
    pub fn new(info: PipelineLayoutInfo, handle: vk::PipelineLayout) -> Self {
        PipelineLayout { info, handle }
    }
}

#[derive(Clone)]
pub struct GraphicsPipeline {
    info: GraphicsPipelineInfo,
    handle: vk::Pipeline,
}

impl GraphicsPipeline {
    pub fn new(info: GraphicsPipelineInfo, handle: vk::Pipeline) -> Self {
        GraphicsPipeline { info, handle }
    }
}

#[derive(Clone)]
pub struct AccelerationStructure {
    info: AccelerationStructureInfo,
    handle: vk::AccelerationStructureKHR,
    address: DeviceAddress,
}

#[derive(Clone)]
pub struct RayTracingPipeline {
    info: RayTracingPipelineInfo,
    handle: vk::Pipeline,
    group_handlers: Arc<[u8]>,
}
