use crate::acceleration_structures::AccelerationStructureInfo;
use crate::buffer::{BufferInfo, DeviceAddress};
use crate::descriptor::{DescriptorSetInfo, DescriptorSetLayoutInfo, DescriptorSizes};
use crate::framebuffer::FramebufferInfo;
use crate::image::{ImageInfo, ImageViewInfo};
use crate::pipeline::{GraphicsPipelineInfo, PipelineLayoutInfo, RayTracingPipelineInfo};
use crate::render_pass::RenderPassInfo;
use crate::shader::ShaderModuleInfo;
use erupt::vk;
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
    allocation_flags: UsageFlags,
}

impl Buffer {
    pub fn new(
        info: BufferInfo,
        handle: vk::Buffer,
        device_address: Option<DeviceAddress>,
        index: usize,
        memory_block: MemoryBlock<vk::DeviceMemory>,
        allocation_flags: UsageFlags,
    ) -> Self {
        Buffer {
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
            allocation_flags,
        }
    }

    pub fn info(&self) -> &BufferInfo {
        &self.inner.info
    }

    pub fn handle(&self) -> vk::Buffer {
        self.inner.handle
    }

    pub fn device_address(&self) -> Option<DeviceAddress> {
        self.inner.device_address
    }

    pub unsafe fn memory_block(&mut self) -> &mut MemoryBlock<vk::DeviceMemory> {
        &mut *self.inner.memory_block.get()
    }
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

#[derive(Clone)]
pub struct Fence {
    handle: vk::Fence,
}

impl Fence {
    pub fn new(handle: vk::Fence) -> Self {
        Fence { handle }
    }
    pub fn handle(&self) -> vk::Fence {
        self.handle
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

    pub fn info(&self) -> &RenderPassInfo {
        &self.info
    }
}

#[derive(Clone)]
pub struct Sampler {
    handle: vk::Sampler,
}

impl Sampler {
    pub fn new(handle: vk::Sampler) -> Self {
        Sampler { handle }
    }

    pub fn handle(&self) -> vk::Sampler {
        self.handle
    }
}

#[derive(Clone)]
pub struct Framebuffer {
    info: FramebufferInfo,
    handle: vk::Framebuffer,
}

impl Framebuffer {
    pub fn new(info: FramebufferInfo, handle: vk::Framebuffer) -> Self {
        Framebuffer { info, handle }
    }

    pub fn info(&self) -> &FramebufferInfo {
        &self.info
    }

    pub fn handle(&self) -> vk::Framebuffer {
        self.handle
    }
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

    pub fn handle(&self) -> vk::DescriptorSet {
        self.handle
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

    pub fn info(&self) -> &GraphicsPipelineInfo {
        &self.info
    }

    pub fn handle(&self) -> vk::Pipeline {
        self.handle
    }
}

#[derive(Clone)]
pub struct AccelerationStructure {
    info: AccelerationStructureInfo,
    handle: vk::AccelerationStructureKHR,
    device_address: DeviceAddress,
}

impl AccelerationStructure {
    pub fn new(
        info: AccelerationStructureInfo,
        handle: vk::AccelerationStructureKHR,
        device_address: DeviceAddress,
    ) -> Self {
        AccelerationStructure {
            info,
            handle,
            device_address,
        }
    }

    pub fn info(&self) -> &AccelerationStructureInfo {
        &self.info
    }

    pub fn handle(&self) -> vk::AccelerationStructureKHR {
        self.handle
    }

    pub fn device_address(&self) -> DeviceAddress {
        self.device_address
    }
}

#[derive(Clone)]
pub struct RayTracingPipeline {
    info: RayTracingPipelineInfo,
    handle: vk::Pipeline,
    group_handlers: Arc<[u8]>,
}

impl RayTracingPipeline {
    pub fn new(
        info: RayTracingPipelineInfo,
        handle: vk::Pipeline,
        group_handlers: Arc<[u8]>,
    ) -> Self {
        RayTracingPipeline {
            info,
            handle,
            group_handlers,
        }
    }

    pub fn info(&self) -> &RayTracingPipelineInfo {
        &self.info
    }

    pub fn handle(&self) -> vk::Pipeline {
        self.handle
    }

    pub fn group_handlers(&self) -> &[u8] {
        &*self.group_handlers
    }
}
