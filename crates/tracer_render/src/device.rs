use crate::buffer::BufferInfo;
use crate::resources::{Buffer, MappableBuffer, Semaphore};
use crate::surface::Surface;
use crate::swapchain::Swapchain;
use crevice::internal::bytemuck::Pod;
use erupt::{vk, DeviceLoader, InstanceLoader};
use gpu_alloc::{GpuAllocator, UsageFlags};
use gpu_alloc_erupt::EruptMemoryDevice;
use parking_lot::Mutex;
use slab::Slab;
use std::sync::Arc;

pub struct DeviceInner {
    handle: DeviceLoader,
    instance: Arc<InstanceLoader>,
    physical_device: vk::PhysicalDevice,
    allocator: Mutex<GpuAllocator<vk::DeviceMemory>>,
    buffers: Mutex<Slab<vk::Buffer>>,
    semaphores: Mutex<Slab<vk::Semaphore>>,
    swapchains: Mutex<Slab<vk::SwapchainKHR>>,
    descriptor_pools: Mutex<Slab<vk::DescriptorPool>>,
    descriptor_set_layouts: Mutex<Slab<vk::DescriptorPool>>,
}

#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

impl Device {
    pub fn new(
        instance: Arc<InstanceLoader>,
        logical: DeviceLoader,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        let allocator = Mutex::new(GpuAllocator::new(
            gpu_alloc::Config::i_am_prototyping(),
            unsafe { gpu_alloc_erupt::device_properties(&instance, physical_device).unwrap() },
        ));
        Device {
            inner: Arc::new(DeviceInner {
                handle: logical,
                instance,
                physical_device,
                allocator,
                buffers: Mutex::new(Slab::with_capacity(1024)),
                semaphores: Mutex::new(Slab::with_capacity(1024)),
                swapchains: Mutex::new(Slab::with_capacity(1024)),
                descriptor_pools: Mutex::new(Slab::with_capacity(1024)),
                descriptor_set_layouts: Mutex::new(Slab::with_capacity(1024)),
            }),
        }
    }

    pub fn instance(&self) -> &InstanceLoader {
        &self.inner.instance
    }

    pub fn handle(&self) -> &DeviceLoader {
        &self.inner.handle
    }

    pub fn swapchains(&self) -> &Mutex<Slab<vk::SwapchainKHR>> {
        &self.inner.swapchains
    }

    pub fn create_buffer(&self, info: BufferInfo, allocation_flags: UsageFlags) -> MappableBuffer {
        let buffer = unsafe {
            self.inner
                .handle
                .create_buffer(
                    &vk::BufferCreateInfoBuilder::new()
                        .size(info.size)
                        .usage(info.usage_flags)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap()
        };

        let mem_requirements = unsafe { self.inner.handle.get_buffer_memory_requirements(buffer) };

        let mem_block = unsafe {
            self.inner
                .allocator
                .lock()
                .alloc(
                    EruptMemoryDevice::wrap(&self.inner.handle),
                    gpu_alloc::Request {
                        size: mem_requirements.size,
                        align_mask: (mem_requirements.alignment - 1) | info.align,
                        usage: allocation_flags,
                        memory_types: mem_requirements.memory_type_bits,
                    },
                )
                .unwrap()
        };

        unsafe {
            self.inner
                .handle
                .bind_buffer_memory(buffer, *mem_block.memory(), mem_block.offset())
                .unwrap()
        }

        let device_address = if allocation_flags.contains(UsageFlags::DEVICE_ADDRESS) {
            let device_address = unsafe {
                self.inner.handle.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfoBuilder::new().buffer(buffer),
                )
            };
            Some(device_address)
        } else {
            None
        };

        let buffer_index = self.inner.buffers.lock().insert(buffer);

        tracing::debug!("Created Buffer {:p}", buffer);
        MappableBuffer::new(
            info,
            buffer,
            device_address,
            buffer_index,
            mem_block,
            allocation_flags,
        )
    }

    pub fn create_buffer_with_data<T: 'static>(&self, info: BufferInfo, data: &[T]) -> Buffer
    where
        T: Pod,
    {
        let mut buffer = self.create_buffer(info, UsageFlags::UPLOAD);

        unsafe {
            let ptr = buffer
                .memory_block()
                .map(
                    EruptMemoryDevice::wrap(&self.inner.handle),
                    0,
                    std::mem::size_of_val(data),
                )
                .expect("Mapping to buffer failed");

            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ptr.as_ptr(),
                std::mem::size_of_val(data),
            );

            buffer
                .memory_block()
                .unmap(EruptMemoryDevice::wrap(&self.inner.handle));
        }
        buffer.into()
    }

    pub fn create_swapchain(&self, surface: &Surface) -> Swapchain {
        Swapchain::new(self, surface)
    }

    pub fn create_semaphore(&self) -> Semaphore {
        let semaphore = unsafe {
            self.handle()
                .create_semaphore(&vk::SemaphoreCreateInfoBuilder::new(), None)
                .unwrap()
        };

        self.inner.semaphores.lock().insert(semaphore);

        Semaphore { handle: semaphore }
    }

    pub fn cleanup(&mut self) {
        let device = self.handle();

        unsafe {
            self.inner
                .swapchains
                .lock()
                .iter()
                .for_each(|(_, &swapchain)| device.destroy_swapchain_khr(Some(swapchain), None));

            self.inner
                .semaphores
                .lock()
                .iter()
                .for_each(|(_, &semaphore)| device.destroy_semaphore(Some(semaphore), None));

            self.handle().destroy_device(None)
        }
    }
}
