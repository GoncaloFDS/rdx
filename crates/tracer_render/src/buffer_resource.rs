use erupt::vk::SharingMode;
use erupt::{vk, DeviceLoader};
use gpu_alloc::{GpuAllocator, MemoryBlock, Request, UsageFlags};
use gpu_alloc_erupt::EruptMemoryDevice;
use std::sync::{Arc, Mutex};

pub struct BufferResource {
    pub buffer: vk::Buffer,
    pub allocation: Option<MemoryBlock<vk::DeviceMemory>>,

    device: Arc<DeviceLoader>,
    allocator: Arc<Mutex<GpuAllocator<vk::DeviceMemory>>>,
}

impl BufferResource {
    pub fn new(
        device: Arc<DeviceLoader>,
        allocator: Arc<Mutex<GpuAllocator<vk::DeviceMemory>>>,
        buffer_size: vk::DeviceSize,
        usage_flags: vk::BufferUsageFlags,
        memory_usage: UsageFlags,
    ) -> Self {
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfoBuilder::new()
                    .size(buffer_size)
                    .usage(usage_flags)
                    .sharing_mode(SharingMode::EXCLUSIVE),
                None,
                None,
            )
        }
        .unwrap();

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer, None) };

        let allocation = unsafe {
            allocator.lock().unwrap().alloc(
                EruptMemoryDevice::wrap(&device),
                Request {
                    size: mem_requirements.size,
                    align_mask: (mem_requirements.alignment),
                    usage: memory_usage,
                    memory_types: mem_requirements.memory_type_bits,
                },
            )
        }
        .unwrap();

        unsafe {
            device
                .bind_buffer_memory(buffer, *allocation.memory(), allocation.offset())
                .unwrap()
        }

        BufferResource {
            buffer,
            allocation: Some(allocation),
            device,
            allocator,
        }
    }

    pub fn store<T: Copy>(&mut self, data: &[T]) {
        let buffer_size = std::mem::size_of::<T>() * data.len();

        unsafe {
            match self.allocation.as_mut().unwrap().map(
                EruptMemoryDevice::wrap(&self.device),
                0,
                buffer_size,
            ) {
                Ok(ptr) => {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr() as *const u8,
                        ptr.as_ptr(),
                        buffer_size,
                    );

                    self.allocation
                        .as_mut()
                        .unwrap()
                        .unmap(EruptMemoryDevice::wrap(&self.device))
                }
                Err(err) => panic!("Error {}", err),
            }
        };
    }

    pub fn get_device_address(&self) -> vk::DeviceAddress {
        unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfoBuilder::new().buffer(self.buffer),
            )
        }
    }
}

impl Drop for BufferResource {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(Some(self.buffer), None);
            self.allocator.lock().unwrap().dealloc(
                EruptMemoryDevice::wrap(&self.device),
                self.allocation.take().unwrap(),
            );
        }
    }
}

#[derive(Default)]
pub struct Texture {
    pub image: vk::Image,
    pub allocation: Option<MemoryBlock<vk::DeviceMemory>>,
    pub descriptor: vk::DescriptorImageInfo,
}
