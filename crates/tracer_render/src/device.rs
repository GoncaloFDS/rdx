use crate::buffer::BufferInfo;
use crate::resources::MappableBuffer;
use erupt::{vk, DeviceLoader, InstanceLoader};
use gpu_alloc::{GpuAllocator, UsageFlags};
use gpu_alloc_erupt::EruptMemoryDevice;
use parking_lot::Mutex;
use slab::Slab;
use std::sync::Arc;

pub struct Inner {
    logical: DeviceLoader,
    physical: vk::PhysicalDevice,
    allocator: Mutex<GpuAllocator<vk::DeviceMemory>>,
    buffers: Mutex<Slab<vk::Buffer>>,
    descriptor_pools: Mutex<Slab<vk::DescriptorPool>>,
    descriptor_set_layouts: Mutex<Slab<vk::DescriptorPool>>,
}

pub struct Device {
    inner: Arc<Inner>,
}

impl Device {
    pub fn new(
        instance: &InstanceLoader,
        logical: DeviceLoader,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        Device {
            inner: Arc::new(Inner {
                logical,
                physical: physical_device,
                allocator: Mutex::new(GpuAllocator::new(
                    gpu_alloc::Config::i_am_prototyping(),
                    unsafe {
                        gpu_alloc_erupt::device_properties(&instance, physical_device).unwrap()
                    },
                )),
                buffers: Mutex::new(Slab::with_capacity(1024)),
                descriptor_pools: Mutex::new(Slab::with_capacity(1024)),
                descriptor_set_layouts: Mutex::new(Slab::with_capacity(1024)),
            }),
        }
    }

    pub fn create_buffer(&self, info: BufferInfo, allocation_flags: UsageFlags) -> MappableBuffer {
        let buffer = unsafe {
            self.inner
                .logical
                .create_buffer(
                    &vk::BufferCreateInfoBuilder::new()
                        .size(info.size)
                        .usage(info.usage_flags)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                    None,
                )
                .unwrap()
        };

        let mem_requirements = unsafe {
            self.inner
                .logical
                .get_buffer_memory_requirements(buffer, None)
        };

        let mem_block = unsafe {
            self.inner
                .allocator
                .lock()
                .alloc(
                    EruptMemoryDevice::wrap(&self.inner.logical),
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
                .logical
                .bind_buffer_memory(buffer, *mem_block.memory(), mem_block.offset())
                .unwrap()
        }

        let device_address = if allocation_flags.contains(UsageFlags::DEVICE_ADDRESS) {
            let device_address = unsafe {
                self.inner.logical.get_buffer_device_address(
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
}
