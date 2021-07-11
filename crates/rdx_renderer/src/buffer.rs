use crate::resources::Buffer;
use crate::util::{align_up, ToErupt};
use erupt::vk;
use gpu_alloc::UsageFlags;
use std::num::NonZeroU64;

pub struct BufferInfo {
    pub align: u64,
    pub size: u64,
    pub usage_flags: vk::BufferUsageFlags,
    pub allocation_flags: UsageFlags,
}

impl BufferInfo {
    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        let is_mask = self
            .align
            .checked_add(1)
            .map_or(false, u64::is_power_of_two);

        is_mask && (align_up(self.align, self.size).is_some())
    }
}

#[derive(Clone)]
pub struct BufferRegion {
    pub buffer: Buffer,
    pub offset: u64,
    pub size: u64,
    pub stride: Option<u64>,
}

impl BufferRegion {
    pub fn whole(buffer: Buffer) -> Self {
        BufferRegion {
            offset: 0,
            size: buffer.info().size,
            buffer,
            stride: None,
        }
    }
}

#[derive(Clone, Copy)]
pub struct DeviceAddress(pub NonZeroU64);

impl DeviceAddress {
    pub fn new(address: u64) -> DeviceAddress {
        NonZeroU64::new(address).map(DeviceAddress).unwrap()
    }

    pub fn offset(&mut self, offset: u64) -> DeviceAddress {
        let value = self.0.get().checked_add(offset).unwrap();
        DeviceAddress(unsafe { NonZeroU64::new_unchecked(value) })
    }
}

impl ToErupt<vk::DeviceOrHostAddressKHR> for DeviceAddress {
    fn to_erupt(&self) -> vk::DeviceOrHostAddressKHR {
        vk::DeviceOrHostAddressKHR {
            device_address: self.0.get(),
        }
    }
}

impl ToErupt<vk::DeviceOrHostAddressConstKHR> for DeviceAddress {
    fn to_erupt(&self) -> vk::DeviceOrHostAddressConstKHR {
        vk::DeviceOrHostAddressConstKHR {
            device_address: self.0.get(),
        }
    }
}
