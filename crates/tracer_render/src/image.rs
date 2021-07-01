use crate::resources::Image;
use erupt::vk;
use std::ops::Range;

pub struct ImageInfo {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub levels: u32,
    pub layers: u32,
    pub samples: vk::SampleCountFlags,
    pub usage: vk::ImageUsageFlags,
}

#[derive(Clone)]
pub struct ImageSubresourceRange {
    pub aspect: vk::ImageAspectFlags,
    pub first_level: u32,
    pub level_count: u32,
    pub first_layer: u32,
    pub layer_count: u32,
}

impl ImageSubresourceRange {
    pub fn new(aspect: vk::ImageAspectFlags, levels: Range<u32>, layers: Range<u32>) -> Self {
        assert!(levels.end >= levels.start);

        assert!(layers.end >= layers.start);

        ImageSubresourceRange {
            aspect,
            first_level: levels.start,
            level_count: levels.end - levels.start,
            first_layer: layers.start,
            layer_count: layers.end - layers.start,
        }
    }
}

pub struct ImageSubresourceLayers {
    pub aspect: vk::ImageAspectFlags,
    pub level: u32,
    pub first_layer: u32,
    pub layer_count: u32,
}

impl ImageSubresourceLayers {
    pub fn new(aspect: vk::ImageAspectFlags, level: u32, layers: Range<u32>) -> Self {
        assert!(layers.end >= layers.start);

        ImageSubresourceLayers {
            aspect,
            level,
            first_layer: layers.start,
            layer_count: layers.end - layers.start,
        }
    }
}

pub struct ImageMemoryBarrier<'a> {
    pub image: &'a Image,
    pub old_layout: Option<vk::ImageLayout>,
    pub new_layout: vk::ImageLayout,
    pub family_transfer: Option<Range<u32>>,
    pub subresource: ImageSubresourceRange,
}

#[derive(Clone)]
pub struct ImageViewInfo {
    pub view_type: vk::ImageViewType,
    pub subresource: ImageSubresourceRange,
    pub image: Image,
}

impl ImageViewInfo {
    pub fn new(image: Image, image_aspect_flags: vk::ImageAspectFlags) -> Self {
        let info = image.info();

        ImageViewInfo {
            view_type: vk::ImageViewType::_2D,
            subresource: ImageSubresourceRange::new(
                image_aspect_flags,
                0..info.layers,
                0..info.layers,
            ),
            image,
        }
    }
}
