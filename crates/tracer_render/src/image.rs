use erupt::vk;
use gpu_alloc::MemoryBlock;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;

pub struct ImageInfo {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: vk::SampleCountFlagBits,
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
                0..info.array_layers,
                0..info.array_layers,
            ),
            image,
        }
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

impl ImageView {
    pub fn new(info: ImageViewInfo, handle: vk::ImageView) -> ImageView {
        ImageView { info, handle }
    }

    pub fn info(&self) -> &ImageViewInfo {
        &self.info
    }

    pub fn handle(&self) -> vk::ImageView {
        self.handle
    }
}
