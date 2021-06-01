use erupt::vk;
use memoffset::offset_of;

#[derive(Debug, Copy, Clone)]
pub struct Vertex {
    pub position: glam::Vec3,
    // pub normal: glam::Vec3,
    // pub color: glam::Vec3,
}

impl Vertex {
    pub fn get_binding_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        [vk::VertexInputBindingDescriptionBuilder::new()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()]
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 1] {
        [
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Self, position) as u32)
                .build(),
            // vk::VertexInputAttributeDescription::builder()
            //     .binding(0)
            //     .location(1)
            //     .format(vk::Format::R32G32B32_SFLOAT)
            //     .offset(offset_of!(Self, normal) as u32)
            //     .build(),
            // vk::VertexInputAttributeDescription::builder()
            //     .binding(0)
            //     .location(2)
            //     .format(vk::Format::R32G32B32_SFLOAT)
            //     .offset(offset_of!(Self, color) as u32)
            //     .build(),
        ]
    }
}
