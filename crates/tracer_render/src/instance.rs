use crate::debug::VALIDATION_LAYER;
use erupt::utils::surface;
use erupt::{vk, EntryLoader, InstanceLoader};
use winit::window::Window;

pub fn create_instance(window: &Window, entry: &EntryLoader) -> InstanceLoader {
    let app_info = vk::ApplicationInfoBuilder::new().api_version(vk::make_api_version(1, 2, 0, 0));

    let mut instance_extensions = surface::enumerate_required_extensions(window).unwrap();
    if cfg!(debug_assertions) {
        instance_extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    #[cfg(target_os = "windows")]
    {
        instance_extensions.push(vk::KHR_WIN32_SURFACE_EXTENSION_NAME);
    }

    let mut instance_layers = Vec::new();
    if cfg!(debug_assertions) {
        instance_layers.push(VALIDATION_LAYER);
    }

    let instance_info = vk::InstanceCreateInfoBuilder::new()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);

    unsafe { InstanceLoader::new(&entry, &instance_info, None).unwrap() }
}
