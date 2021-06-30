use erupt::utils::surface;
use erupt::{vk, InstanceLoader};
use std::sync::Arc;
use winit::window::Window;

struct SurfaceInner {
    pub handle: vk::SurfaceKHR,
}

#[derive(Clone)]
pub struct Surface {
    inner: Arc<SurfaceInner>,
}

impl Surface {
    pub fn new(instance: &InstanceLoader, window: &Window) -> Self {
        Surface {
            inner: Arc::new(SurfaceInner {
                handle: unsafe { surface::create_surface(instance, window, None).unwrap() },
            }),
        }
    }

    pub fn handle(&self) -> vk::SurfaceKHR {
        self.inner.handle
    }
}
