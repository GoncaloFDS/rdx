use erupt::utils::surface;
use erupt::{vk, InstanceLoader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use winit::window::Window;

struct SurfaceInner {
    pub handle: vk::SurfaceKHR,
    pub used: AtomicBool,
}

pub struct Surface {
    inner: Arc<SurfaceInner>,
}

impl Surface {
    pub fn new(instance: &InstanceLoader, window: &Window) -> Self {
        Surface {
            inner: Arc::new(SurfaceInner {
                handle: unsafe { surface::create_surface(instance, window, None).unwrap() },
                used: AtomicBool::new(false),
            }),
        }
    }

    pub fn handle(&self) -> vk::SurfaceKHR {
        self.inner.handle
    }

    pub fn mark_used(&self) {
        self.inner.used.fetch_or(true, Ordering::SeqCst);
    }
}
