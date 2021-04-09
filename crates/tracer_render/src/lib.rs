use ash::version::DeviceV1_0;
use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::window::{WindowCreated, WindowResized};
use bevy::winit::WinitWindows;

use crate::render_context::RenderContext;
use crate::renderer::Renderer;
use crate::swapchain::SwapchainConfig;

mod device_info;
mod render_context;
mod renderer;
mod vk_types;
mod swapchain;
mod mesh;

#[derive(Default)]
pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_startup_system_to_stage(StartupStage::PreStartup, setup.system())
            .add_system(world_update.system())
            .add_system_to_stage(CoreStage::PreUpdate, window_resize.system())
            .add_system_to_stage(CoreStage::Last, world_cleanup.system());
    }
}

fn setup(
    mut commands: Commands,
    mut window_created_events: EventReader<WindowCreated>,
    winit_windows: Res<WinitWindows>,
) {
    let window_id = window_created_events
        .iter()
        .next()
        .map(|event| event.id)
        .unwrap();

    let winit_window = winit_windows.get_window(window_id).unwrap();
    let mut renderer = Renderer::new(winit_window);
    let render_context = RenderContext::new(&renderer);

    commands.insert_resource(renderer);
    commands.insert_resource(render_context);
}

fn world_update(
    mut renderer: ResMut<Renderer>,
    mut render_context: ResMut<RenderContext>,
) {
    renderer.draw_frame(&mut render_context);
}

fn window_resize(
    mut window_resized_event: EventReader<WindowResized>,
    mut renderer_context: ResMut<RenderContext>,
) {
    for event in window_resized_event.iter() {
        if event.width != 0.0 && event.height != 0.0 {
            renderer_context.recreate_swapchain(event.width, event.height);
        }
    }
}

fn world_cleanup(
    mut commands: Commands,
    mut app_exit_events: EventReader<AppExit>,
    renderer: Res<Renderer>,
) {
    if app_exit_events.iter().next().is_some() {
        unsafe {
            renderer.vk_context.device.device_wait_idle().unwrap();
        }

        commands.remove_resource::<RenderContext>();
        commands.remove_resource::<Renderer>();
    }
}
