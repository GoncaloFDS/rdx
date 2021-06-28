use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::window::{WindowCreated, WindowResized};
use bevy::winit::WinitWindows;

use crate::renderer::Renderer;

mod buffer_resource;
mod commands;
mod raytracing;
mod raytracing_builder;
mod renderer;
mod util;
mod vertex;

#[derive(Default)]
pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_startup_system_to_stage(StartupStage::PreStartup, setup.system())
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

    renderer.init();

    commands.insert_resource(renderer);
}

fn window_resize(mut window_resized_event: EventReader<WindowResized>) {
    for event in window_resized_event.iter() {
        if event.width != 0.0 && event.height != 0.0 {
            tracing::debug!("window resized")
        }
    }
}

fn world_cleanup(mut commands: Commands, mut app_exit_events: EventReader<AppExit>) {
    if app_exit_events.iter().next().is_some() {
        commands.remove_resource::<Renderer>();
    }
}
