use bevy::prelude::*;

fn main() {
    App::build()
        .insert_resource(bevy::log::LogSettings {
            level: bevy::utils::tracing::Level::DEBUG,
            ..Default::default()
        })
        .insert_resource(bevy::window::WindowDescriptor {
            width: 800.0,
            height: 600.0,
            title: "tracer".to_string(),
            ..Default::default()
        })
        .add_plugin(bevy::log::LogPlugin::default())
        .add_plugin(bevy::core::CorePlugin::default())
        .add_plugin(bevy::transform::TransformPlugin::default())
        .add_plugin(bevy::diagnostic::DiagnosticsPlugin::default())
        .add_plugin(bevy::diagnostic::LogDiagnosticsPlugin::default())
        .add_plugin(bevy::input::InputPlugin::default())
        .add_plugin(bevy::window::WindowPlugin::default())
        .add_plugin(bevy::winit::WinitPlugin::default())
        .add_plugin(bevy::asset::AssetPlugin::default())
        .add_plugin(bevy::scene::ScenePlugin::default())
        .add_plugin(rdx_renderer::RenderPlugin::default())
        .run()
}
