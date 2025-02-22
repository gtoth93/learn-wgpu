use bytemuck::{Pod, Zeroable};
use glam::{Vec3A, Vec4};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroup, BindGroupLayout, Buffer, BufferUsages, Device,
};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LightData {
    pub position: Vec4,
    pub color: Vec4,
}

unsafe impl Pod for LightData {}
unsafe impl Zeroable for LightData {}

pub struct LightUniform {
    #[allow(dead_code)]
    data: LightData,
    #[allow(dead_code)]
    buffer: Buffer,
}

impl LightUniform {
    #[must_use]
    pub fn new(device: &Device, position: Vec3A, color: Vec3A) -> Self {
        let data = LightData {
            position: position.extend(1.0),
            color: color.extend(1.0),
        };
        let data_slice = &[data];
        let buffer_desc = &BufferInitDescriptor {
            contents: bytemuck::cast_slice(data_slice),
            usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
            label: Some("Light Buffer"),
        };
        let buffer = device.create_buffer_init(buffer_desc);

        Self { data, buffer }
    }
}

pub struct LightBinding {
    pub layout: BindGroupLayout,
    pub bind_group: BindGroup,
}
