use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BufferAddress, BufferDescriptor, BufferUsages, Device,
};

pub trait ToRaw {
    type Output;
    fn to_raw(&self) -> Self::Output;
}

pub struct RawBuffer<R>
where
    R: Copy + Pod + Zeroable,
{
    pub buffer: wgpu::Buffer,
    pub data: Vec<R>,
}

impl<R: Copy + Pod + Zeroable> RawBuffer<R> {
    pub fn from_slice<T: ToRaw<Output = R>>(
        device: &Device,
        data: &[T],
        usage: BufferUsages,
    ) -> Self {
        let raw_data = data.iter().map(ToRaw::to_raw).collect::<Vec<R>>();
        Self::from_vec(device, raw_data, usage)
    }

    #[must_use]
    pub fn from_vec(device: &Device, data: Vec<R>, usage: BufferUsages) -> Self {
        let buffer_desc = &BufferInitDescriptor {
            contents: bytemuck::cast_slice(&data),
            usage,
            label: None,
        };
        let buffer = device.create_buffer_init(buffer_desc);
        Self::from_parts(buffer, data, usage)
    }

    #[must_use]
    pub fn from_parts(buffer: wgpu::Buffer, data: Vec<R>, _usage: BufferUsages) -> Self {
        Self { buffer, data }
    }

    #[must_use]
    pub fn buffer_size(&self) -> BufferAddress {
        (self.data.len() * size_of::<R>()) as BufferAddress
    }
}

pub struct Buffer<U: ToRaw<Output = R>, R: Copy + Pod + Zeroable> {
    pub data: Vec<U>,
    pub raw_buffer: RawBuffer<R>,
    pub usage: BufferUsages,
}

impl<U: ToRaw<Output = R>, R: Copy + Pod + Zeroable> Buffer<U, R> {
    pub fn uniform(device: &Device, datum: U) -> Self {
        let data = vec![datum];
        let usage = BufferUsages::UNIFORM | BufferUsages::COPY_DST;
        Self::with_usage(device, data, usage)
    }

    #[must_use]
    pub fn storage(device: &Device, data: Vec<U>) -> Self {
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        Self::with_usage(device, data, usage)
    }

    #[must_use]
    pub fn staging(device: &Device, other: &Self) -> Self {
        let buffer_size = other.raw_buffer.buffer_size();
        let usage = BufferUsages::COPY_SRC | BufferUsages::MAP_READ | BufferUsages::MAP_WRITE;
        let buffer_desc = &BufferDescriptor {
            size: buffer_size,
            usage,
            label: None,
            mapped_at_creation: false,
        };
        let buffer = device.create_buffer(buffer_desc);
        let raw_buffer = RawBuffer::from_parts(buffer, Vec::new(), usage);
        Self::from_parts(Vec::new(), raw_buffer, usage)
    }

    #[must_use]
    pub fn with_usage(device: &Device, data: Vec<U>, usage: BufferUsages) -> Self {
        let raw_buffer = RawBuffer::from_slice(device, &data, usage);
        Self::from_parts(data, raw_buffer, usage)
    }

    #[must_use]
    pub fn from_parts(data: Vec<U>, raw_buffer: RawBuffer<R>, usage: BufferUsages) -> Self {
        Self {
            data,
            raw_buffer,
            usage,
        }
    }
}
