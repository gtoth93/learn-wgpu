use crate::buffer::RawBuffer;
use image::{DynamicImage, GenericImageView, ImageError};
use std::path::Path;
use thiserror::Error;
use wgpu::{
    AddressMode, BufferAddress, BufferDescriptor, BufferUsages, CompareFunction, Device, Extent3d,
    FilterMode, Origin3d, Queue, Sampler, SamplerDescriptor, SurfaceConfiguration,
    TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0:?}")]
    ImageError(#[from] ImageError),
    #[error("Path is not valid unicode")]
    PathNotUnicode,
}

pub struct Texture<'a> {
    pub texture: wgpu::Texture,
    pub view: TextureView,
    pub sampler: Sampler,
    pub desc: TextureDescriptor<'a>,
}

impl<'a> Texture<'a> {
    pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

    /// # Errors
    ///
    /// Will return `Err` if the image cannot be opened.
    pub fn load<P: AsRef<Path>>(
        device: &Device,
        queue: &Queue,
        path: P,
        is_normal_map: bool,
    ) -> Result<Self, Error> {
        let path_copy = path.as_ref().to_path_buf();
        let label = path_copy.to_str().ok_or(Error::PathNotUnicode)?;
        let img = image::open(path)?;
        let texture = Self::from_image(device, queue, &img, Some(label), is_normal_map);
        Ok(texture)
    }

    #[must_use]
    pub fn from_descriptor(device: &Device, desc: TextureDescriptor<'a>) -> Self {
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&TextureViewDescriptor::default());
        let sampler_desc = &SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: Some(CompareFunction::LessEqual),
            ..Default::default()
        };
        let sampler = device.create_sampler(sampler_desc);

        Self {
            texture,
            view,
            sampler,
            desc,
        }
    }

    /// # Errors
    ///
    /// Will return `Err` if the image cannot be loaded.
    pub fn from_bytes(
        device: &Device,
        queue: &Queue,
        label: Option<&str>,
        is_normal_map: bool,
        bytes: &[u8],
    ) -> Result<Self, Error> {
        let img = image::load_from_memory(bytes)?;
        Ok(Self::from_image(device, queue, &img, label, is_normal_map))
    }

    #[must_use]
    pub fn from_image(
        device: &Device,
        queue: &Queue,
        img: &DynamicImage,
        _label: Option<&str>,
        is_normal_map: bool,
    ) -> Self {
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();

        let size = Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let format = if is_normal_map {
            TextureFormat::Rgba8Unorm
        } else {
            TextureFormat::Rgba8UnormSrgb
        };
        let desc = TextureDescriptor {
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            label: None,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);

        queue.write_texture(
            TexelCopyTextureInfo {
                aspect: TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
            },
            &rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&TextureViewDescriptor::default());
        let sampler_desc = &SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: Some(CompareFunction::Always),
            ..Default::default()
        };
        let sampler = device.create_sampler(sampler_desc);

        Self {
            texture,
            view,
            sampler,
            desc,
        }
    }

    #[must_use]
    pub fn create_depth_texture(device: &Device, config: &SurfaceConfiguration) -> Self {
        let desc = TextureDescriptor {
            label: None,
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[Self::DEPTH_FORMAT],
        };
        Self::from_descriptor(device, desc)
    }

    #[must_use]
    pub fn prepare_buffer_rgba(&self, device: &Device) -> RawBuffer<[f32; 4]> {
        let num_pixels =
            self.desc.size.width * self.desc.size.height * self.desc.size.depth_or_array_layers;

        let buffer_size = num_pixels * size_of::<[f32; 4]>() as u32;
        let buffer_usage = BufferUsages::COPY_DST | BufferUsages::MAP_READ;
        let buffer_desc = BufferDescriptor {
            size: BufferAddress::from(buffer_size),
            usage: buffer_usage,
            label: None,
            mapped_at_creation: false,
        };
        let buffer = device.create_buffer(&buffer_desc);

        let data = Vec::with_capacity(num_pixels as usize);

        RawBuffer::from_parts(buffer, data, buffer_usage)
    }
}
