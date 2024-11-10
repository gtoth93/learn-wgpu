use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use wgpu::{
    AddressMode, CompareFunction, Device, Extent3d, FilterMode, ImageCopyTexture, ImageDataLayout,
    Origin3d, Queue, Sampler, SamplerDescriptor, SurfaceConfiguration, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

#[allow(clippy::struct_field_names, dead_code)]
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: TextureView,
    pub sampler: Sampler,
    pub size: Extent3d, // NEW!
}

impl Texture {
    pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

    pub fn create_depth_texture(
        device: &Device,
        config: &SurfaceConfiguration,
        label: &str,
    ) -> Self {
        let size = Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        };
        let texture_desc = TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[Self::DEPTH_FORMAT],
        };
        let texture = device.create_texture(&texture_desc);
        let view = texture.create_view(&TextureViewDescriptor::default());
        let sampler_desc = SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            compare: Some(CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        };
        let sampler = device.create_sampler(&sampler_desc);

        Self {
            texture,
            view,
            sampler,
            size, // NEW!
        }
    }

    #[allow(dead_code)]
    pub fn create_depth_texture_non_comparison_sampler(
        device: &Device,
        config: &SurfaceConfiguration,
        label: &str,
    ) -> Self {
        let size = Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        };
        let texture_desc = TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[Self::DEPTH_FORMAT],
        };
        let texture = device.create_texture(&texture_desc);
        let view = texture.create_view(&TextureViewDescriptor::default());
        let sampler_desc = SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            compare: None,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        };
        let sampler = device.create_sampler(&sampler_desc);

        Self {
            texture,
            view,
            sampler,
            size, // NEW!
        }
    }

    pub fn from_bytes(
        device: &Device,
        queue: &Queue,
        bytes: &[u8],
        label: &str,
        is_normal_map: bool,
    ) -> Result<Self> {
        let img = image::load_from_memory(bytes)?;
        Ok(Self::from_image(
            device,
            queue,
            &img,
            Some(label),
            is_normal_map,
        ))
    }

    pub fn from_image(
        device: &Device,
        queue: &Queue,
        img: &DynamicImage,
        label: Option<&str>,
        is_normal_map: bool,
    ) -> Self {
        let rgba = img.to_rgba8();
        let (image_width, image_height) = img.dimensions();

        // All textures are stored as 3D, we represent our 2D texture
        // by setting depth to 1.
        let size = Extent3d {
            width: image_width,
            height: image_height,
            depth_or_array_layers: 1,
        };
        // Most images are stored using sRGB, so we need to reflect that here.
        let format = if is_normal_map {
            TextureFormat::Rgba8Unorm
        } else {
            TextureFormat::Rgba8UnormSrgb
        };
        // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
        // COPY_DST means that we want to copy data to this texture
        let usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;

        let texture = Self::create_2d_texture(
            device,
            size.width,
            size.height,
            format,
            usage,
            FilterMode::Linear,
            label,
        );

        queue.write_texture(
            ImageCopyTexture {
                aspect: TextureAspect::All,
                texture: &texture.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
            },
            &rgba,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * image_width),
                rows_per_image: Some(image_height),
            },
            size,
        );

        texture
    }

    pub(crate) fn create_2d_texture(
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: TextureUsages,
        mag_filter: FilterMode,
        label: Option<&str>,
    ) -> Self {
        // All textures are stored as 3D, we represent our 2D texture
        // by setting depth to 1.
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        Self::create_texture(
            device,
            label,
            size,
            format,
            usage,
            TextureDimension::D2,
            mag_filter,
        )
    }

    pub fn create_texture(
        device: &Device,
        label: Option<&str>,
        size: Extent3d,
        format: TextureFormat,
        usage: TextureUsages,
        dimension: TextureDimension,
        mag_filter: FilterMode,
    ) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage,
            // This is the same as with the SurfaceConfig. It
            // specifies what texture formats can be used to
            // create TextureViews for this texture. The base
            // texture format (Rgba8UnormSrgb in this case) is
            // always supported. Note that using a different
            // texture format is not supported on the WebGL2
            // backend.
            view_formats: &[],
        });

        // We don't need to configure the texture view much, so let's
        // let wgpu define it.
        let view = texture.create_view(&TextureViewDescriptor::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            size, // NEW!
        }
    }
}

// NEW!
#[allow(clippy::module_name_repetitions)]
pub struct CubeTexture {
    texture: wgpu::Texture,
    sampler: Sampler,
    view: TextureView,
}

// NEW!
impl CubeTexture {
    #[allow(clippy::too_many_arguments)]
    pub fn create_2d(
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
        mip_level_count: u32,
        usage: TextureUsages,
        mag_filter: FilterMode,
        label: Option<&str>,
    ) -> Self {
        let texture_desc = &TextureDescriptor {
            label,
            size: Extent3d {
                width,
                height,
                // A cube has 6 sides, so we need 6 layers
                depth_or_array_layers: 6,
            },
            mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        };
        let texture = device.create_texture(texture_desc);

        let view_desc = &TextureViewDescriptor {
            label,
            dimension: Some(wgpu::TextureViewDimension::Cube),
            array_layer_count: Some(6),
            ..Default::default()
        };
        let view = texture.create_view(view_desc);

        let sampler_desc = &SamplerDescriptor {
            label,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        };
        let sampler = device.create_sampler(sampler_desc);

        Self {
            texture,
            sampler,
            view,
        }
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    pub fn view(&self) -> &TextureView {
        &self.view
    }

    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }
}
