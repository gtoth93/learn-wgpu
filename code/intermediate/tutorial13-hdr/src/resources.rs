use crate::{
    model::{Material, Mesh, Model, ModelVertex},
    texture::{CubeTexture, Texture},
};
use anyhow::Result;
use glam::{Vec2, Vec3, Vec3A};
use image::{GenericImageView, ImageFormat};
use std::io::{BufReader, Cursor};
use tobj::LoadOptions;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, FilterMode,
    Origin3d, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, ShaderStages,
    StorageTextureAccess, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let mut origin = location.origin().unwrap();
    if !origin.ends_with("learn-wgpu") {
        origin = format!("{}/learn-wgpu", origin);
    }
    let base = reqwest::Url::parse(&format!("{}/", origin,)).unwrap();
    base.join(file_name).unwrap()
}

#[allow(clippy::unused_async)]
pub async fn load_string(file_name: &str) -> Result<String> {
    #[cfg(target_arch = "wasm32")]
    {
        let url = format_url(file_name);
        let txt = reqwest::get(url).await?.text().await?;
        Ok(txt)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::path::Path;
        let path = Path::new(env!("OUT_DIR")).join("res").join(file_name);
        let txt = std::fs::read_to_string(path)?;
        Ok(txt)
    }
}

#[allow(clippy::unused_async)]
pub async fn load_binary(file_name: &str) -> Result<Vec<u8>> {
    #[cfg(target_arch = "wasm32")]
    {
        let url = format_url(file_name);
        let data = reqwest::get(url).await?.bytes().await?.to_vec();
        Ok(data)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::path::Path;
        let path = Path::new(env!("OUT_DIR")).join("res").join(file_name);
        let data = std::fs::read(path)?;
        Ok(data)
    }
}

pub async fn load_texture(
    file_name: &str,
    is_normal_map: bool,
    device: &Device,
    queue: &Queue,
) -> Result<Texture> {
    let data = load_binary(file_name).await?;
    Texture::from_bytes(device, queue, &data, file_name, is_normal_map)
}

#[allow(clippy::too_many_lines)]
pub async fn load_model(
    file_name: &str,
    device: &Device,
    queue: &Queue,
    layout: &BindGroupLayout,
) -> Result<Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let mat_text = load_string(&p).await.unwrap();
            tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
        },
    )
    .await?;

    let mut materials = Vec::new();
    for m in obj_materials? {
        if let tobj::Material {
            diffuse_texture: Some(diffuse_file_name),
            normal_texture: Some(normal_file_name),
            ..
        } = m
        {
            let diffuse_texture = load_texture(&diffuse_file_name, false, device, queue).await?;
            let normal_texture = load_texture(&normal_file_name, true, device, queue).await?;

            materials.push(Material::new(
                device,
                &m.name,
                diffuse_texture,
                normal_texture,
                layout,
            ));
        }
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let mut vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| {
                    if m.mesh.normals.is_empty() {
                        ModelVertex {
                            position: Vec3::new(
                                m.mesh.positions[i * 3],
                                m.mesh.positions[i * 3 + 1],
                                m.mesh.positions[i * 3 + 2],
                            ),
                            tex_coords: Vec2::new(
                                m.mesh.texcoords[i * 2],
                                1.0 - m.mesh.texcoords[i * 2 + 1],
                            ),
                            normal: Vec3::ZERO,
                            // We'll calculate these later
                            tangent: Vec3::ZERO,
                            bitangent: Vec3::ZERO,
                        }
                    } else {
                        ModelVertex {
                            position: Vec3::new(
                                m.mesh.positions[i * 3],
                                m.mesh.positions[i * 3 + 1],
                                m.mesh.positions[i * 3 + 2],
                            ),
                            tex_coords: Vec2::new(
                                m.mesh.texcoords[i * 2],
                                1.0 - m.mesh.texcoords[i * 2 + 1],
                            ),
                            normal: Vec3::new(
                                m.mesh.normals[i * 3],
                                m.mesh.normals[i * 3 + 1],
                                m.mesh.normals[i * 3 + 2],
                            ),
                            // We'll calculate these later
                            tangent: Vec3::ZERO,
                            bitangent: Vec3::ZERO,
                        }
                    }
                })
                .collect::<Vec<_>>();

            let indices = &m.mesh.indices;
            let mut triangles_included: Vec<u16> = vec![0; vertices.len()];

            // Calculate tangents and bitangets. We're going to
            // use the triangles, so we need to loop through the
            // indices in chunks of 3
            for c in indices.chunks(3) {
                let v0 = vertices[c[0] as usize];
                let v1 = vertices[c[1] as usize];
                let v2 = vertices[c[2] as usize];

                let pos0: Vec3A = v0.position.into();
                let pos1: Vec3A = v1.position.into();
                let pos2: Vec3A = v2.position.into();

                let uv0 = v0.tex_coords;
                let uv1 = v1.tex_coords;
                let uv2 = v2.tex_coords;

                // Calculate the edges of the triangle
                let delta_pos1 = pos1 - pos0;
                let delta_pos2 = pos2 - pos0;

                // This will give us a direction to calculate the
                // tangent and bitangent
                let delta_uv1 = uv1 - uv0;
                let delta_uv2 = uv2 - uv0;

                // Solving the following system of equations will
                // give us the tangent and bitangent.
                //     delta_pos1 = delta_uv1.x * T + delta_u.y * B
                //     delta_pos2 = delta_uv2.x * T + delta_uv2.y * B
                // Luckily, the place I found this equation provided
                // the solution!
                let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
                // We flip the bitangent to enable right-handed normal
                // maps with wgpu texture coordinate system
                let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * -r;

                // We'll use the same tangent/bitangent for each vertex in the triangle
                vertices[c[0] as usize].tangent = (tangent + Vec3A::from(v0.tangent)).into();
                vertices[c[1] as usize].tangent = (tangent + Vec3A::from(v1.tangent)).into();
                vertices[c[2] as usize].tangent = (tangent + Vec3A::from(v2.tangent)).into();
                vertices[c[0] as usize].bitangent = (bitangent + Vec3A::from(v0.bitangent)).into();
                vertices[c[1] as usize].bitangent = (bitangent + Vec3A::from(v1.bitangent)).into();
                vertices[c[2] as usize].bitangent = (bitangent + Vec3A::from(v2.bitangent)).into();

                // Used to average the tangents/bitangents
                triangles_included[c[0] as usize] += 1;
                triangles_included[c[1] as usize] += 1;
                triangles_included[c[2] as usize] += 1;
            }

            // Average the tangents/bitangents
            for (i, n) in triangles_included.into_iter().enumerate() {
                let denom = 1.0 / f32::from(n);
                let v = &mut vertices[i];
                v.tangent = (Vec3A::from(v.tangent) * denom).into();
                v.bitangent = (Vec3A::from(v.bitangent) * denom).into();
            }

            let vertex_buffer_label = format!("{file_name:?} Vertex Buffer");
            let vertex_buffer_desc = BufferInitDescriptor {
                label: Some(&vertex_buffer_label),
                contents: bytemuck::cast_slice(&vertices),
                usage: BufferUsages::VERTEX,
            };
            let vertex_buffer = device.create_buffer_init(&vertex_buffer_desc);
            let index_buffer_label = format!("{file_name:?} Index Buffer");
            let index_buffer_desc = BufferInitDescriptor {
                label: Some(&index_buffer_label),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: BufferUsages::INDEX,
            };
            let index_buffer = device.create_buffer_init(&index_buffer_desc);

            tracing::info!("Mesh: {}", m.name);
            Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len().try_into().unwrap(),
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(Model { meshes, materials })
}

pub struct HdrLoader {
    texture_format: TextureFormat,
    equirect_layout: BindGroupLayout,
    equirect_to_cubemap: ComputePipeline,
}

impl HdrLoader {
    pub fn new(device: &Device) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("equirectangular.wgsl"));
        let texture_format = TextureFormat::Rgba32Float;
        let equirect_layout_desc = &BindGroupLayoutDescriptor {
            label: Some("HdrLoader::equirect_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: texture_format,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        };
        let equirect_layout = device.create_bind_group_layout(equirect_layout_desc);

        let pipeline_layout_desc = &PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&equirect_layout],
            push_constant_ranges: &[],
        };
        let pipeline_layout = device.create_pipeline_layout(pipeline_layout_desc);

        let equirect_to_cubemap_desc = &ComputePipelineDescriptor {
            label: Some("equirect_to_cubemap"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("compute_equirect_to_cubemap"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        };
        let equirect_to_cubemap = device.create_compute_pipeline(equirect_to_cubemap_desc);

        Self {
            texture_format,
            equirect_layout,
            equirect_to_cubemap,
        }
    }

    pub fn load_from_equirectangular_bytes(
        &self,
        device: &Device,
        queue: &Queue,
        data: &[u8],
        dst_size: u32,
        label: Option<&str>,
    ) -> Result<CubeTexture> {
        let image = image::load_from_memory_with_format(data, ImageFormat::Hdr)?;
        let (width, height) = image.dimensions();

        let pixels = image.into_rgba32f().into_raw();

        let src = Texture::create_2d_texture(
            device,
            width,
            height,
            self.texture_format,
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            FilterMode::Linear,
            None,
        );

        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &src.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            bytemuck::cast_slice(&pixels),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(src.size.width * u32::try_from(size_of::<[f32; 4]>())?),
                rows_per_image: Some(src.size.height),
            },
            src.size,
        );

        let dst = CubeTexture::create_2d(
            device,
            dst_size,
            dst_size,
            self.texture_format,
            1,
            // We are going to write to `dst` texture so we
            // need to use a `STORAGE_BINDING`.
            TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            FilterMode::Nearest,
            label,
        );

        let dst_view_desc = &TextureViewDescriptor {
            label,
            // Normally, you'd use `TextureViewDimension::Cube`
            // for a cube texture, but we can't use that
            // view dimension with a `STORAGE_BINDING`.
            // We need to access the cube texture layers
            // directly.
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        };
        let dst_view = dst.texture().create_view(dst_view_desc);

        let bind_group_desc = &BindGroupDescriptor {
            label,
            layout: &self.equirect_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&src.view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&dst_view),
                },
            ],
        };
        let bind_group = device.create_bind_group(bind_group_desc);

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        let pass_desc = &ComputePassDescriptor {
            label,
            timestamp_writes: None,
        };
        let mut pass = encoder.begin_compute_pass(pass_desc);

        let num_workgroups = (dst_size + 15) / 16;
        pass.set_pipeline(&self.equirect_to_cubemap);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, num_workgroups, 6);

        drop(pass);

        queue.submit([encoder.finish()]);

        Ok(dst)
    }
}
