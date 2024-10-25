use crate::{
    model::{Material, Mesh, Model, ModelVertex},
    texture::Texture,
};
use anyhow::Result;
use glam::{Vec2, Vec3};
use std::io::{BufReader, Cursor};
use tobj::LoadOptions;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, BufferUsages, Device,
    Queue,
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

pub async fn load_texture(file_name: &str, device: &Device, queue: &Queue) -> Result<Texture> {
    let data = load_binary(file_name).await?;
    Texture::from_bytes(device, queue, &data, file_name)
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
        if let Some(texture_file_name) = m.diffuse_texture {
            let diffuse_texture = load_texture(&texture_file_name, device, queue).await?;
            let bind_group_desc = BindGroupDescriptor {
                layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&diffuse_texture.view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&diffuse_texture.sampler),
                    },
                ],
                label: None,
            };
            let bind_group = device.create_bind_group(&bind_group_desc);

            let material = Material {
                name: m.name,
                diffuse_texture,
                bind_group,
            };
            materials.push(material);
        }
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
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
                        }
                    }
                })
                .collect::<Vec<_>>();

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
