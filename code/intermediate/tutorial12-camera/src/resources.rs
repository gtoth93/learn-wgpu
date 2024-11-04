use crate::{
    model::{Material, Mesh, Model, ModelVertex},
    texture::Texture,
};
use anyhow::Result;
use glam::{Vec2, Vec3, Vec3A};
use std::io::{BufReader, Cursor};
use tobj::LoadOptions;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupLayout, BufferUsages, Device, Queue,
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
