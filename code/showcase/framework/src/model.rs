use crate::texture::{self, Texture};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3A};
use std::{ops::Range, path::Path};
use thiserror::Error;
use tobj::{LoadError, LoadOptions};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt}, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource,
    Buffer, BufferAddress, BufferUsages, Device, IndexFormat, Queue, RenderPass,
    VertexAttribute, VertexBufferLayout,
    VertexStepMode,
};

#[derive(Debug, Error)]
#[allow(clippy::enum_variant_names)]
pub enum Error {
    #[error("{0:?}")]
    ObjLoadError(#[from] LoadError),
    #[error("Directory has no parent")]
    ParentDirectoryMissing,
    #[error("{0:?}")]
    TextureError(#[from] texture::Error),
    #[error("Diffuse texture is missing")]
    DiffuseTextureMissing,
    #[error("Normal texture is missing")]
    NormalTextureMissing,
    #[error("{0:?}")]
    TryFromIntError(#[from] std::num::TryFromIntError),
}

pub trait Vertex {
    fn desc() -> VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ModelVertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
}

impl ModelVertex {
    const ATTRIBS: &'static [VertexAttribute] = &wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
        2 => Float32x3,
        3 => Float32x3,
        4 => Float32x3,
    ];
}

impl Vertex for ModelVertex {
    fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: size_of::<ModelVertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: Self::ATTRIBS,
        }
    }
}

pub struct Material<'a> {
    pub name: String,
    pub diffuse_texture: Texture<'a>,
    pub normal_texture: Texture<'a>,
    pub bind_group: BindGroup,
}

impl<'a> Material<'a> {
    #[must_use]
    pub fn new(
        device: &Device,
        name: &str,
        diffuse_texture: Texture<'a>,
        normal_texture: Texture<'a>,
        layout: &BindGroupLayout,
    ) -> Self {
        let bind_group_desc = &BindGroupDescriptor {
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
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&normal_texture.view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&normal_texture.sampler),
                },
            ],
            label: Some(name),
        };
        let bind_group = device.create_bind_group(bind_group_desc);

        Self {
            name: String::from(name),
            diffuse_texture,
            normal_texture,
            bind_group,
        }
    }
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub num_elements: u32,
    pub material: usize,
}

pub struct Model<'a> {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material<'a>>,
}

impl Model<'_> {
    /// # Errors
    ///
    /// Will return `Err` if the obj file cannot be loaded or
    /// if the obj file is in the root directory, and thus it has no parent directory or
    /// if either the diffuse or normal texture is missing next to the obj file.
    #[allow(clippy::too_many_lines)]
    pub fn load_obj<P: AsRef<Path>>(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        path: P,
    ) -> Result<Self, Error> {
        let (obj_models, obj_materials) = tobj::load_obj(
            path.as_ref(),
            &LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
        )?;

        // We're assuming that the texture files are stored with the obj file
        let containing_folder = path
            .as_ref()
            .parent()
            .ok_or(Error::ParentDirectoryMissing)?;

        let mut materials = Vec::new();
        for mat in obj_materials? {
            let diffuse_path = mat.diffuse_texture;
            let diffuse_texture = Texture::load(
                device,
                queue,
                containing_folder.join(diffuse_path.ok_or(Error::DiffuseTextureMissing)?),
                false,
            )?;

            let normal_path = mat.normal_texture;
            let normal_texture = Texture::load(
                device,
                queue,
                containing_folder.join(normal_path.ok_or(Error::NormalTextureMissing)?),
                true,
            )?;

            let material =
                Material::new(device, &mat.name, diffuse_texture, normal_texture, layout);
            materials.push(material);
        }

        let mut meshes = Vec::new();
        for m in obj_models {
            let mut vertices = Vec::new();
            for i in 0..m.mesh.positions.len() / 3 {
                let vertex = ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                    tex_coords: [m.mesh.texcoords[i * 2], m.mesh.texcoords[i * 2 + 1]],
                    normal: [
                        m.mesh.normals[i * 3],
                        m.mesh.normals[i * 3 + 1],
                        m.mesh.normals[i * 3 + 2],
                    ],
                    // We'll calculate these later
                    tangent: [0.0; 3],
                    bitangent: [0.0; 3],
                };
                vertices.push(vertex);
            }

            let indices = &m.mesh.indices;

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

                let uv0: Vec2 = v0.tex_coords.into();
                let uv1: Vec2 = v1.tex_coords.into();
                let uv2: Vec2 = v2.tex_coords.into();

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
                let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r;

                // We'll use the same tangent/bitangent for each vertex in the triangle
                vertices[c[0] as usize].tangent = tangent.into();
                vertices[c[1] as usize].tangent = tangent.into();
                vertices[c[2] as usize].tangent = tangent.into();

                vertices[c[0] as usize].bitangent = bitangent.into();
                vertices[c[1] as usize].bitangent = bitangent.into();
                vertices[c[2] as usize].bitangent = bitangent.into();
            }

            let vertex_buffer_label = &format!("{:?} Vertex Buffer", path.as_ref());
            let vertex_buffer_desc = &BufferInitDescriptor {
                label: Some(vertex_buffer_label),
                contents: bytemuck::cast_slice(&vertices),
                usage: BufferUsages::VERTEX,
            };
            let vertex_buffer = device.create_buffer_init(vertex_buffer_desc);

            let index_buffer_label = &format!("{:?} Index Buffer", path.as_ref());
            let index_buffer_desc = &BufferInitDescriptor {
                label: Some(index_buffer_label),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: BufferUsages::INDEX,
            };
            let index_buffer = device.create_buffer_init(index_buffer_desc);

            let mesh = Mesh {
                name: m.name,
                vertex_buffer,
                index_buffer,
                num_elements: u32::try_from(m.mesh.indices.len())?,
                material: m.mesh.material_id.unwrap_or(0),
            };
            meshes.push(mesh);
        }

        Ok(Self { meshes, materials })
    }
}

pub trait DrawModel<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        camera_bind_group: &'a BindGroup,
        light_bind_group: &'a BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a BindGroup,
        light_bind_group: &'a BindGroup,
    );

    fn draw_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a BindGroup,
        light_bind_group: &'a BindGroup,
    );
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a BindGroup,
        light_bind_group: &'a BindGroup,
    );
    fn draw_model_instanced_with_material(
        &mut self,
        model: &'a Model,
        material: &'a Material,
        instances: Range<u32>,
        camera_bind_group: &'a BindGroup,
        light_bind_group: &'a BindGroup,
    );
}

impl<'a, 'b> DrawModel<'b> for RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        camera_bind_group: &'b BindGroup,
        light_bind_group: &'b BindGroup,
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b BindGroup,
        light_bind_group: &'b BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), IndexFormat::Uint32);
        self.set_bind_group(0, &material.bind_group, &[]);
        self.set_bind_group(1, camera_bind_group, &[]);
        self.set_bind_group(2, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_model(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b BindGroup,
        light_bind_group: &'b BindGroup,
    ) {
        self.draw_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b BindGroup,
        light_bind_group: &'b BindGroup,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(
                mesh,
                material,
                instances.clone(),
                camera_bind_group,
                light_bind_group,
            );
        }
    }

    fn draw_model_instanced_with_material(
        &mut self,
        model: &'b Model,
        material: &'b Material,
        instances: Range<u32>,
        camera_bind_group: &'b BindGroup,
        light_bind_group: &'b BindGroup,
    ) {
        for mesh in &model.meshes {
            self.draw_mesh_instanced(
                mesh,
                material,
                instances.clone(),
                camera_bind_group,
                light_bind_group,
            );
        }
    }
}

pub trait DrawLight<'a> {
    fn draw_light_mesh(
        &mut self,
        mesh: &'a Mesh,
        camera_bind_group: &'a BindGroup,
        light_bind_group: &'a BindGroup,
    );
    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        instances: Range<u32>,
        camera_bind_group: &'a BindGroup,
        light_bind_group: &'a BindGroup,
    );

    fn draw_light_model(
        &mut self,
        model: &'a Model,
        camera_bind_group: &'a BindGroup,
        light_bind_group: &'a BindGroup,
    );
    fn draw_light_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        camera_bind_group: &'a BindGroup,
        light_bind_group: &'a BindGroup,
    );
}

impl<'a, 'b> DrawLight<'b> for RenderPass<'a>
where
    'b: 'a,
{
    fn draw_light_mesh(
        &mut self,
        mesh: &'b Mesh,
        camera_bind_group: &'b BindGroup,
        light_bind_group: &'b BindGroup,
    ) {
        self.draw_light_mesh_instanced(mesh, 0..1, camera_bind_group, light_bind_group);
    }

    fn draw_light_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        instances: Range<u32>,
        camera_bind_group: &'b BindGroup,
        light_bind_group: &'b BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), IndexFormat::Uint32);
        self.set_bind_group(0, camera_bind_group, &[]);
        self.set_bind_group(1, light_bind_group, &[]);
        self.draw_indexed(0..mesh.num_elements, 0, instances);
    }

    fn draw_light_model(
        &mut self,
        model: &'b Model,
        camera_bind_group: &'b BindGroup,
        light_bind_group: &'b BindGroup,
    ) {
        self.draw_light_model_instanced(model, 0..1, camera_bind_group, light_bind_group);
    }
    fn draw_light_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        camera_bind_group: &'b BindGroup,
        light_bind_group: &'b BindGroup,
    ) {
        for mesh in &model.meshes {
            self.draw_light_mesh_instanced(
                mesh,
                instances.clone(),
                camera_bind_group,
                light_bind_group,
            );
        }
    }
}
