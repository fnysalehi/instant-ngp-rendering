/** @file   testbed_geometry.cu
 *  @author Fatemeh Salehi
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/discrete_distribution.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/random_val.cuh> // helpers to generate random values, directions
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/takikawa_encoding.cuh>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/geometry.h>
#include <neural-graphics-primitives/tinyobj_loader_wrapper.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>
#include <neural-graphics-primitives/geometry_bvh.cuh>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/trainer.h>

namespace ngp {


// TODO: all m_aabb s should be changed to local/node bounding boxes and passed/stored in the node 



// struct MeshPayload {
// 	vec3 dir; // direction of the ray
// 	uint32_t idx; // index of the ray
// 	uint16_t n_steps;
// 	bool alive;
// 	// uint32_t triangle_id; // ID of the triangle that the ray hit
//     // vec3 hit_point; // point where the ray hit the triangle
//     // vec3 normal; // normal of the triangle at the hit point

// };

// struct RaysMeshSoa {	// most probably this is not correct
// #if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
//     void copy_from_other_async(uint32_t n_elements, const RaysMeshSoa& other, cudaStream_t stream) {
//         CUDA_CHECK_THROW(cudaMemcpyAsync(origin, other.origin, n_elements * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
//         CUDA_CHECK_THROW(cudaMemcpyAsync(dir, other.dir, n_elements * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
//         CUDA_CHECK_THROW(cudaMemcpyAsync(payload, other.payload, n_elements * sizeof(MeshPayload), cudaMemcpyDeviceToDevice, stream));
//     }
// #endif

//     void set(vec3* origin, vec3* dir, MeshPayload* payload) {
//         this->origin = origin;
//         this->dir = dir;
//         this->payload = payload;
//     }

//     vec3* origin;	//maybe pos?
//     vec3* dir;
//     MeshPayload* payload;
// };


__global__ void shade_kernel_geometry(
	const uint32_t n_elements,
	BoundingBox aabb,
	float floor_y,
	const ERenderMode mode::Shade,
	const BRDFParams brdf,
	vec3 sun_dir,
	vec3 up_dir,
	mat4x3 camera_matrix,
	vec3* __restrict__ positions,
	vec3* __restrict__ normals,
	float* __restrict__ distances,
	geometryPayload* __restrict__ payloads,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	GeometryPayload& payload = payloads[i];
	if (!aabb.contains(positions[i])) {
		return;
	}

	// The normal in memory isn't normalized yet
	vec3 normal = normalize(normals[i]);
	vec3 pos = positions[i];
	bool floor = false;
	if (pos.y < floor_y + 0.001f && payload.dir.y < 0.f) {
		normal = vec3{0.0f, 1.0f, 0.0f};
		floor = true;
	}

	vec3 cam_pos = camera_matrix[3];
	vec3 cam_fwd = camera_matrix[2];
	float ao = powf(0.92f, payload.n_steps * 0.5f) * (1.f / 0.92f);
	vec3 color;

	float skyam = -dot(normal, up_dir) * 0.5f + 0.5f;
	vec3 suncol = vec3{255.f/255.0f, 225.f/255.0f, 195.f/255.0f} * 4.f * distances[i]; // Distance encodes shadow occlusion. 0=occluded, 1=no shadow
	const vec3 skycol = vec3{195.f/255.0f, 215.f/255.0f, 255.f/255.0f} * 4.f * skyam;
	float check_size = 8.f/aabb.diag().x;
	float check=((int(floorf(check_size*(pos.x-aabb.min.x)))^int(floorf(check_size*(pos.z-aabb.min.z)))) &1) ? 0.8f : 0.2f;
	const vec3 floorcol = vec3{check*check*check, check*check, check};
	color = evaluate_shading(
		floor ? floorcol : brdf.basecolor * brdf.basecolor,
		brdf.ambientcolor * skycol,
		suncol,
		floor ? 0.f : brdf.metallic,
		floor ? 0.f : brdf.subsurface,
		floor ? 1.f : brdf.specular,
		floor ? 0.5f : brdf.roughness,
		0.f,
		floor ? 0.f : brdf.sheen,
		0.f,
		floor ? 0.f : brdf.clearcoat,
		brdf.clearcoat_gloss,
		sun_dir,
		-normalize(payload.dir),
		normal
	);

	frame_buffer[payload.idx] = {color.r, color.g, color.b, 1.0f};
	depth_buffer[payload.idx] = dot(cam_fwd, pos - cam_pos);
}



// separates the "alive" and "dead" elements of the input arrays into two separate arrays
__global__ void compact_kernel_geometry(
	const uint32_t n_elements,
	const float zero_offset,
	vec3* src_positions, float* src_distances, GeometryPayload* src_payloads,
	vec3* dst_positions, float* dst_distances, GeometryPayload* dst_payloads,
	vec3* dst_final_positions, float* dst_final_distances, GeometryPayload* dst_final_payloads,
	BoundingBox aabb,
	uint32_t* counter, uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	GeometryPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_positions[idx] = src_positions[i];
		dst_distances[idx] = src_distances[i];
	} else if (aabb.contains(src_positions[i])) {
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_positions[idx] = src_positions[i];
		dst_final_distances[idx] = 1.0f; // HACK: Distances encode shadowing factor when shading
	}
}

__global__ void init_rays_with_payload_kernel_geometry(
	uint32_t sample_index,
	vec3* __restrict__ positions,
	float* __restrict__ distances,
	GeometryPayload* __restrict__ payloads,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox aabb,
	float floor_y,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Buffer2DView<const vec4> envmap,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	const TriangleOctreeNode* __restrict__ octree_nodes = nullptr,
	int max_octree_depth = 0
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	Ray ray = pixel_to_ray(
		sample_index,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask
	);

	distances[idx] = MAX_DEPTH();
	depth_buffer[idx] = MAX_DEPTH();

	GeometryPayload& payload = payloads[idx];

	if (!ray.is_valid()) {
		payload.dir = ray.d;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		positions[idx] = ray.o;
		return;
	}

	if (plane_z < 0) {
		float n = length(ray.d);
		payload.dir = (1.0f/n) * ray.d;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		positions[idx] = ray.o - plane_z * ray.d;
		depth_buffer[idx] = -plane_z;
		return;
	}

	ray.d = normalize(ray.d);
	float t = max(aabb.ray_intersect(ray.o, ray.d).x, 0.0f);

	ray.advance(t + 1e-6f);

	if (octree_nodes && !TriangleOctree::contains(octree_nodes, max_octree_depth, ray.o)) {
		t = max(0.0f, TriangleOctree::ray_intersect(octree_nodes, max_octree_depth, ray.o, ray.d));
		if (ray.o.y > floor_y && ray.d.y < 0.f) {
			float floor_dist = -(ray.o.y - floor_y) / ray.d.y;
			if (floor_dist > 0.f) {
				t = min(t, floor_dist);
			}
		}

		ray.advance(t + 1e-6f);
	}

	positions[idx] = ray.o;

	// if (envmap) {
	// 	frame_buffer[idx] = read_envmap(envmap, ray.d);
	// }

	payload.dir = ray.d;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = aabb.contains(ray.o);
}


__global__ void init_rays_with_payload_kernel_geometry(
    uint32_t sample_index,
    vec3* __restrict__ positions,
    float* __restrict__ distances,
    GeometryPayload* __restrict__ payloads,
    ivec2 resolution,
    vec2 focal_length,
    mat4x3 camera_matrix,
    vec2 screen_center,
    vec3 parallax_shift,
    bool snap_to_pixel_centers,
    BoundingBox aabb,
    float floor_y,
    float near_distance,
    float plane_z,
    float aperture_size,
    Foveation foveation,
    Buffer2DView<const vec4> envmap,
    vec4* __restrict__ frame_buffer,
    float* __restrict__ depth_buffer,
    Buffer2DView<const uint8_t> hidden_area_mask,
    const TriangleOctreeNode* __restrict__ octree_nodes = nullptr,
    int max_octree_depth = 0
) {
    uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	Ray ray = pixel_to_ray(
		sample_index,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask
	);

	distances[idx] = MAX_DEPTH();	// I don't know what this is for	
	depth_buffer[idx] = MAX_DEPTH();

    GeometryPayload& payload = payloads[idx];

    if (!ray.is_valid()) {
		payload.dir = ray.d;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		positions[idx] = ray.o;
		return;
	}

	if (plane_z < 0) {
		float n = length(ray.d);
		payload.dir = (1.0f/n) * ray.d;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		positions[idx] = ray.o - plane_z * ray.d;
		depth_buffer[idx] = -plane_z;
		return;
	}

	ray.d = normalize(ray.d);
    float t = max(aabb.ray_intersect(ray.o, ray.d).x, 0.0f);

	ray.advance(t + 1e-6f);

    if (octree_nodes && !TriangleOctree::contains(octree_nodes, max_octree_depth, ray.o)) {
        t = max(0.0f, TriangleOctree::ray_intersect(octree_nodes, max_octree_depth, ray.o, ray.d));
        if (ray.o.y > floor_y && ray.d.y < 0.f) {
			float floor_dist = -(ray.o.y - floor_y) / ray.d.y;
			if (floor_dist > 0.f) {
				t = min(t, floor_dist);
			}
		}

		ray.advance(t + 1e-6f);
    }

    positions[idx] = ray.o;

	if (envmap) {
		frame_buffer[idx] = read_envmap(envmap, ray.d);
	}

    payload.dir = ray.d;
    payload.idx = idx;
    payload.n_steps = 0;
    payload.alive = aabb.contains(ray.o);
}

void Testbed::MyTracer::init_rays_from_camera(
	uint32_t sample_index,
	const ivec2& resolution,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	const vec3& parallax_shift,
	bool snap_to_pixel_centers,
	const BoundingBox& aabb,
	float floor_y,
	float near_distance,
	float plane_z,
	float aperture_size,
	const Foveation& foveation,
	const Buffer2DView<const vec4>& envmap,
	vec4* frame_buffer,
	float* depth_buffer,
	const Buffer2DView<const uint8_t>& hidden_area_mask,
	const TriangleOctree* octree,
	uint32_t n_octree_levels,
	cudaStream_t stream
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	size_t n_pixels = (size_t)resolution.x * resolution.y;
	enlarge(n_pixels, stream);

	// defining the grid and block dimensions for launching the CUDA kernel
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1 };
	init_rays_with_payload_kernel_geometry<<<blocks, threads, 0, stream>>>(
		sample_index,
		m_rays[0].pos,
		m_rays[0].distance,
		m_rays[0].payload,
		resolution,
		focal_length,
		camera_matrix,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		aabb,
		floor_y,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		envmap,
		frame_buffer,
		depth_buffer,
		hidden_area_mask,
		octree ? octree->nodes_gpu() : nullptr,
		octree ? n_octree_levels : 0
	);
	m_n_rays_initialized = (uint32_t)n_pixels;
}

uint32_t Testbed::MyTracer::trace_bvh(TriangleBvh* bvh, const Triangle* triangles, cudaStream_t stream) {
	uint32_t n_alive = m_n_rays_initialized;
	m_n_rays_initialized = 0;

	if (!bvh) {
		return 0;
	}

	// Abuse the normal buffer to temporarily hold ray directions
	parallel_for_gpu(stream, n_alive, [payloads=m_rays[0].payload, normals=m_rays[0].normal] __device__ (size_t i) {
		normals[i] = payloads[i].dir;
	});

	//  if optix is available, uses optix.raytrace->invoke
	bvh->ray_trace_gpu(n_alive, m_rays[0].pos, m_rays[0].normal, triangles, stream);
	return n_alive;
}

uint32_t Testbed::MyTracer::trace(
    const Triangle* triangles,
    uint32_t num_triangles,
    float zero_offset,
    float distance_scale,
    float maximum_distance,
    const BoundingBox& aabb,
    const float floor_y,
    const TriangleOctree* octree,
    const uint32_t n_octree_levels,
    cudaStream_t stream
) {
    if (m_n_rays_initialized == 0) {
        return 0;
    }

    CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter, 0, sizeof(uint32_t), stream));

    const uint32_t STEPS_INBETWEEN_COMPACTION = 4;

    uint32_t n_alive = m_n_rays_initialized;
    m_n_rays_initialized = 0;

    uint32_t i = 1;
    uint32_t double_buffer_index = 0;
    while (i < MARCH_ITER) {
        uint32_t step_size = std::min(i, STEPS_INBETWEEN_COMPACTION);

		// double buffer
        GeometryPayload& rays_current = m_rays[(double_buffer_index+1)%2];
        GeometryPayload& rays_tmp = m_rays[double_buffer_index%2];
        ++double_buffer_index;

        // Compact rays that did not diverge yet
        {
            CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
            linear_kernel(compact_kernel_geometry, 0, stream,
                n_alive,
                zero_offset,
                rays_tmp.pos, rays_tmp.distance, rays_tmp.payload,
                rays_current.pos, rays_current.distance, rays_current.payload,
                m_rays_hit.pos, m_rays_hit.distance, m_rays_hit.payload,
                aabb,
                m_alive_counter, m_hit_counter
            );
            CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        }

        if (n_alive == 0) {
            break;
        }

        for (uint32_t j = 0; j < step_size; ++j) {
            linear_kernel(ray_triangle_intersection_kernel, 0, stream,
                n_alive,
                rays_current.pos,
                rays_current.distance,
                rays_current.payload,
                aabb,
                floor_y,
                triangles,
                num_triangles,
                distance_scale,
                maximum_distance
            );
        }

        i += step_size;
    }

    uint32_t n_hit;
    CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    return n_hit;
}

// allocate and distribute workspace memory for rays
void Testbed::MyTracer::enlarge(size_t n_elements, cudaStream_t stream) {
	n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
	auto scratch = allocate_workspace_and_distribute<
		vec3, vec3, float, float, float, float, GeometryPayload, // m_rays[0]
		vec3, vec3, float, float, float, float, GeometryPayload, // m_rays[1]
		vec3, vec3, float, float, float, float, GeometryPayload, // m_rays_hit

		uint32_t,
		uint32_t
	>(
		stream, &m_scratch_alloc,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), std::get<6>(scratch));
	m_rays[1].set(std::get<7>(scratch), std::get<8>(scratch), std::get<9>(scratch), std::get<10>(scratch), std::get<11>(scratch), std::get<12>(scratch), std::get<13>(scratch));
	m_rays_hit.set(std::get<14>(scratch), std::get<15>(scratch), std::get<16>(scratch), std::get<17>(scratch), std::get<18>(scratch), std::get<19>(scratch), std::get<20>(scratch));

	m_hit_counter = std::get<21>(scratch);
	m_alive_counter = std::get<22>(scratch);
}



// todo: tracer for nerfs and tracer for the mesh based objects
// init_rays_from_camera, init_rays_from_data, enlarge
void Testbed::render_geometry(
	cudaStream_t stream,
		CudaDevice& device,
		const CudaRenderBufferView& render_buffer,
		const vec2& focal_length,
		const mat4x3& camera_matrix,
		const vec2& screen_center,
		const Foveation& foveation,
		int visualized_dimension
) {
	float plane_z = m_slice_plane_z + m_scale;
	float distance_scale = 1.f/std::max(m_volume.inv_distance_scale,0.01f);
	auto res = render_buffer.resolution;
}

void Testbed::render_mesh(
    cudaStream_t stream,
	const distance_fun_t& distance_function,
	const normals_fun_t& normals_function,
	const CudaRenderBufferView& render_buffer,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	const Foveation& foveation,
	int visualized_dimension,
	const BoundingBox& aabb
) {
	float plane_z = m_slice_plane_z + m_scale;
	
	auto* octree_ptr = m_meshData.triangle_octree.get();

	MyTracer tracer;

	uint32_t n_octree_levels = octree_ptr ? octree_ptr->depth() : 0;

	BoundingBox mesh_bounding_box = aabb;
	mesh_bounding_box.inflate(m_meshData.zero_offset);
	tracer.init_rays_from_camera(
		render_buffer.spp,
		render_buffer.resolution,
		focal_length,
		camera_matrix,
		screen_center,
		m_parallax_shift,
		m_snap_to_pixel_centers,
		mesh_bounding_box,
		get_floor_y(),
		m_render_near_distance,
		plane_z,
		m_aperture_size,
		foveation,
		m_envmap.inference_view(),
		render_buffer.frame_buffer,
		render_buffer.depth_buffer,
		render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
		octree_ptr,
		n_octree_levels,
		stream
	);

	auto trace = [&](MyTracer& tracer) {
		return tracer.trace_bvh(m_meshData.triangle_bvh.get(), m_meshData.triangles_gpu.data(), stream);
	};

	uint32_t n_hit = trace(tracer);

	RaysMeshSoa& rays_hit = tracer.rays_hit();

	normals_function(n_hit, rays_hit.pos, rays_hit.normal, stream);

	linear_kernel(shade_kernel_geometry, 0, stream,
		n_hit,
		aabb,
		get_floor_y(),
		render_mode,
		m_meshData.brdf,
		normalize(m_sun_dir),
		normalize(m_up_dir),
		camera_matrix,
		rays_hit.pos,
		rays_hit.normal,
		rays_hit.distance,
		rays_hit.payload,
		render_buffer.frame_buffer,
		render_buffer.depth_buffer
	);
}
// 

// /****************************************************/
// /*  The following code is from testbed_nerf.cu file */

void Testbed::NerfTracer::init_rays_from_camera(
	uint32_t sample_index,
	uint32_t padded_output_width,
	uint32_t n_extra_dims,
	const ivec2& resolution,
	const vec2& focal_length,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec4& rolling_shutter,
	const vec2& screen_center,
	const vec3& parallax_shift,
	bool snap_to_pixel_centers,
	const BoundingBox& render_aabb,
	const mat3& render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	const Foveation& foveation,
	const Lens& lens,
	const Buffer2DView<const vec4>& envmap,
	const Buffer2DView<const vec2>& distortion,
	vec4* frame_buffer,
	float* depth_buffer,
	const Buffer2DView<const uint8_t>& hidden_area_mask,
	const uint8_t* grid,
	int show_accel,
	uint32_t max_mip,
	float cone_angle_constant,
	ERenderMode render_mode,
	cudaStream_t stream
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	size_t n_pixels = (size_t)resolution.x * resolution.y;
	enlarge(n_pixels, padded_output_width, n_extra_dims, stream);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1 };
	init_rays_with_payload_kernel_nerf<<<blocks, threads, 0, stream>>>(
		sample_index,
		m_rays[0].payload,
		resolution,
		focal_length,
		camera_matrix0,
		camera_matrix1,
		rolling_shutter,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		render_aabb,
		render_aabb_to_local,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		lens,
		envmap,
		frame_buffer,
		depth_buffer,
		hidden_area_mask,
		distortion,
		render_mode
	);

	m_n_rays_initialized = resolution.x * resolution.y;

	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(vec4), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), stream));

	linear_kernel(advance_pos_nerf_kernel, 0, stream,
		m_n_rays_initialized,
		render_aabb,
		render_aabb_to_local,
		camera_matrix1[2],
		focal_length,
		sample_index,
		m_rays[0].payload,
		grid,
		(show_accel >= 0) ? show_accel : 0,
		max_mip,
		cone_angle_constant
	);
}

uint32_t Testbed::NerfTracer::trace(
	const std::shared_ptr<NerfNetwork<network_precision_t>>& network,
	const BoundingBox& render_aabb,
	const mat3& render_aabb_to_local,
	const BoundingBox& train_aabb,
	const vec2& focal_length,
	float cone_angle_constant,
	const uint8_t* grid,
	ERenderMode render_mode,
	const mat4x3 &camera_matrix,
	float depth_scale,
	int visualized_layer,
	int visualized_dim,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	int show_accel,
	uint32_t max_mip,
	float min_transmittance,
	float glow_y_cutoff,
	int glow_mode,
	const float* extra_dims_gpu,
	cudaStream_t stream
) {
	if (m_n_rays_initialized == 0) {
		return 0;
	}

	CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter, 0, sizeof(uint32_t), stream));

	uint32_t n_alive = m_n_rays_initialized;
	// m_n_rays_initialized = 0;

	uint32_t i = 1;
	uint32_t double_buffer_index = 0;
	while (i < MARCH_ITER) {
		RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
		RaysNerfSoa& rays_tmp = m_rays[double_buffer_index % 2];
		++double_buffer_index;

		// Compact rays that did not diverge yet
		{
			CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
			linear_kernel(compact_kernel_nerf, 0, stream,
				n_alive,
				rays_tmp.rgba, rays_tmp.depth, rays_tmp.payload,
				rays_current.rgba, rays_current.depth, rays_current.payload,
				m_rays_hit.rgba, m_rays_hit.depth, m_rays_hit.payload,
				m_alive_counter, m_hit_counter
			);
			CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		}

		if (n_alive == 0) {
			break;
		}

		// Want a large number of queries to saturate the GPU and to ensure compaction doesn't happen toooo frequently.
		uint32_t target_n_queries = 2 * 1024 * 1024;
		uint32_t n_steps_between_compaction = clamp(target_n_queries / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);

		uint32_t extra_stride = network->n_extra_dims() * sizeof(float);
		PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input, 1, 0, extra_stride);
		linear_kernel(generate_next_nerf_network_inputs, 0, stream,
			n_alive,
			render_aabb,
			render_aabb_to_local,
			train_aabb,
			focal_length,
			camera_matrix[2],
			rays_current.payload,
			input_data,
			n_steps_between_compaction,
			grid,
			(show_accel>=0) ? show_accel : 0,
			max_mip,
			cone_angle_constant,
			extra_dims_gpu
		);
		uint32_t n_elements = next_multiple(n_alive * n_steps_between_compaction, BATCH_SIZE_GRANULARITY);
		GPUMatrix<float> positions_matrix((float*)m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
		GPUMatrix<network_precision_t, RM> rgbsigma_matrix((network_precision_t*)m_network_output, network->padded_output_width(), n_elements);
		network->inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

		if (render_mode == ERenderMode::Normals) {
			network->input_gradient(stream, 3, positions_matrix, positions_matrix);
		} else if (render_mode == ERenderMode::EncodingVis) {
			network->visualize_activation(stream, visualized_layer, visualized_dim, positions_matrix, positions_matrix);
		}

		linear_kernel(composite_kernel_nerf, 0, stream,
			n_alive,
			n_elements,
			i,
			train_aabb,
			glow_y_cutoff,
			glow_mode,
			camera_matrix,
			focal_length,
			depth_scale,
			rays_current.rgba,
			rays_current.depth,
			rays_current.payload,
			input_data,
			m_network_output,
			network->padded_output_width(),
			n_steps_between_compaction,
			render_mode,
			grid,
			rgb_activation,
			density_activation,
			show_accel,
			min_transmittance
		);

		i += n_steps_between_compaction;
	}

	uint32_t n_hit;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	return n_hit;
}

void Testbed::NerfTracer::enlarge(size_t n_elements, uint32_t padded_output_width, uint32_t n_extra_dims, cudaStream_t stream) {
	n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
	size_t num_floats = sizeof(NerfCoordinate) / sizeof(float) + n_extra_dims;
	auto scratch = allocate_workspace_and_distribute<
		vec4, float, NerfPayload, // m_rays[0]
		vec4, float, NerfPayload, // m_rays[1]
		vec4, float, NerfPayload, // m_rays_hit

		network_precision_t,
		float,
		uint32_t,
		uint32_t
	>(
		stream, &m_scratch_alloc,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * padded_output_width,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * num_floats,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements);
	m_rays[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements);
	m_rays_hit.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements);

	m_network_output = std::get<9>(scratch);
	m_network_input = std::get<10>(scratch);

	m_hit_counter = std::get<11>(scratch);
	m_alive_counter = std::get<12>(scratch);
}

std::vector<float> Testbed::Nerf::Training::get_extra_dims_cpu(int trainview) const {
	if (dataset.n_extra_dims() == 0) {
		return {};
	}

	if (trainview < 0 || trainview >= dataset.n_images) {
		throw std::runtime_error{"Invalid training view."};
	}

	const float* extra_dims_src = extra_dims_gpu.data() + trainview * dataset.n_extra_dims();

	std::vector<float> extra_dims_cpu(dataset.n_extra_dims());
	CUDA_CHECK_THROW(cudaMemcpy(extra_dims_cpu.data(), extra_dims_src, dataset.n_extra_dims() * sizeof(float), cudaMemcpyDeviceToHost));

	return extra_dims_cpu;
}

void Testbed::Nerf::Training::update_extra_dims() {
	uint32_t n_extra_dims = dataset.n_extra_dims();
	std::vector<float> extra_dims_cpu(extra_dims_gpu.size());
	for (uint32_t i = 0; i < extra_dims_opt.size(); ++i) {
		const std::vector<float>& value = extra_dims_opt[i].variable();
		for (uint32_t j = 0; j < n_extra_dims; ++j) {
			extra_dims_cpu[i * n_extra_dims + j] = value[j];
		}
	}

	CUDA_CHECK_THROW(cudaMemcpyAsync(extra_dims_gpu.data(), extra_dims_cpu.data(), extra_dims_opt.size() * n_extra_dims * sizeof(float), cudaMemcpyHostToDevice));
}

void Testbed::render_nerf(
	cudaStream_t stream,
	CudaDevice& device,
	const CudaRenderBufferView& render_buffer,
	const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network,
	const uint8_t* density_grid_bitfield,
	const vec2& focal_length,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec4& rolling_shutter,
	const vec2& screen_center,
	const Foveation& foveation,
	int visualized_dimension,
	const Nerf& nerf,
	const BoundingBox& aabb
) {
	float plane_z = m_slice_plane_z + m_scale;
	if (m_render_mode == ERenderMode::Slice) {
		plane_z = -plane_z;
	}

	ERenderMode render_mode = visualized_dimension > -1 ? ERenderMode::EncodingVis : m_render_mode;

	const float* extra_dims_gpu = nerf.get_rendering_extra_dims(stream);

	NerfTracer tracer;

	// Our motion vector code can't undo grid distortions -- so don't render grid distortion if DLSS is enabled.
	// (Unless we're in distortion visualization mode, in which case the distortion grid is fine to visualize.)
	auto grid_distortion =
		nerf.render_with_lens_distortion && (!m_dlss || m_render_mode == ERenderMode::Distortion) ?
		m_distortion.inference_view() :
		Buffer2DView<const vec2>{};

	Lens lens = nerf.render_with_lens_distortion ? nerf.render_lens : Lens{};

	auto resolution = render_buffer.resolution;

	tracer.init_rays_from_camera(
		render_buffer.spp,
		nerf_network->padded_output_width(),
		nerf_network->n_extra_dims(),
		render_buffer.resolution,
		focal_length,
		camera_matrix0,
		camera_matrix1,
		rolling_shutter,
		screen_center,
		m_parallax_shift,
		m_snap_to_pixel_centers,
		m_render_aabb,
		m_render_aabb_to_local,
		m_render_near_distance,
		plane_z,
		m_aperture_size,
		foveation,
		lens,
		m_envmap.inference_view(),
		grid_distortion,
		render_buffer.frame_buffer,
		render_buffer.depth_buffer,
		render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
		density_grid_bitfield,
		nerf.show_accel,
		nerf.max_cascade,
		nerf.cone_angle_constant,
		render_mode,
		stream
	);

	float depth_scale = 1.0f / nerf.training.dataset.scale;
	bool render_2d = m_render_mode == ERenderMode::Slice || m_render_mode == ERenderMode::Distortion;

	uint32_t n_hit;
	if (render_2d) {
		n_hit = tracer.n_rays_initialized();
	} else {
		n_hit = tracer.trace(
			nerf_network,
			m_render_aabb,
			m_render_aabb_to_local,
			aabb,
			focal_length,
			nerf.cone_angle_constant,
			density_grid_bitfield,
			render_mode,
			camera_matrix1,
			depth_scale,
			m_visualized_layer,
			visualized_dimension,
			nerf.rgb_activation,
			nerf.density_activation,
			nerf.show_accel,
			nerf.max_cascade,
			nerf.render_min_transmittance,
			nerf.glow_y_cutoff,
			nerf.glow_mode,
			extra_dims_gpu,
			stream
		);
	}
	RaysNerfSoa& rays_hit = render_2d ? tracer.rays_init() : tracer.rays_hit();

	if (render_2d) {
		// Store colors in the normal buffer
		uint32_t n_elements = next_multiple(n_hit, BATCH_SIZE_GRANULARITY);
		const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + nerf_network->n_extra_dims();
		const uint32_t extra_stride = nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

		GPUMatrix<float> positions_matrix{floats_per_coord, n_elements, stream};
		GPUMatrix<float> rgbsigma_matrix{4, n_elements, stream};

		linear_kernel(generate_nerf_network_inputs_at_current_position, 0, stream, n_hit, m_aabb, rays_hit.payload, PitchedPtr<NerfCoordinate>((NerfCoordinate*)positions_matrix.data(), 1, 0, extra_stride), extra_dims_gpu);

		if (visualized_dimension == -1) {
			nerf_network->inference(stream, positions_matrix, rgbsigma_matrix);
			linear_kernel(compute_nerf_rgba_kernel, 0, stream, n_hit, (vec4*)rgbsigma_matrix.data(), nerf.rgb_activation, nerf.density_activation, 0.01f, false);
		} else {
			nerf_network->visualize_activation(stream, m_visualized_layer, visualized_dimension, positions_matrix, rgbsigma_matrix);
		}

		linear_kernel(shade_kernel_nerf, 0, stream,
			n_hit,
			nerf.render_gbuffer_hard_edges,
			camera_matrix1,
			depth_scale,
			(vec4*)rgbsigma_matrix.data(),
			nullptr,
			rays_hit.payload,
			m_render_mode,
			nerf.training.linear_colors,
			render_buffer.frame_buffer,
			render_buffer.depth_buffer
		);
		return;
	}

	linear_kernel(shade_kernel_nerf, 0, stream,
		n_hit,
		nerf.render_gbuffer_hard_edges,
		camera_matrix1,
		depth_scale,
		rays_hit.rgba,
		rays_hit.depth,
		rays_hit.payload,
		m_render_mode,
		nerf.training.linear_colors,
		render_buffer.frame_buffer,
		render_buffer.depth_buffer
	);

	if (render_mode == ERenderMode::Cost) {
		std::vector<NerfPayload> payloads_final_cpu(n_hit);
		CUDA_CHECK_THROW(cudaMemcpyAsync(payloads_final_cpu.data(), rays_hit.payload, n_hit * sizeof(NerfPayload), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

		size_t total_n_steps = 0;
		for (uint32_t i = 0; i < n_hit; ++i) {
			total_n_steps += payloads_final_cpu[i].n_steps;
		}
		tlog::info() << "Total steps per hit= " << total_n_steps << "/" << n_hit << " = " << ((float)total_n_steps/(float)n_hit);
	}
}

const float* Testbed::Nerf::get_rendering_extra_dims(cudaStream_t stream) const {
	CHECK_THROW(rendering_extra_dims.size() == training.dataset.n_extra_dims());

	if (training.dataset.n_extra_dims() == 0) {
		return nullptr;
	}

	const float* extra_dims_src = rendering_extra_dims_from_training_view >= 0 ?
		training.extra_dims_gpu.data() + rendering_extra_dims_from_training_view * training.dataset.n_extra_dims() :
		rendering_extra_dims.data();

	if (!training.dataset.has_light_dirs) {
		return extra_dims_src;
	}

	// the dataset has light directions, so we must construct a temporary buffer and fill it as requested.
	// we use an extra 'slot' that was pre-allocated for us at the end of the extra_dims array.
	size_t size = training.dataset.n_extra_dims() * sizeof(float);
	float* dims_gpu = training.extra_dims_gpu.data() + training.dataset.n_images * training.dataset.n_extra_dims();
	CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, extra_dims_src, size, cudaMemcpyDeviceToDevice, stream));
	vec3 light_dir = warp_direction(normalize(light_dir));
	CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, &light_dir, min(size, sizeof(vec3)), cudaMemcpyHostToDevice, stream));
	return dims_gpu;
}


std::vector<vec3> load_stl(const fs::path& path) {
	std::vector<vec3> vertices;

	std::ifstream f{native_string(path), std::ios::in | std::ios::binary};
	if (!f) {
		throw std::runtime_error{fmt::format("Mesh file '{}' not found", path.str())};
	}

	uint32_t buf[21] = {};
	f.read((char*)buf, 4 * 21);
	if (f.gcount() < 4 * 21) {
		throw std::runtime_error{fmt::format("Mesh file '{}' too small for STL header", path.str())};
	}

	uint32_t nfaces = buf[20];
	if (memcmp(buf, "solid", 5) == 0 || buf[20] == 0) {
		throw std::runtime_error{fmt::format("ASCII STL file '{}' not supported", path.str())};
	}

	vertices.reserve(nfaces * 3);
	for (uint32_t i = 0; i < nfaces; ++i) {
		f.read((char*)buf, 50);
		if (f.gcount() < 50) {
			nfaces = i;
			break;
		}

		vertices.push_back(*(vec3*)(buf + 3));
		vertices.push_back(*(vec3*)(buf + 6));
		vertices.push_back(*(vec3*)(buf + 9));
	}

	return vertices;
}

GeometryBvhNode Testbed::load_mesh(const fs::path& data_path, vec3 center) {

	GeometryBvhNode mesh_node;
    mesh_node.type = NodeType::MESH;

	tlog::info() << "Loading mesh from '" << data_path << "'";
	auto start = std::chrono::steady_clock::now();

	std::vector<vec3> vertices;
	if (equals_case_insensitive(data_path.extension(), "obj")) {
		vertices = load_obj(data_path.str());
	} else if (equals_case_insensitive(data_path.extension(), "stl")) {
		vertices = load_stl(data_path.str());
	} else {
		throw std::runtime_error{"mesh data path must be a mesh in ascii .obj or binary .stl format."};
	}

	// The expected format is
	// [v1.x][v1.y][v1.z][v2.x]...
	size_t n_vertices = vertices.size();
	size_t n_triangles = n_vertices/3;

	// Compute the AABB of the mesh
	vec3 inf(std::numeric_limits<float>::infinity());
	BoundingBox aabb (inf, inf);

	for (size_t i = 0; i < n_vertices; ++i) {
	    aabb.enlarge(vertices[i]);
	}

	// Normalize the vertices.
	for (size_t i = 0; i < n_vertices; ++i) {
	    vertices[i] = (vertices[i] - aabb.center) / max(aabb.diag());
	}

	// Store the center and scale for later use.
	// I dont think we need center! and maybe scale is not needed as well
	// if not needed delete from the struct
	mesh_node.data.mesh.mesh_center = aabb.center;
	mesh_node.data.mesh.mesh_scale = max(aabb.diag());

	// Normalize vertex coordinates to lie within [0,1]^3.
	// This way, none of the constants need to carry around
	// bounding box factors.
	for (size_t i = 0; i < n_vertices; ++i) {
		vertices[i] = (vertices[i] - aabb.min - 0.5f * aabb.diag()) / mesh_node.data.mesh.mesh_scale  + 0.5f;
	}

	mesh_node.data.mesh.triangles_cpu.resize(n_triangles);
	for (size_t i = 0; i < n_vertices; i += 3) {
		mesh_node.data.mesh.triangles_cpu[i/3] = {vertices[i+0], vertices[i+1], vertices[i+2]};
	}

	if (!mesh_node.data.mesh.triangle_bvh) {
		mesh_node.data.mesh.triangle_bvh = TriangleBvh::make();
	}

	mesh_node.data.mesh.triangle_bvh->build(mesh_node.data.mesh.triangles_cpu, 8);
	mesh_node.data.mesh.triangles_gpu.resize_and_copy_from_host(mesh_node.data.mesh.triangles_cpu);

	// initializes optix and creates OptiX program raytrace
	mesh_node.data.mesh.triangle_bvh->build_optix(mesh_node.data.mesh.triangles_gpu, m_stream.get());

	mesh_node.data.mesh.triangle_octree.reset(new TriangleOctree{});
	mesh_node.data.mesh.triangle_octree->build(*mesh_node.data.mesh.triangle_bvh, mesh_node.data.mesh.triangles_cpu, 10);

	m_bounding_radius = length(vec3(0.5f));

	// Compute discrete probability distribution for later sampling of the mesh's surface
	mesh_node.data.mesh.triangle_weights.resize(n_triangles);
	for (size_t i = 0; i < n_triangles; ++i) {
		triangle_weights.triangle_weights[i] = triangle_weights.triangles_cpu[i].surface_area();
	}
	triangle_weights.triangle_distribution.build(triangle_weights.triangle_weights);

	// Move CDF to gpu
	triangle_weights.triangle_cdf.resize_and_copy_from_host(triangle_weights.triangle_distribution.cdf);

	// Clear training data as it's no longer representative
	// of the previously loaded mesh.. but don't clear the network.
	// Perhaps it'll look interesting while morphing from one mesh to another.
	triangle_weights.training.idx = 0;
	triangle_weights.training.size = 0;

	tlog::success() << "Loaded mesh after " << tlog::durationToString(std::chrono::steady_clock::now() - start);
	tlog::info() << "  n_triangles=" << n_triangles << " aabb=" << m_raw_aabb;


	node.bb = aabb; // not sure about the correct bounding box
    node.left_idx = -1; // Set to -1 for leaf node
    node.right_idx = -1; // Set to -1 for leaf node

	return node;
}

// should I make a unique pointer and then return it?
GeometryBvhNode Testbed::load_empty_mesh_node(vec3 center) {

    GeometryBvhNode mesh_node;
    mesh_node.type = NodeType::Mesh;
	mesh_node.data.mesh = MeshData{};

	// init the meshData

    mesh_node.bb = BoundingBox{center, center+vec3(0.5f)};

    mesh_node.left_idx = -1; 
    mesh_node.right_idx = -1; 

    return mesh_node;
}


GeometryBvhNode Testbed::load_nerf(const fs::path& data_path, vec3 center) {
	
	
	GeometryBvhNode nerf_node;
    nerf_node.type = NodeType::Nerf;


	if (!data_path.empty()) {
		std::vector<fs::path> json_paths;
		if (data_path.is_directory()) {
			for (const auto& path : fs::directory{data_path}) {
				if (path.is_file() && equals_case_insensitive(path.extension(), "json")) {
					json_paths.emplace_back(path);
				}
			}
		} else if (equals_case_insensitive(data_path.extension(), "json")) {
			json_paths.emplace_back(data_path);
		} else {
			throw std::runtime_error{"NeRF data path must either be a json file or a directory containing json files."};
		}

		const auto prev_aabb_scale = nerf.training.dataset.aabb_scale;

		nerf_node.data.nerf.training.dataset = ngp::load_nerf(json_paths, nerf_node.data.nerf.sharpen);

		nerf_node.data.nerf.rgb_activation = nerf_node.data.nerf.training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

	nerf_node.data.nerf.training.n_images_for_training = (int)nerf_node.data.nerf.training.dataset.n_images;

	nerf_node.data.nerf.training.dataset.update_metadata();

	nerf_node.data.nerf.training.cam_pos_gradient.resize(nerf_node.data.nerf.training.dataset.n_images, vec3(0.0f));
	nerf_node.data.nerf.training.cam_pos_gradient_gpu.resize_and_copy_from_host(nerf_node.data.nerf.training.cam_pos_gradient);

	nerf_node.data.nerf.training.cam_exposure.resize(nerf_node.data.nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-3f));
	nerf_node.data.nerf.training.cam_pos_offset.resize(nerf_node.data.nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-4f));
	nerf_node.data.nerf.training.cam_rot_offset.resize(nerf_node.data.nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
	nerf_node.data.nerf.training.cam_focal_length_offset = AdamOptimizer<vec2>(1e-5f);

	nerf_node.data.nerf.training.cam_rot_gradient.resize(nerf_node.data.nerf.training.dataset.n_images, vec3(0.0f));
	nerf_node.data.nerf.training.cam_rot_gradient_gpu.resize_and_copy_from_host(nerf_node.data.nerf.training.cam_rot_gradient);

	nerf_node.data.nerf.training.cam_exposure_gradient.resize(nerf_node.data.nerf.training.dataset.n_images, vec3(0.0f));
	nerf_node.data.nerf.training.cam_exposure_gpu.resize_and_copy_from_host(nerf_node.data.nerf.training.cam_exposure_gradient);
	nerf_node.data.nerf.training.cam_exposure_gradient_gpu.resize_and_copy_from_host(nerf_node.data.nerf.training.cam_exposure_gradient);

	nerf_node.data.nerf.training.cam_focal_length_gradient = vec2(0.0f);
	nerf_node.data.nerf.training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&nerf_node.data.nerf.training.cam_focal_length_gradient, 1);

	nerf_node.data.nerf.reset_extra_dims(m_rng);
	nerf_node.data.nerf.training.optimize_extra_dims = nerf_node.data.nerf.training.dataset.n_extra_learnable_dims > 0;

	if (nerf_node.data.nerf.training.dataset.has_rays) {
		nerf_node.data.nerf.training.near_distance = 0.0f;
	}

	nerf_node.data.nerf.training.update_transforms();

	if (!nerf_node.data.nerf.training.dataset.metadata.empty()) {
		nerf_node.data.nerf.render_lens = nerf_node.data.nerf.training.dataset.metadata[0].lens;
		m_screen_center = vec2(1.f) - nerf_node.data.nerf.training.dataset.metadata[0].principal_point;
	}

	if (!is_pot(nerf_node.data.nerf.training.dataset.aabb_scale)) {
		throw std::runtime_error{fmt::format("NeRF dataset's `aabb_scale` must be a power of two, but is {}.", nerf_node.data.nerf.training.dataset.aabb_scale)};
	}

	int max_aabb_scale = 1 << (NERF_CASCADES()-1);
	if (nerf_node.data.nerf.training.dataset.aabb_scale > max_aabb_scale) {
		throw std::runtime_error{fmt::format(
			"NeRF dataset must have `aabb_scale <= {}`, but is {}. "
			"You can increase this limit by factors of 2 by incrementing `NERF_CASCADES()` and re-compiling.",
			max_aabb_scale, nerf_node.data.nerf.training.dataset.aabb_scale
		)};
	}

	// Todo: come up with better plan!!

	BoundingBox aabb = BoundingBox{center, center};
	aabb.inflate(0.5f * std::min(1 << (NERF_CASCADES()-1), nerf_node.data.nerf.training.dataset.aabb_scale));

	m_render_aabb_to_local = nerf_node.data.nerf.training.dataset.render_aabb_to_local;	//not sure about this
	
	if (!nerf_node.data.nerf.training.dataset.render_aabb.is_empty()) {
		render_aabb = nerf_node.data.nerf.training.dataset.render_aabb.intersection(aabb);
	}

	nerf_node.data.nerf.max_cascade = 0;
	while ((1 << nerf_node.data.nerf.max_cascade) < nerf_node.data.nerf.training.dataset.aabb_scale) {
		++nerf_node.data.nerf.max_cascade;
	}

	// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
	// stepping in larger scenes.
	nerf_node.data.nerf.cone_angle_constant = nerf_node.data.nerf.training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);

	m_up_dir = nerf_node.data.nerf.training.dataset.up;
	}

	nerf_node.bb = aabb;
    nerf_node.left_idx = -1; 
    nerf_node.right_idx = -1; 

    return nerf_node;
}

GeometryBvhNode Testbed::load_empty_nerf_node(vec3 center) {
	m_data_path = {};
	GeometryBvhNode nerf_node;
    nerf_node.type = NodeType::Nerf;
    nerf_node.data.nerf = Nerf{}; // I'm not sure if this is necessary

	nerf_node.data.nerf.training.dataset = ngp::create_empty_nerf_dataset(0, 1, false);
	load_nerf(m_data_path);
	nerf_node.data.nerf.training.n_images_for_training = 0;
    
	nerf_node.bb = BoundingBox{center, center+vec3(0.5f)};

    nerf_node.left_idx = -1; 
    nerf_node.right_idx = -1; 

    return nerf_node;
}

// future work: add a threadpool to make the loading simultaneous
void Testbed::load_scene(const fs::path& data_path) {


	/**
	 * [
    {
        "center": [0.0, 0.0, 0.0],
        "path": "path/to/geometry.obj",
        "type": "Mesh"
    },
    {
        "center": [1.0, 1.0, 1.0],
        "path": "path/to/geometry.json",
        "type": "Nerf"
    }
    // ... more geometries ...
	 *]
	 * 
	*/
	if (!data_path.empty()) {
		if (m_geometry.geometry_bvh) {
			m_geometry.geometry_bvh.reset();
		}
		if (m_geometry.bvh_nodes.size() > 0) {
			m_geometry.bvh_nodes.clear();
		}
		if (!m_geometry.geometry_bvh) {
			m_geometry.geometry_bvh = GeometryBvh::make();
		}

		std::ifstream file(data_path);
		if (!file) {
			throw std::runtime_error{fmt::format("Geometry file '{}' not found", path.str())};
		}
        
		nlohmann::json geometries = nlohmann::json::parse(file, nullptr, true, true);

        if (!geometries.is_array()) {
            throw std::runtime_error{"Geometry file must contain an array of geometry metadata."};
        }

        size_t n_nodes = geometries.size();

		m_geometry.bvh_nodes.resize(n_nodes);

		for(size_t i = 0; i < n_nodes; ++i) {
			auto& geometry = geometries[i];
            fs::path model_path = geometry["path"];

			std::string type = geometry["type"];
			std::vector<float> center = geometry["center"];
            vec3 center_vec(center[0], center[1], center[2]);

            if (type == "Mesh") {
                m_geometry.bvh_nodes[i] = load_mesh(model_path, center_vec);
            } else if (type == "Nerf") {
                m_geometry.bvh_nodes[i] = load_nerf(model_path, center_vec);
            } else {
                throw std::runtime_error{"Geometry type must be either 'Mesh' or 'Nerf'."};
            }

			// if (equals_case_insensitive(model_path.extension(), "obj") || equals_case_insensitive(model_path.extension(), "stl")) {
			// 	m_geometry.bvh_nodes[i] = load_mesh(model_path);
			// } else if (equals_case_insensitive(model_path.extension(), "json")) {
			// 	m_geometry.bvh_nodes[i] = load_nerf(model_path);
			// } else {
			// 	throw std::runtime_error{"mesh data path must be a mesh in ascii .obj or binary .stl format or nerf in json format."};
			// }
		}
	}
	
	else {

		m_geometry.bvh_nodes.resize(2);
    	
		m_geometry.bvh_nodes[0] = load_empty_mesh_node(vec3(0.0f));
    	m_geometry.bvh_nodes[1]= load_empty_nerf_node(vec3(1.0f));
		
	}
		
	
	m_geometry.geometry_bvh->build(m_geometry.geometry_bvh, 8);

	

}

}