/** @file   testbed_geometry.cu
 *  @author Fatemeh Salehi
 */

#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/discrete_distribution.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/random_val.cuh>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/takikawa_encoding.cuh>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/geometry.h>
#include <neural-graphics-primitives/tinyobj_loader_wrapper.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/geometry_bvh.cuh>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/trainer.h>

namespace ngp {


// TODO: all m_aabb s should be changed to local/node bounding boxes and passed/stored in the node 

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;

Testbed::NetworkDims Testbed::network_dims_geometry() const {
	NetworkDims dims;
	dims.n_input = 3;
	dims.n_output = 1;
	dims.n_pos = 3;
	return dims;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ inline float square(float x) { return x * x; }
__device__ inline float mix(float a, float b, float t) { return a + (b - a) * t; }
__device__ inline vec3 mix(const vec3& a, const vec3& b, float t) { return a + (b - a) * t; }

__device__ inline float SchlickFresnel(float u) {
	float m = __saturatef(1.0 - u);
	return square(square(m)) * m;
}

__device__ inline float G1(float NdotH, float a) {
	if (a >= 1.0) { return 1.0 / PI(); }
	float a2 = square(a);
	float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
	return (a2 - 1.0) / (PI() * log(a2) * t);
}

__device__ inline float G2(float NdotH, float a) {
	float a2 = square(a);
	float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
	return a2 / (PI() * t * t);
}

__device__ inline float SmithG_GGX(float NdotV, float alphaG) {
	float a = alphaG * alphaG;
	float b = NdotV * NdotV;
	return 1.0 / (NdotV + sqrtf(a + b - a * b));
}


__device__ vec3 evaluate_shading_geometry(
	const vec3& base_color,
	const vec3& ambient_color, // :)
	const vec3& light_color, // :)
	float metallic,
	float subsurface,
	float specular,
	float roughness,
	float specular_tint,
	float sheen,
	float sheen_tint,
	float clearcoat,
	float clearcoat_gloss,
	vec3 L,
	vec3 V,
	vec3 N
) {
	float NdotL = dot(N, L);
	float NdotV = dot(N, V);

	vec3 H = normalize(L + V);
	float NdotH = dot(N, H);
	float LdotH = dot(L, H);

	// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
	// and mix in diffuse retro-reflection based on roughness
	float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV);
	vec3 amb = (ambient_color * mix(0.2f, FV, metallic));
	amb *= base_color;
	if (NdotL < 0.f || NdotV < 0.f) {
		return amb;
	}

	float luminance = dot(base_color, vec3{0.3f, 0.6f, 0.1f});

	// normalize luminance to isolate hue and saturation components
	vec3 Ctint = base_color * (1.f/(luminance+0.00001f));
	vec3 Cspec0 = mix(mix(vec3(1.0f), Ctint, specular_tint) * specular * 0.08f, base_color, metallic);
	vec3 Csheen = mix(vec3(1.0f), Ctint, sheen_tint);

	float Fd90 = 0.5f + 2.0f * LdotH * LdotH * roughness;
	float Fd = mix(1, Fd90, FL) * mix(1.f, Fd90, FV);

	// Based on Hanrahan-Krueger BRDF approximation of isotropic BSSRDF
	// 1.25 scale is used to (roughly) preserve albedo
	// Fss90 used to "flatten" retroreflection based on roughness
	float Fss90 = LdotH * LdotH * roughness;
	float Fss = mix(1.0f, Fss90, FL) * mix(1.0f, Fss90, FV);
	float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - 0.5f) + 0.5f);

	// Specular
	float a= std::max(0.001f, square(roughness));
	float Ds = G2(NdotH, a);
	float FH = SchlickFresnel(LdotH);
	vec3 Fs = mix(Cspec0, vec3(1.0f), FH);
	float Gs = SmithG_GGX(NdotL, a) * SmithG_GGX(NdotV, a);

	// sheen
	vec3 Fsheen = FH * sheen * Csheen;

	// clearcoat (ior = 1.5 -> F0 = 0.04)
	float Dr = G1(NdotH, mix(0.1f, 0.001f, clearcoat_gloss));
	float Fr = mix(0.04f, 1.0f, FH);
	float Gr = SmithG_GGX(NdotL, 0.25f) * SmithG_GGX(NdotV, 0.25f);

	float CCs=0.25f * clearcoat * Gr * Fr * Dr;
	vec3 brdf = (float(1.0f / PI()) * mix(Fd, ss, subsurface) * base_color + Fsheen) * (1.0f - metallic) +
		Gs * Fs * Ds + vec3{CCs, CCs, CCs};
	return vec3(brdf * light_color) * NdotL + amb;
}

__global__ void advance_pos_kernel_mesh_geometry(
	const uint32_t n_elements,
	const float zero_offset,
	vec3* __restrict__ positions,
	float* __restrict__ distances,
	GeometryPayload* __restrict__ payloads,
	BoundingBox aabb,
	float floor_y,
	float distance_scale,
	float maximum_distance,
	float k,
	float* __restrict__ prev_distances,
	float* __restrict__ total_distances,
	float* __restrict__ min_visibility
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	GeometryPayload& payload = payloads[i];
	if (!payload.alive) {
		return;
	}

	float distance = distances[i] - zero_offset;

	distance *= distance_scale;

	// Advance by the predicted distance
	vec3 pos = positions[i];
	pos += distance * payload.dir;

	if (pos.y < floor_y && payload.dir.y<0.f) {
		float floor_dist = -(pos.y-floor_y)/payload.dir.y;
		distance += floor_dist;
		pos += floor_dist * payload.dir;
		payload.alive=false;
	}

	positions[i] = pos;

	if (total_distances && distance > 0.0f) {
		// From https://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
		float total_distance = total_distances[i];
		float y = distance*distance / (2.0f * prev_distances[i]);
		float d = sqrtf(distance*distance - y*y);

		min_visibility[i] = fminf(min_visibility[i], k * d / fmaxf(0.0f, total_distance - y));
		prev_distances[i] = distance;
		total_distances[i] = total_distance + distance;
	}

	bool stay_alive = distance > maximum_distance && fabsf(distance / 2) > 3*maximum_distance;
	if (!stay_alive) {
		payload.alive = false;
		return;
	}

	if (!aabb.contains(pos)) {
		payload.alive = false;
		return;
	}

	payload.n_steps++;
}

__global__ void perturb_mesh_samples(uint32_t n_elements, const vec3* __restrict__ perturbations, vec3* __restrict__ positions, float* __restrict__ distances) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	vec3 perturbation = perturbations[i];
	positions[i] += perturbation;

	// Small epsilon above 1 to ensure a triangle is always found.
	distances[i] = length(perturbation) * 1.001f;
}

__global__ void prepare_shadow_rays_geometry(const uint32_t n_elements,
	vec3 sun_dir,
	vec3* __restrict__ positions,
	vec3* __restrict__ normals,
	float* __restrict__ distances,
	float* __restrict__ prev_distances,
	float* __restrict__ total_distances,
	float* __restrict__ min_visibility,
	GeometryPayload* __restrict__ payloads,
	BoundingBox aabb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	GeometryPayload& payload = payloads[i];

	// Step back a little along the ray to prevent self-intersection
	vec3 view_pos = positions[i] + normalize(faceforward(normals[i], payload.dir, normals[i])) * 1e-3f;
	vec3 dir = normalize(sun_dir);

	float t = fmaxf(aabb.ray_intersect(view_pos, dir).x + 1e-6f, 0.0f);
	view_pos += t * dir;

	positions[i] = view_pos;

	if (!aabb.contains(view_pos)) {
		distances[i] = MAX_DEPTH();
		payload.alive = false;
		min_visibility[i] = 1.0f;
		return;
	}

	distances[i] = MAX_DEPTH();
	payload.idx = i;
	payload.dir = dir;
	payload.n_steps = 0;
	payload.alive = true;

	if (prev_distances) {
		prev_distances[i] = 1e20f;
	}

	if (total_distances) {
		total_distances[i] = 0.0f;
	}

	if (min_visibility) {
		min_visibility[i] = 1.0f;
	}
}

__global__ void write_shadow_ray_result_geometry(const uint32_t n_elements, BoundingBox aabb, const vec3* __restrict__ positions, const GeometryPayload* __restrict__ shadow_payloads, const float* __restrict__ min_visibility, float* __restrict__ shadow_factors) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	shadow_factors[shadow_payloads[i].idx] = aabb.contains(positions[i]) ? 0.0f : min_visibility[i];
}

__global__ void shade_kernel_mesh_geometry(
	const uint32_t n_elements,
	BoundingBox aabb,
	float floor_y,
	const ERenderMode mode,
	const BRDFParams brdf,
	vec3 sun_dir,
	vec3 up_dir,
	mat4x3 camera_matrix,
	vec3* __restrict__ positions,
	vec3* __restrict__ normals,
	float* __restrict__ distances,
	GeometryPayload* __restrict__ payloads,
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
	switch (mode) {
		case ERenderMode::AO: color = vec3(powf(0.92f, payload.n_steps)); break;
		case ERenderMode::Shade: {
			float skyam = -dot(normal, up_dir) * 0.5f + 0.5f;
			vec3 suncol = vec3{255.f/255.0f, 225.f/255.0f, 195.f/255.0f} * 4.f * distances[i]; // Distance encodes shadow occlusion. 0=occluded, 1=no shadow
			const vec3 skycol = vec3{195.f/255.0f, 215.f/255.0f, 255.f/255.0f} * 4.f * skyam;
			float check_size = 8.f/aabb.diag().x;
			float check=((int(floorf(check_size*(pos.x-aabb.min.x)))^int(floorf(check_size*(pos.z-aabb.min.z)))) &1) ? 0.8f : 0.2f;
			const vec3 floorcol = vec3{check*check*check, check*check, check};
			color = evaluate_shading_geometry(
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
		} break;
		case ERenderMode::Depth: color = vec3(dot(cam_fwd, pos - cam_pos)); break;
		case ERenderMode::Positions: {
			color = (pos - 0.5f) / 2.0f + 0.5f;
		} break;
		case ERenderMode::Normals: color = 0.5f * normal + 0.5f; break;
		case ERenderMode::Cost: color = vec3((float)payload.n_steps / 30); break;
		case ERenderMode::EncodingVis: color = normals[i]; break;
	}

	frame_buffer[payload.idx] = {color.r, color.g, color.b, 1.0f};
	depth_buffer[payload.idx] = dot(cam_fwd, pos - cam_pos);
}

__global__ void compact_kernel_shadow_mesh_geometry(
	const uint32_t n_elements,
	const float zero_offset,
	vec3* src_positions, float* src_distances, GeometryPayload* src_payloads, float* src_prev_distances, float* src_total_distances, float* src_min_visibility,
	vec3* dst_positions, float* dst_distances, GeometryPayload* dst_payloads, float* dst_prev_distances, float* dst_total_distances, float* dst_min_visibility,
	vec3* dst_final_positions, float* dst_final_distances, GeometryPayload* dst_final_payloads, float* dst_final_prev_distances, float* dst_final_total_distances, float* dst_final_min_visibility,
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
		dst_prev_distances[idx] = src_prev_distances[i];
		dst_total_distances[idx] = src_total_distances[i];
		dst_min_visibility[idx] = src_min_visibility[i];
	} else { // For shadow rays, collect _all_ final samples to keep track of their partial visibility
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_positions[idx] = src_positions[i];
		dst_final_distances[idx] = src_distances[i];
		dst_final_prev_distances[idx] = src_prev_distances[i];
		dst_final_total_distances[idx] = src_total_distances[i];
		dst_final_min_visibility[idx] = aabb.contains(src_positions[i]) ? 0.0f : src_min_visibility[i];
	}
}

// separates the "alive" and "dead" elements of the input arrays into two separate arrays
__global__ void compact_kernel_mesh_geometry(
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

__global__ void scale_to_aabb_kernel_geometry(uint32_t n_elements, BoundingBox aabb, vec3* __restrict__ inout) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	inout[i] = aabb.min + inout[i] * aabb.diag();
}
__global__ void scale_iou_counters_kernel_geometry(uint32_t n_elements, uint32_t *counters, float scale) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	counters[i] = uint32_t(roundf(counters[i]*scale));
}

__global__ void assign_float_geometry(uint32_t n_elements, float value, float* __restrict__ out) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	out[i] = value;
}

__global__ void init_rays_with_payload_kernel_mesh_geometry(
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
	Buffer2DView<const uint8_t> hidden_area_mask
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

	positions[idx] = ray.o;

	if (envmap) {
		frame_buffer[idx] = read_envmap(envmap, ray.d);
	}

	payload.dir = ray.d;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = aabb.contains(ray.o);
}

__host__ __device__ uint32_t sample_discrete_geometry(float uniform_sample, const float* __restrict__ cdf, int length) {
	return binary_search(uniform_sample, cdf, length);
}

__global__ void sample_uniform_on_triangle_kernel_geometry(uint32_t n_elements, const float* __restrict__ cdf, uint32_t length, const Triangle* __restrict__ triangles, vec3* __restrict__ sampled_positions) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	vec3 sample = sampled_positions[i];
	uint32_t tri_idx = sample_discrete_geometry(sample.x, cdf, length);

	sampled_positions[i] = triangles[tri_idx].sample_uniform_position(sample.yz());
}

///////////////////////////////////////////////////////////////////////////////////////////////\


__global__ void extract_srgb_with_activation_geometry(const uint32_t n_elements,	const uint32_t rgb_stride, const float* __restrict__ rgbd, float* __restrict__ rgb, ENerfActivation rgb_activation, bool from_linear) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	float c = network_to_rgb(rgbd[elem_idx*4 + dim_idx], rgb_activation);
	if (from_linear) {
		c = linear_to_srgb(c);
	}

	rgb[elem_idx*rgb_stride + dim_idx] = c;
}

__global__ void mark_untrained_density_grid_geometry(const uint32_t n_elements,  float* __restrict__ grid_out,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	bool clear_visible_voxels
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint32_t level = i / NERF_GRID_N_CELLS();
	uint32_t pos_idx = i % NERF_GRID_N_CELLS();

	uint32_t x = morton3D_invert(pos_idx>>0);
	uint32_t y = morton3D_invert(pos_idx>>1);
	uint32_t z = morton3D_invert(pos_idx>>2);

	float voxel_size = scalbnf(1.0f / NERF_GRIDSIZE(), level);
	vec3 pos = (vec3{(float)x, (float)y, (float)z} / (float)NERF_GRIDSIZE() - 0.5f) * scalbnf(1.0f, level) + 0.5f;

	vec3 corners[8] = {
		pos + vec3{0.0f,       0.0f,       0.0f      },
		pos + vec3{voxel_size, 0.0f,       0.0f      },
		pos + vec3{0.0f,       voxel_size, 0.0f      },
		pos + vec3{voxel_size, voxel_size, 0.0f      },
		pos + vec3{0.0f,       0.0f,       voxel_size},
		pos + vec3{voxel_size, 0.0f,       voxel_size},
		pos + vec3{0.0f,       voxel_size, voxel_size},
		pos + vec3{voxel_size, voxel_size, voxel_size},
	};

	// Number of training views that need to see a voxel cell
	// at minimum for that cell to be marked trainable.
	// Floaters can be reduced by increasing this value to 2,
	// but at the cost of certain reconstruction artifacts.
	const uint32_t min_count = 1;
	uint32_t count = 0;

	for (uint32_t j = 0; j < n_training_images && count < min_count; ++j) {
		const auto& xform = training_xforms[j].start;
		const auto& m = metadata[j];

		if (m.lens.mode == ELensMode::FTheta || m.lens.mode == ELensMode::LatLong || m.lens.mode == ELensMode::Equirectangular) {
			// FTheta lenses don't have a forward mapping, so are assumed seeing everything. Latlong and equirect lenses
			// by definition see everything.
			++count;
			continue;
		}

		for (uint32_t k = 0; k < 8; ++k) {
			// Only consider voxel corners in front of the camera
			vec3 dir = normalize(corners[k] - xform[3]);
			if (dot(dir, xform[2]) < 1e-4f) {
				continue;
			}

			// Check if voxel corner projects onto the image plane, i.e. uv must be in (0, 1)^2
			vec2 uv = pos_to_uv(corners[k], m.resolution, m.focal_length, xform, m.principal_point, vec3(0.0f), {}, m.lens);

			// `pos_to_uv` is _not_ injective in the presence of lens distortion (which breaks down outside of the image plane).
			// So we need to check whether the produced uv location generates a ray that matches the ray that we started with.
			Ray ray = uv_to_ray(0.0f, uv, m.resolution, m.focal_length, xform, m.principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, m.lens);
			if (distance(normalize(ray.d), dir) < 1e-3f && uv.x > 0.0f && uv.y > 0.0f && uv.x < 1.0f && uv.y < 1.0f) {
				++count;
				break;
			}
		}
	}

	if (clear_visible_voxels || (grid_out[i] < 0) != (count < min_count)) {
		grid_out[i] = (count >= min_count) ? 0.f : -1.f;
	}
}

__global__ void generate_grid_samples_nerf_uniform_geometry(ivec3 res_3d, const uint32_t step, BoundingBox render_aabb, mat3 render_aabb_to_local, BoundingBox train_aabb, NerfPosition* __restrict__ out) {
	// check grid_in for negative values -> must be negative on output
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x >= res_3d.x || y >= res_3d.y || z >= res_3d.z) {
		return;
	}

	uint32_t i = x + y * res_3d.x + z * res_3d.x * res_3d.y;
	vec3 pos = vec3{(float)x, (float)y, (float)z} / vec3(res_3d - 1);
	pos = transpose(render_aabb_to_local) * (pos * (render_aabb.max - render_aabb.min) + render_aabb.min);
	out[i] = { warp_position(pos, train_aabb), warp_dt(MIN_CONE_STEPSIZE()) };
}

// generate samples for uniform grid including constant ray direction
__global__ void generate_grid_samples_nerf_uniform_dir_geometry(ivec3 res_3d, const uint32_t step, BoundingBox render_aabb, mat3 render_aabb_to_local, BoundingBox train_aabb, vec3 ray_dir, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims, bool voxel_centers) {
	// check grid_in for negative values -> must be negative on output
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x >= res_3d.x || y >= res_3d.y || z >= res_3d.z) {
		return;
	}

	uint32_t i = x+ y*res_3d.x + z*res_3d.x*res_3d.y;
	vec3 pos;
	if (voxel_centers) {
		pos = vec3{(float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f} / vec3(res_3d);
	} else {
		pos = vec3{(float)x, (float)y, (float)z} / vec3(res_3d - 1);
	}

	pos = transpose(render_aabb_to_local) * (pos * (render_aabb.max - render_aabb.min) + render_aabb.min);

	network_input(i)->set_with_optional_extra_dims(warp_position(pos, train_aabb), warp_direction(ray_dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
}

__global__ void generate_grid_samples_nerf_nonuniform_geometry(const uint32_t n_elements, default_rng_t rng, const uint32_t step, BoundingBox aabb, const float* __restrict__ grid_in, NerfPosition* __restrict__ out, uint32_t* __restrict__ indices, uint32_t n_cascades, float thresh) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// 1 random number to select the level, 3 to select the position.
	rng.advance(i*4);
	uint32_t level = (uint32_t)(random_val(rng) * n_cascades) % n_cascades;

	// Select grid cell that has density
	uint32_t idx;
	for (uint32_t j = 0; j < 10; ++j) {
		idx = ((i+step*n_elements) * 56924617 + j * 19349663 + 96925573) % NERF_GRID_N_CELLS();
		idx += level * NERF_GRID_N_CELLS();
		if (grid_in[idx] > thresh) {
			break;
		}
	}

	// Random position within that cellq
	uint32_t pos_idx = idx % NERF_GRID_N_CELLS();

	uint32_t x = morton3D_invert(pos_idx>>0);
	uint32_t y = morton3D_invert(pos_idx>>1);
	uint32_t z = morton3D_invert(pos_idx>>2);

	vec3 pos = ((vec3{(float)x, (float)y, (float)z} + random_val_3d(rng)) / (float)NERF_GRIDSIZE() - 0.5f) * scalbnf(1.0f, level) + 0.5f;

	out[i] = { warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE()) };
	indices[i] = idx;
}

__global__ void splat_grid_samples_nerf_max_nearest_neighbor_geometry(const uint32_t n_elements, const uint32_t* __restrict__ indices, const network_precision_t* network_output, float* __restrict__ grid_out, ENerfActivation rgb_activation, ENerfActivation density_activation) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint32_t local_idx = indices[i];

	// Current setting: optical thickness of the smallest possible stepsize.
	// Uncomment for:   optical thickness of the ~expected step size when the observer is in the middle of the scene
	uint32_t level = 0;//local_idx / NERF_GRID_N_CELLS();

	float mlp = network_to_density(float(network_output[i]), density_activation);
	float optical_thickness = mlp * scalbnf(MIN_CONE_STEPSIZE(), level);

	// Positive floats are monotonically ordered when their bit pattern is interpretes as uint.
	// uint atomicMax is thus perfectly acceptable.
	atomicMax((uint32_t*)&grid_out[local_idx], __float_as_uint(optical_thickness));
}

__global__ void grid_samples_half_to_float_geometry(const uint32_t n_elements, BoundingBox aabb, float* dst, const network_precision_t* network_output, ENerfActivation density_activation, const NerfPosition* __restrict__ coords_in, const float* __restrict__ grid_in, uint32_t max_cascade) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// let's interpolate for marching cubes based on the raw MLP output, not the density (exponentiated) version
	//float mlp = network_to_density(float(network_output[i * padded_output_width]), density_activation);
	float mlp = float(network_output[i]);

	if (grid_in) {
		vec3 pos = unwarp_position(coords_in[i].p, aabb);
		float grid_density = cascaded_grid_at(pos, grid_in, mip_from_pos(pos, max_cascade));
		if (grid_density < NERF_MIN_OPTICAL_THICKNESS()) {
			mlp = -10000.0f;
		}
	}

	dst[i] = mlp;
}

__global__ void ema_grid_samples_nerf_geometry(const uint32_t n_elements,
	float decay,
	const uint32_t count,
	float* __restrict__ grid_out,
	const float* __restrict__ grid_in
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float importance = grid_in[i];

	// float ema_debias_old = 1 - (float)powf(decay, count);
	// float ema_debias_new = 1 - (float)powf(decay, count+1);

	// float filtered_val = ((grid_out[i] * decay * ema_debias_old + importance * (1 - decay)) / ema_debias_new);
	// grid_out[i] = filtered_val;

	// Maximum instead of EMA allows capture of very thin features.
	// Basically, we want the grid cell turned on as soon as _ANYTHING_ visible is in there.

	float prev_val = grid_out[i];
	float val = (prev_val<0.f) ? prev_val : fmaxf(prev_val * decay, importance);
	grid_out[i] = val;
}

__global__ void decay_sharpness_grid_nerf_geometry(const uint32_t n_elements, float decay, float* __restrict__ grid) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	grid[i] *= decay;
}

__global__ void grid_to_bitfield_geometry(
	const uint32_t n_elements,
	const uint32_t n_nonzero_elements,
	const float* __restrict__ grid,
	uint8_t* __restrict__ grid_bitfield,
	const float* __restrict__ mean_density_ptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	if (i >= n_nonzero_elements) {
		grid_bitfield[i] = 0;
		return;
	}

	uint8_t bits = 0;

	float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

	NGP_PRAGMA_UNROLL
	for (uint8_t j = 0; j < 8; ++j) {
		bits |= grid[i*8+j] > thresh ? ((uint8_t)1 << j) : 0;
	}

	grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool_geometry(const uint32_t n_elements,
	const uint8_t* __restrict__ prev_level,
	uint8_t* __restrict__ next_level
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint8_t bits = 0;

	NGP_PRAGMA_UNROLL
	for (uint8_t j = 0; j < 8; ++j) {
		// If any bit is set in the previous level, set this
		// level's bit. (Max pooling.)
		bits |= prev_level[i*8+j] > 0 ? ((uint8_t)1 << j) : 0;
	}

	uint32_t x = morton3D_invert(i>>0) + NERF_GRIDSIZE()/8;
	uint32_t y = morton3D_invert(i>>1) + NERF_GRIDSIZE()/8;
	uint32_t z = morton3D_invert(i>>2) + NERF_GRIDSIZE()/8;

	next_level[morton3D(x, y, z)] |= bits;
}

__device__ void advance_pos_nerf_geometry(
	NerfPayload& payload,
	const BoundingBox& render_aabb,
	const mat3& render_aabb_to_local,
	const vec3& camera_fwd,
	const vec2& focal_length,
	uint32_t sample_index,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	float cone_angle_constant
) {
	if (!payload.alive) {
		return;
	}

	vec3 origin = payload.origin;
	vec3 dir = payload.dir;
	vec3 idir = vec3(1.0f) / dir;

	float cone_angle = calc_cone_angle(dot(dir, camera_fwd), focal_length, cone_angle_constant);

	float t = advance_n_steps(payload.t, cone_angle, ld_random_val(sample_index, payload.idx * 786433));
	t = if_unoccupied_advance_to_next_occupied_voxel(t, cone_angle, {origin, dir}, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
	if (t >= MAX_DEPTH()) {
		payload.alive = false;
	} else {
		payload.t = t;
	}
}

__global__ void advance_pos_nerf_kernel_geometry(
	const uint32_t n_elements,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	vec3 camera_fwd,
	vec2 focal_length,
	uint32_t sample_index,
	NerfPayload* __restrict__ payloads,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	float cone_angle_constant
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	advance_pos_nerf_geometry(payloads[i], render_aabb, render_aabb_to_local, camera_fwd, focal_length, sample_index, density_grid, min_mip, max_mip, cone_angle_constant);
}

__global__ void generate_nerf_network_inputs_from_positions_geometry(const uint32_t n_elements, BoundingBox aabb, const vec3* __restrict__ pos, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	vec3 dir = normalize(pos[i] - 0.5f); // choose outward pointing directions, for want of a better choice
	network_input(i)->set_with_optional_extra_dims(warp_position(pos[i], aabb), warp_direction(dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
}

__global__ void generate_nerf_network_inputs_at_current_position_geometry(const uint32_t n_elements, BoundingBox aabb, const NerfPayload* __restrict__ payloads, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	vec3 dir = payloads[i].dir;
	network_input(i)->set_with_optional_extra_dims(warp_position(payloads[i].origin + dir * payloads[i].t, aabb), warp_direction(dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
}

__device__ vec4 compute_nerf_rgba_geometry(const vec4& network_output, ENerfActivation rgb_activation, ENerfActivation density_activation, float depth, bool density_as_alpha = false) {
	vec4 rgba = network_output;

	float density = network_to_density(rgba.a, density_activation);
	float alpha = 1.f;
	if (density_as_alpha) {
		rgba.a = density;
	} else {
		rgba.a = alpha = clamp(1.f - __expf(-density * depth), 0.0f, 1.0f);
	}

	rgba.rgb() = network_to_rgb_vec(rgba.rgb(), rgb_activation) * alpha;
	return rgba;
}

__global__ void compute_nerf_rgba_kernel_geometry(const uint32_t n_elements, vec4* network_output, ENerfActivation rgb_activation, ENerfActivation density_activation, float depth, bool density_as_alpha = false) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	network_output[i] = compute_nerf_rgba_geometry(network_output[i], rgb_activation, density_activation, depth, density_as_alpha);
}

__global__ void generate_next_nerf_network_inputs_geometry(
	const uint32_t n_elements,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	BoundingBox train_aabb,
	vec2 focal_length,
	vec3 camera_fwd,
	NerfPayload* __restrict__ payloads,
	PitchedPtr<NerfCoordinate> network_input,
	uint32_t n_steps,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	float cone_angle_constant,
	const float* extra_dims
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	vec3 origin = payload.origin;
	vec3 dir = payload.dir;
	vec3 idir = vec3(1.0f) / dir;

	float cone_angle = calc_cone_angle(dot(dir, camera_fwd), focal_length, cone_angle_constant);

	float t = payload.t;

	for (uint32_t j = 0; j < n_steps; ++j) {
		// ray marching
		t = if_unoccupied_advance_to_next_occupied_voxel(t, cone_angle, {origin, dir}, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
		if (t >= MAX_DEPTH()) {
			payload.n_steps = j;
			return;
		}

		float dt = calc_dt(t, cone_angle);
		network_input(i + j * n_elements)->set_with_optional_extra_dims(warp_position(origin + dir * t, train_aabb), warp_direction(dir), warp_dt(dt), extra_dims, network_input.stride_in_bytes); // XXXCONE
		t += dt;
	}

	payload.t = t;
	payload.n_steps = n_steps;
}

__global__ void composite_kernel_nerf_geometry(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t current_step,
	BoundingBox aabb,
	float glow_y_cutoff,
	int glow_mode,
	mat4x3 camera_matrix,
	vec2 focal_length,
	float depth_scale,
	vec4* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* payloads,
	PitchedPtr<NerfCoordinate> network_input,
	const network_precision_t* __restrict__ network_output,
	uint32_t padded_output_width,
	uint32_t n_steps,
	ERenderMode render_mode,
	const uint8_t* __restrict__ density_grid,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	int show_accel,
	float min_transmittance
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	vec4 local_rgba = rgba[i];
	float local_depth = depth[i];
	vec3 origin = payload.origin;
	vec3 cam_fwd = camera_matrix[2];
	// Composite in the last n steps
	uint32_t actual_n_steps = payload.n_steps;
	uint32_t j = 0;

	for (; j < actual_n_steps; ++j) {
		tvec<network_precision_t, 4> local_network_output;
		local_network_output[0] = network_output[i + j * n_elements + 0 * stride];
		local_network_output[1] = network_output[i + j * n_elements + 1 * stride];
		local_network_output[2] = network_output[i + j * n_elements + 2 * stride];
		local_network_output[3] = network_output[i + j * n_elements + 3 * stride];
		const NerfCoordinate* input = network_input(i + j * n_elements);
		vec3 warped_pos = input->pos.p;
		vec3 pos = unwarp_position(warped_pos, aabb);

		float T = 1.f - local_rgba.a;
		float dt = unwarp_dt(input->dt);
		float alpha = 1.f - __expf(-network_to_density(float(local_network_output[3]), density_activation) * dt);
		if (show_accel >= 0) {
			alpha = 1.f;
		}
		float weight = alpha * T;

		vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);

		if (glow_mode) { // random grid visualizations ftw!
#if 0
			if (0) {  // extremely startrek edition
				float glow_y = (pos.y - (glow_y_cutoff - 0.5f)) * 2.f;
				if (glow_y>1.f) glow_y=max(0.f,21.f-glow_y*20.f);
				if (glow_y>0.f) {
					float line;
					line =max(0.f,cosf(pos.y*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.x*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.z*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.y*4.f*3.141592653589793f * 16.f)-0.975f);
					line+=max(0.f,cosf(pos.x*4.f*3.141592653589793f * 16.f)-0.975f);
					line+=max(0.f,cosf(pos.z*4.f*3.141592653589793f * 16.f)-0.975f);
					glow_y=glow_y*glow_y*0.5f + glow_y*line*25.f;
					rgb.y+=glow_y;
					rgb.z+=glow_y*0.5f;
					rgb.x+=glow_y*0.25f;
				}
			}
#endif
			float glow = 0.f;

			bool green_grid = glow_mode & 1;
			bool green_cutline = glow_mode & 2;
			bool mask_to_alpha = glow_mode & 4;

			// less used?
			bool radial_mode = glow_mode & 8;
			bool grid_mode = glow_mode & 16; // makes object rgb go black!

			{
				float dist;
				if (radial_mode) {
					dist = distance(pos, camera_matrix[3]);
					dist = min(dist, (4.5f - pos.y) * 0.333f);
				} else {
					dist = pos.y;
				}

				if (grid_mode) {
					glow = 1.f / max(1.f, dist);
				} else {
					float y = glow_y_cutoff - dist; // - (ii*0.005f);
					float mask = 0.f;
					if (y > 0.f) {
						y *= 80.f;
						mask = min(1.f, y);
						//if (mask_mode) {
						//	rgb.x=rgb.y=rgb.z=mask; // mask mode
						//} else
						{
							if (green_cutline) {
								glow += max(0.f, 1.f - abs(1.f -y)) * 4.f;
							}

							if (y>1.f) {
								y = 1.f - (y - 1.f) * 0.05f;
							}

							if (green_grid) {
								glow += max(0.f, y / max(1.f, dist));
							}
						}
					}
					if (mask_to_alpha) {
						weight *= mask;
					}
				}
			}

			if (glow > 0.f) {
				float line;
				line  = max(0.f, cosf(pos.y * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				if (grid_mode) {
					glow = /*glow*glow*0.75f + */ glow * line * 15.f;
					rgb.y = glow;
					rgb.z = glow * 0.5f;
					rgb.x = glow * 0.25f;
				} else {
					glow = glow * glow * 0.25f + glow * line * 15.f;
					rgb.y += glow;
					rgb.z += glow * 0.5f;
					rgb.x += glow * 0.25f;
				}
			}
		} // glow

		if (render_mode == ERenderMode::Normals) {
			// Network input contains the gradient of the network output w.r.t. input.
			// So to compute density gradients, we need to apply the chain rule.
			// The normal is then in the opposite direction of the density gradient (i.e. the direction of decreasing density)
			vec3 normal = -network_to_density_derivative(float(local_network_output[3]), density_activation) * warped_pos;
			rgb = normalize(normal);
		} else if (render_mode == ERenderMode::Positions) {
			rgb = (pos - 0.5f) / 2.0f + 0.5f;
		} else if (render_mode == ERenderMode::EncodingVis) {
			rgb = warped_pos;
		} else if (render_mode == ERenderMode::Depth) {
			rgb = vec3(dot(cam_fwd, pos - origin) * depth_scale);
		} else if (render_mode == ERenderMode::AO) {
			rgb = vec3(alpha);
		}

		if (show_accel >= 0) {
			uint32_t mip = max((uint32_t)show_accel, mip_from_pos(pos));
			uint32_t res = NERF_GRIDSIZE() >> mip;
			int ix = pos.x * res;
			int iy = pos.y * res;
			int iz = pos.z * res;
			default_rng_t rng(ix + iy * 232323 + iz * 727272);
			rgb.x = 1.f - mip * (1.f / (NERF_CASCADES() - 1));
			rgb.y = rng.next_float();
			rgb.z = rng.next_float();
		}

		local_rgba += vec4(rgb * weight, weight);
		if (weight > payload.max_weight) {
			payload.max_weight = weight;
			local_depth = dot(cam_fwd, pos - camera_matrix[3]);
		}

		if (local_rgba.a > (1.0f - min_transmittance)) {
			local_rgba /= local_rgba.a;
			break;
		}
	}

	if (j < n_steps) {
		payload.alive = false;
		payload.n_steps = j + current_step;
	}

	rgba[i] = local_rgba;
	depth[i] = local_depth;
}

__global__ void generate_training_samples_nerf_geometry(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t max_samples,
	const uint32_t n_rays_total,
	default_rng_t rng,
	uint32_t* __restrict__ ray_counter,
	uint32_t* __restrict__ numsteps_counter,
	uint32_t* __restrict__ ray_indices_out,
	Ray* __restrict__ rays_out_unnormalized,
	uint32_t* __restrict__ numsteps_out,
	PitchedPtr<NerfCoordinate> coords_out,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	const uint8_t* __restrict__ density_grid,
	uint32_t max_mip,
	bool max_level_rand_training,
	float* __restrict__ max_level_ptr,
	bool snap_to_pixel_centers,
	bool train_envmap,
	float cone_angle_constant,
	Buffer2DView<const vec2> distortion,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 cdf_res,
	const float* __restrict__ extra_dims_gpu,
	uint32_t n_extra_dims
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	uint32_t img = image_idx(i, n_rays, n_rays_total, n_training_images, cdf_img);
	ivec2 resolution = metadata[img].resolution;

	rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
	vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, cdf_res, img);

	// Negative values indicate masked-away regions
	size_t pix_idx = pixel_idx(uv, resolution, 0);
	if (read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type).x < 0.0f) {
		return;
	}

	float max_level = max_level_rand_training ? (random_val(rng) * 2.0f) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

	float motionblur_time = random_val(rng);

	const vec2 focal_length = metadata[img].focal_length;
	const vec2 principal_point = metadata[img].principal_point;
	const float* extra_dims = extra_dims_gpu + img * n_extra_dims;
	const Lens lens = metadata[img].lens;

	const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, motionblur_time);

	Ray ray_unnormalized;
	const Ray* rays_in_unnormalized = metadata[img].rays;
	if (rays_in_unnormalized) {
		// Rays have been explicitly supplied. Read them.
		ray_unnormalized = rays_in_unnormalized[pix_idx];

		/* DEBUG - compare the stored rays to the computed ones
		const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, 0.f);
		Ray ray2;
		ray2.o = xform[3];
		ray2.d = f_theta_distortion(uv, principal_point, lens);
		ray2.d = (xform.block<3, 3>(0, 0) * ray2.d).normalized();
		if (i==1000) {
			printf("\n%d uv %0.3f,%0.3f pixel %0.2f,%0.2f transform from [%0.5f %0.5f %0.5f] to [%0.5f %0.5f %0.5f]\n"
				" origin    [%0.5f %0.5f %0.5f] vs [%0.5f %0.5f %0.5f]\n"
				" direction [%0.5f %0.5f %0.5f] vs [%0.5f %0.5f %0.5f]\n"
			, img,uv.x, uv.y, uv.x*resolution.x, uv.y*resolution.y,
				training_xforms[img].start[3].x,training_xforms[img].start[3].y,training_xforms[img].start[3].z,
				training_xforms[img].end[3].x,training_xforms[img].end[3].y,training_xforms[img].end[3].z,
				ray_unnormalized.o.x,ray_unnormalized.o.y,ray_unnormalized.o.z,
				ray2.o.x,ray2.o.y,ray2.o.z,
				ray_unnormalized.d.x,ray_unnormalized.d.y,ray_unnormalized.d.z,
				ray2.d.x,ray2.d.y,ray2.d.z);
		}
		*/
	} else {
		ray_unnormalized = uv_to_ray(0, uv, resolution, focal_length, xform, principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, lens, distortion);
		if (!ray_unnormalized.is_valid()) {
			ray_unnormalized = {xform[3], xform[2]};
		}
	}

	vec3 ray_d_normalized = normalize(ray_unnormalized.d);

	vec2 tminmax = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
	float cone_angle = calc_cone_angle(dot(ray_d_normalized, xform[2]), focal_length, cone_angle_constant);

	// The near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x = fmaxf(tminmax.x, 0.0f);
	
	// calculate the starting point of the ray within the bounding box
	float startt = advance_n_steps(tminmax.x, cone_angle, random_val(rng));
	vec3 idir = vec3(1.0f) / ray_d_normalized;

	// first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t = startt;
	vec3 pos;

	// checking if the current position is within the bounding box and if the density grid is occupied at the current position.
	// counting the number of steps
	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < NERF_STEPS()) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos, max_mip);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			++j;
			t += dt;
		} else {
			t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
		}
	}
	if (j == 0 && !train_envmap) {
		return;
	}
	uint32_t numsteps = j;
	uint32_t base = atomicAdd(numsteps_counter, numsteps);	 // first entry in the array is a counter
	if (base + numsteps > max_samples) {
		return;
	}

	coords_out += base;

	uint32_t ray_idx = atomicAdd(ray_counter, 1);

	ray_indices_out[ray_idx] = i;
	rays_out_unnormalized[ray_idx] = ray_unnormalized;
	numsteps_out[ray_idx*2+0] = numsteps;
	numsteps_out[ray_idx*2+1] = base;

	vec3 warped_dir = warp_direction(ray_d_normalized);
	t=startt;
	j=0;

	// enters second loop: it steps through the ray again
	// storing the position, direction, and time step of each step in the output array.
	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos, max_mip);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
			++j;
			t += dt;
		} else {
			t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
		}
	}

	if (max_level_rand_training) {
		max_level_ptr += base;
		for (j = 0; j < numsteps; ++j) {
			max_level_ptr[j] = max_level;
		}
	}
}


__global__ void compute_loss_kernel_train_nerf_geometry(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const uint32_t max_samples_compacted,
	const uint32_t* __restrict__ rays_counter,
	float loss_scale,
	int padded_output_width,
	Buffer2DView<const vec4> envmap,
	float* __restrict__ envmap_gradient,
	const ivec2 envmap_resolution,
	ELossType envmap_loss_type,
	vec3 background_color,
	EColorSpace color_space,
	bool train_with_random_bg_color,
	bool train_in_linear_colors,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const network_precision_t* network_output,
	uint32_t* __restrict__ numsteps_counter,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<const NerfCoordinate> coords_in,
	PitchedPtr<NerfCoordinate> coords_out,
	network_precision_t* dloss_doutput,
	ELossType loss_type,
	ELossType depth_loss_type,
	float* __restrict__ loss_output,
	bool max_level_rand_training,
	float* __restrict__ max_level_compacted_ptr,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	bool snap_to_pixel_centers,
	float* __restrict__ error_map,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 error_map_res,
	const ivec2 error_map_cdf_res,
	const float* __restrict__ sharpness_data,
	ivec2 sharpness_resolution,
	float* __restrict__ sharpness_grid,
	float* __restrict__ density_grid,
	const float* __restrict__ mean_density_ptr,
	uint32_t max_mip,
	const vec3* __restrict__ exposure,
	vec3* __restrict__ exposure_gradient,
	float depth_supervision_lambda,
	float near_distance
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];

	coords_in += base;
	network_output += base * padded_output_width;

	float T = 1.f;

	float EPSILON = 1e-4f;

	vec3 rgb_ray = vec3(0.0f);
	vec3 hitpoint = vec3(0.0f);

	float depth_ray = 0.f;
	uint32_t compacted_numsteps = 0;
	vec3 ray_o = rays_in_unnormalized[i].o;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
		if (T < EPSILON) {
			break;
		}

		const tvec<network_precision_t, 4> local_network_output = *(tvec<network_precision_t, 4>*)network_output;
		const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
		const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		const float dt = unwarp_dt(coords_in.ptr->dt);
		float cur_depth = distance(pos, ray_o);
		float density = network_to_density(float(local_network_output[3]), density_activation);


		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray += weight * rgb;
		hitpoint += weight * pos;
		depth_ray += weight * cur_depth;
		T *= (1.f - alpha);

		network_output += padded_output_width;
		coords_in += 1;
	}
	hitpoint /= (1.0f - T);

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

	float img_pdf = 1.0f;
	uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img, &img_pdf);
	ivec2 resolution = metadata[img].resolution;

	float uv_pdf = 1.0f;
	vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_cdf_res, img, &uv_pdf);
	float max_level = max_level_rand_training ? (random_val(rng) * 2.0f) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level
	rng.advance(1); // motionblur_time

	if (train_with_random_bg_color) {
		background_color = random_val_3d(rng);
	}
	vec3 pre_envmap_background_color = background_color = srgb_to_linear(background_color);

	// Composit background behind envmap
	vec4 envmap_value;
	vec3 dir;
	if (envmap) {
		dir = normalize(rays_in_unnormalized[i].d);
		envmap_value = read_envmap(envmap, dir);
		background_color = envmap_value.rgb() + background_color * (1.0f - envmap_value.a);
	}

	vec3 exposure_scale = exp(0.6931471805599453f * exposure[img]);
	// vec3 rgbtarget = composit_and_lerp(uv, resolution, img, training_images, background_color, exposure_scale);
	// vec3 rgbtarget = composit(uv, resolution, img, training_images, background_color, exposure_scale);
	vec4 texsamp = read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type);

	vec3 rgbtarget;
	if (train_in_linear_colors || color_space == EColorSpace::Linear) {
		rgbtarget = exposure_scale * texsamp.rgb() + (1.0f - texsamp.a) * background_color;

		if (!train_in_linear_colors) {
			rgbtarget = linear_to_srgb(rgbtarget);
			background_color = linear_to_srgb(background_color);
		}
	} else if (color_space == EColorSpace::SRGB) {
		background_color = linear_to_srgb(background_color);
		if (texsamp.a > 0) {
			rgbtarget = linear_to_srgb(exposure_scale * texsamp.rgb() / texsamp.a) * texsamp.a + (1.0f - texsamp.a) * background_color;
		} else {
			rgbtarget = background_color;
		}
	}

	if (compacted_numsteps == numsteps) {
		// support arbitrary background colors
		rgb_ray += T * background_color;
	}

	// Step again, this time computing loss
	network_output -= padded_output_width * compacted_numsteps; // rewind the pointer
	coords_in -= compacted_numsteps;

	uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
	compacted_numsteps = min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
	numsteps_in[i*2+0] = compacted_numsteps;
	numsteps_in[i*2+1] = compacted_base;
	if (compacted_numsteps == 0) {
		return;
	}

	max_level_compacted_ptr += compacted_base;
	coords_out += compacted_base;

	dloss_doutput += compacted_base * padded_output_width;

	LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray, loss_type);
	lg.loss /= img_pdf * uv_pdf;

	float target_depth = length(rays_in_unnormalized[i].d) * ((depth_supervision_lambda > 0.0f && metadata[img].depth) ? read_depth(uv, resolution, metadata[img].depth) : -1.0f);
	LossAndGradient lg_depth = loss_and_gradient(vec3(target_depth), vec3(depth_ray), depth_loss_type);
	float depth_loss_gradient = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.gradient.x : 0;

	// Note: dividing the gradient by the PDF would cause unbiased loss estimates.
	// Essentially: variance reduction, but otherwise the same optimization.
	// We _dont_ want that. If importance sampling is enabled, we _do_ actually want
	// to change the weighting of the loss function. So don't divide.
	// lg.gradient /= img_pdf * uv_pdf;

	float mean_loss = mean(lg.loss);
	if (loss_output) {
		loss_output[i] = mean_loss / (float)n_rays;
	}

	if (error_map) {
		const vec2 pos = clamp(uv * vec2(error_map_res) - 0.5f, 0.0f, vec2(error_map_res) - (1.0f + 1e-4f));
		const ivec2 pos_int = pos;
		const vec2 weight = pos - vec2(pos_int);

		ivec2 idx = clamp(pos_int, 0, resolution - 2);

		auto deposit_val = [&](int x, int y, float val) {
			atomicAdd(&error_map[img * product(error_map_res) + y * error_map_res.x + x], val);
		};

		if (sharpness_data && aabb.contains(hitpoint)) {
			ivec2 sharpness_pos = clamp(ivec2(uv * vec2(sharpness_resolution)), 0, sharpness_resolution - 1);
			float sharp = sharpness_data[img * product(sharpness_resolution) + sharpness_pos.y * sharpness_resolution.x + sharpness_pos.x] + 1e-6f;

			// The maximum value of positive floats interpreted in uint format is the same as the maximum value of the floats.
			float grid_sharp = __uint_as_float(atomicMax((uint32_t*)&cascaded_grid_at(hitpoint, sharpness_grid, mip_from_pos(hitpoint, max_mip)), __float_as_uint(sharp)));
			grid_sharp = fmaxf(sharp, grid_sharp); // atomicMax returns the old value, so compute the new one locally.

			mean_loss *= fmaxf(sharp / grid_sharp, 0.01f);
		}

		deposit_val(idx.x,   idx.y,   (1 - weight.x) * (1 - weight.y) * mean_loss);
		deposit_val(idx.x+1, idx.y,        weight.x  * (1 - weight.y) * mean_loss);
		deposit_val(idx.x,   idx.y+1, (1 - weight.x) *      weight.y  * mean_loss);
		deposit_val(idx.x+1, idx.y+1,      weight.x  *      weight.y  * mean_loss);
	}

	loss_scale /= n_rays;

	const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
	const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

	// now do it again computing gradients
	vec3 rgb_ray2 = { 0.f,0.f,0.f };
	float depth_ray2 = 0.f;
	T = 1.f;
	for (uint32_t j = 0; j < compacted_numsteps; ++j) {
		if (max_level_rand_training) {
			max_level_compacted_ptr[j] = max_level;
		}
		// Compact network inputs
		NerfCoordinate* coord_out = coords_out(j);
		const NerfCoordinate* coord_in = coords_in(j);
		coord_out->copy(*coord_in, coords_out.stride_in_bytes);

		const vec3 pos = unwarp_position(coord_in->pos.p, aabb);
		float depth = distance(pos, ray_o);

		float dt = unwarp_dt(coord_in->dt);
		const tvec<network_precision_t, 4> local_network_output = *(tvec<network_precision_t, 4>*)network_output;
		const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
		const float density = network_to_density(float(local_network_output[3]), density_activation);
		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray2 += weight * rgb;
		depth_ray2 += weight * depth;
		T *= (1.f - alpha);

		// we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
		const vec3 suffix = rgb_ray - rgb_ray2;
		const vec3 dloss_by_drgb = weight * lg.gradient;

		tvec<network_precision_t, 4> local_dL_doutput;

		// chain rule to go from dloss/drgb to dloss/dmlp_output
		local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
		local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
		local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

		float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
		const float depth_suffix = depth_ray - depth_ray2;
		const float depth_supervision = depth_loss_gradient * (T * depth - depth_suffix);

		float dloss_by_dmlp = density_derivative * (
			dt * (dot(lg.gradient, T * rgb - suffix) + depth_supervision)
		);

		//static constexpr float mask_supervision_strength = 1.f; // we are already 'leaking' mask information into the nerf via the random bg colors; setting this to eg between 1 and  100 encourages density towards 0 in such regions.
		//dloss_by_dmlp += (texsamp.a<0.001f) ? mask_supervision_strength * weight : 0.f;

		local_dL_doutput[3] =
			loss_scale * dloss_by_dmlp +
			(float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
			(float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);
			;

		*(tvec<network_precision_t, 4>*)dloss_doutput = local_dL_doutput;

		dloss_doutput += padded_output_width;
		network_output += padded_output_width;
	}

	if (exposure_gradient) {
		// Assume symmetric loss
		vec3 dloss_by_dgt = -lg.gradient / uv_pdf;

		if (!train_in_linear_colors) {
			dloss_by_dgt /= srgb_to_linear_derivative(rgbtarget);
		}

		// 2^exposure * log(2)
		vec3 dloss_by_dexposure = loss_scale * dloss_by_dgt * exposure_scale * 0.6931471805599453f;
		atomicAdd(&exposure_gradient[img].x, dloss_by_dexposure.x);
		atomicAdd(&exposure_gradient[img].y, dloss_by_dexposure.y);
		atomicAdd(&exposure_gradient[img].z, dloss_by_dexposure.z);
	}

	if (compacted_numsteps == numsteps && envmap_gradient) {
		vec3 loss_gradient = lg.gradient;
		if (envmap_loss_type != loss_type) {
			loss_gradient = loss_and_gradient(rgbtarget, rgb_ray, envmap_loss_type).gradient;
		}

		vec3 dloss_by_dbackground = T * loss_gradient;
		if (!train_in_linear_colors) {
			dloss_by_dbackground /= srgb_to_linear_derivative(background_color);
		}

		tvec<network_precision_t, 4> dL_denvmap;
		dL_denvmap[0] = loss_scale * dloss_by_dbackground.x;
		dL_denvmap[1] = loss_scale * dloss_by_dbackground.y;
		dL_denvmap[2] = loss_scale * dloss_by_dbackground.z;


		float dloss_by_denvmap_alpha = -dot(dloss_by_dbackground, pre_envmap_background_color);

		// dL_denvmap[3] = loss_scale * dloss_by_denvmap_alpha;
		dL_denvmap[3] = (network_precision_t)0;

		deposit_envmap_gradient(dL_denvmap, envmap_gradient, envmap_resolution, dir);
	}
}


__global__ void compute_cam_gradient_train_nerf_geometry(
	const uint32_t n_rays,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const BoundingBox aabb,
	const uint32_t* __restrict__ rays_counter,
	const TrainingXForm* training_xforms,
	bool snap_to_pixel_centers,
	vec3* cam_pos_gradient,
	vec3* cam_rot_gradient,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<NerfCoordinate> coords,
	PitchedPtr<NerfCoordinate> coords_gradient,
	float* __restrict__ distortion_gradient,
	float* __restrict__ distortion_gradient_weight,
	const ivec2 distortion_resolution,
	vec2* cam_focal_length_gradient,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 error_map_res
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	if (numsteps == 0) {
		// The ray doesn't matter. So no gradient onto the camera
		return;
	}

	uint32_t base = numsteps_in[i*2+1];
	coords += base;
	coords_gradient += base;

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);
	ivec2 resolution = metadata[img].resolution;

	const mat4x3& xform = training_xforms[img].start;

	Ray ray = rays_in_unnormalized[i];
	ray.d = normalize(ray.d);
	Ray ray_gradient = { vec3(0.0f), vec3(0.0f) };

	// Compute ray gradient
	for (uint32_t j = 0; j < numsteps; ++j) {
		const vec3 warped_pos = coords(j)->pos.p;
		const vec3 pos_gradient = coords_gradient(j)->pos.p * warp_position_derivative(warped_pos, aabb);
		ray_gradient.o += pos_gradient;
		const vec3 pos = unwarp_position(warped_pos, aabb);

		// Scaled by t to account for the fact that further-away objects' position
		// changes more rapidly as the direction changes.
		float t = distance(pos, ray.o);
		const vec3 dir_gradient = coords_gradient(j)->dir.d * warp_direction_derivative(coords(j)->dir.d);
		ray_gradient.d += pos_gradient * t + dir_gradient;
	}

	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());
	float uv_pdf = 1.0f;

	vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_res, img, &uv_pdf);

	if (distortion_gradient) {
		// Projection of the raydir gradient onto the plane normal to raydir,
		// because that's the only degree of motion that the raydir has.
		vec3 orthogonal_ray_gradient = ray_gradient.d - ray.d * dot(ray_gradient.d, ray.d);

		// Rotate ray gradient to obtain image plane gradient.
		// This has the effect of projecting the (already projected) ray gradient from the
		// tangent plane of the sphere onto the image plane (which is correct!).
		vec3 image_plane_gradient = inverse(mat3(xform)) * orthogonal_ray_gradient;

		// Splat the resulting 2D image plane gradient into the distortion params
		deposit_image_gradient(image_plane_gradient.xy() / uv_pdf, distortion_gradient, distortion_gradient_weight, distortion_resolution, uv);
	}

	if (cam_pos_gradient) {
		// Atomically reduce the ray gradient into the xform gradient
		NGP_PRAGMA_UNROLL
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_pos_gradient[img][j], ray_gradient.o[j] / uv_pdf);
		}
	}

	if (cam_rot_gradient) {
		// Rotation is averaged in log-space (i.e. by averaging angle-axes).
		// Due to our construction of ray_gradient.d, ray_gradient.d and ray.d are
		// orthogonal, leading to the angle_axis magnitude to equal the magnitude
		// of ray_gradient.d.
		vec3 angle_axis = cross(ray.d, ray_gradient.d);

		// Atomically reduce the ray gradient into the xform gradient
		NGP_PRAGMA_UNROLL
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_rot_gradient[img][j], angle_axis[j] / uv_pdf);
		}
	}
}

__global__ void compute_extra_dims_gradient_train_nerf_geometry(
	const uint32_t n_rays,
	const uint32_t n_rays_total,
	const uint32_t* __restrict__ rays_counter,
	float* extra_dims_gradient,
	uint32_t n_extra_dims,
	const uint32_t n_training_images,
	const uint32_t* __restrict__ ray_indices_in,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<NerfCoordinate> coords_gradient,
	const float* __restrict__ cdf_img
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	if (numsteps == 0) {
		// The ray doesn't matter. So no gradient onto the camera
		return;
	}
	uint32_t base = numsteps_in[i*2+1];
	coords_gradient += base;
	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);

	extra_dims_gradient += n_extra_dims * img;

	for (uint32_t j = 0; j < numsteps; ++j) {
		const float *src = coords_gradient(j)->get_extra_dims();
		for (uint32_t k = 0; k < n_extra_dims; ++k) {
			atomicAdd(&extra_dims_gradient[k], src[k]);
		}
	}
}

__global__ void shade_kernel_nerf_geometry(
	const uint32_t n_elements,
	bool gbuffer_hard_edges,
	mat4x3 camera_matrix,
	float depth_scale,
	vec4* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* __restrict__ payloads,
	ERenderMode render_mode,
	bool train_in_linear_colors,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements || render_mode == ERenderMode::Distortion) return;
	NerfPayload& payload = payloads[i];

	vec4 tmp = rgba[i];
	if (render_mode == ERenderMode::Normals) {
		vec3 n = normalize(tmp.xyz());
		tmp.rgb() = (0.5f * n + 0.5f) * tmp.a;
	} else if (render_mode == ERenderMode::Cost) {
		float col = (float)payload.n_steps / 128;
		tmp = {col, col, col, 1.0f};
	} else if (gbuffer_hard_edges && render_mode == ERenderMode::Depth) {
		tmp.rgb() = vec3(depth[i] * depth_scale);
	} else if (gbuffer_hard_edges && render_mode == ERenderMode::Positions) {
		vec3 pos = camera_matrix[3] + payload.dir / dot(payload.dir, camera_matrix[2]) * depth[i];
		tmp.rgb() = (pos - 0.5f) / 2.0f + 0.5f;
	}

	if (!train_in_linear_colors && (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Slice)) {
		// Accumulate in linear colors
		tmp.rgb() = srgb_to_linear(tmp.rgb());
	}

	frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.a);
	if (render_mode != ERenderMode::Slice && tmp.a > 0.2f) {
		depth_buffer[payload.idx] = depth[i];
	}
}

__global__ void compact_kernel_nerf_geometry(
	const uint32_t n_elements,
	vec4* src_rgba, float* src_depth, NerfPayload* src_payloads,
	vec4* dst_rgba, float* dst_depth, NerfPayload* dst_payloads,
	vec4* dst_final_rgba, float* dst_final_depth, NerfPayload* dst_final_payloads,
	uint32_t* counter, uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_rgba[idx] = src_rgba[i];
		dst_depth[idx] = src_depth[i];
	} else if (src_rgba[i].a > 0.001f) {
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_rgba[idx] = src_rgba[i];
		dst_final_depth[idx] = src_depth[i];
	}
}

__global__ void init_rays_with_payload_kernel_nerf_geometry(
	uint32_t sample_index,
	NerfPayload* __restrict__ payloads,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Lens lens,
	Buffer2DView<const vec4> envmap,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Buffer2DView<const vec2> distortion,
	ERenderMode render_mode
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

	vec2 pixel_offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
	vec2 uv = vec2{(float)x + pixel_offset.x, (float)y + pixel_offset.y} / vec2(resolution);
	mat4x3 camera = get_xform_given_rolling_shutter({camera_matrix0, camera_matrix1}, rolling_shutter, uv, ld_random_val(sample_index, idx * 72239731));

	Ray ray = uv_to_ray(
		sample_index,
		uv,
		resolution,
		focal_length,
		camera,
		screen_center,
		parallax_shift,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask,
		lens,
		distortion
	);

	NerfPayload& payload = payloads[idx];
	payload.max_weight = 0.0f;

	depth_buffer[idx] = MAX_DEPTH();

	if (!ray.is_valid()) {
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}

	if (plane_z < 0) {
		float n = length(ray.d);
		payload.origin = ray.o;
		payload.dir = (1.0f/n) * ray.d;
		payload.t = -plane_z*n;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		depth_buffer[idx] = -plane_z;
		return;
	}

	if (render_mode == ERenderMode::Distortion) {
		vec2 uv_after_distortion = pos_to_uv(ray(1.0f), resolution, focal_length, camera, screen_center, parallax_shift, foveation);

		frame_buffer[idx].rgb() = to_rgb((uv_after_distortion - uv) * 64.0f);
		frame_buffer[idx].a = 1.0f;
		depth_buffer[idx] = 1.0f;
		payload.origin = ray(MAX_DEPTH());
		payload.alive = false;
		return;
	}

	ray.d = normalize(ray.d);

	if (envmap) {
		frame_buffer[idx] = read_envmap(envmap, ray.d);
	}

	float t = fmaxf(render_aabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;

	if (!render_aabb.contains(render_aabb_to_local * ray(t))) {
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}

	payload.origin = ray.o;
	payload.dir = ray.d;
	payload.t = t;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = true;
}

static constexpr float MIN_PDF = 0.01f;

__global__ void construct_cdf_2d_geometry(
	uint32_t n_images,
	uint32_t height,
	uint32_t width,
	const float* __restrict__ data,
	float* __restrict__ cdf_x_cond_y,
	float* __restrict__ cdf_y
) {
	const uint32_t y = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t img = threadIdx.y + blockIdx.y * blockDim.y;
	if (y >= height || img >= n_images) return;

	const uint32_t offset_xy = img * height * width + y * width;
	data += offset_xy;
	cdf_x_cond_y += offset_xy;

	float cum = 0;
	for (uint32_t x = 0; x < width; ++x) {
		cum += data[x] + 1e-10f;
		cdf_x_cond_y[x] = cum;
	}

	cdf_y[img * height + y] = cum;
	float norm = __frcp_rn(cum);

	for (uint32_t x = 0; x < width; ++x) {
		cdf_x_cond_y[x] = (1.0f - MIN_PDF) * cdf_x_cond_y[x] * norm + MIN_PDF * (float)(x+1) / (float)width;
	}
}

__global__ void construct_cdf_1d_geometry(
	uint32_t n_images,
	uint32_t height,
	float* __restrict__ cdf_y,
	float* __restrict__ cdf_img
) {
	const uint32_t img = threadIdx.x + blockIdx.x * blockDim.x;
	if (img >= n_images) return;

	cdf_y += img * height;

	float cum = 0;
	for (uint32_t y = 0; y < height; ++y) {
		cum += cdf_y[y];
		cdf_y[y] = cum;
	}

	cdf_img[img] = cum;

	float norm = __frcp_rn(cum);
	for (uint32_t y = 0; y < height; ++y) {
		cdf_y[y] = (1.0f - MIN_PDF) * cdf_y[y] * norm + MIN_PDF * (float)(y+1) / (float)height;
	}
}

__global__ void safe_divide_geometry(const uint32_t num_elements, float* __restrict__ inout, const float* __restrict__ divisor) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	float local_divisor = divisor[i];
	inout[i] = local_divisor > 0.0f ? (inout[i] / local_divisor) : 0.0f;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// mesh
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Testbed::MyTracer::init_rays_from_camera_mesh(
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
	cudaStream_t stream
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	size_t n_pixels = (size_t)resolution.x * resolution.y;
	enlarge_mesh(n_pixels, stream);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1 };
	init_rays_with_payload_kernel_mesh_geometry<<<blocks, threads, 0, stream>>>(
		sample_index,
		m_rays_mesh[0].pos,
		m_rays_mesh[0].distance,
		m_rays_mesh[0].payload,
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
		hidden_area_mask
	);
	m_n_rays_initialized_mesh = (uint32_t)n_pixels;
	// tlog::info() << "m_n_rays_initialized_mesh: " << m_n_rays_initialized_mesh;
}

void Testbed::MyTracer::init_rays_from_data_mesh(uint32_t n_elements, const RaysMeshSoa& data, cudaStream_t stream) {
	enlarge_mesh(n_elements, stream);
	m_rays_mesh[0].copy_from_other_async(n_elements, data, stream);
	m_n_rays_initialized_mesh = n_elements;
}

uint32_t Testbed::MyTracer::trace_mesh_bvh(GeometryBvh* bvh, const MeshData* meshes, cudaStream_t stream) {
	uint32_t n_alive = m_n_rays_initialized_mesh;
	m_n_rays_initialized_mesh = 0;

	if (!bvh) {
		return 0;
	}

	// Abuse the normal buffer to temporarily hold ray directions
	parallel_for_gpu(stream, n_alive, [payloads=m_rays_mesh[0].payload, normals=m_rays_mesh[0].normal] __device__ (size_t i) {
		normals[i] = payloads[i].dir;
	});

	bvh->ray_trace_mesh_gpu(n_alive, m_rays_mesh[0].pos, m_rays_mesh[0].normal, meshes, stream);
	return n_alive;
}

// allocate and distribute workspace memory for rays
void Testbed::MyTracer::enlarge_mesh(size_t n_elements, cudaStream_t stream) {
	n_elements = next_multiple(n_elements, size_t(BATCH_SIZE_GRANULARITY));
	auto scratch = allocate_workspace_and_distribute<
		vec3, vec3, float, float, float, float, GeometryPayload, // m_rays[0]
		vec3, vec3, float, float, float, float, GeometryPayload, // m_rays[1]
		vec3, vec3, float, float, float, float, GeometryPayload, // m_rays_hit

		uint32_t,
		uint32_t
	>(
		stream, &m_scratch_alloc_mesh,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements, n_elements, n_elements, n_elements, n_elements,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays_mesh[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), std::get<6>(scratch));
	m_rays_mesh[1].set(std::get<7>(scratch), std::get<8>(scratch), std::get<9>(scratch), std::get<10>(scratch), std::get<11>(scratch), std::get<12>(scratch), std::get<13>(scratch));
	m_rays_hit_mesh.set(std::get<14>(scratch), std::get<15>(scratch), std::get<16>(scratch), std::get<17>(scratch), std::get<18>(scratch), std::get<19>(scratch), std::get<20>(scratch));

	m_hit_counter_mesh = std::get<21>(scratch);
	m_alive_counter_mesh = std::get<22>(scratch);
}


void Testbed::render_geometry_mesh(
	cudaStream_t stream,
	const normals_fun_t& normals_function,
	const CudaRenderBufferView& render_buffer,
	const vec2& focal_length,
	const mat4x3& camera_matrix,
	const vec2& screen_center,
	const Foveation& foveation,
	int visualized_dimension
) {
	// tlog::info() << "render_geometry_mesh start ";
	
	// switch(m_render_mode) {
    //     case ERenderMode::AO: tlog::info() << "AO"; break;
    //     case ERenderMode::Shade: tlog::info() << "Shade"; break;
    // 	case ERenderMode::Normals: tlog::info() << "Normals"; break;
    // 	case ERenderMode::Positions: tlog::info() << "Positions"; break;
    // 	case ERenderMode::Depth: tlog::info() << "Depth"; break;
    // 	case ERenderMode::Distortion: tlog::info() << "Distortion"; break;
    // 	case ERenderMode::Cost: tlog::info() << "Cost"; break;
    // 	case ERenderMode::Slice: tlog::info() << "Slice"; break;
    // 	case ERenderMode::NumRenderModes: tlog::info() << "NumRenderModes"; break;
    // 	case ERenderMode::EncodingVis: tlog::info() << "EncodingVis"; break;
    // 	default: tlog::info() << "Unknown"; break;

    // }
	tlog::info() << "Camera Matrix:";
	for (int i = 0; i < 4; ++i) {
	    for (int j = 0; j < 3; ++j) {
	        tlog::info() << camera_matrix[i][j];
	    }
	}

	float plane_z = m_slice_plane_z + m_scale;
	if (m_render_mode == ERenderMode::Slice) {
		plane_z = -plane_z;
	}

	MyTracer tracer;

	BoundingBox mesh_bounding_box = m_geometry.geometry_mesh_bvh->get_nodes()[0].bb;	//the biggest bb, the root node
	// tlog::info() << "setting bounding box" ;
	// tlog::info() << "mesh_bounding_box.min: (" << mesh_bounding_box.min.x << ", " << mesh_bounding_box.min.y << ", " << mesh_bounding_box.min.z << ")";
	// tlog::info() << "mesh_bounding_box.max: (" << mesh_bounding_box.max.x << ", " << mesh_bounding_box.max.y << ", " << mesh_bounding_box.max.z << ")";
	mesh_bounding_box.inflate(m_geometry.zero_offset);
	
	tracer.init_rays_from_camera_mesh(
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
		stream
	);

	bool gt_raytrace = true;

	auto trace = [&](MyTracer& tracer) {
		return tracer.trace_mesh_bvh(m_geometry.geometry_mesh_bvh.get(), m_geometry.mesh_cpu.data(), stream);
	};

	uint32_t n_hit;
	if (m_render_mode == ERenderMode::Slice) {
		n_hit = tracer.n_rays_initialized();
	} else {
		n_hit = trace(tracer);
	}


	RaysMeshSoa& rays_hit = m_render_mode == ERenderMode::Slice || gt_raytrace ? tracer.rays_init() : tracer.rays_hit();

	// if (m_render_mode == ERenderMode::Slice) {
	// 	if (visualized_dimension == -1) {
	// 		distance_function(n_hit, rays_hit.pos, rays_hit.distance, stream);
	// 		extract_dimension_pos_neg_kernel<float><<<n_blocks_linear(n_hit*3), N_THREADS_LINEAR, 0, stream>>>(n_hit*3, 0, 1, 3, rays_hit.distance, CM, (float*)rays_hit.normal);
	// 	} else {
	// 		// Store colors in the normal buffer
	// 		uint32_t n_elements = next_multiple(n_hit, BATCH_SIZE_GRANULARITY);

	// 		GPUMatrix<float> positions_matrix((float*)rays_hit.pos, 3, n_elements);
	// 		GPUMatrix<float> colors_matrix((float*)rays_hit.normal, 3, n_elements);
	// 		m_network->visualize_activation(stream, m_visualized_layer, visualized_dimension, positions_matrix, colors_matrix);
	// 	}
	// }


	ERenderMode render_mode = (visualized_dimension > -1 || m_render_mode == ERenderMode::Slice) ? ERenderMode::EncodingVis : m_render_mode;
	if (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Normals) {
		normals_function(n_hit, rays_hit.pos, rays_hit.normal, stream);

		if (render_mode == ERenderMode::Shade && n_hit > 0) {
			// Shadow rays towards the sun
			MyTracer shadow_tracer;

			shadow_tracer.init_rays_from_data_mesh(n_hit, rays_hit, stream);
			shadow_tracer.set_trace_shadow_rays(true);
			shadow_tracer.set_shadow_sharpness(m_geometry.shadow_sharpness);
			RaysMeshSoa& shadow_rays_init = shadow_tracer.rays_init();
			linear_kernel(prepare_shadow_rays_geometry, 0, stream,
				n_hit,
				normalize(m_sun_dir),
				shadow_rays_init.pos,
				shadow_rays_init.normal,
				shadow_rays_init.distance,
				shadow_rays_init.prev_distance,
				shadow_rays_init.total_distance,
				shadow_rays_init.min_visibility,
				shadow_rays_init.payload,
				mesh_bounding_box
			);

			uint32_t n_hit_shadow = trace(shadow_tracer);
			auto& shadow_rays_hit = gt_raytrace ? shadow_tracer.rays_init() : shadow_tracer.rays_hit();

			linear_kernel(write_shadow_ray_result_geometry, 0, stream,
				n_hit_shadow,
				mesh_bounding_box,
				shadow_rays_hit.pos,
				shadow_rays_hit.payload,
				shadow_rays_hit.min_visibility,
				rays_hit.distance
			);
		}
	} else if (render_mode == ERenderMode::EncodingVis && m_render_mode != ERenderMode::Slice) {
		// HACK: Store colors temporarily in the normal buffer
		uint32_t n_elements = next_multiple(n_hit, BATCH_SIZE_GRANULARITY);

		GPUMatrix<float> positions_matrix((float*)rays_hit.pos, 3, n_elements);
		GPUMatrix<float> colors_matrix((float*)rays_hit.normal, 3, n_elements);
		m_network->visualize_activation(stream, m_visualized_layer, visualized_dimension, positions_matrix, colors_matrix);
	}
	

	// this has a bug with write_shadow_ray_result_geometry
	// fix it for the future

	// if ( n_hit > 0) {
	// 	// Shadow rays towards the sun
	// 	MyTracer shadow_tracer;
	// 	shadow_tracer.init_rays_from_data_mesh(n_hit, rays_hit, stream);
	// 	shadow_tracer.set_trace_shadow_rays(true);
	// 	shadow_tracer.set_shadow_sharpness(m_geometry.shadow_sharpness);
	// 	RaysMeshSoa& shadow_rays_init = shadow_tracer.rays_init();
	// 	linear_kernel(prepare_shadow_rays_geometry, 0, stream,
	// 		n_hit,
	// 		normalize(m_sun_dir),
	// 		shadow_rays_init.pos,
	// 		shadow_rays_init.normal,
	// 		shadow_rays_init.distance,
	// 		shadow_rays_init.prev_distance,
	// 		shadow_rays_init.total_distance,
	// 		shadow_rays_init.min_visibility,
	// 		shadow_rays_init.payload,
	// 		mesh_bounding_box
	// 	);
	// 	uint32_t n_hit_shadow = trace(shadow_tracer);
	// 	auto& shadow_rays_hit =  shadow_tracer.rays_init();
	// 	linear_kernel(write_shadow_ray_result_geometry, 0, stream,
	// 		n_hit_shadow,
	// 		mesh_bounding_box,
	// 		shadow_rays_hit.pos,
	// 		shadow_rays_hit.payload,
	// 		shadow_rays_hit.min_visibility,
	// 		rays_hit.distance
	// 	);
	// }

	linear_kernel(shade_kernel_mesh_geometry, 0, stream,
		n_hit,
		mesh_bounding_box,
		get_floor_y(),
		m_render_mode,
		m_geometry.brdf,	//not sure how to handle brdf
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

	if (render_mode == ERenderMode::Cost) {
		std::vector<GeometryPayload> payloads_final_cpu(n_hit);
		CUDA_CHECK_THROW(cudaMemcpyAsync(payloads_final_cpu.data(), rays_hit.payload, n_hit * sizeof(SdfPayload), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		size_t total_n_steps = 0;
		for (uint32_t i = 0; i < n_hit; ++i) {
			total_n_steps += payloads_final_cpu[i].n_steps;
		}

		tlog::info() << "Total steps per hit= " << total_n_steps << "/" << n_hit << " = " << ((float)total_n_steps/(float)n_hit);
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// nerf
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// todo: change the bounding box to the nerf bounding box from the nerf bvh
void Testbed::MyTracer::init_rays_from_camera_nerf(
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
	enlarge_nerf(n_pixels, padded_output_width, n_extra_dims, stream);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1 };
	init_rays_with_payload_kernel_nerf_geometry<<<blocks, threads, 0, stream>>>(
		sample_index,
		m_rays_nerf[0].payload,
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

	m_n_rays_initialized_nerf = resolution.x * resolution.y;

	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays_nerf[0].rgba, 0, m_n_rays_initialized_nerf * sizeof(vec4), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays_nerf[0].depth, 0, m_n_rays_initialized_nerf * sizeof(float), stream));

	linear_kernel(advance_pos_nerf_kernel_geometry, 0, stream,
		m_n_rays_initialized_nerf,
		render_aabb,
		render_aabb_to_local,
		camera_matrix1[2],
		focal_length,
		sample_index,
		m_rays_nerf[0].payload,
		grid,
		(show_accel >= 0) ? show_accel : 0,
		max_mip,
		cone_angle_constant
	);
}

uint32_t Testbed::MyTracer::trace_nerf_bvh(GeometryBvh* bvh, const Nerf* nerfs, cudaStream_t stream) {
	uint32_t n_alive = m_n_rays_initialized_nerf;
	m_n_rays_initialized_nerf = 0;

	if (!bvh) {
		return 0;
	}

	// // Abuse the normal buffer to temporarily hold ray directions
	// parallel_for_gpu(stream, n_alive, [payloads=m_rays_nerf[0].payload, normals=m_rays_nerf[0].normal] __device__ (size_t i) {
	// 	normals[i] = payloads[i].dir;
	// });

	// bvh->ray_trace_nerf_gpu(n_alive, m_rays_nerf[0].pos, m_rays_nerf[0].normal, nerfs, stream);
	return n_alive;
}

uint32_t Testbed::MyTracer::trace_nerf(
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
	if (m_n_rays_initialized_nerf == 0) {
		return 0;
	}

	CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter_nerf, 0, sizeof(uint32_t), stream));

	uint32_t n_alive = m_n_rays_initialized_nerf;
	// m_n_rays_initialized = 0;

	uint32_t i = 1;
	uint32_t double_buffer_index = 0;
	while (i < MARCH_ITER) {
		RaysNerfSoa& rays_current = m_rays_nerf[(double_buffer_index + 1) % 2];
		RaysNerfSoa& rays_tmp = m_rays_nerf[double_buffer_index % 2];
		++double_buffer_index;

		// Compact rays that did not diverge yet
		{
			CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter_nerf, 0, sizeof(uint32_t), stream));
			linear_kernel(compact_kernel_nerf_geometry, 0, stream,
				n_alive,
				rays_tmp.rgba, rays_tmp.depth, rays_tmp.payload,
				rays_current.rgba, rays_current.depth, rays_current.payload,
				m_rays_hit_nerf.rgba, m_rays_hit_nerf.depth, m_rays_hit_nerf.payload,
				m_alive_counter_nerf, m_hit_counter_nerf
			);
			CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter_nerf, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		}

		if (n_alive == 0) {
			break;
		}

		// Want a large number of queries to saturate the GPU and to ensure compaction doesn't happen toooo frequently.
		uint32_t target_n_queries = 2 * 1024 * 1024;
		uint32_t n_steps_between_compaction = clamp(target_n_queries / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);

		uint32_t extra_stride = network->n_extra_dims() * sizeof(float);
		PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input_nerf, 1, 0, extra_stride);
		linear_kernel(generate_next_nerf_network_inputs_geometry, 0, stream,
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
		GPUMatrix<float> positions_matrix((float*)m_network_input_nerf, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
		GPUMatrix<network_precision_t, RM> rgbsigma_matrix((network_precision_t*)m_network_output_nerf, network->padded_output_width(), n_elements);
		network->inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

		if (render_mode == ERenderMode::Normals) {
			network->input_gradient(stream, 3, positions_matrix, positions_matrix);
		} else if (render_mode == ERenderMode::EncodingVis) {
			network->visualize_activation(stream, visualized_layer, visualized_dim, positions_matrix, positions_matrix);
		}

		linear_kernel(composite_kernel_nerf_geometry, 0, stream,
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
			m_network_output_nerf,
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
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter_nerf, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	return n_hit;
}

void Testbed::MyTracer::enlarge_nerf(size_t n_elements, uint32_t padded_output_width, uint32_t n_extra_dims, cudaStream_t stream) {
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
		stream, &m_scratch_alloc_nerf,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * padded_output_width,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * num_floats,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays_nerf[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements);
	m_rays_nerf[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements);
	m_rays_hit_nerf.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements);

	m_network_output_nerf = std::get<9>(scratch);
	m_network_input_nerf = std::get<10>(scratch);

	m_hit_counter_nerf = std::get<11>(scratch);
	m_alive_counter_nerf = std::get<12>(scratch);
}

void Testbed::render_geometry_nerf(
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
	int visualized_dimension
) {
	float plane_z = m_slice_plane_z + m_scale;
	if (m_render_mode == ERenderMode::Slice) {
		plane_z = -plane_z;
	}

	ERenderMode render_mode = visualized_dimension > -1 ? ERenderMode::EncodingVis : m_render_mode;

	const float* extra_dims_gpu = m_nerf.get_rendering_extra_dims(stream);

	NerfTracer tracer;

	// Our motion vector code can't undo grid distortions -- so don't render grid distortion if DLSS is enabled.
	// (Unless we're in distortion visualization mode, in which case the distortion grid is fine to visualize.)
	auto grid_distortion =
		m_nerf.render_with_lens_distortion && (!m_dlss || m_render_mode == ERenderMode::Distortion) ?
		m_distortion.inference_view() :
		Buffer2DView<const vec2>{};

	Lens lens = m_nerf.render_with_lens_distortion ? m_nerf.render_lens : Lens{};

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
		m_nerf.show_accel,
		m_nerf.max_cascade,
		m_nerf.cone_angle_constant,
		render_mode,
		stream
	);

	float depth_scale = 1.0f / m_nerf.training.dataset.scale;
	bool render_2d = m_render_mode == ERenderMode::Slice || m_render_mode == ERenderMode::Distortion;

	uint32_t n_hit;
	if (render_2d) {
		n_hit = tracer.n_rays_initialized();
	} else {
		n_hit = tracer.trace(
			nerf_network,
			m_render_aabb,
			m_render_aabb_to_local,
			m_aabb,
			focal_length,
			m_nerf.cone_angle_constant,
			density_grid_bitfield,
			render_mode,
			camera_matrix1,
			depth_scale,
			m_visualized_layer,
			visualized_dimension,
			m_nerf.rgb_activation,
			m_nerf.density_activation,
			m_nerf.show_accel,
			m_nerf.max_cascade,
			m_nerf.render_min_transmittance,
			m_nerf.glow_y_cutoff,
			m_nerf.glow_mode,
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

		linear_kernel(generate_nerf_network_inputs_at_current_position_geometry, 0, stream, n_hit, m_aabb, rays_hit.payload, PitchedPtr<NerfCoordinate>((NerfCoordinate*)positions_matrix.data(), 1, 0, extra_stride), extra_dims_gpu);

		if (visualized_dimension == -1) {
			nerf_network->inference(stream, positions_matrix, rgbsigma_matrix);
			linear_kernel(compute_nerf_rgba_kernel_geometry, 0, stream, n_hit, (vec4*)rgbsigma_matrix.data(), m_nerf.rgb_activation, m_nerf.density_activation, 0.01f, false);
		} else {
			nerf_network->visualize_activation(stream, m_visualized_layer, visualized_dimension, positions_matrix, rgbsigma_matrix);
		}

		linear_kernel(shade_kernel_nerf_geometry, 0, stream,
			n_hit,
			m_nerf.render_gbuffer_hard_edges,
			camera_matrix1,
			depth_scale,
			(vec4*)rgbsigma_matrix.data(),
			nullptr,
			rays_hit.payload,
			m_render_mode,
			m_nerf.training.linear_colors,
			render_buffer.frame_buffer,
			render_buffer.depth_buffer
		);
		return;
	}

	linear_kernel(shade_kernel_nerf_geometry, 0, stream,
		n_hit,
		m_nerf.render_gbuffer_hard_edges,
		camera_matrix1,
		depth_scale,
		rays_hit.rgba,
		rays_hit.depth,
		rays_hit.payload,
		m_render_mode,
		m_nerf.training.linear_colors,
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


///////////////////////////////////////////////////////////////////////////////////////////////////
// loading functions
///////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<vec3> geometry_load_stl(const fs::path& path) {
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

void Testbed::load_mesh(MeshData* mesh, const fs::path& data_path, vec3 center) {

	tlog::info() << "Loading mesh from '" << data_path << "'";
	auto start = std::chrono::steady_clock::now();

	std::vector<vec3> vertices;
	if (equals_case_insensitive(data_path.extension(), "obj")) {
		vertices = load_obj(data_path.str());
	} else if (equals_case_insensitive(data_path.extension(), "stl")) {
		vertices = geometry_load_stl(data_path.str());
	} else {
		throw std::runtime_error{"mesh data path must be a mesh in ascii .obj or binary .stl format."};
	}

	// The expected format is
	// [v1.x][v1.y][v1.z][v2.x]...
	size_t n_vertices = vertices.size();
	size_t n_triangles = n_vertices/3;

	// for (size_t i = 0; i < n_vertices; ++i) {
	// 	tlog::info() << "Vertex " << i << ": [" << vertices[i].x << ", " << vertices[i].y << ", " << vertices[i].z << "]";

	// }

	// Compute the AABB of the mesh
	vec3 inf(std::numeric_limits<float>::infinity());
	BoundingBox aabb (inf, -inf);

	for (size_t i = 0; i < n_vertices; ++i) {
	    aabb.enlarge(vertices[i]);
	}

	// // Normalize the vertices.
	// for (size_t i = 0; i < n_vertices; ++i) {
	//     vertices[i] = (vertices[i] - aabb.center()) / max(aabb.diag());

	// }


	// Store the center and scale for later use.
	// I dont think we need center! and maybe scale is not needed as well
	// if not needed delete from the struct
	(*mesh).center = center;
	(*mesh).scale = max(aabb.diag());

	// Normalize vertex coordinates to lie within [0,1]^3.
	// This way, none of the constants need to carry around
	// bounding box factors.
	// for (size_t i = 0; i < n_vertices; ++i) {
	// 	vertices[i] = (vertices[i] - aabb.min - 0.5f * aabb.diag()) / (*mesh).scale  + 0.5f;
	// }

	(*mesh).triangles_cpu.resize(n_triangles);
	for (size_t i = 0; i < n_vertices; i += 3) {
		(*mesh).triangles_cpu[i/3] = {vertices[i+0], vertices[i+1], vertices[i+2]};
	}

	if (!(*mesh).triangle_bvh) {
		(*mesh).triangle_bvh = TriangleBvh::make();
	}

	(*mesh).triangle_bvh->build((*mesh).triangles_cpu, 8);
	(*mesh).triangles_gpu.resize_and_copy_from_host((*mesh).triangles_cpu);

	// initializes optix and creates OptiX program raytrace
	// (*mesh).triangle_bvh->build_optix((*mesh).triangles_gpu, m_stream.get());

	// (*mesh).triangle_octree.reset(new TriangleOctree{});
	// (*mesh).triangle_octree->build(*(*mesh).triangle_bvh, (*mesh).triangles_cpu, 10);

	// Compute discrete probability distribution for later sampling of the (*mesh)'s surface
	(*mesh).triangle_weights.resize(n_triangles);
	for (size_t i = 0; i < n_triangles; ++i) {
		(*mesh).triangle_weights[i] = (*mesh).triangles_cpu[i].surface_area();
	}
	(*mesh).triangle_distribution.build((*mesh).triangle_weights);

	// Move CDF to gpu
	(*mesh).triangle_cdf.resize_and_copy_from_host((*mesh).triangle_distribution.cdf);

	// Clear training data as it's no longer representative
	// of the previously loaded mesh.. but don't clear the network.
	// Perhaps it'll look interesting while morphing from one mesh to another.
	// mesh.training.idx = 0;
	// mesh.training.size = 0;

	tlog::success() << "Loaded mesh after " << tlog::durationToString(std::chrono::steady_clock::now() - start);
	tlog::info() << "  n_triangles=" << n_triangles;

}

void Testbed::load_empty_mesh(MeshData* mesh, vec3 center) {
	(*mesh).center = center;
	(*mesh).scale = 1.0f;

}

void Testbed::load_nerf_post(Nerf* nerf, const vec3 center) { // moved the second half of load_nerf here
	(*nerf).rgb_activation = (*nerf).training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

	(*nerf).training.n_images_for_training = (int)(*nerf).training.dataset.n_images;

	(*nerf).training.dataset.update_metadata();

	(*nerf).training.cam_pos_gradient.resize((*nerf).training.dataset.n_images, vec3(0.0f));
	(*nerf).training.cam_pos_gradient_gpu.resize_and_copy_from_host((*nerf).training.cam_pos_gradient);

	(*nerf).training.cam_exposure.resize((*nerf).training.dataset.n_images, AdamOptimizer<vec3>(1e-3f));
	(*nerf).training.cam_pos_offset.resize((*nerf).training.dataset.n_images, AdamOptimizer<vec3>(1e-4f));
	(*nerf).training.cam_rot_offset.resize((*nerf).training.dataset.n_images, RotationAdamOptimizer(1e-4f));
	(*nerf).training.cam_focal_length_offset = AdamOptimizer<vec2>(1e-5f);

	(*nerf).training.cam_rot_gradient.resize((*nerf).training.dataset.n_images, vec3(0.0f));
	(*nerf).training.cam_rot_gradient_gpu.resize_and_copy_from_host((*nerf).training.cam_rot_gradient);

	(*nerf).training.cam_exposure_gradient.resize((*nerf).training.dataset.n_images, vec3(0.0f));
	(*nerf).training.cam_exposure_gpu.resize_and_copy_from_host((*nerf).training.cam_exposure_gradient);
	(*nerf).training.cam_exposure_gradient_gpu.resize_and_copy_from_host((*nerf).training.cam_exposure_gradient);

	(*nerf).training.cam_focal_length_gradient = vec2(0.0f);
	(*nerf).training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&(*nerf).training.cam_focal_length_gradient, 1);

	(*nerf).reset_extra_dims(m_rng);
	(*nerf).training.optimize_extra_dims = (*nerf).training.dataset.n_extra_learnable_dims > 0;

	if ((*nerf).training.dataset.has_rays) {
		(*nerf).training.near_distance = 0.0f;
	}

	(*nerf).training.update_transforms();

	if (!(*nerf).training.dataset.metadata.empty()) {
		(*nerf).render_lens = (*nerf).training.dataset.metadata[0].lens;
		m_screen_center = vec2(1.f) - (*nerf).training.dataset.metadata[0].principal_point;
	}

	if (!is_pot((*nerf).training.dataset.aabb_scale)) {
		throw std::runtime_error{fmt::format("(*nerf) dataset's `aabb_scale` must be a power of two, but is {}.", (*nerf).training.dataset.aabb_scale)};
	}

	int max_aabb_scale = 1 << (NERF_CASCADES()-1);
	if ((*nerf).training.dataset.aabb_scale > max_aabb_scale) {
		throw std::runtime_error{fmt::format(
			"(*nerf) dataset must have `aabb_scale <= {}`, but is {}. "
			"You can increase this limit by factors of 2 by incrementing `(*nerf)_CASCADES()` and re-compiling.",
			max_aabb_scale, (*nerf).training.dataset.aabb_scale
		)};
	}

	// m_aabb = BoundingBox{vec3(0.5f), vec3(0.5f)};
	// m_aabb.inflate(0.5f * std::min(1 << (NERF_CASCADES()-1), (*nerf).training.dataset.aabb_scale));
	// m_raw_aabb = m_aabb;
	// m_render_aabb = m_aabb;
	// m_render_aabb_to_local = (*nerf).training.dataset.render_aabb_to_local;
	// if (!(*nerf).training.dataset.render_aabb.is_empty()) {
	// 	m_render_aabb = (*nerf).training.dataset.render_aabb.intersection(m_aabb);
	// }
	
	// tlog::info() << "AABB: " << m_aabb;

	(*nerf).max_cascade = 0;
	while ((1 << (*nerf).max_cascade) < (*nerf).training.dataset.aabb_scale) {
		++(*nerf).max_cascade;
	}

	// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
	// stepping in larger scenes.
	(*nerf).cone_angle_constant = (*nerf).training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);

	m_up_dir = (*nerf).training.dataset.up;

	(*nerf).center = center;
	(*nerf).scale = (*nerf).training.dataset.aabb_scale;
}

void Testbed::load_nerf(Nerf* nerf, const fs::path& data_path, const vec3 center) {
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

		(*nerf).training.dataset = ngp::load_nerf(json_paths, (*nerf).sharpen);

	}

	load_nerf_post(nerf, center);
}

void Testbed::load_empty_nerf(Nerf* nerf, const vec3 center) {
	fs::path data_path = {};
	(*nerf).training.dataset = ngp::create_empty_nerf_dataset(0, 1, false);
	load_nerf(m_data_path);
	(*nerf).training.n_images_for_training = 0;
}

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
 	size_t mesh_count = 0;
    size_t nerf_count = 0;


	if (!data_path.empty()) {
		if (m_geometry.geometry_mesh_bvh) {
			m_geometry.geometry_mesh_bvh.reset();
		}
		if (m_geometry.geometry_nerf_bvh) {
			m_geometry.geometry_nerf_bvh.reset();
		}
		std::ifstream f{native_string(data_path)};
		nlohmann::json jsonfile = nlohmann::json::parse(f, nullptr, true, true);

		// std::cout << geometries.dump(4) << std::endl;

        if (jsonfile.empty()) {
            throw std::runtime_error{"Geometry file must contain an array of geometry metadata."};
        }

		nlohmann::json geometries = jsonfile["geometry"];

        // Count the number of Mesh and Nerf types
        for(auto& geometry : geometries) {
            std::string type = geometry["type"];
            if (type == "Mesh") {
                ++mesh_count;
            } else if (type == "Nerf") {
                ++nerf_count;
            }
        }

		std::cout << "Mesh count: " << mesh_count << std::endl;
		std::cout << "Nerf count: " << nerf_count << std::endl;

        // Resize the vectors
        m_geometry.mesh_cpu.resize(mesh_count);
        m_geometry.nerf_cpu.resize(nerf_count);

        size_t mesh_index = 0;
        size_t nerf_index = 0;

        // Load the geometries
        for(auto& geometry : geometries) {
            fs::path model_path = geometry["path"];
            std::string type = geometry["type"];
            std::vector<float> center = geometry["center"];
            vec3 center_vec(center[0], center[1], center[2]);
			
            if (type == "Mesh") {
				// Todo: move constructor and delete copy constructor
                load_mesh(&m_geometry.mesh_cpu[mesh_index++],model_path, center_vec);
            } else if (type == "Nerf") {
                Nerf nerf;
                load_nerf(&m_geometry.nerf_cpu[nerf_index++], model_path, center_vec);
            }
        }
	}
	
	else {

		m_geometry.mesh_cpu.resize(1);
    	m_geometry.nerf_cpu.resize(1);
		
		load_empty_mesh(&m_geometry.mesh_cpu[0], vec3(0.0f));
		load_empty_nerf(&m_geometry.nerf_cpu[0], vec3(0.0f));
		
	}
	if(mesh_count > 0) {
		m_geometry.geometry_mesh_bvh = GeometryBvh::make();
		// at the end we want each leaf to contain only one geometry
		tlog::info() << "Building mesh bvh";
		m_geometry.geometry_mesh_bvh->build_mesh(m_geometry.mesh_cpu, 1);
		tlog::info() << "Built mesh bvh";
	}
	if(nerf_count > 0) {
		m_geometry.geometry_nerf_bvh = GeometryBvh::make();
		m_geometry.geometry_nerf_bvh->build_nerf(m_geometry.nerf_cpu, 1);
	}

	tlog::success() << "Loaded scene";

}

}