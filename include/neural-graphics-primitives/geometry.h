/** @file   geometry.h
 *  @author Fatemeh Salehi
 */

#pragma once

#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/gpu_memory.h>

namespace ngp {

struct GeometryPayload {
	vec3 dir;
	uint32_t idx;
	uint16_t n_steps;
	bool alive;
};

struct RaysMeshSoa {
#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
    void copy_from_other_async(uint32_t n_elements, const RaysMeshSoa& other, cudaStream_t stream) {
        CUDA_CHECK_THROW(cudaMemcpyAsync(pos, other.pos, n_elements * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(normal, other.normal, n_elements * sizeof(vec3), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(distance, other.distance, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(prev_distance, other.prev_distance, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(total_distance, other.total_distance, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(payload, other.payload, n_elements * sizeof(GeometryPayload), cudaMemcpyDeviceToDevice, stream));
    }
#endif

	void set(vec3* pos, vec3* normal, float* distance, float* prev_distance, float* total_distance, GeometryPayload* payload) {
        this->pos = pos;
        this->normal = normal;
        this->distance = distance;
        this->prev_distance = prev_distance;
        this->total_distance = total_distance;
        this->payload = payload;
    }
    vec3* pos;
	vec3* normal;
	float* distance;
	float* prev_distance;
	float* total_distance;
	// float* min_visibility;   not sure if we need this
    // maybe add 
    GeometryPayload* payload;
};

// struct BRDFParams {
// 	float metallic=0.f;
// 	float subsurface=0.f;
// 	float specular=1.f;
// 	float roughness=0.5f;
// 	float sheen=0.f;
// 	float clearcoat=0.f;
// 	float clearcoat_gloss=0.f;
// 	vec3 basecolor = {0.8f, 0.8f, 0.8f};
// 	vec3 ambientcolor = {0.0f, 0.0f, 0.0f};
// };

}
