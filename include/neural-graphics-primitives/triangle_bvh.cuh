/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   triangle_bvh.cuh
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/triangle.cuh>

#include <tiny-cuda-nn/gpu_memory.h>

#include <memory>

namespace ngp {

struct TriangleBvhNode {
	BoundingBox bb;
	int left_idx; // negative values indicate leaves
	int right_idx;
};

// Sorting networks from http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html#N4L5D3
template <uint32_t N, typename T>
__host__ __device__ void sorting_network(T values[N]) {
	static_assert(N <= 8, "Sorting networks are only implemented up to N==8");
	if (N <= 1) {
		return;
	} else if (N == 2) {
		compare_and_swap(values[0], values[1]);
	} else if (N == 3) {
		compare_and_swap(values[0], values[2]);
		compare_and_swap(values[0], values[1]);
		compare_and_swap(values[1], values[2]);
	} else if (N == 4) {
		compare_and_swap(values[0], values[2]);
		compare_and_swap(values[1], values[3]);
		compare_and_swap(values[0], values[1]);
		compare_and_swap(values[2], values[3]);
		compare_and_swap(values[1], values[2]);
	} else if (N == 5) {
		compare_and_swap(values[0], values[3]);
		compare_and_swap(values[1], values[4]);

		compare_and_swap(values[0], values[2]);
		compare_and_swap(values[1], values[3]);

		compare_and_swap(values[0], values[1]);
		compare_and_swap(values[2], values[4]);

		compare_and_swap(values[1], values[2]);
		compare_and_swap(values[3], values[4]);

		compare_and_swap(values[2], values[3]);
	} else if (N == 6) {
		compare_and_swap(values[0], values[5]);
		compare_and_swap(values[1], values[3]);
		compare_and_swap(values[2], values[4]);

		compare_and_swap(values[1], values[2]);
		compare_and_swap(values[3], values[4]);

		compare_and_swap(values[0], values[3]);
		compare_and_swap(values[2], values[5]);

		compare_and_swap(values[0], values[1]);
		compare_and_swap(values[2], values[3]);
		compare_and_swap(values[4], values[5]);

		compare_and_swap(values[1], values[2]);
		compare_and_swap(values[3], values[4]);
	} else if (N == 7) {
		compare_and_swap(values[0], values[6]);
		compare_and_swap(values[2], values[3]);
		compare_and_swap(values[4], values[5]);

		compare_and_swap(values[0], values[2]);
		compare_and_swap(values[1], values[4]);
		compare_and_swap(values[3], values[6]);

		compare_and_swap(values[0], values[1]);
		compare_and_swap(values[2], values[5]);
		compare_and_swap(values[3], values[4]);

		compare_and_swap(values[1], values[2]);
		compare_and_swap(values[4], values[6]);

		compare_and_swap(values[2], values[3]);
		compare_and_swap(values[4], values[5]);

		compare_and_swap(values[1], values[2]);
		compare_and_swap(values[3], values[4]);
		compare_and_swap(values[5], values[6]);
	} else if (N == 8) {
		compare_and_swap(values[0], values[2]);
		compare_and_swap(values[1], values[3]);
		compare_and_swap(values[4], values[6]);
		compare_and_swap(values[5], values[7]);

		compare_and_swap(values[0], values[4]);
		compare_and_swap(values[1], values[5]);
		compare_and_swap(values[2], values[6]);
		compare_and_swap(values[3], values[7]);

		compare_and_swap(values[0], values[1]);
		compare_and_swap(values[2], values[3]);
		compare_and_swap(values[4], values[5]);
		compare_and_swap(values[6], values[7]);

		compare_and_swap(values[2], values[4]);
		compare_and_swap(values[3], values[5]);

		compare_and_swap(values[1], values[4]);
		compare_and_swap(values[3], values[6]);

		compare_and_swap(values[1], values[2]);
		compare_and_swap(values[3], values[4]);
		compare_and_swap(values[5], values[6]);
	}
}

template <typename T, int MAX_SIZE=32>
class FixedStack {
public:
	__host__ __device__ void push(T val) {
		if (m_count >= MAX_SIZE-1) {
			printf("WARNING TOO BIG\n");
		}
		m_elems[m_count++] = val;
	}

	__host__ __device__ T pop() {
		return m_elems[--m_count];
	}

	__host__ __device__ bool empty() const {
		return m_count <= 0;
	}

private:
	T m_elems[MAX_SIZE];
	int m_count = 0;
};

using FixedIntStack = FixedStack<int>;

struct DistAndIdx {
	float dist;
	uint32_t idx;

	// Sort in descending order!
	__host__ __device__ bool operator<(const DistAndIdx& other) {
		return dist < other.dist;
	}
};

template <typename T>
__host__ __device__ void inline compare_and_swap(T& t1, T& t2) {
	if (t1 < t2) {
		T tmp{t1}; t1 = t2; t2 = tmp;
	}
}

__host__ __device__ std::pair<int, float> trianglebvh_ray_intersect(const vec3& ro, const vec3& rd, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles);

class TriangleBvh {
public:
	virtual void signed_distance_gpu(uint32_t n_elements, EMeshSdfMode mode, const vec3* gpu_positions, float* gpu_distances, const Triangle* gpu_triangles, bool use_existing_distances_as_upper_bounds, cudaStream_t stream) = 0;
	virtual void ray_trace_gpu(uint32_t n_elements, vec3* gpu_positions, vec3* gpu_directions, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
	virtual bool touches_triangle(const BoundingBox& bb, const Triangle* __restrict__ triangles) const = 0;
	virtual void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) = 0;
	virtual void build_optix(const GPUMemory<Triangle>& triangles, cudaStream_t stream) = 0;

	static std::unique_ptr<TriangleBvh> make();

	TriangleBvhNode* nodes_gpu() const {
		return m_nodes_gpu.data();
	}

protected:
	std::vector<TriangleBvhNode> m_nodes;
	GPUMemory<TriangleBvhNode> m_nodes_gpu;
	TriangleBvh() {};
};

}
