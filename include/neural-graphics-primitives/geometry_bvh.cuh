/** @file   geometry_bvh.cuh
 *  @author Fatemeh Salehi
 */

#pragma once

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/triangle.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/nerf.h>
#include <neural-graphics-primitives/mesh.h>

#include <tiny-cuda-nn/gpu_memory.h>

#include <memory>

namespace ngp {

enum class NodeType : int {
        MESH,
        NERF
};

struct GeometryBvhNode {
	BoundingBox bb;
	int left_idx; // negative values indicate leaves
	int right_idx;
    NodeType type;
    union {
            MeshData* mesh;
            Nerf* nerf;
    } data;
};

using FixedIntStack = FixedStack<int>;


__host__ __device__ std::pair<int, float> geometrybvh_ray_intersect(const vec3& ro, const vec3& rd, const GeometryBvhNode* __restrict__ bvhnodes);

class GeometryBvh {
public:
	virtual void ray_trace_gpu(uint32_t n_elements, vec3* gpu_positions, vec3* gpu_directions, const GeometryBvhNode* m_nodes_gpu, cudaStream_t stream) = 0;
	virtual void build(std::vector<GeometryBvhNode>& nodes, uint32_t n_primitives_per_leaf) = 0;
	virtual void build_optix(const GPUMemory<GeometryBvhNode>& nodes, cudaStream_t stream) = 0;

	static std::unique_ptr<GeometryBvh> make();


	GeometryBvhNode* nodes_gpu() const {
		return m_nodes_gpu.data();
	}

	// For debugging: storing intersected rays with leaf nodes bounding boxes
	__device__ void store_intersecting_ray(const vec3& ro, const vec3& p) {
        intersectedRays.push_back({ro, p});
    }

	std::vector<std::pair<vec3, vec3>> getIntersectedRays() const {
		return intersectedRays;
	}

protected:
	std::vector<std::pair<vec3, vec3>> intersectedRays;
	std::vector<GeometryBvhNode> m_nodes;
	GPUMemory<GeometryBvhNode> m_nodes_gpu;
	GeometryBvh() {};
};


}
