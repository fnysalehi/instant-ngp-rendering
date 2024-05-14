/** @file   geometry_bvh.cuh
 *  @author Fatemeh Salehi
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/sdf.h>
#include <neural-graphics-primitives/nerf_device.cuh>
#include <neural-graphics-primitives/triangle.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>

namespace ngp {

// struct BRDFParams;
struct MeshData {
    	
		float mesh_scale;

    	GPUMemory<Triangle> triangles_gpu;
    	std::vector<Triangle> triangles_cpu;
    	std::vector<float> triangle_weights;
    	DiscreteDistribution triangle_distribution;
    	GPUMemory<float> triangle_cdf;
    	std::shared_ptr<TriangleBvh> triangle_bvh;

    	int octree_depth_target = 0;
		float zero_offset = 0;

		vec3 center = vec3(0.0f);

		BRDFParams brdf;

	};

}