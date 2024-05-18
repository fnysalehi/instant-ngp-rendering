/** @file   geometry_bvh.cuh
 *  @author Fatemeh Salehi
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/sdf.h>
#include <neural-graphics-primitives/nerf_device.cuh>
#include <neural-graphics-primitives/triangle.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>


namespace ngp {

struct MeshData {
    	
	MeshData() 
    : scale(1.0f), 
      center(vec3(0.0f)), 
    //   brdf(BRDFParams()), 
      octree_depth_target(0), 
      zero_offset(0) 
    {
    }

	MeshData& operator=(const MeshData& other) {
        if (this != &other) {
            scale = other.scale;
            triangles_gpu = other.triangles_gpu;
            triangles_cpu = other.triangles_cpu;
            triangle_weights = other.triangle_weights;
            triangle_distribution = other.triangle_distribution;
            triangle_cdf = other.triangle_cdf;
            triangle_bvh = other.triangle_bvh;
            triangle_octree = other.triangle_octree;
            octree_depth_target = other.octree_depth_target;
            zero_offset = other.zero_offset;
            center = other.center;
            // brdf = other.brdf;
        }
        return *this;
    }
	
	float scale = 1.0f;
    GPUMemory<Triangle> triangles_gpu;
    std::vector<Triangle> triangles_cpu;
    std::vector<float> triangle_weights;
    DiscreteDistribution triangle_distribution;
    GPUMemory<float> triangle_cdf;
    std::shared_ptr<TriangleBvh> triangle_bvh;
	
	std::shared_ptr<TriangleOctree> triangle_octree;
    int octree_depth_target = 0;
	float zero_offset = 0;
	vec3 center = vec3(0.0f);
	// BRDFParams brdf;

	};

}