/** @file   geometry_bvh.cu
 *  @author Fatemeh Salehi
 */

#include <neural-graphics-primitives/common_host.h>
#include <neural-graphics-primitives/geometry_bvh.cuh>

#include <tiny-cuda-nn/gpu_memory.h>

#include <stack>

namespace ngp {

constexpr float MAX_DIST = 10.0f;

NGP_HOST_DEVICE BoundingBox::BoundingBox(MeshData* begin, MeshData* end) {
    // Initialize the bounding box to the first point of the first triangle of the first mesh
    min = max = begin->triangles_cpu[0].a;
    for (auto it = begin; it != end; ++it) {
        enlarge(*it);
    }
}

NGP_HOST_DEVICE void BoundingBox::enlarge(const MeshData& mesh) {
	// add the translation here insted of in the build, for now I assume the center for meshes are not considered!
	for (const Triangle& triangle : mesh.triangles_cpu) {
	    // Enlarge the bounding box to include the current triangle's points
	    enlarge(triangle.a);
	    enlarge(triangle.b);
	    enlarge(triangle.c);

	}
}

NGP_HOST_DEVICE BoundingBox::BoundingBox(Nerf* begin, Nerf* end) {
	min = max = begin->center;
	inflate(begin->scale);
	for (auto it = begin; it != end; ++it) {
		enlarge(*it);
	}
}

NGP_HOST_DEVICE void BoundingBox::enlarge(const Nerf& other){
	BoundingBox otherBox(other.center, other.center);
    otherBox.inflate(other.scale);
	enlarge(otherBox);
}

// __global__ void raytrace_kernel(uint32_t n_elements, vec3* __restrict__ positions, vec3* __restrict__ directions, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles);
__global__ void mesh_raytrace_kernel(uint32_t n_elements, vec3* __restrict__ positions, vec3* __restrict__ directions, const GeometryBvhNode* __restrict__ nodes, const MeshData* __restrict__ meshes);		
__global__ void nerf_raytrace_kernel(uint32_t n_elements, vec3* __restrict__ positions, vec3* __restrict__ directions, const GeometryBvhNode* __restrict__ nodes, const Nerf* __restrict__ nerfs);

template <uint32_t BRANCHING_FACTOR>
class GeometryBvhWithBranchingFactor : public GeometryBvh {
public:

	__host__ __device__ static std::pair<int, float> ray_intersect(const vec3& ro, const vec3& rd, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
		FixedIntStack query_stack;
		query_stack.push(0);

		float mint = MAX_DIST;
		int shortest_idx = -1;

		while (!query_stack.empty()) {
			int idx = query_stack.pop();

			const TriangleBvhNode& node = bvhnodes[idx];
			// checks if it's a leaf node
			if (node.left_idx < 0) {

				// checks each triangle in the leaf node for intersection with the ray, updating mint and shortest_idx if a closer intersection is found
				int end = -node.right_idx-1;
				for (int i = -node.left_idx-1; i < end; ++i) {
					float t = triangles[i].ray_intersect(ro, rd);
					if (t < mint) {
						mint = t;
						shortest_idx = i;
					}
				}
			} 
			// calculates the intersection of the ray with the bounding boxes of the node's children and sorts them by distance
			else {
				DistAndIdx children[BRANCHING_FACTOR];

				uint32_t first_child = node.left_idx;

				NGP_PRAGMA_UNROLL
				for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
					children[i] = {bvhnodes[i+first_child].bb.ray_intersect(ro, rd).x, i+first_child};
				}

				sorting_network<BRANCHING_FACTOR>(children);

				// pushes the indices ofchildren with the closest bounding boxes (intersect with the ray) to the query stack
				NGP_PRAGMA_UNROLL
				for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
					if (children[i].dist < mint) {
						query_stack.push(children[i].idx);
					}
				}
			}
		}

		return {shortest_idx, mint};
	}


	__host__ __device__ static std::tuple<int, int, float> ray_intersect(const vec3& ro, const vec3& rd, const GeometryBvhNode* __restrict__ meshbvhnodes, const MeshData* __restrict__ meshes) {
		FixedIntStack query_stack;
		query_stack.push(0);

		float mint = MAX_DIST;
		int shortest_idx = -1;
		int mesh_idx = -1;

		while (!query_stack.empty()) {
			int idx = query_stack.pop();

			const GeometryBvhNode& node = meshbvhnodes[idx];
			
			float t = node.bb.ray_intersect(ro, rd).x;

        	// If the ray intersects the bounding box of the node
			if(t < std::numeric_limits<float>::max())
			{	

				// If it's a leaf node
				if (node.left_idx < 0) {
                	// checks each triangle in the mesh of the leaf node for intersection with the ray, updating mint and shortest_idx if a closer intersection is found
					
					int meshIdx = -node.left_idx-1;
					const MeshData& mesh = meshes[meshIdx];

					const TriangleBvhNode* bvhnodes = mesh.triangle_bvh->nodes_gpu();
					const Triangle* triangles = mesh.triangles_gpu.data();

					auto result = GeometryBvhWithBranchingFactor<BRANCHING_FACTOR>::ray_intersect(ro, rd, bvhnodes, triangles);

					// auto result = ray_intersect(ro, rd, mesh.triangle_bvh, mesh.triangles_gpu);
					
					if (result.second < mint) {
						shortest_idx = result.first;
						mint = result.second;
						mesh_idx = meshIdx;
					}	
				}
            	// If it's not a leaf node
				// same as trinagle bvh
				else {
					DistAndIdx children[BRANCHING_FACTOR];

					uint32_t first_child = node.left_idx;

					NGP_PRAGMA_UNROLL
					for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
						children[i] = {meshbvhnodes[i+first_child].bb.ray_intersect(ro, rd).x, i+first_child};
					}

					sorting_network<BRANCHING_FACTOR>(children);

					// pushes the indices ofchildren with the closest bounding boxes (intersect with the ray) to the query stack
					NGP_PRAGMA_UNROLL
					for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
						if (children[i].dist < mint) {
							query_stack.push(children[i].idx);
						}
					}
				}
			}
		}

		 return {mesh_idx, shortest_idx, mint};
	}

	__host__ __device__ static std::pair<int, float> ray_intersect(const vec3& ro, const vec3& rd, const GeometryBvhNode* __restrict__ nerfbvhnodes, const Nerf* __restrict__ nerfs) {
		FixedIntStack query_stack;
		query_stack.push(0);

		float mint = MAX_DIST;
		int shortest_idx = -1;

		while (!query_stack.empty()) {
			int idx = query_stack.pop();

			const GeometryBvhNode& node = nerfbvhnodes[idx];
			
			float t = node.bb.ray_intersect(ro, rd).x;

        	// If the ray intersects the bounding box of the node
			if(t < std::numeric_limits<float>::max())
			{	

				// If it's a leaf node
				if (node.left_idx < 0) {
					if (t < mint) {
						mint = t;
						shortest_idx = -node.left_idx-1;	//not sure
					}		
				}
            	// If it's not a leaf node
				// same as trinagle bvh
				else {
					DistAndIdx children[BRANCHING_FACTOR];

					uint32_t first_child = node.left_idx;

					NGP_PRAGMA_UNROLL
					for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
						children[i] = {nerfbvhnodes[i+first_child].bb.ray_intersect(ro, rd).x, i+first_child};
					}

					sorting_network<BRANCHING_FACTOR>(children);

					// pushes the indices ofchildren with the closest bounding boxes (intersect with the ray) to the query stack
					NGP_PRAGMA_UNROLL
					for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
						if (children[i].dist < mint) {
							query_stack.push(children[i].idx);
						}
					}
				}
			}
		}

		return {shortest_idx, mint};
	}

	
	
	void ray_trace_gpu(uint32_t n_elements, vec3* gpu_positions, vec3* gpu_directions, const MeshData* __restrict__ meshes, const Nerf* __restrict__ nerfs, cudaStream_t stream) override {
		linear_kernel(mesh_raytrace_kernel, 0, stream,
			n_elements,
			gpu_positions,
			gpu_directions,
			m_nodes_gpu.data(),
			meshes	
		);

		// it will be executed in sequence! maybe I can change it in the future
		linear_kernel(nerf_raytrace_kernel, 0, stream,
			n_elements,
			gpu_positions,
			gpu_directions,
			m_nodes_gpu.data(),
			nerfs	
		);
			
	}

	void ray_trace_mesh_gpu(uint32_t n_elements, vec3* gpu_positions, vec3* gpu_directions, const MeshData* __restrict__ meshes, cudaStream_t stream) override {
		linear_kernel(mesh_raytrace_kernel, 0, stream,
			n_elements,
			gpu_positions,
			gpu_directions,
			m_nodes_gpu.data(),
			meshes	
		);
			
	}

	void ray_trace_nerf_gpu(uint32_t n_elements, vec3* gpu_positions, vec3* gpu_directions, const Nerf* __restrict__ nerfs, cudaStream_t stream) override {
		linear_kernel(nerf_raytrace_kernel, 0, stream,
			n_elements,
			gpu_positions,
			gpu_directions,
			m_nodes_gpu.data(),
			nerfs	
		);
			
	}

	void build_mesh(std::vector<MeshData>& meshes, uint32_t n_primitives_per_leaf) override {
		m_nodes.clear();

		tlog::info() << "Building Mesh GeometryBvh with branching factor " << BRANCHING_FACTOR;
		
		// Root
		m_nodes.emplace_back();
		auto bb = BoundingBox(meshes.data(), meshes.data() + meshes.size());
		m_nodes.front().bb = bb;
		tlog::info() << " main aabb=" <<bb;


		struct BuildNode {
			int node_idx;
			std::vector<MeshData>::iterator begin;
			std::vector<MeshData>::iterator end;
		};

		std::stack<BuildNode> build_stack;
		build_stack.push({0, std::begin(meshes), std::end(meshes)});

		while (!build_stack.empty()) {
			const BuildNode& curr = build_stack.top();
			size_t node_idx = curr.node_idx;

			std::array<BuildNode, BRANCHING_FACTOR> c;
			c[0].begin = curr.begin;
			c[0].end = curr.end;

			build_stack.pop();

			// Partition the triangles into the children
			int number_c = 1;
			while (number_c < BRANCHING_FACTOR) {
				for (int i = number_c - 1; i >= 0; --i) {
					auto& child = c[i];

					// Choose axis with maximum standard deviation
					vec3 mean = vec3(0.0f);
					for (auto it = child.begin; it != child.end; ++it) {
						mean += it->center; // In the traingle bvh they use centroid instead of center!
					}
					mean /= (float)std::distance(child.begin, child.end);

					vec3 var = vec3(0.0f);
					for (auto it = child.begin; it != child.end; ++it) {
						vec3 diff = it->center - mean;
						var += diff * diff;
					}
					var /= (float)std::distance(child.begin, child.end);

					float max_val = max(var);
					int axis = var.x == max_val ? 0 : (var.y == max_val ? 1 : 2);

					auto m = child.begin + std::distance(child.begin, child.end)/2;
					std::nth_element(child.begin, m, child.end, [&](const MeshData& mesh1, const MeshData& mesh2) { return mesh1.center[0]+mesh1.center[1]+mesh1.center[2] < mesh2.center[0]+mesh2.center[1]+mesh2.center[2]; });

					c[i*2].begin = c[i].begin;
					c[i*2+1].end = c[i].end;
					c[i*2].end = c[i*2+1].begin = m;
				}

				number_c *= 2;
			}

			// Create next build nodes
			m_nodes[node_idx].left_idx = (int)m_nodes.size();
			for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
				auto& child = c[i];
				assert(child.begin != child.end);
				child.node_idx = (int)m_nodes.size();

				m_nodes.emplace_back();
				m_nodes.back().bb = BoundingBox(&*child.begin, &*child.end);
				// tlog::info() << " aabb=" << m_nodes.back().bb;


				if (std::distance(child.begin, child.end) <= n_primitives_per_leaf) {
					// I am not sure but I think this should be chnaged to this!
					int idx = -(int)std::distance(std::begin(meshes), child.begin)-1;
					m_nodes.back().left_idx = idx;
					m_nodes.back().right_idx = idx;
					// m_nodes.back().left_idx = -(int)std::distance(std::begin(meshes), child.begin)-1;
					// m_nodes.back().right_idx = -(int)std::distance(std::begin(meshes), child.end)-1;
				} else {
					build_stack.push(child);
				}
			}
			m_nodes[node_idx].right_idx = (int)m_nodes.size();
		}

		m_nodes_gpu.resize_and_copy_from_host(m_nodes);

		tlog::success() << "Built GeometryBvh: nodes=" << m_nodes.size();
	}

	void build_nerf(std::vector<Nerf>& nerfs, uint32_t n_primitives_per_leaf) override {
		m_nodes.clear();

		tlog::info() << "Building Nerf GeometryBvh with branching factor " << BRANCHING_FACTOR;
		
		// Root
		m_nodes.emplace_back();
		auto bb = BoundingBox(nerfs.data(), nerfs.data() + nerfs.size());
		m_nodes.front().bb = bb;
		tlog::info() << " main aabb=" <<bb;


		struct BuildNode {
			int node_idx;
			std::vector<Nerf>::iterator begin;
			std::vector<Nerf>::iterator end;
		};

		std::stack<BuildNode> build_stack;
		build_stack.push({0, std::begin(nerfs), std::end(nerfs)});

		while (!build_stack.empty()) {
			const BuildNode& curr = build_stack.top();
			size_t node_idx = curr.node_idx;

			std::array<BuildNode, BRANCHING_FACTOR> c;
			c[0].begin = curr.begin;
			c[0].end = curr.end;

			build_stack.pop();

			// Partition the triangles into the children
			int number_c = 1;
			while (number_c < BRANCHING_FACTOR) {
				for (int i = number_c - 1; i >= 0; --i) {
					auto& child = c[i];

					// Choose axis with maximum standard deviation
					vec3 mean = vec3(0.0f);
					for (auto it = child.begin; it != child.end; ++it) {
						mean += it->center; // In the traingle bvh they use centroid instead of center!
					}
					mean /= (float)std::distance(child.begin, child.end);

					vec3 var = vec3(0.0f);
					for (auto it = child.begin; it != child.end; ++it) {
						vec3 diff = it->center - mean;
						var += diff * diff;
					}
					var /= (float)std::distance(child.begin, child.end);

					float max_val = max(var);
					int axis = var.x == max_val ? 0 : (var.y == max_val ? 1 : 2);

					auto m = child.begin + std::distance(child.begin, child.end)/2;
					std::nth_element(child.begin, m, child.end, [&](const Nerf& nerf1, const Nerf& nerf2) { return nerf1.center[0]+nerf1.center[1]+nerf1.center[2] < nerf2.center[0]+nerf2.center[1]+nerf2.center[2]; });

					c[i*2].begin = c[i].begin;
					c[i*2+1].end = c[i].end;
					c[i*2].end = c[i*2+1].begin = m;
				}

				number_c *= 2;
			}

			// Create next build nodes
			m_nodes[node_idx].left_idx = (int)m_nodes.size();
			for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
				auto& child = c[i];
				assert(child.begin != child.end);
				child.node_idx = (int)m_nodes.size();

				m_nodes.emplace_back();
				m_nodes.back().bb = BoundingBox(&*child.begin, &*child.end);
				tlog::info() << " aabb=" << m_nodes.back().bb;


				if (std::distance(child.begin, child.end) <= n_primitives_per_leaf) {
					m_nodes.back().left_idx = -(int)std::distance(std::begin(nerfs), child.begin)-1;
					m_nodes.back().right_idx = -(int)std::distance(std::begin(nerfs), child.end)-1;
				} else {
					build_stack.push(child);
				}
			}
			m_nodes[node_idx].right_idx = (int)m_nodes.size();
		}

		m_nodes_gpu.resize_and_copy_from_host(m_nodes);

		tlog::success() << "Built GeometryBvh: nodes=" << m_nodes.size();
	}


	void build_optix(const GPUMemory<GeometryBvhNode>& nodes, cudaStream_t stream) override {
// #ifdef NGP_OPTIX
// 		// m_optix.available = optix::initialize();
// 		// if (m_optix.available) {
// 		// 	m_optix.gas = std::make_unique<optix::Gas>(nodes, g_optix, stream);	// Todo: Implement optix::Gas
// 		// 	m_optix.raytrace = std::make_unique<optix::Program<Raytrace>>((const char*)optix_ptx::raytrace_ptx, sizeof(optix_ptx::raytrace_ptx), g_optix);
// 		// 	tlog::success() << "Built OptiX GAS and shaders";
// 		// } else {
// 		// 	tlog::warning() << "Falling back to slower GeometryBvh::ray_intersect.";
// 		// }
// #else //NGP_OPTIX
// 		tlog::warning() << "OptiX was not built. Falling back to slower GeometryBvh::ray_intersect.";
// #endif //NGP_OPTIX
	}

	GeometryBvhWithBranchingFactor() {}

private:
// #ifdef NGP_OPTIX
// 	struct {
// 		std::unique_ptr<optix::Gas> gas;
// 		std::unique_ptr<optix::Program<Raytrace>> raytrace;
// 		bool available = true;
// 	} m_optix;
// #endif //NGP_OPTIX
};

using GeometryBvh4 = GeometryBvhWithBranchingFactor<4>;
using GeometryBvh1 = GeometryBvhWithBranchingFactor<2>;

std::unique_ptr<GeometryBvh> GeometryBvh::make() {
	return std::unique_ptr<GeometryBvh>(new GeometryBvh1());
}

__global__ void mesh_raytrace_kernel(uint32_t n_elements, vec3* __restrict__ positions, vec3* __restrict__ directions, const GeometryBvhNode* __restrict__ nodes, const MeshData* __restrict__ meshes) {		
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;	//index i of the current thread.
	if (i >= n_elements) return;

	auto pos = positions[i];
	auto dir = directions[i];

	auto p = GeometryBvh1::ray_intersect(pos, dir, nodes, meshes);
	// first element is the index of the intersected mesh
	// second element is the index of the intersected triangle
	// third element is the distance to the intersection

	// new positions = intersection points
	positions[i] = pos + std::get<2>(p) * dir;

    // if a mesh was hit, p.first is its triangle index and it updates the direction of the ray to the normal of the intersected traingle, 
    // otherwise p.first is -1.

	if (std::get<0>(p) >= 0) {
		auto& mesh = meshes[std::get<0>(p)];
		directions[i] = mesh.triangles_gpu[static_cast<size_t>(std::get<1>(p))].normal();
	}
}
__global__ void nerf_raytrace_kernel(uint32_t n_elements, vec3* __restrict__ positions, vec3* __restrict__ directions, const GeometryBvhNode* __restrict__ nodes, const Nerf* __restrict__ nerfs) {		
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;	//index i of the current thread.
	if (i >= n_elements) return;

	auto pos = positions[i];
	auto dir = directions[i];

	auto p = GeometryBvh4::ray_intersect(pos, dir, nodes, nerfs);
	// first element is the index of the intersected geometry and the second element is the distance to the intersection.
	
	// new positions = intersection points
	positions[i] = pos + p.second * dir;

	// not sure if the direction sho9uld be updated or not
}

}


