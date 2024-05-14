/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf_device.cuh>

#ifdef NGP_PYTHON
#  include <pybind11/pybind11.h>
#  include <pybind11/numpy.h>
#endif


namespace ngp {

struct NerfCounters {
		GPUMemory<uint32_t> numsteps_counter; // number of steps each ray took
		GPUMemory<uint32_t> numsteps_counter_compacted; // number of steps each ray took
		GPUMemory<float> loss;

		uint32_t rays_per_batch = 1<<12;
		uint32_t n_rays_total = 0;
		uint32_t measured_batch_size = 0;
		uint32_t measured_batch_size_before_compaction = 0;

		void prepare_for_training_steps(cudaStream_t stream);
		float update_after_training(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream);
	};

struct Nerf {
		struct Training {
			NerfDataset dataset;
			int n_images_for_training = 0; // how many images to train from, as a high watermark compared to the dataset size
			int n_images_for_training_prev = 0; // how many images we saw last time we updated the density grid

			struct ErrorMap {
				GPUMemory<float> data;
				GPUMemory<float> cdf_x_cond_y;
				GPUMemory<float> cdf_y;
				GPUMemory<float> cdf_img;
				std::vector<float> pmf_img_cpu;
				ivec2 resolution = {16, 16};
				ivec2 cdf_resolution = {16, 16};
				bool is_cdf_valid = false;
			} error_map;

			std::vector<TrainingXForm> transforms;
			GPUMemory<TrainingXForm> transforms_gpu;

			std::vector<vec3> cam_pos_gradient;
			GPUMemory<vec3> cam_pos_gradient_gpu;

			std::vector<vec3> cam_rot_gradient;
			GPUMemory<vec3> cam_rot_gradient_gpu;

			GPUMemory<vec3> cam_exposure_gpu;
			std::vector<vec3> cam_exposure_gradient;
			GPUMemory<vec3> cam_exposure_gradient_gpu;

			vec2 cam_focal_length_gradient = vec2(0.0f);
			GPUMemory<vec2> cam_focal_length_gradient_gpu;

			std::vector<AdamOptimizer<vec3>> cam_exposure;
			std::vector<AdamOptimizer<vec3>> cam_pos_offset;
			std::vector<RotationAdamOptimizer> cam_rot_offset;
			AdamOptimizer<vec2> cam_focal_length_offset = AdamOptimizer<vec2>(0.0f);

			GPUMemory<float> extra_dims_gpu; // if the model demands a latent code per training image, we put them in here.
			GPUMemory<float> extra_dims_gradient_gpu;
			std::vector<VarAdamOptimizer> extra_dims_opt;

			std::vector<float> get_extra_dims_cpu(int trainview) const;

			float extrinsic_l2_reg = 1e-4f;
			float extrinsic_learning_rate = 1e-3f;

			float intrinsic_l2_reg = 1e-4f;
			float exposure_l2_reg = 0.0f;

			NerfCounters counters_rgb;

			bool random_bg_color = true;
			bool linear_colors = false;
			ELossType loss_type = ELossType::L2;
			ELossType depth_loss_type = ELossType::L1;
			bool snap_to_pixel_centers = true;
			bool train_envmap = false;

			bool optimize_distortion = false;
			bool optimize_extrinsics = false;
			bool optimize_extra_dims = false;
			bool optimize_focal_length = false;
			bool optimize_exposure = false;
			bool render_error_overlay = false;
			float error_overlay_brightness = 0.125f;
			uint32_t n_steps_between_cam_updates = 16;
			uint32_t n_steps_since_cam_update = 0;

			bool sample_focal_plane_proportional_to_error = false;
			bool sample_image_proportional_to_error = false;
			bool include_sharpness_in_error = false;
			uint32_t n_steps_between_error_map_updates = 128;
			uint32_t n_steps_since_error_map_update = 0;
			uint32_t n_rays_since_error_map_update = 0;

			float near_distance = 0.1f;
			float density_grid_decay = 0.95f;
			default_rng_t density_grid_rng;
			int view = 0;

			float depth_supervision_lambda = 0.f;

			GPUMemory<float> sharpness_grid;

			void set_camera_intrinsics(int frame_idx, float fx, float fy = 0.0f, float cx = -0.5f, float cy = -0.5f, float k1 = 0.0f, float k2 = 0.0f, float p1 = 0.0f, float p2 = 0.0f, float k3 = 0.0f, float k4 = 0.0f, bool is_fisheye = false);
			void set_camera_extrinsics_rolling_shutter(int frame_idx, mat4x3 camera_to_world_start, mat4x3 camera_to_world_end, const vec4& rolling_shutter, bool convert_to_ngp = true);
			void set_camera_extrinsics(int frame_idx, mat4x3 camera_to_world, bool convert_to_ngp = true);
			mat4x3 get_camera_extrinsics(int frame_idx);
			void update_transforms(int first = 0, int last = -1);
			void update_extra_dims();

#ifdef NGP_PYTHON
			void set_image(int frame_idx, pybind11::array_t<float> img, pybind11::array_t<float> depth_img, float depth_scale);
#endif

			void reset_camera_extrinsics();
			void export_camera_extrinsics(const fs::path& path, bool export_extrinsics_in_quat_format = true);
		} training = {};

		GPUMemory<float> density_grid; // NERF_GRIDSIZE()^3 grid of EMA smoothed densities from the network
		GPUMemory<uint8_t> density_grid_bitfield;
		uint8_t* get_density_grid_bitfield_mip(uint32_t mip);
		GPUMemory<float> density_grid_mean;
		uint32_t density_grid_ema_step = 0;

		uint32_t max_cascade = 0;

		ENerfActivation rgb_activation = ENerfActivation::Exponential;
		ENerfActivation density_activation = ENerfActivation::Exponential;

		vec3 light_dir = vec3(0.5f);
		// which training image's latent code should be used for rendering
		int rendering_extra_dims_from_training_view = 0;
		GPUMemory<float> rendering_extra_dims;

		void reset_extra_dims(default_rng_t &rng);
		const float* get_rendering_extra_dims(cudaStream_t stream) const;

		int show_accel = -1;

		float sharpen = 0.f;

		float cone_angle_constant = 1.f/256.f;

		bool visualize_cameras = false;
		bool render_with_lens_distortion = false;
		Lens render_lens = {};

		float render_min_transmittance = 0.01f;
		bool render_gbuffer_hard_edges = false;

		float glow_y_cutoff = 0.f;
		int glow_mode = 0;

		int find_closest_training_view(mat4x3 pose) const;
		void set_rendering_extra_dims_from_training_view(int trainview);
		void set_rendering_extra_dims(const std::vector<float>& vals);
		std::vector<float> get_rendering_extra_dims_cpu() const;
}; 

struct RaysNerfSoa {
#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
	void copy_from_other_async(const RaysNerfSoa& other, cudaStream_t stream) {
		CUDA_CHECK_THROW(cudaMemcpyAsync(rgba, other.rgba, size * sizeof(vec4), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(depth, other.depth, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK_THROW(cudaMemcpyAsync(payload, other.payload, size * sizeof(NerfPayload), cudaMemcpyDeviceToDevice, stream));
	}
#endif

	void set(vec4* rgba, float* depth, NerfPayload* payload, size_t size) {
		this->rgba = rgba;
		this->depth = depth;
		this->payload = payload;
		this->size = size;
	}

	vec4* rgba;
	float* depth;
	NerfPayload* payload;
	size_t size;
};

}
