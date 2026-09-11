typedef unsigned int uint;
#include <helper_math.h>
#include "raytracing.h"

extern "C" __global__ void voxel_weighting(
    // large data structures
    float* volume,
    // metadata, same as static_backprojection.cu
    float* view_stack,
//     float3* centers_time,
    float3* voi_centers,
    uint* projection_indices,
    uint width,
    uint height,
    uint depth,
    float voxel_scale,
    float voxel_size,
    float pixel_size,
    float step_size,
    uint num_projections_subset,
    uint detector_width,
    uint detector_height,
    uint roi_u_start,
    uint roi_v_start,
    uint detector_rwidth,
    uint detector_rheight
    ) {
    uint3 stride = make_uint3(blockDim.x*gridDim.x, blockDim.y*gridDim.y, blockDim.z*gridDim.z);
    float3 signs = make_float3(1.f, -1.f, -1.f);
    float3 vol_rshape = make_float3(width, height, depth);  // roi
    float3 vol_shape = make_float3(detector_width, detector_width, detector_height);  // standard

    for (uint x = blockDim.x * blockIdx.x + threadIdx.x; x < width; x += stride.x) {
        for (uint y = blockDim.y * blockIdx.y + threadIdx.y; y < height; y += stride.y) {
            for (uint z = blockDim.z * blockIdx.z + threadIdx.z; z < depth; z += stride.z) {
                float3 voxel = make_float3(x, y, z);
                uint voxel_index = z * height * width + y * width + x;
                float3 pixel = make_float3(-1.f, -1.f, -1.f);  // will be overwritten
                float weight = 1.f;
                uint hit_count = 0;
                for (uint subset_index = 0; subset_index < num_projections_subset; subset_index ++) {
                    float3 voi_center = voi_centers[subset_index];
                    pixel = project_voxel(voxel, view_stack, subset_index, signs, voi_center, voxel_size * voxel_scale, pixel_size,
                                          detector_width, detector_height, roi_u_start, roi_v_start);
                    if (pixel_in_bounds(pixel, detector_rwidth, detector_rheight)){
                        hit_count += 1;
                    }
                }
                if (hit_count > 0) {
                    volume[voxel_index] = volume[voxel_index] / hit_count / step_size / voxel_size * 10.;
                } else {
                    volume[voxel_index] = 0.f;
                }
            }
        }
    }
}