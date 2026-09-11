typedef unsigned int uint;
#include <helper_math.h>
#include "raytracing.h"

extern "C" __global__ void ray_weighting(
    // large data structures
    // no volumeTex
    float* sinogram,
    // metadata, same as static_projection.cu
    float* view_stack,
    float3* voi_centers,
    float3* box_min,
    float3* box_max,
    uint* projection_indices,
//     uint width,
//     uint height,
//     uint depth,
    uint num_subviews,
    uint detector_rheight,
    uint num_projections_subset,
    uint detector_rwidth,
    uint detector_height,
    uint detector_width,
    float pixel_size,
    float step_size,
    float voxel_scale,
    float voxel_size,
    uint roi_u_start,
    uint roi_v_start
    ) {
    uint3 stride = make_uint3(blockDim.x*gridDim.x,blockDim.y*gridDim.y, blockDim.z*gridDim.z);
    float3 signs = make_float3(1.f, -1.f, -1.f);
//     float3 vol_rshape = make_float3(width, height, depth);  // roi
//     float3 vol_shape = make_float3(detector_width, detector_width, detector_height);  // standard
    float min_distance, max_distance;

    for (uint row = blockDim.x * blockIdx.x + threadIdx.x; row < detector_rheight; row += stride.x) {
        for (uint subset_index = blockDim.y * blockIdx.y + threadIdx.y; subset_index < num_projections_subset; subset_index += stride.y) {
            for (uint col = blockDim.z * blockIdx.z + threadIdx.z; col < detector_rwidth; col += stride.z) {
                uint ray_index = row * num_projections_subset * detector_width + subset_index * detector_width + col;
                float2 pixel = make_float2(col + roi_u_start, row + roi_v_start);  // u,v
                float2 pixel_mm = pixel_in_mm(pixel, detector_width, detector_height, pixel_size, signs);
                float length = 0.f;
                uint subview = 0;
                uint subview_index = subset_index * num_subviews + subview;
                float3 origin = ray_origin(view_stack, subview_index);
                float3 direction = ray_direction(pixel_mm, view_stack, subview_index);
                direction = normalize(direction);
                int hit = intersect_box(origin, direction, *box_min, *box_max, &min_distance, &max_distance);
                if(!hit) continue;
                length = max_distance - min_distance;

                if (length > voxel_size) {
                    length = length / 10;
                    sinogram[ray_index] = sinogram[ray_index] / length;
                } else {
                    sinogram[ray_index] = 0;
                }
            }
        }
    }
}