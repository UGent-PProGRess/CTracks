typedef unsigned int uint;
#include <helper_math.h>
#include "raytracing.h"

extern "C" __global__ void static_projection(
    // large data structures
    cudaTextureObject_t volumeTex,
    float* sinogram,
    // metadata
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
//     float3 vol_rshape = make_float3(width, height, depth);  // voi, voxel scale
//     float3 vol_shape = make_float3(detector_width, detector_width, detector_height);  // standard xyz, voxel_scale 1
    float min_distance, max_distance;

    for (uint row = blockDim.x * blockIdx.x + threadIdx.x; row < detector_rheight; row += stride.x) {
        for (uint subset_index = blockDim.y * blockIdx.y + threadIdx.y; subset_index < num_projections_subset; subset_index += stride.y) {
            for (uint col = blockDim.z * blockIdx.z + threadIdx.z; col < detector_rwidth; col += stride.z) {
                float acc = 0.f;
                uint ray_index = row * num_projections_subset * detector_rwidth + subset_index * detector_rwidth + col;
                float2 pixel = make_float2(col + roi_u_start, row + roi_v_start);  // u,v
                float2 pixel_mm = pixel_in_mm(pixel, detector_width, detector_height, pixel_size, signs);
                for (uint subview = 0; subview < num_subviews; subview ++) {
                    uint subview_index = subset_index * num_subviews + subview;
//                     float3 vol_center = 0.5f * vol_rshape + (0.5f * vol_shape - centers_time[subset_index]);
                    float3 voi_center = voi_centers[0];
                    float3 origin = ray_origin(view_stack, subview_index);
                    float3 direction = ray_direction(pixel_mm, view_stack, subview_index);
                    direction = normalize(direction);
                    int hit = intersect_box(origin, direction, *box_min, *box_max, &min_distance, &max_distance);
                    if(!hit) continue;
//                     if ((row == 200) && (col == 250) && (subset_index == 0)) {
//                         printf("\nmin: %3.1f, max: %3.1f", min_distance, max_distance);
//                     }
                    uint num_steps = int(floorf((max_distance - min_distance) / step_size));
                    float new_step_size = max((max_distance - min_distance - 1 * step_size) / num_steps,
                                              0.5f * step_size); // don't let it become zero or negative

                    for (float step = min_distance + 0.5 * new_step_size; step <= max_distance - 0.5 * new_step_size; step += new_step_size) {
                        float3 pos = origin + step * direction;
                        // pos to voxel units, adjusting for sample motion
                        pos = world_to_voxel(pos, signs, voi_center, voxel_size * voxel_scale);
                        float val = tex3D<float>(volumeTex, pos.x + 0.5f, pos.y + 0.5f, pos.z + 0.5f);
                        acc += val * new_step_size;
                    }
                }
                sinogram[ray_index] += acc / num_subviews / 10.f;  // 1/mm to 1/cm
            }
        }
    }
}