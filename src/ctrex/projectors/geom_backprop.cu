// Cone view differentiation
// Input: (Nv, Np, Nu) grad_output
// Output: (Nv, Np, Nu, 16) gradients with respect to each parameter of the view
// This output is reduced outside the kernel to produce (Np, 16) gradients of the view matrices

typedef unsigned int uint;
#include <helper_math.h>
#include "raytracing.h"

__device__ __forceinline__
float3 project_vector(float3 v, float3 unit_direction, float inv_norm) {
    // project v onto the plane perpendicular to unit_direction, scaled by 1/|direction|
    float dotdv = unit_direction.x * v.x + unit_direction.y * v.y + unit_direction.z * v.z;
    return make_float3(
        (v.x - dotdv * unit_direction.x) * inv_norm,
        (v.y - dotdv * unit_direction.y) * inv_norm,
        (v.z - dotdv * unit_direction.z) * inv_norm
        );
}

__device__ void compute_rayparam_derivatives(
    float2 pixel,
    float t,
    float3 direction,
    float3 unit_direction,
    float3 d_contrib[16]
) {
    // compute normalization Jacobian: (I - direction*direction^T)/|dir|
    float inv_norm = rsqrtf(direction.x*direction.x + direction.y*direction.y + direction.z*direction.z);
    float u = pixel.x;
    float v = pixel.y;

    // u_axis (indices 0,4,8)
    d_contrib[ 0] = t * project_vector(make_float3(u,0,0), unit_direction, inv_norm);
    d_contrib[ 4] = t * project_vector(make_float3(0,u,0), unit_direction, inv_norm);
    d_contrib[ 8] = t * project_vector(make_float3(0,0,u), unit_direction, inv_norm);

    // normal axis not used (indices 1, 5, 9)
    d_contrib[ 1] = make_float3(0,0,0);
    d_contrib[ 5] = make_float3(0,0,0);
    d_contrib[ 9] = make_float3(0,0,0);
    d_contrib[15] = make_float3(0,0,0);

    // v_axis (indices 2,6,10)
    d_contrib[ 2] = t * project_vector(make_float3(v,0,0), unit_direction, inv_norm);
    d_contrib[ 6] = t * project_vector(make_float3(0,v,0), unit_direction, inv_norm);
    d_contrib[10] = t * project_vector(make_float3(0,0,v), unit_direction, inv_norm);

    // detector_center (indices 3,7,11)
    d_contrib[ 3] = t * project_vector(make_float3(1,0,0), unit_direction, inv_norm);
    d_contrib[ 7] = t * project_vector(make_float3(0,1,0), unit_direction, inv_norm);
    d_contrib[11] = t * project_vector(make_float3(0,0,1), unit_direction, inv_norm);

    // origin (indices 12,13,14) → direct contribution, no normalization
    d_contrib[12] = direction * 0.f - project_vector(make_float3(1,0,0), unit_direction, inv_norm); // x
    d_contrib[13] = direction * 0.f - project_vector(make_float3(0,1,0), unit_direction, inv_norm); // y
    d_contrib[14] = direction * 0.f - project_vector(make_float3(0,0,1), unit_direction, inv_norm); // z
}



// grad_per_ray has shape [num_projections, detector_rheight, detector_rwidth, 16]
// Each thread writes its own (16,) vector → no atomics needed
extern "C" __global__ void geom_backprop(
    cudaTextureObject_t volumeTex,
    const float* __restrict__ grad_output,  // (Nv, Np, Nu)
    float* grad_view_sino,       // (Nv, Np, Nu, 16)
    // metadata
    float* view_stack,
    float3* centers_time,
    uint* projection_indices,
    uint width,
    uint height,
    uint depth,
    uint num_subviews,
    uint detector_rheight,
    uint num_projections,
    uint detector_rwidth,
    uint detector_height,
    uint detector_width,
    float pixel_size,
    float min_distance,
    float step_size,
    float max_distance,
    float voxel_scale,
    float voxel_size,
    uint roi_u_start,
    uint roi_v_start
    ) {
    uint3 stride = make_uint3(blockDim.x*gridDim.x,blockDim.y*gridDim.y, blockDim.z*gridDim.z);
    float3 signs = make_float3(1.f, -1.f, -1.f);
    float3 vol_rshape = make_float3(width, height, depth);  // roi, voxel scale
    float3 vol_shape = make_float3(detector_width, detector_width, detector_height);  // standard, voxel_scale 1

    for (uint row = blockDim.x * blockIdx.x + threadIdx.x; row < detector_rheight; row += stride.x) {
        for (uint proj = blockDim.y * blockIdx.y + threadIdx.y; proj < num_projections; proj += stride.y) {
            for (uint col = blockDim.z * blockIdx.z + threadIdx.z; col < detector_rwidth; col += stride.z) {
                uint ray_index = row * num_projections * detector_width + proj * detector_width + col;
                float2 pixel = make_float2(col + roi_u_start, row + roi_v_start);  // u,v
                float2 pixel_mm = pixel_in_mm(pixel, detector_width, detector_height, pixel_size, signs);
                float grad_view[16] = {0.0f};
                for (uint subview = 0; subview < num_subviews; subview ++) {
                    uint projection_index = subview * num_projections + proj;
                    float3 vol_center = 0.5f * vol_rshape + 0.5f * vol_shape - centers_time[projection_index];
                    float3 origin = ray_origin(view_stack, projection_index);
                    float3 direction = ray_direction(pixel_mm, view_stack, projection_index);
                    float3 unit_direction = normalize(direction);

                    for (float step = min_distance; step < max_distance; step += step_size) {
                        float3 pos = origin + step * voxel_size * unit_direction;
                        // pos to voxel units
                        pos = world_to_voxel(pos, signs, vol_center, voxel_size * voxel_scale);
                        if (voxel_in_2dbounds(pos, width, height)) {
                            float3 grad_x = tex3Dgrad(volumeTex, pos + 0.5f, 0.5f);

                            // d_contrib[k] = ∂r/∂param_k (3D vector)
                            float3 d_contrib[16];
                            compute_rayparam_derivatives(pixel_mm, step, direction, unit_direction, d_contrib);

                            for (int k = 0; k < 16; k++) {
                                float dotval = grad_x.x * step_size * signs.x * d_contrib[k].x +
                                               grad_x.y * step_size * signs.y * d_contrib[k].y +
                                               grad_x.z * step_size * signs.z * d_contrib[k].z;
                                grad_view[k] += grad_output[ray_index] * dotval;
                            }
                        }
                    }
                }

                // write 16 scalars
                for (uint k = 0; k < 16; k++) {
                    grad_view_sino[ray_index * 16 + k] = grad_view[k] / num_subviews;
                }
            }
        }
    }
}
