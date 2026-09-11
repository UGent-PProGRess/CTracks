
#ifndef RAYTRACING_H
#define RAYTRACING_H

#include <helper_math.h>

__device__ float3 voxel_to_world(float3 voxel, float3 signs, float3 vol_center, float voxel_size) {
    float3 world_coordinate = signs * voxel_size * (voxel - vol_center);
    return world_coordinate;
}

__device__ float3 world_to_voxel(float3 world_coordinate, float3 signs, float3 vol_center, float voxel_size) {
    float3 voxel_coordinate = world_coordinate * signs / voxel_size + vol_center;
    return voxel_coordinate;
}


__device__ bool pixel_in_bounds(float3 pixel, uint detector_rwidth, uint detector_rheight) {
    float edge = 0;
    if (pixel.x < edge) return false;
    if (pixel.y < edge) return false;
    if (pixel.x > detector_rwidth - edge) return false;
    if (pixel.y > detector_rheight - edge) return false;
    return true;
}

__device__ bool voxel_in_bounds(float3 voxel, uint width, uint height, uint depth) {
    if (voxel.x < 0) return false;
    if (voxel.y < 0) return false;
    if (voxel.z < 0) return false;
    if (voxel.x > width) return false;
    if (voxel.y > height) return false;
    if (voxel.z > depth) return false;
    return true;
}

__device__ bool voxel_in_2dbounds(float3 voxel, uint width, uint height) {
    if (voxel.x < 0) return false;
    if (voxel.y < 0) return false;
    if (voxel.x > width) return false;
    if (voxel.y > height) return false;
    return true;
}

__host__ __device__ bool intersect_box(float3 origin, float3 direction, float3 boxmin, float3 boxmax, float *tnear, float *tfar){
    /*
     * https://github.com/tpn/cuda-samples/blob/master/v9.0/2_Graphics/volumeFiltering/volumeRender_kernel.cu
     * CTLab/PolyProject/polyproject.cu on https://github.ugent.be/UGCT/CTLab/blob/master/PolyProject/polyproject.cu
     * Calculates the intersections between the ray and the box (containing the sample)
     *
     * Ray: origin + direction of ray (take r.d normalized to get correct distance of ray in pixel - tfar-tnear)
     * boxmin: minimum of six corners - e.g. (-1mm,-1mm,-1mm)
     * boxmax: maximum of six corners - e.g. (1mm,1mm,1mm)
     * tnear: origin + direction * tnear is first intersection
     * tfar:  origin + direction * tfar is last intersection
    */

    // compute intersection of ray with all six box planes
    float3 invR = make_float3(1.0f) / direction;
    float3 tbot = invR * (boxmin - origin);
    float3 ttop = invR * (boxmax - origin);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

__device__ float3 project_voxel(
    const float3 voxel,
    const float* views,   // (num_projections, 4, 4) flattened
    int projection_index,
    float3 signs,
    float3 vol_center,
    float voxel_size,
    float pixel_size,
    float detector_width,
    float detector_height,
    uint roi_u_start,
    uint roi_v_start) {
    int mat_offset = projection_index * 16;

    // Extract rotation matrix and translation vector from views
    float3 u_axis = make_float3(views[mat_offset + 0], views[mat_offset + 4], views[mat_offset + 8]);
    float3 v_axis = make_float3(views[mat_offset + 2], views[mat_offset + 6], views[mat_offset +10]);
    float3 detector_center = make_float3(views[mat_offset + 3], views[mat_offset + 7], views[mat_offset + 11]);
    float3 source_pos = make_float3(views[mat_offset + 12], views[mat_offset + 13], views[mat_offset + 14]);

    float3 detector_normal = normalize(cross(u_axis, v_axis));

    // Project voxel to detector
    float3 voxel_world = voxel_to_world(voxel, signs, vol_center, voxel_size);
    float3 source_to_point = voxel_world - source_pos;
    float projected_source_distance = dot(detector_center, detector_normal);
    float projected_point_distance = dot(source_to_point, detector_normal);

    float3 projected_point = source_to_point / projected_point_distance * projected_source_distance - detector_center;

    // Orthogonalize
    float uus = dot(u_axis, u_axis);
    float vvs = dot(v_axis, v_axis);
    float uvs = dot(u_axis, v_axis);

    float3 tu = (vvs * u_axis - uvs * v_axis) / (uus * vvs - uvs * uvs);
    float3 tv = (uus * v_axis - uvs * u_axis) / (uus * vvs - uvs * uvs);

    float u_proj = 0.5f * detector_width + dot(projected_point, tu) / pixel_size - roi_u_start;
    float v_proj = 0.5f * detector_height - dot(projected_point, tv) / pixel_size - roi_v_start;

    float dist_obj = sqrtf(dot(source_to_point, source_to_point));  // norm
    float dist_det = sqrtf(dot(projected_point + detector_center, projected_point + detector_center));  // norm
    float mag = dist_det / dist_obj;

    float3 pixel = make_float3(u_proj, v_proj, mag);
    return pixel;
}

__device__ float3 ray_origin(const float* views, uint projection_index) {
    uint mat_offset = projection_index * 16;
    float3 origin = make_float3(views[mat_offset + 12], views[mat_offset + 13], views[mat_offset + 14]);
    return origin;
}

__device__ float2 pixel_in_mm(float2 pixel, uint detector_width, uint detector_height, float pixel_size, float3 signs){
    return make_float2(signs.x * pixel_size * (pixel.x - 0.5f * detector_width),  // u
                       signs.z * pixel_size * (pixel.y - 0.5f * detector_height));  // v
}

__device__ float3 ray_direction(float2 world_pixel, const float* views, uint projection_index) {
    int mat_offset = projection_index * 16;
    float3 u_axis = make_float3(views[mat_offset + 0], views[mat_offset + 4], views[mat_offset + 8]);
    float3 v_axis = make_float3(views[mat_offset + 2], views[mat_offset + 6], views[mat_offset +10]);
    float3 detector_center = make_float3(views[mat_offset + 3], views[mat_offset + 7], views[mat_offset + 11]);
    float3 direction = world_pixel.x * u_axis + world_pixel.y * v_axis + detector_center;
    return direction;
}

__device__ float3 tex3Dgrad(cudaTextureObject_t volumeTex, float3 pos, float delta) {
    float fx1 = tex3D<float>(volumeTex, pos.x+delta, pos.y, pos.z);
    float fx0 = tex3D<float>(volumeTex, pos.x-delta, pos.y, pos.z);
    float fy1 = tex3D<float>(volumeTex, pos.x, pos.y+delta, pos.z);
    float fy0 = tex3D<float>(volumeTex, pos.x, pos.y-delta, pos.z);
    float fz1 = tex3D<float>(volumeTex, pos.x, pos.y, pos.z+delta);
    float fz0 = tex3D<float>(volumeTex, pos.x, pos.y, pos.z-delta);

    float3 g;
    g.x = (fx1 - fx0) / 2 / delta;
    g.y = (fy1 - fy0) / 2 / delta;
    g.z = (fz1 - fz0) / 2 / delta;
    return g;
}

#endif

