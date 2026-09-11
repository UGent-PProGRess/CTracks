#ifndef HELPER_TEXT_H
#define HELPER_TEXT_H

/*! \file helper_text.h
 *  \brief Include helper_text.h to use these functions.
 *  This is an extension to the Texture and Surface object API files of cuda
 *  Contains functions to get and set texture and surface values more easily with vectors written as float2's, float3's etc.
 */

#include <helper_math.h>
// #include <stdio.h>

////////////////////////////////////////////////////////////////////////////////
// Simple texture getters and setters
////////////////////////////////////////////////////////////////////////////////

// 3D texture reads

template<typename T> inline __device__ float getVoxel(texture<T, cudaTextureType3D, cudaReadModeNormalizedFloat> tex, float3 voxel_pos) {
    return tex3D(tex, voxel_pos.x, voxel_pos.y, voxel_pos.z);
}

template<typename T> inline __device__ T getVoxel(texture<T, cudaTextureType3D, cudaReadModeElementType> tex, float3 voxel_pos) {
    return tex3D(tex, voxel_pos.x, voxel_pos.y, voxel_pos.z);
}

template<typename T, enum cudaTextureReadMode readMode> inline __device__ T getVoxel(texture<T, cudaTextureType3D, readMode> tex, float2 pixel_pos, uint buffer_index) {
    return tex3D(tex,pixel_pos.x,pixel_pos.y,buffer_index);
}

template<typename T, enum cudaTextureReadMode readMode> inline __device__ T getVoxel(texture<T, cudaTextureType3D, readMode> tex, float x, float y, float z) {
    return tex3D(tex,x,y,z);
}

// 2D layered texture reads

template<typename T> inline __device__ float getVoxel(texture<T, cudaTextureType2DLayered, cudaReadModeNormalizedFloat> tex, float3 voxel_pos) {
    return tex2DLayered(tex, voxel_pos.x, voxel_pos.y, voxel_pos.z);
}

template<typename T> inline __device__ T getVoxel(texture<T, cudaTextureType2DLayered, cudaReadModeElementType> tex, float3 voxel_pos) {
    return tex2DLayered(tex, voxel_pos.x, voxel_pos.y, voxel_pos.z);
}

template<typename T, enum cudaTextureReadMode readMode> inline __device__ T getPixel(texture<T, cudaTextureType2DLayered, readMode> tex, float2 pixel_pos, uint buffer_index) {
    return tex2DLayered(tex,pixel_pos.x,pixel_pos.y,buffer_index);
}

template<typename T, enum cudaTextureReadMode readMode> inline __device__ T getPixel(texture<T, cudaTextureType2DLayered, readMode> tex, float x, float y, uint buffer_index) {
    return tex3D(tex,x,y,buffer_index);
}

// surface read and write

template<typename T> inline __device__ T getVoxel(surface<void, cudaSurfaceType3D> surf, float3 voxel_pos) {
    return surf3Dread<T>(surf, sizeof(T)*__float2int_rn(voxel_pos.x), __float2int_rn(voxel_pos.y), __float2int_rn(voxel_pos.z));
}

template<typename T> inline __device__ T getVoxel(surface<void, cudaSurfaceType2DLayered> surf, float2 voxel_pos, uint buffer_index) {
    return surf2DLayeredread<T>(surf, sizeof(T)*__float2int_rn(voxel_pos.x), __float2int_rn(voxel_pos.y), buffer_index);
}

template<typename S, typename V> __device__ void setVoxel(V val, S surf, float3 voxel_pos) {
    surf3Dwrite(val, surf, sizeof(V)*voxel_pos.x, voxel_pos.y, voxel_pos.z, cudaBoundaryModeZero);
}

template<typename S, typename V> inline __device__ void setVoxel(V val, S surf, float2 pixel_pos, uint buffer_index) {
    surf2DLayeredwrite(val,surf,sizeof(V)*pixel_pos.x, pixel_pos.y,buffer_index,cudaBoundaryModeZero);
}

template<typename V> __device__ void setVoxel(V val, surface<void, cudaSurfaceType3D> surf, uint x, uint y, uint z) {
    surf3Dwrite(val, surf, sizeof(V)*x, y, z, cudaBoundaryModeZero);
}

template<typename V> inline __device__ void setVoxel(V val, surface<void, cudaSurfaceType2DLayered> surf, uint x, uint y, uint buffer_index) {
    surf2DLayeredwrite(val,surf,sizeof(V)*x, y,buffer_index,cudaBoundaryModeZero);
}


#endif // HELPER_TEXT_H
