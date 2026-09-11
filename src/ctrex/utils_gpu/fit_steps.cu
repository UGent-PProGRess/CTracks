typedef unsigned int uint;
#include <helper_math.h>

extern "C" __global__ void fit_steps(
    float* dynamic_volume,
    float* volume0,
    float* volume1,
    float* time_step,
    uint width,
    uint height,
    uint depth,
    uint duration
    ) {
    uint3 stride = make_uint3(blockDim.x*gridDim.x,blockDim.y*gridDim.y, blockDim.z*gridDim.z);

    if (duration < 2) return;

    for (uint z = blockDim.z * blockIdx.z + threadIdx.z; z < depth; z += stride.z) {
    for (uint y = blockDim.y * blockIdx.y + threadIdx.y; y < height; y += stride.y) {
    for (uint x = blockDim.x * blockIdx.x + threadIdx.x; x < width; x += stride.x) {
        size_t zyx_index = z * (height * width) + y * (width) + x;
        float mu;
        float lin = 0.f;  // sum of linear mu
        float sqr = 0.f;  // sum of square mu

        for (uint t = 0; t < duration; t++) {
            size_t tzyx_index = t * (depth * height * width) + zyx_index;
            mu = dynamic_volume[tzyx_index];
            lin += mu;   sqr += mu * mu;
        }

        float num0 = 0;     float num1 = duration;
        float lin0 = 0.f;   float lin1 = lin;  // sum of linear mu
        float sqr0 = 0.f;   float sqr1 = sqr;  // sum of square mu
        float best_cost = 1e30;
        float cost = 0.f;
        float mu0_new = 0.f;
        float mu1_new = 0.f;
        float t01_new = 0.f;

        for (uint t = 0; t < duration - 1; t++) {
            size_t tzyx_index = t * (depth * height * width) + zyx_index;
            mu = dynamic_volume[tzyx_index];
            num0 += 1;  lin0 += mu;   sqr0 += mu * mu;
            num1 -= 1;  lin1 -= mu;   sqr1 -= mu * mu;
//             cost = sqr0 - lin0 * lin0 / num0 + sqr1 - lin1 * lin1 / num1;  // sum of squared differences
            cost = (sqr0 - lin0 * lin0 / num0) / num0 + (sqr1 - lin1 * lin1 / num1) / num1; // sum of variances

            if (cost < best_cost){
                best_cost = cost;
                mu0_new = lin0 / num0;
                mu1_new = lin1 / num1;
                t01_new = t + 0.5f;
            }
        }
        volume0[zyx_index] = mu0_new;
        volume1[zyx_index] = mu1_new;
        time_step[zyx_index] = t01_new;
    }}}
}