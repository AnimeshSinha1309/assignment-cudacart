#include <string>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include "image.hu"
#include "gpu.hu"
#include "generate.hu"

using namespace CudaProj::Cartpole;

int main(int argc, char** argv)
{
    auto device = initialize_gpu();
    auto works = print_cuda_info();
    std::string filename = "output.png";

    ImageMatrix cartpole(1000, ImageRow(1000));
    for (int i = 0; i < 1000; i++)
        for (int j = 0; j < 1000; j++)
            draw_scene(cartpole, i, j);

    CpuImage o_image(cartpole);
    CudaImage d_image(o_image);

    dim3 blocks_per_grid {10, 10, 1};
    dim3 threads_per_block{1, 1, 3};
    color_background<<<blocks_per_grid, threads_per_block>>>(
        d_image.image_ptr(), d_image.image_size().first, d_image.image_size().second);

    CpuImage h_image(d_image);
    auto matrix = o_image.to_matrix();
    save_image(matrix);
}
