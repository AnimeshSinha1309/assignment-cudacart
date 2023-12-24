#include <string>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include "image.hu"
#include "gpu.hu"
#include "generate.hu"

int main(int argc, char** argv)
{
    auto device = CudaProj::Cartpole::initialize_gpu();
    auto works = CudaProj::Cartpole::print_cuda_info();
    std::string filename = "output.png";
    CudaProj::Cartpole::CudaImage d_image(100, 100);

    dim3 blocks_per_grid, threads_per_block;
    blocks_per_grid.x = 10;
    blocks_per_grid.y = 10;
    blocks_per_grid.z = 1;
    threads_per_block.x = 10;
    threads_per_block.y = 10;
    threads_per_block.z = 3;

    CudaProj::Cartpole::color_red<<<blocks_per_grid, threads_per_block>>>(
        d_image.image_ptr(), d_image.image_size().first, d_image.image_size().second);

    CudaProj::Cartpole::CpuImage h_image(d_image);
    auto matrix = h_image.to_matrix();
}
