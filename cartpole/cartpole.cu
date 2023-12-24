#include <string>

#include <cuda_runtime.h>
#include <npp.h>

#include "image.hu"
#include "gpu.hu"

int main(int argc, char** argv)
{
    auto device = CudaProj::Cartpole::initialize_gpu();
    auto works = CudaProj::Cartpole::print_cuda_info();
    std::string filename = "output.png";
    CudaProj::Cartpole::CudaImage d_image(100, 100);
    // d_image.color_red();
    // CudaProj::Cartpole::CpuImage h_image(d_image);
}
