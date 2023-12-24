#include "gpu.hu"

#include <iostream>

#include <npp.h>
#include <cuda_runtime.h>

namespace CudaProj::Cartpole
{

inline void check_cuda_error(
    cudaError_t error,
    char const *const func,
    const char *const file,
    int const line
)
{
    if (error)
    {
        std::cerr << "Encountered cuda error" << file << ":" << line
            << " error_code=" << error << "" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int initialize_gpu() {
    int device_count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        std::cerr << "No devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int device_id = 0, compute_mode = -1, major = 0, minor = 0;
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&compute_mode, cudaDevAttrComputeMode, device_id));
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
    if (compute_mode == cudaComputeModeProhibited)
    {
        std::cerr <<  "Error: device is running in <Compute Mode Prohibited>,"
            " no threads can use cudaSetDevice()" << std::endl;
        return -1;
    }
    if (major < 1)
    {
        std::cerr << "GPU device does not support CUDA" << std::endl;
        exit(EXIT_FAILURE);
    }

    CHECK_CUDA_ERROR(cudaSetDevice(device_id));
    return device_id;
}

bool print_cuda_info()
{
  const NppLibraryVersion *library_version = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", library_version->major, library_version->minor,
         library_version->build);

  int driver_version, runtime_version;
  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);

  std::cout << "CUDA Driver  Version: " << driver_version / 1000 << "." << (driver_version % 100) / 10 << std::endl;
  std::cout << "CUDA Runtime Version: " << runtime_version / 1000 << "." << (runtime_version % 100) / 10 << std::endl;

  bool works_on_platform = check_cuda_capabilities(1, 0);
  return works_on_platform;
}

inline bool check_cuda_capabilities(int major_version, int minor_version)
{
    int device_id;
    int major = 0, minor = 0;

    CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));

    if ((major > major_version) || (major == major_version && minor >= minor_version))
    {
        std::cout << "Device " << device_id << ":, with version " << major << "." << minor << " detected" << std::endl;
        return true;
    }
    else
    {
        std::cout << "No GPU device was with version " << major << "." << minor << " detected" << std::endl;
        return false;
    }
}

}
