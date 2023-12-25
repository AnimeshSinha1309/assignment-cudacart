#include "generate.hu"


namespace CudaProj::Cartpole
{

__global__ void color_background(Npp8u* ptr, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            ptr[(i * cols + j) * 3 + 0] = static_cast<Npp8u>(255);
            ptr[(i * cols + j) * 3 + 1] = static_cast<Npp8u>(0);
            ptr[(i * cols + j) * 3 + 2] = static_cast<Npp8u>(0);
        }
    }
}

}
