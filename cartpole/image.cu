#include "image.hu"

#include <iostream>

namespace CudaProj::Cartpole
{

CpuImage::CpuImage(int rows, int cols)
    : mRows(rows), mCols(cols)
{
    mImage = (Npp8u*) malloc(sizeof(Npp8u) * mCols * mRows * 3);
    mPitch = mCols * sizeof(Npp8u) * 3;
}

CpuImage::CpuImage(std::vector<std::vector<std::array<Npp8u, 3>>> image)
{
    CpuImage(image.size(), image[0].size());
    for (int i = 0; i < mRows; i++)
    {
        for (int j = 0; j < mCols; j++)
        {
            mImage[(i * mCols + j) * 3 + 0] = image[i][j][0];
            mImage[(i * mCols + j) * 3 + 1] = image[i][j][1];
            mImage[(i * mCols + j) * 3 + 2] = image[i][j][2];
        }
    }
}

CpuImage::CpuImage(CudaImage image) : mRows(image.mRows), mCols(image.mCols)
{
    mImage = (Npp8u*) malloc(sizeof(Npp8u) * mCols * mRows * 3);
    mPitch = image.mPitch;
    cudaError_t eResult = cudaMemcpy2D(
        mImage, mPitch, image.mImage, image.mPitch,
        mRows * sizeof(Npp8u), mCols, cudaMemcpyDeviceToHost);
    if (eResult != cudaSuccess)
        throw std::runtime_error("Cuda Memcpy from device to host failed." + eResult);

    int pixel1 = mImage[0], pixel2 = mImage[1], pixel3 = mImage[2];
    std::cout << "Data: " << pixel1 << ' ' << pixel2 << ' ' << pixel3 << std::endl;
}

CpuImage::~CpuImage()
{
    free(mImage);
}

CudaImage::CudaImage(int width, int height)
    : mRows(width), mCols(height)
{
    mImage = nppiMalloc_8u_C3(mRows, mCols, &mPitch);
}

CudaImage::CudaImage(CpuImage image) : mRows(image.mRows), mCols(image.mCols)
{
    mImage = nppiMalloc_8u_C3(mRows, mCols, &mPitch);
    cudaError_t eResult = cudaMemcpy2D(
        mImage, mPitch, image.mImage, image.mPitch,
        mRows * sizeof(Npp8u), mCols, cudaMemcpyHostToDevice);
    if (eResult != cudaSuccess)
        throw std::runtime_error("Cuda Memcpy from host to device failed.");
}

CudaImage::~CudaImage()
{
    nppiFree(mImage);
}

Npp8u* CudaImage::image_ptr()
{
    return mImage;
}

std::pair<int, int> CudaImage::image_size()
{
    return {mRows, mCols};
}

std::vector<std::vector<std::array<Npp8u, 3>>> CpuImage::to_matrix()
{
    auto matrix = std::vector<std::vector<std::array<Npp8u, 3>>>(mRows, std::vector<std::array<Npp8u, 3>>(mCols));
    for (int i = 0; i < mRows; i++)
    {
        for (int j = 0; j < mCols; j++)
        {
            matrix[i][j][0] = mImage[(i * mCols + j) * 3 + 0];
            matrix[i][j][1] = mImage[(i * mCols + j) * 3 + 1];
            matrix[i][j][2] = mImage[(i * mCols + j) * 3 + 2];
        }
    }
    return matrix;
}

}
