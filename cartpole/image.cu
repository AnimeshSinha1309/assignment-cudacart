#include "image.hu"

namespace CudaProj::Cartpole
{

CpuImage::CpuImage(int rows, int cols)
    : mRows(rows), mCols(cols)
{
    mImage = (Npp8u*) malloc(sizeof(Npp8u) * mCols * mRows * 3);
}

CpuImage::CpuImage(std::vector<std::vector<std::array<Npp8u, 3>>> image)
{
    CpuImage(image.size(), image[0].size());
    for (int i = 0; i < image.size(); i++)
    {
        for (int j = 0; j < image[i].size(); j++)
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
    cudaError_t eResult = cudaMemcpy2D(
        mImage, mPitch, image.mImage, image.mPitch,
        mRows * sizeof(Npp8u), mCols, cudaMemcpyDeviceToHost);
    if (eResult != cudaSuccess)
        throw std::runtime_error("Cuda Memcpy from device to host failed.");
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

void CudaImage::color_red()
{
    for (int i = 0; i < mRows; i++)
    {
        for (int j = 0; j < mCols; j++)
        {
            mImage[(i * mCols + j) * 3 + 0] = 255;
            mImage[(i * mCols + j) * 3 + 1] = 0;
            mImage[(i * mCols + j) * 3 + 2] = 0;
        }
    }
}

}
