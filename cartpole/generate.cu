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

__host__ void draw_scene(ImageMatrix& cartpole, int i, int j)
{
    cartpole[i][j][0] = 255;
    cartpole[i][j][1] = 255;
    cartpole[i][j][2] = 255;

    // Make the floor
    if (i >= 800 && i < 840)
    {
        cartpole[i][j][0] = 0;
        cartpole[i][j][1] = 0;
        cartpole[i][j][2] = 0;
    }

    // Make the pole
    std::pair<double, double> r = {i - 710, j - 500};
    std::pair<double, double> u = {sin(30), cos(30)};
    int cl = r.first * u.first + r.second * u.second;
    int cw = r.second * u.first - r.first * u.second;
    if (cl > 0 && cl < 500 && cw > -10 && cw < 10)
    {
        cartpole[i][j][0] = 0;
        cartpole[i][j][1] = 0;
        cartpole[i][j][2] = 0;
    }

    // Making the cart
    if (i >= 700 && i < 800)
    {
        if (j >= 400 && j < 600)
        {
            cartpole[i][j][0] = 53;
            cartpole[i][j][1] = 69;
            cartpole[i][j][2] = 149;
        }
    }
}

}
