#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <windows.h>   
#include <iostream>

using namespace std;
using namespace cv;

// gray     (overload)
__global__ void imageAdd(uchar* img1, uchar* img2, uchar* imgres, int length) {
    // 一维数据索引计算（万能计算方法）
    int tid = blockIdx.z * (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z) \
        + blockIdx.y * gridDim.x * (blockDim.x * blockDim.y * blockDim.z) \
        + blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) \
        + threadIdx.z * (blockDim.x * blockDim.y) \
        + threadIdx.y * blockDim.x \
        + threadIdx.x;

    if (tid < length) {
        imgres[tid] = (img1[tid] + img2[tid]) / 2;
    }
}

// rgb      (overload)
__global__ void imageAdd(uchar3* img1, uchar3* img2, uchar3* imgres, int length) {
    int tid = blockIdx.z * (gridDim.x * gridDim.y) * (blockDim.x * blockDim.y * blockDim.z) \
              + blockIdx.y * gridDim.x * (blockDim.x * blockDim.y * blockDim.z) \
              + blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) \
              + threadIdx.z * (blockDim.x * blockDim.y) \
              + threadIdx.y * blockDim.x \
              + threadIdx.x;
    if (tid < length) {
        imgres[tid].x = (img1[tid].x + img2[tid].x) / 2;
        imgres[tid].y = (img1[tid].y + img2[tid].y) / 2;
        imgres[tid].z = (img1[tid].z + img2[tid].z) / 2;
    }
}

// gpu: gray Image Overload
void gpu_GrayImageOverload(Mat &img1_host, Mat &img2_host, int row, int col) {
    int length = row * col;
    //  memory size
    int memSize = length * sizeof(uchar);

    // device memory
    uchar* img1_device;
    uchar* img2_device;
    uchar* imgRes_device;

    //  device memory malloc
    cudaMalloc((void**)&img1_device, memSize);
    cudaMalloc((void**)&img2_device, memSize);
    cudaMalloc((void**)&imgRes_device, memSize);

    // copy host to device
    cudaMemcpy(img1_device, img1_host.data, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(img2_device, img2_host.data, memSize, cudaMemcpyHostToDevice);

    // setting parameters and run the kernel function
    dim3 grid(1 + (length / (32 * 32 + 1)), 1, 1);      // grid
    dim3 block(32, 32, 1);                              // block
    // gray image overload
    imageAdd << <grid, block >> > (img1_device, img2_device, imgRes_device, length);

    // copy device to host
    Mat imgRes_host = Mat::zeros(row, col, CV_8UC1);
    cudaMemcpy(imgRes_host.data, imgRes_device, memSize, cudaMemcpyDeviceToHost);

    // show source and result images
    imshow("img1", img1_host);
    imshow("img2", img2_host);
    imshow("imgres", imgRes_host);

    waitKey(0);

    // free
    cudaFree(img1_device);
    cudaFree(img2_device);
    cudaFree(imgRes_device);
}


// gpu: rgb Image Overload
void gpu_RgbImageOverload(Mat &img1rgb_host, Mat &img2rgb_host, int rgb_row, int rgb_col) {
    int rgbLength = rgb_row * rgb_col;

    int rgbMemSize = rgbLength * sizeof(uchar3);

    uchar3* rgbImg1_device;
    uchar3* rgbImg2_device;
    uchar3* rgbImgRes_device;

    //  device momory malloc
    cudaMalloc((void**)&rgbImg1_device, rgbMemSize);
    cudaMalloc((void**)&rgbImg2_device, rgbMemSize);
    cudaMalloc((void**)&rgbImgRes_device, rgbMemSize);

    cudaMemcpy(rgbImg1_device, img1rgb_host.data, rgbMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(rgbImg2_device, img2rgb_host.data, rgbMemSize, cudaMemcpyHostToDevice);

    dim3 rgbGrid(1 + (rgbLength / (32 * 32 + 1)), 1, 1);
    dim3 rgbBlock(32, 32, 1);
    //  rgb image  overload
    imageAdd << <rgbGrid, rgbBlock >> > (rgbImg1_device, rgbImg2_device, rgbImgRes_device, rgbLength);

    Mat rgbImgRes_host = Mat::zeros(rgb_row, rgb_col, CV_8UC3);
    cudaMemcpy(rgbImgRes_host.data, rgbImgRes_device, rgbMemSize, cudaMemcpyDeviceToHost);

    imshow("rgbImg1", img1rgb_host);
    imshow("rgbImg2", img2rgb_host);
    imshow("rgbImgRes", rgbImgRes_host);
    
    waitKey();

    cudaFree(rgbImg1_device);
    cudaFree(rgbImg2_device);
    cudaFree(rgbImgRes_device);

}

// cpu: gray Image Overload
void cpu_GrayImageOverload(Mat &img1, Mat &img2, int row, int col) {
    
    Mat imgRes = Mat::zeros(Size(col, row), CV_8UC1);
    
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            imgRes.data[i*col + j] = (img1.data[i*col + j] + img2.data[i*col+ j]) / 2;
        }
    }

    imshow("cpu", imgRes);

    waitKey();

}



int main() {
    // source images, gray
    Mat img1_host = imread("img1.jpg", IMREAD_GRAYSCALE);
    Mat img2_host = imread("img2.jpg", IMREAD_GRAYSCALE);
    //  image 1 = image 2    
    int row = img1_host.rows;
    int col = img1_host.cols;

    cpu_GrayImageOverload(img1_host, img2_host, row, col);


    /*
    double start = GetTickCount();
    //  Calculation
    gpu_GrayImageOverload(img1_host, img2_host, row, col);
    double end = GetTickCount();
    cout << "gpu gray image processing time: " << end - start << "\n";
    */

    /*
    // rgb
    Mat rgbimg1_host = imread("img1.jpg");
    Mat rgbimg2_host = imread("img2.jpg");

    int rgb_row = rgbimg1_host.rows;
    int rgb_col = rgbimg1_host.cols;
 
    start = GetTickCount();
    //  Calculation
    gpu_RgbImageOverload(rgbimg1_host, rgbimg2_host, rgb_row, rgb_col);
    end = GetTickCount();
    cout << "gpu rgb image processing time: " << end - start << "\n";
    */





   

    system("pause");
    return 0;
}
