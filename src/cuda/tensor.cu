#include <tensor.cuh>

void tensor_add(const Storage *a, const Storage *b, Storage *c);
__global__ void tensor_add_h(const float *a, const float *b, float *c,
                             std::size_t size);

void tensor_sub(const Storage *a, const Storage *b, Storage *c);
__global__ void tensor_sub_h(const float *a, const float *b, float *c,
                             std::size_t size);

void tensor_mul(const Storage *a, const Storage *b, Storage *c);
__global__ void tensor_mul_h(const float *a, const float *b, float *c,
                             std::size_t size);

void tensor_div(const Storage *a, const Storage *b, Storage *c);
__global__ void tensor_div_h(const float *a, const float *b, float *c,
                             std::size_t size);

void tensor_matmul(const Storage *a, const Storage *b, Storage *c);
__global__ void tensor_matmul_h(const float *a, const float *b, float *c,
                                std::size_t *shape_a, std::size_t *shape_b,
                                std::size_t dims);

void tensor_transpose(const Storage *a, unsigned int dim0, unsigned int dim1,
                      Storage *c);
__global__ void tensor_transpose_h(const float *a, unsigned int dim0,
                                   unsigned int dim1, float *c,
                                   std::size_t *shape_a, std::size_t dims);

void tensor_log_softmax(const Storage *a, unsigned int dim, Storage *c);
__global__ void tensor_log_softmax_h(const float *a, unsigned int dim, float *c,
                                     std::size_t *shape_a, std::size_t dims);

void tensor_mean(const Storage *a, unsigned int dim, Storage *c);
__global__ void tensor_mean_h(const float *a, unsigned int dim, float *c,
                              std::size_t *shape_a, std::size_t dims);

void tensor_pow(const Storage *a, unsigned int e, Storage *c);
__global__ void tensor_pow_h(const float *a, unsigned int e, float *c,
                             std::size_t *shape_a, std::size_t dims);

void tensor_log(const Storage *a, Storage *c);
__global__ void tensor_log_h(const float *a, float *c), std::size_t *shape_a,
    std::size_t dims;

void tensor_exp(const Storage *a, Storage *c);
__global__ void tensor_exp_h(const float *a, float *c, std::size_t *shape_a,
                             std::size_t dims);

void tensor_sigmoid(const Storage *a, Storage *c);
__global__ void tensor_sigmoid_h(const float *a, float *c, std::size_t *shape_a,
                                 std::size_t dims);

void tensor_tanh(const Storage *a, Storage *c);
__global__ void tensor_tanh_h(const float *a, float *c, std::size_t *shape_a,
                              std::size_t dims);

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, Cd.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}