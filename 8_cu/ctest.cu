#include <cstdlib>
#include <cstdio>
#include <cassert>
#define dbg_print(x) print(#x": "); print(x); print("\n");
#define dbg_print_layout(x) print(#x": "); print_layout(x); print("\n");

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cublas_v2.h>
// #include <cublas.h>

using namespace cute;

inline void checkCublasError(cublasStatus_t status, const char* msg = "") {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << msg << " code " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}
template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}
template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    printf("%3d  ", layout(m));
  }
  printf("\n");
}

template <typename T, int br, int bc, int bk, typename TiledMma>
__global__ void cute_gemm(const T* Aptr, const T* Bptr, T* Cptr, int m, int n, int k, TiledMma MmaC) {
  using TA = float;
  using TB = float;
  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<br>{}, Int<bk>{}), make_coord(ix, _));
  Tensor gB = local_tile(B, make_tile(Int<bc>{}, Int<bk>{}), make_coord(iy, _));
  Tensor gC = local_tile(C, make_tile(Int<br>{}, Int<bc>{}), make_coord(ix, iy));
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN) 
  
  ThrMMA thr_mma = MmaC.get_thread_slice(threadIdx.x);
  Tensor tAgA = thr_mma.partition_A(gA); //// (MMA, MMA_M, MMA_K, num_tile_k)
  Tensor tBgB = thr_mma.partition_B(gB);
  Tensor tCgC = thr_mma.partition_C(gC);
  // print(size(tAgA));

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _ ,0));
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));

   if(ix == 0 && iy == 0 && threadIdx.x == 0) {
    print("thr_mma ");
    print(thr_mma);
    print("\n");
    print("A ");
    print(A);
    print("\n");
    print("gA ");
    print(gA);

    print("\n");
  //   print("gB");
  //   print(gB);
  //   print("\n");
  //   print("gC");
  //   print(gC);
  //   print("\n");
    print("tAgA ");
    print(tAgA);
    print("\n");
    print("tBgB");
    print(tBgB);
    print("\n");
    print("tCgC ");
    print(tCgC);
    print("\n");
    print("tArA ");
    print(tArA);
    print("\n");
    print("tBrB");  
    print(tBrB);
    print("\n");
    print("tCrC ");
    print(tCrC);  
  }

  clear(tCrC);
  int num_tiles = size<2>(gA);
  if(ix == 0 && iy == 0 && threadIdx.x == 0) {
  print("num_tile_k");
  print(num_tiles);
  print("\n");
  }

  for(int i = 0; i < num_tiles; ++i) {
    cute::copy(tAgA(_, _, _, i), tArA);
    cute::copy(tBgB(_, _, _, i), tBrB);
    cute::gemm(MmaC, tCrC, tArA, tBrB, tCrC);
    // print("happy!\n");
  }
  cute::copy(tCrC, tCgC);
}

template<class T, int bM, int bN, int bK,
        class TiledMMA>
__global__ void gemm_device(const T* Aptr, const T* Bptr, T* Cptr, 
                            int m, int n, int k,
                            TiledMMA tiled_mma) {
  using namespace cute;
  using TA = float;
  using TB = float;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{})); //(m,k) row-major
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{})); //(n,k) row-major
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{})); //(m,n) row-major

  // Get the appropriate blocks for this thread block
  int ix = blockIdx.x;
  int iy = blockIdx.y;             
  Tensor gA = local_tile(A, make_tile(Int<bM>{}, Int<bK>{}), make_coord(ix, _));  // (b_M,b_K,num_tile_k)
  Tensor gB = local_tile(B, make_tile(Int<bN>{}, Int<bK>{}), make_coord(iy, _));  // (b_N,b_K,num_tile_k)
  Tensor gC = local_tile(C, make_tile(Int<bM>{}, Int<bN>{}), make_coord(ix, iy)); // (b_M,b_N)

  ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

  Tensor tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
  Tensor tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
  Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)
  

  auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_K, MMA_N)
  auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)

  clear(tCrC); 
  int num_tile_k = size<2>(gA);

  #pragma unroll 1
  for(int itile = 0; itile < num_tile_k; ++itile) {
    cute::copy(tAgA(_, _, _, itile), tArA);
    cute::copy(tBgB(_, _, _, itile), tBrB);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
  }

  cute::copy(tCrC, tCgC); 
}

int main() {
  using namespace cute;
  using Element = __half;


  constexpr int M = 4096;
  constexpr int N = 1024;
  constexpr int K = 512;

  // Define a tensor shape with dynamic extents (m, n)
  // Allocate and initialize
  thrust::host_vector<Element> h_A(M*K);
  thrust::host_vector<Element> h_B(K*N);
  thrust::host_vector<Element> h_C(M*N);
  thrust::host_vector<Element> h_C_ref(M*N);

  for (size_t i = 0; i < h_A.size(); ++i) {
    auto rand_value = rand() % 10 - 5;
    h_A[i] = static_cast<Element>(rand_value);
  }
  for (size_t i = 0; i < h_B.size(); ++i) {
    auto rand_value = rand() % 10 - 5;
    h_B[i] = static_cast<Element>(rand_value);
  }
  for (size_t i = 0; i < h_C.size(); ++i) {
    h_C[i] = static_cast<Element>(0.0f);
    h_C_ref[i] = static_cast<Element>(0.0f);
  }

  thrust::device_vector<Element> d_A = h_A;
  thrust::device_vector<Element> d_B = h_B;
  thrust::device_vector<Element> d_C = h_C;
  thrust::device_vector<Element> d_C_ref = h_C_ref;

  const int bM = 128;
  const int bN = 128;
  const int bK = 32;
  static constexpr int kNWarps = 4;
  using MMA_Atom_Arch  = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
  using TiledMma = TiledMMA<
                        MMA_Atom_Arch
                        ,Layout<Shape<Int<kNWarps>, _1, _1>>
                        // ,Layout<Shape<_1, _1, _1>>
                        ,Tile<Int<kNWarps* 16>, _8, _16>
                        >;
  TiledMMA MmaC = TiledMma{};
  print("MmaC\n");
  print_latex(TiledMma{});
  print("MmaC\n");

  dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
  dim3 dimBlock(size(MmaC));
  print("dimBlock");
  print(dimBlock);
  print(size(MmaC));
  print("%n");

  const Element* Aptr = thrust::raw_pointer_cast(d_A.data());
  const Element* Bptr = thrust::raw_pointer_cast(d_B.data());
  Element* Cptr = thrust::raw_pointer_cast(d_C.data());

  cute_gemm<Element, bM, bN, bK, TiledMma><<<dimGrid, dimBlock>>>(Aptr, Bptr, Cptr, M , N, K, MmaC);
  cudaDeviceSynchronize();

   // Initialize cuBLAS
  cudaSetDevice(0);
  cublasHandle_t handle;
  checkCublasError(cublasCreate(&handle), "cuBLAS initialization failed");
  half alpha = half(1.0f);
  half beta = half(0.0f);
  Element* Cptr_ref = thrust::raw_pointer_cast(d_C_ref.data());
  checkCublasError(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    Bptr, K,
                    Aptr, K,
                    &beta,
                    Cptr_ref, N), "cuBLAS SGEMM failed");

  h_C = d_C;
  h_C_ref = d_C_ref;
  for (int i = 0; i < M*N; i++) {
    if (std::abs(__half2float(h_C[i]) - __half2float(h_C_ref[i])) > 0.01) {
      std::cerr << "Error. h_C[" << i << "]: " << __half2float(h_C[i]) << ",   h_C_ref[" << i << "]: " << __half2float(h_C_ref[i]) << std::endl;
      return -1;
    }
  }
  printf("Success!\n");
  cudaDeviceSynchronize();


  return 0;
}