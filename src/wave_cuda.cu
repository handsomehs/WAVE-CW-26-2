// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "wave_cuda.h"

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <array>
#include <cstdlib>
#include <string_view>
#include <utility>

// Free helper macro to check for CUDA errors!
#define CUDA_CHECK(expr) do { \
    cudaError_t res = expr; \
    if (res != cudaSuccess) \
      throw std::runtime_error(std::format(__FILE__ ":{} CUDA error: {}", __LINE__, cudaGetErrorString(res))); \
  } while (0)

namespace {

enum FaceDir : int {
    XP = 0,
    XM = 1,
    YP = 2,
    YM = 3,
    ZP = 4,
    ZM = 5,
    NFACE = 6,
};

enum class MpiTransferMode {
    auto_mode,
    device,
    host,
};

MpiTransferMode mpi_transfer_mode_from_env() {
    const char* env = std::getenv("AWAVE_MPI_MODE");
    if (!env || env[0] == '\0') return MpiTransferMode::auto_mode;
    std::string_view v(env);
    if (v == "host" || v == "HOST" || v == "cpu" || v == "CPU") return MpiTransferMode::host;
    if (v == "cuda" || v == "CUDA" || v == "device" || v == "DEVICE" || v == "gpu" || v == "GPU")
        return MpiTransferMode::device;
    return MpiTransferMode::auto_mode;
}

bool overlap_enabled_from_env() {
    const char* env = std::getenv("AWAVE_CUDA_OVERLAP");
    if (!env || env[0] == '\0') return true;
    std::string_view v(env);
    if (v == "0" || v == "off" || v == "OFF" || v == "false" || v == "FALSE") return false;
    return true;
}

bool tile_enabled_from_env() {
    const char* env = std::getenv("AWAVE_CUDA_TILE");
    if (!env || env[0] == '\0') return false;
    std::string_view v(env);
    if (v == "1" || v == "on" || v == "ON" || v == "true" || v == "TRUE") return true;
    return false;
}

bool mpi_waitsome_enabled_from_env() {
    const char* env = std::getenv("AWAVE_MPI_WAITSOME");
    if (!env || env[0] == '\0') return false;
    std::string_view v(env);
    if (v == "0" || v == "off" || v == "OFF" || v == "false" || v == "FALSE") return false;
    return true;
}

int block_mode_from_env() {
    const char* env = std::getenv("AWAVE_CUDA_BLOCK");
    if (!env || env[0] == '\0') return 6;
    int v = std::atoi(env);
    if (v < 0 || v > 6) return 6;
    return v;
}

bool boundary_split_enabled_from_env() {
    const char* env = std::getenv("AWAVE_CUDA_BOUNDARY_SPLIT");
    if (!env || env[0] == '\0') return true;
    std::string_view v(env);
    if (v == "0" || v == "off" || v == "OFF" || v == "false" || v == "FALSE") return false;
    return true;
}

bool damp_branchless_enabled_from_env() {
    const char* env = std::getenv("AWAVE_CUDA_DAMP_BRANCHLESS");
    if (!env || env[0] == '\0') return false;
    std::string_view v(env);
    if (v == "0" || v == "off" || v == "OFF" || v == "false" || v == "FALSE") return false;
    return true;
}

bool z_padding_enabled_from_env() {
    const char* env = std::getenv("AWAVE_CUDA_ZPAD");
    if (!env || env[0] == '\0') return false;
    std::string_view v(env);
    if (v == "0" || v == "off" || v == "OFF" || v == "false" || v == "FALSE") return false;
    return true;
}

std::size_t padded_size(shape_t const& local_shape, int u_stride_y) {
    auto [nx, ny, nz] = local_shape;
    (void)nz;
    return static_cast<std::size_t>(nx + 2) * static_cast<std::size_t>(ny + 2) * static_cast<std::size_t>(u_stride_y);
}

std::size_t coeff_size(shape_t const& local_shape) {
    auto [nx, ny, nz] = local_shape;
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
}

struct HaloFaceBuffers {
    int neigh = -1;
    std::size_t count = 0;
    double* send_d = nullptr;
    double* recv_d = nullptr;
    double* send_h = nullptr;
    double* recv_h = nullptr;

    bool active() const {
        return neigh >= 0 && count > 0 && send_d && recv_d;
    }
};

struct HaloMpiRequests {
    std::array<MPI_Request, NFACE> recv{};
    std::array<MPI_Request, NFACE> send{};
    int nrecv = 0;
    int nsend = 0;
};

// Trigger the first kernel launch path early so one-time CUDA initialisation
// does not distort the first timed chunk.
__global__ void warmup_kernel(double* __restrict__ u_now) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        u_now[0] = u_now[0];
    }
}

__global__ void pack_face_x(double const* __restrict__ u,
                            double* __restrict__ face,
                            int i_src,
                            int ny, int nz,
                            int u_stride_x, int u_stride_y) {
    int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int count = ny * nz;
    if (t >= count) return;
    int j = t / nz;
    int k = t - j * nz;
    int idx = i_src * u_stride_x + (j + 1) * u_stride_y + (k + 1);
    face[t] = u[idx];
}

__global__ void unpack_face_x(double const* __restrict__ face,
                              double* __restrict__ u,
                              int i_dst,
                              int ny, int nz,
                              int u_stride_x, int u_stride_y) {
    int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int count = ny * nz;
    if (t >= count) return;
    int j = t / nz;
    int k = t - j * nz;
    int idx = i_dst * u_stride_x + (j + 1) * u_stride_y + (k + 1);
    u[idx] = face[t];
}

__global__ void pack_face_y(double const* __restrict__ u,
                            double* __restrict__ face,
                            int j_src,
                            int nx, int nz,
                            int u_stride_x, int u_stride_y) {
    int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int count = nx * nz;
    if (t >= count) return;
    int i = t / nz;
    int k = t - i * nz;
    int idx = (i + 1) * u_stride_x + j_src * u_stride_y + (k + 1);
    face[t] = u[idx];
}

__global__ void unpack_face_y(double const* __restrict__ face,
                              double* __restrict__ u,
                              int j_dst,
                              int nx, int nz,
                              int u_stride_x, int u_stride_y) {
    int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int count = nx * nz;
    if (t >= count) return;
    int i = t / nz;
    int k = t - i * nz;
    int idx = (i + 1) * u_stride_x + j_dst * u_stride_y + (k + 1);
    u[idx] = face[t];
}

__global__ void pack_face_z(double const* __restrict__ u,
                            double* __restrict__ face,
                            int k_src,
                            int nx, int ny,
                            int u_stride_x, int u_stride_y) {
    int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int count = nx * ny;
    if (t >= count) return;
    int i = t / ny;
    int j = t - i * ny;
    int idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + k_src;
    face[t] = u[idx];
}

__global__ void unpack_face_z(double const* __restrict__ face,
                              double* __restrict__ u,
                              int k_dst,
                              int nx, int ny,
                              int u_stride_x, int u_stride_y) {
    int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int count = nx * ny;
    if (t >= count) return;
    int i = t / ny;
    int j = t - i * ny;
    int idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + k_dst;
    u[idx] = face[t];
}

__device__ __forceinline__ double damped_update(double center,
                                                double prev,
                                                double value,
                                                double d,
                                                double dt,
                                                bool branchless) {
    if (branchless) {
        double inv_den = 1.0 / (1.0 + d * dt);
        double num = 1.0 - d * dt;
        return (2.0 * center - num * prev + value) * inv_den;
    }
    if (d == 0.0) {
        return 2.0 * center - prev + value;
    }
    double inv_den = 1.0 / (1.0 + d * dt);
    double num = 1.0 - d * dt;
    value *= inv_den;
    return 2.0 * inv_den * center - num * inv_den * prev + value;
}

// one-kernel update: each thread computes one point,
// branching on the damping field to apply the correct update.
__global__ void step_kernel(double const* __restrict__ u_prev,
                            double const* __restrict__ u_now,
                            double* __restrict__ u_next,
                            double const* __restrict__ cs2,
                            double const* __restrict__ damp,
                            int nx, int ny, int nz,
                            int u_stride_x, int u_stride_y,
                            int coeff_stride_x, int coeff_stride_y,
                            double factor, double dt,
                            bool damp_branchless) {
    int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (i >= nx || j >= ny || k >= nz) return;

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
    int c_idx = i * coeff_stride_x + j * coeff_stride_y + k;

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2[c_idx] * lap;

    double d = damp[c_idx];
    u_next[u_idx] = damped_update(center, u_prev[u_idx], value, d, dt, damp_branchless);
}

// Interior-only update (excludes the outermost active cells in each dimension).
// This region does not depend on halo data from neighbouring MPI ranks.
__global__ void step_kernel_interior(double const* __restrict__ u_prev,
                                     double const* __restrict__ u_now,
                                     double* __restrict__ u_next,
                                     double const* __restrict__ cs2,
                                     double const* __restrict__ damp,
                                     int nx, int ny, int nz,
                                     int u_stride_x, int u_stride_y,
                                     int coeff_stride_x, int coeff_stride_y,
                                     double factor, double dt,
                                     bool damp_branchless) {
    int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
    int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) + 1;
    if (i >= (nx - 1) || j >= (ny - 1) || k >= (nz - 1)) return;

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
    int c_idx = i * coeff_stride_x + j * coeff_stride_y + k;

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2[c_idx] * lap;

    double d = damp[c_idx];
    u_next[u_idx] = damped_update(center, u_prev[u_idx], value, d, dt, damp_branchless);
}

// Interior-only update using shared-memory tiling for u_now values.
// The tile stores a one-cell halo in each dimension to reduce global reads.
__global__ void step_kernel_interior_tiled(double const* __restrict__ u_prev,
                                           double const* __restrict__ u_now,
                                           double* __restrict__ u_next,
                                           double const* __restrict__ cs2,
                                           double const* __restrict__ damp,
                                           int nx, int ny, int nz,
                                           int u_stride_x, int u_stride_y,
                                           int coeff_stride_x, int coeff_stride_y,
                                           double factor, double dt,
                                           bool damp_branchless) {
    extern __shared__ double s_u[];

    int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
    int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z) + 1;

    bool active = (i < (nx - 1)) && (j < (ny - 1)) && (k < (nz - 1));

    int sx = static_cast<int>(blockDim.x) + 2;
    int sy = static_cast<int>(blockDim.y) + 2;

    int li = static_cast<int>(threadIdx.z) + 1;
    int lj = static_cast<int>(threadIdx.y) + 1;
    int lk = static_cast<int>(threadIdx.x) + 1;

    auto sidx = [sy, sx](int ii, int jj, int kk) {
        return (ii * sy + jj) * sx + kk;
    };

    if (active) {
        int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
        s_u[sidx(li, lj, lk)] = u_now[u_idx];

        if (threadIdx.z == 0 || i == 1) {
            s_u[sidx(0, lj, lk)] = u_now[u_idx - u_stride_x];
        }
        if (threadIdx.z == blockDim.z - 1 || i == (nx - 2)) {
            s_u[sidx(static_cast<int>(blockDim.z) + 1, lj, lk)] = u_now[u_idx + u_stride_x];
        }

        if (threadIdx.y == 0 || j == 1) {
            s_u[sidx(li, 0, lk)] = u_now[u_idx - u_stride_y];
        }
        if (threadIdx.y == blockDim.y - 1 || j == (ny - 2)) {
            s_u[sidx(li, static_cast<int>(blockDim.y) + 1, lk)] = u_now[u_idx + u_stride_y];
        }

        if (threadIdx.x == 0 || k == 1) {
            s_u[sidx(li, lj, 0)] = u_now[u_idx - 1];
        }
        if (threadIdx.x == blockDim.x - 1 || k == (nz - 2)) {
            s_u[sidx(li, lj, static_cast<int>(blockDim.x) + 1)] = u_now[u_idx + 1];
        }
    }

    __syncthreads();
    if (!active) return;

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
    int c_idx = i * coeff_stride_x + j * coeff_stride_y + k;

    double center = s_u[sidx(li, lj, lk)];
    double lap = s_u[sidx(li - 1, lj, lk)] + s_u[sidx(li + 1, lj, lk)]
               + s_u[sidx(li, lj - 1, lk)] + s_u[sidx(li, lj + 1, lk)]
               + s_u[sidx(li, lj, lk - 1)] + s_u[sidx(li, lj, lk + 1)]
               - 6.0 * center;
    double value = factor * cs2[c_idx] * lap;

    double d = damp[c_idx];
    u_next[u_idx] = damped_update(center, u_prev[u_idx], value, d, dt, damp_branchless);
}

// Boundary-only update for active cells on the local subdomain boundary.
// This runs after halo exchange has completed.
__global__ void step_kernel_boundary(double const* __restrict__ u_prev,
                                     double const* __restrict__ u_now,
                                     double* __restrict__ u_next,
                                     double const* __restrict__ cs2,
                                     double const* __restrict__ damp,
                                     int nx, int ny, int nz,
                                     int u_stride_x, int u_stride_y,
                                     int coeff_stride_x, int coeff_stride_y,
                                     double factor, double dt,
                                     bool damp_branchless) {
    int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (i >= nx || j >= ny || k >= nz) return;

    bool interior = (i > 0 && i < (nx - 1)) &&
                    (j > 0 && j < (ny - 1)) &&
                    (k > 0 && k < (nz - 1));
    if (interior) return;

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
    int c_idx = i * coeff_stride_x + j * coeff_stride_y + k;

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2[c_idx] * lap;

    double d = damp[c_idx];
    u_next[u_idx] = damped_update(center, u_prev[u_idx], value, d, dt, damp_branchless);
}

__global__ void step_kernel_boundary_x_faces(double const* __restrict__ u_prev,
                                             double const* __restrict__ u_now,
                                             double* __restrict__ u_next,
                                             double const* __restrict__ cs2,
                                             double const* __restrict__ damp,
                                             int nx, int ny, int nz,
                                             int u_stride_x, int u_stride_y,
                                             int coeff_stride_x, int coeff_stride_y,
                                             double factor, double dt,
                                             bool damp_branchless) {
    int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int plane = ny * nz;
    int count = 2 * plane;
    if (t >= count) return;

    int face = t / plane;
    int rem = t - face * plane;
    int j = rem / nz;
    int k = rem - j * nz;
    int i = (face == 0) ? 0 : (nx - 1);

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
    int c_idx = i * coeff_stride_x + j * coeff_stride_y + k;

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2[c_idx] * lap;

    double d = damp[c_idx];
    u_next[u_idx] = damped_update(center, u_prev[u_idx], value, d, dt, damp_branchless);
}

__global__ void step_kernel_boundary_y_faces(double const* __restrict__ u_prev,
                                             double const* __restrict__ u_now,
                                             double* __restrict__ u_next,
                                             double const* __restrict__ cs2,
                                             double const* __restrict__ damp,
                                             int nx, int ny, int nz,
                                             int u_stride_x, int u_stride_y,
                                             int coeff_stride_x, int coeff_stride_y,
                                             double factor, double dt,
                                             bool damp_branchless) {
    int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int slab = (nx - 2) * nz;
    int count = 2 * slab;
    if (t >= count) return;

    int face = t / slab;
    int rem = t - face * slab;
    int i = rem / nz + 1;
    int k = rem - (i - 1) * nz;
    int j = (face == 0) ? 0 : (ny - 1);

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
    int c_idx = i * coeff_stride_x + j * coeff_stride_y + k;

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2[c_idx] * lap;

    double d = damp[c_idx];
    u_next[u_idx] = damped_update(center, u_prev[u_idx], value, d, dt, damp_branchless);
}

__global__ void step_kernel_boundary_z_faces(double const* __restrict__ u_prev,
                                             double const* __restrict__ u_now,
                                             double* __restrict__ u_next,
                                             double const* __restrict__ cs2,
                                             double const* __restrict__ damp,
                                             int nx, int ny, int nz,
                                             int u_stride_x, int u_stride_y,
                                             int coeff_stride_x, int coeff_stride_y,
                                             double factor, double dt,
                                             bool damp_branchless) {
    int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int slab = (nx - 2) * (ny - 2);
    int count = 2 * slab;
    if (t >= count) return;

    int face = t / slab;
    int rem = t - face * slab;
    int i = rem / (ny - 2) + 1;
    int j = rem - (i - 1) * (ny - 2) + 1;
    int k = (face == 0) ? 0 : (nz - 1);

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
    int c_idx = i * coeff_stride_x + j * coeff_stride_y + k;

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2[c_idx] * lap;

    double d = damp[c_idx];
    u_next[u_idx] = damped_update(center, u_prev[u_idx], value, d, dt, damp_branchless);
}

} // namespace

// This struct can hold any data you need to manage running on the device
//
// Allocate with std::make_unique when you create the simulation
// object below, in `from_cpu_sim`.
struct CudaImplementationData {
    int device = 0;
    int local_rank = 0;

    int nx = 0;
    int ny = 0;
    int nz = 0;
    int u_stride_y = 0;
    int u_stride_x = 0;
    int coeff_stride_y = 0;
    int coeff_stride_x = 0;

    std::size_t u_size = 0;
    std::size_t c_size = 0;

    double* d_prev = nullptr;
    double* d_now = nullptr;
    double* d_next = nullptr;
    double* d_cs2 = nullptr;
    double* d_damp = nullptr;

    std::array<HaloFaceBuffers, NFACE> faces{};

    cudaStream_t compute_stream = nullptr;
    cudaStream_t halo_stream = nullptr;
    cudaEvent_t pack_ready = nullptr;
    cudaEvent_t interior_done = nullptr;
    cudaEvent_t boundary_done = nullptr;
    bool use_host_staging = false;
    int last_host_sync_time = -1;

    CudaImplementationData(Params const& params,
                           Decomposition const& decomp,
                           uField const& u,
                           array3d const& cs2,
                           array3d const& damp) {
        (void)params;
        nvtx3::scoped_range r{"initialise"};

        MPI_Comm local_comm = MPI_COMM_NULL;
        MPI_Comm_split_type(decomp.comm, MPI_COMM_TYPE_SHARED, decomp.rank, MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_free(&local_comm);

        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev <= 0) {
            throw std::runtime_error("No CUDA device visible to this MPI rank");
        }
        device = local_rank % ndev;
        CUDA_CHECK(cudaSetDevice(device));

        auto mode = mpi_transfer_mode_from_env();
        use_host_staging = (mode == MpiTransferMode::host);

        auto shape = decomp.local_shape;
        nx = static_cast<int>(shape[0]);
        ny = static_cast<int>(shape[1]);
        nz = static_cast<int>(shape[2]);
        bool z_padding_enabled = z_padding_enabled_from_env();
        u_stride_y = z_padding_enabled ? (((nz + 2) + 15) & ~15) : (nz + 2);
        u_stride_x = (ny + 2) * u_stride_y;
        coeff_stride_y = nz;
        coeff_stride_x = ny * nz;

        u_size = padded_size(shape, u_stride_y);
        c_size = coeff_size(shape);

        CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&halo_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&pack_ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&interior_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&boundary_done, cudaEventDisableTiming));

        CUDA_CHECK(cudaMalloc(&d_prev, u_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_now, u_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_next, u_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cs2, c_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_damp, c_size * sizeof(double)));

        std::size_t host_u_stride_y = static_cast<std::size_t>(nz + 2);
        std::size_t row_bytes = host_u_stride_y * sizeof(double);
        std::size_t row_count = static_cast<std::size_t>(nx + 2) * static_cast<std::size_t>(ny + 2);
        std::size_t device_pitch = static_cast<std::size_t>(u_stride_y) * sizeof(double);

        CUDA_CHECK(cudaMemcpy2DAsync(d_prev, device_pitch, u.prev().data(), row_bytes,
                                     row_bytes, row_count, cudaMemcpyHostToDevice, compute_stream));
        CUDA_CHECK(cudaMemcpy2DAsync(d_now, device_pitch, u.now().data(), row_bytes,
                                     row_bytes, row_count, cudaMemcpyHostToDevice, compute_stream));
        CUDA_CHECK(cudaMemcpy2DAsync(d_next, device_pitch, u.next().data(), row_bytes,
                                     row_bytes, row_count, cudaMemcpyHostToDevice, compute_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_cs2, cs2.data(), c_size * sizeof(double), cudaMemcpyHostToDevice, compute_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_damp, damp.data(), c_size * sizeof(double), cudaMemcpyHostToDevice, compute_stream));

        auto& [px, py, pz] = decomp.mpi_shape;
        auto& [pi, pj, pk] = decomp.mpi_idx;

        auto alloc_face = [this](FaceDir dir, int neigh, std::size_t count) {
            if (neigh < 0) return;
            auto& f = faces[static_cast<int>(dir)];
            f.neigh = neigh;
            f.count = count;
            CUDA_CHECK(cudaMalloc(&f.send_d, count * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&f.recv_d, count * sizeof(double)));
            if (use_host_staging) {
                CUDA_CHECK(cudaMallocHost(&f.send_h, count * sizeof(double)));
                CUDA_CHECK(cudaMallocHost(&f.recv_h, count * sizeof(double)));
            }
        };

        if (pi < px - 1)
            alloc_face(XP, decomp.rank_layout(pi + 1, pj, pk), static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz));
        if (pi > 0)
            alloc_face(XM, decomp.rank_layout(pi - 1, pj, pk), static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz));

        if (pj < py - 1)
            alloc_face(YP, decomp.rank_layout(pi, pj + 1, pk), static_cast<std::size_t>(nx) * static_cast<std::size_t>(nz));
        if (pj > 0)
            alloc_face(YM, decomp.rank_layout(pi, pj - 1, pk), static_cast<std::size_t>(nx) * static_cast<std::size_t>(nz));

        if (pk < pz - 1)
            alloc_face(ZP, decomp.rank_layout(pi, pj, pk + 1), static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
        if (pk > 0)
            alloc_face(ZM, decomp.rank_layout(pi, pj, pk - 1), static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));

        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        warmup_kernel<<<1, 1, 0, compute_stream>>>(d_now);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        out("[rank {}] local_rank {} -> cuda:{} ({}) mpi_mode={} zpad={}",
            decomp.rank, local_rank, device, prop.name, use_host_staging ? "host" : "device",
            z_padding_enabled ? "on" : "off");
    }

    void sync_host_fields(uField& u, bool copy_prev) {
        if (last_host_sync_time == u.time()) return;
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(halo_stream));
        std::size_t host_u_stride_y = static_cast<std::size_t>(nz + 2);
        std::size_t row_bytes = host_u_stride_y * sizeof(double);
        std::size_t row_count = static_cast<std::size_t>(nx + 2) * static_cast<std::size_t>(ny + 2);
        std::size_t device_pitch = static_cast<std::size_t>(u_stride_y) * sizeof(double);
        CUDA_CHECK(cudaMemcpy2D(u.now().data(), row_bytes, d_now, device_pitch,
                                row_bytes, row_count, cudaMemcpyDeviceToHost));
        if (copy_prev) {
            CUDA_CHECK(cudaMemcpy2D(u.prev().data(), row_bytes, d_prev, device_pitch,
                                    row_bytes, row_count, cudaMemcpyDeviceToHost));
        }
        last_host_sync_time = u.time();
    }

    ~CudaImplementationData() {
        for (auto& f : faces) {
            if (f.send_d) cudaFree(f.send_d);
            if (f.recv_d) cudaFree(f.recv_d);
            if (f.send_h) cudaFreeHost(f.send_h);
            if (f.recv_h) cudaFreeHost(f.recv_h);
        }
        if (d_prev) cudaFree(d_prev);
        if (d_now) cudaFree(d_now);
        if (d_next) cudaFree(d_next);
        if (d_cs2) cudaFree(d_cs2);
        if (d_damp) cudaFree(d_damp);
        if (pack_ready) cudaEventDestroy(pack_ready);
        if (interior_done) cudaEventDestroy(interior_done);
        if (boundary_done) cudaEventDestroy(boundary_done);
        if (compute_stream) cudaStreamDestroy(compute_stream);
        if (halo_stream) cudaStreamDestroy(halo_stream);
    }
};

static void launch_pack_face(CudaImplementationData& impl, FaceDir dir) {
    auto& f = impl.faces[static_cast<int>(dir)];
    if (!f.active()) return;
    int threads = 256;
    int blocks = ceildiv(static_cast<int>(f.count), threads);
    switch (dir) {
    case XP:
        pack_face_x<<<blocks, threads, 0, impl.halo_stream>>>(impl.d_now, f.send_d, impl.nx,
                                                          impl.ny, impl.nz,
                                                          impl.u_stride_x, impl.u_stride_y);
        break;
    case XM:
        pack_face_x<<<blocks, threads, 0, impl.halo_stream>>>(impl.d_now, f.send_d, 1,
                                                          impl.ny, impl.nz,
                                                          impl.u_stride_x, impl.u_stride_y);
        break;
    case YP:
        pack_face_y<<<blocks, threads, 0, impl.halo_stream>>>(impl.d_now, f.send_d, impl.ny,
                                                          impl.nx, impl.nz,
                                                          impl.u_stride_x, impl.u_stride_y);
        break;
    case YM:
        pack_face_y<<<blocks, threads, 0, impl.halo_stream>>>(impl.d_now, f.send_d, 1,
                                                          impl.nx, impl.nz,
                                                          impl.u_stride_x, impl.u_stride_y);
        break;
    case ZP:
        pack_face_z<<<blocks, threads, 0, impl.halo_stream>>>(impl.d_now, f.send_d, impl.nz,
                                                          impl.nx, impl.ny,
                                                          impl.u_stride_x, impl.u_stride_y);
        break;
    case ZM:
        pack_face_z<<<blocks, threads, 0, impl.halo_stream>>>(impl.d_now, f.send_d, 1,
                                                          impl.nx, impl.ny,
                                                          impl.u_stride_x, impl.u_stride_y);
        break;
    case NFACE:
        break;
    }
}

static void launch_unpack_face(CudaImplementationData& impl, FaceDir dir) {
    auto& f = impl.faces[static_cast<int>(dir)];
    if (!f.active()) return;
    int threads = 256;
    int blocks = ceildiv(static_cast<int>(f.count), threads);
    switch (dir) {
    case XP:
        unpack_face_x<<<blocks, threads, 0, impl.halo_stream>>>(f.recv_d, impl.d_now, impl.nx + 1,
                                                            impl.ny, impl.nz,
                                                            impl.u_stride_x, impl.u_stride_y);
        break;
    case XM:
        unpack_face_x<<<blocks, threads, 0, impl.halo_stream>>>(f.recv_d, impl.d_now, 0,
                                                            impl.ny, impl.nz,
                                                            impl.u_stride_x, impl.u_stride_y);
        break;
    case YP:
        unpack_face_y<<<blocks, threads, 0, impl.halo_stream>>>(f.recv_d, impl.d_now, impl.ny + 1,
                                                            impl.nx, impl.nz,
                                                            impl.u_stride_x, impl.u_stride_y);
        break;
    case YM:
        unpack_face_y<<<blocks, threads, 0, impl.halo_stream>>>(f.recv_d, impl.d_now, 0,
                                                            impl.nx, impl.nz,
                                                            impl.u_stride_x, impl.u_stride_y);
        break;
    case ZP:
        unpack_face_z<<<blocks, threads, 0, impl.halo_stream>>>(f.recv_d, impl.d_now, impl.nz + 1,
                                                            impl.nx, impl.ny,
                                                            impl.u_stride_x, impl.u_stride_y);
        break;
    case ZM:
        unpack_face_z<<<blocks, threads, 0, impl.halo_stream>>>(f.recv_d, impl.d_now, 0,
                                                            impl.nx, impl.ny,
                                                            impl.u_stride_x, impl.u_stride_y);
        break;
    case NFACE:
        break;
    }
}

static HaloMpiRequests halo_start_exchange(Decomposition const& d,
                                           CudaImplementationData& impl) {
    nvtx3::scoped_range r{"halo_start"};
    std::array<FaceDir, NFACE> dirs = {XP, XM, YP, YM, ZP, ZM};
    HaloMpiRequests reqs{};
    reqs.recv.fill(MPI_REQUEST_NULL);
    reqs.send.fill(MPI_REQUEST_NULL);

    for (auto dir : dirs) {
        launch_pack_face(impl, dir);
    }
    CUDA_CHECK(cudaGetLastError());

    if (impl.use_host_staging) {
        for (auto dir : dirs) {
            auto& f = impl.faces[static_cast<int>(dir)];
            if (!f.active()) continue;
            CUDA_CHECK(cudaMemcpyAsync(f.send_h, f.send_d, f.count * sizeof(double), cudaMemcpyDeviceToHost, impl.halo_stream));
        }
    }

    // Ensure send buffers are ready before MPI uses them.
    CUDA_CHECK(cudaEventRecord(impl.pack_ready, impl.halo_stream));
    CUDA_CHECK(cudaEventSynchronize(impl.pack_ready));

    for (auto dir : dirs) {
        int idx = static_cast<int>(dir);
        auto& f = impl.faces[static_cast<int>(dir)];
        if (!f.active()) continue;
        void* recv_ptr = impl.use_host_staging ? static_cast<void*>(f.recv_h) : static_cast<void*>(f.recv_d);
        MPI_Irecv(recv_ptr, static_cast<int>(f.count), MPI_DOUBLE, f.neigh, 0, d.comm, &reqs.recv[idx]);
        ++reqs.nrecv;
    }

    for (auto dir : dirs) {
        int idx = static_cast<int>(dir);
        auto& f = impl.faces[static_cast<int>(dir)];
        if (!f.active()) continue;
        void* send_ptr = impl.use_host_staging ? static_cast<void*>(f.send_h) : static_cast<void*>(f.send_d);
        MPI_Isend(send_ptr, static_cast<int>(f.count), MPI_DOUBLE, f.neigh, 0, d.comm, &reqs.send[idx]);
        ++reqs.nsend;
    }

    return reqs;
}

static void halo_finish_exchange(CudaImplementationData& impl,
                                 HaloMpiRequests& reqs,
                                 bool waitsome_enabled) {
    nvtx3::scoped_range r{"halo_finish"};
    std::array<FaceDir, NFACE> dirs = {XP, XM, YP, YM, ZP, ZM};

    if (waitsome_enabled && reqs.nrecv > 0) {
        std::array<int, NFACE> done_idx{};
        int pending = reqs.nrecv;
        while (pending > 0) {
            int outcount = 0;
            MPI_Waitsome(NFACE, reqs.recv.data(), &outcount, done_idx.data(), MPI_STATUSES_IGNORE);
            if (outcount == MPI_UNDEFINED) break;
            pending -= outcount;
            for (int i = 0; i < outcount; ++i) {
                int idx = done_idx[static_cast<std::size_t>(i)];
                if (idx < 0 || idx >= NFACE) continue;
                auto dir = static_cast<FaceDir>(idx);
                auto& f = impl.faces[static_cast<std::size_t>(idx)];
                if (!f.active()) continue;
                if (impl.use_host_staging) {
                    CUDA_CHECK(cudaMemcpyAsync(f.recv_d, f.recv_h, f.count * sizeof(double), cudaMemcpyHostToDevice, impl.halo_stream));
                }
                launch_unpack_face(impl, dir);
            }
        }
        if (reqs.nsend > 0) {
            MPI_Waitall(NFACE, reqs.send.data(), MPI_STATUSES_IGNORE);
        }
    } else {
        if (reqs.nrecv > 0) {
            MPI_Waitall(NFACE, reqs.recv.data(), MPI_STATUSES_IGNORE);
        }
        if (reqs.nsend > 0) {
            MPI_Waitall(NFACE, reqs.send.data(), MPI_STATUSES_IGNORE);
        }

        if (impl.use_host_staging) {
            for (auto dir : dirs) {
                auto& f = impl.faces[static_cast<int>(dir)];
                if (!f.active()) continue;
                CUDA_CHECK(cudaMemcpyAsync(f.recv_d, f.recv_h, f.count * sizeof(double), cudaMemcpyHostToDevice, impl.halo_stream));
            }
        }

        for (auto dir : dirs) {
            launch_unpack_face(impl, dir);
        }
    }

    CUDA_CHECK(cudaGetLastError());
}

static void step_cpu(
    Decomposition const& decomp, Params const& params,
    view3d const cs2, view3d const damp,
    view3d const u_prev, view3d const u_now, view3d u_next
) {
    auto d2 = params.dx * params.dx;
    auto dt = params.dt;
    auto factor = dt*dt / d2;
    auto [nx, ny, nz] = decomp.local_shape;
    for (unsigned i = 0; i < nx; ++i) {
        auto ii = i + 1;
        for (unsigned j = 0; j < ny; ++j) {
            auto jj = j + 1;
            for (unsigned k = 0; k < nz; ++k) {
                auto kk = k + 1;
                // Simple approximation of Laplacian
                auto value = factor * cs2(i, j, k) * (
                        u_now(ii - 1, jj, kk) + u_now(ii + 1, jj, kk) +
                        u_now(ii, jj - 1, kk) + u_now(ii, jj + 1, kk) +
                        u_now(ii, jj, kk - 1) + u_now(ii, jj, kk + 1)
                        - 6.0 * u_now(ii, jj, kk)
                );
                // Deal with the damping field
                auto& d = damp(i, j, k);
                if (d == 0.0) {
                    u_next(ii, jj, kk) = 2.0 * u_now(ii, jj, kk) - u_prev(ii, jj, kk) + value;
                } else {
                    auto inv_denominator = 1.0 / (1.0 + d * dt);
                    auto numerator = 1.0 - d * dt;
                    value *= inv_denominator;
                    u_next(ii, jj, kk) = 2.0 * inv_denominator * u_now(ii, jj, kk) -
                                             numerator * inv_denominator * u_prev(ii, jj, kk) + value;
                }
            }
        }
    }
}

static void halo_exchange_host(Decomposition const& d, view3d field) {
    std::array<MPI_Request, 12> reqs;
    MPI_Request* next_req = reqs.data();

    auto& [px, py, pz] = d.mpi_shape;
    auto& [pi, pj, pk] = d.mpi_idx;
    auto& [nx, ny, nz] = d.local_shape;

    if (pi < px - 1) {
        auto neigh = d.rank_layout(pi+1, pj, pk);
        MPI_Irecv(&field(nx+1u, 1u, 1u), 1, d.types->x, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(nx, 1u, 1u), 1, d.types->x, neigh, 0, d.comm, next_req++);
    }
    if (pi > 0) {
        auto neigh = d.rank_layout(pi-1, pj, pk);
        MPI_Irecv(&field(0u, 1u, 1u), 1, d.types->x, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(1u, 1u, 1u), 1, d.types->x, neigh, 0, d.comm, next_req++);
    }

    if (pj < py - 1) {
        auto neigh = d.rank_layout(pi, pj+1, pk);
        MPI_Irecv(&field(1u, ny+1u, 1u), 1, d.types->y, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(1u, ny, 1u), 1, d.types->y, neigh, 0, d.comm, next_req++);
    }
    if (pj > 0) {
        auto neigh = d.rank_layout(pi, pj-1, pk);
        MPI_Irecv(&field(1u, 0u, 1u), 1, d.types->y, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(1u, 1u, 1u), 1, d.types->y, neigh, 0, d.comm, next_req++);
    }

    if (pk < pz - 1) {
        auto neigh = d.rank_layout(pi, pj, pk+1);
        MPI_Irecv(&field(0u, 0u, nz+1u), 1, d.types->z, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(0u, 0u, nz), 1, d.types->z, neigh, 0, d.comm, next_req++);
    }
    if (pk > 0) {
        auto neigh = d.rank_layout(pi, pj, pk-1);
        MPI_Irecv(&field(0u, 0u, 0u), 1, d.types->z, neigh, 0, d.comm, next_req++);
        MPI_Isend(&field(0u, 0u, 1u), 1, d.types->z, neigh, 0, d.comm, next_req++);
    }

    int nreqs = static_cast<int>(next_req - reqs.data());
    if (nreqs > 0) {
        MPI_Waitall(nreqs, reqs.data(), MPI_STATUSES_IGNORE);
    }
}

CudaWaveSimulation::CudaWaveSimulation() = default;
CudaWaveSimulation::CudaWaveSimulation(CudaWaveSimulation&&) noexcept = default;
CudaWaveSimulation& CudaWaveSimulation::operator=(CudaWaveSimulation&&) noexcept = default;
CudaWaveSimulation::~CudaWaveSimulation() = default;

CudaWaveSimulation CudaWaveSimulation::from_cpu_sim(const fs::path& cp, const WaveSimulation& source) {
    CudaWaveSimulation ans;
    auto rank = source.decomp.rank;
    outroot(rank, "Initialising {} simulation as copy of {}...", ans.ID(), source.ID());
    ans.params = source.params;
    ans.decomp = source.decomp;
    ans.u = source.u.clone();
    ans.sos = source.sos.clone();
    ans.cs2 = source.cs2.clone();
    ans.damp = source.damp.clone();

    ans.checkpoint = cp;

    if (source.h5) {
        ans.h5 = H5IO::from_params(cp, ans.params, ans.decomp);
        outroot(rank, "Write initial conditions to {}", ans.checkpoint.c_str());
        ans.h5.put_params(ans.params);
        ans.h5.put_damp(ans.damp);
        ans.h5.put_sos(ans.sos);
        ans.append_u_fields();
    } else {
        outroot(rank, "IO off, skipping");
    }

    int count = 0;
    auto err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        outroot(rank, "No CUDA device found; CUDA simulation will run on CPU (cudaGetDeviceCount={} '{}', count={})",
                static_cast<int>(err), cudaGetErrorString(err), count);
        return ans;
    }

    ans.impl = std::make_unique<CudaImplementationData>(ans.params, ans.decomp, ans.u, ans.cs2, ans.damp);

    return ans;
}

void CudaWaveSimulation::append_u_fields() {
    if (impl) {
        bool need_copyback = static_cast<bool>(h5) || (u.time() == params.nsteps);
        if (need_copyback) {
            impl->sync_host_fields(u, true);
        }
    }
    if (h5) {
        h5.append_u(u);
    }
}

void CudaWaveSimulation::run(int n) {
    nvtx3::scoped_range r{"run"};
    if (!impl) {
        for (int i = 0; i < n; ++i) {
            halo_exchange_host(decomp, u.now());
            step_cpu(decomp, params, cs2, damp, u.prev(), u.now(), u.next());
            u.advance();
        }
        return;
    }

    auto& impl = *this->impl;
    double dt = params.dt;
    double factor = dt * dt / (params.dx * params.dx);
    bool overlap_enabled = overlap_enabled_from_env();
    bool tile_enabled = tile_enabled_from_env();
    bool waitsome_enabled = mpi_waitsome_enabled_from_env();
    bool boundary_split_enabled = boundary_split_enabled_from_env();
    bool damp_branchless_enabled = damp_branchless_enabled_from_env();
    int block_mode = block_mode_from_env();
    bool have_interior = (impl.nx > 2) && (impl.ny > 2) && (impl.nz > 2);

    dim3 block_ijk;
    switch (block_mode) {
    case 1:
        block_ijk = dim3(16, 8, 2);
        break;
    case 2:
        block_ijk = dim3(8, 8, 4);
        break;
    case 3:
        block_ijk = dim3(64, 2, 2);
        break;
    case 4:
        block_ijk = dim3(128, 2, 1);
        break;
    case 5:
        block_ijk = dim3(32, 8, 1);
        break;
    case 6:
        block_ijk = dim3(32, 4, 4);
        break;
    case 0:
    default:
        block_ijk = dim3(32, 4, 4);
        break;
    }
    dim3 grid_full(ceildiv(impl.nz, static_cast<int>(block_ijk.x)),
                   ceildiv(impl.ny, static_cast<int>(block_ijk.y)),
                   ceildiv(impl.nx, static_cast<int>(block_ijk.z)));
    dim3 grid_interior;
    if (have_interior) {
        grid_interior = dim3(ceildiv(impl.nz - 2, static_cast<int>(block_ijk.x)),
                             ceildiv(impl.ny - 2, static_cast<int>(block_ijk.y)),
                             ceildiv(impl.nx - 2, static_cast<int>(block_ijk.z)));
    }
    std::size_t interior_shmem_bytes =
            static_cast<std::size_t>(block_ijk.z + 2) *
            static_cast<std::size_t>(block_ijk.y + 2) *
            static_cast<std::size_t>(block_ijk.x + 2) *
            sizeof(double);

    for (int i = 0; i < n; ++i) {
        HaloMpiRequests reqs = halo_start_exchange(decomp, impl);

        if (have_interior) {
            if (tile_enabled) {
                step_kernel_interior_tiled<<<grid_interior, block_ijk, interior_shmem_bytes, impl.compute_stream>>>(
                        impl.d_prev,
                        impl.d_now,
                        impl.d_next,
                        impl.d_cs2,
                        impl.d_damp,
                        impl.nx,
                        impl.ny,
                        impl.nz,
                        impl.u_stride_x,
                        impl.u_stride_y,
                        impl.coeff_stride_x,
                        impl.coeff_stride_y,
                        factor,
                        dt,
                        damp_branchless_enabled);
            } else {
                step_kernel_interior<<<grid_interior, block_ijk, 0, impl.compute_stream>>>(
                        impl.d_prev,
                        impl.d_now,
                        impl.d_next,
                        impl.d_cs2,
                        impl.d_damp,
                        impl.nx,
                        impl.ny,
                        impl.nz,
                        impl.u_stride_x,
                        impl.u_stride_y,
                        impl.coeff_stride_x,
                        impl.coeff_stride_y,
                        factor,
                        dt,
                        damp_branchless_enabled);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(impl.interior_done, impl.compute_stream));

        if (!overlap_enabled) {
            CUDA_CHECK(cudaEventSynchronize(impl.interior_done));
        }

        halo_finish_exchange(impl, reqs, waitsome_enabled);

        if (have_interior) {
            if (boundary_split_enabled) {
            int threads = 256;
            int x_count = 2 * impl.ny * impl.nz;
            int y_count = 2 * (impl.nx - 2) * impl.nz;
            int z_count = 2 * (impl.nx - 2) * (impl.ny - 2);

            step_kernel_boundary_x_faces<<<ceildiv(x_count, threads), threads, 0, impl.halo_stream>>>(
                impl.d_prev,
                impl.d_now,
                impl.d_next,
                impl.d_cs2,
                impl.d_damp,
                impl.nx,
                impl.ny,
                impl.nz,
                impl.u_stride_x,
                impl.u_stride_y,
                impl.coeff_stride_x,
                impl.coeff_stride_y,
                factor,
                dt,
                damp_branchless_enabled);

            step_kernel_boundary_y_faces<<<ceildiv(y_count, threads), threads, 0, impl.halo_stream>>>(
                impl.d_prev,
                impl.d_now,
                impl.d_next,
                impl.d_cs2,
                impl.d_damp,
                impl.nx,
                impl.ny,
                impl.nz,
                impl.u_stride_x,
                impl.u_stride_y,
                impl.coeff_stride_x,
                impl.coeff_stride_y,
                factor,
                dt,
                damp_branchless_enabled);

            step_kernel_boundary_z_faces<<<ceildiv(z_count, threads), threads, 0, impl.halo_stream>>>(
                impl.d_prev,
                impl.d_now,
                impl.d_next,
                impl.d_cs2,
                impl.d_damp,
                impl.nx,
                impl.ny,
                impl.nz,
                impl.u_stride_x,
                impl.u_stride_y,
                impl.coeff_stride_x,
                impl.coeff_stride_y,
                factor,
                dt,
                damp_branchless_enabled);
            } else {
            step_kernel_boundary<<<grid_full, block_ijk, 0, impl.halo_stream>>>(
                impl.d_prev,
                impl.d_now,
                impl.d_next,
                impl.d_cs2,
                impl.d_damp,
                impl.nx,
                impl.ny,
                impl.nz,
                impl.u_stride_x,
                impl.u_stride_y,
                impl.coeff_stride_x,
                impl.coeff_stride_y,
                factor,
                dt,
                damp_branchless_enabled);
            }
        } else {
                    step_kernel<<<grid_full, block_ijk, 0, impl.halo_stream>>>(
                    impl.d_prev,
                    impl.d_now,
                    impl.d_next,
                    impl.d_cs2,
                    impl.d_damp,
                    impl.nx,
                    impl.ny,
                    impl.nz,
                    impl.u_stride_x,
                    impl.u_stride_y,
                    impl.coeff_stride_x,
                    impl.coeff_stride_y,
                    factor,
                    dt,
                    damp_branchless_enabled);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(impl.boundary_done, impl.halo_stream));

        // Keep all step dependencies on the device side so the CPU can queue
        // the next iteration without blocking on per-step event synchronizes.
        CUDA_CHECK(cudaStreamWaitEvent(impl.compute_stream, impl.boundary_done, 0));
        CUDA_CHECK(cudaStreamWaitEvent(impl.halo_stream, impl.interior_done, 0));

        auto* old_prev = impl.d_prev;
        impl.d_prev = impl.d_now;
        impl.d_now = impl.d_next;
        impl.d_next = old_prev;

        u.advance();
        impl.last_host_sync_time = -1;
    }

    // Ensure all communications and kernels complete before returning.
    CUDA_CHECK(cudaStreamSynchronize(impl.compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(impl.halo_stream));
}
