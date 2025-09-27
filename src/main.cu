#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "hex.cuh"
#include "keccak.cuh"
#include "rlp.cuh"

__constant__ uint8_t c_deployer[20];
__constant__ uint8_t c_init_hash[32];
__constant__ uint8_t c_target_prefix[20];
__constant__ int c_cmp_bytes;
__constant__ int c_has_odd;
__constant__ uint8_t c_last_mask;

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(_err)          \
                      << " (" << __FILE__ << ":" << __LINE__ << ")"        \
                      << std::endl;                                          \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__global__ void grind(uint64_t salt_base, uint64_t *hit_salt, int *found_flag) {
    if (atomicAdd(found_flag, 0)) return;

    const uint64_t idx = salt_base + blockIdx.x * blockDim.x + threadIdx.x;

    uint8_t salt[32];
#pragma unroll
    for (int b = 0; b < 32; ++b) {
        salt[31 - b] = static_cast<uint8_t>((idx >> (b * 8)) & 0xFF);
    }

    uint8_t pre[85];
    pre[0] = 0xFF;
#pragma unroll
    for (int i = 0; i < 20; ++i) pre[1 + i] = c_deployer[i];
#pragma unroll
    for (int i = 0; i < 32; ++i) pre[21 + i] = salt[i];
#pragma unroll
    for (int i = 0; i < 32; ++i) pre[53 + i] = c_init_hash[i];

    uint8_t h1[32];
    keccak256(pre, sizeof(pre), h1);

    uint8_t proxy[20];
#pragma unroll
    for (int i = 0; i < 20; ++i) proxy[i] = h1[12 + i];

    uint8_t rlp_bytes[23];
    rlp_addr_nonce(rlp_bytes, proxy);

    uint8_t h2[32];
    keccak256(rlp_bytes, sizeof(rlp_bytes), h2);

    const uint8_t *address = h2 + 12;

    const int cmp_bytes = c_cmp_bytes;
    const int has_odd = c_has_odd;
    const uint8_t mask = c_last_mask;

#pragma unroll
    for (int i = 0; i < 20; ++i) {
        if (i < cmp_bytes - (has_odd ? 1 : 0)) {
            if (address[i] != c_target_prefix[i]) return;
        } else if (has_odd && i == cmp_bytes - 1) {
            if ((address[i] & mask) != (c_target_prefix[i] & mask)) return;
            break;
        } else {
            break;
        }
    }

    if (atomicCAS(found_flag, 0, 1) == 0) {
        *hit_salt = idx;
    }
}

std::string keccak_hex(const std::string &input) {
    std::vector<uint8_t> bytes(input.begin(), input.end());
    uint8_t digest[32];
    keccak256(bytes.data(), bytes.size(), digest);
    return bytes_to_hex(digest, 32, false);
}

std::string checksum_address(const uint8_t address[20]) {
    const std::string lower = bytes_to_hex(address, 20, false);
    const std::string hash = keccak_hex(lower);
    std::string out = "0x";
    out.reserve(42);
    for (std::size_t i = 0; i < lower.size(); ++i) {
        char c = lower[i];
        if (c >= 'a' && c <= 'f') {
            const int nibble = hex_char_to_value(hash[i]);
            if (nibble > 7) {
                c = static_cast<char>(c - 'a' + 'A');
            }
        }
        out.push_back(c);
    }
    return out;
}

void usage(const char *prog) {
    std::cerr << "Usage: " << prog << " --deployer <addr> --init-hash <hash> --prefix <hex>" << std::endl;
}

int main(int argc, char **argv) {
    std::string deployer_hex;
    std::string init_hex;
    std::string prefix_hex;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--deployer" && i + 1 < argc) {
            deployer_hex = argv[++i];
        } else if (arg == "--init-hash" && i + 1 < argc) {
            init_hex = argv[++i];
        } else if (arg == "--prefix" && i + 1 < argc) {
            prefix_hex = argv[++i];
        } else {
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (deployer_hex.empty() || init_hex.empty() || prefix_hex.empty()) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::array<uint8_t, 20> deployer{};
    std::array<uint8_t, 32> init_hash{};
    std::array<uint8_t, 20> target{};

    if (!parse_hex_exact(deployer_hex, deployer.data(), deployer.size())) {
        std::cerr << "Invalid deployer address" << std::endl;
        return EXIT_FAILURE;
    }
    if (!parse_hex_exact(init_hex, init_hash.data(), init_hash.size())) {
        std::cerr << "Invalid init hash" << std::endl;
        return EXIT_FAILURE;
    }

    int cmp_bytes = 0;
    int has_odd = 0;
    uint8_t last_mask = 0;
    if (!parse_prefix(prefix_hex, target, cmp_bytes, has_odd, last_mask)) {
        std::cerr << "Invalid prefix" << std::endl;
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_deployer, deployer.data(), deployer.size()));
    CUDA_CHECK(cudaMemcpyToSymbol(c_init_hash, init_hash.data(), init_hash.size()));
    CUDA_CHECK(cudaMemcpyToSymbol(c_target_prefix, target.data(), target.size()));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cmp_bytes, &cmp_bytes, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_has_odd, &has_odd, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_last_mask, &last_mask, sizeof(uint8_t)));

    uint64_t *d_hit_salt = nullptr;
    int *d_found_flag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hit_salt, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_found_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_found_flag, 0, sizeof(int)));

    const dim3 block_dim(256);
    const dim3 grid_dim(4096);
    const uint64_t stride = static_cast<uint64_t>(block_dim.x) * grid_dim.x;

    uint64_t salt_base = 0;
    uint64_t host_hit_salt = 0;

    while (true) {
        grind<<<grid_dim, block_dim>>>(salt_base, d_hit_salt, d_found_flag);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int found = 0;
        CUDA_CHECK(cudaMemcpy(&found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if (found) {
            CUDA_CHECK(cudaMemcpy(&host_hit_salt, d_hit_salt, sizeof(uint64_t), cudaMemcpyDeviceToHost));
            break;
        }
        salt_base += stride;
    }

    CUDA_CHECK(cudaFree(d_hit_salt));
    CUDA_CHECK(cudaFree(d_found_flag));

    std::array<uint8_t, 32> salt_bytes{};
    for (int b = 0; b < 32; ++b) {
        salt_bytes[31 - b] = static_cast<uint8_t>((host_hit_salt >> (8 * b)) & 0xFF);
    }

    uint8_t pre[85];
    pre[0] = 0xFF;
    std::memcpy(pre + 1, deployer.data(), deployer.size());
    std::memcpy(pre + 21, salt_bytes.data(), salt_bytes.size());
    std::memcpy(pre + 53, init_hash.data(), init_hash.size());

    uint8_t h1[32];
    keccak256(pre, sizeof(pre), h1);

    uint8_t proxy[20];
    for (int i = 0; i < 20; ++i) proxy[i] = h1[12 + i];

    uint8_t rlp_bytes[23];
    rlp_addr_nonce(rlp_bytes, proxy);

    uint8_t h2[32];
    keccak256(rlp_bytes, sizeof(rlp_bytes), h2);
    const uint8_t *final_address = h2 + 12;

    if (!starts_with_prefix(final_address, target, cmp_bytes, has_odd, last_mask)) {
        std::cerr << "Internal error: found salt does not satisfy prefix" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string salt_hex = bytes_to_hex(salt_bytes.data(), salt_bytes.size(), true);
    const std::string address_checksum = checksum_address(final_address);

    std::cout << "salt: 0x" << salt_hex << std::endl;
    std::cout << "address: " << address_checksum << std::endl;

    return EXIT_SUCCESS;
}

