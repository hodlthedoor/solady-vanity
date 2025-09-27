#pragma once

#include <cstdint>
#include <cstddef>

#if defined(__CUDA_ARCH__)
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

HD uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

HD void keccak_f1600(uint64_t state[25]) {
    static constexpr uint64_t RC[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL,
        0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL,
        0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL,
        0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL,
        0x0000000080000001ULL, 0x8000000080008008ULL
    };

    static constexpr int ROTC[25] = {
         0,  1, 62, 28, 27,
        36, 44,  6, 55, 20,
         3, 10, 43, 25, 39,
        41, 45, 15, 21,  8,
        18,  2, 61, 56, 14
    };

    static constexpr int PILN[25] = {
         0, 10,  7, 11, 17,
        18,  3,  5, 16,  8,
        21, 24,  4, 15, 23,
        19, 13, 12,  2, 20,
        14, 22,  9,  6,  1
    };

    uint64_t bc[5];
    uint64_t temp;

#if defined(__CUDA_ARCH__)
#pragma unroll 24
#endif
    for (int round = 0; round < 24; ++round) {
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int i = 0; i < 5; ++i) {
            bc[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        }
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int i = 0; i < 5; ++i) {
            temp = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
            for (int j = 0; j < 25; j += 5) {
                state[j + i] ^= temp;
            }
        }

        temp = state[1];
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int i = 0; i < 24; ++i) {
            const int pi = PILN[i];
            const uint64_t current = state[pi];
            state[pi] = rotl64(temp, ROTC[i + 1]);
            temp = current;
        }

#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int j = 0; j < 25; j += 5) {
            uint64_t a0 = state[j];
            uint64_t a1 = state[j + 1];
            uint64_t a2 = state[j + 2];
            uint64_t a3 = state[j + 3];
            uint64_t a4 = state[j + 4];
            state[j]     = a0 ^ ((~a1) & a2);
            state[j + 1] = a1 ^ ((~a2) & a3);
            state[j + 2] = a2 ^ ((~a3) & a4);
            state[j + 3] = a3 ^ ((~a4) & a0);
            state[j + 4] = a4 ^ ((~a0) & a1);
        }

        state[0] ^= RC[round];
    }
}

HD void keccak256(const uint8_t *data, std::size_t len, uint8_t out[32]) {
    uint64_t state[25];
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (int i = 0; i < 25; ++i) state[i] = 0;

    constexpr std::size_t rate = 136;

    std::size_t offset = 0;
    while (len >= rate) {
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (std::size_t i = 0; i < rate / 8; ++i) {
            uint64_t lane = 0;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
            for (int b = 0; b < 8; ++b) {
                lane |= static_cast<uint64_t>(data[offset + 8 * i + b]) << (8 * b);
            }
            state[i] ^= lane;
        }
        keccak_f1600(state);
        offset += rate;
        len -= rate;
    }

    uint8_t block[rate];
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (std::size_t i = 0; i < rate; ++i) block[i] = 0;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (std::size_t i = 0; i < len; ++i) {
        block[i] = data[offset + i];
    }
    block[len] ^= 0x01;
    block[rate - 1] ^= 0x80;

#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (std::size_t i = 0; i < rate / 8; ++i) {
        uint64_t lane = 0;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int b = 0; b < 8; ++b) {
            lane |= static_cast<uint64_t>(block[8 * i + b]) << (8 * b);
        }
        state[i] ^= lane;
    }
    keccak_f1600(state);

#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (int i = 0; i < 4; ++i) {
        uint64_t lane = state[i];
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (int b = 0; b < 8; ++b) {
            out[i * 8 + b] = static_cast<uint8_t>((lane >> (8 * b)) & 0xFF);
        }
    }
}

#undef HD

