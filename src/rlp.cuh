#pragma once

#include <cstdint>

#if defined(__CUDA_ARCH__)
#define HD_INLINE __device__ __forceinline__
#else
#define HD_INLINE inline
#endif

HD_INLINE void rlp_addr_nonce(uint8_t out[23], const uint8_t address[20]) {
    out[0] = 0xD6;
    out[1] = 0x94;
#if defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (int i = 0; i < 20; ++i) {
        out[2 + i] = address[i];
    }
    out[22] = 0x01;
}

#undef HD_INLINE

