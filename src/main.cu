#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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
    if (*reinterpret_cast<volatile int *>(found_flag)) return;

    const uint64_t idx = salt_base + blockIdx.x * blockDim.x + threadIdx.x;

    uint8_t pre[85];
    pre[0] = 0xFF;
#pragma unroll
    for (int i = 0; i < 20; ++i) pre[1 + i] = c_deployer[i];
#pragma unroll
    for (int i = 0; i < 32; ++i) pre[21 + i] = 0;
    uint64_t tmp_idx = idx;
#pragma unroll
    for (int b = 0; b < static_cast<int>(sizeof(idx)); ++b) {
        pre[52 - b] = static_cast<uint8_t>(tmp_idx & 0xFF);
        tmp_idx >>= 8;
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) pre[53 + i] = c_init_hash[i];

    uint8_t h1[32];
    keccak256(pre, sizeof(pre), h1);

    uint8_t rlp_bytes[23];
    rlp_addr_nonce(rlp_bytes, h1 + 12);

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

std::string format_duration(double seconds) {
    if (!std::isfinite(seconds)) {
        return "∞";
    }
    if (seconds < 1.0) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << seconds << "s";
        return oss.str();
    }
    const double max_seconds = static_cast<double>(std::numeric_limits<long long>::max());
    if (seconds >= max_seconds) {
        return "∞";
    }
    const long long total_seconds = static_cast<long long>(std::llround(seconds));
    long long remaining = total_seconds;
    const long long days = remaining / 86'400;
    remaining %= 86'400;
    const long long hours = remaining / 3'600;
    remaining %= 3'600;
    const long long minutes = remaining / 60;
    const long long secs = remaining % 60;

    std::ostringstream oss;
    if (days > 0) {
        oss << days << "d ";
    }
    if (hours > 0 || days > 0) {
        oss << hours << "h ";
    }
    if (minutes > 0 || hours > 0 || days > 0) {
        oss << minutes << "m ";
    }
    oss << secs << "s";
    return oss.str();
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

    const std::string clean_prefix = strip_0x(prefix_hex);

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

    const std::size_t prefix_nibbles = clean_prefix.size();
    long double expected_hashes_per_hit = 1.0L;
    for (std::size_t i = 0; i < prefix_nibbles; ++i) {
        expected_hashes_per_hit *= 16.0L;
    }
    const double expected_hashes_per_hit_double = static_cast<double>(expected_hashes_per_hit);

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
    uint64_t total_checked = 0;
    uint64_t total_found = 0;
    auto start_time = std::chrono::steady_clock::now();
    auto last_status = start_time;

    std::size_t last_status_length = 0;
    const auto clear_status_line = [&]() {
        if (last_status_length > 0) {
            std::cout << '\r' << std::string(last_status_length, ' ') << '\r' << std::flush;
            last_status_length = 0;
        }
    };
    const auto print_status_line = [&](const std::string &line) {
        std::cout << '\r' << line;
        if (line.size() < last_status_length) {
            std::cout << std::string(last_status_length - line.size(), ' ');
        }
        std::cout << std::flush;
        last_status_length = line.size();
    };
    const auto format_status_line = [&](double mh_per_second, double elapsed_seconds,
                                        double expected_seconds) {
        std::ostringstream line;
        double display_rate = mh_per_second;
        const char *unit = "MH/s";
        if (display_rate >= 1000.0) {
            display_rate /= 1000.0;
            unit = "GH/s";
        }
        line << "[Status] Found " << total_found << " | " << std::fixed << std::setprecision(2)
             << display_rate << ' ' << unit
             << " | Runtime " << format_duration(elapsed_seconds)
             << " | ETA " << format_duration(expected_seconds);
        return line.str();
    };

    std::ofstream result_file("results.txt", std::ios::out | std::ios::app);
    if (!result_file) {
        std::cerr << "Failed to open results.txt for writing" << std::endl;
        return EXIT_FAILURE;
    }
    result_file.seekp(0, std::ios::end);

    print_status_line(format_status_line(0.0, 0.0, std::numeric_limits<double>::infinity()));

    while (true) {
        grind<<<grid_dim, block_dim>>>(salt_base, d_hit_salt, d_found_flag);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        salt_base += stride;
        total_checked += stride;

        const auto now = std::chrono::steady_clock::now();
        if (now - last_status >= std::chrono::seconds(1)) {
            const double elapsed = std::chrono::duration<double>(now - start_time).count();
            const double hashes_per_second = elapsed > 0.0
                                               ? static_cast<double>(total_checked) / elapsed
                                               : 0.0;
            const double mh_per_second = hashes_per_second / 1'000'000.0;
            const double expected_seconds =
                hashes_per_second > 0.0
                    ? expected_hashes_per_hit_double / hashes_per_second
                    : std::numeric_limits<double>::infinity();
            print_status_line(format_status_line(mh_per_second, elapsed, expected_seconds));
            last_status = now;
        }

        int found = 0;
        CUDA_CHECK(cudaMemcpy(&found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if (found) {
            uint64_t host_hit_salt = 0;
            CUDA_CHECK(cudaMemcpy(&host_hit_salt, d_hit_salt, sizeof(uint64_t), cudaMemcpyDeviceToHost));

            std::array<uint8_t, 32> salt_bytes{};
            uint64_t tmp_host_salt = host_hit_salt;
            for (int b = 0; b < static_cast<int>(sizeof(host_hit_salt)); ++b) {
                salt_bytes[31 - b] = static_cast<uint8_t>(tmp_host_salt & 0xFF);
                tmp_host_salt >>= 8;
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

            clear_status_line();

            std::cout << "[Hit] salt: 0x" << salt_hex << std::endl;
            std::cout << "[Hit] address: " << address_checksum << std::endl;

            result_file << "salt: 0x" << salt_hex << ", address: " << address_checksum
                         << std::endl;
            result_file.flush();

            ++total_found;

            const double elapsed = std::chrono::duration<double>(now - start_time).count();
            const double hashes_per_second = elapsed > 0.0
                                               ? static_cast<double>(total_checked) / elapsed
                                               : 0.0;
            const double mh_per_second = hashes_per_second / 1'000'000.0;
            const double expected_seconds =
                hashes_per_second > 0.0
                    ? expected_hashes_per_hit_double / hashes_per_second
                    : std::numeric_limits<double>::infinity();
            std::cout << "[Hit] elapsed: " << format_duration(elapsed) << std::endl;
            std::cout << "[Hit] expected time at current rate: "
                      << format_duration(expected_seconds) << std::endl;
            print_status_line(format_status_line(mh_per_second, elapsed, expected_seconds));

            CUDA_CHECK(cudaMemset(d_found_flag, 0, sizeof(int)));
        }
    }

    return EXIT_SUCCESS;
}

