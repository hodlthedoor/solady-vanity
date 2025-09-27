#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "hex.cuh"
#include "keccak.cuh"
#include "rlp.cuh"

std::string keccak_hex(const uint8_t *data, std::size_t len) {
    std::array<uint8_t, 32> digest{};
    keccak256(data, len, digest.data());
    return bytes_to_hex(digest.data(), digest.size(), false);
}

std::string keccak_hex(const std::string &input) {
    return keccak_hex(reinterpret_cast<const uint8_t *>(input.data()), input.size());
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

int main() {
    // Basic Keccak-256 test vectors to ensure the permutation is correct on host builds.
    const std::string expected_empty = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470";
    const std::string expected_abc = "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45";
    const std::string actual_empty = keccak_hex(nullptr, 0);
    const std::string actual_abc = keccak_hex("abc");
    assert(actual_empty == expected_empty);
    assert(actual_abc == expected_abc);

    const std::string deployer_hex = "0xBA203fFDB6727c59e31D73d66290fFb47728e4Cb";
    const std::string init_hash_hex = "0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f";
    const std::string salt_hex = "0x0000000000000000000000000000000000000000000000000000000009b5beb5";
    const std::string expected_address = "0xd000007fB7b2683688561Ff7Fa6c107D2206A5B8";

    std::array<uint8_t, 20> deployer{};
    std::array<uint8_t, 32> init_hash{};
    std::array<uint8_t, 32> salt{};

    assert(parse_hex_exact(deployer_hex, deployer.data(), deployer.size()));
    assert(parse_hex_exact(init_hash_hex, init_hash.data(), init_hash.size()));
    assert(parse_hex_exact(salt_hex, salt.data(), salt.size()));

    std::array<uint8_t, 85> pre{};
    pre[0] = 0xFF;
    std::memcpy(pre.data() + 1, deployer.data(), deployer.size());
    std::memcpy(pre.data() + 21, salt.data(), salt.size());
    std::memcpy(pre.data() + 53, init_hash.data(), init_hash.size());

    std::array<uint8_t, 32> h1{};
    keccak256(pre.data(), pre.size(), h1.data());

    std::array<uint8_t, 20> proxy{};
    for (int i = 0; i < 20; ++i) {
        proxy[i] = h1[12 + i];
    }

    std::array<uint8_t, 23> rlp_bytes{};
    rlp_addr_nonce(rlp_bytes.data(), proxy.data());

    std::array<uint8_t, 32> h2{};
    keccak256(rlp_bytes.data(), rlp_bytes.size(), h2.data());

    std::array<uint8_t, 20> final_address{};
    for (int i = 0; i < 20; ++i) {
        final_address[i] = h2[12 + i];
    }

    const std::string checksum = checksum_address(final_address.data());
    assert(checksum == expected_address);

    return 0;
}
