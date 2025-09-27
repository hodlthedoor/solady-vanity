#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <string>

inline std::string strip_0x(const std::string &input) {
    if (input.size() >= 2 && input[0] == '0' && (input[1] == 'x' || input[1] == 'X')) {
        return input.substr(2);
    }
    return input;
}

inline int hex_char_to_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

inline bool parse_hex_exact(const std::string &hex, uint8_t *out, std::size_t bytes) {
    const std::string clean = strip_0x(hex);
    if (clean.size() != bytes * 2) {
        return false;
    }
    for (std::size_t i = 0; i < bytes; ++i) {
        const int hi = hex_char_to_value(clean[2 * i]);
        const int lo = hex_char_to_value(clean[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

inline std::string bytes_to_hex(const uint8_t *data, std::size_t len, bool uppercase = false) {
    std::string out;
    out.resize(len * 2);
    const char *digits = uppercase ? "0123456789ABCDEF" : "0123456789abcdef";
    for (std::size_t i = 0; i < len; ++i) {
        const uint8_t byte = data[i];
        out[2 * i] = digits[byte >> 4];
        out[2 * i + 1] = digits[byte & 0x0F];
    }
    return out;
}

inline bool parse_prefix(const std::string &hex, std::array<uint8_t, 20> &target,
                         int &cmp_bytes, int &has_odd, uint8_t &last_mask) {
    const std::string clean = strip_0x(hex);
    if (clean.empty() || clean.size() > 40) {
        return false;
    }
    for (auto &b : target) b = 0;
    const std::size_t nibbles = clean.size();
    cmp_bytes = static_cast<int>((nibbles + 1) / 2);
    has_odd = static_cast<int>(nibbles & 1U);
    last_mask = has_odd ? 0xF0 : 0x00;
    std::size_t i = 0;
    while (i + 1 < nibbles) {
        const int hi = hex_char_to_value(clean[i]);
        const int lo = hex_char_to_value(clean[i + 1]);
        if (hi < 0 || lo < 0) return false;
        target[i / 2] = static_cast<uint8_t>((hi << 4) | lo);
        i += 2;
    }
    if (has_odd) {
        const int hi = hex_char_to_value(clean.back());
        if (hi < 0) return false;
        target[cmp_bytes - 1] = static_cast<uint8_t>(hi << 4);
    }
    return true;
}

inline bool starts_with_prefix(const uint8_t address[20], const std::array<uint8_t, 20> &target,
                               int cmp_bytes, int has_odd, uint8_t last_mask) {
    const int full_bytes = cmp_bytes - (has_odd ? 1 : 0);
    for (int i = 0; i < full_bytes; ++i) {
        if (address[i] != target[i]) {
            return false;
        }
    }
    if (has_odd) {
        const int idx = cmp_bytes - 1;
        if ((address[idx] & last_mask) != (target[idx] & last_mask)) {
            return false;
        }
    }
    return true;
}

