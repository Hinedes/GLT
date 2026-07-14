#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct TokenBuffer {
    std::vector<int32_t> tokens;
    int num_sequences = 0;
    int seq_len = 0;
};

bool load_bin(const std::string& path, TokenBuffer& buf);
