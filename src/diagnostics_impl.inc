inline std::string parity_dump_path(const std::string& name) {
    if (g_cfg.parity_dump_dir.empty()) return {};
    std::string pd = g_cfg.parity_dump_dir;
    if (!pd.empty() && pd.back() != '/' && pd.back() != '\\') pd += '/';
    return pd + name;
}

void dump_f32_tensor_raw(const at::Tensor& t, const std::string& path) {
    if (path.empty()) return;
    auto cpu = t.detach().to(torch::kFloat32).contiguous().to(torch::kCPU);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(cpu.data_ptr<float>()),
            static_cast<std::streamsize>(cpu.numel() * sizeof(float)));
}

void dump_f32_buffer_raw(const float* data, size_t numel, const std::string& path) {
    if (path.empty()) return;
    std::vector<float> host(numel);
    HIP_CHECK(hipMemcpy(host.data(), data, numel * sizeof(float), hipMemcpyDeviceToHost));
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(host.data()),
            static_cast<std::streamsize>(numel * sizeof(float)));
}

static inline float bf16_to_f32_host(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static inline uint16_t f32_to_bf16_host(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t rounding_bias = ((bits >> 16) & 1u) + 0x7FFFu;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

void dump_bf16_buffer_raw(const uint16_t* data, size_t numel, const std::string& path) {
    if (path.empty()) return;
    std::vector<uint16_t> host(numel);
    HIP_CHECK(hipMemcpy(host.data(), data, numel * sizeof(uint16_t), hipMemcpyDeviceToHost));
    std::vector<float> out(numel);
    for (size_t i = 0; i < numel; ++i) out[i] = bf16_to_f32_host(host[i]);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(out.data()),
            static_cast<std::streamsize>(out.size() * sizeof(float)));
}
