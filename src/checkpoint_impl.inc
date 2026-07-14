// ============================================================================
// Phase 5: Safetensors Checkpoint Save/Load (GRFT Magic Format)
// Operation: Serializes all 108 delta weight matrices (float32->bfloat16)
// into a safetensors file prefixed with 4-byte magic "GRFT" for file-type
// identification. Load validates magic header (backward-compatible with
// legacy files lacking it), then validates shape, dtype, and session_id.
// ============================================================================
static std::string build_safetensors_header(int domain_id, int64_t& total_bf16_bytes) {
    int64_t total_elems = static_cast<int64_t>(g_cfg.total_projections)
                        * g_cfg.slice_dim * g_cfg.h_slice_dim;
    total_bf16_bytes = total_elems * 2;

    std::ostringstream j;
    j << "{"
      << "\"__metadata__\":{"
      << "\"session_id\":\"" << g_cfg.session_id << "\","
      << "\"domain_id\":\"" << domain_id << "\","
      << "\"slice_dim\":\"" << g_cfg.slice_dim << "\","
      << "\"h_slice_dim\":\"" << g_cfg.h_slice_dim << "\","
      << "\"hidden_dim\":\"" << g_cfg.hidden_dim << "\","
      << "\"num_layers\":\"" << g_cfg.num_layers << "\""
      << "},"
      << "\"delta_weights\":{"
      << "\"dtype\":\"BF16\","
      << "\"shape\":[" << g_cfg.total_projections << "," << g_cfg.slice_dim << "," << g_cfg.h_slice_dim << "],"
      << "\"data_offsets\":[0," << total_bf16_bytes << "]"
      << "}"
      << "}";
    return j.str();
}

void save_safetensors(const std::string& path, const GraftWeights& graft) {
    if (fs::exists(path) && fs::is_directory(path)) {
        throw std::runtime_error("save_safetensors: path is a directory, not a file: " + path);
    }
    {
        fs::path parent = fs::path(path).parent_path();
        if (!parent.empty() && !fs::exists(parent)) {
            throw std::runtime_error("save_safetensors: parent directory does not exist: " + parent.string());
        }
    }
    int64_t weight_elems = static_cast<int64_t>(g_cfg.total_projections) * g_cfg.slice_dim * g_cfg.h_slice_dim;

    std::vector<float> host_weights(static_cast<size_t>(weight_elems));
    HIP_CHECK(hipMemcpy(host_weights.data(), graft.d_weight,
                        static_cast<size_t>(weight_elems) * sizeof(float),
                        hipMemcpyDeviceToHost));

    std::vector<uint16_t> host_bf16(static_cast<size_t>(weight_elems));
    for (size_t i = 0; i < static_cast<size_t>(weight_elems); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &host_weights[i], sizeof(bits));
        // Round-to-nearest-even, matching f32_to_bf16_kernel (line 1449)
        uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
        bits += rounding_bias;
        host_bf16[i] = static_cast<uint16_t>(bits >> 16);
    }

    int64_t bf16_bytes;
    std::string header = build_safetensors_header(graft.domain_id, bf16_bytes);
    int64_t header_len = static_cast<int64_t>(header.size());
    int64_t padding = (8 - (header_len % 8)) % 8;
    int64_t header_size = header_len + padding;

    std::ofstream f(path, std::ios::binary);
    uint32_t magic = GRFT_MAGIC;
    f.write(reinterpret_cast<const char*>(&magic), 4);
    uint8_t ver = GRFT_VERSION;
    f.write(reinterpret_cast<const char*>(&ver), 1);
    uint64_t header_size_u64 = static_cast<uint64_t>(header_size);
    f.write(reinterpret_cast<const char*>(&header_size_u64), 8);
    f.write(header.data(), header_len);
    for (int64_t i = 0; i < padding; ++i) f.put(0);
    f.write(reinterpret_cast<const char*>(host_bf16.data()), bf16_bytes);

    size_t mb = static_cast<size_t>(bf16_bytes) / (1024 * 1024);

    std::ifstream verify(path, std::ios::binary | std::ios::ate);
    if (!verify) {
        throw std::runtime_error("save_safetensors: failed to open checkpoint for verification: " + path);
    }
    std::streamsize actual_size = verify.tellg();
    if (actual_size <= 0) {
        throw std::runtime_error("save_safetensors: checkpoint file is zero bytes: " + path);
    }

    std::cout << "  Saved " << mb << " MB (safetensors, bfloat16, GRFT magic)\n";
    std::cout << "  Checkpoint: " << path << " (" << actual_size << " bytes)\n";
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

static void write_gemm_profile_stats_json(const std::string& output_path) {
    if (!g_cfg.profile_graft_gemms) return;

    fs::path stats_path(output_path);
    stats_path.replace_extension(".graft_gemm_stats.json");
    std::ofstream f(stats_path);
    if (!f) {
        std::cerr << "[graft_gemm_stats] failed to write " << stats_path.string() << std::endl;
        return;
    }

    f << "{\n";
    f << "  \"graft_gemm_backend\": \"" << graft_gemm_backend_name(g_cfg.graft_gemm_backend) << "\",\n";
    f << "  \"stats\": {\n";
    const GraftGemmKind kinds[] = {
        GraftGemmKind::ExpandForward,
        GraftGemmKind::ContractForward,
        GraftGemmKind::ExpandWeightGrad,
        GraftGemmKind::ContractWeightGrad,
        GraftGemmKind::ExpandInputGrad,
        GraftGemmKind::ContractInputGrad,
    };
    for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); ++i) {
        auto kind = kinds[i];
        const auto it = g_gemm_profile_stats.find(kind);
        GraftGemmProfileStats stats{};
        if (it != g_gemm_profile_stats.end()) {
            stats = it->second;
        }
        f << "    \"" << graft_gemm_kind_name(kind) << "\": {";
        f << "\"call_count\":" << stats.call_count << ",";
        f << "\"rocblas_calls\":" << stats.rocblas_calls << ",";
        f << "\"hipblaslt_calls\":" << stats.hipblaslt_calls << ",";
        f << "\"aten_fallback_calls\":" << stats.aten_fallback_calls << ",";
        f << "\"fallback_calls\":" << stats.fallback_calls << ",";
        f << "\"total_ms\":" << stats.total_ms << ",";
        f << "\"host_approx_ms\":" << stats.host_approx_ms << ",";
        f << "\"host_approx\":" << (stats.host_approx ? "true" : "false") << ",";
        f << "\"workspace_bytes\":" << stats.workspace_bytes << ",";
        f << "\"m\":" << stats.m << ",";
        f << "\"n\":" << stats.n << ",";
        f << "\"k\":" << stats.k << ",";
        f << "\"lda\":" << stats.lda << ",";
        f << "\"ldb\":" << stats.ldb << ",";
        f << "\"ldc\":" << stats.ldc << ",";
        f << "\"transa\":" << stats.transa << ",";
        f << "\"transb\":" << stats.transb << ",";
        f << "\"dtype_a\":" << stats.dtype_a << ",";
        f << "\"dtype_b\":" << stats.dtype_b << ",";
        f << "\"dtype_c\":" << stats.dtype_c << ",";
        f << "\"compute_type\":" << stats.compute_type << ",";
        f << "\"alpha\":" << stats.alpha << ",";
        f << "\"beta\":" << stats.beta << ",";
        f << "\"backend\":\"" << json_escape(stats.backend) << "\",";
        f << "\"fallback_reason\":\"" << json_escape(stats.fallback_reason) << "\"";
        f << "}";
        if (i + 1 < sizeof(kinds) / sizeof(kinds[0])) f << ",";
        f << "\n";
    }
    f << "  }\n";
    f << "}\n";
}

[[nodiscard]] bool load_safetensors_checkpoint(const std::string& path, int expected_domain_id, VramArena& arena, GraftWeights& graft) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "Error: cannot open checkpoint " << path << std::endl;
        return false;
    }
    std::streamsize file_size = f.tellg();
    f.seekg(0);
    std::vector<char> buf(static_cast<size_t>(file_size));
    if (!f.read(buf.data(), file_size)) {
        std::cerr << "Error: failed to read " << path << std::endl;
        return false;
    }

    // Check for GRFT magic header + version byte (backward-compatible with legacy files)
    char* safetensors_start = buf.data();
    int64_t safetensors_size = static_cast<int64_t>(file_size);
    if (file_size >= 5 && *reinterpret_cast<uint32_t*>(buf.data()) == GRFT_MAGIC) {
        uint8_t ver = static_cast<uint8_t>(buf[4]);
        safetensors_start += 5;
        safetensors_size -= 5;
        std::cout << "  Detected GRFT magic header, version " << static_cast<int>(ver) << "\n";
    } else {
        std::cout << "  Legacy checkpoint (no GRFT magic header)\n";
    }

    safetensors_File sf{};
    char* err = safetensors_file_init(safetensors_start, safetensors_size, &sf);
    if (err) {
        std::cerr << "Error: safetensors parse failed: " << err << std::endl;
        free(sf.tensors);
        free(sf.metadata);
        return false;
    }

    int idx = safetensors_lookup(&sf, "delta_weights");
    if (idx < 0) {
        std::cerr << "Error: 'delta_weights' not found in " << path << std::endl;
        free(sf.tensors);
        free(sf.metadata);
        return false;
    }
    auto& tensor = sf.tensors[idx];

    if (tensor.dtype != SAFETENSORS_BF16) {
        std::cerr << "Error: delta_weights dtype is not BF16\n";
        free(sf.tensors);
        free(sf.metadata);
        return false;
    }
    if (tensor.n_dimensions != 3 ||
        tensor.shape[0] != g_cfg.total_projections ||
        tensor.shape[1] != g_cfg.slice_dim ||
        tensor.shape[2] != g_cfg.h_slice_dim) {
        std::cerr << "Error: delta_weights shape mismatch. "
                  << "Expected [" << g_cfg.total_projections << "," << g_cfg.slice_dim << "," << g_cfg.h_slice_dim << "], "
                  << "got [" << tensor.shape[0] << "," << tensor.shape[1] << "," << tensor.shape[2] << "]\n";
        free(sf.tensors);
        free(sf.metadata);
        return false;
    }

    int file_domain = -1;
    std::string file_session_id;
    for (int i = 0; i < sf.num_metadata; i++) {
        auto& [name, value] = sf.metadata[i];
        std::string name_str(name.ptr, name.len);
        std::string value_str(value.ptr, value.len);
        if (name_str == "domain_id") file_domain = std::stoi(value_str);
        if (name_str == "session_id") file_session_id = value_str;
    }

    if (!g_cfg.session_id.empty() && !file_session_id.empty() && g_cfg.session_id != file_session_id) {
        std::cerr << "Error: session_id mismatch! Checkpoint has '" << file_session_id
                  << "', current training session is '" << g_cfg.session_id << "'.\n"
                  << "  Grafts from different training runs cannot be mixed.\n";
        free(sf.tensors);
        free(sf.metadata);
        return false;
    }
    if (file_domain == -1) {
        std::cerr << "Warning: checkpoint missing domain_id metadata; allowing legacy load.\n";
    } else if (file_domain != expected_domain_id) {
        std::cerr << "Error: checkpoint domain_id mismatch. Checkpoint has " << file_domain
                  << ", current --domain-id is " << expected_domain_id << ".\n"
                  << "Refusing to load graft into wrong Axis ARW slice.\n";
        free(sf.tensors);
        free(sf.metadata);
        return false;
    }

    int64_t weight_elems = static_cast<int64_t>(g_cfg.total_projections) * g_cfg.slice_dim * g_cfg.h_slice_dim;
    uint16_t* src = static_cast<uint16_t*>(tensor.ptr);
    std::vector<float> host_f32(static_cast<size_t>(weight_elems));
    for (int64_t i = 0; i < weight_elems; ++i) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&host_f32[static_cast<size_t>(i)], &bits, sizeof(bits));
    }

    HIP_CHECK(hipMemcpy(graft.d_weight, host_f32.data(),
                        static_cast<size_t>(weight_elems) * sizeof(float),
                        hipMemcpyHostToDevice));
    refresh_graft_weight_bf16(arena, graft);

    size_t mb = static_cast<size_t>(weight_elems) * sizeof(uint16_t) / (1024 * 1024);
    if (file_domain == -1) {
        std::cout << "  Loaded legacy checkpoint (missing domain_id metadata)"
                  << " (" << mb << " MB, safetensors bfloat16)\n";
    } else {
        std::cout << "  Loaded checkpoint domain_id=" << file_domain
                  << " (" << mb << " MB, safetensors bfloat16)\n";
    }
    free(sf.tensors);
    free(sf.metadata);
    return true;
}
