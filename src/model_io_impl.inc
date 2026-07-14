// ============================================================================
// Phase 2: Base Model Weight Loading (Safetensors -> ATen bf16 CUDA tensors)
// Operation: Scans safetensors files (single file or directory), parses
// headers, loads all SmolLM3-3B weights (embed_tokens, 36x Q/K/V/O/layernorm/
// gate/up/down, final_norm, lm_head) into bf16 CUDA tensors via st_to_aten().
// Validates dimensionality and shape constraints, handles weight tying
// (lm_head = embed_tokens if separate weight not found).
// ============================================================================

// --------------------------------------------------------------------------
// st_to_aten ” Safetensors descriptor -> ATen CUDA tensor
// Operation: Maps the safetensors dtype (BF16/F32/F16) to the equivalent
// torch ScalarType, clones the CPU data from the file buffer, then moves
// to CUDA device. Returns empty tensor on unsupported dtype.
// --------------------------------------------------------------------------
at::Tensor st_to_aten(safetensors_TensorDescriptor& td) {
    std::vector<int64_t> shape(td.n_dimensions);
    for (int d = 0; d < td.n_dimensions; d++) shape[d] = td.shape[d];

    at::ScalarType stype;
    switch (td.dtype) {
        case SAFETENSORS_BF16: stype = torch::kBFloat16; break;
        case SAFETENSORS_F32:  stype = torch::kFloat32;  break;
        case SAFETENSORS_F16:  stype = torch::kHalf;     break;
        default:
            std::cerr << "  Unsupported safetensors dtype " << td.dtype << "\n";
            return at::Tensor();
    }

    auto cpu = torch::TensorOptions().dtype(stype).device(torch::kCPU);
    auto t = torch::from_blob(td.ptr, shape, cpu).clone();
    return t.to(torch::kCUDA);
}

// --------------------------------------------------------------------------
// load_named_tensor ” Find and load a named tensor from safetensors files
// Operation: Iterates over all loaded safetensors files, looks up the
// tensor by name via safetensors_lookup(), converts and returns it as
// an ATen CUDA tensor. Returns empty tensor if not found (caller handles).
// --------------------------------------------------------------------------
at::Tensor load_named_tensor(std::vector<std::pair<std::vector<uint8_t>, safetensors_File>>& files,
                             const std::string& name) {
    for (size_t fi = 0; fi < files.size(); fi++) {
        safetensors_File* sf = &files[fi].second;
        int idx = safetensors_lookup(sf, name.c_str());
        if (idx >= 0) {
            auto t = st_to_aten(sf->tensors[idx]);
            if (t.defined()) {
                std::cout << "  Loaded " << name << " [" << t.sizes() << "]\n";
                return t;
            }
        }
    }
    return at::Tensor();  // not found
}

[[nodiscard]] bool load_base_model_safetensors(const std::string& safetensors_path) {
    std::cout << "[Component 1] Loading SmolLM3-3B weights (bfloat16) via safetensors.h\n";

    // Collect .safetensors files (single file or directory)
    std::vector<fs::path> st_files;
    if (fs::is_regular_file(safetensors_path)) {
        st_files.push_back(safetensors_path);
    } else if (fs::is_directory(safetensors_path)) {
        for (auto& entry : fs::directory_iterator(safetensors_path)) {
            if (entry.path().extension() == ".safetensors")
                st_files.push_back(entry.path());
        }
        std::sort(st_files.begin(), st_files.end());
    } else {
        std::cerr << "Error: safetensors path '" << safetensors_path << "' is not a file or directory\n";
        return false;
    }

    if (st_files.empty()) {
        std::cerr << "Error: no .safetensors files found at '" << safetensors_path << "'\n";
        return false;
    }

    // Load all files into memory and parse headers
    std::vector<std::pair<std::vector<uint8_t>, safetensors_File>> files;
    for (auto& p : st_files) {
        std::ifstream f(p.string(), std::ios::binary | std::ios::ate);
        if (!f) { std::cerr << "Error: cannot open " << p << "\n"; return false; }
        size_t sz = f.tellg();
        f.seekg(0);
        std::vector<uint8_t> buf(sz);
        f.read(reinterpret_cast<char*>(buf.data()), sz);

        safetensors_File sf;
        char* err = safetensors_file_init(buf.data(), sz, &sf);
        if (err) {
            std::cerr << "Error parsing " << p << ": " << err << "\n";
            return false;
        }
        std::cout << "  " << p.filename().string() << ": " << sf.num_tensors << " tensors\n";
        files.emplace_back(std::move(buf), sf);
    }

    g_model_weights.layers.resize(g_cfg.num_layers);

    // Embedding
    g_model_weights.embed_tokens_weight = load_named_tensor(files, "model.embed_tokens.weight");

    // Per-layer weights
    for (int l = 0; l < g_cfg.num_layers; ++l) {
        std::string p = "model.layers." + std::to_string(l) + ".";
        auto& w = g_model_weights.layers[l];
        w.input_layernorm_weight          = load_named_tensor(files, p + "input_layernorm.weight");
        w.q_proj_weight                   = load_named_tensor(files, p + "self_attn.q_proj.weight");
        w.k_proj_weight                   = load_named_tensor(files, p + "self_attn.k_proj.weight");
        w.v_proj_weight                   = load_named_tensor(files, p + "self_attn.v_proj.weight");
        w.o_proj_weight                   = load_named_tensor(files, p + "self_attn.o_proj.weight");
        w.post_attention_layernorm_weight = load_named_tensor(files, p + "post_attention_layernorm.weight");
        w.gate_proj_weight                = load_named_tensor(files, p + "mlp.gate_proj.weight");
        w.up_proj_weight                  = load_named_tensor(files, p + "mlp.up_proj.weight");
        w.down_proj_weight                = load_named_tensor(files, p + "mlp.down_proj.weight");
    }

    // Final norm and LM head (may be tied)
    g_model_weights.final_norm_weight = load_named_tensor(files, "model.norm.weight");
    g_model_weights.lm_head_weight    = load_named_tensor(files, "lm_head.weight");
    if (!g_model_weights.lm_head_weight.defined()) {
        std::cout << "  lm_head.weight not found ” tying with embed_tokens.weight\n";
        g_model_weights.lm_head_weight = g_model_weights.embed_tokens_weight;
    }
    // Ensure lm_head_weight has shape [vocab_size, hidden_dim] for at::linear
    if (g_model_weights.lm_head_weight.defined()) {
        auto lw = g_model_weights.lm_head_weight;
        if (lw.dim() == 2 && lw.size(0) == g_cfg.vocab_size && lw.size(1) == g_cfg.hidden_dim) {
            // Already in correct orientation ” nothing to do
        } else if (lw.dim() == 2 && lw.size(0) == g_cfg.hidden_dim && lw.size(1) == g_cfg.vocab_size) {
            std::cout << "  Transposing lm_head_weight from [" << g_cfg.hidden_dim << ", " << g_cfg.vocab_size
                      << "] to [" << g_cfg.vocab_size << ", " << g_cfg.hidden_dim << "]\n";
            g_model_weights.lm_head_weight = lw.t().contiguous();
        } else if (lw.dim() == 2 && g_cfg.vocab_size == g_cfg.hidden_dim && lw.size(0) == g_cfg.hidden_dim && lw.size(1) == g_cfg.hidden_dim) {
            std::cerr << "  Warning: lm_head_weight is square (" << g_cfg.hidden_dim << "x" << g_cfg.hidden_dim
                      << "), assuming [g_cfg.vocab_size, g_cfg.hidden_dim] orientation.\n";
        } else {
            std::cerr << "Error: lm_head_weight has unexpected shape ["
                      << (lw.dim() >= 1 ? lw.size(0) : -1)
                      << (lw.dim() >= 2 ? ", " + std::to_string(lw.size(1)) : "")
                      << "], expected [" << g_cfg.vocab_size << ", " << g_cfg.hidden_dim << "].\n";
            return false;
        }
    }

    // Validate all required weights are defined
    auto& w = g_model_weights;
    if (!w.embed_tokens_weight.defined()) {
        std::cerr << "Error: model.embed_tokens.weight not found in safetensors.\n";
        return false;
    }
    if (!w.final_norm_weight.defined()) {
        std::cerr << "Error: model.norm.weight not found in safetensors.\n";
        return false;
    }
    if (!w.lm_head_weight.defined()) {
        std::cerr << "Error: lm_head.weight not found and fallback (embed. tied) also missing.\n";
        return false;
    }
    for (int l = 0; l < g_cfg.num_layers; ++l) {
        auto& lw = w.layers[l];
        std::string prefix = "  Layer " + std::to_string(l) + ": ";
        if (!lw.input_layernorm_weight.defined()) { std::cerr << prefix << "input_layernorm.weight missing\n"; return false; }
        if (!lw.q_proj_weight.defined())          { std::cerr << prefix << "q_proj.weight missing\n"; return false; }
        if (!lw.k_proj_weight.defined())          { std::cerr << prefix << "k_proj.weight missing\n"; return false; }
        if (!lw.v_proj_weight.defined())          { std::cerr << prefix << "v_proj.weight missing\n"; return false; }
        if (!lw.o_proj_weight.defined())          { std::cerr << prefix << "o_proj.weight missing\n"; return false; }
        if (!lw.post_attention_layernorm_weight.defined()) { std::cerr << prefix << "post_attention_layernorm.weight missing\n"; return false; }
        if (!lw.gate_proj_weight.defined())       { std::cerr << prefix << "gate_proj.weight missing\n"; return false; }
        if (!lw.up_proj_weight.defined())         { std::cerr << prefix << "up_proj.weight missing\n"; return false; }
        if (!lw.down_proj_weight.defined())       { std::cerr << prefix << "down_proj.weight missing\n"; return false; }
        // PATCH P2-14: Pre-cast gate/up/down to f32 once. Frozen weights, safe to cache.
        lw.gate_proj_weight_f32 = lw.gate_proj_weight.to(torch::kFloat32);
        lw.up_proj_weight_f32   = lw.up_proj_weight.to(torch::kFloat32);
        lw.down_proj_weight_f32 = lw.down_proj_weight.to(torch::kFloat32);
    }

    // Free file buffers (tensors were cloned, so safe)
    for (size_t fi = 0; fi < files.size(); fi++) {
        auto& sf = files[fi].second;
        free(sf.tensors);
        free(sf.metadata);
    }
    files.clear();

    std::cout << "[Component 1] Weights loaded in bfloat16 on CUDA.\n";
    return true;
}