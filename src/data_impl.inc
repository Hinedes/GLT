// ============================================================================
// Phase 4: Binary Token Data Loading (.bin files)
// Operation: Reads pre-tokenized sequences from binary files with format:
// [num_sequences:int32][seq_len:int32][tokens:int32...]. Domain data and
// OOD (out-of-domain) data are loaded separately with seq_len validation.
// During training, batches are constructed as half domain + half OOD.
// ============================================================================

[[nodiscard]] bool load_bin(const std::string& path, TokenBuffer& buf) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Error: cannot open " << path << std::endl;
        return false;
    }
    int32_t num_sequences, seq_len;
    f.read(reinterpret_cast<char*>(&num_sequences), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&seq_len), sizeof(int32_t));
    if (!f) {
        std::cerr << "Error: failed reading header from " << path << std::endl;
        return false;
    }
    // Trust-boundary validation: header is attacker-controlled. Reject
    // non-positive dims and overflow/truncation before any allocation.
    if (num_sequences <= 0 || seq_len <= 0) {
        std::cerr << "Error: invalid header in " << path
                  << " (num_sequences=" << num_sequences << ", seq_len=" << seq_len << ")\n";
        return false;
    }
    int64_t total = static_cast<int64_t>(num_sequences) * seq_len;
    // Cap token count to avoid reserve() throwing on a crafted header.
    if (total > (static_cast<int64_t>(1) << 30)) {
        std::cerr << "Error: token count " << total << " in " << path
                  << " exceeds safety limit\n";
        return false;
    }
    // Verify the file actually contains `total` int32 tokens after the header.
    std::streampos header_end = f.tellg();
    f.seekg(0, std::ios::end);
    int64_t file_bytes = static_cast<int64_t>(f.tellg());
    f.seekg(header_end);
    int64_t remaining = file_bytes - static_cast<int64_t>(header_end);
    if (remaining < total * static_cast<int64_t>(sizeof(int32_t))) {
        std::cerr << "Error: truncated token data in " << path
                  << " (expected " << total * sizeof(int32_t)
                  << " bytes, found " << remaining << ")\n";
        return false;
    }
    buf.tokens.resize(total);
    buf.num_sequences = num_sequences;
    buf.seq_len = seq_len;
    f.read(reinterpret_cast<char*>(buf.tokens.data()), total * sizeof(int32_t));
    if (!f) {
        std::cerr << "Error: failed reading token data from " << path << std::endl;
        return false;
    }
    std::cout << "  Loaded " << path << ": " << num_sequences << " sequences, " << seq_len << " tokens each\n";
    return true;
}
