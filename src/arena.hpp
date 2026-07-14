#pragma once
#include "types.hpp"

template <typename T>
[[nodiscard]] T* arena_offset(void* base, size_t& offset, size_t bytes);

void allocate_arena(VramArena& arena, int batch_size, int seq_len);
void free_arena(VramArena& arena);
void free_arena_noexcept(VramArena& arena) noexcept;
