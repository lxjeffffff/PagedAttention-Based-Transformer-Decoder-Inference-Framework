#pragma once

struct CPUAttentionConfig {
    int batch_size;
    int num_heads;
    int head_dim;
    int seq_len;
    bool use_cache = true;
    bool use_int8 = false;
};
