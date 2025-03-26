#pragma once

#include <cuda_runtime.h>

class PageTable {
public:
    PageTable();
    ~PageTable();

    void init(int num_beams, int num_heads, int num_tiles);

    void clear();

    void assign(int beam_id, int head_id, int tile_id, int page_id);

    __device__ int lookup(int beam_id, int head_id, int tile_id) const;

    int* device_data() const;

    void sync_to_gpu();

    void remove(int beam_id, int head_id, int tile_id);

private:
    int num_beams_;
    int num_heads_;
    int num_tiles_;
    int total_entries_;

    int* d_table_;  // Device-side linear page table array
    std::vector<int> host_table_;

    void allocate_device_table();
    void free_device_table();

    __host__ __device__ int index(int beam_id, int head_id, int tile_id) const;
};

inline __host__ __device__
int PageTable::index(int beam_id, int head_id, int tile_id) const {
    return beam_id * (num_heads_ * num_tiles_) + head_id * num_tiles_ + tile_id;
}

inline __device__
int PageTable::lookup(int beam_id, int head_id, int tile_id) const {
    int idx = index(beam_id, head_id, tile_id);
    if (idx < 0 || idx >= total_entries_) return -1;
    return d_table_[idx];
}
