#include "page_table.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>

PageTable::PageTable()
    : num_beams_(0), num_heads_(0), num_tiles_(0),
      total_entries_(0), d_table_(nullptr) {}

PageTable::~PageTable() {
    free_device_table();
}

void PageTable::init(int num_beams, int num_heads, int num_tiles) {
    num_beams_ = num_beams;
    num_heads_ = num_heads;
    num_tiles_ = num_tiles;
    total_entries_ = num_beams_ * num_heads_ * num_tiles_;

    allocate_device_table();

    // Initialize to -1 (indicates no page assigned)
    cudaMemset(d_table_, -1, total_entries_ * sizeof(int));
    // Initialize host table
    host_table_.resize(total_entries_, -1);
}

void PageTable::allocate_device_table() {
    size_t bytes = total_entries_ * sizeof(int);
    cudaError_t status = cudaMalloc(&d_table_, bytes);
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed in PageTable: " << cudaGetErrorString(status) << "\n";
    }
}

void PageTable::free_device_table() {
    if (d_table_) {
        cudaFree(d_table_);
        d_table_ = nullptr;
    }
}

void PageTable::clear() {
    if (d_table_) {
        cudaMemset(d_table_, -1, total_entries_ * sizeof(int));
    }
}

void PageTable::assign(int beam_id, int head_id, int tile_id, int page_id) {
    int idx = index(beam_id, head_id, tile_id);
    assert(idx >= 0 && idx < total_entries_);
    cudaMemcpy(d_table_ + idx, &page_id, sizeof(int), cudaMemcpyHostToDevice);
}

int* PageTable::device_data() const {
    return d_table_;
}

void PageTable::sync_to_gpu() {
    size_t bytes = total_entries_ * sizeof(int);
    cudaMemcpy(d_table_, host_table_.data(), bytes, cudaMemcpyHostToDevice);
}

void PageTable::remove(int beam_id, int head_id, int tile_id) {
    int idx = index(beam_id, head_id, tile_id);
    host_table_[idx] = -1; // host_table_ is a temp array of CPU
}