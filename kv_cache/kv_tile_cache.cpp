#include "kv_tile_cache.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <fstream>

template<typename T>
KVTileCache<T>::KVTileCache()
    : key_buffer_(nullptr), value_buffer_(nullptr),
      tile_size_(0), head_dim_(0), total_pages_(0) {}

template<typename T>
KVTileCache<T>::~KVTileCache() {
    free_buffers();
}

template<typename T>
void KVTileCache<T>::init(int num_pages, int tile_size, int head_dim) {
    tile_size_ = tile_size;
    head_dim_ = head_dim;
    total_pages_ = num_pages;
    allocate_buffers();
    page_table_.init(total_pages_, head_dim_, tile_size_);
}

template<typename T>
void KVTileCache<T>::resize(int new_num_pages, int new_tile_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_buffers();
    tile_size_ = new_tile_size;
    total_pages_ = new_num_pages;
    allocate_buffers();
    tile_to_page_map_.clear();
    lru_list_.clear();
    page_table_.clear();
    page_table_.init(total_pages_, head_dim_, tile_size_);
}

template<typename T>
void KVTileCache<T>::allocate_buffers() {
    size_t bytes = total_pages_ * tile_size_ * head_dim_ * sizeof(T);
    cudaMalloc(&key_buffer_, bytes);
    cudaMalloc(&value_buffer_, bytes);
}

template<typename T>
void KVTileCache<T>::free_buffers() {
    if (key_buffer_) cudaFree(key_buffer_);
    if (value_buffer_) cudaFree(value_buffer_);
}

template<typename T>
T* KVTileCache<T>::get_key_ptr(int page_id) {
    assert(page_id < total_pages_);
    return key_buffer_ + page_id * tile_size_ * head_dim_;
}

template<typename T>
T* KVTileCache<T>::get_value_ptr(int page_id) {
    assert(page_id < total_pages_);
    return value_buffer_ + page_id * tile_size_ * head_dim_;
}

template<typename T>
void KVTileCache<T>::register_tile(int beam_id, int head_id, int tile_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    TileKey key{beam_id, head_id, tile_id};

    if (tile_to_page_map_.find(key) == tile_to_page_map_.end()) {
        evict_if_needed();
        int new_page_id = tile_to_page_map_.size();
        tile_to_page_map_[key] = new_page_id;
        page_table_.assign(beam_id, head_id, tile_id, new_page_id);
    }

    update_lru(key);
}

template<typename T>
void KVTileCache<T>::update_lru(const TileKey& key) {
    auto it = lru_map_.find(key);
    if (it != lru_map_.end()) {
        lru_list_.erase(it->second);
    }
    lru_list_.push_front(key);
    lru_map_[key] = lru_list_.begin();
}

template<typename T>
void KVTileCache<T>::evict_if_needed() {
    if (tile_to_page_map_.size() >= total_pages_) {
        auto last = lru_list_.back();
        tile_to_page_map_.erase(last);
        lru_map_.erase(last);
        page_table_.remove(last.beam_id, last.head_id, last.tile_id);
        lru_list_.pop_back();
    }
}

template<typename T>
void KVTileCache<T>::sync_page_table_to_gpu() {
    page_table_.sync_to_gpu(); // batch updating
}

template<typename T>
void KVTileCache<T>::save_to_file(const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    size_t bytes = total_pages_ * tile_size_ * head_dim_ * sizeof(T);
    std::vector<T> host_key(bytes / sizeof(T)), host_value(bytes / sizeof(T));
    cudaMemcpy(host_key.data(), key_buffer_, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_value.data(), value_buffer_, bytes, cudaMemcpyDeviceToHost);
    ofs.write((char*)host_key.data(), bytes);
    ofs.write((char*)host_value.data(), bytes);
}

template<typename T>
void KVTileCache<T>::load_from_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    size_t bytes = total_pages_ * tile_size_ * head_dim_ * sizeof(T);
    std::vector<T> host_key(bytes / sizeof(T)), host_value(bytes / sizeof(T));
    ifs.read((char*)host_key.data(), bytes);
    ifs.read((char*)host_value.data(), bytes);
    cudaMemcpy(key_buffer_, host_key.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(value_buffer_, host_value.data(), bytes, cudaMemcpyHostToDevice);
}

template class KVTileCache<float>;
template class KVTileCache<half>;
