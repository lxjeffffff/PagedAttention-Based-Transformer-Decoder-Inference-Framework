#pragma once
#include "page_table.hpp"
#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <list>

template<typename T>
class KVTileCache {
public:
    KVTileCache();
    ~KVTileCache();

    void init(int num_pages, int tile_size, int head_dim);
    void resize(int new_num_pages, int new_tile_size);

    T* get_key_ptr(int page_id);
    T* get_value_ptr(int page_id);

    __device__ const T* KVTileCache<T>::get(int beam_id, int head_id, int tile_id, char type) const {
        int page_id = page_table_.lookup(beam_id, head_id, tile_id);
        if (page_id < 0 || page_id >= total_pages_) return nullptr;
        int offset = page_id * tile_size_ * head_dim_;
        return (type == 'k') ? key_buffer_ + offset : value_buffer_ + offset;
    }

    template<typename T>
    __device__ T* KVTileCache<T>::get_write_ptr(int beam_id, int head_id, int tile_id, char type) {
        int page_id = page_table_.lookup(beam_id, head_id, tile_id);
        if (page_id < 0 || page_id >= total_pages_) return nullptr;
        int offset = page_id * tile_size_ * head_dim_;
        return (type == 'k') ? key_buffer_ + offset : value_buffer_ + offset;
    }

    void register_tile(int beam_id, int head_id, int tile_id);

    void save_to_file(const std::string& path);
    void load_from_file(const std::string& path);

    void sync_page_table_to_gpu();

private:
    T* key_buffer_;
    T* value_buffer_;

    int tile_size_;
    int head_dim_;
    int total_pages_;

    PageTable page_table_;

    struct TileKey {
        int beam_id;
        int head_id;
        int tile_id;
        bool operator==(const TileKey& other) const {
            return beam_id == other.beam_id && head_id == other.head_id && tile_id == other.tile_id;
        }
    };

    struct TileKeyHash {
        size_t operator()(const TileKey& key) const {
            return ((std::hash<int>()(key.beam_id) ^
                    (std::hash<int>()(key.head_id) << 1)) >> 1) ^
                    (std::hash<int>()(key.tile_id) << 1);
        }
    };

    std::unordered_map<TileKey, int, TileKeyHash> tile_to_page_map_;
    std::list<TileKey> lru_list_;
    std::unordered_map<TileKey, std::list<TileKey>::iterator, TileKeyHash> lru_map_;
    std::mutex mutex_;

    void allocate_buffers();
    void free_buffers();

    void update_lru(const TileKey& key);
    void evict_if_needed();
};
