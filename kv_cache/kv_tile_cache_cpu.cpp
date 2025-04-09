#include "kv_tile_cache_cpu.hpp"
#include <cstring>
#include <algorithm>

// ---- struct TileIndex ----
template <typename T>
bool KVTileCacheCPU<T>::TileIndex::operator==(const TileIndex& other) const {
    return batch_id == other.batch_id &&
           head_id == other.head_id &&
           tile_id == other.tile_id;
}

template <typename T>
std::size_t KVTileCacheCPU<T>::TileIndexHash::operator()(const TileIndex& idx) const {
    return std::hash<int>()(idx.batch_id) ^
           (std::hash<int>()(idx.head_id) << 1) ^
           (std::hash<int>()(idx.tile_id) << 2);
}

// ---- constructor ----
template <typename T>
KVTileCacheCPU<T>::KVTileCacheCPU(int max_size, int tile_size)
    : max_size_(max_size), tile_size_(tile_size) {}

// ---- put single tile ----
template <typename T>
void KVTileCacheCPU<T>::put(int batch_id, int head_id, int tile_id, const T* data) {
    TileIndex idx{batch_id, head_id, tile_id};
    std::unique_lock lock(mutex_);

    auto it = cache_.find(idx);
    if (it != cache_.end()) {
        std::memcpy(it->second.data(), data, tile_size_ * sizeof(T));
    } else {
        if (lru_list_.size() >= max_size_) {
            auto last = lru_list_.back();
            cache_.erase(last);
            lru_map_.erase(last);
            lru_list_.pop_back();
        }
        cache_[idx] = std::vector<T>(data, data + tile_size_);
    }
    updateLRU(idx);
}

// ---- put batch of tiles ----
template <typename T>
void KVTileCacheCPU<T>::put_batch(const std::vector<TileIndex>& indices, const std::vector<T*>& data_ptrs) {
    std::unique_lock lock(mutex_);
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto& idx = indices[i];
        const T* data = data_ptrs[i];
        auto it = cache_.find(idx);
        if (it != cache_.end()) {
            std::memcpy(it->second.data(), data, tile_size_ * sizeof(T));
        } else {
            if (lru_list_.size() >= max_size_) {
                auto last = lru_list_.back();
                cache_.erase(last);
                lru_map_.erase(last);
                lru_list_.pop_back();
            }
            cache_[idx] = std::vector<T>(data, data + tile_size_);
        }
        updateLRU(idx);
    }
}

// ---- get tile ----
template <typename T>
const T* KVTileCacheCPU<T>::get(int batch_id, int head_id, int tile_id) {
    TileIndex idx{batch_id, head_id, tile_id};
    std::shared_lock lock(mutex_);
    auto it = cache_.find(idx);
    if (it != cache_.end()) {
        updateLRU(idx);
        return it->second.data();
    }
    return nullptr;
}

// ---- tile size ----
template <typename T>
int KVTileCacheCPU<T>::get_tile_size() const {
    return tile_size_;
}

// ---- save to file ----
template <typename T>
void KVTileCacheCPU<T>::save(const std::string& path) {
    std::unique_lock lock(mutex_);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for saving: " + path);
    }
    int size = cache_.size();
    out.write((char*)&size, sizeof(int));
    for (const auto& [idx, data] : cache_) {
        out.write((char*)&idx, sizeof(TileIndex));
        out.write((char*)data.data(), tile_size_ * sizeof(T));
    }
}

// ---- load from file ----
template <typename T>
void KVTileCacheCPU<T>::load(const std::string& path) {
    std::unique_lock lock(mutex_);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for loading: " + path);
    }
    int size;
    in.read((char*)&size, sizeof(int));
    for (int i = 0; i < size; ++i) {
        TileIndex idx;
        in.read((char*)&idx, sizeof(TileIndex));
        std::vector<T> data(tile_size_);
        in.read((char*)data.data(), tile_size_ * sizeof(T));
        cache_[idx] = std::move(data);
        lru_list_.push_front(idx);
        lru_map_[idx] = lru_list_.begin();  // sync lru_map_
    }
}

// ---- update LRU list ----
template <typename T>
void KVTileCacheCPU<T>::updateLRU(const TileIndex& idx) {
    auto it = lru_map_.find(idx);
    if (it != lru_map_.end()) {
        lru_list_.erase(it->second);
    }
    lru_list_.push_front(idx);
    lru_map_[idx] = lru_list_.begin();
}

template class KVTileCacheCPU<float>;
template class KVTileCacheCPU<int8_t>;
template class KVTileCacheCPU<uint16_t>;  // optional half precision
