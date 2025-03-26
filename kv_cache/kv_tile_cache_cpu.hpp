#pragma once
#include <unordered_map>
#include <vector>
#include <tuple>
#include <mutex>
#include <shared_mutex>
#include <list>
#include <string>
#include <fstream>

template <typename T = float>
class KVTileCacheCPU {
public:
    struct TileIndex {
        int batch_id;
        int head_id;
        int tile_id;

        bool operator==(const TileIndex& other) const;
    };

    struct TileIndexHash {
        std::size_t operator()(const TileIndex& idx) const;
    };

    KVTileCacheCPU(int max_size, int tile_size);

    void put(int batch_id, int head_id, int tile_id, const T* data);

    void put_batch(const std::vector<TileIndex>& indices, const std::vector<T*>& data_ptrs);

    const T* get(int batch_id, int head_id, int tile_id);

    int get_tile_size() const;

    void save(const std::string& path);
    void load(const std::string& path);

private:
    void updateLRU(const TileIndex& idx);

    int max_size_;
    int tile_size_;

    std::unordered_map<TileIndex, std::vector<T>, TileIndexHash> cache_;
    std::list<TileIndex> lru_list_;
    std::unordered_map<TileIndex, typename std::list<TileIndex>::iterator, TileIndexHash> lru_map_;
    mutable std::shared_mutex mutex_;
};
