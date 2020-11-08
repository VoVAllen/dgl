/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/cache.cc
 * \brief DGL cache implementation
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <dgl/runtime/object.h>
#include <dgl/runtime/registry.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>

#include <vector>
#include <chrono>
#include <ctime>
#include <unordered_map>
#include <unordered_set>
#include <memory>

#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {

template<class IdType, int Size>
class cache_line {
  char data[(sizeof(IdType) + sizeof(int32_t)) * Size];

 public:
  cache_line() {
    for (int i = 0; i < Size; i++) {
      set(i, -1, -1);
    }
  }

  void set(int idx, IdType id, int32_t loc) {
    reinterpret_cast<IdType *>(data)[idx] = id;
    reinterpret_cast<int32_t *>(data + sizeof(IdType) * Size)[idx] = loc;
  }

  IdType get_id(int idx) const {
    return reinterpret_cast<const IdType *>(data)[idx];
  }

  int32_t get_loc(int idx) const {
    return reinterpret_cast<const int32_t *>(data + sizeof(IdType) * Size)[idx];
  }

  bool is_init(int idx) const {
    return this->get_id(idx) != -1;
  }

  int get_valid_entries() const {
    int valid = 0;
    for (int i = 0; i < Size; i++) {
      valid += is_init(i);
    }
    return valid;
  }

  int find(IdType id) const {
    for (int i = 0; i < Size; i++) {
      if (get_id(i) == id)
        return i;
    }
    return -1;
  }

  int find_empty_entry() const {
    for (int i = 0; i < Size; i++) {
      if (!is_init(i))
        return i;
    }
    return -1;
  }
};

typedef int32_t IdType;
typedef int64_t CacheIdxType;
const int CACHE_LINE_SIZE = 8;

/*
 * This is a cache index. It maps an Id (e.g., node or edge) to the location of a slot in the cache.
 * It is implemented as a set-associative cache. It uses a hashtable to store the locations of
 * a fixed number of cache slots and the cache management (e.g., eviction) only happens inside
 * a hashtable slot.
 */
class SACacheIndex : public runtime::Object {
  std::vector<cache_line<IdType, CACHE_LINE_SIZE>> index;
  CacheIdxType cache_size;

  const cache_line<IdType, CACHE_LINE_SIZE> &get_line(IdType id) const {
    // TODO(zhengda) we need a better way to index it.
    return index[id % index.size()];
  }

  cache_line<IdType, CACHE_LINE_SIZE> &get_line(IdType id) {
    // TODO(zhengda) we need a better way to index it.
    return index[id % index.size()];
  }

 public:
  // cache_size is the number of entries in the cache.
  explicit SACacheIndex(size_t cache_size) {
    index.resize(cache_size / CACHE_LINE_SIZE);
    CacheIdxType cache_idx = 0;
    // here we allocate slots in the actual cache to each entry in the cache index.
    for (size_t i = 0; i < index.size(); i++) {
      for (int j = 0; j < CACHE_LINE_SIZE; j++) {
        index[i].set(j, -1, cache_idx++);
      }
    }
    this->cache_size = cache_idx;
  }

  virtual ~SACacheIndex() = default;

  template<class LookupType>
  void add(const LookupType *ids, size_t len, CacheIdxType *locs) {
    for (size_t i = 0; i < len; i++) {
      cache_line<IdType, CACHE_LINE_SIZE> &line = get_line(ids[i]);
      int idx = line.find_empty_entry();
      if (idx >= 0) {
        CacheIdxType cache_loc = line.get_loc(idx);
        line.set(idx, ids[i], cache_loc);
        locs[i] = cache_loc;
      } else {
        locs[i] = -1;
      }
    }
  }

  int64_t get_cache_size() const {
    return this->cache_size;
  }

  int64_t get_valid_entries() const  {
    size_t valid = 0;
    for (size_t i = 0; i < index.size(); i++)
      valid += index[i].get_valid_entries();
    return valid;
  }

  size_t get_capacity() const {
    return index.size() * CACHE_LINE_SIZE;
  }

  size_t get_space() const {
    return index.size() * sizeof(cache_line<IdType, CACHE_LINE_SIZE>);
  }

  template<class LookupType>
  void lookup(const LookupType *ids, int64_t len, CacheIdxType *locs,
              LookupType *return_ids) const {
#pragma omp parallel for
    for (int64_t i = 0; i < len; i++) {
      const cache_line<IdType, CACHE_LINE_SIZE> &line = get_line(ids[i]);
      int entry_idx = line.find(ids[i]);
      if (entry_idx == -1) {
        // If the id dosn't exist, the location is set to the end of the cache.
        locs[i] = cache_size;
        return_ids[i] = -1;
      } else {
        locs[i] = line.get_loc(entry_idx);
        return_ids[i] = ids[i];
      }
    }
  }

  static constexpr const char* _type_key = "cache.SACache";
  DGL_DECLARE_OBJECT_TYPE_INFO(SACacheIndex, runtime::Object);
};

typedef std::shared_ptr<SACacheIndex> CachePtr;

// Define CacheRef
DGL_DEFINE_OBJECT_REF(CacheRef, SACacheIndex);

DGL_REGISTER_GLOBAL("cache._CAPI_DGLCacheCreate")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    size_t cache_size = args[0];
    auto cache_ptr = std::make_shared<SACacheIndex>(cache_size);
    *rv = CacheRef(cache_ptr);
  });

DGL_REGISTER_GLOBAL("cache._CAPI_DGLGetCacheSize")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CacheRef c = args[0];
    auto cache_ptr = c.sptr();
    *rv = cache_ptr->get_cache_size();
  });

DGL_REGISTER_GLOBAL("cache._CAPI_DGLGetNumOccupied")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CacheRef c = args[0];
    auto cache_ptr = c.sptr();
    *rv = cache_ptr->get_valid_entries();
  });

DGL_REGISTER_GLOBAL("cache._CAPI_DGLCacheLookup")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CacheRef c = args[0];
    auto cache_ptr = c.sptr();
    IdArray ids = args[1];
    NDArray locs = args[2];
    IdArray out_ids = args[3];
    int64_t len = ids->shape[0];
    CHECK_EQ(locs->shape[0], len);
    CHECK_EQ(out_ids->shape[0], len);
    CHECK_EQ(ids->dtype.bits, out_ids->dtype.bits);
    CHECK_EQ(ids->dtype.code, out_ids->dtype.code);
    CHECK_EQ(locs->dtype.code, kDLInt);
    CHECK_EQ(locs->dtype.bits, sizeof(CacheIdxType) * 8);

    ATEN_ID_TYPE_SWITCH(ids->dtype, IdType, {
      const IdType *ids_data = static_cast<IdType *>(ids->data);
      CacheIdxType *locs_data = static_cast<CacheIdxType *>(locs->data);
      IdType *out_ids_data = static_cast<IdType *>(out_ids->data);
      cache_ptr->lookup(ids_data, len, locs_data, out_ids_data);
    });
    *rv = locs;
  });

DGL_REGISTER_GLOBAL("cache._CAPI_DGLCacheAddData")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    CacheRef c = args[0];
    auto cache_ptr = c.sptr();
    IdArray ids = args[1];
    NDArray locs = args[2];
    int64_t len = ids->shape[0];
    CHECK_EQ(locs->shape[0], len);
    CHECK_EQ(locs->dtype.code, kDLInt);
    CHECK_EQ(locs->dtype.bits, sizeof(CacheIdxType) * 8);

    ATEN_ID_TYPE_SWITCH(ids->dtype, IdType, {
      const IdType *ids_data = static_cast<IdType *>(ids->data);
      CacheIdxType *locs_data = static_cast<CacheIdxType *>(locs->data);
      cache_ptr->add(ids_data, len, locs_data);
    });
    *rv = locs;
  });

}  // namespace dgl
