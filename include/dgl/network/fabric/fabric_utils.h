#pragma once

#include <arpa/inet.h>
#include <dgl/network/msg_queue.h>
#include <dmlc/logging.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_tagged.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <iostream>
#include <sstream>
#include <string>

#define check_err(ret, msg)                           \
  do {                                                \
    if (ret != 0) {                                   \
      LOG(FATAL) << msg << ". Return Code: " << ret   \
                 << ". ERROR: " << fi_strerror(-ret); \
    }                                                 \
  } while (false)

namespace dgl {
namespace network {

typedef std::unordered_map<int, std::shared_ptr<MessageQueue>> QueueMap;

enum FabricMsgTag : uint64_t {
  kSizeMsg = 0x0000000000010000,
  kDataMsg = 0x0000000000020000,
  kCtrlAddrMsg = 0x0000000000030000,
  kFiAddrMsg = 0x0000000000040000,
  kIgnoreMsg = 0x0000000000050000,
};

static const uint64_t MsgTagMask = 0x00000000FFFF0000;
static const uint64_t IdMask = 0x000000000000FFFF;

static const int FABRIC_VERSION = FI_VERSION(1, 10);

struct FabricDeleter {
  void operator()(fi_info* info) {
    if (info) fi_freeinfo(info);
  }
  void operator()(fid* fid) {
    if (fid) fi_close(fid);
  }
  void operator()(fid_domain* fid) {
    if (fid) fi_close((fid_t)fid);
  }
  void operator()(fid_fabric* fid) {
    if (fid) fi_close((fid_t)fid);
  }
  void operator()(fid_cq* fid) {
    if (fid) fi_close((fid_t)fid);
  }
  void operator()(fid_av* fid) {
    if (fid) fi_close((fid_t)fid);
  }
  void operator()(fid_ep* fid) {
    if (fid) fi_close((fid_t)fid);
  }
  void operator()(fid_eq* fid) {
    if (fid) fi_close((fid_t)fid);
  }
};

static void HandleCQError(struct fid_cq* cq) {
  struct fi_cq_err_entry err_entry;
  int ret = fi_cq_readerr(cq, &err_entry, 1);
  if (ret == FI_EADDRNOTAVAIL) {
    LOG(WARNING) << "fi_cq_readerr: FI_EADDRNOTAVAIL";
  } else if (ret < 0) {
    LOG(FATAL) << "fi_cq_readerr failed. Return Code: " << ret << ". ERROR: "
               << fi_cq_strerror(cq, err_entry.prov_errno, err_entry.err_data,
                                 nullptr, err_entry.err_data_size);
  } else {
    check_err(-err_entry.err, "fi_cq_read failed. retrieved error: ");
  }
}
template <typename T>
using UniqueFabricPtr = std::unique_ptr<T, FabricDeleter>;

// static int ofi_str_to_sin(const char* str, struct sockaddr_in* sin,
//                           size_t* len) {
//   // ;
//   char ip[64];
//   int ret;

//   *len = sizeof(*sin);
//   sin = reinterpret_cast<struct sockaddr_in*>(calloc(1, *len));
//   if (!sin) return -FI_ENOMEM;

//   sin->sin_family = AF_INET;
//   ret = sscanf(str, "%*[^:]://:%" SCNu16, &sin->sin_port);
//   if (ret == 1) goto match_port;

//   ret = sscanf(str, "%*[^:]://%64[^:]:%" SCNu16, ip, &sin->sin_port);
//   if (ret == 2) goto match_ip;

//   ret = sscanf(str, "%*[^:]://%64[^:/]", ip);
//   if (ret == 1) goto match_ip;

//   LOG(ERROR) << "Malformed FI_ADDR_STR: " << str;
// err:
//   // free(sin);
//   LOG(ERROR) << "ERR: " << str;
//   return -FI_EINVAL;

// match_ip:
//   ip[sizeof(ip) - 1] = '\0';
//   ret = inet_pton(AF_INET, ip, &sin->sin_addr);
//   if (ret != 1) {
//     LOG(ERROR) << "Unable to convert IPv4 address: " << ip;
//     goto err;
//   }

// match_port:
//   sin->sin_port = htons(sin->sin_port);
//   // *addr = sin;
//   return 0;
// }

struct FabricAddr {
  // endpoint name
  char name[64] = {};
  // length of endpoint name
  size_t len = sizeof(name);

  std::string DebugStr() const {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < len; i++) {
      ss << std::to_string(name[i]) << ",";
    }
    ss << "]";
    return ss.str();
  }

  std::string str() const { return std::string(name, len); }

  void CopyFrom(void* ep_name, const size_t ep_name_len) {
    len = ep_name_len;
    memcpy(name, ep_name, sizeof(name));
  }

  void CopyTo(char* ep_name, size_t* ep_name_len) {
    *(ep_name_len) = len;
    memcpy(ep_name, name, sizeof(name));
  }
};
}  // namespace network
}  // namespace dgl