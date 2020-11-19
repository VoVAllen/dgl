#pragma once

#include <dgl/network/msg_queue.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include "fabric_endpoint.h"
#include "fabric_utils.h"

namespace dgl {
namespace network {

class FabricCommunicatorContext {
  static bool HandleCompletionEvent(const struct fi_cq_tagged_entry& cq_entry,
                                    FabricEndpoint* fep, QueueMap* msg_queue_) {
    uint64_t flags = cq_entry.flags;
    if ((flags & FI_SEND) == FI_SEND) {
      if ((cq_entry.op_context) and
          ((cq_entry.tag & SenderIdMask) == kDataMsg)) {
        Message* msg_ptr = reinterpret_cast<Message*>(cq_entry.op_context);
        if (msg_ptr->deallocator != nullptr) {
          msg_ptr->deallocator(msg_ptr);
        }
        delete msg_ptr;
      }
    } else {
      CHECK_EQ((flags & FI_RECV), FI_RECV);
      CHECK_EQ((flags & FI_TAGGED), FI_TAGGED);
      uint64_t tag = cq_entry.tag & MsgTagMask;
      uint64_t sender_id = cq_entry.tag & SenderIdMask;
      if (tag == kSizeMsg) {
        CHECK(cq_entry.len == sizeof(int64_t)) << "Invalid size message";
        LOG(INFO) << "recved size: " << *(int64_t*)cq_entry.buf;
        int64_t data_size = *(int64_t*)cq_entry.buf;
        LOG(INFO) << "data size: " << data_size;
        char* buffer = nullptr;
        if (data_size == 0) {  // Indicate receiver should exit
          return false;
        }
        try {
          buffer = new char[data_size];
        } catch (const std::bad_alloc&) {
          LOG(FATAL) << "Cannot allocate enough memory for message, "
                     << "(message size: " << data_size << ")";
        }
        // TODO
        // free(cq_entry.buf);  // Free size buffer
        // Receive from specific sender
        fep->Recv(buffer, data_size, kDataMsg | sender_id,
                  FI_ADDR_UNSPEC, false, ~(MsgTagMask | SenderIdMask));
      } else if (tag == kDataMsg) {
        Message* msg = new Message();
        msg->data = reinterpret_cast<char*>(cq_entry.buf);
        msg->size = cq_entry.len;
        msg->deallocator = DefaultMessageDeleter;
        msg_queue_->at(sender_id)
          ->Add(msg, ((cq_entry.tag & MsgIdMask) >> 32));
        int64_t* size_buffer =
          new int64_t;  // can be optimized with buffer pool
        fep->Recv(size_buffer, sizeof(int64_t), kSizeMsg, FI_ADDR_UNSPEC, false,
                  ~MsgTagMask);  // can use FI_ADDR_UNSPEC flag
      } else {
        if (tag != kIgnoreMsg) {
          LOG(INFO) << "Invalid tag";
        }
      }
    }
    return true;
  }

  static void PollCompletionQueue(std::shared_ptr<FabricEndpoint> fep,
                                  std::shared_ptr<QueueMap> queue) {
    struct fi_cq_tagged_entry cq_entries[kMaxConcurrentWorkRequest];
    bool keep_polling = true;
    while (keep_polling) {
      int ret = fi_cq_read(fep->fabric_ctx->cq.get(), cq_entries,
                           kMaxConcurrentWorkRequest);
      if (ret == -FI_EAGAIN) {
        continue;
      } else if (ret == -FI_EAVAIL) {
        HandleCQError(fep->fabric_ctx->cq.get());
      } else if (ret < 0) {
        check_err(ret, "fi_cq_read failed");
      } else {
        CHECK_NE(ret, 0) << "at least one completion event is expected";
        for (int i = 0; i < ret; ++i) {
          if (!HandleCompletionEvent(cq_entries[i], fep.get(), queue.get())) {
            keep_polling = false;
          };
        }
      }
    }
  };

 public:
  static std::shared_ptr<QueueMap> GetQueueMap() {
    static std::shared_ptr<QueueMap> queue;
    if (!queue) {
      queue = std::make_shared<QueueMap>();
    }
    return queue;
  }

  static std::shared_ptr<std::thread> StartPolling() {
    static std::shared_ptr<std::thread> polling_thread;
    if (!polling_thread) {
      polling_thread = std::make_shared<std::thread>(
        PollCompletionQueue, GetEndpoint(), GetQueueMap());
    }
    return polling_thread;
  }
  static std::shared_ptr<FabricEndpoint> GetEndpoint() {
    static std::shared_ptr<FabricEndpoint> fep;
    if (!fep) {
      fep = std::make_shared<FabricEndpoint>();
    }
    return fep;
  }
};
}  // namespace network
}  // namespace dgl