#pragma once

#include <dgl/network/msg_queue.h>
#include <dgl/runtime/ndarray.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <utility>  // for pair

namespace dgl {
namespace network {
class FabricMessageQueue {
 public:
  /*!
   * \brief MessageQueue constructor
   * \param queue_size size (bytes) of message queue
   * \param num_producers number of producers, use 1 by default
   */
  FabricMessageQueue(){};

  /*!
   * \brief MessageQueue deconstructor
   */
  ~FabricMessageQueue() {}

  /*!
   * \brief Add message to the queue
   * \param msg data message
   * \param is_blocking Blocking if cannot add, else return
   * \return Status code
   */
  STATUS Add(Message* msg, int64_t id) {
    // LOG(INFO) << "Add " << id << " to: " << this;
    queue_[id] = msg;
    return ADD_SUCCESS;
  };

  /*!
   * \brief Remove message from the queue
   * \param msg pointer of data msg
   * \param is_blocking Blocking if cannot remove, else return
   * \return Status code
   */
  STATUS Remove(Message* msg) {
    //   queue_[current_idx].compare_exchange_strong()
    if (queue_[current_idx]) {
      Message* msg_in_queue = queue_[current_idx].load();
    //   msg->data = ret_msg->data;
    //   msg->size = ret_msg->size;
    //   msg->deallocator = ret_msg->deallocator;
      *msg = *msg_in_queue;
      queue_[current_idx] = nullptr;
      delete msg_in_queue;
    //   LOG(INFO) << "Remove " << current_idx << " from: " << this;
      current_idx = (current_idx + 1) & 0xFFFF;
      return REMOVE_SUCCESS;
    } else {
      return QUEUE_EMPTY;
    }
  };

  /*!
   * \brief Signal that producer producer_id will no longer produce anything
   * \param producer_id An integer uniquely to identify a producer thread
   */
  void SignalFinished(int producer_id){};

 public:
  /*!
   * \brief message queue
   */
  std::array<std::atomic<Message*>, 65536> queue_;

  /*!
   * \brief Size of the queue in bytes
   */
  int64_t current_idx;
};

}  // namespace network
}  // namespace dgl