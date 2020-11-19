#include <dgl/network/fabric/fabric_communicator.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <unistd.h>

#include "rdma/fi_domain.h"

namespace dgl {
namespace network {

bool FabricReceiver::Wait(const char* addr, int num_sender) {
  num_sender_ = num_sender;
  ctrl_ep =
    std::unique_ptr<FabricEndpoint>(FabricEndpoint::CreateCtrlEndpoint(addr));
  for (size_t i = 0; i < num_sender; i++) {
    FabricAddrInfo addr;
    ctrl_ep->Recv(&addr, sizeof(FabricAddrInfo), kCtrlAddrMsg, FI_ADDR_UNSPEC,
                  true);
    ctrl_peer_fi_addr[i] = ctrl_ep->AddPeerAddr(&addr.addr);
    receiver_id_ = addr.receiver_id;
    ctrl_ep->Recv(&addr, sizeof(FabricAddrInfo), kFiAddrMsg,
                  ctrl_peer_fi_addr[i], true);
    peer_fi_addr[i] = fep->AddPeerAddr(&addr.addr);
    // fi_to_id[peer_fi_addr[i]] = i;
    FabricAddrInfo info = {.sender_id = static_cast<int>(i),
                           .receiver_id = receiver_id_,
                           .addr = fep->fabric_ctx->addr};
    ctrl_ep->Send(&info, sizeof(FabricAddrInfo), kFiAddrMsg,
                  ctrl_peer_fi_addr[i],
                  true);  // Send back server address
    msg_queue_->insert({i, std::make_shared<FabricMessageQueue>()});
  }
  for (size_t i = 0; i < peer_fi_addr.size() * 4; i++)
  {
    // can be optimized with buffer pool
    // Will be freed in HandleCompletionEvent
    int64_t* size_buffer = new int64_t;
    // Issue recv events
    fep->Recv(size_buffer, sizeof(int64_t), kSizeMsg, FI_ADDR_UNSPEC, false,
              ~MsgTagMask);
  }

  FabricCommunicatorContext::StartPolling();

  return true;
}

STATUS FabricReceiver::Recv(Message* msg, int* send_id) {
  // loop until get a message
  for (;;) {
    for (auto& mq : *msg_queue_) {
      *send_id = mq.first;
      // We use non-block remove here
      STATUS code = msg_queue_->at(*send_id)->Remove(msg);
      if (code == QUEUE_EMPTY) {
        continue;  // jump to the next queue
      } else {
        return code;
      }
    }
  }
  FabricCommunicatorContext::StartPolling()->join();
}

STATUS FabricReceiver::RecvFrom(Message* msg, int send_id) {
  // Get message from specified message queue
  STATUS code = QUEUE_EMPTY;
  while (code != REMOVE_SUCCESS) {
    code = msg_queue_->at(send_id)->Remove(msg);
  }
  return code;
}

void FabricReceiver::Finalize() {
  // Send a signal to tell the message queue to finish its job
  for (auto& mq : *msg_queue_) {
    // wait until queue is empty
    //     while (mq.second->Empty() == false) {
    // #ifdef _WIN32
    //       // just loop
    // #else  // !_WIN32
    //       usleep(1000);
    // #endif  // _WIN32
    //     }
    int ID = mq.first;
    mq.second->SignalFinished(ID);
  }
  // Block main thread until all socket-threads finish their jobs
  // FabricCommunicatorContext::polling_thread->join();
}
}  // namespace network
}  // namespace dgl
