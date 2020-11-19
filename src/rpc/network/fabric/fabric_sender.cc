#include <dgl/network/fabric/fabric_communicator.h>
#include <dgl/network/fabric/fabric_communicator_context.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <netinet/in.h>
#include <unistd.h>

namespace dgl {
namespace network {

void FabricSender::AddReceiver(const char* addr, int recv_id) {
  struct sockaddr_in sin = {};
  FabricAddr ctrl_fi_addr;
  sin.sin_family = AF_INET;
  std::vector<std::string> substring, hostport;
  SplitStringUsing(addr, "//", &substring);
  SplitStringUsing(substring[1], ":", &hostport);
  CHECK(inet_pton(AF_INET, hostport[0].c_str(), &sin.sin_addr) > 0)
    << "Invalid ip";
  sin.sin_port = htons(stoi(hostport[1]));
  ctrl_fi_addr.CopyFrom(&sin, sizeof(sin));
  ctrl_peer_fi_addr[recv_id] = ctrl_ep->AddPeerAddr(&ctrl_fi_addr);
}

void FabricSender::Finalize() {
  int64_t* exit_code = new int64_t(0);
  for (auto& kv : peer_fi_addr) {
    fep->Send(exit_code, sizeof(int64_t), kSizeMsg | sender_id, kv.second);
  }
  FabricCommunicatorContext::StartPolling()->join();
}

bool FabricSender::Connect() {
  sender_id = -1;
  struct FabricAddrInfo addrs;
  for (auto& kv : ctrl_peer_fi_addr) {
    // Send local control info to remote
    FabricAddrInfo ctrl_info = {.sender_id = -1,
                                .receiver_id = kv.first,
                                .addr = ctrl_ep->fabric_ctx->addr};
    ctrl_ep->Send(&ctrl_info, sizeof(FabricAddrInfo), kCtrlAddrMsg, kv.second,
                  true);
    FabricAddrInfo fep_info = {
      .sender_id = -1, .receiver_id = kv.first, .addr = fep->fabric_ctx->addr};
    ctrl_ep->Send(&fep_info, sizeof(FabricAddrInfo), kFiAddrMsg, kv.second,
                  true);
    ctrl_ep->Recv(&addrs, sizeof(FabricAddrInfo), kFiAddrMsg, FI_ADDR_UNSPEC,
                  true);
    peer_fi_addr[kv.first] = fep->AddPeerAddr(&addrs.addr);
    sender_id = addrs.sender_id;
    msg_ids[kv.first] = 0;
    // msg_queue_[kv.first] = std::make_shared<MessageQueue>(queue_size_);
  }
  FabricCommunicatorContext::StartPolling();
  return true;
}

STATUS FabricSender::Send(Message msg, int recv_id) {
  CHECK_NOTNULL(msg.data);
  CHECK_GT(msg.size, 0);
  CHECK_GE(recv_id, 0);
  Message* msg_copy = new Message();
  *msg_copy = msg;
  LOG(INFO) << "Send Size: "<< msg.size;
  fep->Send(&msg.size, sizeof(msg.size), (msg_ids[recv_id] << 32) | kSizeMsg | sender_id,
            peer_fi_addr[recv_id]);
  fep->Send(msg.data, msg.size, (msg_ids[recv_id] << 32) | kDataMsg | sender_id,
            peer_fi_addr[recv_id], false, msg_copy);
  msg_ids[recv_id] = (msg_ids[recv_id] + 1) & 0xFFFF;
  return ADD_SUCCESS;
}
}  // namespace network
}  // namespace dgl