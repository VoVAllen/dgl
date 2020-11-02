#pragma once

#include <dgl/network/common.h>
#include <dgl/network/fabric/fabric_utils.h>
#include <dmlc/logging.h>
#include <netinet/in.h>
#include <rdma/fabric.h>

#include <string>

namespace dgl {
namespace network {

// struct sock_in ConvertToSockIn(const char *addr) {}

class FabricProvider {
 public:
  FabricProvider() {}

  FabricProvider(std::string prov_name) : prov_name(prov_name) {
    UniqueFabricPtr<struct fi_info> hints(fi_allocinfo());
    hints->ep_attr->type = FI_EP_RDM;  // Reliable Datagram
    hints->caps = FI_TAGGED | FI_MSG | FI_DIRECTED_RECV;
    if (prov_name != "shm") {
      hints->domain_attr->threading = FI_THREAD_COMPLETION;
      hints->mode = FI_CONTEXT;
      hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
      hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
      hints->tx_attr->msg_order = FI_ORDER_SAS;
      hints->rx_attr->msg_order = FI_ORDER_SAS;
    }
    // hints->domain_attr->av_type = FI_AV_TABLE;
    hints->fabric_attr->prov_name = strdup(prov_name.c_str());

    // fi_getinfo
    struct fi_info *info_;
    int ret =
      fi_getinfo(FABRIC_VERSION, nullptr, nullptr, 0, hints.get(), &info_);
    info.reset(info_);

    CHECK_NE(ret, -FI_ENODATA) << "Could not find any optimal provider";
    check_err(ret, "fi_getinfo failed");
    // struct fi_info *providers = info.get();
    // int i = 0;
    // while (providers) {
    // LOG(INFO) << "Found a fabric provider [" << i << "] "
    //           << providers->fabric_attr->prov_name << ":"
    //           << fi_tostr(&providers->addr_format, FI_TYPE_ADDR_FORMAT);
    //   i++;
    //   providers = providers->next;
    // }
  }

  static FabricProvider *CreateTcpProvider(const char *addr) {
    UniqueFabricPtr<struct fi_info> hints(fi_allocinfo());
    hints->ep_attr->type = FI_EP_RDM;  // Reliable Datagram
    hints->caps = FI_TAGGED | FI_MSG | FI_DIRECTED_RECV;
    hints->domain_attr->threading = FI_THREAD_COMPLETION;
    hints->mode = FI_CONTEXT;
    hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
    hints->domain_attr->data_progress = FI_PROGRESS_MANUAL;
    hints->addr_format = FI_SOCKADDR_IN;
    hints->tx_attr->msg_order = FI_ORDER_SAS;
    hints->rx_attr->msg_order = FI_ORDER_SAS;
    // hints->domain_attr->av_type = FI_AV_TABLE;
    // char *prov = new char("tcp");
    hints->fabric_attr->prov_name = strdup("udp");

    struct sockaddr_in *sdr = new struct sockaddr_in();

    struct fi_info *info_;
    int ret = -1;
    if (addr) {
      std::vector<std::string> substring, hostport;
      SplitStringUsing(addr, "//", &substring);
      SplitStringUsing(substring[1], ":", &hostport);
      // ofi_str_to_sin(addr, sdr, &hints->src_addrlen);
      // hints->src_addr = &sdr;
      ret = fi_getinfo(FABRIC_VERSION, hostport[0].c_str(), hostport[1].c_str(),
                       FI_SOURCE, hints.get(), &info_);
    } else {
      ret =
        fi_getinfo(FABRIC_VERSION, nullptr, nullptr, 0, hints.get(), &info_);
    };

    // fi_getinfo
    FabricProvider *provider = new FabricProvider();
    // provider->prov_name = "tcp";
    provider->info.reset(info_);

    CHECK_NE(ret, -FI_ENODATA) << "Could not find any optimal provider";
    check_err(ret, "fi_getinfo failed");
    return provider;
  }
  std::string prov_name;

  UniqueFabricPtr<fi_info> info;
};
}  // namespace network
}  // namespace dgl