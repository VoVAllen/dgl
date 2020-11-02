""" API for transferring data to/from the GPU over second stream.A """


from .. import backend as F
from .. import ndarray
from .. import utils
from .._ffi.function import _init_api

class Transfer(object):
    """ Class for representing an asynchronous transfer. """
    def __init__(self, transfer_id, handle):
        """ Create a new Transfer object.

        Parameters
        ----------
        transfer_id : int
            The id of the asynchronous tranfer.
        handle : DGLAsyncTransferer
            The handle of the DGLAsyncTransferer object that initiated the
            transfer.
        """

        self._transfer_id = transfer_id
        self._handle = handle

    def wait(self):
        """ Wait for this transfer to finish, and return the result.

        Returns
        -------
        Tensor
            The new tensor on the target context.
        """
        res_tensor = _CAPI_DGLAsyncTransfererWait(self._handle, self._transfer_id)
        return F.zerocopy_from_dgl_ndarray(res_tensor)



class AsyncTransferer(object):
    """ Class for initiating asynchronous copies to/from the GPU on a second
    GPU stream.

    To initiate a transfer to a GPU:

    >>> tensor_cpu = torch.ones(100000).pin_memory()
    >>> transferer = dgl.dataloading.AsyncTransferer(torch.device(0))
    >>> future = transferer.async_copy(tensor_cpu, torch.device(0))

    And then to wait for the transfer to finish and get a copy of the tensor on
    the GPU.

    >>> tensor_gpu = future.wait()


    """
    def __init__(self, device):
        """ Create a new AsyncTransferer object.

        Parameters
        ----------
        device : Device or context object.
            The context in which the second stream will be created. Must be a
            GPU context for the copy to be asynchronous.
        """
        if isinstance(device, ndarray.DGLContext):
            ctx = device
        else:
            ctx = utils.to_dgl_context(device)
        self._handle = _CAPI_DGLAsyncTransfererCreate(ctx)

    def async_copy(self, tensor, device):
        """ Initiate an asynchronous copy on the internal stream. For this call
        to be asynchronous, the context the AsyncTranserer is created with must
        be a GPU context, and the input tensor must be in pinned memory.

        Currently, transfers from the GPU to the CPU, and CPU to CPU, will
        be synchronous.

        Parameters
        ----------
        tensor : Tensor
            The tensor to transfer.
        device : Device or context object.
            The context to transfer to.

        Returns
        -------
        Transfer
            A Transfer object that can be waited on to get the tensor in the
            new context.
        """
        if isinstance(device, ndarray.DGLContext):
            ctx = device
        else:
            ctx = utils.to_dgl_context(device)

        tensor = F.zerocopy_to_dgl_ndarray(tensor)

        transfer_id = _CAPI_DGLAsyncTransfererStartTransfer(self._handle, tensor, ctx)
        return Transfer(transfer_id, self._handle)


class Prefetcher:
    '''This is a prefetcher defined on top of a data loader.

    The way of loading data is defined by users. The user-defined data loading
    has to be done asynchronously.

    Parameters
    ----------
    dataloader : iterator
        It defines how to sample subgraphs.
    load_data : a callable.
        It defines the logic of loading data for subgraphs. Each time it consumes a bundle of
        subgraphs. It has a field `bundle_size` that defines the number of subgraphs consumed
        by the data loading function. This issues data loading requests and returns `futures`
        for the loaded data.
    num_prefetch : int
        The number of prefetch.
    '''
    def __init__(self, dataloader, load_data, num_prefetch):
        self.dataloader = dataloader
        self.load_data = load_data
        self.num_prefetch = num_prefetch
        self.buf = []
        self.ready_data = []

    def _bundle_prefetch(self, num):
        """prefetch a pre-defined number of subgraphs.
        """
        res = []
        try:
            for i in range(num):
                res.append(next(self.it))
        except StopIteration:
            pass
        return res

    def __iter__(self):
        '''When we create an iterator, we should start to issue requests to prefetch data.
        '''
        self.it = iter(self.dataloader)
        for i in range(self.num_prefetch):
            bundle = self._bundle_prefetch(self.load_data.bundle_size)
            if len(bundle) == 0:
                break
            # Issue data loading request.
            res, future = self.load_data(bundle)
            self.buf.append((res, future))
        return self

    def __next__(self):
        '''Returning data.

        We have issued multiple requests. We need to wait for one of the requests to be complete.
        Once a request is complete, we will issue another request.
        '''
        if len(self.ready_data) > 0:
            return self.ready_data.pop(0)

        if len(self.buf) == 0:
            raise StopIteration()
        else:
            res, future = self.buf.pop(0)
            # Prefetch a new bundle.
            bundle = self._bundle_prefetch(self.load_data.bundle_size)
            if len(bundle) > 0:
                # Issue requests.
                next_res, next_future = self.load_data(bundle)
                self.buf.append((next_res, next_future))
            data = future()
            self.ready_data = [(r, d) for r, d in zip(res, data)]
            return self.ready_data.pop(0)


_init_api("dgl.dataloading.async_transferer")
