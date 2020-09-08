set -e
rm dgl -rf
git clone --recurse https://github.com/vovallen/dgl.git
cd dgl
git checkout euler_bench
git submodule update --init --recursive
mkdir build
cd build
cmake -DUSE_CUDA=ON ..
make -j16
cd ..
cd python
python3.7 -m pip install -e .
# python 
# python3.7 -m awscli s3 sync s3://dgl-data/distributed/partitioned_data/ ~/data/
# python3.7 -m awscli s3 cp  s3://dgl-data/distributed/partitioned_data/ogbn-paper100M/ogb-paper100M.json /home/ubuntu/data/ogbn-paper100M/ogb-paper100M.json
# set -e 
# cd dgl
# git pull
# cd build 
# cmake -DUSE_CUDA=ON ..
# make -j16
# cd ..
# cd python
# sudo apt install -y python3-pip
# python3.7 -m pip install -e .
# export PYTHONPATH=/home/ubuntu/dgl_distributed/dgl/examples/pytorch/graphsage
# python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr=172.31.4.236 --master_port=1234 train_dist.py --graph-name ogb-product --ip_config ip_config.txt --num-epochs 3 --num-client 20 --batch-size 1000 --lr 0.1
# set -e

# sudo su
# apt update
# apt install -y libhugetlbfs-tests
# echo 2048 > /proc/sys/vm/nr_hugepages
# sysctl -w vm.nr_hugepages=2048
# grep HugePages_Total /proc/meminfo

# mkdir -p /mnt/hugetlbfs
# mount -t hugetlbfs none /mnt/hugetlbfs
# chown ubuntu:ubuntu /mnt/hugetlbfs
# hugeadm --set-recommended-shmmax

