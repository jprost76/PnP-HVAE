import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import hostlist

def setup_ddp(port):
    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
    #gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")
    os.environ["MASTER_ADDR"] = hostnames[0]
    #os.environ["MASTER_PORT"] = str(12345 + int(min(gpu_ids)))
    os.environ["MASTER_PORT"] = port
    print(hostnames)

    H = {}
    H['rank'] = int(os.environ['SLURM_PROCID'])
    H['local_rank'] = int(os.environ['SLURM_LOCALID'])
    H['world_size'] = int(os.environ['SLURM_NTASKS'])
    # bug if -c is not specified !!!
    #H['cpus_per_task'] = int(os.environ['SLURM_CPUS_PER_TASK'])

    # initialize the process group
    dist.init_process_group("nccl", rank=H['rank'], world_size=H['world_size'])
    torch.cuda.set_device(H['local_rank'])
    return H
