from multiprocessing import shared_memory

import random
import pickle
import time
import copy
import torch
import torch.distributed as dist
from .cfg_holder import cfg_unique_holder as cfguh

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

def is_ddp():
    return dist.is_available() and dist.is_initialized()

def get_rank(type='local'):
    ddp = is_ddp()
    global_rank = dist.get_rank() if ddp else 0
    local_world_size = torch.cuda.device_count()
    if type == 'global':
        return global_rank
    elif type == 'local':
        return global_rank % local_world_size
    elif type == 'node':
        return global_rank // local_world_size
    elif type == 'all':
        return global_rank, \
            global_rank % local_world_size, \
            global_rank // local_world_size
    else:
        assert False, 'Unknown type'

def get_world_size(type='local'):
    ddp = is_ddp()
    global_rank = dist.get_rank() if ddp else 0
    global_world_size = dist.get_world_size() if ddp else 1
    local_world_size = torch.cuda.device_count()
    if type == 'global':
        return global_world_size
    elif type == 'local':
        return local_world_size
    elif type == 'node':
        return global_world_size // local_world_size
    elif type == 'all':
        return global_world_size, local_world_size, \
            global_world_size // local_world_size
    else:
        assert False, 'Unknown type'

class barrier_lock(object):
    def __init__(self, n):
        self.n = n
        id = int(random.random()*10000) + int(time.time())*10000
        self.lock_shmname = 'barrier_lock_{}'.format(id)
        lock_shm = shared_memory.SharedMemory(
            name=self.lock_shmname, create=True, size=n)
        for i in range(n):
            lock_shm.buf[i] = 0
        lock_shm.close()

    def destroy(self):
        try:
            lock_shm = shared_memory.SharedMemory(
                name=self.lock_shmname)
            lock_shm.close()
            lock_shm.unlink()
        except:
            return

    def wait(self, k):
        lock_shm = shared_memory.SharedMemory(
            name=self.lock_shmname)
        assert lock_shm.buf[k] == 0, 'Two waits on the same id is not allowed.'
        lock_shm.buf[k] = 1
        if k == 0:
            while sum([lock_shm.buf[i]==0 for i in range(self.n)]) != 0:
                pass
            for i in range(self.n):
                lock_shm.buf[i] = 0
            return 
        else:
            while lock_shm.buf[k] != 0:
                pass

class nodewise_sync_global(object):
    """
    This is the global part of nodewise_sync that need to call at master process
        before spawn.
    """
    def __init__(self):
        self.local_world_size = get_world_size('local')
        self.b_lock = barrier_lock(self.local_world_size)
        id = int(random.random()*10000) + int(time.time())*10000
        self.id_shmname = 'nodewise_sync_id_shm_{}'.format(id)

    def destroy(self):
        self.b_lock.destroy()
        try:
            shm = shared_memory.SharedMemory(name=self.id_shmname)
            shm.close()
            shm.unlink()
        except:
            return

@singleton
class nodewise_sync(object):
    """
    A class that centralize nodewise sync activities.
    The backend is multiprocess sharememory, not torch, as torch not support this.
    """
    def __init__(self):
        pass

    def copy_global(self, reference):
        self.local_world_size = reference.local_world_size
        self.b_lock = reference.b_lock
        self.id_shmname = reference.id_shmname
        return self

    def local_init(self):
        self.ddp = is_ddp()
        self.global_rank, self.local_rank, self.node_rank = get_rank('all')
        self.global_world_size, self.local_world_size, self.nodes = get_world_size('all')
        if self.local_rank == 0:
            temp = int(random.random()*10000) + int(time.time())*10000
            temp = pickle.dumps(temp)
            shm = shared_memory.SharedMemory(
                name=self.id_shmname, create=True, size=len(temp))
            shm.close()
        return self

    def random_sync_id(self):
        assert self.local_rank is not None, 'Not initialized!'
        if self.local_rank == 0:
            sync_id = int(random.random()*10000) + int(time.time())*10000
            data = pickle.dumps(sync_id)
            shm = shared_memory.SharedMemory(name=self.id_shmname)
            shm.buf[0:len(data)] = data[0:len(data)]
            self.barrier()
            shm.close()
        else:
            self.barrier()
            shm = shared_memory.SharedMemory(name=self.id_shmname)
            sync_id = pickle.loads(shm.buf)
            shm.close()
        return sync_id

    def barrier(self):
        self.b_lock.wait(self.local_rank)

    def broadcast_r0(self, data=None):
        assert self.local_rank is not None, 'Not initialized!'
        id = self.random_sync_id()
        shmname = 'broadcast_r0_{}'.format(id)
        if self.local_rank == 0:
            assert data!=None, 'Rank 0 needs to input data!'
            data = pickle.dumps(data)
            datan = len(data)
            load_info_shm = shared_memory.SharedMemory(
                name=shmname, create=True, size=datan)
            load_info_shm.buf[0:datan] = data[0:datan]
            self.barrier()
            self.barrier()
            load_info_shm.close()
            load_info_shm.unlink()
            return None
        else:
            assert data==None, 'Rank other than 1 should input None as data!'
            self.barrier()
            shm = shared_memory.SharedMemory(name=shmname)
            data = pickle.loads(shm.buf)
            shm.close()
            self.barrier()
            return data

    def destroy(self):
        self.barrier.destroy()
        try:
            shm = shared_memory.SharedMemory(name=self.id_shmname)
            shm.close()
            shm.unlink()
        except:
            return

