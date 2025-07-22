import multiprocessing
import random
import time
import threading

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from slime.backends.sglang_utils.sglang_engine import SglangEngine
from slime.ray.buffer import Buffer
from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import find_available_port, get_host_info, run_router
from .utils import Lock

from tracer import tracepoint_module_setup, TracePoint
import os
import asyncio

# Global variables to prevent multiple router instances
_router_started = False
_router_lock = threading.Lock()

@ray.remote
class RolloutRayActor(RayActor):
    def __init__(self, args, rank: int, task_id: int):
        self.args = args
        self.rank = rank
        self.task_id = task_id

    def init(self, dist_init_addr, port, nccl_port, use_local_engine=False):
        # build infer engine
        self.infer_engine = SglangEngine(
            args=self.args,
            rank=self.rank,
            dist_init_addr=dist_init_addr,
            port=port,
            nccl_port=nccl_port,
            use_local_engine=use_local_engine,
        )

        if self.args.offload:
            # offload the engine to the CPU
            self.infer_engine.sleep()

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self.infer_engine.init_process_group(
            master_address, master_port, rank_offset, world_size, group_name, backend
        )

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
        return self.infer_engine.update_weights_from_distributed(names, dtypes, shapes, group_name)

    def update_weights_from_tensor(self, ipc_handles):
        return self.infer_engine.update_weights_from_tensor(ipc_handles)

    def reset_prefix_cache(self):
        self.infer_engine.reset_prefix_cache()

    def sleep(self, level=1):
        self.infer_engine.sleep(level=level)

    def wake_up(self):
        self.infer_engine.wake_up()

    def pause_generation(self):
        self.infer_engine.pause_generation()

    def continue_generation(self):
        self.infer_engine.continue_generation()

    def local_generate(self, prompt, sampling_params):
        """
        使用本地引擎进行推理
        """
        if hasattr(self.infer_engine, 'llm') and hasattr(self.infer_engine.llm, 'generate'):
            return self.infer_engine.llm.generate(prompt, sampling_params)
        else:
            raise NotImplementedError("Local generation not supported by current engine")


def create_rollout_engines(args, task_id, pg, use_local_engine=False):
    print(f'{torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'{torch.cuda.get_device_name(i)} ({i})')
    if args.debug_train_only:
        return []

    num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, 8)
    num_engines = args.rollout_num_gpus // num_gpu_per_engine

    pg, reordered_bundle_indices = pg
    print("8"*100)
    print(f"{reordered_bundle_indices}")
    print("8"*100)
    rollout_engines = []
    for i in range(num_engines):
        num_gpus = 0.5
        num_cpus = num_gpus

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )

        rollout_engines.append(
            RolloutRayActor.options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(args, rank=i, task_id=task_id)
        )

    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. dist_init_addr port
    # 4. other ports for dp_attention, which is of size 4 + dp_size
    num_engines_per_node = max(1, min(8, args.rollout_num_gpus) // args.rollout_num_gpus_per_engine)
    addr_and_ports = [{} for _ in range(num_engines)]
    for rank, engine in enumerate(rollout_engines):
        if rank % num_engines_per_node != 0:
            continue

        def get_addr_and_ports():
            # use small ports to prevent ephemeral port between 32768 and 65536.
            start_port = 10000

            def port(consecutive=1):
                nonlocal start_port
                _, port = ray.get(
                    engine._get_current_node_ip_and_free_port.remote(
                        start_port=start_port,
                        consecutive=consecutive,
                    )
                )
                start_port = port + consecutive
                return port

            def addr():
                addr, _ = ray.get(engine._get_current_node_ip_and_free_port.remote())
                return addr

            return addr, port

        get_addr, get_port = get_addr_and_ports()

        for i in range(num_engines_per_node):
            addr_and_ports[rank + i]["port"] = get_port()
            addr_and_ports[rank + i]["nccl_port"] = get_port()

        if args.rollout_num_gpus_per_engine > 8:
            num_node_per_engine = args.rollout_num_gpus_per_engine // 8
            if rank % num_node_per_engine == 0:
                # this is the first node in the engine, we need to allocate the dist_init_addr port
                dist_init_addr = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"
                for i in range(num_node_per_engine):
                    addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
        else:
            for i in range(num_engines_per_node):
                addr_and_ports[rank + i]["dist_init_addr"] = f"{get_addr()}:{get_port(6 + args.sglang_dp_size)}"

    for i in range(num_engines):
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        print(f"Ports for engine {i}: {addr_and_ports[i]}")

    # TODO: don't ray.get here to overlap train actor init with rollout engine init.
    # somehow if we don't sync here, the --debug-rollout-only mode will crash.
    if use_local_engine:
        print("Initializing rollout engines with local SGLang engines")
        # 对于本地引擎，我们传递use_local_engine参数
        init_handles = []
        for engine, ports in zip(rollout_engines, addr_and_ports):
            # 修改init调用以传递use_local_engine参数
            init_handles.append(engine.init.remote(
                dist_init_addr=ports["dist_init_addr"],
                port=ports["port"],
                nccl_port=ports["nccl_port"],
                use_local_engine=True
            ))
    else:
        print("Initializing rollout engines with HTTP server engines")
        init_handles = [engine.init.remote(**ports) for engine, ports in zip(rollout_engines, addr_and_ports)]
    
    ray.get(init_handles)

    return rollout_engines


class RolloutGroup:
    def __init__(self, args, task_id, pg, use_local_engine=False):
        self.args = args
        self.task_id = task_id
        self.use_local_engine = use_local_engine
        
        # 如果使用本地引擎，则不需要启动router
        if not use_local_engine:
            self.start_router()
        else:
            print("Using local engines, skipping router startup")
            
        tracepoint_module_setup()
        tp = TracePoint(f"buffer_create{self.task_id}", "1")
        tp.begin()
        self.data_buffer = Buffer.options(
            num_cpus=1,
            num_gpus=0,
        ).remote(args,use_local_engine=use_local_engine)
        tp.end()
        print(f'{torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'{torch.cuda.get_device_name(i)} ({i})')
        tp = TracePoint(f"create_rollout_engines{self.task_id}", "1")
        tp.begin()
        self.all_rollout_engines = create_rollout_engines(args, task_id, pg, use_local_engine)
        tp.end()
        nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // 8)
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        self.rollout_engines = self.all_rollout_engines[::nodes_per_engine]
        self.rollout_engine_lock = Lock.options(
            num_cpus=1,
            num_gpus=0,
        ).remote()

    def start_router(self):
        global _router_started
        
        if self.args.sglang_router_ip is not None:
            return
            
        with _router_lock:
            # Double check to prevent race conditions
            if _router_started or self.args.sglang_router_ip is not None:
                return

            from sglang_router.launch_router import RouterArgs

            self.args.sglang_router_ip = get_host_info()[1]
            self.args.sglang_router_port = find_available_port(random.randint(3000, 4000))

            router_args = RouterArgs(
                host=self.args.sglang_router_ip,
                port=self.args.sglang_router_port,
                balance_abs_threshold=0,
            )

            if hasattr(router_args, "log_level"):
                router_args.log_level = "warn"

            try:
                process = multiprocessing.Process(
                    target=run_router,
                    args=(router_args,),
                )
                process.daemon = True  # Set the process as a daemon
                process.start()
                # Wait 3 seconds
                time.sleep(3)
                if process.is_alive():
                    _router_started = True
                    print(f"SGLang router launched at {self.args.sglang_router_ip}:{self.args.sglang_router_port}")
                else:
                    print("Failed to start SGLang router")
                    raise RuntimeError("Router process failed to start")
            except Exception as e:
                print(f"Error starting router: {e}")
                raise

    async def async_generate(self, rollout_id, evaluation=False):
        print(f'{torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'{torch.cuda.get_device_name(i)} ({i})')
        return await self.data_buffer.generate.remote(rollout_id, evaluation=evaluation)

    async def async_reset_prefix_cache(self):
        return await asyncio.gather(*[engine.reset_prefix_cache.remote() for engine in self.rollout_engines])

    async def async_offload(self):
        return await asyncio.gather(*[engine.sleep.remote() for engine in self.rollout_engines])

    async def async_onload(self):
        return await asyncio.gather(*[engine.wake_up.remote() for engine in self.rollout_engines])