import ray
import copy
import asyncio


from slime.engine import PipeEngine
from slime.utils.arguments import parse_args
import os
import torch

NUM_TASKS=1
async def main(args):
    # initialize ray if not initialized
    if not ray.is_initialized():
        print("--"*50)
        # ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "RAY_DEBUG_POST_MORTEM": "1"}})
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})
    print(f"Process CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"PyTorch device count: {torch.cuda.device_count()}")
    print('ray cluster resources:', ray.cluster_resources())
    print(ray.get_gpu_ids())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    tasks_args = []
    tasks_args.append(args)
    print(tasks_args)
    tasks_args = [copy.deepcopy(args) for _ in range(NUM_TASKS)]
    pipeEngine = PipeEngine(tasks_args)
    await pipeEngine.init_task()
    await pipeEngine.run()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
    print("finish")