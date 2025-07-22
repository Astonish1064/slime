import ray
import asyncio

from slime.ray.placement_group import create_placement_groups

from .task import Task
from tracer import tracepoint_module_setup, TracePoint
class PipeEngine():
    _instance = None
    
    def __new__(cls,tasks_args):
        """Singleton pattern to ensure only one instance of PipeEngine exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, tasks_args):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.tasks_args = tasks_args
        self.task_num = len(tasks_args)
        self.tasks = []

        self.train_lock= asyncio.Lock()
        self.rollout_lock = asyncio.Lock()
        
        self.pgs = create_placement_groups(self.tasks_args[0])
        locks = [self.train_lock, self.rollout_lock]
        self.tasks = [Task(args, task_id+1, self.pgs,locks, self.task_num) for task_id, args in enumerate(self.tasks_args)]

    async def init_task(self):
        """
        初始化所有任务
        """
        print("Initializing tasks...")
        await asyncio.gather(*[task.init() for task in self.tasks])
        print("All tasks initialized.")

    async def run(self):
        """
        启动所有任务的训练
        """
        print("Starting concurrent training tasks...")

        # 并发执行所有任务的训练
        await asyncio.gather(*[self.train(task_id) for task_id in range(self.task_num)])

        print("All concurrent training tasks finished.")

    async def train(self,task_id):
        print("Starting concurrent training tasks...")
        start_rollout_ids = self.tasks[task_id].start_rollout_ids
        for rollout_id in range(self.tasks[task_id].args.start_rollout_id, self.tasks[task_id].args.num_rollout):

            if self.tasks[task_id].args.eval_interval is not None and rollout_id == 0:
                await asyncio.gather(*[self.tasks[task_id].rollout_group.async_generate(rollout_id, evaluation=True), self.tasks[task_id].train_group.async_eval(rollout_id)])

            tp= TracePoint(f"rollout_generate{self.tasks[task_id].task_id}_{rollout_id}", "1")
            tp.begin()
            await self.tasks[task_id].rollout_group.async_generate(rollout_id)
            tp.end()

            if self.tasks[task_id].args.offload:
                await self.tasks[task_id].rollout_group.async_offload()

            tp= TracePoint(f"train{self.tasks[task_id].task_id}_{rollout_id}", "1")
            tp.begin()
            await asyncio.gather(*self.tasks[task_id].train_group.async_train(rollout_id))
            tp.end()

            if self.tasks[task_id].args.save_interval is not None and (
                (rollout_id + 1) % self.tasks[task_id].args.save_interval == 0
                or (self.tasks[task_id].args.num_rollout_per_epoch is not None and (rollout_id + 1) % self.tasks[task_id].args.num_rollout_per_epoch == 0)
            ):
                await asyncio.gather(*self.tasks[task_id].train_group.async_save_model(rollout_id))
                if self.tasks[task_id].args.rollout_global_dataset:
                    await self.tasks[task_id].rollout_group.data_buffer.save.remote(rollout_id)

            if self.tasks[task_id].args.offload:
                await asyncio.gather(
                    *[self.tasks[task_id].train_group.async_offload(),
                    self.tasks[task_id].rollout_group.async_onload()]
                )
            
            tp= TracePoint(f"update_weights{self.tasks[task_id].task_id}_{rollout_id}", "1")
            tp.begin()
            await asyncio.gather(*self.tasks[task_id].train_group.async_update_weights())
            tp.end()

            if self.tasks[task_id].args.eval_interval is not None and (
                (rollout_id + 1) % self.tasks[task_id].args.eval_interval == 0
                or (self.tasks[task_id].args.num_rollout_per_epoch is not None and (rollout_id + 1) % self.tasks[task_id].args.num_rollout_per_epoch == 0)
            ):
                await asyncio.gather(*[self.tasks[task_id].rollout_group.async_generate(rollout_id, evaluation=True), self.tasks[task_id].train_group.async_eval(rollout_id)])


        print("All concurrent training tasks finished.")