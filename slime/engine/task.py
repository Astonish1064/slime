import ray
import asyncio
from ray.util.placement_group import placement_group
from slime.ray.placement_group import (create_actor_group,create_rollout_group)
from tracer import tracepoint_module_setup,TracePoint

class Task:
    '''
    contains:
    -placement_group
    -train_group
    -rollout_group
    '''

    def __init__(self,args,task_id,pgs,locks,task_num):
        self.args = args
        self.pgs = pgs
        self.locks = locks
        self.task_id = task_id
        self.task_num = task_num
        self.train_group = None
        self.rollout_group = None
        self.offset:dict={
            'train_actor':0,
            'rollout_actor':1,
        }
    
    async def init(self):
        tracepoint_module_setup()
        tp= TracePoint(f"create_train_group{self.task_id}", "1")
        tp.begin()
        self.train_group = await create_actor_group(
                args=self.args,
                pg=self.pgs["actor"],
                task_id=self.task_id,
            )
        tp.end()
        tp= TracePoint(f"create_rollout_group{self.task_id}", "1")
        tp.begin()
        use_local_engine = getattr(self.args, 'use_local_sglang_engine', False)

        self.rollout_group = await create_rollout_group(
                args=self.args,
                pg=self.pgs["rollout"],
                task_id=self.task_id,
                use_local_engine=use_local_engine
            )
        tp.end()
        num_rollout_per_epoch = None

        if self.args.num_rollout is None:
            num_rollout_per_epoch = await self.rollout_group.data_buffer.get_num_rollout_per_epoch.remote()
            self.args.num_rollout = num_rollout_per_epoch * self.args.num_epoch
        assert self.args.num_rollout > 0

        # sync the initialization (model initalization, load checkpoint, etc.)
        tp= TracePoint(f"init_trainer_{self.task_id}", "1")
        tp.begin()
        start_rollout_ids = await self.train_group.async_init(
            self.args,role="actor",with_ref=self.args.kl_coef != 0 or self.args.use_kl_loss
        )
        assert len(set(start_rollout_ids)) == 1
        if self.args.start_rollout_id is None:
            self.args.start_rollout_id = start_rollout_ids[0]
        tp.end()
        await asyncio.sleep(3)  # 确保异步操作完成
        if self.args.rollout_global_dataset:
            await self.rollout_group.data_buffer.load.remote(self.args.start_rollout_id - 1)
        tp= TracePoint(f"init_weight_update_connections{self.task_id}", "1")
        tp.begin()
        await asyncio.gather(self.train_group.async_init_weight_update_connections(self.rollout_group))
        tp.end()
        

        if self.args.offload:
            await self.rollout_group.async_onload()

        await self.train_group.async_update_weights()

        self.start_rollout_ids=start_rollout_ids


    
    async def train(self):
        for rollout_id in range(self.args.start_rollout_id, self.args.num_rollout):

            if self.args.eval_interval is not None and rollout_id == 0:
                await asyncio.gather(*[self.rollout_group.async_generate(rollout_id, evaluation=True), self.train_group.async_eval(rollout_id)])
            tp= TracePoint(f"rollout_generate{self.task_id}_{rollout_id}", "1")
            tp.begin()
            await asyncio.gather(*self.rollout_group.async_generate(rollout_id))
            tp.end()
            if self.args.offload:
                await self.rollout_group.async_offload()
            tp= TracePoint(f"train{self.task_id}_{rollout_id}", "1")
            tp.begin()
            await self.train_group.async_train(rollout_id)
            tp.end()

            if self.args.save_interval is not None and (
                (rollout_id + 1) % self.args.save_interval == 0
                or (self.args.num_rollout_per_epoch is not None and (rollout_id + 1) % self.args.num_rollout_per_epoch == 0)
            ):
                await asyncio.gather(*self.train_group.async_save_model(rollout_id))
                if self.args.rollout_global_dataset:
                    ray.get(self.rollout_group.data_buffer.save.remote(rollout_id))
            
            if self.args.offload:
                await asyncio.gather(
                    *[self.train_group.async_offload(),
                    self.rollout_group.async_onload()]
                )
            tp= TracePoint(f"update_weights{self.task_id}_{rollout_id}", "1")
            tp.begin()
            await self.train_group.async_update_weights()

            if self.args.eval_interval is not None and (
                (rollout_id + 1) % self.args.eval_interval == 0
                or (self.args.num_rollout_per_epoch is not None and (rollout_id + 1) % self.args.num_rollout_per_epoch == 0)
            ):
                await asyncio.gather(*[self.rollout_group.async_generate(rollout_id, evaluation=True), self.train_group.async_eval(rollout_id)])



    


