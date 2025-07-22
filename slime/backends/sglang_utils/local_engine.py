import dataclasses
import os
from typing import TYPE_CHECKING

import sglang as sgl
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    pass


def get_base_gpu_id(args, rank):
    num_gpus = min(8, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % 8
    else:
        num_actor_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % 8
    return start_index


class SglangLocalEngine:
    """
    本地SGLang引擎，直接调用模型而无需HTTP服务器
    """

    def __init__(self, args, rank, dist_init_addr, port, nccl_port):
        self.args = args
        self.rank = rank
        
        # remove the CUDA_VISIBLE_DEVICES set by ray and use base_gpu_id
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        nnodes = max(1, args.rollout_num_gpus_per_engine // 8)
        node_rank = rank % nnodes
        
        # 构造SGLang引擎参数 - 使用ServerArgs的参数格式
        server_args_dict = {
            "model_path": args.hf_checkpoint,
            "trust_remote_code": True,
            "random_seed": args.seed + rank,
            # memory
            "enable_memory_saver": args.offload,
            # distributed - 本地引擎仍需要这些参数用于多卡推理
            "tp_size": args.rollout_num_gpus_per_engine,
            "dp_size": getattr(args, 'sglang_dp_size', 1),
            "pp_size": getattr(args, 'sglang_pp_size', 1),
            "ep_size": getattr(args, 'sglang_ep_size', 1),
            # 添加分布式相关参数，虽然是本地引擎但多卡推理仍需要
            "nnodes": nnodes,
            "node_rank": node_rank,
            "base_gpu_id": get_base_gpu_id(args, rank),
            # 设置日志级别避免过多输出
            "log_level": "warning",
        }

        # 添加其他sglang参数
        for attr in dataclasses.fields(ServerArgs):
            if hasattr(args, f"sglang_{attr.name}") and attr.name not in server_args_dict:
                server_args_dict[attr.name] = getattr(args, f"sglang_{attr.name}")

        try:
            # 创建SGLang本地引擎
            print(f"Creating SGLang local engine with rank {rank}, args: {server_args_dict}")
            self.llm = sgl.Engine(**server_args_dict)
            print(f"Successfully created SGLang local engine with rank {rank}")
        except Exception as e:
            print(f"Error creating SGLang local engine: {e}")
            raise

    def generate(self, prompt, sampling_params):
        """
        直接调用本地引擎进行推理
        """
        try:
            # 调用SGLang Engine的generate方法
            result = self.llm.generate(
                prompt=prompt,
                sampling_params=sampling_params
            )
            
            # SGLang Engine返回的结果格式需要处理
            if isinstance(result, list) and len(result) > 0:
                # 对于单个prompt，取第一个结果
                return result[0]
            else:
                return result
                
        except Exception as e:
            print(f"Error in local engine generation: {e}")
            raise


    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        """
        初始化权重更新组 - 与HTTP引擎保持接口一致
        """
        print(f"[LOCAL ENGINE] Initializing weights update group: {group_name}")
        print(f"[LOCAL ENGINE] Master: {master_address}:{master_port}, rank_offset: {rank_offset}, world_size: {world_size}")
        
        try:
            self.llm.init_weights_update_group(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
                group_name=group_name,
                backend=backend
            )
            return True
            
        except Exception as e:
            print(f"[LOCAL ENGINE] ✗ Failed to initialize weights update group: {e}")
            import traceback
            traceback.print_exc()
            raise

    def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
        """
        从分布式训练更新权重
        """
        print(f"Updating weights from distributed for group: {group_name}")
        try:
            if hasattr(self.llm, 'update_weights_from_distributed'):
                self.llm.update_weights_from_distributed(names, dtypes, shapes, group_name)
        except Exception as e:
            print(f"Error updating weights from distributed: {e}")

    def update_weights_from_tensor(self, ipc_handles):
        """
        从张量更新权重
        """
        print("Updating weights from tensor for local engine")
        try:
            self.llm.update_weights_from_tensor(ipc_handles)
        except Exception as e:
            print(f"Error updating weights from tensor: {e}")

    def flush_cache(self):
        """重置前缀缓存"""
        try:
            self.llm.flush_cache()
            print("Flushed cache for local engine")
        except Exception as e:
            print(f"Error resetting prefix cache: {e}")

    def release_memory_occupation(self):
        """释放内存占用"""
        try:
            self.llm.release_memory_occupation()
            print("Released memory occupation for local engine")
        except Exception as e:
            print(f"Error releasing memory occupation: {e}")

    def pause_generation(self):
        """暂停生成"""
        try:
            self.llm.tokenizer_manager.pause_generation()
            print("Pausing generation for local engine")
        except Exception as e:
            print(f"Error pausing generation: {e}")
            return False

    def continue_generation(self):
        """恢复生成"""
        try:
            self.llm.tokenizer_manager.continue_generation()
            print("Continuing generation for local engine")
        except Exception as e:
            print(f"Error continuing generation: {e}")
            return False

    def shutdown(self):
        """关闭引擎"""
        try:
            self.llm.shutdown()
            print("Shutdown local engine")
        except Exception as e:
            print(f"Error shutting down local engine: {e}")
