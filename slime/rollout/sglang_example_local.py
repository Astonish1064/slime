import asyncio
import copy

from tqdm import tqdm
from transformers import AutoTokenizer

from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout"]


class GenerateState(metaclass=SingletonMeta):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        if self.tokenizer.chat_template is None:
            if "qwen2-vl" in args.hf_checkpoint.lower():
                self.tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>' + '\n' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        from .rollout_functions import reward_fn

        self.reward_fn = reward_fn

        self.batch_size = args.rollout_batch_size
        self.remaining_batch_size = 0

    def submit_generate_tasks(self, samples: list[list[Sample]]):
        # 对于本地引擎，我们将直接处理样本而不是提交到任务队列
        for sample_group in samples:
            for sample in sample_group:
                # 这里可以添加预处理逻辑
                pass
        self.remaining_batch_size += len(samples)


async def generate(args, sample: Sample, sampling_params, local_engine=None) -> Sample:
    """
    使用本地SGLang引擎进行推理,替代网络调用
    """
    state = GenerateState(args)

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if len(sample.response) > 0:
        response_token_ids = state.tokenizer(sample.response, add_special_tokens=False)["input_ids"]
        sampling_params["max_new_tokens"] -= len(response_token_ids)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    
    if sampling_params["max_new_tokens"] == 0:
        return sample

    # Handle partial rollout samples: continue generation from existing response
    input_text = sample.prompt + sample.response

    # 如果传入了本地引擎，使用本地引擎；否则需要获取引擎实例
    if local_engine is None:
        # 这里需要从某个地方获取本地引擎实例
        # 可能需要修改架构来传递引擎实例
        raise ValueError("Local engine must be provided for local generation")

    try:
        # 直接调用本地引擎
        output = local_engine.generate(
            prompt=input_text,
            sampling_params=sampling_params
        )
        
        # 提取生成的文本
        generated_text = output.get("text", "")
        
        # 移除输入部分，只保留新生成的部分
        if generated_text.startswith(input_text):
            new_text = generated_text[len(input_text):]
        else:
            new_text = generated_text
            
        sample.response += new_text

        # 设置完成状态
        sample.status = Sample.Status.COMPLETED

        # 计算tokens
        prompt_tokens_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
        response_token_ids = state.tokenizer(sample.response, add_special_tokens=False)["input_ids"]
        sample.tokens = prompt_tokens_ids + response_token_ids
        sample.response_length = len(response_token_ids)

    except Exception as e:
        print(f"Error in local generation: {e}")
        sample.status = Sample.Status.ABORTED

    return sample


async def generate_and_rm(args, sample: Sample, sampling_params: dict, local_engine=None, evaluation=False) -> Sample:
    """
    使用本地引擎进行生成和奖励模型评估
    """
    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response and sample.reward is not None
        return sample

    state = GenerateState(args)

    # Generate response using local engine
    sample = await generate_local(args, sample, sampling_params, local_engine)

    if sample.status == Sample.Status.ABORTED:
        return sample

    # Apply reward model
    if not evaluation:
        sample = await async_rm(args, sample, state.reward_fn)

    return sample


async def generate_and_rm_group(args, samples: list[Sample], sampling_params: dict, local_engine=None, evaluation=False) -> list[Sample]:
    """
    批量处理样本组，使用本地引擎
    """
    try:
        # 为每个样本应用生成和奖励模型
        tasks = []
        for sample in samples:
            task = generate_and_rm_local(args, sample, sampling_params, local_engine, evaluation)
            tasks.append(task)
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        completed_samples = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing sample {i}: {result}")
                samples[i].status = Sample.Status.ABORTED
                completed_samples.append(samples[i])
            else:
                completed_samples.append(result)
        
        return completed_samples
        
    except Exception as e:
        print(f"Error in generate_and_rm_group_local: {e}")
        # 标记所有样本为中止状态
        for sample in samples:
            sample.status = Sample.Status.ABORTED
        return samples


def generate_rollout(args, rollout_id, data_buffer, local_engines=None, evaluation=False):
    """
    使用本地引擎生成rollout的主函数
    """
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_agent_rollout_local(args, rollout_id, data_buffer, local_engines, evaluation))


async def generate_agent_rollout(args, rollout_id, data_buffer, local_engines=None, evaluation=False):
    """
    本地引擎版本的agent rollout生成
    """
    print(f"Starting local agent rollout {rollout_id} with evaluation={evaluation}")
    
    if local_engines is None:
        raise ValueError("Local engines must be provided for local rollout generation")
    
    # 加载数据集
    dataset = Dataset.load_from_url(args.prompt_data)
    
    # 构造采样参数
    sampling_params = {
        "max_new_tokens": args.rollout_max_response_len,
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
    }
    
    # 处理数据
    all_samples = []
    engine_idx = 0  # 轮询使用引擎
    
    for data_item in dataset:
        # 为每个数据项创建多个样本
        for _ in range(args.n_samples_per_prompt):
            sample = Sample(
                prompt=data_item['prompt'],
                response="",
                status=Sample.Status.PENDING,
                instance_id=data_item.get('instance_id', f"{rollout_id}_{len(all_samples)}")
            )
            
            # 选择引擎（负载均衡）
            current_engine = local_engines[engine_idx % len(local_engines)]
            engine_idx += 1
            
            # 生成样本
            try:
                processed_sample = await generate_and_rm_local(
                    args, sample, sampling_params.copy(), current_engine, evaluation
                )
                all_samples.append(processed_sample)
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                sample.status = Sample.Status.ABORTED
                all_samples.append(sample)
    
    # 将样本添加到数据缓冲区
    if data_buffer:
        try:
            data_buffer.add_samples(all_samples)
            print(f"Added {len(all_samples)} samples to data buffer")
        except Exception as e:
            print(f"Error adding samples to buffer: {e}")
    
    print(f"Completed local agent rollout {rollout_id}, generated {len(all_samples)} samples")
    return all_samples


