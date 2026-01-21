from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
try:
    import torch_npu
except ImportError:
    pass

class ShiftConfidence(DllmAlgorithm):
    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.num_small_blocks = config.algorithm_config.get("num_small_blocks", 1)
        self.alg = config.algorithm_config.get("alg", "confidence_threshold")
        self.eos_token_id = config.algorithm_config.get("eos_token_id")
        self.next_token_cache = None

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:

        prompt_flag = forward_batch.input_ids[0] != self.mask_id

        small_block_length = self.block_size // self.num_small_blocks
        if self.block_size % self.num_small_blocks != 0:
            raise ValueError(f"block_size ({self.block_size}) must be divisible by num_small_blocks ({self.num_small_blocks})")
        
        mask_index = forward_batch.input_ids == self.mask_id
        mask_count = torch.sum(mask_index).item()
            
        if not prompt_flag:
            if self.next_token_cache is not None:
                forward_batch.input_ids[0] = self.next_token_cache
                self.next_token_cache = None
            for small_block_idx in range(self.num_small_blocks):
                small_block_start = small_block_idx * small_block_length
                small_block_end = small_block_start + small_block_length

                while True:
                    mask_index = forward_batch.input_ids[small_block_start:small_block_end] == self.mask_id
                    mask_count = torch.sum(mask_index).item()
                    if mask_count == 0:
                        break
                    out = model_runner.forward(
                        forward_batch, pp_proxy_tensors=None
                    )
                    logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
                    original_logits = logits_output.full_logits #shape(32,153376)
                    shifted_logits = torch.cat([original_logits[:1, :], original_logits[:-1, :]], dim=0)
                    shifted_logits = shifted_logits[small_block_start:small_block_end]
                    x = torch.argmax(shifted_logits, dim=-1)
                    p = torch.squeeze(
                        torch.gather(
                            F.softmax(shifted_logits, dim=-1),
                            dim=-1,
                            index=torch.unsqueeze(x,-1),
                        ),
                        -1,
                    )
                    x = torch.where(mask_index, x, forward_batch.input_ids[small_block_start:small_block_end])
                    confidence = torch.where(mask_index, p, -np.inf)

                    if self.alg=="confidence_threshold":
                        transfer_index = confidence > self.threshold
                    else:
                        best_idx = torch.argmax(confidence)
                        transfer_index = F.one_hot(best_idx, num_classes=confidence.shape[0]).to(torch.bool)

                    num_transfer = transfer_index.sum().item()

                    if num_transfer == 0:
                        _, select_index = torch.topk(confidence, k=1)
                        transfer_index[select_index] = True

                    forward_batch.input_ids[small_block_start:small_block_end][transfer_index] = x[transfer_index]
                if self.eos_token_id and (forward_batch.input_ids[small_block_start:small_block_end] == self.eos_token_id).any():
                    break
        out = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        last_logits = logits_output.full_logits[len(forward_batch.input_ids)-mask_count-1, :] 
        self.next_token_cache = torch.argmax(last_logits, dim=-1)
        if prompt_flag:
            next_token_ids = forward_batch.input_ids[len(forward_batch.input_ids):] 
        else :
            next_token_ids = forward_batch.input_ids 
        return logits_output, next_token_ids, can_run_cuda_graph

Algorithm = ShiftConfidence