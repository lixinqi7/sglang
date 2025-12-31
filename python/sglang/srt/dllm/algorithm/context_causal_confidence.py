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

def sample_tokens(logits, margin_confidence=False, neg_entropy=False):
    probs = torch.softmax(logits, dim=-1)
    confidence, x0 = probs.max(dim=-1)    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    elif margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[..., 0] - sorted_probs[..., 1]
    return confidence, x0

class ContextCausalConfidence(DllmAlgorithm):
    def __init__(
        self,
        config: DllmConfig,
    ):
        config.use_context_causal_block_diffusion = True
        config.pad_id = 0
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.num_small_blocks = config.algorithm_config.get("num_small_blocks", 1)
        self.alg = config.algorithm_config.get("alg", "entropy")
        self.next_token_cache = None
        self.pad_len = 0
        self.pad_id = config.pad_id

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        if self.alg == "entropy":
            neg_entropy = True
        else :
            neg_entropy = False

        prompt_flag = forward_batch.input_ids[0] != self.mask_id
        small_block_length = self.block_size // self.num_small_blocks
        if self.block_size % self.num_small_blocks != 0:
            raise ValueError(f"block_size ({self.block_size}) must be divisible by num_small_blocks ({self.num_small_blocks})")
        mask_index = forward_batch.input_ids == self.mask_id
        mask_count = torch.sum(mask_index).item()
        if forward_batch.model_specific_states is None:
            forward_batch.model_specific_states = {}
        if prompt_flag:
            pad_len = torch.sum(forward_batch.input_ids == self.pad_id).item()
            if pad_len!=self.pad_len:
                self.pad_len = pad_len
        forward_batch.model_specific_states["pad_len"] = self.pad_len

        if self.pad_len:
            padding_mask = torch.ones(forward_batch.seq_lens[0], dtype=torch.bool, device=forward_batch.input_ids.device).unsqueeze(0)
            padding_mask[:, :self.pad_len] = False
            padding_mask = torch.logical_and(
                padding_mask.unsqueeze(1).unsqueeze(-2),
                padding_mask.unsqueeze(1).unsqueeze(-1),
            )
            
        if not prompt_flag:
            if self.next_token_cache is not None:
                forward_batch.input_ids[0] = self.next_token_cache
                self.next_token_cache = None
            forward_batch.model_specific_states["is_prefill"] = False
            if not self.pad_len:
                forward_batch.model_specific_states["attention_mask"] = None
            else:
                forward_batch.model_specific_states["attention_mask"] = padding_mask
            for small_block_idx in range(self.num_small_blocks):
                small_block_start = small_block_idx * small_block_length
                small_block_end = small_block_start + small_block_length
                for _ in range(small_block_length):
                    mask_index = forward_batch.input_ids[small_block_start:small_block_end] == self.mask_id
                    mask_count = torch.sum(mask_index).item()
                    if mask_count == 0:
                        break
                    out = model_runner.forward(
                        forward_batch, pp_proxy_tensors=None
                    )
                    logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
                    original_logits = logits_output.full_logits #shape(32,153376)
                    dummy_row = torch.full_like(original_logits[:1, :], float('-inf'))
                    shifted_logits = torch.cat([dummy_row, original_logits[:-1, :]], dim=0)
                    shifted_logits = shifted_logits[small_block_start:small_block_end]
                    confidence, x = sample_tokens(
                        shifted_logits,
                        temperature=0.0,  
                        neg_entropy=neg_entropy  
                    )
                    x = torch.where(mask_index, x, forward_batch.input_ids[small_block_start:small_block_end])
                    confidence = torch.where(mask_index, confidence, -np.inf)
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
        forward_batch.model_specific_states["is_prefill"] = True
        causal_mask = torch.tril(torch.ones(forward_batch.seq_lens[0], forward_batch.seq_lens[0], device=forward_batch.input_ids.device, dtype=torch.bool))[None, None, :, :]
        if not self.pad_len:
            forward_batch.model_specific_states["attention_mask"] = causal_mask
        else:
            forward_batch.model_specific_states["attention_mask"] = (padding_mask & causal_mask)
        out = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        last_logits = logits_output.full_logits[len(forward_batch.input_ids)-mask_count-1, :] #
        confidence, predicted_token_id = sample_tokens(
            last_logits,
            temperature=0.0,
            neg_entropy=neg_entropy
        )
        self.next_token_cache = predicted_token_id
        if prompt_flag:
            next_token_ids = forward_batch.input_ids[len(forward_batch.input_ids):] 
        else :
            next_token_ids = forward_batch.input_ids 
        return logits_output, next_token_ids, can_run_cuda_graph

Algorithm = ContextCausalConfidence