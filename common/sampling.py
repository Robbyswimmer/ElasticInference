"""Shared sampling logic used by prefill and decode workers."""
import torch


def apply_sampling(logits, sampling_params, generated_ids=None):
    """Apply sampling parameters to logits and return a sampled token id.

    Args:
        logits: 1-D float tensor of shape (vocab_size,).
        sampling_params: inference_pb2.SamplingParams proto message.
        generated_ids: Optional list of previously generated token ids
                       (used for repetition penalty).

    Returns:
        Sampled token id (int).
    """
    # Repetition penalty
    if generated_ids and sampling_params.repetition_penalty > 1.0:
        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] /= sampling_params.repetition_penalty
            else:
                logits[token_id] *= sampling_params.repetition_penalty

    # Temperature
    temp = sampling_params.temperature if sampling_params.temperature > 0 else 1.0
    logits = logits / temp

    # Top-k filtering
    if sampling_params.top_k > 0:
        top_k = min(sampling_params.top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = float("-inf")

    # Top-p (nucleus) filtering
    if 0.0 < sampling_params.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        cutoff_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= sampling_params.top_p
        sorted_logits[cutoff_mask] = float("-inf")
        logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
