import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemoryReport:
    total_params: int
    weight_bytes: int
    kv_cache_bytes: int
    layer_activation_bytes: int
    full_logits_bytes: int
    last_token_logits_bytes: int


def human_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024 ** 3)
    mib = num_bytes / (1024 ** 2)
    if gib >= 1:
        return f"{gib:.3f} GiB ({mib:.1f} MiB)"
    return f"{mib:.1f} MiB"


def parse_dtype_bytes(dtype: str) -> int:
    mapping = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def estimate_memory(config_path: str, seq_len: int = 1024, batch_size: int = 1) -> MemoryReport:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))

    h = cfg["hidden_size"]
    i = cfg["intermediate_size"]
    l = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    dtype_bytes = parse_dtype_bytes(cfg["torch_dtype"])

    # 参数量估算（Qwen3/LLaMA风格，SwiGLU MLP，无偏置）
    embed_params = vocab * h

    q_proj = h * h
    k_proj = h * kv_heads * head_dim
    v_proj = h * kv_heads * head_dim
    o_proj = h * h
    attn_params = q_proj + k_proj + v_proj + o_proj

    gate_proj = h * i
    up_proj = h * i
    down_proj = i * h
    mlp_params = gate_proj + up_proj + down_proj

    # 每层2个RMSNorm权重
    norm_params_per_layer = 2 * h

    per_layer_params = attn_params + mlp_params + norm_params_per_layer
    final_norm_params = h

    # tie_word_embeddings=true，因此不重复计算lm_head
    total_params = embed_params + l * per_layer_params + final_norm_params

    weight_bytes = total_params * dtype_bytes

    # KV Cache（推理use_cache=true）
    # K + V, shape: [batch, layers, seq, kv_heads, head_dim]
    kv_elements = batch_size * l * seq_len * kv_heads * head_dim * 2
    kv_cache_bytes = kv_elements * dtype_bytes

    # 单层激活粗略峰值（仅推理，不含反向）
    # hidden + qkv + swiglu(gate/up)
    layer_activation_elements = batch_size * seq_len * (h + (h + kv_heads * head_dim * 2) + 2 * i)
    layer_activation_bytes = layer_activation_elements * dtype_bytes

    # logits内存：全序列 vs 仅最后一个token
    full_logits_bytes = batch_size * seq_len * vocab * dtype_bytes
    last_token_logits_bytes = batch_size * vocab * dtype_bytes

    return MemoryReport(
        total_params=total_params,
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_cache_bytes,
        layer_activation_bytes=layer_activation_bytes,
        full_logits_bytes=full_logits_bytes,
        last_token_logits_bytes=last_token_logits_bytes,
    )


def main() -> None:
    path = "qwen3-0.6b-config.json"
    seq_len = 1024
    batch_size = 1
    report = estimate_memory(path, seq_len=seq_len, batch_size=batch_size)

    print("=== Qwen3-0.6B 显存估算（基于配置文件）===")
    print(f"配置文件: {path}")
    print(f"输入长度: {seq_len}, batch_size: {batch_size}")
    print()
    print(f"总参数量(估算): {report.total_params:,}")
    print(f"模型权重显存({human_bytes(report.weight_bytes)}), dtype=bfloat16")
    print(f"KV Cache显存: {human_bytes(report.kv_cache_bytes)}")
    print(f"单层激活峰值(粗略): {human_bytes(report.layer_activation_bytes)}")
    print(f"Logits显存(全序列输出): {human_bytes(report.full_logits_bytes)}")
    print(f"Logits显存(仅最后token): {human_bytes(report.last_token_logits_bytes)}")
    print()

    total_with_full_logits = report.weight_bytes + report.kv_cache_bytes + report.layer_activation_bytes + report.full_logits_bytes
    total_with_last_logits = report.weight_bytes + report.kv_cache_bytes + report.layer_activation_bytes + report.last_token_logits_bytes

    print(f"整体估算(保留全序列logits): {human_bytes(total_with_full_logits)}")
    print(f"整体估算(仅最后token logits): {human_bytes(total_with_last_logits)}")


if __name__ == "__main__":
    main()
