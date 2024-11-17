# LLMs in TT-NN
Authors:
## Contents
- [LLMs in TT-NN](#llms-in-tt-nn)
  - [Contents](#contents)
  - [1. Overview](#1-overview)
  - [2. Modules](#2-modules)
    - [2.1 Embedding](#21-embedding)
    - [2.2 RoPE](#22-rope)
    - [2.3 Norm](#23-norm)
    - [2.4 Attention](#24-attention)
    - [2.5 MLP](#25-mlp)
    - [2.6 Decoder](#26-decoder)
    - [2.7 LM Head](#27-lm-head)
  - [3. Features](#3-features)
    - [3.1 Generative Decoding](#31-generative-decoding)
    - [3.2 Prefill and Decode](#32-prefill-and-decode)
    - [3.3 Multi-Device](#33-multi-device)
    - [3.4 Continuous Batching](#34-continuous-batching)
    - [3.5 vLLM Integration](#34-vllm-integration)
  - [4. Best Practices and Optimizations](#4-best-practices-and-optimizations)
    - [4.1 Tracing](#41-tracing)
    - [4.2 Async Mode](#42-async-mode)
    - [4.3 Multiple CQs](#43-multiple-cqs)
    - [4.4 Op Configs](#44-op-configs)
    - [4.5 Accuracy](#45-accuracy)
    - [4.6 Performance Analysis](#46-performance-analysis)
    - [4.7 Misc. Performance Optimizations](#47-misc-performance-optimizations)
    - [4.8 Module Tests](#48-module-tests)
    - [4.9 Performance Testing](#49-performance-testing)
    - [4.10 Common Pitfalls](#410-common-pitfalls)
      - [4.10.1 Error Messages](#4101-error-messages)
      - [4.10.2 Shard Spec Mismatches](#4102-shard-spec-mismatches)
      - [4.10.3 Ethernet Dispatch Cores](#4103-ethernet-dispatch-cores)
      - [4.10.4 Hangs](#4104-hangs)
        - [4.10.4.1 Tracing](#41041-tracing)
        - [4.10.4.2 Large Matmuls](#41042-large-matmuls)

## 1. Overview
## 2. Modules
### 2.1 Embedding
### 2.2 RoPE
  - Iterative update system
  - When to use our fused op
### 2.3 Norm
  - Replicated layernorm vs distributed layernorm
    - Layernorm/rmsnorm weights in row major / wrapped around tile size trick
### 2.4 Attention

Attention in TT-NN is implemented in custom TT-NN kernels. In PyTorch, the attention op is usually implemented in the following way with 6 steps:

1. QKV projections matmuls
2. Reshape Q, K, V to match the expected input shape for the attention op
3. Apply RoPE to Q and K
4. Cache K and V
5. Scaled Dot Product Attention
6. Output reshape and output matmul

For example, the Llama model is implemented as follows:
```python
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        # (1) QKV projections matmuls
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # (2) Reshape Q, K, V to match the expected input shape for the attention op
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # (3) Apply RoPE to Q and K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # (4) Cache K and V
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # (5) Scaled Dot Product Attention
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        output = torch.scaled_dot_product_attention(xq, keys, values, attn_mask=mask)

        # (6) Output reshape and output matmul
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
```

The generic `torch` implementation is agnostic to **prefill** and **decode** modes, however, our implementation differientiates them. To learn more about the differences between the two modes and how we handle them in TT-NN, please see [3.2 Prefill and Decode](#32-prefill-and-decode). In general, our high performance attention module uses specialized implementations for each mode as they have different memory and compute patterns and bottlenecks, requiring different optimizations.

The rest of this section will organized as follows. We split the attention module into two parts -- **prefill** and **decode** -- and describe the 6 steps implementations for each. Then, we discuss some limitations of the current implementation and useful facts that will help with debugging and performance optimization.

Some common terminology used in this section:
| Term | Description |
| --- | --- |
| bsz | batch size |
| batch_id | batch index (used for prefill) |
| cur_pos/cur_pos_tensor | list/tensor of current positions in the sequence for each batch |
| cache_len | length of the KV cache |
| seqlen | sequence length |
| dim | hidden dimension of input x |
| head_dim | hidden dimension of Q, K, V |
| n_q_heads | number of heads in Q |
| n_kv_heads | number of heads in K, V |

### 2.4.1 Attention Prefill
The attention module in prefill mode expects input shape `(1, bsz=1, seqlen, hidden_dim)` and outputs a tensor of the same shape. Note that `bsz=1` is required. For multiple batches, we simply run prefill iteratively and populate the KV cache at `batch_id`.

An end-to-end example of the prefill attention module is in the `models/demos/llama3/tt/llama_attention.py` file, under the `forward_prefill` method. In short, we break down the attention module in prefill mode into the following steps:
1. QKV projections matmuls.
   - We combine the QKV projection weights into a single tensor, and perform standard `ttnn.linear`. Example:
     ```python
     xqkv_fused = ttnn.linear(x, wqkv, dtype=ttnn.bfloat16)
     ```
   - Input/Output shapes:
      ```python
      (1, 1, seqlen, dim) -> (1, 1, seqlen, (n_q_heads+2*n_kv_heads)*head_dim)
      ```

2. Reshape Q, K, V to match the expected input shape for scaled dot product attention.
   - We split the fused QKV tensor into individual Q, K, V tensors using a custom optimized TM op, `ttnn.experimental.nlp_create_qkv_heads`. Example:
     ```python
     Q, K, V = ttnn.experimental.nlp_create_qkv_heads(xqkv_fused, num_heads=n_q_heads, num_kv_heads=n_kv_heads, transpose_k_heads=False)
     ```
   - Input/Output shapes:
      ```python
      (1, 1, seqlen, (n_q_heads+2*n_kv_heads)*head_dim) -> (1, n_q_heads, seqlen, head_dim), (1, n_kv_heads, seqlen, head_dim), (1, n_kv_heads, seqlen, head_dim)
      ```

3. Apply RoPE to Q and K
   - We apply the RoPE transformation to Q and K using the rotary embedding op outlined in [2.2 RoPE](#22-rope). The input/output shapes remain the same as in step 2.

4. Cache K and V
   - We populate the KV cache at `batch_id` with the current K and V tensors using the `ttnn.fill_cache` op. Example:
     ```python
     ttnn.fill_cache(K_cache, K, batch_id)
     ttnn.fill_cache(V_cache, V, batch_id)
     ```
   - If page table is used, we use the `ttnn.experimental.paged_fill_cache` op. Example:
     ```python
     ttnn.experimental.paged_fill_cache(K_cache, K, page_table, batch_idx=batch_id)
     ttnn.experimental.paged_fill_cache(V_cache, V, page_table, batch_idx=batch_id)
     ```

5. Scaled Dot Product Attention
   - We perform scaled dot product attention using our custom flash attention kernel, `ttnn.transformer.scaled_dot_product_attention`. It takes in the following arguments:
     - `q`: Query tensor of shape `(1, n_q_heads, seqlen, head_dim)`.
     - `k`: Key tensor of shape `(1, n_kv_heads, cache_len, head_dim)`.
     - `v`: Value tensor of shape `(1, n_kv_heads, cache_len, head_dim)`.
     - `attn_mask`: Defaults to `None`. [b x 1 x cache_len x seqlen]. Head broadcasting is implied.
     - `is_causal`: bool, defaults to `true`. Whether to apply causal masking.
     - `scale`: float, defaults to `None`.
     - `program_config`: Defaults to `None`.
     - `compute_kernel_config`: Defaults to `None`.

   - For general prefilling phase use cases with causal attention, it is recommended to set `is_causal=True`. This removes the need for `attn_mask` and attention scores are computed in the lower triangular half of the attention matrix. For example:
     ```python
     attn_output = ttnn.transformer.scaled_dot_product_attention(Q,K,V,is_causal=True)
     ```

   - For non-causal attention, `attn_mask` must be provided. An example is in the cross attention case in visual language models. For example:
     ```python
     attn_output = ttnn.transformer.scaled_dot_product_attention(Q,K,V,attn_mask=mask, is_causal=False)
     ```

6. Output reshape and output matmul
   - At last, we use `ttnn.experimental.nlp_concat_heads` to reshape the output of the attention op, followed by a standard `ttnn.linear` to do the output projection. Example:
     ```python
     attn_output = ttnn.experimental.nlp_concat_heads(attn_output)
     output = ttnn.linear(attn_output, wo)
     ```
   - Input/Output shapes:
     ```python
     (1, n_q_heads, seqlen, head_dim) -> (1, 1, seqlen, hidden_dim) -> (1, 1, seqlen, hidden_dim)
     ```

### 2.4.2 Attention Decode
The attention module in decode mode expects input shape `(1, seqlen=1, bsz, hidden_dim)` and outputs a tensor of the same shape. Decode mode expects sequence length of 1 and parallelizes over batch size due to the auto-regressive nature of decoding.

An end-to-end example of the decode attention module is in the `models/demos/llama3/tt/llama_attention.py` file, under the `forward_decode` method. The decode mode is broken down into the following steps:

1. QKV projections matmuls.
   - This works the same as in prefill mode, using `ttnn.linear`. Note that the input shape is `(1, 1, bsz, dim)` instead of `(1, 1, seqlen, dim)`.
   - Input/Output shapes:
      ```python
      (1, 1, bsz, dim) -> (1, 1, bsz, (n_q_heads+2*n_kv_heads)*head_dim)
      ```

2. Reshape Q, K, V to match the expected input shape for scaled dot product attention.
   - We split the fused QKV tensor into individual Q, K, V tensors using `ttnn.experimental.nlp_create_qkv_heads_decode`. Note that this is a different op than `ttnn.experimental.nlp_create_qkv_heads` used in prefill mode. Example:
     ```python
     Q, K, V = ttnn.experimental.nlp_create_qkv_heads_decode(
      xqkv_fused,
      num_heads=n_q_heads,
      num_kv_heads=n_kv_heads,
      memory_config=ttnn.MemoryConfig(
          ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1
      )
     )
     ```
   - **Input/Output shapes**: The output is height sharded across the batch dimension on `bsz` number of cores.
      ```python
      (1, 1, bsz, (n_q_heads+2*n_kv_heads)*head_dim) -> (1, bsz, n_q_heads, head_dim), (1, bsz, n_kv_heads, head_dim), (1, bsz, n_kv_heads, head_dim)
      ```

3. Apply RoPE to Q and K
   - Again, we apply the RoPE transformation to Q and K using the rotary embedding op outlined in [2.2 RoPE](#22-rope). The input/output shapes remain the same as in step 2.

4. Cache K and V
   - We populate the KV cache at `cur_pos` for all batches with the current K and V tensors using the `ttnn.experimental.paged_update_cache` op. This op takes in an optional `page_table` argument to support paged KV cache updates. Example:
     ```python
     ttnn.experimental.paged_update_cache(keys, K, update_idxs=cur_pos, page_table=page_table)
     ttnn.experimental.paged_update_cache(values, V, update_idxs=cur_pos, page_table=page_table)
     ```
   - If current position is `cur_pos_tensor`, a `ttnn.Tensor` rather than a list, we use the `update_idxs_tensor` argument instead:
     ```python
     ttnn.experimental.paged_update_cache(keys, K, update_idxs_tensor=cur_pos_tensor, page_table=page_table)
     ```

5. Scaled Dot Product Attention Decode
   - We perform scaled dot product attention using our custom flash attention kernel optimized for decode mode, `ttnn.transformer.scaled_dot_product_attention_decode` and `ttnn.transformer.paged_scaled_dot_product_attention_decode` for paged KV cache.
   - `ttnn.transformer.scaled_dot_product_attention_decode` takes in the following arguments:
     - `q`: Query tensor of shape `(1, bsz, n_q_heads, head_dim)`.
     - `k`: Key tensor of shape `(1, bsz, cache_len, head_dim)`.
     - `v`: Value tensor of shape `(1, bsz, cache_len, head_dim)`.
     - `is_causal`: bool, defaults to `true`. Whether to apply causal masking.
     - `attn_mask`: Optional attention mask tensor. Defaults to `None` and only used if `is_causal=False`.
     - `cur_pos`: (Required for is_causal=True) List of current positions in the sequence for each batch. Defaults to `None`. Must be provided if `cur_pos_tensor` is not provided.
     - `cur_pos_tensor`: (Required for is_causal=True) Optional current position tensor. Defaults to `None`. Must be provided if `cur_pos` is not provided.
     - `scale`: Optional scale factor. Defaults to `None`.
     - `program_config`: Optional program configuration. Defaults to `None`.
     - `compute_kernel_config`: Optional compute kernel configuration. Defaults to `None`.
     - `memory_config`: Optional memory configuration for output tensor. Defaults to `None`.
   - `ttnn.transformer.paged_scaled_dot_product_attention_decode` takes in the same arguments as `ttnn.transformer.scaled_dot_product_attention_decode`, but also takes in an additional `page_table_tensor` argument.
   - For general decode use cases, it is recommended to set `is_causal=True`. This removes the need for `attn_mask` which greatly reduces memory bandwidth usage. For example:
     ```python
     attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(Q, K, V, cur_pos_tensor=cur_pos, page_table=page_table)
     ```
   - For non-causal attention, `attn_mask` must be provided. An example is in the cross attention case in visual language models. For example:
     ```python
     attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(Q, K, V, attn_mask=mask, is_causal=False)
     ```

6. Output reshape and output matmul
   - Lastly, we use `ttnn.experimental.nlp_concat_heads_decode` to reshape the output of the attention op, followed by a standard `ttnn.linear` to do the output projection. Example:
     ```python
     attn_output = ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads=n_q_heads)
     output = ttnn.linear(attn_output, wo)
     ```
   - Input/Output shapes:
     ```python
     (1, bsz, n_q_heads, head_dim) -> (1, 1, bsz, hidden_dim) -> (1, 1, bsz, hidden_dim)
     ```

### 2.4.3 Miscellaneous Facts
Flash attention and flash decode are the major ops for attention. They are optimized over for latency and throughput, and perform much better than vanilla implementations. If you are interested in how they work, please refer to our [Flash Attention Tech Report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/FlashAttention/FlashAttention.md).

TLDR -- here are some useful things about the attention ops to keep in mind that will help you write efficient and bug-free code:

1. **Program Configs** in flash attention (and flash decode) ops:
   The Program config has the following parameters:
   - `compute_with_storage_grid_size`: The size of the grid size.
   - `q_chunk_size`: The size of a chunk to process at a time for Q.
   - `k_chunk_size`: The size of a chunk to process at a time for K and V.
   - `exp_approx_mode`: Whether to use the exponential approximation mode for softmax.
   - `max_cores_per_head_batch`: The maximum number of cores to use for each head batch in flash decode.

   Flash attention processes Q, K, V in chunks of size `q_chunk_size` and `k_chunk_size`. The chunk size must be a power of 2 and a multiple of 32. By default, the chunk size is set to 512, but you should experiment with different values to find the best performance. Flash attention is parallelized on the cores specified in `compute_with_storage_grid_size`. For example, if you are running on a grid size of 8x8, then flash attention is parallelized over 64 cores. The parallelization is divided by batch, then by head, then by the number of Q chunks.

   Flash decode processes the entire Q (since query in decode mode is small) and K/V in chunks of size `k_chunk_size`. As a result, the `q_chunk_size` field is not used for flash decode. It is parallelized over the cores specified in `compute_with_storage_grid_size`. The parallelization is divided by batch, then by kv_head. In many cases, there will be more cores than `heads*batch`, so this is why flash decode is needed because it allows for multiple cores to process a single head. In extreme cases where there are too many cores to process a single head, the noc bandwidth between cores will become the bottleneck. We experimentally found out that more than 16 cores per head batch no longer provides any benefits and starts degrading performance. The `max_cores_per_head_batch` field is used to limit the number of cores used for each head batch for flash decode, and is set to 16 by default.

   Lastly, the `exp_approx_mode` field is to set the exponential approximation mode for softmax in flash attention and flash decode. We recommend setting this to `true` for small `seqlen/chunk_size` values. For large `seqlen/chunk_size` values, the error introduced by the exponential approximation can accumulate through chunk accumulation, causing major degradation in pcc. For example in Llama3 models, we use `q_chunk_size` and `k_chunk_size` of 512, and `exp_approx_mode` set to `false` for long sequence lengths greater than 16K.

2. **Current Position Tensor** for flash decode and kv cache ops:

   In decode mode, you can either provide a list of current positions, or a tensor. The tensor version can be more efficient because it supports **tracing**. To learn more about what is tracing and how to use it, please refer to [4.1 Tracing](#41-tracing). In short, tracing requires the traced variables to be statically known at the compile time, so if you provide a list of current positions, you cannot modify it for the next token generation. However, if you provide a tensor, the position values are stored in device memory and can be updated using binary addition op, e.g. `ttnn.add`.

### 2.5 MLP
### 2.6 Decoder
### 2.7 LM Head
## 3. Features
### 3.1 Generative Decoding
### 3.2 Prefill and Decode
  - submodules, tests
  - how to combine prefill and decode,
  - slicing prefill to fit in L1
### 3.3 Multi-Device
  - device mesh
  - column parallel followed by row parallel
  - sharding, CCL ops, reducing CCL overheads, etc.
### 3.4 Continuous Batching
  - quick intro and how it is implemented in demos.
### 3.5 vLLM Integration
  - Our vLLM repo and what's needed to integrate with it.
## 4. Best Practices and Optimizations
### 4.1 Tracing
  - link to existing doc, why it helps decode more
### 4.2 Async Mode
### 4.3 Multiple CQs
  - how to feed back output to input and read output asyncronously
### 4.4 Op Configs
  - Writing correct program configs and shard specs
  - Deciding how many cores to run an op on
    - Why did we use 16 cores for MLP
  - Which matmul to use when @Colman Glagovich
    - 1d, 2d, dram-sharded, ...
  - Implicitly padding weights in program config for matmuls
### 4.5 Accuracy
  - How we measure it (PCC, perplexity, top-1/top-5, end-user tests, benchmarking)
  - How much PCC is enough? Rules of thumb.
  - Accuracy tests
  - Debugging PCC issues
### 4.6 Performance Analysis
  - Performance tooling, tracy
### 4.7 Misc. Performance Optimizations
  - Which dim to shard matmuls on
  - DRAM-sharding
  - Avoiding sharded to interleaved calls
### 4.8 Module Tests
### 4.9 Performance Testing
### 4.10 Common Pitfalls
#### 4.10.1 Error Messages
  - Running out of L1
  - Shard spec and program config mismatches
  - For some TTNN ops (e.g. ttnn.all_gather) it's not supported to pass -1 in the dim argument.
    - You'll see an error related to op invocation where the arguments don't match
#### 4.10.2 Shard Spec Mismatches
#### 4.10.3 Ethernet Dispatch Cores
  - link to any other description, and mention it is needed for N300 and T3K
#### 4.10.4 Hangs
##### 4.10.4.1 Tracing
  - Host communications cause tracing to hang
  - Running without async mode enabled causes tracing to hang
  - Careful with print in tracing
##### 4.10.4.2 Large Matmuls
  - Large matmuls hanging? Link to appropriate ticket with workaround
  - Issue is being investigated with a workaround of setting the output subblock to 1,1 and grid size to 8x7
