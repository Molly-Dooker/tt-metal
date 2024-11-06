# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
from typing import List, Union

import ttnn

from models.demos.llama3.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3.tt.llama_model import TtTransformer
from models.demos.llama3.tt.model_config import TtModelArgs
from models.demos.llama3.tt.llama_common import get_prefill_rot_mat, HostEmbedding, get_rot_transformation_mat


class TtLlamaModelForGeneration:
    def __init__(self, model_args, mesh_device, state_dict, dtype=ttnn.bfloat8_b):
        self.model_args = model_args
        self.mesh_device = mesh_device

        self.tt_embd = TtLlamaEmbedding(
            mesh_device=mesh_device,
            args=model_args,
            weight_cache_path=model_args.weight_cache_path(dtype),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
        )
        self.tt_model = TtTransformer(
            args=model_args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
        )
        self.embd = HostEmbedding(model_args)
        state_dict_prefix = model_args.get_state_dict_prefix("", None)
        self.embd.load_state_dict({"emb.weight": state_dict[f"{state_dict_prefix}tok_embeddings.weight"]})

        del state_dict

        transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim)
        self.transformation_mats = ttnn.from_torch(
            transformation_mat_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, n_layers=None):
        n_layers = 1
        instruct_mode = "Instruct" in hf_config._name_or_path
        model_args = TtModelArgs(mesh_device, instruct=instruct_mode, max_batch_size=max_batch_size)
        if n_layers is not None:
            model_args.n_layers = n_layers
        state_dict = model_args.load_state_dict()
        return cls(model_args, mesh_device, state_dict)

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def forward(
        self,
        tokens: Union[List[List[int]], torch.Tensor],
        input_positions: torch.Tensor,
        is_prefill,
        page_table=None,
        kv_cache=None,
    ):
        if is_prefill:
            assert isinstance(tokens, list), "tokens should be List[List[int]] for prefill mode"
            return self.prefill_forward(
                tokens,
                input_positions,
                page_table=page_table,
                kv_cache=kv_cache,
            )
        else:
            assert isinstance(tokens, torch.Tensor), "tokens should be torch.Tensor for decode mode"
            return self.decode_forward(tokens, input_positions, page_table=page_table, kv_cache=kv_cache)

    def prefill_forward(
        self, tokens: List[List[int]], prompt_lens, page_table=None, kv_cache=None, return_padded_lens=False
    ):
        batch_size = len(tokens)
        output_logits = torch.zeros(batch_size, self.model_args.vocab_size)

        input_tokens_prefill, padded_prefill_lens = self._preprocess_tokens_prefill(tokens)

        # Prefill embeddings are on host since we need to mask out the tokens after the prefill length after embeddings are computed
        pt_prefill_input = [
            self.embd(input_tokens_prefill[b]).view(1, padded_prefill_lens[b], -1) for b in range(batch_size)
        ]

        for batch_id in range(batch_size):
            prefill_seq_len = padded_prefill_lens[batch_id]
            rot_mats_prefill = get_prefill_rot_mat(
                self.model_args.head_dim, self.model_args.max_seq_len, self.mesh_device, seq_len=prefill_seq_len
            )
            if prompt_lens[batch_id] < prefill_seq_len:
                pt_prefill_input[batch_id][
                    :, prompt_lens[batch_id] :, :
                ] = 0  # Zero out the tokens after the prefill length

            prefill_input = self.model_args.prepare_inputs_ttnn_prefill(
                pt_prefill_input[batch_id],
            )

            tt_out = self.tt_model(
                prefill_input,
                None,  # Current position
                rot_mats_prefill,
                self.transformation_mats,
                user_id=batch_id,
                mode="prefill",
                get_last_token=((prompt_lens[batch_id] - 1) // 32) * 32,
            )

            pt_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))[
                0, 0, (prompt_lens[batch_id] - 1) % 32, :
            ]
            ttnn.deallocate(tt_out)
            output_logits[batch_id] = pt_out

        if return_padded_lens:
            return output_logits, padded_prefill_lens
        return output_logits

    def decode_forward(self, tokens: torch.Tensor, current_pos, page_table=None, kv_cache=None):
        pass

    def _preprocess_tokens_prefill(self, tokens: List[List[int]]):
        input_tokens_prefill = []
        padded_prefill_lens = []

        # Always prefill the nearest power of 2 for each user. This means that the majority of cases we will prefill more tokens than needed.
        # To avoid issues, we keep track of the decoding position to decode correctly the user's prompt
        for encoded in tokens:
            # Prefill size is nearest power of 2
            prefill_seq_len = max(2 ** math.ceil(math.log(len(encoded), 2)), 128)

            # Initial prefill tensors full of pad tokens
            input_tokens_prefill_i = torch.full((1, prefill_seq_len), 0, dtype=torch.int32)
            input_tokens_prefill_i[0, : len(encoded[:])] = torch.tensor(encoded[:]).to(input_tokens_prefill_i)
            input_tokens_prefill.append(input_tokens_prefill_i)

            # Keep the correct decoding position of each user
            padded_prefill_lens.append(prefill_seq_len)

        return input_tokens_prefill, padded_prefill_lens
