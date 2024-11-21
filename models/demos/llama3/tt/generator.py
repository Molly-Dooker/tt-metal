# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List
import ttnn
import torch
import PIL
from loguru import logger

from llama_models.llama3.api.datatypes import (
    InterleavedTextMedia,
    StopReason,
)

from llama_models.llama3.reference_impl.generation import (
    ChatPrediction,
    CompletionPrediction,
    TokenResult,
    sample_top_p,
)
from models.demos.llama3.tt.llama_common import (
    copy_host_to_device,
    get_padded_prefill_len,
    num_blocks_in_seq,
    get_block_size,
)

from llama_models.llama3.api.chat_format import create_vision_mask


class LlamaGenerator:
    def __init__(self, model, model_args, mesh_device, vllm=False, tokenizer=None, formatter=None):
        """
        Creating a LlamaVision wrapper requires only a mesh_device and model_args.
        With model_args you have the checkpoint location, can specify max batch size
        and max seqlen, and other model specific parameters.

        LlamaVision is general to text and chat.

        For bringup, make this class general to any backend implementation, as long as it takes torch tensors and returns torch tensors.

        """
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self.vllm = vllm
        self.tokenizer = tokenizer
        self.formatter = formatter

        if self.vllm:
            # TODO: if possible, obtain these from llama3 api since tokenizer is None
            self.vision_token = 128256

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size):
        from models.demos.llama3.demo.simple_vision_demo import create_multimodal_model

        max_seq_len = 512  # TODO: Increase to 131072 once it's verified to work
        model_args, model = create_multimodal_model(mesh_device, max_batch_size, max_seq_len)
        return cls(model, model_args, mesh_device, vllm=True)

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def prefill_forward_text(self, tokens: torch.Tensor, page_table=None, kv_cache=None, prompt_lens=None):
        batch, batch_seq_len = tokens.shape
        output_logits = torch.zeros(batch, 1, self.model_args.vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)

        if page_table is not None:
            assert isinstance(
                page_table, torch.Tensor
            ), "page_table must be a torch.Tensor when passing into prefill_forward"

        for user_id in range(batch):
            seq_len = prompt_lens[user_id]
            last_token_idx = seq_len - 1

            prefill_seq_len = get_padded_prefill_len(seq_len)
            prefill_ids = torch.cat(
                [tokens[user_id : user_id + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
            )
            if page_table is not None:
                block_size = get_block_size(kv_cache)
                num_padding_blocks = num_blocks_in_seq(prefill_seq_len, block_size) - num_blocks_in_seq(
                    seq_len, block_size
                )
                page_table_user = torch.cat(
                    [page_table, torch.zeros(batch, num_padding_blocks, dtype=torch.int32)], dim=-1
                )

            logits = self.prefill_forward_single_user_text(
                prefill_ids,
                page_table=page_table_user if page_table is not None else None,
                user_id=user_id,
                last_token_idx=last_token_idx,
                kv_cache=kv_cache,
            )

            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[user_id] = logits

        return output_logits

    def prefill_forward_single_user_text(self, tokens, page_table, user_id, last_token_idx, kv_cache=None):
        prefill_input, rot_mats_prefill, page_table_tt = self.model.prepare_inputs_prefill(
            tokens,
            page_table=page_table,
        )

        tt_logits = self.model.ttnn_prefill_forward(
            prefill_input,
            rot_mats=rot_mats_prefill,
            user_id=user_id,
            page_table=page_table_tt,
            get_last_token=(last_token_idx // 32) * 32,
        )

        logits = self.model.process_output_prefill(tt_logits, last_token_idx=(last_token_idx % 32))

        return logits

    def decode_forward_text(
        self,
        tokens,
        current_pos,
        page_table=None,
    ):
        """
        Performs text decode step.
        Returns logits
        """
        tt_tokens, tt_current_pos, tt_rot_mats, tt_page_table = self.model.prepare_inputs_decode(
            tokens, current_pos, page_table
        )

        tt_logits = self.model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mats=tt_rot_mats,
            page_table=tt_page_table,
        )

        logits = self.model.process_output_decode(tt_logits)
        return logits

    def capture_trace_text(
        self,
        tokens,
        current_pos,
        page_table=None,
    ):
        """
        Captures a trace for the decode_forward method.
        """

        # Compile run
        self.decode_forward_text(tokens, current_pos, page_table)

        # Get inputs ready for trace run
        host_inputs = self.model.prepare_decode_inputs_host(tokens, current_pos, page_table)

        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        transformed_inputs = self.model.transform_decode_inputs_device(*device_inputs)
        tt_out_trace = self.model.ttnn_decode_forward(*transformed_inputs)

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)

        return trace_id, tt_out_trace, *device_inputs

    def decode_forward_trace_text(
        self,
        trace_id,
        device_inputs,
        tt_out_trace,
        tokens,
        current_pos,
        page_table=None,
    ):
        host_inputs = self.model.prepare_decode_inputs_host(tokens, current_pos, page_table)

        device_inputs = copy_host_to_device(
            host_tensors=host_inputs,
            device_tensors=device_inputs,
        )

        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        logits = self.model.process_output_decode(tt_out_trace)

        return logits

    def prefill_forward_single_user(
        self,
        vision_images,
        vision_mask,
        tokens,
        xattn_caches,
        user_id,
        total_len,
        prefill_len,
    ):
        """
        Performs vision encode step then text prefill.
        Returns (xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits)
        """
        B = tokens.shape[0]
        vision_tokens, cross_attention_masks, full_text_row_masked_out_mask = self.model.compute_vision_tokens_masks(
            batch_images=[vision_images],
            batch_masks=[vision_mask],
            total_len=total_len,
        )

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            tt_position_id,
            rot_mats,
        ) = self.model.prepare_inputs_prefill(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, prefill_len=prefill_len
        )

        tt_logits = self.model.ttnn_prefill_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_full_text_mask_expand_11SD,
            xattn_caches,
            tt_position_id,
            rot_mats,
            user_id,
            vision_tokens,
        )

        logits = self.model.process_output_prefill(tt_logits, B, prefill_len)

        return xattn_caches, cross_attention_masks, full_text_row_masked_out_mask, logits

    # TODO: merge this function with prefill_forward
    def prefill_forward_vllm(
        self,
        tokens: torch.Tensor,
        images: List[PIL.Image.Image],
        xattn_caches,
        start_pos,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
    ):
        # TODO: modify vLLM to return list of tokens for prefill instead of tensor with padding

        batch, batch_seq_len = tokens.shape
        output_logits = torch.zeros(batch, 1, self.model_args.vocab_size)

        max_gen_len = self.model_args.max_seq_len - 1  # TODO: double check what this should be

        cross_attention_masks_out = []
        full_text_row_masked_out_masks = []

        for user_id in range(batch):
            logger.info(f"Prefilling User {user_id}")
            vision_image = images[user_id : user_id + 1]
            prompt_tokens = [int(tokens[user_id, i]) for i in range(prompt_lens[user_id])]
            vision_mask = create_vision_mask(prompt_tokens, self.vision_token)

            prefill_len = len(prompt_tokens)
            total_len = prefill_len + max_gen_len  # Prepares mask for full length of output
            prompt_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long).reshape(1, -1)  # B, S

            (
                xattn_caches,
                cross_attention_masks,
                full_text_row_masked_out_mask,
                logits,
            ) = self.prefill_forward_single_user(
                vision_image,
                vision_mask,
                prompt_tokens_tensor,
                xattn_caches,
                user_id=user_id,
                total_len=total_len,
                prefill_len=prefill_len,
            )

            cross_attention_masks_out.append(cross_attention_masks)
            full_text_row_masked_out_masks.append(full_text_row_masked_out_mask)

            output_logits[user_id] = logits[:, -1:, :]

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        return output_logits, cross_attention_masks_out, full_text_row_masked_out_masks

    def prefill_forward(self, vision_images, vision_masks, tokens: torch.Tensor, xattn_caches, total_lens, prompt_lens):
        """
        Batched version of prefill_forward_single_user for vision model.
        """
        batch, batch_seq_len = tokens.shape
        output_logits = torch.zeros(batch, max(prompt_lens), self.model_args.vocab_size)
        output_xattn_masks = []
        output_full_text_row_masked_out_masks = []

        for user_id in range(batch):
            print(f"Prefilling User {user_id}")
            seq_len = prompt_lens[user_id]
            (
                xattn_caches,
                cross_attention_masks,
                full_text_row_masked_out_mask,
                logits,
            ) = self.prefill_forward_single_user(
                vision_images=vision_images[user_id],
                vision_mask=vision_masks[user_id],
                tokens=tokens[user_id : user_id + 1, :seq_len],  # Keep batch dimension
                xattn_caches=xattn_caches,
                user_id=user_id,
                total_len=total_lens[user_id],
                prefill_len=seq_len,
            )
            output_logits[user_id, :seq_len] = logits
            output_xattn_masks.append(cross_attention_masks)
            output_full_text_row_masked_out_masks.append(full_text_row_masked_out_mask)

        return output_logits, output_xattn_masks, output_full_text_row_masked_out_masks

    def decode_forward(
        self,
        start_pos,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        xattn_caches,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
    ):
        """
        Performs text decode step.
        Returns logits
        """

        # forward_decode should be traced callable
        # decorator does compilation, capture, execute
        B, S = tokens.shape
        assert S == 1

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rot_mats,
        ) = self.model.prepare_inputs_decode(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, position_id=start_pos
        )

        tt_logits = self.model.ttnn_decode_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            xattn_caches,
            tt_position_id,
            tt_rot_mats,
        )

        logits = self.model.process_output_decode(tt_logits, B, S)
        return logits

    def capture_trace(
        self,
        position_id,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        xattn_caches,
    ):
        """
        Captures a trace for the decode_forward method.
        """
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rot_mats,
        ) = self.model.prepare_inputs_decode(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, position_id=position_id
        )

        # Compile run
        tt_logits_rm = self.model.ttnn_decode_forward(
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            xattn_caches,
            tt_position_id,
            tt_rot_mats,
        )

        # Get inputs ready for trace run
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rope_id,
        ) = self.model.prepare_decode_inputs_host(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, position_id
        )

        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rope_id,
        ) = self.model.copy_host_to_device(
            (tt_h, tt_xattn_mask, tt_full_text_mask_expand_1NSH, tt_position_id, tt_rope_id)
        )

        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        B = tokens.shape[0]
        # Do on-device transformations of inputs before forward
        (
            tt_rot_mats,
            tt_xattn_mask_transform,
            tt_full_text_mask_expand_1NSH_transform,
        ) = self.model.transform_decode_inputs_device(
            tt_rope_id,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            B=B,
        )

        tt_logits_rm = self.model.ttnn_decode_forward(
            tt_h,
            tt_xattn_mask_transform,
            tt_full_text_mask_expand_1NSH_transform,
            xattn_caches,
            tt_position_id,
            tt_rot_mats,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)

        return trace_id, tt_logits_rm, tt_h, tt_xattn_mask, tt_full_text_mask_expand_1NSH, tt_position_id, tt_rope_id

    def decode_forward_trace(
        self,
        position_id,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        xattn_caches,  # TODO: unused since captured in trace?
        trace_id,
        trace_logits_rm,
        trace_h,
        trace_xattn_mask,
        trace_full_text_mask_expand_1NSH,
        trace_position_id,
        trace_rope_id,
    ):
        (
            tt_h,
            tt_xattn_mask,
            tt_full_text_mask_expand_1NSH,
            tt_position_id,
            tt_rope_id,
        ) = self.model.prepare_decode_inputs_host(
            tokens, cross_attention_masks, full_text_row_masked_out_mask, position_id=position_id
        )

        self.model.copy_host_to_device(
            host_tensors=(tt_h, tt_xattn_mask, tt_full_text_mask_expand_1NSH, tt_position_id, tt_rope_id),
            device_tensors=(
                trace_h,
                trace_xattn_mask,
                trace_full_text_mask_expand_1NSH,
                trace_position_id,
                trace_rope_id,
            ),
        )

        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)

        B, S = tokens.shape
        logits = self.model.process_output_decode(trace_logits_rm, B=B, S=S)

        return logits

    def easy_trace(
        self,
        position_id,
        tokens,
        cross_attention_masks,
        full_text_row_masked_out_mask,
        xattn_caches,
    ):
        """
        Tracing is easy! Just call this method and we'll handle tracing for you.
        """
        if not hasattr(self, "trace_id"):
            (
                trace_id,
                tt_logits_rm,
                tt_h,
                tt_xattn_mask,
                tt_full_text_mask_expand_1NSH,
                tt_position_id,
                tt_rope_id,
            ) = self.capture_trace(
                position_id,
                tokens,
                cross_attention_masks,
                full_text_row_masked_out_mask,
                xattn_caches,
            )
            self.trace_id = trace_id
            self.trace_inputs = {
                "tt_h": tt_h,
                "tt_xattn_mask": tt_xattn_mask,
                "tt_full_text_mask_expand_1NSH": tt_full_text_mask_expand_1NSH,
                "tt_position_id": tt_position_id,
                "tt_rope_id": tt_rope_id,
            }
            self.trace_outputs = {
                "tt_logits_rm": tt_logits_rm,
            }

        return self.decode_forward_trace(
            position_id,
            tokens,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            xattn_caches,
            self.trace_id,
            self.trace_outputs["tt_logits_rm"],
            self.trace_inputs["tt_h"],
            self.trace_inputs["tt_xattn_mask"],
            self.trace_inputs["tt_full_text_mask_expand_1NSH"],
            self.trace_inputs["tt_position_id"],
            self.trace_inputs["tt_rope_id"],
        )

    def generate(
        self,
        model_input,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ):
        # Do initial prefill
        vision_images = model_input.vision.images
        vision_mask = model_input.vision.mask
        prompt_tokens = model_input.tokens
        prefill_len = len(prompt_tokens)
        total_len = prefill_len + max_gen_len  # Prepares mask for full length of output

        prompt_tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long).reshape(1, -1)  # B, S
        # Suboptimal to allocate caches every time
        xattn_caches = self.model.setup_cache(self.model_args.max_batch_size)
        (
            xattn_caches,
            cross_attention_masks,
            full_text_row_masked_out_mask,
            logits,
        ) = self.prefill_forward_single_user(
            vision_images,
            vision_mask,
            prompt_tokens_tensor,
            xattn_caches,
            user_id=0,
            total_len=total_len,
            prefill_len=prefill_len,
        )

        def sample(logits):
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            return next_token, self.tokenizer.decode(next_token.tolist())

        next_token, text = sample(logits)

        yield TokenResult(
            token=next_token[0].item(),
            text=text,
        )

        for gen_idx in range(max_gen_len - 1):
            position_id = torch.tensor([prefill_len + gen_idx])
            next_token_tensor = next_token.reshape(1, 1)  # B, S

            logits = self.decode_forward(
                position_id,
                next_token_tensor,
                [cross_attention_masks],
                [full_text_row_masked_out_mask],
                xattn_caches,
            )

            next_token, text = sample(logits)
            yield TokenResult(
                token=next_token[0].item(),
                text=text,
            )

    def chat_completion(
        self,
        messages,
        temperature=0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model.configuration.max_seq_len:
            max_gen_len = self.model.configuration.max_seq_len - 1

        tokens = []

        stop_reason = None
        for result in self.generate(
            model_input=self.formatter.encode_dialog_prompt(messages, tool_prompt_format=False),
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(result.token)
            if result.text == "<|eot_id|>":
                stop_reason = StopReason.end_of_turn
            elif result.text == "<|eom_id|>":
                stop_reason = StopReason.end_of_message

        if stop_reason is None:
            stop_reason = StopReason.out_of_tokens

        message = self.formatter.decode_assistant_message(tokens, stop_reason)

        return ChatPrediction(generation=message)

    def text_completion(
        self,
        content: InterleavedTextMedia,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len=None,
    ):
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model.configuration.max_seq_len:
            max_gen_len = self.model.configuration.max_seq_len - 1

        model_input = self.formatter.encode_content(content)

        tokens = []

        for result in self.generate(
            model_input=model_input,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            tokens.append(result.token)

        generation = self.tokenizer.decode(tokens)

        return CompletionPrediction(generation=generation)
