"""A2S model definition.

Copyright PolyAI Limited.
"""
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

import constants as c
from modules import masking_logic
from modules.conformer import Conformer
from modules.masking_logic import (State, mask_by_random_topk,
                                   sample_from_logits, state_init)
from utils import load_checkpoint


class Pheme(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.model = TTSConformer(hp)
        self.cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=self.hp.label_smoothing,
            ignore_index=self.hp.n_codes
        )
        if self.hp.pretrained_path:
            self.load()
        else:
            self.apply(self.init_weights)

        if self.hp.only_inference:
            self.model.eval()

        self.save_hyperparameters()

    def load(self):
        state_dict = load_checkpoint(self.hp.pretrained_path)
        print(f"Parameters loaded from {self.hp.pretrained_path}")
        self.load_state_dict(state_dict, strict=True)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module._fill_padding_idx_with_zero()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def configure_optimizers(self):
        optimizer_adam = optim.AdamW(
            self.parameters(), lr=self.hp.lr,
            betas=(self.hp.adam_beta1, self.hp.adam_beta2))

        # Learning rate scheduler
        num_training_steps = self.hp.training_step
        num_warmup_steps = self.hp.warmup_step
        num_flat_steps = int(self.hp.optim_flat_percent * num_training_steps)

        def lambda_lr(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            elif current_step < (num_warmup_steps + num_flat_steps):
                return 1.0
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(
                    max(1, num_training_steps - (num_warmup_steps + num_flat_steps))  # noqa
                ),
            )

        scheduler_adam = {
            "scheduler": optim.lr_scheduler.LambdaLR(
                optimizer_adam, lambda_lr),
            "interval": "step",
        }
        return [optimizer_adam], [scheduler_adam]

    def top_k_accuracy(self, y_true, y_pred_probabilities, k):
        _, sorted_indices = torch.sort(y_pred_probabilities, descending=True)

        # Get the top-k predictions
        top_k_indices = sorted_indices[:, :k]
        expanded_y_true = y_true.unsqueeze(1).expand_as(top_k_indices)

        # Check if true labels exist in top-k predictions
        hits = torch.sum(torch.eq(top_k_indices, expanded_y_true))
        accuracy = hits.item() / (len(y_true) + 1e-7)

        return accuracy

    def training_step(self, batch, batch_idx):
        # Sample training level
        rvq_level = torch.randint(
            0, min(self.hp.first_n_lvls, self.hp.n_cluster_groups),(1,)).item()

        target, chosen_tokens, _, _ = self.model(
            batch["tts_quantize_input"], rvq_level, batch["semantic_tokens"],
            batch["quantization_lengths"],
            speaker_emb=batch["speaker"],
            min_seq_length=batch["quantization_lengths"].min().item())

        # Mask targets and labels
        mask = chosen_tokens
        target = target[mask]

        labels = batch["tts_quantize_input"][:, :, rvq_level]
        labels = labels[mask]

        loss = self.cross_entropy(target, labels)
        acc = (target.argmax(-1) == labels).float().mean()
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, prog_bar=True)
        self.log(
            f"train/acc_lvl_{rvq_level}", acc, on_step=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        speaker_emb = batch["speaker"]
        acoustic_tokens = batch["tts_quantize_input"]
        semantic_tokens = batch["semantic_tokens"]

        if self.hp.only_inference:
            self.inference(
                acoustic_tokens, semantic_tokens, self.hp.first_n_lvls)
        else:
            rvq_level = torch.randint(
                0, min(self.hp.first_n_lvls, self.hp.n_cluster_groups),(1,)
            ).item()

            # FIXME: edge case
            if len(semantic_tokens.shape) == 3:
                semantic_tokens = rearrange(semantic_tokens, "B 1 T -> B T")

            target, chosen_tokens, _, _ = self.model(
                acoustic_tokens, rvq_level, semantic_tokens,
                torch.tensor([acoustic_tokens.shape[1]]).to(self.device),
                speaker_emb=speaker_emb, 
                min_seq_length=acoustic_tokens.shape[1]
            )

            target = target[chosen_tokens]
            labels = acoustic_tokens[:, :, rvq_level][chosen_tokens]
            loss = self.cross_entropy(target, labels)
            
            acc = (target.argmax(-1) == labels).float().mean()
            acc_5 = self.top_k_accuracy(labels, target, 5)

            self.log(
                f"val/dataset_{dataloader_idx}/loss",
                loss,
                on_epoch=True,
                logger=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"val/dataset_{dataloader_idx}/acc_lvl",
                acc,
                on_epoch=True,
                logger=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"val/dataset_{dataloader_idx}/acc_lvl_{rvq_level}",
                acc,
                on_epoch=True,
                logger=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"val/dataset_{dataloader_idx}/acc_top_5",
                acc_5,
                on_epoch=True,
                logger=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"val/dataset_{dataloader_idx}/acc_top_5_lvl_{rvq_level}",
                acc_5,
                on_epoch=True,
                logger=True,
                add_dataloader_idx=False,
            )

    def compute_stats(self, logits, labels, mask_ratio=0, rvq_level=0):
        acc = (logits.argmax(-1) == labels).float().mean()
        acc_5 = self.top_k_accuracy(labels, logits, 5)
        acc_10 = self.top_k_accuracy(labels, logits, 10)

        idx = torch.randperm(logits.shape[0])
        logits_shuffled = logits[idx]
        random = self.top_k_accuracy(labels, logits_shuffled, 10)
        print(f"Mask ratio: {mask_ratio}, Level {rvq_level}: acc {acc},"
              f"acc 5 {acc_5}, acc 10 {acc_10}, quasi random {random}")


class TTSConformer(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.padding_id = self.hp.n_codes

        additional_codes = [c.PAD, c.SPKR_1, c.SPKR_2]

        self.embedding = nn.ModuleList(
            [
                nn.Embedding(
                    self.hp.n_codes + len(additional_codes),
                    self.hp.hidden_size,
                    padding_idx=self.padding_id)
                for _ in range(self.hp.n_cluster_groups)
            ]
        )

        # Additional modules
        self.semantic_embedding = nn.Embedding(
            self.hp.n_semantic_codes + len(additional_codes),
            self.hp.hidden_size,
            padding_idx=self.padding_id)

        if self.hp.use_spkr_emb:
            self.spkr_linear = nn.Linear(c.SPKR_EMB_SIZE, self.hp.hidden_size)

        self.conformer = Conformer(
            dim=self.hp.hidden_size,
            num_layers=self.hp.enc_nlayers,
            heads=self.hp.nheads,
            dim_head=64,
            ff_mult=4,  # 512*4=2048
            conv_expansion_factor=2,
            conv_kernel_size=self.hp.depthwise_conv_kernel_size,
            attn_dropout=self.hp.dropout,
            ff_dropout=self.hp.dropout,
            conv_dropout=self.hp.dropout,
            attn_flash=True,
            t5_rel_pos_bias=False
        )

        self.heads = nn.ModuleList(
            [
                nn.Linear(
                    self.hp.hidden_size,
                    self.hp.n_codes + len(additional_codes)
                ) 
                for _ in range(self.hp.n_cluster_groups)
            ]
        )

    def build_mask_from_lengths(self, length, max_len=None):
        max_len = max_len or length.max().item()
        mask = torch.arange(
            max_len, device=length.device)[None, :] >= length[:, None]
        return mask.bool()

    @torch.no_grad()
    def create_mask(
            self, B, T, lengths, mask_ratio=None, start_t=None, 
            min_seq_length=None
    ):
        # 1. Define the random length of condition tokens given the shortest
        # audio in the batch
        if start_t is None:
            start_t = torch.randint(1, min_seq_length - 1, (1,)).item()

        # 2. Mask other tokens - sample different masking levels per
        if mask_ratio is None:
            ratio = torch.rand(1).item()
            mask_ratio = masking_logic.schedule(ratio)

        # Create a random tensor with values between 0 and 1
        random_tensor = torch.rand(
            (B, T - start_t), dtype=torch.float).to(self.device)
        # Create a mask where values less than p are set to True
        initial_mask = random_tensor < mask_ratio
        length_mask = self.build_mask_from_lengths(
            lengths - start_t, T - start_t)
        # we can't pick up tokens past token lengths
        initial_mask = torch.logical_and(initial_mask, ~length_mask)

        # Constrain ratio to always include some samples
        # If all are False let's pick up at least one:
        if torch.sum(initial_mask) == 0:
            choose_steps = torch.randint(low=0, high=(T - start_t), size=(B,))
            initial_mask[torch.arange(B), choose_steps] = torch.tensor(
                True, device=self.device)

        # 3. Add condition tokens containing information
        acoustic_token_mask = torch.cat(
            (torch.full((B, start_t), False, device=self.device), initial_mask),  # noqa
            1
        )

        return acoustic_token_mask, start_t, mask_ratio

    def process_input(
            self, data, lengths, rvq_level, min_seq_length=None,
            mask_ratio=None, start_t=None, acoustic_token_mask=None
    ):
        """
            data: (B,  T, code_level, D)
            rvq_level: int
        """
        B = data.size(0)
        T = data.size(1)
        level_data = data[:, :, rvq_level, :]  # [B, T, C, D] -> [B, T, D]

        # Choose acoustic tokens to mask
        if acoustic_token_mask is None:
            acoustic_token_mask, start_t, mask_ratio = self.create_mask(
                B, T, lengths, mask_ratio=mask_ratio, start_t=start_t,
                min_seq_length=min_seq_length)
            # Remove code information from chosen tokens
            level_data[acoustic_token_mask, :] = 0

        # Embed only lower rvq_level
        lower_code_data = data[:, :, :rvq_level, :].sum(dim=2)

        # Combine with chosen tokens at rvq_level.
        # Note: all tokens at rvq_level+1: will be discarded.
        summed_data = torch.add(lower_code_data, level_data)

        return summed_data, acoustic_token_mask, mask_ratio, start_t

    def forward(
        self, x, code_level, semantic_tokens, lengths,
        speaker_emb=None, min_seq_length=10, mask_ratio=None, start_t=None,
        acoustic_token_mask=None
    ):
        # FIXME: parallelize this
        batch = []
        for lvl, embed in enumerate(self.embedding[:(code_level + 1)]):
            batch.append(embed(x[:, :, lvl]))  # [B T D]

        x = torch.stack(batch, dim=2)  # [B T C D]
        x, acoustic_token_mask, mask_ratio, start_t = self.process_input(
            x, lengths, code_level, min_seq_length=min_seq_length,
            mask_ratio=mask_ratio, start_t=start_t,
            acoustic_token_mask=acoustic_token_mask
        )

        # Add phoneme embeddings
        # Cross attention for all tokens?

        # Add semantic tokens
        # HACK ME
        semantic_emb = self.semantic_embedding(semantic_tokens)
        x = torch.add(x, semantic_emb)
        # FIXME pfb30

        # Merge different modalities
        if self.hp.use_spkr_emb:
            spkr_emb = F.normalize(speaker_emb, dim=-1)
            spkr_emb = self.spkr_linear(
                F.dropout(spkr_emb, self.hp.speaker_embed_dropout)
            )
            x = torch.add(x, spkr_emb)

        output_frames = self.conformer(x, None)

        x = self.heads[code_level](output_frames)

        return x, acoustic_token_mask, mask_ratio, start_t

    @torch.no_grad()
    def inference(
        self, codes, semantic_tokens,
        length: torch.LongTensor, rvq_levels=7,
        mask_ratio=0.99, maskgit_inference=True,
        start_t: Union[torch.LongTensor, None] = None,
        speaker_emb=None, steps=16
    ):
        # Use half of the recording for the conditioning
        if start_t is None:
            start_t = torch.tensor(int((codes.shape[1]) / 2)).long()

        start_t = start_t.item()

        for rvq_level in range(rvq_levels):
            original_codes = torch.clone(codes)
            if rvq_level == 0 and maskgit_inference:
                codes = self.multi_step_inference(
                    original_codes, semantic_tokens, length,
                    start_t=start_t, vamp_filtering=False, 
                    speaker_emb=speaker_emb, steps=16
                )
            else:
                codes = self.one_step_inference(
                    original_codes, semantic_tokens, length,
                    code_level=rvq_level,
                    mask_ratio=mask_ratio, start_t=start_t, 
                    speaker_emb=speaker_emb
                )

            codes = rearrange(codes, 'T C -> 1 T C')

        # Remove any padding left
        codes = rearrange(codes, '1 T C -> 1 C T')
        codes = torch.where(codes >= self.hp.n_codes, 0, codes)
        acoustic_tokens = codes
        semantic_tokens = rearrange(semantic_tokens, 'b c -> b 1 c')
        semantic_tokens = torch.where(
            semantic_tokens >= self.hp.n_codes, 0, semantic_tokens)
        codes = torch.cat([semantic_tokens, acoustic_tokens], dim=1)

        return codes

    @torch.no_grad()
    def one_step_inference(
        self, original_codes, semantic_tokens, lengths, code_level=0, 
        mask_ratio=0.99, start_t=0, inference_setup="argmax", speaker_emb=None
    ):  
        codes = torch.clone(original_codes)
        logits, _, _, _ = self.forward(
            codes, code_level, semantic_tokens, lengths, 
            mask_ratio=mask_ratio, start_t=start_t,
            speaker_emb=speaker_emb, acoustic_token_mask=False)

        if inference_setup == "argmax":
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_indeces = torch.argmax(probs, dim=-1)

        if inference_setup == "sampling":
            top_indeces = torch.distributions.Categorical(
                logits=logits).sample()

        codes = rearrange(codes, '1 T C -> T C')
        codes[start_t:, code_level] = top_indeces[0, start_t:]

        return codes

    @torch.no_grad()
    def multi_step_inference(
        self, original_codes, semantic_tokens, lengths,
        start_t: torch.LongTensor=None,
        choice_temperature=1.0, start_iter=0,
        steps=16, vamp_filtering=False, speaker_emb=None
    ):
        codes = torch.clone(original_codes)
        code_level = 0
        _, seq_len, _ = original_codes.shape
        mask_token_id = self.padding_id

        # Get true codes for the prompt
        prompt_mask = codes[:, :start_t, code_level]
    
        # Fill up rest with masks
        mask = torch.full(
            (1, seq_len - start_t), mask_token_id, device=self.device) 
        inputs = torch.cat((prompt_mask, mask), 1)

        num_mask_tokens_at_start = torch.sum(inputs == mask_token_id, axis=-1)

        # Initializes state
        state = state_init(inputs, steps, start_iter=start_iter)

        def loop_cond_fn(state):
            """Beam search loop termination condition."""
            not_at_end = (state.cur_index < steps)
            return not_at_end

        while loop_cond_fn(state):
            """Beam search loop state update function."""
            step = state.cur_index
            # Current input ids: [batch_size, seq_length].
            cur_ids = state.cur_seqs

            # Calls model on current seqs to get next-iteration seqs.
            with torch.no_grad():
                logits, _, _, _ = self.forward(
                    rearrange(inputs, 'B T -> B T 1'), 
                    code_level,
                    semantic_tokens, lengths, 
                    acoustic_token_mask=False,
                    speaker_emb=speaker_emb)

            # Samples the ids using categorical sampling:
            if vamp_filtering:
                typical_mass = 0.2
                typical_min_tokens = 1
                top_p = None
                sample_cutoff = 0.5
                typical_filtering = False
                sampled_ids, selected_probs = sample_from_logits(
                    logits, sample=((step / steps) <= sample_cutoff),
                    temperature=choice_temperature,
                    typical_filtering=typical_filtering,
                    typical_mass=typical_mass,
                    typical_min_tokens=typical_min_tokens,
                    top_k=None, top_p=top_p, return_probs=True,
                )
            else:
                sampled_ids = torch.distributions.Categorical(
                    logits=logits).sample()

            # Just updates the masked tokens.
            unknown_map = (cur_ids == mask_token_id)
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)
            # Defines the mask ratio for the next round. The number to mask out
            # is determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1. * (step + 1) / steps
            mask_ratio = masking_logic.schedule(ratio)

            # Updates final seqs with the current sampled_ids.
            final_seqs = torch.clone(state.final_seqs)
            final_seqs[:, step, :] = sampled_ids
            # Computes the probabilities of each selected tokens.
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Extract the probabilities of sampled ids
            selected_probs = torch.squeeze(
                torch.take_along_dim(
                    probs, torch.unsqueeze(sampled_ids, -1) , -1),
                -1
            )

            # Ignores the tokens given in the input 
            # by overwriting their confidence.
            selected_probs = torch.where(
                unknown_map, selected_probs, torch.inf)
            # Gets mask lens for each sample in the 
            # batch according to the mask ratio.
            num_to_mask = torch.unsqueeze(
                torch.floor(num_mask_tokens_at_start * mask_ratio), 1)

            # Keeps at least one of prediction in this 
            # round and also masks out at least
            # one and for the next iteration
            num_to_mask = torch.maximum(
                torch.tensor(1), 
                torch.minimum(
                    torch.sum(unknown_map, dim=-1, keepdim=True) - 1,
                    num_to_mask)
            )
            # Adds noise for randomness
            masking = mask_by_random_topk(
                num_to_mask, selected_probs, choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            sampled_ids = torch.where(masking, mask_token_id, sampled_ids)  

            state = State(
                cur_index=state.cur_index + 1,
                cur_seqs=sampled_ids,
                final_seqs=final_seqs
            )

        codes = torch.clone(original_codes)
        codes = rearrange(codes, '1 T C -> T C')
        codes[:, 0] = state.final_seqs[0][-1]

        return codes
