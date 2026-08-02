#!/usr/bin/env python
"""Flow-matching chunk policy (PLAN.md section 2.3).

Rectified-flow head grafted onto the vendored ACT transformer (act/modeling_act.py),
the standard conversion used by pi0 / LeRobot-pi0 / X-IL / Much-Ado:

- NO CVAE: the latent z, VAE encoder and KL term are deleted — the flow noise sample
  x0 replaces z as the stochasticity source.
- Obs encoder identical to ACT's (state + env_state token projections). The unchanged
  ``ACTEncoder`` and sinusoidal-embedding helper are imported from modeling_act; the
  decoder classes are COPIED here (copy-and-modify per PLAN 2.3, not inheritance) with
  ONE modification: the self-attention takes a ``key_padding_mask`` so padded /
  loss-censored chunk positions cannot influence the velocity predicted at valid
  positions (keeps the ``action_is_pad`` loss censor exact — without it, garbage x_tau
  values at masked positions would leak into valid tokens through attention).
- Decoder input tokens = linear projection of the noisy action chunk x_tau (B, S, A)
  + fixed sinusoidal chunk-position embeddings; bidirectional self-attention over the
  action tokens, cross-attention to the encoder obs tokens.
- Time conditioning choice: sinusoidal embedding of tau -> 2-layer MLP -> ADDED TO
  EVERY DECODER INPUT TOKEN. This is the simpler of the two PLAN 2.3 variants (the
  AdaLN alternative would require rewriting the vendored decoder's LayerNorms) and
  fits the vendored decoder structure without touching its layer internals.
- Output head: linear to action_dim per token = predicted velocity v_hat.

Training (rectified flow): x_tau = (1 - tau) * x0 + tau * x1 with x0 ~ N(0, I) and
tau ~ U(0, 1) per sample; target v = x1 - x0; masked elementwise MSE (the
``action_is_pad`` censor multiplies in exactly as it did for ACT's L1).

Inference: Euler integration x <- x + v_hat(x, tau) * dtau from tau=0 to tau=1 over
``num_inference_steps`` (default 10, PLAN 2.3 / pi0 default).

Same external interface as the vendored ACTPolicy: ``forward(batch) -> (loss,
loss_dict)`` and ``predict_action_chunk(batch) -> (B, chunk_size, action_dim)``
returning NORMALIZED actions — normalization lives outside the policy
(act/normalize.py), same contract as ACT.
"""

import math
from itertools import chain

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_act import ACTConfig
from .modeling_act import (
    ACTION,
    OBS_ENV_STATE,
    OBS_STATE,
    ACTEncoder,
    create_sinusoidal_pos_embedding,
    get_activation_fn,
)


class FlowMatchingPolicy(nn.Module):
    """Rectified-flow action-chunk policy on the ACT transformer backbone."""

    config_class = ACTConfig
    name = "flow"

    def __init__(self, config: ACTConfig, **kwargs):
        super().__init__()
        config.validate_features()
        self.config = config
        # Extra knob not in the vendored ACTConfig (which we must not modify):
        # getattr-with-default pattern, set as a plain attribute by train_flow.py.
        self.num_inference_steps = int(getattr(config, "num_inference_steps", 10))
        self.model = FlowTransformer(config)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Rectified-flow training loss on a (normalized) batch."""
        x1 = batch[ACTION]  # (B, chunk_size, action_dim)
        x0 = torch.randn_like(x1)
        tau = torch.rand(x1.shape[0], 1, 1, device=x1.device, dtype=x1.dtype)
        x_tau = (1 - tau) * x0 + tau * x1
        v_target = x1 - x0

        pad = batch["action_is_pad"]  # (B, chunk_size) bool
        # Never mask EVERY key of a sample (an all-masked attention row yields NaNs);
        # all-pad samples contribute nothing to the loss anyway.
        key_padding_mask = pad & ~pad.all(dim=1, keepdim=True)

        v_hat = self.model(batch, x_tau, tau.reshape(-1), key_padding_mask=key_padding_mask)

        sq_err = F.mse_loss(v_hat, v_target, reduction="none")
        valid_mask = ~pad.unsqueeze(-1)
        num_valid = valid_mask.sum() * sq_err.shape[-1]
        mse_loss = (sq_err * valid_mask).sum() / num_valid.clamp_min(1)

        return mse_loss, {"mse": mse_loss.item()}

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        generator: torch.Generator | None = None,
        x0: Tensor | None = None,
    ) -> Tensor:
        """Euler-integrate the flow from x0 ~ N(0, I) at tau=0 to tau=1.

        ``generator`` (optional) seeds x0 for deterministic sampling (residual-RL
        stages need reproducible chunks). ``x0`` (optional) fixes the noise outright:
        (chunk_size, action_dim) is shared across the batch — the chunk becomes a
        deterministic function of the observation (PLAN section 6: ResiP-style
        deterministic base for residual RL) — while (B, chunk_size, action_dim) gives
        each batch element its own noise (EXP07 x0-steering: per-env steered x0).
        Returns NORMALIZED actions.
        """
        self.eval()

        ref = batch[OBS_STATE] if OBS_STATE in batch else batch[OBS_ENV_STATE]
        batch_size, device = ref.shape[0], ref.device
        shape = (batch_size, self.config.chunk_size, self.config.action_feature.shape[0])
        if x0 is not None:
            x0 = x0.to(device)
            x = x0.clone() if x0.dim() == 3 else x0.unsqueeze(0).expand(batch_size, -1, -1).clone()
        elif generator is not None:
            # torch.randn requires the generator's device; sample there, then move.
            x = torch.randn(shape, generator=generator, device=generator.device).to(device)
        else:
            x = torch.randn(shape, device=device)

        # Obs tokens are tau-independent: encode once, reuse across Euler steps.
        encoder_out, encoder_pos_embed = self.model.encode(batch)

        dt = 1.0 / self.num_inference_steps
        for i in range(self.num_inference_steps):
            tau = torch.full((batch_size,), i * dt, device=device)
            v_hat = self.model.velocity(encoder_out, encoder_pos_embed, x, tau)
            x = x + v_hat * dt
        return x


class FlowTransformer(nn.Module):
    """ACT's encoder-decoder transformer with the CVAE removed and the decoder queries
    replaced by projected noisy-chunk tokens (velocity prediction network)."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        if config.image_features:
            raise NotImplementedError("FlowMatchingPolicy is state-only (no image backbone).")

        # Transformer encoder over obs tokens [(robot_state), (env_state)] — same
        # structure as ACT minus the latent token.
        self.encoder = ACTEncoder(config)
        self.decoder = FlowDecoder(config)
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], config.dim_model
            )
        n_1d_tokens = int(bool(config.robot_state_feature)) + int(bool(config.env_state_feature))
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        # Noisy-chunk input projection + fixed sinusoidal chunk-position embeddings
        # (replaces ACT's learned DETR-style decoder queries).
        self.action_in_proj = nn.Linear(config.action_feature.shape[0], config.dim_model)
        self.register_buffer(
            "decoder_pos_embed",
            create_sinusoidal_pos_embedding(config.chunk_size, config.dim_model).unsqueeze(1),
        )  # (chunk_size, 1, dim_model)

        # tau in [0, 1] -> sinusoidal embedding -> 2-layer MLP.
        self.time_mlp = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.SiLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )

        # Final velocity head on the decoder output.
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in ACT."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Obs -> encoder output tokens. Returns (encoder_out (S_obs, B, D),
        encoder_pos_embed (S_obs, 1, D)); tau-independent, computed once per chunk."""
        tokens = []
        if self.config.robot_state_feature:
            tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        encoder_in_tokens = torch.stack(tokens, dim=0)  # (S_obs, B, D)
        encoder_in_pos_embed = self.encoder_1d_feature_pos_embed.weight.unsqueeze(1)  # (S_obs, 1, D)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        return encoder_out, encoder_in_pos_embed

    def velocity(
        self,
        encoder_out: Tensor,
        encoder_pos_embed: Tensor,
        x_tau: Tensor,
        tau: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict v_hat (B, chunk_size, action_dim) for noisy chunk x_tau at time tau (B,)."""
        decoder_in = self.action_in_proj(x_tau).transpose(0, 1)  # (chunk_size, B, D)
        # Time conditioning: added to every decoder input token (see module docstring).
        time_embed = self.time_mlp(create_sinusoidal_time_embedding(tau, self.config.dim_model))
        decoder_in = decoder_in + time_embed.unsqueeze(0)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            decoder_pos_embed=self.decoder_pos_embed,
            encoder_pos_embed=encoder_pos_embed,
            key_padding_mask=key_padding_mask,
        )
        return self.action_head(decoder_out.transpose(0, 1))  # (B, chunk_size, action_dim)

    def forward(
        self,
        batch: dict[str, Tensor],
        x_tau: Tensor,
        tau: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        encoder_out, encoder_pos_embed = self.encode(batch)
        return self.velocity(encoder_out, encoder_pos_embed, x_tau, tau, key_padding_mask)


def create_sinusoidal_time_embedding(tau: Tensor, dimension: int) -> Tensor:
    """(B,) tau in [0, 1] -> (B, dimension) sinusoidal embedding.

    Same 10000^(-2i/D) frequency progression as create_sinusoidal_pos_embedding; tau is
    scaled by 1000 (diffusers/pi0 convention) so the unit interval spans a useful range
    of phases instead of collapsing into the near-zero regime of every sinusoid.
    """
    half = dimension // 2
    freqs = torch.exp(
        torch.arange(half, device=tau.device, dtype=torch.float32) * (-math.log(10000.0) / half)
    )
    args = tau.float().unsqueeze(-1) * 1000.0 * freqs  # (B, half)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class FlowDecoder(nn.Module):
    """COPIED from modeling_act.ACTDecoder; modification: forwards ``key_padding_mask``
    to each layer (padded/censored action tokens excluded from self-attention keys)."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([FlowDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
                key_padding_mask=key_padding_mask,
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class FlowDecoderLayer(nn.Module):
    """COPIED from modeling_act.ACTDecoderLayer; modification: the (bidirectional)
    self-attention over action tokens takes ``key_padding_mask`` so masked chunk
    positions cannot influence valid ones (exactness of the action_is_pad censor)."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            encoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            decoder_pos_embed: (DS, 1, C) positional embedding for the queries (from the decoder).
            key_padding_mask: (B, DS) bool, True = exclude this action token from the
                self-attention keys (padded / loss-censored chunk positions).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x
