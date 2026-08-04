#!/usr/bin/env python
"""Vision flow-matching chunk policy for the slot task (VISION_PLAN section 1).

Ported VERBATIM from ``act/modeling_flow_vision.py`` (eva_bc EXP08 step 3) apart from this
docstring: it depends only on configuration_act / modeling_act / modeling_flow, and
``slot_act/modeling_flow.py`` is byte-identical to ``act/modeling_flow.py`` (checked). Keeping it
a verbatim port is deliberate -- any future divergence should show up as a diff, not as drift.

FlowMatchingPolicy (slot_act/modeling_flow.py) with the env_state token replaced by camera
tokens, using the vendored ACT image path (act/modeling_act.py): torchvision ResNet
backbone -> layer4 feature map -> 1x1-conv projection -> one encoder token per spatial
cell with 2D sinusoidal position embeddings. At 160x90 input, resnet18 gives 5x3 = 15
tokens per camera; encoder sequence = [robot_state, wrist 15, workspace 15] = 31 tokens.

Per the modeling_flow convention, the transformer is COPIED from FlowTransformer and
modified (not inherited): __init__ swaps the env_state projection for the backbone
stack, and encode() appends the camera tokens. FlowDecoder, the time embedding and the
flow train/inference math are imported unchanged.

Inputs (VISION_PLAN section 0 no-privileged-info contract). The slot task's privileged obs is
34-D -- ``jp8 | jv8 | block_pose7 | slot_frame4 | last_action7`` -- and the student gets the
first 16 and the last 7. That is 23-D, the same width as the pick-place student by coincidence
(its privileged obs is 41-D), not by construction:
    observation.state          (B, 23) float32   jp8 + jv8 + last_action7
    observation.images.wrist   (B, 3, 90, 160) float32 in [0, 1]
    observation.images.workspace  same
ImageNet mean/std normalization is applied INSIDE encode() (fixed constants, matching
the pretrained backbone init) -- shards stay uint8, the outside MeanStdNormalizer
touches only state/action keys.
"""

from itertools import chain

import einops
import torch
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .configuration_act import ACTConfig
from .modeling_act import (
    OBS_IMAGES,
    OBS_STATE,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    create_sinusoidal_pos_embedding,
)
from .modeling_flow import FlowDecoder, FlowMatchingPolicy, create_sinusoidal_time_embedding

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class FlowMatchingVisionPolicy(FlowMatchingPolicy):
    """Rectified-flow chunk policy on state + camera tokens. Same external interface
    as FlowMatchingPolicy (forward -> (loss, dict), predict_action_chunk -> chunks in
    NORMALIZED action space); the flow math is inherited unchanged."""

    name = "flow_vision"

    def __init__(self, config: ACTConfig, **kwargs):
        # bypass FlowMatchingPolicy.__init__ (it builds the state-only transformer)
        nn.Module.__init__(self)
        config.validate_features()
        self.config = config
        self.num_inference_steps = int(getattr(config, "num_inference_steps", 10))
        self.model = FlowVisionTransformer(config)

    def _stack_images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if OBS_IMAGES not in batch:
            batch = dict(batch)  # shallow copy; don't mutate the caller's dict
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return super().forward(self._stack_images(batch))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return super().predict_action_chunk(self._stack_images(batch), **kwargs)


class FlowVisionTransformer(nn.Module):
    """COPIED from modeling_flow.FlowTransformer; modifications: the env_state input
    projection is replaced by the ACT image backbone stack, and encode() appends the
    per-camera spatial tokens. velocity()/forward() are identical."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        if not config.image_features:
            raise ValueError("FlowMatchingVisionPolicy requires image features; use FlowMatchingPolicy for state-only.")
        if config.env_state_feature:
            raise ValueError("env_state is privileged -- forbidden here (VISION_PLAN section 0).")

        self.encoder = ACTEncoder(config)
        self.decoder = FlowDecoder(config)
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        n_1d_tokens = int(bool(config.robot_state_feature))
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        # Image backbone stack, same construction as modeling_act.ACT.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
        self.encoder_img_feat_input_proj = nn.Conv2d(
            backbone_model.fc.in_features, config.dim_model, kernel_size=1
        )
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
        self.register_buffer("imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        # Noisy-chunk projection + fixed chunk-position embeddings (as FlowTransformer).
        self.action_in_proj = nn.Linear(config.action_feature.shape[0], config.dim_model)
        self.register_buffer(
            "decoder_pos_embed",
            create_sinusoidal_pos_embedding(config.chunk_size, config.dim_model).unsqueeze(1),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_model),
            nn.SiLU(),
            nn.Linear(config.dim_model, config.dim_model),
        )
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier init of the transformer (NOT the pretrained backbone)."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Obs -> encoder tokens: [robot_state] + per-camera spatial tokens."""
        encoder_in_tokens = []
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))

        # Camera features -> tokens (modeling_act pattern: per-token lists; the 2D pos
        # embedding has batch dim 1 and broadcasts. Images arrive float in [0, 1]).
        for img in batch[OBS_IMAGES]:
            img = (img - self.imagenet_mean) / self.imagenet_std
            cam_features = self.backbone(img)["feature_map"]
            cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = self.encoder_img_feat_input_proj(cam_features)
            cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
            cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)  # (S, B, D)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)  # (S, 1, D)

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
        decoder_in = self.action_in_proj(x_tau).transpose(0, 1)
        time_embed = self.time_mlp(create_sinusoidal_time_embedding(tau, self.config.dim_model))
        decoder_in = decoder_in + time_embed.unsqueeze(0)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            decoder_pos_embed=self.decoder_pos_embed,
            encoder_pos_embed=encoder_pos_embed,
            key_padding_mask=key_padding_mask,
        )
        return self.action_head(decoder_out.transpose(0, 1))

    def forward(
        self,
        batch: dict[str, Tensor],
        x_tau: Tensor,
        tau: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        encoder_out, encoder_pos_embed = self.encode(batch)
        return self.velocity(encoder_out, encoder_pos_embed, x_tau, tau, key_padding_mask)
