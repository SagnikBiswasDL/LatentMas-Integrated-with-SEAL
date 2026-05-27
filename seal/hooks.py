"""Model-agnostic SEAL steering via forward hooks."""

from __future__ import annotations

from typing import List, Optional

import torch


class SealSteering:
    def __init__(
        self,
        *,
        vector_path: str,
        layer: int,
        coef: float,
        mode: str,
        device: torch.device,
    ) -> None:
        self.layer = layer
        self.coef = coef
        self.mode = mode.lower()
        self.device = device
        self.steer_vec = torch.load(vector_path, map_location=device, weights_only=True)
        self.latent_active = False
        self.text_active = False
        self.text_mask: Optional[torch.Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, model: torch.nn.Module) -> None:
        layers = self._get_layers(model)
        if self.layer < 0 or self.layer >= len(layers):
            raise ValueError(f"Invalid steering layer {self.layer}; model has {len(layers)} layers.")

        def hook(_module, _inputs, output):
            if not (self.latent_active or self.text_active):
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3:
                return output

            updated = hidden.clone()
            vec = self.steer_vec.to(device=updated.device, dtype=updated.dtype)

            if self.latent_active and self.mode in {"latent", "both"}:
                updated[:, -1, :] = updated[:, -1, :] + self.coef * vec
            elif self.text_active and self.mode in {"text", "both"} and self.text_mask is not None:
                mask = self.text_mask.to(device=updated.device)
                if mask.any():
                    updated[mask, -1, :] = updated[mask, -1, :] + self.coef * vec

            if isinstance(output, tuple):
                return (updated,) + output[1:]
            return updated

        self._handles.append(layers[self.layer].register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def set_latent(self, active: bool) -> None:
        self.latent_active = active and self.mode in {"latent", "both"}
        if self.latent_active:
            self.text_active = False
            self.text_mask = None

    def set_text_mask(self, mask: Optional[torch.Tensor]) -> None:
        if self.mode not in {"text", "both"}:
            self.text_active = False
            self.text_mask = None
            return
        self.text_active = mask is not None and bool(mask.any())
        self.text_mask = mask
        if self.text_active:
            self.latent_active = False

    @staticmethod
    def _get_layers(model: torch.nn.Module):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError("Could not locate transformer layers for SEAL hook.")
