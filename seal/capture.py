"""Forward-hook capture of last-token hidden states at a single layer.

Used to extract a *native* LatentMAS steering direction: we record the
`hidden[:, -1, :]` vector at the chosen layer for every Judger decode step, so
the captured states live in exactly the same place the SEAL hook later injects
(`seal/hooks.py` registers its hook on the same `model.model.layers[layer]`).
That keeps extraction and injection on the identical module — no off-by-one
between where the direction is measured and where it is added.
"""

from __future__ import annotations

from typing import List, Optional

import torch


class HiddenCapture:
    def __init__(self, layer: int) -> None:
        self.layer = layer
        self.active = False
        self.buffer: List[torch.Tensor] = []
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def register(self, model: torch.nn.Module) -> None:
        layers = self._get_layers(model)
        if self.layer < 0 or self.layer >= len(layers):
            raise ValueError(
                f"Invalid capture layer {self.layer}; model has {len(layers)} layers."
            )

        def hook(_module, _inputs, output):
            if not self.active:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3:
                return output
            # [B, D] for the most recent position; store on CPU in fp32.
            self.buffer.append(hidden[:, -1, :].detach().to(torch.float32).cpu())
            return output

        self._handle = layers[self.layer].register_forward_hook(hook)

    def reset(self) -> None:
        self.buffer = []

    def stacked(self) -> torch.Tensor:
        """Return captured steps as [T, B, D] (empty tensor if nothing captured)."""
        if not self.buffer:
            return torch.empty(0)
        return torch.stack(self.buffer, dim=0)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @staticmethod
    def _get_layers(model: torch.nn.Module):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError("Could not locate transformer layers for hidden capture.")
