# Pilot results (GSM8K, 2026-05-28)

| File | Description |
|------|-------------|
| [`summary.json`](summary.json) | Headline comparison + delta |
| [`baseline_summary.json`](baseline_summary.json) | LatentMAS without SEAL |
| [`seal_summary.json`](seal_summary.json) | LatentMAS + SEAL |
| [`seal_vector_extraction.json`](seal_vector_extraction.json) | Vector build metadata |

Full verbose run logs (per-problem traces) were written on RunPod to `results/pilot/latent_mas.json` and `latent_mas_seal.json` via `tee`; those files are not in git due to size. See [`docs/pilot_report.md`](../../docs/pilot_report.md) for analysis.
