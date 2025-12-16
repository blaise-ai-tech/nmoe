# HYDRA Judge Head (`hydra_judge.pt`)

This repository ships a small PyTorch `state_dict` checkpoint:

- `nmoe/data/hydra_judge.pt`

## What it is

`hydra_judge.pt` contains weights for the HYDRA "judge head" used by the data
pipeline to grade documents. The head is intended to be loaded on top of a
*frozen* `gpt-oss-20B` backbone (the backbone weights are **not** included in
this repository).

## Licensing

This file is distributed as part of this repository and is intended to be
covered by the repository license (Apache-2.0), unless otherwise noted.

Using the judge head requires separately obtaining the `gpt-oss-20B` backbone
and complying with the backbone's license and terms.

## Security note

This is a PyTorch checkpoint. Do not `torch.load` untrusted checkpoints.
