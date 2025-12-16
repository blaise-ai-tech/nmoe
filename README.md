NMoE
====

NMoE is an opinionated Mixture-of-Experts training library hard-targeted to
NVIDIA Blackwell B200 (`sm_100a`) with RDEP expert parallelism.

This repository is **container-first**: the supported way to build and run is
via the Dockerfiles in `docker/`.

What this repo is not:
- Not a general-purpose distributed training stack.
- Not a tensor-parallel framework.
- Not a "works everywhere" codebase (off-target GPUs are intentionally rejected).

Quickstart (Docker)
-------------------

Build the base image:

    docker build -f docker/Dockerfile.base -t nmoe:base .

Build the single-node training image:

    docker build -f docker/Dockerfile.train --build-arg BASE_IMAGE=nmoe:base -t nmoe:train .

Build the multi-node (patched NVSHMEM) image:

    docker build -f docker/Dockerfile.dist --build-arg BASE_IMAGE=nmoe:base -t nmoe:dist .

Run training (example):

    python -m nmoe.train configs/moonlet.toml

Data inputs
-----------

Training consumes token shards (`.npy`) and supports two distinct workflows:

- **Direct shards (research / small runs)**: set `data_path` to a directory containing `.npy` shards (no flow TOMLs).
- **Flows (production / large runs)**: set `flow_mode`, `mixture_toml`, and `flow_profiles_toml` for deterministic dataset mixing and exact resume.

Dataset cleaning / augmentation (HYDRA grading, K2 rephrasing) is a separate preprocessing pipeline and may run on non‑B200 hardware; training itself hard-targets B200.

HYDRA judge head artifact
-------------------------

This repository includes `nmoe/data/hydra_judge.pt`, a judge head checkpoint
intended to be loaded on top of a frozen `gpt-oss-20B` backbone. The backbone
weights are not included here.

See `nmoe/data/HYDRA_JUDGE_HEAD.md`.

Licensing
---------

The repository is licensed under Apache-2.0; see `LICENSE` and `NOTICE`.
Some files include third-party work under other licenses; see
`THIRD_PARTY_NOTICES.md`.
