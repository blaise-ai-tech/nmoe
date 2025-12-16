#!/usr/bin/env python3
"""
Production data preprocessing pipeline: HYDRA quality grading + K2 rephrasing.

Usage:
    python pipeline.py \
        --dataset fineweb \
        --max-docs 100000 \
        --output /data/fineweb_edu \
        --checkpoint /data/checkpoints/gpt-oss-20b \
        --heads-dir /workspace/nmoe/nmoe/data \
        --calibration /workspace/nmoe/nmoe/data
"""
import argparse
import json
import sys
from pathlib import Path

from nmoe.log import info, warning, error


def run_pipeline(dataset_name: str, max_docs: int, output_dir: str,
                 checkpoint_path: str, heads_dir: str, calibration_dir: str,
                 num_versions: int = 10, shard_index: int = 0, num_shards: int = 1):
    """Run complete preprocessing pipeline on a dataset.

    Steps:
    1. Download documents from HuggingFace
    2. HYDRA quality grading (20B + Judge Head)
    3. Extract kept documents (quality >= threshold)
    4. K2 rephrasing (style-diverse augmentation)

    Args:
        dataset_name: Dataset identifier ('fineweb' or 'dolma')
        max_docs: Maximum documents to process
        output_dir: Output directory path
        checkpoint_path: Path to gpt-oss-20b checkpoint
        heads_dir: Directory containing hydra_judge.pt
        calibration_dir: Directory containing calibration_summary.json
        num_versions: Number of rephrased versions per document (default: 10)
        shard_index: Shard index for parallel processing (0-indexed, default: 0)
        num_shards: Total number of shards (default: 1)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Pipeline spec for deterministic resume/re-run.
    spec_path = output_path / "pipeline_spec.json"
    raw_docs_path = output_path / "raw_docs.jsonl"
    grades_dir = output_path / 'hydra_grades'
    summary_path = grades_dir / "summary.json"
    scores_path = grades_dir / "quality_scores.jsonl"
    kept_docs_path = output_path / 'kept_docs.jsonl'
    rephrased_path = output_path / 'rephrased_docs.jsonl'
    shards_dir = output_path / 'training_shards'

    if num_shards <= 0:
        error(f"num_shards must be > 0 (got {num_shards})")
        return 1
    if shard_index < 0 or shard_index >= num_shards:
        error(f"shard_index out of range: {shard_index} (num_shards={num_shards})")
        return 1

    max_docs_shard = (max_docs + num_shards - 1) // num_shards

    # Dataset configurations
    datasets = {
        'fineweb': {
            'hf_dataset': 'HuggingFaceFW/fineweb-edu',
            'hf_subset': 'sample-10BT',
            'text_field': 'text',
            # Prefer dataset-provided ids when available.
            'id_field': 'id',
        },
        'dolma': {
            'hf_dataset': 'allenai/dolma3_dolmino_mix-10B-1025',
            'hf_subset': None,
            'text_field': 'text',
            'id_field': 'id',
        }
    }

    if dataset_name not in datasets:
        error(f"Unknown dataset '{dataset_name}'")
        error(f"Available: {list(datasets.keys())}")
        return 1

    config = datasets[dataset_name]

    info(f"\n{'='*80}")
    info(f"NMOE DATA PREPROCESSING PIPELINE")
    info(f"{'='*80}")
    info(f"Dataset: {config['hf_dataset']}")
    info(f"Max docs (global): {max_docs:,}")
    info(f"Max docs (this shard): {max_docs_shard:,}")
    info(f"K2 versions: {num_versions}")
    info(f"Shard: {shard_index + 1}/{num_shards}")
    info(f"Output: {output_dir}")
    info(f"{'='*80}\n")

    # Spec: enforce deterministic resume (same args => same outputs).
    pipeline_spec = {
        "dataset_name": dataset_name,
        "hf_dataset": config["hf_dataset"],
        "hf_subset": config["hf_subset"],
        "text_field": config["text_field"],
        "id_field": config.get("id_field"),
        "max_docs_global": int(max_docs),
        "max_docs_shard": int(max_docs_shard),
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "num_versions": int(num_versions),
        "checkpoint_path": str(checkpoint_path),
        "heads_dir": str(heads_dir),
        "calibration_dir": str(calibration_dir),
    }

    if spec_path.exists():
        try:
            existing = json.loads(spec_path.read_text())
        except Exception as e:
            error(f"✗ Failed to read {spec_path}: {e}")
            return 1
        if existing != pipeline_spec:
            error("✗ Output directory contains a different pipeline_spec.json")
            error(f"  Path: {spec_path}")
            error("  Use a new output directory (recommended), or manually delete the old one.")
            return 1
    else:
        tmp_path = spec_path.with_name(spec_path.name + ".tmp")
        tmp_path.write_text(json.dumps(pipeline_spec, indent=2, sort_keys=True))
        tmp_path.replace(spec_path)

    # Step 0: Ensure checkpoint exists
    checkpoint = Path(checkpoint_path)
    lock_file = checkpoint.parent / '.checkpoint_download.lock'

    if not checkpoint.exists() or not (checkpoint / 'config.json').exists():
        if shard_index == 0:
            # Only shard 0 downloads to avoid parallel overwrites
            info(f"[Step 0/4] Downloading gpt-oss-20b checkpoint to {checkpoint_path}...")
            from huggingface_hub import snapshot_download
            import fcntl

            checkpoint.parent.mkdir(parents=True, exist_ok=True)

            # Acquire exclusive lock
            with lock_file.open('w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

                # Double-check after acquiring lock (another process may have downloaded)
                if not (checkpoint / 'config.json').exists():
                    snapshot_download(
                        repo_id='openai/gpt-oss-20b',
                        local_dir=str(checkpoint),
                        local_dir_use_symlinks=False
                    )
                    info(f"✓ Checkpoint downloaded")
                else:
                    info(f"✓ Checkpoint already available")

                # Lock released automatically on exit
        else:
            # Other shards wait for shard 0 to finish
            info(f"[Step 0/4] Waiting for shard 0 to download checkpoint...")
            import time
            max_wait = 3600  # 1 hour timeout
            waited = 0
            while not (checkpoint / 'config.json').exists() and waited < max_wait:
                time.sleep(10)
                waited += 10
                if waited % 60 == 0:
                    info(f"  Still waiting... ({waited}s elapsed)")

            if not (checkpoint / 'config.json').exists():
                error(f"✗ Timeout waiting for checkpoint download")
                return 1
            info(f"✓ Checkpoint ready")
    else:
        info(f"[Step 0/4] Using existing checkpoint at {checkpoint_path}")

    # Step 1: Download documents from HuggingFace
    if raw_docs_path.exists():
        info(f"[Step 1/4] Using existing raw docs at {raw_docs_path}")
    else:
        info(f"[Step 1/4] Downloading {max_docs_shard:,} documents from {config['hf_dataset']}...")

        from nmoe.data.sources import HuggingFaceSource

        source = HuggingFaceSource(
            dataset=config['hf_dataset'],
            split='train',
            subset=config['hf_subset'],
            text_field=config['text_field'],
            id_field=config.get('id_field'),
            streaming=True,
        )

        # Apply sharding if parallel processing
        if num_shards > 1:
            source = source.shard(num_shards=num_shards, index=shard_index)

        doc_count = 0
        with raw_docs_path.open('w', encoding='utf-8') as f:
            for doc in source:
                if doc_count >= max_docs_shard:
                    break
                if doc.text and doc.text.strip():
                    f.write(json.dumps({'text': doc.text, 'id': doc.doc_id}, ensure_ascii=False) + '\n')
                    doc_count += 1
                    if doc_count % 10000 == 0:
                        info(f"  Downloaded {doc_count:,}/{max_docs_shard:,} documents...")

        info(f"✓ Downloaded {doc_count:,} documents")

    # Step 2: HYDRA Quality Grading
    if summary_path.exists() and scores_path.exists():
        info(f"\n[Step 2/4] Using existing HYDRA grades at {grades_dir}")
    else:
        info(f"\n[Step 2/4] Running HYDRA grading (20B + Judge Head)...")

        from nmoe.data.hydra import cmd_grade
        import argparse

        grades_dir.mkdir(parents=True, exist_ok=True)

        args = argparse.Namespace(
            input_jsonl=str(raw_docs_path),
            input_docids=None,
            checkpoint=checkpoint_path,
            heads_dir=heads_dir,
            calibration=calibration_dir,
            out=str(grades_dir),
            max_ctx=4096,
            max_batch=16,
        )

        try:
            returncode = cmd_grade(args)
            if returncode != 0:
                error(f"✗ HYDRA grading failed with return code {returncode}")
                return 1
        except Exception as e:
            error(f"✗ HYDRA grading failed: {e}")
            return 1

        if not summary_path.exists():
            error(f"✗ HYDRA summary not found: {summary_path}")
            return 1

    try:
        with summary_path.open() as f:
            summary = json.load(f)
    except json.JSONDecodeError as e:
        error(f"✗ Failed to parse HYDRA summary: {e}")
        return 1

    required_fields = ['total', 'kept', 'band', 'dropped']
    missing = [f for f in required_fields if f not in summary]
    if missing:
        error(f"✗ HYDRA summary missing fields: {missing}")
        return 1

    info(f"✓ HYDRA grading complete:")
    info(f"  Total: {summary['total']:,}")
    info(f"  Kept: {summary['kept']:,} ({summary['kept']/summary['total']*100:.1f}%)")
    info(f"  Band: {summary['band']:,} ({summary['band']/summary['total']*100:.1f}%)")
    info(f"  Dropped: {summary['dropped']:,} ({summary['dropped']/summary['total']*100:.1f}%)")

    # Step 3: Extract kept documents
    info(f"\n[Step 3/4] Extracting kept documents (quality >= threshold)...")

    if kept_docs_path.exists():
        kept_count = 0
        with kept_docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    kept_count += 1
        info(f"✓ Using existing kept docs: {kept_count:,} rows")
    else:
        kept_count = 0

        # Build lookup of kept doc IDs and their scores.
        kept_scores = {}
        try:
            with scores_path.open('r') as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        j = json.loads(line)
                        if j.get('decision') == 'keep':
                            kept_scores[j['doc_id']] = j['aggregated']
                    except json.JSONDecodeError as e:
                        warning(f"  Warning: Invalid JSON at line {line_no} in scores: {e}")
                        continue
        except Exception as e:
            error(f"✗ Failed to read quality scores: {e}")
            return 1

        # Stream through raw docs and write kept ones.
        try:
            with raw_docs_path.open('r') as in_f, kept_docs_path.open('w', encoding='utf-8') as out_f:
                for line_no, line in enumerate(in_f, 1):
                    if not line.strip():
                        continue
                    try:
                        j = json.loads(line)
                        doc_id = j.get('id')
                        if doc_id and doc_id in kept_scores:
                            out_f.write(json.dumps({
                                'text': j['text'],
                                'id': doc_id,
                                'score': kept_scores[doc_id]
                            }, ensure_ascii=False) + '\n')
                            kept_count += 1
                    except json.JSONDecodeError as e:
                        warning(f"  Warning: Invalid JSON at line {line_no} in raw docs: {e}")
                        continue
        except Exception as e:
            error(f"✗ Failed to extract kept documents: {e}")
            return 1

    info(f"✓ Extracted {kept_count:,} kept documents")

    # Step 4: K2 Rephrasing (style-diverse augmentation)
    if kept_count > 0:
        if rephrased_path.exists():
            total_rephrased = 0
            with rephrased_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        total_rephrased += 1
            info(f"\n[Step 4/4] Using existing rephrases: {total_rephrased:,} rows")
        else:
            info(f"\n[Step 4/4] Running K2 rephrasing ({num_versions} versions per document)...")
            info(f"  Processing {kept_count:,} documents...")

            from nmoe.data.model import BatchedGenerator
            from nmoe.data.rephrase import rephrase_batch, get_encoding

            # Load kept documents (bounded by shard size).
            texts = []
            doc_ids = []
            with kept_docs_path.open('r', encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    texts.append(j['text'])
                    doc_ids.append(j['id'])

            # Initialize generator
            gen = BatchedGenerator(checkpoint_path, max_batch=8, max_seq_len=8192)
            enc = get_encoding()

            # Rephrase in batches (to avoid memory spikes)
            batch_size = 100
            total_rephrased = 0

            with rephrased_path.open('w', encoding='utf-8') as f:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_ids = doc_ids[i:i+batch_size]

                    if (i // batch_size) % 10 == 0:
                        info(f"  Rephrasing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")

                    results = rephrase_batch(
                        gen, enc, batch_texts,
                        num_versions=num_versions,
                        chunk_size=2048,
                        use_style_diversity=True,
                        max_new=8192,  # HIGH: ~2K reasoning + 4K output + buffer
                        temperature=0.8,
                        top_p=0.9
                    )

                    # Write batch results
                    for doc_idx, versions in enumerate(results):
                        for ver_idx, text in enumerate(versions):
                            obj = {
                                'text': text,
                                'id': f'{batch_ids[doc_idx]}_v{ver_idx}',
                                'original_id': batch_ids[doc_idx]
                            }
                            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                            total_rephrased += 1

            info(f"✓ Generated {total_rephrased:,} rephrased versions")
    else:
        info(f"\n[Step 4/4] Skipping K2 rephrasing (no documents kept)")
        total_rephrased = 0
        rephrased_path = None

    # Step 5: Tokenization to training shards
    if total_rephrased > 0:
        manifest_path = shards_dir / "manifest.json"
        if manifest_path.exists():
            info(f"\n[Step 5/5] Using existing training shards at {shards_dir}")
        else:
            info(f"\n[Step 5/5] Tokenizing to training shards...")

            from nmoe.data.cli import cmd_prep
            import argparse

            # One shard namespace per pipeline shard output directory; shard rotation
            # is handled by tokens_per_shard in ShardedWriter.
            prep_args = argparse.Namespace(
                source='jsonl',
                paths=[str(rephrased_path)],
                output=str(shards_dir),
                name=dataset_name,
                version='v1',
                tokenizer='o200k_harmony',
                vocab_size=201088,
                eos_token_id=199999,
                num_shards=1,
                tokens_per_shard=500_000_000,
                workers=8,
                batch_size=1000,
                text_field='text',
                min_tokens=10,
                max_tokens=1_000_000,
                parallel=False,
                dataset=None,
                split=None,
                subset=None,
            )

            try:
                returncode = cmd_prep(prep_args)
                if returncode != 0:
                    error(f"✗ Tokenization failed with return code {returncode}")
                    return 1
                info(f"✓ Training shards created at {shards_dir}")
            except Exception as e:
                error(f"✗ Tokenization failed: {e}")
                return 1
    else:
        info(f"\n[Step 5/5] Skipping tokenization (no rephrased documents)")
        shards_dir = None

    # Final summary
    doc_total = int(summary.get("total", 0))
    info(f"\n{'='*80}")
    info(f"PIPELINE COMPLETE")
    info(f"{'='*80}")
    info(f"Output directory: {output_path}")
    info(f"")
    info(f"Results:")
    info(f"  Raw documents: {doc_total:,}")
    info(f"  Kept (high quality): {kept_count:,} ({kept_count/doc_total*100 if doc_total > 0 else 0:.1f}%)")
    info(f"  Rephrased versions: {total_rephrased:,}")
    info(f"  Augmentation factor: {total_rephrased/kept_count if kept_count > 0 else 0:.1f}x")
    info(f"")
    info(f"Files:")
    info(f"  {raw_docs_path}")
    info(f"  {kept_docs_path}")
    if kept_count > 0:
        info(f"  {rephrased_path}")
    if shards_dir:
        info(f"  {shards_dir}")
    info(f"{'='*80}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Run HYDRA + K2 preprocessing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU
  python pipeline.py \
    --dataset fineweb \
    --max-docs 100000 \
    --output /data/fineweb_edu \
    --checkpoint /data/checkpoints/gpt-oss-20b \
    --heads-dir /workspace/nmoe/nmoe/data \
    --calibration /workspace/nmoe/nmoe/data

  # Parallel processing (manual sharding)
  python pipeline.py \
    --dataset fineweb \
    --max-docs 100000 \
    --output /data/shard_0 \
    --checkpoint /data/checkpoints/gpt-oss-20b \
    --heads-dir /workspace/nmoe/nmoe/data \
    --calibration /workspace/nmoe/nmoe/data \
    --shard-index 0 \
    --num-shards 4
        """
    )
    parser.add_argument('--dataset', required=True, choices=['fineweb', 'dolma'],
                        help='Dataset to process')
    parser.add_argument('--max-docs', type=int, required=True,
                        help='Maximum documents to process')
    parser.add_argument('--output', required=True,
                        help='Output directory')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to gpt-oss-20b checkpoint')
    parser.add_argument('--heads-dir', required=True,
                        help='Directory containing hydra_judge.pt')
    parser.add_argument('--calibration', required=True,
                        help='Directory containing calibration_summary.json')
    parser.add_argument('--num-versions', type=int, default=10,
                        help='Number of rephrased versions per document (default: 10)')
    parser.add_argument('--shard-index', type=int,
                        help='Shard index for parallel processing (default: from JOB_COMPLETION_INDEX env or 0)')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Total number of shards (default: 1)')

    args = parser.parse_args()

    # Get shard index from arg, env var, or default to 0
    import os
    shard_index = args.shard_index
    if shard_index is None:
        shard_index = int(os.getenv('JOB_COMPLETION_INDEX', '0'))

    return run_pipeline(args.dataset, args.max_docs, args.output,
                       args.checkpoint, args.heads_dir, args.calibration,
                       args.num_versions, shard_index, args.num_shards)


if __name__ == '__main__':
    sys.exit(main())
