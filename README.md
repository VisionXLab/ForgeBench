# ForgeBench

Evaluation code for [FORGE-Bench](https://huggingface.co/datasets/aaaabcd/FORGE-Bench), a benchmark of industrial image-to-video generation.

The runner samples each task video at fixed points, collects judge and CV evidence, and writes per-sample scores and model summaries. The [scoring method](docs/SCORING_METHOD.md) describes the six axes, event coverage, operator caps, and aggregation.

## Code

| Path | Purpose |
|---|---|
| `eval/run_eval.py` | Run evaluation for a model and task manifest |
| `eval/video_protocol.py` | Select frames for judges and CV operators |
| `eval/operator_evidence.py` | Measure local changes, temporal breaks, and joint motion |
| `eval/llm_judge*.py` | Anthropic and OpenAI-compatible judge backends |
| `scoring/per_sample.py` | Score one task |
| `scoring/aggregate.py` | Aggregate tasks and estimate confidence intervals |
| `scripts/combine_eval_shards.py` | Merge evaluation shards against the task manifest |
| `scripts/reaggregate_cached_results.py` | Rescore saved judge results |

## Run

Install Python 3.10 or newer and the dependencies:

```bash
python -m pip install -r requirements.txt
```

Put videos under one directory, named `{task_id}.mp4`. Category subdirectories are supported. The task manifest should contain paths to the corresponding reference images.

```bash
python -m eval.run_eval \
  --model my_model \
  --video_dir /path/to/videos \
  --samples_json /path/to/video_generation_500_samples.json \
  --output_dir results \
  --llm_provider openai_compat
```

Set `OPENAI_COMPAT_API_KEY` for an OpenAI-compatible endpoint, or use `--llm_provider anthropic` with `ANTHROPIC_API_KEY`. Results are written to `results/my_model/` as task records, `per_sample.json`, `aggregate.json`, `run_metadata.json`, and `report.json`.

The task set and generated videos are available from the [dataset release](https://huggingface.co/datasets/aaaabcd/FORGE-Bench).

## Tests

```bash
python -m pip install pytest
python -m pytest -q
```
