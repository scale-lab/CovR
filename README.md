# 🔍 CovR
CovR: Coverage-Aware Hardware Verification via Reasoning-Guided Reinforcement Learnin

## Overview

CovR generates SystemVerilog testbenches that target RTL code coverage. It combines simulation feedback and agentic self-refinement with supervised fine-tuning on reasoning-annotated examples and reinforcement learning using coverage-driven rewards.

The framework has three components:

- **Data generation:** generate and refine testbenches using simulation and coverage feedback, then annotate them with reasoning traces.
- **Model training:** specialize a language model through supervised fine-tuning (SFT), followed by Group Relative Policy Optimization (GRPO).
- **Verification:** generate test stimuli directly or refine them through the agentic workflow to improve line, toggle, condition, branch, and FSM coverage.

CovR produces **stimulus-only testbenches**. 


## Dataset

The paper introduces a synthetic dataset of **16,514 specification–RTL–reasoning–testbench tuples**. 

Dataset is hosted on 🤗 [Hugging Face](https://huggingface.co/datasets/scale-lab/CovR). 

## Models

| Model | Base model |
|---|---|
| [CovR-Qwen3-4B-Instruct-2507](https://huggingface.co/scale-lab/CovR-Qwen3-4B-Instruct-2507) | Qwen/Qwen3-4B-Instruct-2507 |
| [CovR-Qwen2.5-Coder-7B-Instruct](https://huggingface.co/scale-lab/CovR-Qwen2.5-Coder-7B-Instruct) | Qwen/Qwen2.5-Coder-7B-Instruct |


## Install Dependencies

Install dependecies in a virtual environment: 

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

The verification depends synopsys tool chains (VCS & URG). 

## Run Direct Inference 

Generate testbenches in a single pass (no agentic refinement) with the base model plus the CovR LoRA adapter. Run this inside a GPU allocation from the repository root, using the workflow environment from [Install Dependencies](#install-dependencies):

```bash
hf download scale-lab/CovR-Qwen3-4B-Instruct-2507 --local-dir ./models/CovR-Qwen3-4B-Instruct-2507

python core/train/evaluator.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --lora ./models/CovR-Qwen3-4B-Instruct-2507 \
  --eval ./data/eval_dataset.jsonl \
  --run_name covr_qwen3_4b_direct \
  --output_dir ./output-eval/direct \
  --num_samples 10 \
  --temperature 0.1 \
  --max_seq_length 32000 \
  --max_new_tokens 10000 \
  --with_reasoning \
  --use_vllm \
  --generate_only
```

- Use `--model Qwen/Qwen2.5-Coder-7B-Instruct` with the [Coder-7B adapter](https://huggingface.co/scale-lab/CovR-Qwen2.5-Coder-7B-Instruct) for the other model.
- Use `--eval ./data/eval_cvdp.jsonl` for CVDP.
- Answers are saved in a timestamped directory under `--output_dir` as `answers_rank*.jsonl`. Drop `--generate_only` to also run Synopsys VCS/URG evaluation (requires the simulator).


## Run Agentic Workflow 

Run the self-reflection workflow from the repository root. This example uses `gpt-4o-mini` for all agents on CVDP. Activate your workflow environment, set `OPENAI_API_KEY`, and configure Synopsys VCS/URG before running: 

```bash
python -m core.baselines.self_reflection_loop.main \
  --prompter-model gpt-4o-mini \
  --tb-model gpt-4o-mini \
  --enhancer-model gpt-4o-mini \
  --checker-model gpt-4o-mini \
  --reflector-model gpt-4o-mini \
  --curator-model gpt-4o-mini \
  --fixer-model gpt-4o-mini \
  --dataset ./data/eval_cvdp.jsonl \
  --iters 5 \
  --num_samples 10 \
  --sample-workers 1 \
  --problem-workers 1 \
  --max_seq_length 32000 \
  --max_new_tokens 10000 \
  --output ./output-eval/self_reflection/gpt4omini_cvdp
```

- Use `--dataset ./data/eval_dataset.jsonl` for VerilogEval + RTLLM, and choose a corresponding output directory.
- `--iters` limits each repair and coverage-enhancement loop; `--num_samples` sets the number of trajectories per design.
- Outputs are saved in a timestamped directory under `--output`, including `answers_llm4dv.jsonl`.
- Add `--generate-only` to skip final saved-answer evaluation. Simulation and coverage feedback still run during self-reflection.

Use finetuned models for generator agent: 

For fine-tuned CovR models, use the [Qwen3-4B launcher](jobs/agentic/self_reflection_loop/qwen4b_lora.sh) or [Coder-7B launcher](jobs/agentic/self_reflection_loop/qwen2.5_coder_lora.sh). They use separate vLLM endpoints for the base agents and LoRA generator/enhancer. Update the local adapter path, environment activation, simulator settings, and Slurm resources before submitting a launcher.

## Training 
### Supervised Fine-tuning

The SFT entry point is [core/train/sft.py](core/train/sft.py), with YAML configurations under [core/train/cfg/experiments/sft](core/train/cfg/experiments/sft). After configuring the training environment and dataset paths, an example command is:

```bash
python -m core.train.sft \
  --config core/train/cfg/experiments/sft/qwen4b_instruct/sft_r512_a512.yaml \
  --with_reasoning \
  --completion_only_loss \
  --run_name covr_qwen3_4b_sft \
  --output_dir ./output/sft \
  --generate_only
```

Review [jobs/sft](jobs/sft) for the experiment launchers and their environment requirements.

### Reinforcement Learning

The GRPO entry point is [core/train/grpo.py](core/train/grpo.py). Configurations are under [core/train/cfg/experiments/grpo](core/train/cfg/experiments/grpo), and experiment launchers are under [jobs/grpo](jobs/grpo).

GRPO starts from an SFT checkpoint and uses simulation-derived rewards. Configure the SFT checkpoint, dataset, simulator, coverage tools, and compute resources before launching. The supplied job scripts are cluster-specific experiment references.

## Repository Structure

```text
core/
  baselines/        Baseline verification workflows
  data/             Dataset preparation utilities
  data_generation/  Testbench generation and refinement
  metrics/          Evaluation and coverage metrics
  plots/            Result analysis and figures
  train/            SFT, GRPO, inference, prompts, and configurations
data/               Evaluation inputs and extraction utilities
jobs/               Experiment and Slurm launch scripts
```

## Citation

```bibtex
@inproceedings{abdelatty2026covr,
  title={CovR: Coverage-Aware Hardware Verification via Reasoning-Guided Reinforcement Learning},
  author={Abdelatty, Manar and Nouh, Maryam and Reda, Sherief},
  booktitle={2026 ACM/IEEE International Symposium on Machine Learning for CAD (MLCAD)},
  year={2026}
}
```

