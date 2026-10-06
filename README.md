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

| Model | Base model | RL checkpoint | Access |
|---|---|---:|---|
| [CovR-Qwen3-4B-Instruct-2507](https://huggingface.co/scale-lab/CovR-Qwen3-4B-Instruct-2507) | Qwen/Qwen3-4B-Instruct-2507 | 500 | Private |
| [CovR-Qwen2.5-Coder-7B-Instruct](https://huggingface.co/scale-lab/CovR-Qwen2.5-Coder-7B-Instruct) | Qwen/Qwen2.5-Coder-7B-Instruct | 600 | Private |

Both repositories contain **PEFT LoRA adapters**, tokenizer assets, exact prompt templates, and a Python inference example. Loading an adapter also requires its base model. Private access requires a Hugging Face account with permission to the model repository.

## Install Dependencies

Use **Python 3.11** on Linux with an NVIDIA GPU and a compatible CUDA driver for local model serving/training. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

[requirements.txt](requirements.txt) is based on the LLM4TB `unsloth-b200` environment used by the self-reflection launchers. It includes PyTorch 2.10.0, vLLM 0.18.0, Transformers 4.57.6, and Unsloth 2026.5.2. Two pins resolve conflicts in the original environment: `dill==0.4.0` and `langchain-openai==1.1.10`. OpenAI remains at 2.24.0 to satisfy vLLM's version constraint. Dependency metadata was checked; a fresh installation and GPU execution have not yet been validated.

Original environment snapshots are available in [requirements/unsloth-b200.txt](requirements/unsloth-b200.txt) and [requirements/trl-vllm-clean-env.txt](requirements/trl-vllm-clean-env.txt). The first preserves the original conflicts for reference. The second records the separate training/evaluation environment with different TRL and PEFT versions. Install it in a separate virtual environment when reproducing launchers that use `trl-vllm-clean-env`; do not combine the snapshots.

Configure Synopsys VCS/URG and their licenses separately; they are not pip packages. Set `OPENAI_API_KEY` for OpenAI-backed agents and run `hf auth login` for private model access. Update environment activation paths in the job scripts to match your environment.


## Run Direct Inference 

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


### Inference setup

For the standalone model example, use a Python environment with PyTorch, Transformers, PEFT, Accelerate, and the Hugging Face Hub client:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch 'transformers>=4.51.0,<5' 'peft>=0.19.1' accelerate huggingface_hub
hf auth login
```

Choose a PyTorch build compatible with your CUDA environment. The example loads the base model and adapter in BF16, so use a compatible GPU with sufficient memory. On a cluster, run model inference and training inside a GPU allocation.

These minimal dependencies cover only standalone inference. For the workflow environment, use [Install Dependencies](#install-dependencies) above; keep the minimal inference environment separate from the pinned workflow environment.

### Inference

Download the model's example and run it:

```bash
hf download scale-lab/CovR-Qwen3-4B-Instruct-2507 \
  example_inference.py --local-dir examples/qwen3_4b
python examples/qwen3_4b/example_inference.py
```

For Coder-7B:

```bash
hf download scale-lab/CovR-Qwen2.5-Coder-7B-Instruct \
  example_inference.py --local-dir examples/qwen25_coder_7b
python examples/qwen25_coder_7b/example_inference.py
```

Edit `spec` and `rtl` in the downloaded example to use your design. The example downloads the exact prompt templates, loads the base model and LoRA adapter, and applies the tokenizer's chat template.

### Prompt format

Use the system message and user template in each model repository's `prompt_templates.json`. The same prompt definitions are available in [core/train/prompts.py](core/train/prompts.py).

The user message includes:

```text
<spec>
Your natural-language design specification.
</spec>
<rtl>
Your complete RTL code.
</rtl>
<context>
Optional prior testbench, simulation errors, or coverage feedback.
</context>
Write a SystemVerilog testbench that maximizes coverage of the given RTL. Target coverage only in your testbench: Try to achieve as high coverage as possible, including line, toggle, FSM, branch, condition. Also, don't include assertion statements in the testbench; they'll be handled separately.
```

Leave context empty for initial generation. Apply the tokenizer's chat template with `add_generation_prompt=True`. The requested response has a `<think>...</think>` section followed by a `<testbench>...</testbench>` section containing a fenced Verilog code block. Extract the testbench code for compilation and simulation.

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

### Evaluation and Agentic Refinement

[core/train/evaluator.py](core/train/evaluator.py) supports answer generation and evaluation of saved answers. The two-stage pipeline in [jobs/evaluator](jobs/evaluator) separates generation from Synopsys VCS evaluation.

- Use `data/eval_dataset.jsonl` for the combined VerilogEval and RTLLM inputs.
- Use `data/eval_cvdp.jsonl` for CVDP inputs.
- Agentic self-refinement launchers are under [jobs/agentic/self_reflection_loop](jobs/agentic/self_reflection_loop).
- Coverage analysis and plotting utilities are under [core/metrics](core/metrics) and [core/plots](core/plots).

The paper's evaluation uses Synopsys VCS and URG. Configure your licensed installation and update paths, environment activation, model locations, and Slurm settings in the launchers before running them. Generated testbenches must be compiled and simulated; model output alone does not establish coverage or correctness.

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

## License

This project is licensed under the [MIT License](LICENSE). Base models and third-party datasets remain subject to their respective licenses.
