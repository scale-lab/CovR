
"""Creates an instruciton dataset (Instruction and Output pairs)
"""

import os
import json
import argparse
from typing import Any

import yaml
from torch.utils.data import Dataset as TorchDataset

from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import login
from transformers import AutoTokenizer
from core.utils import get_logger, write_ground_truth_files
from core.train.utils import plot_data_lengths
from core.train.model import get_model
from core.train.prompts import USER_PROMPT_SECTIONED, ASSISTANT_PROMPT_SECTIONED_W_REASONING, ASSISTANT_PROMPT_SECTIONED_WO_REASONING, SYS_PROMPT, SYS_PROMPT_WO_REASONING, LLM4_COV_SYSTEM_PROMPT, LLM4COV_USER_TEMPLATE
from core.data_generation.utils import read_jsonl


class HFDatasetWrapper(TorchDataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        return {k: self.hf_dataset[k][idx] for k in self.hf_dataset.column_names}
    
    
class InstructDataset:

    def __init__(self, train_json, test_json, train_split, valid_split, model, tokenizer, shuffle=True, filter_max_seq=True, max_seq_length=5028, grpo=False, with_reasoning=False, use_llm4cov_prompt=False, completion_only_loss=False) -> None:
        self.shuffle = shuffle
        self.model = model 
        self.tokenizer = tokenizer
        self.train_split = train_split
        self.valid_split = valid_split
        self.completion_only_loss = completion_only_loss and not grpo

        if train_json: 
            train_dataset = self.create_train_set(
                json_path=train_json, 
                grpo=grpo, 
                with_reasoning=with_reasoning, 
                use_llm4cov_prompt=use_llm4cov_prompt,
                completion_only_loss=self.completion_only_loss
            )
            if self.completion_only_loss:
                train_dataset = train_dataset.map(self._tokenize_prompt_completion)
            else:
                train_dataset = train_dataset.map(lambda samples: tokenizer(samples["prompt"]))

        test_dataset = self.create_test_set(json_path=test_json, with_reasoning=with_reasoning, use_llm4cov_prompt=use_llm4cov_prompt)

        # test_dataset = test_dataset.map(lambda samples: tokenizer(samples["prompt"]))

        if filter_max_seq and train_json:
            print("Dataset Length before dropping >= max_seq_length: ", len(train_dataset))
            print(f"Max train sequence length (before dropping) is: {max([len(x['input_ids']) for x in train_dataset])}")
            train_dataset = train_dataset.filter(lambda example: len(example['input_ids']) <= max_seq_length)
            print("Dataset Length after dropping >= max_seq_length: ", len(train_dataset))
            
        if train_json: 
            train_valid_dataset = train_dataset.train_test_split(
                train_size=self.train_split, 
                test_size=self.valid_split, 
                shuffle=shuffle, 
                seed=0
            )

            self.dataset = DatasetDict({
                'train': train_valid_dataset['train'],
                'valid': train_valid_dataset['test'],
                'test': test_dataset
            })
        else:
            self.dataset = DatasetDict({
                'test': test_dataset
            })

    def create_train_set(self, json_path, grpo, with_reasoning, use_llm4cov_prompt, completion_only_loss):
        natural_specs = []
        testbenches = []
        rtl_codes = []
        top_modules = []
        prompts = []
        completions = []

        data = read_jsonl(json_path)

        for  sample in data:         
            natural_specs.append(sample["natural_spec"])
            testbenches.append(sample["testbench"])
            rtl_codes.append(sample["rtl_code"])
            top_modules.append(sample["top_module"])
            if with_reasoning:
                reasoning = self._extract_best_reasoning_text(
                    sample.get("best_reasoning_trace", sample.get("tb_reasoning_gen_output", ""))
                )
                # reasoning_trace = sample["best_reasoning_trace"]['reasoning']
            else: 
                reasoning = ""

            if completion_only_loss and not grpo:
                prompt, completion = self.preprocess_prompt_completion(
                    natural_spec=sample["natural_spec"],
                    rtl_code=sample["rtl_code"],
                    testbench=sample["testbench"],
                    reasoning=reasoning,
                    top_module=sample.get("top_module"),
                    dut_instance_name=sample.get("dut_instance_name"),
                    with_reasoning=with_reasoning,
                    use_llm4cov_prompt=use_llm4cov_prompt
                )
                completions.append(completion)
            else:
                prompt = self.preprocess(
                    natural_spec=sample["natural_spec"],
                    rtl_code=sample["rtl_code"],
                    testbench=sample["testbench"],
                    reasoning=reasoning,
                    # reasoning=reasoning_trace,
                    top_module=sample.get("top_module"),
                    dut_instance_name=sample.get("dut_instance_name"),
                    with_reasoning=with_reasoning,
                    test=grpo,
                    use_llm4cov_prompt=use_llm4cov_prompt
                )
            
            prompts.append(prompt)

        dataset_dict = {
            "natural_spec": natural_specs,
            "testbench": testbenches,
            "rtl_code": rtl_codes,
            "top_module": top_modules,
            "prompt": prompts
        }
        if completion_only_loss and not grpo:
            dataset_dict["completion"] = completions

        train_dataset = Dataset.from_dict(dataset_dict)

        train_dataset = train_dataset.shuffle(seed=0)   

        return train_dataset 
    

    def create_test_set(self, json_path, with_reasoning, use_llm4cov_prompt):
        natural_specs = []
        testbenches = []
        rtl_codes = []
        top_modules = []
        prompts = []
        sample_id = []
        target_percentages = []

        data = read_jsonl(json_path)

        for  sample in data:         
            natural_specs.append(sample["natural_spec"])
            testbenches.append(sample["testbench"])
            rtl_codes.append(sample["rtl_code"])
            top_modules.append(sample.get("top_module", "TopModule"))
            sample_id.append(sample["sample_id"])
            
            dut_instance = sample.get("dut_instance_name", "")
            
            target_percentage = 100.0
            matched_target = None
            for target in sample.get("targets", []):
                if target.get("inst_name") == dut_instance:
                    target_percentage = target["target_percentage"]
                    matched_target = target
                    break

            target_percentages.append(target_percentage)
            print(
                "DEBUG target_percentage:",
                f"sample_id={sample['sample_id']}",
                f"dut_instance={dut_instance!r}",
                f"matched_inst={matched_target.get('inst_name') if matched_target else None!r}",
                f"target_percentage={target_percentage}",
            )
          
            prompt = self.preprocess(
                natural_spec=sample["natural_spec"],
                rtl_code=sample["rtl_code"],
                testbench=sample["testbench"],
                reasoning="",
                top_module=sample.get("top_module", "TopModule"),
                dut_instance_name=dut_instance,
                test=True,
                with_reasoning=with_reasoning,
                use_llm4cov_prompt=use_llm4cov_prompt
            )

            prompts.append(prompt)

        test_dataset = Dataset.from_dict({
            "natural_spec": natural_specs,
            "testbench": testbenches,
            "rtl_code": rtl_codes,
            "top_module": top_modules,
            "prompt": prompts,
            "sample_id": sample_id,
            "target_percentage": target_percentages
        })

        test_dataset = test_dataset.shuffle(seed=0)   

        return test_dataset  


    def _extract_best_reasoning_text(self, value: Any) -> str:
        if isinstance(value, list) and value:
            return self._extract_best_reasoning_text(value[0])
        if isinstance(value, dict):
            return str(value.get("reasoning", ""))
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.startswith("{") and cleaned.endswith("}"):
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict):
                        return str(parsed.get("reasoning", parsed.get("reason", "")))
                except Exception:
                    pass
            return cleaned
        return ""

    def _tokenize_prompt_completion(self, sample):
        prompt_tokens = self.tokenizer(sample["prompt"])
        full_tokens = self.tokenizer(sample["prompt"] + sample["completion"])
        prompt_length = min(len(prompt_tokens["input_ids"]), len(full_tokens["input_ids"]))
        completion_length = len(full_tokens["input_ids"]) - prompt_length
        full_tokens["completion_mask"] = [0] * prompt_length + [1] * completion_length
        return full_tokens

    def preprocess_prompt_completion(
        self,
        natural_spec: str,
        rtl_code: str,
        testbench: str,
        reasoning: str,
        top_module: str | None = None,
        dut_instance_name: str | None = None,
        with_reasoning=False,
        use_llm4cov_prompt=False
    ):
        prompt = self.preprocess(
            natural_spec=natural_spec,
            rtl_code=rtl_code,
            testbench=testbench,
            reasoning=reasoning,
            top_module=top_module,
            dut_instance_name=dut_instance_name,
            test=True,
            with_reasoning=with_reasoning,
            use_llm4cov_prompt=use_llm4cov_prompt
        )
        full_prompt = self.preprocess(
            natural_spec=natural_spec,
            rtl_code=rtl_code,
            testbench=testbench,
            reasoning=reasoning,
            top_module=top_module,
            dut_instance_name=dut_instance_name,
            test=False,
            with_reasoning=with_reasoning,
            use_llm4cov_prompt=use_llm4cov_prompt
        )
        if full_prompt.startswith(prompt):
            completion = full_prompt[len(prompt):]
        else:
            completion = self._assistant_answer(
                testbench=testbench,
                reasoning=reasoning,
                top_module=top_module,
                with_reasoning=with_reasoning,
                use_llm4cov_prompt=use_llm4cov_prompt,
            )
        return prompt, completion

    def _assistant_answer(
        self,
        testbench: str,
        reasoning: str,
        top_module: str | None = None,
        with_reasoning=False,
        use_llm4cov_prompt=False,
    ):
        if use_llm4cov_prompt:
            filename_top = self._sanitize_filename_top(top_module)
            return f"filename: tb_{filename_top}.sv\n```systemverilog\n{testbench}\n```"
        if with_reasoning:
            return ASSISTANT_PROMPT_SECTIONED_W_REASONING(reasoning=reasoning, testbench=testbench)
        return ASSISTANT_PROMPT_SECTIONED_WO_REASONING(testbench=testbench)

    def _sanitize_filename_top(self, top_module: str | None) -> str:
        top = (top_module or "dut").strip() or "dut"
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in top)

    def _llm4cov_context(self, natural_spec: str, rtl_code: str) -> str:
        rtl_part = f"// File: dut.sv\n{rtl_code.strip()}"
        if natural_spec and natural_spec.strip():
            spec_part = f"// File: design_requirements.txt\n{natural_spec.strip()}"
        else:
            spec_part = "// No specification files provided."
        return "\n\n".join([rtl_part, spec_part])

    def _llm4cov_instructions(
        self,
        top_module: str | None = None,
        dut_instance_name: str | None = None,
    ) -> str:
        top_module = (top_module or "").strip()
        dut_instance_name = (dut_instance_name or "dut").strip() or "dut"
        design_clause = (
            f"The top module is '{top_module}', and its instance name in the testbench "
            f"should be '{dut_instance_name}'. "
            if top_module
            else f"The DUT instance name in the testbench should be '{dut_instance_name}'. "
        )
        return (
            "Generate a SystemVerilog testbench for the given RTL design. "
            + design_clause
            + "Target coverage only in your testbench: Try to achieve as high coverage as "
            "possible, including line, toggle, FSM, branch, etc. Also, don't include "
            "assertion statements in the testbench; they'll be handled separately."
        )

    def preprocess(
        self,        
        natural_spec: str,
        rtl_code: str,
        testbench: str,
        reasoning: str, # TODO: make use of reasoning traces
        top_module: str | None = None,
        dut_instance_name: str | None = None,
        test = False,
        with_reasoning = False,
        use_llm4cov_prompt = False
    ):

        if use_llm4cov_prompt: 
            instructions = self._llm4cov_instructions(
                top_module=top_module,
                dut_instance_name=dut_instance_name,
            )
            user_prompt = LLM4COV_USER_TEMPLATE(
                context=self._llm4cov_context(natural_spec=natural_spec, rtl_code=rtl_code),
                instructions=instructions,
            )
            system_prompt = LLM4_COV_SYSTEM_PROMPT
        else:
            if with_reasoning:
                system_prompt = SYS_PROMPT
            else:
                system_prompt = SYS_PROMPT_WO_REASONING
            # TODO: Fix context
            user_prompt = USER_PROMPT_SECTIONED(natural_spec=natural_spec, rtl_code=rtl_code, context="")

        if test: 
            assistant_answer=""
        else:
            assistant_answer = self._assistant_answer(
                testbench=testbench,
                reasoning=reasoning,
                top_module=top_module,
                with_reasoning=with_reasoning,
                use_llm4cov_prompt=use_llm4cov_prompt,
            )

        # TODO: MAKE SURE THAT SFTTrainer Doesn't apply the chat template internally
        full_prompt = self.model.generate_chat_template(
            tokenizer = self.tokenizer,
            system_prompt=system_prompt, 
            user_prompt=user_prompt, 
            assistant_answer=assistant_answer,
            add_generation_prompt=test

        )
        return full_prompt
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help="Path to training JSON file ", required=False, default="./output/data_merged_all/all_merged_deduplicated.jsonl")
    parser.add_argument('--test_veval', type=str, help="Path to test veval dataset JSON file ", required=False, default="./data/eval_dataset.jsonl")
    parser.add_argument('--test_cvdp', type=str, help="Path to test cvdp dataset JSON file ", required=False, default="./data/eval_cvdp.jsonl")
    parser.add_argument('--output_dir', type=str, help="Path to output directory ", required=False, default="./output/sft")
    args = parser.parse_args()

    train = args.train
    test_veval = args.test_veval
    test_cvdp = args.test_cvdp
    output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(os.path.join(output_dir, "dataset.log"))
    
    access_token = os.getenv('HF_ACCESS_TOKEN')
    login(token=access_token)

    model_name = "qwen2.5-coder"
    model = get_model(name=model_name, load_model=False)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model.model_instruct_ids[model_name], 
        token=access_token, 
    ) 
    
    completion_only_loss = True 

    instruct_dataset = InstructDataset(
        train_json=train,     
        test_json=test_veval,
        train_split=0.8,
        valid_split=0.2,
        model=model,
        tokenizer=tokenizer,
        max_seq_length=8028,
        shuffle=True,
        filter_max_seq=True,
        use_llm4cov_prompt=False,
        with_reasoning=True,
        completion_only_loss=completion_only_loss
    )    
    
    if completion_only_loss: 
        logger.info(f"CYAN: Training Prompt: {instruct_dataset.dataset['train']['prompt'][0]}")
        logger.info(f"CYAN: Training Completion: {instruct_dataset.dataset['train']['completion'][0]}")
        logger.info(f"MAGENTA: Test Prompt: {instruct_dataset.dataset['test']['prompt'][0]}")
        logger.info(f"MAGENTA: Test target percentages: {instruct_dataset.dataset['test']['target_percentage']}")
    else:
        logger.info(f"CYAN: Training Prompt: {instruct_dataset.dataset['train']['prompt'][0]}")
        logger.info(f"MAGENTA: Test Prompt: {instruct_dataset.dataset['test']['prompt'][0]}")
        logger.info(f"MAGENTA: Test target percentages: {instruct_dataset.dataset['test']['target_percentage']}")

        
    lengths = [len(x['input_ids']) for x in instruct_dataset.dataset['train']]

    logger.info(f"CYAN: Max train sequence length is: {max(lengths)}")

    plot_data_lengths(
        tokenized_train_dataset=instruct_dataset.dataset["train"], 
        tokenized_val_dataset=instruct_dataset.dataset["valid"], 
        save_path=os.path.join(output_dir, "data_lengths_filtered.png")
    )



if __name__ == "__main__":
    main()
