
import os 
import tempfile

from core.data_generation.agent import Agent 


class LMConfigs:

    def __init__(
        self,
        prompter_lm,
        fixer_lm,
        tb_generator_lm,
        enhancer_lm,
        checker_lm,
        reflector_lm,
        curator_lm,
        max_seq_length,
        max_new_tokens,
        temperature=0.1,
        use_vllm=True,
        lora=None,
        generator_lora=None,
        api_base=None,
        api_key="EMPTY",
        generator_api_base=None,
        generator_api_key=None,
        enhancer_api_base=None,
        enhancer_api_key=None,
        seed=0,
    ) -> None:
        self.temperature = temperature
        self.seed = seed
        self.generator_api_base = generator_api_base
        self.generator_api_key = generator_api_key or api_key
        self.enhancer_api_base = enhancer_api_base
        self.enhancer_api_key = enhancer_api_key or api_key
        self.generator_lora = None if generator_api_base else (generator_lora or lora)

        self.prompter_lm_name = prompter_lm 
        agent = Agent(
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens
        )
        model_cache = {}

        def resolve(model_name, lora_path=None, model_api_base=api_base, model_api_key=api_key):
            cache_key = (model_name, lora_path, model_api_base, seed)
            if cache_key not in model_cache:
                model_cache[cache_key] = agent.get_model(
                    model=model_name,
                    temperature=temperature,
                    use_vllm=use_vllm,
                    lora_path=lora_path,
                    api_base=model_api_base,
                    api_key=model_api_key,
                    seed=seed,
                )
            return model_cache[cache_key]

        self.prompter_llm_model = resolve(prompter_lm)
        
        self.fixer_llm_model = resolve(fixer_lm)

        self.tb_generator_llm_model = resolve(
            tb_generator_lm,
            lora_path=self.generator_lora,
            model_api_base=self.generator_api_base or api_base,
            model_api_key=self.generator_api_key,
        )

        self.enhancer_llm_model = resolve(
            enhancer_lm,
            model_api_base=self.enhancer_api_base or api_base,
            model_api_key=self.enhancer_api_key,
        )

        self.reflector_llm_model = resolve(reflector_lm)

        self.curator_llm_model = resolve(curator_lm)   

        self.checker_llm_model = resolve(checker_lm)


class FlowConfigs:

    def __init__(self, lm_config, max_refine_iters=5, max_enhance_iters=5, graph_recursion_limit=25, output_dir=None, testbench_style="self_checking", wo_reflector=False, wo_curator=False, data_gen_mode=True) -> None:
        self.lm_config  = lm_config
        self.max_refine_iters= max_refine_iters
        self.max_enhance_iters = max_enhance_iters
        self.graph_recursion_limit = graph_recursion_limit 
        self.testbench_style = testbench_style
        self.wo_reflector = wo_reflector
        self.wo_curator = wo_curator
        self.data_gen_mode = data_gen_mode

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            self.output_dir = output_dir
        else:
            self.output_dir = tempfile.mkdtemp()        
