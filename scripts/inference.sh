python -m model.inference \
    --model_name qwen25 \
    --prompt_path /fs-computility/ai-shen/gutianle/SV-DPO/LLM-SafeCheck/safety_decoding/data/advbench_harmful_behaviors.json \
    --save_path /fs-computility/ai-shen/gutianle/SV-DPO/attack/advbench_harmful_behaviors.json \
    --prompt_key "goal"