python -m vllm.entrypoints.openai.api_server \
        --model /fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-7B-Instruct \
        --served-model-name qwen25 \
        --dtype bfloat16 \
        --tensor-parallel-size 4 \
        --host "0.0.0.0" \
        --port 8010