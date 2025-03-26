# LLM-SafeCheck
This repository will integrate some work on LLM Safety Evaluation.

## Features
SafeCheck uses vLLMs to create a unified inference interface.

## Usage Introduction
1. Start a service using vLLM (install it via `pip install vllm`)
2. Execute the script located at `./scripts/serve.sh` or alternatively, run the following commands directly in your terminal:
```bash
python -m vllm.entrypoints.openai.api_server \
      --model {model_name_or_path} \
      --served-model-name {model_proxy_name} \
      --dtype bfloat16 \
      --tensor-parallel-size 4 \
      --host "0.0.0.0" \
      --port 8010
```
3. Use the script `./scripts/inference.sh` to locally save the generated results for further analysis.
```bash
python -m model.inference \
    --model_name qwen25 \
    --prompt_path {prompt_path} \
    --save_path {local_path_to_save_generated_results} \
    --prompt_key "goal"
```


## Road Map
- [x] SafeDecoding
- [x] Flames

## References
SafeDecoding
```
@misc{xu2024safedecoding,
      title={SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding}, 
      author={Zhangchen Xu and Fengqing Jiang and Luyao Niu and Jinyuan Jia and Bill Yuchen Lin and Radha Poovendran},
      year={2024},
      eprint={2402.08983},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
Flames
```
@misc{huang2023flames,
      title={Flames: Benchmarking Value Alignment of Chinese Large Language Models}, 
      author={Kexin Huang and Xiangyang Liu and Qianyu Guo and Tianxiang Sun and Jiawei Sun and Yaru Wang and Zeyang Zhou and Yixu Wang and Yan Teng and Xipeng Qiu and Yingchun Wang and Dahua Lin},
      year={2023},
      eprint={2311.06899},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```