'''
inference and save generation results
'''
import json
import argparse
from tqdm import tqdm
from openai import OpenAI

def run(args):
    with open(args.prompt_path, 'r') as f:
        data = json.load(f)
    client = OpenAI(
        api_key = 'EMPTY',
        base_url = 'http://localhost:8010/v1'
    )
    with open(args.save_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(data):
            prompt = sample[args.prompt_key]
            try:
                completion = client.chat.completions.create(
                    messages = [{
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': prompt
                            }
                        ]
                    }],
                    model = args.model_name
                )
                f.write(json.dumps({
                    'question': prompt,
                    'answer': completion.choices[0].message.content
                }, ensure_ascii=False) + '\n')
                f.flush()
            except:
                print('Error')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Inference and save generation results")
    parser.add_argument('--model_name', type=str, default='sft_mix4', help='Model name')
    parser.add_argument('--prompt_path', type=str, default='/fs-computility/ai-shen/shared/VauAI/share/verifier/data/value_testset_0228.json', help='Prompt path')
    parser.add_argument('--save_path', type=str, default='infernce.json', help='Save path')
    parser.add_argument('--prompt_key', type=str, default='prompt')
    
    args = parser.parse_args()
    
    run(args)
    