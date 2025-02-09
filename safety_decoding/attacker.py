'''
Attack LLM using different jailbreak methods
'''
import os
import time
import json
import torch
import argparse
from tqdm import tqdm
from model.model import LLM
from datasets import load_dataset
from utils import load_conversation_template

def get_args():
    parser = argparse.ArgumentParser(description='Attack LLM using different jailbreak methods')
    # Experimental Settings
    parser.add_argument('--model_name_or_path', type=str, default='llama2', help='Model name or path')
    parser.add_argument('--template_name', type=str, default='llama-2', help='Template name')
    parser.add_argument('--attacker', type=str, default='gcg', help='Jailbreak Methods')
    parser.add_argument('--whitebox_attacker', action='store_true', help='Whitebox Attacker')
    parser.add_argument('--output_directory', type=str, default='output', help='Output directory')
    return parser.parse_args()

def load_jailbreak_prompts(model_name, attack_name):
    if attack_name == 'AdvBench':
        with open('data/harmful_behaviors_custom.json', 'r', encoding='utf-8') as f:
            attack_prompts = json.load(f)
    elif attack_name in ['GCG', 'AutoDAN', 'PAIR']:
        attack_prompts = load_dataset('flydust/SafeDecoding-Attackers', split='train')
        attack_prompts = attack_prompts.filter(lambda x: x['source'] == attack_name)
        if model_name in ['vicuna', 'llama2', 'guanaco']:
            attack_prompts = attack_prompts.filter(lambda x: x['model'] == model_name)
        elif model_name == 'dolphin':
            attack_prompts = attack_prompts.filter(lambda x: x['model'] == 'llama2')
        elif model_name == 'falcon':
            if attack_name == 'GCG':
                attack_prompts = attack_prompts.filter(lambda x: x['model'] == 'llama2')
            else:
                attack_prompts = attack_prompts.filter(lambda x: x['model'] == model_name)
    elif attack_name == 'DeepInception':
        attack_prompts = load_dataset('flydust/SafeDecoding-Attackers', split='train')
        attack_prompts = attack_prompts.filter(lambda x: x['source'] == 'DeepInception')
    elif attack_name == 'custom':
        with open('data/custom_prompts.json', 'r', encoding='utf-8') as f:
            attack_prompts = json.load(f)
    else:
        raise ValueError('Invalid attack name')
    return attack_prompts

def main():
    args = get_args()
    # load template
    conv_template = load_conversation_template(args.template_name)
    # load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LLM(args.model_name_or_path, device, args.whitebox_attacker)
    # choose attacker
    attack_prompts = load_jailbreak_prompts(args.template_name, args.attacker)
    # set output json
    output_json = {}
    output_json['config'] = {
        'model_name_or_path': args.model_name_or_path,
        'attacker': args.attacker,
        'template_name': args.template_name,
        'whitebox_attacker': args.whitebox_attacker
    }
    output_json['data'] = []
    # attack
    for prompt in tqdm(attack_prompts):
        response = model.generate(prompt, conv_template)
        output_json['data'].append({
            'prompt': prompt,
            'response': response
        })
    # save output json
    current_time = time.localtime()
    time_str = str(time.strftime('%Y-%m-%d-%H-%M-%S', current_time))
    os.makedirs(args.output_directory, exist_ok=True)
    output_path = os.path.join(args.output_directory, f'{args.attacker}_{args.template_name}_{time_str}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=4)
    print(f'Output saved at {output_path}')