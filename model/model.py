'''
model.py
'''
import copy
import asyncio
import aiohttp
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

# opensource_model -- oldversion
class LLM:

    def __init__(self, model_name_or_path, device, whitebox_attacker=False):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map={'': device}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.whitebox_attacker = whitebox_attacker
        self.device = device

    def generate(self, prompt, conv_template):
        conv_template = copy(conv_template)
        conv_template.append_message(
            conv_template.roles[0],
            prompt
        )
        conv_template.append_message(
            conv_template.roles[1],
            None
        )
        prompt = conv_template.get_prompt()
        if conv_template.name == 'llama-2' and not self.whitebox_attacker:
            prompt += ' '
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'][0].unsqueeze(0)
        inputs['attention_mask'] = inputs['attention_mask'][0].unsqueeze(0)
        inputs = inputs.to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
class vLLM:
    
    def __init__(self, api_key, base_url, model_name):
        self.base_url = base_url
        self.client = OpenAI(
            api_key = api_key,
            base_url = base_url
        )
        self.model_name = model_name
    
    def generate(self, prompt):
        completion = self.client.chat.completions.create(
            messages = [{
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt
                    }
                ]
            }],
            model = self.model_name
        )
        return completion.choices[0].message.content

    async def generate_async(self, session, api_url, prompt):
        header = {"Content-Type": "application/json"}
        payload = {
            'messages': [{
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt
                    }
                ]
            }],
            'model': self.model_name
        }
        async with session.post(api_url, headers=header, json=payload) as resp:
            if resp.status != 200:
                raise Exception(f"Request failed with status {resp.status}")
            data = await resp.json()
            return data['choices'][0]['message']['content']
    
    async def parallel_generate(self, prompts, max_concurrent):
        semaphore = asyncio.Semaphore(max_concurrent)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for prompt in tqdm(prompts):
                async def sem_task(prompt=prompt):
                    async with semaphore:
                        return await self.generate_async(session, self.base_url, prompt)
                tasks.append(sem_task(prompt))
            results = []
            for future in asyncio.as_completed(tasks):
                result = await future
                results.append(result)
            return results
    
    def parallel(self, prompts, max_concurrent):
        results = asyncio.run(
            self.parallel_generate(prompts, max_concurrent=max_concurrent)
        )
        return results