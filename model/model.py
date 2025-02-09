'''
model.py
'''
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer

# opensource_model
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
