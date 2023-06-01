import torch 

import argparse
import os
import sys
import yaml 
from tqdm import tqdm
import json 

sys.path.append(os.environ.get("SAFARI_PATH", "."))

from safari.models.sequence.long_conv_lm import ConvLMHeadModel

from transformers import AutoTokenizer, GPT2LMHeadModel
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import GPT2Tokenizer

try:
    from tokenizers import Tokenizer  
except:
    pass

# https://github.com/openai/gpt-2/issues/131#issuecomment-492786058
def preprocess(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


class LAMBADA:
    "LAMBADA (OpenAI) benchmark"
    def __init__(self, data_dir=None, use_stop_filter:bool=False):
        data_dir = os.environ.get("DATA_DIR", data_dir)
        lambada_path = os.path.join(data_dir + "/lambada/lambada_openai/lambada_test.jsonl")
        self.data = [preprocess(json.loads(line)['text']) for line in open(lambada_path)] 
        self.use_stop_filter = use_stop_filter 
    
    def run(self, model_cfg, ckpt_path):
        
        model, tokenizer = self.load_model(model_cfg, ckpt_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        if isinstance(tokenizer, Tokenizer):
            vocab_size = tokenizer.get_vocab_size()
        else:
            vocab_size = tokenizer.vocab_size

        stop_filter = torch.zeros(vocab_size, device=device)
        if self.use_stop_filter:
            token_to_idx = {tokenizer.decode([i]):i for i in range(vocab_size)}
            for word in STOP_WORDS:
                if ' '+word in token_to_idx:
                    stop_filter[token_to_idx[' '+word]] = -float('inf')
            
        results = []
        for prompt in tqdm(self.data):
            target = prompt.split(" ")[-1]
            
            if isinstance(tokenizer, Tokenizer):
                tokenized_prompt = tokenizer.encode(prompt).ids
                target_tokenized = tokenizer.encode(' '+target).ids
            else:
                tokenized_prompt = tokenizer.encode(prompt)
                target_tokenized = tokenizer(' '+target)['input_ids']
            out = model(torch.tensor([tokenized_prompt]).to(device=device))
            
            if type(out) == tuple: out = out[0]
            logits = out.logits[0][:-1, :vocab_size] # seq_len - 1, vocab_size

            logits = logits + stop_filter[None]
            preds = logits.argmax(-1)            
            acc = all([pred == answer for pred, answer 
                       in zip(preds[-len(target_tokenized):], target_tokenized)
                    ]
                )
            results.append(acc)
            
        print(f"Accuracy {torch.tensor(results).float().mean().item()*100:4.2f}")
        
            
    def load_model(self, model_cfg, ckpt_path):
        config = yaml.load(open(model_cfg, 'r'), Loader=yaml.FullLoader)
        model = ConvLMHeadModel(**config['model_config'])
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
        if config['tokenizer_name'] == 'gpt2':
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            tokenizer = None 
        return model, tokenizer
        
        
if __name__ == "__main__":
    
    SAFARI_PATH = os.getenv('SAFARI_PATH', '.')
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data",
        help="Path to data",
    )    
    
    parser.add_argument(
        "--model_cfg",
        default=f"{SAFARI_PATH}/configs/evals/hyena_small_150b.yaml",
    )
    
    parser.add_argument(
        "--ckpt_path",
        default=f"",
        help="Path to model state dict checkpoint"
    )
        
    parser.add_argument(
        "--stop_word_filter",
        type=bool,
        default=False,
        help="Filter out stop words",
    )
        
    args = parser.parse_args()
        
    task = LAMBADA(data_dir=args.data_dir, use_stop_filter=args.stop_word_filter)
    task.run(args.model_cfg, args.ckpt_path)