from genai.model import Credentials, Model
from genai.schemas import GenerateParams
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import requests

def make_genai_request(engine, api_key, prompts, decoding_method='sample', max_new_tokens=128, min_new_tokens=1,
                       stream=False, temperature=0.7, top_k=50, top_p=1, stop_sequences=None):
    params = GenerateParams(decoding_method=decoding_method, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                            stream=stream, temperature=temperature, top_k=top_k, top_p=top_p, stop_sequences=stop_sequences)
    creds = Credentials(api_key, api_endpoint="https://bam-api.res.ibm.com/v1")
    chat = Model(engine, params=params, credentials=creds)
    responses = chat.generate(prompts)
    results = []
    for response in responses:
        data = {
            "prompt": response.input_text,
            "response": response.generated_text,
        }
        results.append(data)        
    return results

def count_string_tokens(string):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    token_count = torch.tensor(tokenizer.encode(string)).unsqueeze(0).shape[1]
    return token_count


def make_hf_api_request(engine, api_key, prompts, decode_sample, max_new_tokens=128, temperature=0.7, top_k=50, top_p=1):
    #API_ENDPOINT = "https://api-inference.huggingface.co/models/" + engine
    API_ENDPOINT = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    print(API_ENDPOINT)
    headers = {"Authorization": f"Bearer {api_key}"}
    responses = []
    for prompt in prompts:
        #payload = {"inputs": prompt, 
                   #"parameters": {"do_sample": decode_sample, "max_new_tokens": 250, "temperature": temperature, "top_k": top_k, "top_p": top_p, "return_full_text": False},  
                   #"options": {"wait_for_model": True}}
        
        payload = {"inputs": prompt, 
                   "parameters": {"do_sample": decode_sample, "max_new_tokens": 250, "temperature": temperature, "top_k": top_k, "top_p": top_p},  
                   "options": {"wait_for_model": True}}
        responses.append(hf_query(payload, API_ENDPOINT, headers))
    return responses

def hf_query(prompt, endpoint, headers):
    data = json.dumps(prompt)
    response = requests.request("POST", endpoint, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

