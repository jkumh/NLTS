import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from jax import grad, vmap
from data.serialize import serialize_arr, SerializerSettings

MODEL_NAME = 'language_models/gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model_obj = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model_obj.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_obj.to(device)

def tokenize_fn(text, model=None):
    """Using GPT-2 tokenizer to tokenize text, return list of token ids"""
    return tokenizer.encode(text)

def get_allowed_ids(strs, model=None):
    ids = []
    for s in strs:
        token_ids = tokenizer.encode(s, add_special_tokens=False)
        ids.extend(token_ids)
    return list(set(ids))

def gpt2_completion_fn(input_str, steps, settings, num_samples, temp,model=None,**kwargs):
    """
    Using local GPT-2 to simulate the generation logic of ChatCompletion
    """
    #chatgpt_sys_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
    #extra_input = "Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
    
    #full_prompt = f"{chatgpt_sys_message}\n\nUser: {extra_input}{input_str}{settings.time_sep}"
    full_prompt=f'Sequence:{input_str}{settings.time_sep}'
    input_ids = tokenizer.encode(full_prompt, return_tensors='pt').to(device)

    max_position_embeddings = getattr(model_obj.config, "n_positions", 1024)


    input_tokens_count = len(tokenizer.encode(input_str))
    avg_tokens_per_step = input_tokens_count / len(input_str.split(settings.time_sep))
    max_new_tokens = int(avg_tokens_per_step * steps)


    max_input_length = max_position_embeddings - max_new_tokens
    
    if input_ids.shape[1] > max_input_length:
        print(f"Warning: Input sequence too long ({input_ids.shape[1]}). Truncating to {max_input_length}.")

        input_ids = input_ids[:, -max_input_length:]
   
    outputs = model_obj.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_samples,
        temperature=temp,
        do_sample=True if temp > 0 else False,
        pad_token_id=tokenizer.eos_token_id,
    )

    completions = []
    for output in outputs:
        generated_text = tokenizer.decode(output[len(input_ids[0]):], skip_special_tokens=True)
        completions.append(generated_text)
    
    return completions

def gpt2_nll_fn(input_arr, target_arr, settings:SerializerSettings, transform, count_seps=True, temp=1, model=None, **kwargs):
    """
    Calculate the NLL of the target sequence under local GPT-2, including context truncation if necessary. The input and target sequences are transformed and serialized according to the provided settings before being fed into the model. The function handles cases where the combined length of input and target exceeds the model's maximum context length by prioritizing the target sequence and truncating the input as needed. The NLL is computed based on the log probabilities of the target tokens, with an option to include or exclude separator tokens in the calculation. Additionally, a Jacobian determinant adjustment is applied to account for transformations of the target sequence.
    """

    # serialize input and target sequences according to settings
    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    
    # get model's max context length
    max_context_len = getattr(model_obj.config, "n_positions", 1024)
    
    # encode input and target to token ids
    input_ids = tokenizer.encode(input_str, add_special_tokens=False)
    target_ids = tokenizer.encode(target_str, add_special_tokens=False)
    
    total_len = len(input_ids) + len(target_ids)

    if total_len > max_context_len:
       
        allowed_input_len = max_context_len - len(target_ids)
        
        if allowed_input_len <= 0:
            
            input_ids = []
            target_ids = target_ids[-max_context_len:]
            print(f"Warning: Target too long, truncated target to {max_context_len}")
        else:
            input_ids = input_ids[-allowed_input_len:]
            # print(f"Warning: Sequence too long, truncated input to {allowed_input_len}")


    tokens = torch.tensor([input_ids + target_ids]).to(device)
    target_start_idx = len(input_ids) # target 从这个索引开始（在 shift 之前）
    # ----------------------

    with torch.no_grad():
        outputs = model_obj(tokens)
        logits = outputs.logits  

    log_probs = torch.log_softmax(logits / temp, dim=-1)

    shift_log_probs = log_probs[:, :-1, :]
    shift_labels = tokens[:, 1:]

    token_logprobs = torch.gather(shift_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs.cpu().numpy()[0]

    target_token_logprobs = token_logprobs[target_start_idx - 1:]
    
    target_tokens_str = [tokenizer.decode([t]) for t in target_ids]

    seps_mask = np.array([t == settings.time_sep for t in target_tokens_str])
    
    digits_bits = -target_token_logprobs[~seps_mask].sum()
    seps_bits = -target_token_logprobs[seps_mask].sum()
    
    BPD = digits_bits / len(target_arr)
    if count_seps:
        BPD += seps_bits / len(target_arr)
        
    transformed_nll = BPD - settings.prec * np.log(settings.base)

    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    
    return transformed_nll - avg_logdet_dydx