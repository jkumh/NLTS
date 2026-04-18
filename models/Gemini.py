from data.serialize import serialize_arr, SerializerSettings
import openai
import tiktoken
import numpy as np
from jax import grad,vmap

# tokenizer better for single (1|scale 2|differ nonstationary )
# tokenizer for multi-
# use word or pic - 

def tokenize_fn(str,model):
    """
    Retrieve the token IDs for a string for a specific GPT model.

    Args:
        str (list of str): str to be tokenized.
        model (str): Name of the LLM model.

    Returns:
        list of int: List of corresponding token IDs.
    """
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    return encoding.encode(str)

def get_other_allowed_ids(strs,model):
    """
    Retrieve the token IDs for a given list of strings for a specific GPT model.

    Args:
        strs (list of str): strs to be converted.
        model (str): Name of the LLM model.

    Returns:
        list of int: List of corresponding token IDs.
    """
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    ids = []
    for s in strs:
        id = encoding.encode(s)
        ids.extend(id)
    return ids

def other_completion_fn(model, input_str, steps, settings, num_samples, temp):
    """
    Generate text completions from GPT using OpenAI's API.

    Args:
        model (str): Name of the GPT-3 model to use.
        input_str (str): Serialized input time series data.
        steps (int): Number of time steps to predict.
        settings (SerializerSettings): Serialization settings.
        num_samples (int): Number of completions to generate.
        temp (float): Temperature for sampling.

    Returns:
        list of str: List of generated samples.
    """
    avg_tokens_per_step = len(tokenize_fn(input_str, model)) / len(input_str.split(settings.time_sep))
    
    # Define logit bias to prevent GPT-3 from producing unwanted tokens
    logit_bias = {}
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)] 
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0]  # Remove empty tokens like an implicit plus sign
    
    chatgpt_sys_message = "You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas."
    extra_input = "Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n"
    
    all_choices = []  # List to collect all responses
    
    for _ in range(num_samples):
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": chatgpt_sys_message},
                {"role": "user", "content": extra_input + input_str + settings.time_sep}
            ],
            max_tokens=int(avg_tokens_per_step * steps),
            temperature=temp,
            logit_bias=logit_bias,
            n=1,  # Request only one response per call
        )
        
        # Append the generated choices to the all_choices list
        all_choices.extend([choice.message.content for choice in response.choices])
    
    return all_choices

