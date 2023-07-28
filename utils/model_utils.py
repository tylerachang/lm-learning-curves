"""
Transformer model utilities for the monolingual pre-training analyses.
"""

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoConfig, AutoTokenizer, AlbertTokenizer,
    AutoModelForCausalLM, AutoModelForMaskedLM, GPT2LMHeadModel)

from utils.model_overrides import GPT2ModelOverride

"""
Loads a model from a directory.
If checkpoint is provided (an integer for number of steps), loads that
checkpoint; otherwise, loads the final model.
Returns the config, tokenizer, and model.
The model_type is bert or gpt2.
Places model on cuda if available.
"""
def load_model(model_dir, model_type, checkpoint=None, tokenizer_path_override=None,
               config_path_override=None, cache_dir="hf_cache", override_for_hidden_states=True):
    model_type = model_type.lower()
    # Load config.
    config_path = os.path.join(model_dir, "config.json") if config_path_override is None else config_path_override
    config = AutoConfig.from_pretrained(config_path, cache_dir=cache_dir)
    # Load tokenizer.
    tokenizer_path = model_dir if tokenizer_path_override is None else tokenizer_path_override
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir)
    except:
        # If passing in a raw tokenizer model file, assume ALBERT sentencepiece model.
        print("WARNING: attempting to use local sentencepiece model file as tokenizer.")
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path, cache_dir=cache_dir)
    # Overwrite special token ids in the configs.
    config.bos_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.sep_token_id
    config.pad_token_id = tokenizer.pad_token_id
    if model_type == "bert":
        max_seq_len = config.max_position_embeddings
    elif model_type == "gpt2":
        max_seq_len = config.n_positions
    # Load model.
    if checkpoint is not None:
        model_dir = os.path.join(model_dir, "checkpoint-" + str(checkpoint))
    print("Loading from directory: {}".format(model_dir))
    if model_type == "gpt2": # GPT2LMHeadModel.
        model = AutoModelForCausalLM.from_pretrained(model_dir, config=config, cache_dir=cache_dir)
        # Override so that final layer hidden states are saved before the final
        # layer norm, to be comparable with other layers.
        if type(model) == GPT2LMHeadModel and override_for_hidden_states:
            print("Overriding model to save final layer hidden states correctly.")
            overridden_gpt2 = GPT2ModelOverride.from_pretrained(model_dir, config=config, cache_dir=cache_dir)
            model.transformer = overridden_gpt2
        else:
            print("WARNING: may need to override model to save final layer hidden states correctly.")
    elif model_type == "bert": # BertForMaskedLM.
        model = AutoModelForMaskedLM.from_pretrained(model_dir, config=config, cache_dir=cache_dir)
    model.resize_token_embeddings(len(tokenizer))
    # Load onto GPU.
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return config, tokenizer, model


"""
Convert a list of examples (token id lists) to a batch.
Inputs should already include CLS and SEP tokens. Because this function
does not know the maximum sequence length, examples should already be truncated.
All sequences will be padded to the length of the longest example, so this
should be called per batch.
Labels are set to None, assuming these examples are only used for
representation analysis.
"""
def prepare_tokenized_examples(tokenized_examples, tokenizer):
    # Convert into a tensor.
    tensor_examples = [torch.tensor(e, dtype=torch.long) for e in tokenized_examples]
    # Shape: (batch_size, sequence_len).
    input_ids = pad_sequence(tensor_examples, batch_first=True,
                             padding_value=tokenizer.pad_token_id)
    attention_mask = input_ids != tokenizer.pad_token_id
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": None}
    if torch.cuda.is_available():
        inputs["input_ids"] = inputs["input_ids"].cuda()
        inputs["attention_mask"] = inputs["attention_mask"].cuda()
    return inputs


"""
Output the hidden states given examples (lists of token_ids). Assumes examples
have been truncated. Outputs tensor of shape n_tokens x hidden_size (where
n_tokens concatenates tokens in all examples). Handles batching and example
tensorizing.
"""
def get_hidden_states(model, examples, batch_size, tokenizer, layer):
    # Create batches.
    batches = []
    i = 0
    while i+batch_size <= len(examples):
        batches.append(examples[i:i+batch_size])
        i += batch_size
    if len(examples) % batch_size != 0:
        batches.append(examples[i:])
    # Run evaluation.
    model.eval()
    with torch.no_grad():
        all_hidden_states = None
        for batch_i in tqdm(range(len(batches))):
            # Adds padding.
            inputs = prepare_tokenized_examples(batches[batch_i], tokenizer)
            # Run model.
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            labels=inputs["labels"],
                            output_hidden_states=True, return_dict=True)
            hidden_states = outputs["hidden_states"][layer].detach()
            del outputs # Delete before the next batch runs.
            hidden_size = hidden_states.shape[-1]
            if all_hidden_states is None:
                all_hidden_states = np.zeros((0, hidden_size))
            hidden_states = hidden_states.reshape(-1, hidden_size)
            # Remove pad tokens.
            # Shape: n_tokens x hidden_size
            hidden_states = hidden_states[inputs["attention_mask"].flatten(), :]
            hidden_states = hidden_states.detach().cpu() # Send to CPU so not all need to be held on GPU.
            all_hidden_states = np.concatenate((all_hidden_states, hidden_states), axis=0)
    print("Extracted {} hidden states.".format(all_hidden_states.shape[0]))
    return all_hidden_states


"""
Output the surprisals given examples (lists of token_ids). Assumes examples
have been truncated. Outputs tensor of shape (n_tokens) (where
n_tokens concatenates tokens in all examples). The first token in each example
has no surprisal because there is no prediction for the first token.
Handles batching and example tensorizing.
"""
def get_autoregressive_surprisals(model, examples, batch_size, tokenizer):
    # Create batches.
    batches = []
    i = 0
    while i+batch_size <= len(examples):
        batches.append(examples[i:i+batch_size])
        i += batch_size
    if len(examples) % batch_size != 0:
        batches.append(examples[i:])
    # Run evaluation.
    model.eval()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=-1)
        all_surprisals = []
        for batch_i in tqdm(range(len(batches))):
            # Adds padding.
            inputs = prepare_tokenized_examples(batches[batch_i], tokenizer)
            # Run model.
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            labels=inputs["labels"],
                            output_hidden_states=True, return_dict=True)
            # Note: logits pre-softmax.
            # Shape: (batch_size, seq_len, vocab_size).
            logits = outputs["logits"].detach()
            logits = logits[:, :-1, :] # Ignore last prediction, because no corresponding label.
            vocab_size = logits.shape[-1]
            del outputs
            # Surprisals for these labels.
            labels = inputs["input_ids"][:, 1:]
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100
            labels = labels.flatten()
            labels = labels[labels != -100]
            logits = logits.reshape(-1, vocab_size)
            probs = softmax(logits)[labels != -100, :]
            label_probs = torch.gather(probs, dim=-1, index=labels.reshape(-1, 1)).flatten()
            surprisals = -1.0 * torch.log2(label_probs).cpu()
            all_surprisals.append(np.array(surprisals))
        all_surprisals = np.concatenate(all_surprisals, axis=0)
    print("Computed {} surprisals.".format(all_surprisals.shape[0]))
    return all_surprisals


"""
Autoregressively generates token ids, given an input prompt (list of token ids).
Returns the generated token ids.
"""
def generate_text(model, input_ids, tokenizer, max_seq_len=128, temperature=0.0):
    if len(input_ids) > max_seq_len:
        return input_ids[:max_seq_len]
    # Iteratively fill in tokens.
    output_ids = []
    while len(input_ids) + len(output_ids) < max_seq_len:
        inputs = prepare_tokenized_examples([input_ids + output_ids], tokenizer)
        # Note: here, labels are None (because not computing loss).
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"],
                        output_hidden_states=False, return_dict=True)
        # Note: logits pre-softmax.
        logits = outputs["logits"].detach()
        logits = logits[0] # First example only.
        del outputs
        # Logits for the last token.
        index_logits = logits[-1, :]
        softmax = torch.nn.Softmax(dim=0)
        probs = softmax(index_logits)
        if temperature > 0.0:
            probs = torch.pow(probs, 1.0 / temperature)
            fill_id = torch.multinomial(probs, 1).item() # Automatically rescales probs.
        else:
            fill_id = torch.argmax(probs).item()
        output_ids.append(fill_id)
    return output_ids
