"""
Script to generate text, or fill in a masked token.
Sample usage:

python3 lm-learning-curves/generate_text.py \
--model_dir="models/gpt2_0" --model_type="gpt2" \
--checkpoint=100000 --max_seq_len=128 --temperature=0.0 \
--text="This is a"

python3 lm-learning-curves/generate_text.py \
--model_dir="models/bert_0" --model_type="bert" \
--checkpoint=100000 --max_seq_len=128 --temperature=0.0 \
--text="This is a [MASK]."

"""

import argparse
import torch
import random
from tqdm import tqdm

from utils.data_utils import get_checkpoints
from utils.model_utils import load_model, prepare_tokenized_examples

def create_parser():
    parser = argparse.ArgumentParser()
    # The model directory, possibly containing checkpoints too.
    # This should be generated from the pretraining scripts in:
    # https://github.com/tylerachang/word-acquisition-language-models
    parser.add_argument('--model_dir', required=True)
    # If different from the model directory.
    parser.add_argument('--tokenizer', default="")
    # gpt2 or bert.
    parser.add_argument('--model_type', default="gpt2")
    # Checkpoint step, or None to use final model.
    # Will round to the nearest checkpoint.
    parser.add_argument('--checkpoint', type=int, default=None)
    # The prefix for the generated text if gpt2, or the text with [MASK] to
    # be filled in for bert.
    parser.add_argument('--text', required=True)
    parser.add_argument('--max_seq_len', type=int, default=128)
    # If too low but nonzero, may throw an error.
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--hf_cache', default="hf_cache")
    return parser


def main(args):
    # Get nearest checkpoint.
    print("Finding checkpoints...")
    checkpoints = get_checkpoints(args.model_dir) # Already sorted.
    if args.checkpoint is None or len(checkpoints) == 0:
        checkpoint = None
        print("Using final model.")
    elif args.checkpoint in checkpoints:
        checkpoint = args.checkpoint
        print("Using checkpoint step: {}".format(checkpoint))
    else:
        checkpoint = min(checkpoints, key=lambda x: abs(x-args.checkpoint))
        print("Nearest checkpoint step: {}".format(checkpoint))

    print("Loading config, tokenizer, and model...")
    tokenizer_path_override = args.tokenizer if args.tokenizer else None
    config, tokenizer, model = load_model(args.model_dir, args.model_type,
            checkpoint=checkpoint, cache_dir=args.hf_cache,
            tokenizer_path_override=tokenizer_path_override,
            override_for_hidden_states=False)

    print("Encoding text...")
    curr_example = tokenizer.encode(args.text.strip(), add_special_tokens=False)
    curr_example.insert(0, tokenizer.cls_token_id)
    if args.model_type == "bert":
        id_to_fill = tokenizer.mask_token_id
        curr_example.append(tokenizer.sep_token_id)
    elif args.model_type == "gpt2":
        id_to_fill = tokenizer.pad_token_id
        # Fill with pad tokens for autoregressive.
        while len(curr_example) < args.max_seq_len:
            curr_example.append(tokenizer.pad_token_id)
    # Truncate if necessary.
    if len(curr_example) > args.max_seq_len:
        curr_example = curr_example[:args.max_seq_len]

    print("Generating tokens...")
    # Iteratively fill in tokens.
    while id_to_fill in curr_example:
        inputs = prepare_tokenized_examples([curr_example], tokenizer)
        # Note: here, labels are None (because not computing loss).
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"],
                        output_hidden_states=False, return_dict=True)
        # Note: logits pre-softmax.
        logits = outputs["logits"].detach()
        logits = logits[0] # First example only.
        del outputs
        potential_indices = [i for i in range(len(curr_example)) if curr_example[i] == id_to_fill]
        index_to_fill = min(potential_indices) # Fill in order.
        if args.model_type == "gpt2":
            index_logits = logits[index_to_fill-1, :]
        elif args.model_type == "bert":
            index_logits = logits[index_to_fill, :]
        softmax = torch.nn.Softmax(dim=0)
        probs = softmax(index_logits)
        if args.temperature > 0.0:
            probs = torch.pow(probs, 1.0 / args.temperature)
            fill_id = torch.multinomial(probs, 1).item() # Automatically rescales probs.
        else:
            fill_id = torch.argmax(probs).item()
        curr_example[index_to_fill] = fill_id

    print("\nOutput example: {}".format(curr_example))
    output = tokenizer.decode(curr_example)
    print("Decoded:")
    print(output)
    print("DONE.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
