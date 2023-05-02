"""
Computes surprisals at each checkpoint for a set of examples.
Outputs surprisal curves in batches (shape: n_tokens, n_checkpoints):
[output_dir]/surprisals_tokens_[start]-[end].npy
Sample usage:

python3 lm-learning-curves/get_autoregressive_surprisals.py \
--model_dir="models/gpt2_0" --batch_size=32 \
--sequences_file="datasets_tokenized_split/en_tokenized_eval_100000.txt" \
--output_dir="surprisals/gpt2_0" --max_sequences=100000
--save_tokens_batch_size=1270000

"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from utils.model_utils import load_model, get_autoregressive_surprisals
from utils.data_utils import get_checkpoints, get_examples

def create_parser():
    parser = argparse.ArgumentParser()
    # The main model directory, potentially containing checkpoint directories.
    parser.add_argument('--model_dir', type=str, required=True)
    # Only gpt2.
    parser.add_argument('--model_type', type=str, default="gpt2")
    parser.add_argument('--output_dir', type=str, required=True)
    # Tokenized examples. Each line should be a space-separated list of token ids.
    # Should be truncated and include special tokens (CLS, SEP), but un-padded.
    parser.add_argument('--sequences_file', type=str, required=True)
    parser.add_argument('--max_sequences', type=int, default=100000)
    parser.add_argument('--max_seq_len', type=int, default=128)
    # Batch size when running the model.
    parser.add_argument('--batch_size', type=int, default=32)
    # Save example surprisal curves in batches (curve per token).
    parser.add_argument('--save_tokens_batch_size', type=int, default=1270000)
    parser.add_argument('--hf_cache', type=str, default='hf_cache')
    return parser


def main(args):
    print("Identifying checkpoints.")
    checkpoints = get_checkpoints(args.model_dir)
    print("Checkpoints: {}".format(checkpoints))

    print("Loading examples.")
    examples = get_examples(args.sequences_file, max_examples=args.max_sequences)
    n_token_surprisals = 0
    for example_i, example in enumerate(examples):
        if len(example) > args.max_seq_len:
            examples[example_i] = example[:args.max_seq_len]
        n_token_surprisals += len(example) - 1 # No surprisal for first token in each example, because no prediction.

    # Run for all checkpoints.
    os.makedirs(args.output_dir, exist_ok=True)
    for checkpoint in checkpoints:
        print("Running checkpoint: {}.".format(checkpoint))
        config, tokenizer, model = load_model(args.model_dir, args.model_type,
                checkpoint=checkpoint, cache_dir=args.hf_cache, override_for_hidden_states=False)
        surprisals = get_autoregressive_surprisals(model, examples, args.batch_size, tokenizer)
        assert n_token_surprisals == surprisals.shape[0]
        del config, tokenizer, model
        outpath = os.path.join(args.output_dir, "surprisals_checkpoint-{}.npy".format(checkpoint))
        np.save(outpath, surprisals, allow_pickle=False)

    # Re-batch, to save per example instead of per checkpoint.
    print("Re-batching per example instead of per checkpoint.")
    for start_i in tqdm(range(0, n_token_surprisals, args.save_tokens_batch_size)):
        end_i = min(start_i+args.save_tokens_batch_size, n_token_surprisals)
        tokens_batch = np.zeros((end_i-start_i, len(checkpoints)))
        for checkpoint_i, checkpoint in enumerate(checkpoints):
            inpath = os.path.join(args.output_dir, "surprisals_checkpoint-{}.npy".format(checkpoint))
            checkpoint_surprisals = np.load(inpath)[start_i:end_i]
            tokens_batch[:, checkpoint_i] = checkpoint_surprisals
        outpath = os.path.join(args.output_dir, "surprisals_tokens_{0}-{1}.npy".format(start_i, end_i))
        np.save(outpath, tokens_batch, allow_pickle=False)
    print("Saved; deleting checkpoint batches.")
    for checkpoint in checkpoints:
        path = os.path.join(args.output_dir, "surprisals_checkpoint-{}.npy".format(checkpoint))
        os.remove(path)

    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
