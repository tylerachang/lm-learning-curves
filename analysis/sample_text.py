"""
Generates text from randomly sampled prompts, using different model checkpoints.
Samples prompts from the annotator examples. Outputs a tsv.
"""

import os
from tqdm import tqdm
import codecs
import random
import pandas as pd

import sys
sys.path.append('lm-learning-curves')
from utils.data_utils import get_checkpoints
from utils.data_utils import get_examples as get_file_examples
from utils.annotator import CurveAnnotator
from utils.model_utils import generate_text, load_model

MODEL_DIR = 'models/gpt2_0'
MODEL_TYPE = 'gpt2'
CHECKPOINT_STEPS = [0, 100, 1000, 10000, 100000, 1000000]
N_PROMPTS = 100
MIN_PROMPT_LENGTH = 0
MAX_PROMPT_LENGTH = 64
TEMPERATURE = 0.30
MAX_SEQ_LEN = 128
OUTPATH = 'sample_responses_gpt2_0.tsv'
# Loads from this path, or samples from annotator examples then saves here.
PROMPT_PATH = 'sample_prompts.txt'  # Tokenized prompts.


# Get prompts.
if os.path.isfile(PROMPT_PATH):
    prompts = get_file_examples(PROMPT_PATH)
    print('Loaded {} prompts.'.format(len(prompts)))
else:
    # Sample prompts.
    sequences_path = 'datasets_tokenized_split/en_tokenized_eval_100000.txt'
    annotator = CurveAnnotator('annotators/gpt2_0')
    examples = annotator.get_examples(sequences_path)
    del annotator
    random.shuffle(examples)
    # Select in order after shuffling.
    prompts = []
    for example in examples:
        if len(example) >= MIN_PROMPT_LENGTH and len(example) <= MAX_PROMPT_LENGTH:
            prompts.append(example)
            if len(prompts) >= N_PROMPTS:
                break
    if len(prompts) < N_PROMPTS:
        print('WARNING: not enough prompts satisfying min and max prompt length.')
    # Save to outpath.
    outfile = codecs.open(PROMPT_PATH, 'wb', encoding='utf-8')
    for prompt in prompts:
        outfile.write(' '.join([str(id) for id in prompt]))
        outfile.write('\n')
    outfile.close()
    print('Saved {} prompts.'.format(len(prompts)))

# Run for each desired checkpoint.
dataframe = pd.DataFrame()
for checkpoint_step in CHECKPOINT_STEPS:
    # Get nearest checkpoint.
    checkpoints = get_checkpoints(MODEL_DIR) # Already sorted.
    if checkpoint_step in checkpoints:
        print("Using checkpoint step: {}".format(checkpoint_step))
    else:
        checkpoint_step = min(checkpoints, key=lambda x: abs(x-checkpoint_step))
        print("Nearest checkpoint step: {}".format(checkpoint_step))
    # Load model.
    print("Loading config, tokenizer, and model...")
    config, tokenizer, model = load_model(MODEL_DIR, MODEL_TYPE,
            checkpoint=checkpoint_step, cache_dir='hf_cache', override_for_hidden_states=False)
    # Run for each prompt.
    for prompt in tqdm(prompts):
        row_dict = dict()
        row_dict['model'] = MODEL_DIR
        row_dict['checkpoint'] = checkpoint_step
        row_dict['temperature'] = TEMPERATURE
        row_dict['prompt'] = tokenizer.decode(prompt)
        response = generate_text(model, prompt, tokenizer,
                max_seq_len=MAX_SEQ_LEN, temperature=TEMPERATURE)
        row_dict['response'] = tokenizer.decode(response)
        dataframe = dataframe.append(row_dict, ignore_index=True)
dataframe.to_csv(OUTPATH, sep='\t', index=False, encoding='utf-8')
print('Done.')
