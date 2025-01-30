# lm-learning-curves
Code for the paper, [Characterizing Learning Curves During Language Model Pre-Training: Learning, Forgetting, and Stability](https://arxiv.org/abs/2308.15419) (2024).
Includes code for extracting and annotating learning curves from language models during pre-training.

## Pre-train language models.
Language models are pre-trained using: https://github.com/tylerachang/word-acquisition-language-models.
See original repository for details.
<pre>
git clone https://github.com/tylerachang/lm-learning-curves.git
git clone https://github.com/tylerachang/word-acquisition-language-models.git
</pre>
Requirements in requirements_pretraining.txt (Python 3.7.12).
Assume raw text dataset is in: datasets/en.txt (in our case, 128M lines).
First, sample lines to train the tokenizer:
<pre>
mkdir datasets
printf "en.txt\t128000000\n" > datasets/input_line_counts.tsv
printf "en.txt\t10000000\n" > datasets/tokenizer_line_counts.tsv
python3 word-acquisition-language-models/scripts/sample_lines.py --input_dir="datasets" \
--output_path="datasets/tokenizer_sampled_lines.txt" \
--output_line_counts="datasets/tokenizer_line_counts.tsv" \
--input_line_counts="datasets/input_line_counts.tsv"
</pre>
Train the tokenizer:
<pre>
mkdir tokenizer
python3 word-acquisition-language-models/scripts/train_spm_tokenizer.py \
--input_file="datasets/tokenizer_sampled_lines.txt" \
--output="tokenizer/spm" \
--vocab_size=50000 --sample_size=10000000
python3 word-acquisition-language-models/scripts/convert_spm_to_hf_tokenizer.py \
--input="tokenizer/spm.model" \
--output_dir="hf_tokenizer" \
--multiple_of=2048
</pre>
Tokenize the pre-training dataset:
<pre>
python3 word-acquisition-language-models/scripts/tokenize_dataset.py \
--tokenizer="hf_tokenizer" \
--input_file="datasets/en.txt" \
--output_file="datasets_tokenized/en_tokenized.txt" \
--max_segments=-1 --max_seq_len=128
</pre>
Split into train/eval/test:
<pre>
python3 word-acquisition-language-models/scripts/split_datasets.py --dataset_dir="datasets_tokenized" \
--train_proportion=0.80 --eval_proportion=0.10 --test_proportion=0.10
head -10000 datasets_tokenized_split/en_tokenized_eval.txt > datasets_tokenized_split/en_tokenized_eval_10000.txt
head -100000 datasets_tokenized_split/en_tokenized_eval.txt > datasets_tokenized_split/en_tokenized_eval_100000.txt
</pre>
Pre-train (one GPU device):
<pre>
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name="hf_tokenizer" \
--config_name="lm-learning-curves/gpt2_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file="datasets_tokenized_split/en_tokenized_eval_10000.txt" \
--per_device_train_batch_size=128 --gradient_accumulation_steps=2 \
--per_device_eval_batch_size=64 \
--evaluation_strategy="steps" --overwrite_save_strategy="exponential" \
--eval_steps=1000 \
--max_steps=1000000 \
--warmup_steps=10000 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file="datasets_tokenized_split/en_tokenized_train.txt" \
--seed=42 \
--output_dir="models/gpt2_0"
</pre>
To use multiple GPU devices, replace the first line above with (and update per-device batch size):
<pre>
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29401 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
</pre>
To test a language model checkpoint for text generation:
<pre>
python3 lm-learning-curves/generate_text.py \
--model_dir="models/gpt2_0" --model_type="gpt2" \
--checkpoint=100000 --max_seq_len=128 --temperature=0.0 \
--text="This is"
</pre>

## Extract learning curves.
Extract surprisal curves for evaluation examples.
Requirements in requirements_pretraining.txt (Python 3.7.12).
Saving in batches of 1270000 because each sequence (length 128) has surprisals for 127 tokens.
<pre>
python3 lm-learning-curves/get_autoregressive_surprisals.py \
--model_dir="models/gpt2_0" --batch_size=32 \
--sequences_file="datasets_tokenized_split/en_tokenized_eval_100000.txt" \
--output_dir="surprisals/gpt2_0" --max_sequences=100000 \
--save_tokens_batch_size=1270000
</pre>

## Annotate learning curves.
This script runs the following:
* Samples the 1M evaluation tokens in context (saved in annotators/gpt2_0/tokens_mask.npy).
* Fits GAM curves.
* Gets age of acquisition (AoA) scores using sigmoids and GAMs.
* Computes n-gram surprisals.
* Computes the contextual diversity score for each token (frequency-adjusted).
* Annotates POS tags.
* Runs UMAP to embed the learning curves in 2D.

Requirements in requirements_analysis.txt (Python 3.8.17; using a different Python version for compatibility with pyGAM).
<pre>
python3 lm-learning-curves/annotate_curves.py \
--training_file="datasets_tokenized_split/en_tokenized_train.txt" \
--sequences_file="datasets_tokenized_split/en_tokenized_eval_100000.txt" \
--annotator_cache="annotators/gpt2_0" \
--surprisals_dir="surprisals/gpt2_0" --per_sequence=10
</pre>

## Run analyses.
Sample text completions, run linear regressions to predict learning curve metrics, and generate figures.
Note that relative paths are hard-coded in these files.
Assumes that the code above has been run for five pre-training runs (gpt2_[0-4]).
<pre>
python3 lm-learning-curves/analysis/sample_text.py
python3 lm-learning-curves/analysis/run_regressions.py
python3 lm-learning-curves/analysis/generate_figures.py
</pre>
Sampling text completions can also be run with generate_text.py (see pre-training section above).

## Citation.
<pre>
@article{chang-etal-2024-characterizing-learning,
  title={Characterizing Learning Curves During Language Model Pre-Training: Learning, Forgetting, and Stability},
  author={Tyler A. Chang and Zhuowen Tu and Benjamin K. Bergen},
  journal={Transactions of the Association for Computational Linguistics},
  year={2024},
  url={https://arxiv.org/abs/2308.15419}
}
</pre>
