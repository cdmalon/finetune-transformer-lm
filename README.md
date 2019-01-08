# finetune-transformer-lm

Code and model for the paper "Improving Language Understanding by Generative Pre-Training"

## Entailment

This modified version of the code supports the option "--dataset entailment",
to train models for entailment problems such as SNLI, following the
description in the paper.

Training input is expected to be in files named `train.premise`,
`train.hypothesis`, and `train.label`.  These files should use one line per
example.  The premise and hypothesis will be tokenized by Spacy.
Each label should be 0, 1, or 2 (ESIM convention is to use
0 for entailment, 1 for neutral, and 2 for contradiction).  Similarly,
development and testing sets should be put in files named `dev.premise`,
`test.premise`, etc.  These nine files are expected in `data_dir`.

Train with a command like:
```
python train.py --dataset entailment --desc entailment --submit --analysis --data_dir /path/to/data --n_gpu 3 --submission_dir output/submission --save_dir output/save --log_dir output/log
```

I've also added a prediction script, allowing you to obtain model predictions
separately from the training.  If the test files are data/test.premise, etc.,
then the command is like:
```
python predict.py --desc entailment --dataset entailment --model_file output/save/entailment/best_params.jl --test_prefix data/test --n_ctx 348 --result_file result.tsv
```

To run the prediction, you need to supply the amount of context that the model
was trained with, as the value of the `--n_ctx` option.  Generally, this
depends on the lengths of the examples in your training set.  If you didn't
remember this value from training time, you can compute it from the saved
model by taking the number of entries in the embedding matrix, and subtracting
the size of the vocabulary (40478, from `encoder_bpe_40000.json`) and the
number of special tokens (3), as these remaining embeddings are used
to encode each of the positions up to `n_ctx`.  The number of entries in the
embedding matrix will be reported in the error message you get if you choose
the wrong value.

-- Christopher Malon (cdmalon)

## ROCStories

Currently this code implements the ROCStories Cloze Test result reported in the paper by running:
`python train.py --dataset rocstories --desc rocstories --submit --analysis --data_dir [path to data here]`

Note: The code is currently non-deterministic due to various GPU ops. The median accuracy of 10 runs with this codebase (using default hyperparameters) is 85.8% - slightly lower than the reported single run of 86.5% from the paper. 

The ROCStories dataset can be downloaded from the associated [website](http://cs.rochester.edu/nlp/rocstories/).
