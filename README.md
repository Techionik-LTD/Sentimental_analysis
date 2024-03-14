# Sentimental_analysis
sentimental_analysis model for feedback Analysis
<br>
Its a complete project related to sentimental analysis.
<br>
In this project firstly scrape the data for the url.
<br>
then preprocess the data.remove emojis article or etc...
<br>
then load the dataset into model and train the bert model into our dataset with hundred p[erscent accuracy.
<br>
after Training we push the model into huging face hub for make a flask_api 

Fine-tuning with BERT
Important: All results on the paper were fine-tuned on a single Cloud TPU, which has 64GB of RAM. It is currently not possible to re-produce most of the BERT-Large results on the paper using a GPU with 12GB - 16GB of RAM, because the maximum batch size that can fit in memory is too small. We are working on adding code to this repository which allows for much larger effective batch size on the GPU. See the section on out-of-memory issues for more details.

This code was tested with TensorFlow 1.11.0. It was tested with Python2 and Python3 (but more thoroughly with Python2, since this is what's used internally in Google).

The fine-tuning examples which use BERT-Base should be able to run on a GPU that has at least 12GB of RAM using the hyperparameters given.

Fine-tuning with Cloud TPUs
Most of the examples below assumes that you will be running training/evaluation on your local machine, using a GPU like a Titan X or GTX 1080.

However, if you have access to a Cloud TPU that you want to train on, just add the following flags to run_classifier.py or run_squad.py:

<b>--use_tpu=True \
  --tpu_name=$TPU_NAME</b>
Please see the Google Cloud TPU tutorial for how to use Cloud TPUs. Alternatively, you can use the Google Colab notebook "BERT FineTuning with Cloud TPUs".

On Cloud TPUs, the pretrained model and the output directory will need to be on Google Cloud Storage. For example, if you have a bucket named some_bucket, you might use the following flags instead:

  --output_dir=gs://some_bucket/my_output_dir/
The unzipped pre-trained model files can also be found in the Google Cloud Storage folder gs://bert_models/2018_10_18. For example:

export BERT_BASE_DIR=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12
Sentence (and sentence-pair) classification tasks
Before running this example you must download the GLUE data by running this script and unpack it to some directory $GLUE_DIR. Next, download the BERT-Base checkpoint and unzip it to some directory $BERT_BASE_DIR.

This example code fine-tunes BERT-Base on the Microsoft Research Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples and can fine-tune in a few minutes on most GPUs.

export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue
<b>
python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/mrpc_output/
</b>
You should see output like this:
<b>
***** Eval results *****
  eval_accuracy = 0.845588
  eval_loss = 0.505248
  global_step = 343
  loss = 0.505248</b>
This means that the Dev set accuracy was 84.55%. Small sets like MRPC have a high variance in the Dev set accuracy, even when starting from the same pre-training checkpoint. If you re-run multiple times (making sure to point to different output_dir), you should see results between 84% and 88%.

A few other pre-trained models are implemented off-the-shelf in run_classifier.py, so it should be straightforward to follow those examples to use BERT for any single-sentence or sentence-pair classification task.

Note: You might see a message Running train on CPU. This really just means that it's running on something other than a Cloud TPU, which includes a GPU.

Prediction from classifier
Once you have trained your classifier you can use it in inference mode by using the --do_predict=true command. You need to have a file named test.tsv in the input folder. Output will be created in file called test_results.tsv in the output folder. Each line will contain output for each sample, columns are the class probabilities.

export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue
export TRAINED_CLASSIFIER=/path/to/fine/tuned/classifier
<b>
python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/tmp/mrpc_output/</b>
SQuAD 1.1
The Stanford Question Answering Dataset (SQuAD) is a popular question answering benchmark dataset. BERT (at the time of the release) obtains state-of-the-art results on SQuAD with almost no task-specific network architecture modifications or data augmentation. However, it does require semi-complex data pre-processing and post-processing to deal with (a) the variable-length nature of SQuAD context paragraphs, and (b) the character-level answer annotations which are used for SQuAD training. This processing is implemented and documented in run_squad.py.

To run on SQuAD, you will first need to download the dataset. The SQuAD website does not seem to link to the v1.1 datasets any longer, but the necessary files can be found here:

train-v1.1.json
dev-v1.1.json
evaluate-v1.1.py
Download these to some directory $SQUAD_DIR.

The state-of-the-art SQuAD results from the paper currently cannot be reproduced on a 12GB-16GB GPU due to memory constraints (in fact, even batch size 1 does not seem to fit on a 12GB GPU using BERT-Large). However, a reasonably strong BERT-Base model can be trained on the GPU with these hyperparameters:

python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
The dev set predictions will be saved into a file called predictions.json in the output_dir:

python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./squad/predictions.json
Which should produce an output like this:

{"f1": 88.41249612335034, "exact_match": 81.2488174077578}
You should see a result similar to the 88.5% reported in the paper for BERT-Base.

If you have access to a Cloud TPU, you can train with BERT-Large. Here is a set of hyperparameters (slightly different than the paper) which consistently obtain around 90.5%-91.0% F1 single-system trained only on SQuAD:

python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME
For example, one random run with these parameters produces the following Dev scores:

{"f1": 90.87081895814865, "exact_match": 84.38978240302744}
If you fine-tune for one epoch on TriviaQA before this the results will be even better, but you will need to convert TriviaQA into the SQuAD json format.

SQuAD 2.0
This model is also implemented and documented in run_squad.py.

To run on SQuAD 2.0, you will first need to download the dataset. The necessary files can be found here:

train-v2.0.json
dev-v2.0.json
evaluate-v2.0.py
Download these to some directory $SQUAD_DIR.

On Cloud TPU you can run with BERT-Large as follows:

python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True
We assume you have copied everything from the output directory to a local directory called ./squad/. The initial dev set predictions will be at ./squad/predictions.json and the differences between the score of no answer ("") and the best non-null answer for each question will be in the file ./squad/null_odds.json

Run this script to tune a threshold for predicting null versus non-null answers:

python $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.json ./squad/predictions.json --na-prob-file ./squad/null_odds.json

Assume the script outputs "best_f1_thresh" THRESH. (Typical values are between -1.0 and -5.0). You can now re-run the model to generate predictions with the derived threshold or alternatively you can extract the appropriate answers from ./squad/nbest_predictions.json.

python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True \
  --null_score_diff_threshold=$THRESH
Out-of-memory issues
All experiments in the paper were fine-tuned on a Cloud TPU, which has 64GB of device RAM. Therefore, when using a GPU with 12GB - 16GB of RAM, you are likely to encounter out-of-memory issues if you use the same hyperparameters described in the paper.

The factors that affect memory usage are:

max_seq_length: The released models were trained with sequence lengths up to 512, but you can fine-tune with a shorter max sequence length to save substantial memory. This is controlled by the max_seq_length flag in our example code.

train_batch_size: The memory usage is also directly proportional to the batch size.

Model type, BERT-Base vs. BERT-Large: The BERT-Large model requires significantly more memory than BERT-Base.

Optimizer: The default optimizer for BERT is Adam, which requires a lot of extra memory to store the m and v vectors. Switching to a more memory efficient optimizer can reduce memory usage, but can also affect the results. We have not experimented with other optimizers for fine-tuning.

Using the default training scripts (run_classifier.py and run_squad.py), we benchmarked the maximum batch size on single Titan X GPU (12GB RAM) with TensorFlow 1.11.0:

System	Seq Length	Max Batch Size
BERT-Base	64	64
...	128	32
...	256	16
...	320	14
...	384	12
...	512	6
BERT-Large	64	12
...	128	6
...	256	2
...	320	1
...	384	0
...	512	0
Unfortunately, these max batch sizes for BERT-Large are so small that they will actually harm the model accuracy, regardless of the learning rate used. We are working on adding code to this repository which will allow much larger effective batch sizes to be used on the GPU. The code will be based on one (or both) of the following techniques:

Gradient accumulation: The samples in a minibatch are typically independent with respect to gradient computation (excluding batch normalization, which is not used here). This means that the gradients of multiple smaller minibatches can be accumulated before performing the weight update, and this will be exactly equivalent to a single larger update.

Gradient checkpointing: The major use of GPU/TPU memory during DNN training is caching the intermediate activations in the forward pass that are necessary for efficient computation in the backward pass. "Gradient checkpointing" trades memory for compute time by re-computing the activations in an intelligent way.

However, this is not implemented in the current release.

Using BERT to extract fixed feature vectors (like ELMo)
In certain cases, rather than fine-tuning the entire pre-trained model end-to-end, it can be beneficial to obtained pre-trained contextual embeddings, which are fixed contextual representations of each input token generated from the hidden layers of the pre-trained model. This should also mitigate most of the out-of-memory issues.

As an example, we include the script extract_features.py which can be used like this:

# Sentence A and Sentence B are separated by the ||| delimiter for sentence
# pair tasks like question answering and entailment.
# For single sentence inputs, put one sentence per line and DON'T use the
# delimiter.
echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' > /tmp/input.txt

python extract_features.py \
  --input_file=/tmp/input.txt \
  --output_file=/tmp/output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
This will create a JSON file (one line per line of input) containing the BERT activations from each Transformer layer specified by layers (-1 is the final hidden layer of the Transformer, etc.)

Note that this script will produce very large output files (by default, around 15kb for every input token).

If you need to maintain alignment between the original and tokenized words (for projecting training labels), see the Tokenization section below.

Note: You may see a message like Could not find trained model in model_dir: /tmp/tmpuB5g5c, running initialization to predict. This message is expected, it just means that we are using the init_from_checkpoint() API rather than the saved model API. If you don't specify a checkpoint or specify an invalid checkpoint, this script will complain.

Tokenization
For sentence-level tasks (or sentence-pair) tasks, tokenization is very simple. Just follow the example code in run_classifier.py and extract_features.py. The basic procedure for sentence-level tasks is:

Instantiate an instance of tokenizer = tokenization.FullTokenizer

Tokenize the raw text with tokens = tokenizer.tokenize(raw_text).

Truncate to the maximum sequence length. (You can use up to 512, but you probably want to use shorter if possible for memory and speed reasons.)

Add the [CLS] and [SEP] tokens in the right place.

Word-level and span-level tasks (e.g., SQuAD and NER) are more complex, since you need to maintain alignment between your input text and output text so that you can project your training labels. SQuAD is a particularly complex example because the input labels are character-based, and SQuAD paragraphs are often longer than our maximum sequence length. See the code in run_squad.py to show how we handle this.

Before we describe the general recipe for handling word-level tasks, it's important to understand what exactly our tokenizer is doing. It has three main steps:

Text normalization: Convert all whitespace characters to spaces, and (for the Uncased model) lowercase the input and strip out accent markers. E.g., John Johanson's, → john johanson's,.

Punctuation splitting: Split all punctuation characters on both sides (i.e., add whitespace around all punctuation characters). Punctuation characters are defined as (a) Anything with a P* Unicode class, (b) any non-letter/number/space ASCII character (e.g., characters like $ which are technically not punctuation). E.g., john johanson's, → john johanson ' s ,

WordPiece tokenization: Apply whitespace tokenization to the output of the above procedure, and apply WordPiece tokenization to each token separately. (Our implementation is directly based on the one from tensor2tensor, which is linked). E.g., john johanson ' s , → john johan ##son ' s ,

The advantage of this scheme is that it is "compatible" with most existing English tokenizers. For example, imagine that you have a part-of-speech tagging task which looks like this:

Input:  John Johanson 's   house
Labels: NNP  NNP      POS NN
The tokenized output will look like this:

Tokens: john johan ##son ' s house
Crucially, this would be the same output as if the raw text were John Johanson's house (with no space before the 's).

If you have a pre-tokenized representation with word-level annotations, you can simply tokenize each input word independently, and deterministically maintain an original-to-tokenized alignment:

### Input
orig_tokens = ["John", "Johanson", "'s",  "house"]
labels      = ["NNP",  "NNP",      "POS", "NN"]

### Output
bert_tokens = []

# Token map will be an int -> int mapping between the `orig_tokens` index and
# the `bert_tokens` index.
orig_to_tok_map = []

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

bert_tokens.append("[CLS]")
for orig_token in orig_tokens:
  orig_to_tok_map.append(len(bert_tokens))
  bert_tokens.extend(tokenizer.tokenize(orig_token))
bert_tokens.append("[SEP]")

# bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
# orig_to_tok_map == [1, 2, 4, 6]
Now orig_to_tok_map can be used to project labels to the tokenized representation.

There are common English tokenization schemes which will cause a slight mismatch between how BERT was pre-trained. For example, if your input tokenization splits off contractions like do n't, this will cause a mismatch. If it is possible to do so, you should pre-process your data to convert these back to raw-looking text, but if it's not possible, this mismatch is likely not a big deal.

Pre-training with BERT
We are releasing code to do "masked LM" and "next sentence prediction" on an arbitrary text corpus. Note that this is not the exact code that was used for the paper (the original code was written in C++, and had some additional complexity), but this code does generate pre-training data as described in the paper.

Here's how to run the data generation. The input is a plain text file, with one sentence per line. (It is important that these be actual sentences for the "next sentence prediction" task). Documents are delimited by empty lines. The output is a set of tf.train.Examples serialized into TFRecord file format.

You can perform sentence segmentation with an off-the-shelf NLP toolkit such as spaCy. The create_pretraining_data.py script will concatenate segments until they reach the maximum sequence length to minimize computational waste from padding (see the script for more details). However, you may want to intentionally add a slight amount of noise to your input data (e.g., randomly truncate 2% of input segments) to make it more robust to non-sentential input during fine-tuning.

This script stores all of the examples for the entire input file in memory, so for large data files you should shard the input file and call the script multiple times. (You can pass in a file glob to run_pretraining.py, e.g., tf_examples.tf_record*.)

The max_predictions_per_seq is the maximum number of masked LM predictions per sequence. You should set this to around max_seq_length * masked_lm_prob (the script doesn't do that automatically because the exact value needs to be passed to both scripts).

python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
Here's how to run the pre-training. Do not include init_checkpoint if you are pre-training from scratch. The model configuration (including vocab size) is specified in bert_config_file. This demo code only pre-trains for a small number of steps (20), but in practice you will probably want to set num_train_steps to 10000 steps or more. The max_seq_length and max_predictions_per_seq parameters passed to run_pretraining.py must be the same as create_pretraining_data.py.

python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
This will produce an output like this:

***** Eval results *****
  global_step = 20
  loss = 0.0979674
  masked_lm_accuracy = 0.985479
  masked_lm_loss = 0.0979328
  next_sentence_accuracy = 1.0
  next_sentence_loss = 3.45724e-05
Note that since our sample_text.txt file is very small, this example training will overfit that data in only a few steps and produce unrealistically high accuracy numbers.

Pre-training tips and caveats
If using your own vocabulary, make sure to change vocab_size in bert_config.json. If you use a larger vocabulary without changing this, you will likely get NaNs when training on GPU or TPU due to unchecked out-of-bounds access.
If your task has a large domain-specific corpus available (e.g., "movie reviews" or "scientific papers"), it will likely be beneficial to run additional steps of pre-training on your corpus, starting from the BERT checkpoint.
The learning rate we used in the paper was 1e-4. However, if you are doing additional steps of pre-training starting from an existing BERT checkpoint, you should use a smaller learning rate (e.g., 2e-5).
Current BERT models are English-only, but we do plan to release a multilingual model which has been pre-trained on a lot of languages in the near future (hopefully by the end of November 2018).
Longer sequences are disproportionately expensive because attention is quadratic to the sequence length. In other words, a batch of 64 sequences of length 512 is much more expensive than a batch of 256 sequences of length 128. The fully-connected/convolutional cost is the same, but the attention cost is far greater for the 512-length sequences. Therefore, one good recipe is to pre-train for, say, 90,000 steps with a sequence length of 128 and then for 10,000 additional steps with a sequence length of 512. The very long sequences are mostly needed to learn positional embeddings, which can be learned fairly quickly. Note that this does require generating the data twice with different values of max_seq_length.
If you are pre-training from scratch, be prepared that pre-training is computationally expensive, especially on GPUs. If you are pre-training from scratch, our recommended recipe is to pre-train a BERT-Base on a single preemptible Cloud TPU v2, which takes about 2 weeks at a cost of about $500 USD (based on the pricing in October 2018). You will have to scale down the batch size when only training on a single Cloud TPU, compared to what was used in the paper. It is recommended to use the largest batch size that fits into TPU memory.
Pre-training data
We will not be able to release the pre-processed datasets used in the paper. For Wikipedia, the recommended pre-processing is to download the latest dump, extract the text with WikiExtractor.py, and then apply any necessary cleanup to convert it into plain text.

Unfortunately the researchers who collected the BookCorpus no longer have it available for public download. The Project Guttenberg Dataset is a somewhat smaller (200M word) collection of older books that are public domain.

Common Crawl is another very large collection of text, but you will likely have to do substantial pre-processing and cleanup to extract a usable corpus for pre-training BERT.

Learning a new WordPiece vocabulary
This repository does not include code for learning a new WordPiece vocabulary. The reason is that the code used in the paper was implemented in C++ with dependencies on Google's internal libraries. For English, it is almost always better to just start with our vocabulary and pre-trained models. For learning vocabularies of other languages, there are a number of open source options available. However, keep in mind that these are not compatible with our tokenization.py library:

Google's SentencePiece library

tensor2tensor's WordPiece generation script

Rico Sennrich's Byte Pair Encoding library

Using BERT in Colab
If you want to use BERT with Colab, you can get started with the notebook "BERT FineTuning with Cloud TPUs". At the time of this writing (October 31st, 2018), Colab users can access a Cloud TPU completely for free. Note: One per user, availability limited, requires a Google Cloud Platform account with storage (although storage may be purchased with free credit for signing up with GCP), and this capability may not longer be available in the future. Click on the BERT Colab that was just linked for more information.
