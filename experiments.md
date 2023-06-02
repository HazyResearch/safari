Configs for various experiments.

Adding `wandb=null` to any command line turns off logging.

Some of these datasets may require downloading and preparing data, documented in the [src/dataloaders](./src/dataloaders/) subdirectory.

## Long Range Arena (LRA)

You can use these configs to reproduce the best results on LRA using a random initialization.

```
python -m train experiment=lra/long-conv-lra-listops
python -m train experiment=lra/long-conv-lra-imdb
python -m train experiment=lra/long-conv-lra-cifar
python -m train experiment=lra/long-conv-lra-aan
python -m train experiment=lra/long-conv-lra-pathfinder
python -m train experiment=lra/long-conv-lra-pathx
```

## CIFAR-10

```
python -m train experiment=cifar/long-conv-cifar
```

The above command line reproduces our best sequential CIFAR model.

## Language Modeling Synthetics

You can use these commands to run synthetics from the H3 paper:
```
python -m train experiment=synthetics/associative_recall/transformer
python -m train experiment=synthetics/associative_recall/s4d
python -m train experiment=synthetics/associative_recall/h3

python -m train experiment=synthetics/induction_head/transformer
python -m train experiment=synthetics/induction_head/s4d
python -m train experiment=synthetics/induction_head/h3
```

You should get scores >97 for Transformer and H3 for both tasks, and worse scores for S4D.

Note that the train accuracy is being computed over the entire sequence.
If you swap it out with another layer and train accuracy goes to 100, that probably means that your layer is non-causal. 
This is a common failure mode of models tested on these synthetics, consider testing causality numerically (e.g. checking whether gradients
only flow "backwards in time" i.e. 
$$\forall j > i: \frac{\partial y_i}{\partial x_j} \approx 0$$

To run the synthetics from the Hyena paper, you can use these commands:
```
python -m train experiment=synthetics/associative_recall/hyena-131k-30vs.yaml 
```
You can also customize sequence length and vocabulary sizes directly through the command line (or by creating a custom config).
We recommend a quick run with vocabulary size 10 and sequence length 256 to verify the pipeline is working correctly
```
python -m train experiment=synthetics/associative_recall/ dataset.input_seq_len=$SEQ_LEN dataset.vocab_size=$VOCAB_SIZE dataset.data_idr=$DATA_DIR
```
Note that dataset generation for >100k sequence lengths can take a while. If you pass a DATA_DIR to the script, the dataset will be saved after generation, and loaded for any other run with the same sequence length and vocabulary size.

Hyena (2 layers) should reach >90 accuracy on the 30 vocabulary size, 131k sequence length associative recall task ([here:](https://api.wandb.ai/links/zymrael/pnw1nckm) for an example wandb log). Other models should get worse scores (Transformers will be a pain to train on 131k, good luck!).


## PILE

To train on the PILE, you will need to initialize the FlashAttention submodule in this repository and install it, along with fast fused [kernels](https://github.com/HazyResearch/flash-attention/tree/main/training):
```
cd safari
git submodule update --init
cd flash-attention
git submodule update --init
pip install -e .

cd ./csrc/fused_dense_lib && pip install .
cd ../xentropy && pip install .
cd ../layer_norm && pip install .
```
You should also install the FFT convolution library:
```
cd safari
cd csrc/fftconv && pip install .
```

Next, prepare the data by following the instructions in the FlashAttention training [scripts](https://github.com/HazyResearch/flash-attention/blob/main/training/README.md).

Then you can run these commands to train on PILE.
If you downloaded data to `DATA_DIR` in the previous step, you will need to set `DATA_PATH=$DATA_DIR` to get the data in this repo.
```
python -m train experiment=pile/h3
python -m train experiment=pile/h3-conv
```
The H3-conv experiment will run the model in the long convolutions paper.

You can also run this to train for fewer tokens. Make sure to have the learning rate decay properly at the end of training (set `train.scheduler.t_initial` equal to `trainer.max_steps`:
```
python -m train experiment=pile/h3-50b-tokens trainer.max_steps=10000 train.scheduler.t_initial=10000 # 5B tokens
python -m train experiment=pile/h3-conv-50b-tokens trainer.max_steps=10000 train.scheduler.t_initial=10000 # 5B tokens
```
These commands train for 5B tokens.

To train a small Hyena model for 150 billion tokens:
```
python -m train experiment=pile/hyena-150b-tokens
```
We provide a [wandb log](https://api.wandb.ai/links/hazy-research/uzoya5mf) as a reference for typical training behavior.

To recreate the experiments in the Hyena paper, you should adjust the max steps and scheduler to decay at 5B, 10B, or 15B tokens:
```
python -m train experiment=pile/hyena-150b-tokens trainer.max_steps=10000 train.scheduler.t_initial=10000 # 5B tokens
python -m train experiment=pile/hyena-150b-tokens trainer.max_steps=20000 train.scheduler.t_initial=20000 # 10B tokens
python -m train experiment=pile/hyena-150b-tokens trainer.max_steps=30000 train.scheduler.t_initial=30000 # 15B tokens
```

## Downstream Evaluations

Hyena small checkpoint is available at `https://huggingface.co/Zymrael/hyena-small-150b-tok`.
Download directly and move to CKPT_PATH.

To evaluate your language model on LAMBADA (OpenAI preprocessing with full last word accuracy, see: https://github.com/EleutherAI/lm-evaluation-harness/issues/356 and https://github.com/openai/gpt-2/issues/131#issuecomment-497136199 for differences between LAMBADA versions).

```
CUDA_VISIBLE_DEVICES=$ID python evals/lambada.py --ckpt_path $CKPT_PATH --data_dir $DATA_DIR
```
Dataset should first be downloaded from `https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl` and saved to `$DATA_DIR/lambada/lambada_openai/lambada_test.jsonl`.

To evaluate additional models, write a custom `eval` config in `src/configs/evals/`. 

For Hyena small (153M, trained for 150B tokens), you should see `32.95` accuracy without stop word filter and `44.30` with stop word filter.  

## ImageNet

To run Hyena on ImageNet using ViT (swapping attention for Hyena layers), use this command:

```
python -m train wandb=null experiment=imagenet/hyena-vit
```
