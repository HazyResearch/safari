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

## PILE

To train on the PILE, you will need to initialize the FlashAttention submodule in this repository and install it, along with fast fused [kernels](https://github.com/HazyResearch/flash-attention/tree/main/training):
```
cd safari
git submodule update --init
cd flash-attention
git submodule update --init
pip install -e .

cd ../csrc/fused_dense_lib && pip install .
cd ../csrc/xentropy && pip install .
cd ../csrc/layer_norm && pip install .
```
You should also install the FFT convolution library:
```
cd safari
cd csrc/fft_conv && pip install .
```

Next, prepare the data by following the instructions in the FlashAttention training [scripts](https://github.com/HazyResearch/flash-attention/blob/main/training/README.md).

Then you can run these commands to train on PILE.
If you downloaded data to `DATA_DIR` in the previous step, you will need to set `DATA_PATH=$DATA_DIR` to get the data in this repo.
```
python -m train experiment/pile/h3
python -m train experiment/pile/h3-conv
```
The H3-conv experiment will run the model in the long convolutions paper.

You can also run this to train for fewer tokens:
```
python -m train experiment/pile/h3-50b-tokens trainer.max_steps=10000 train.scheduler.t_initial=10000
python -m train experiment/pile/h3-conv-50b-tokens trainer.max_steps=10000 train.scheduler.t_initial=10000
```
These commands train for 5B tokens.