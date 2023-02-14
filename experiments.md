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
