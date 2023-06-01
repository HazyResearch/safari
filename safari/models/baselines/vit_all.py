"""
The original Vision Transformer (ViT) from timm, copyright belongs to / Copyright 2020 Ross Wightman
"""
import math
import logging

from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg
from timm.models.layers import PatchEmbed, Mlp, trunc_normal_, lecun_normal_

from src.models.sequence.base import SequenceModule
from src.models.nn.components import Normalization
from src.models.sequence.block import SequenceResidualBlock
from src.utils.config import to_list, to_dict

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'classifier': 'head',
        **kwargs,
    }


default_cfgs = {
    # patch models (my experiments)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class VisionTransformer(SequenceModule):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        d_model=768,
        depth=12,
        expand=4,
        representation_size=None,
        distilled=False,
        dropout=0.,
        drop_path_rate=0.,
        embed_layer=PatchEmbed,
        norm='layer',
        weight_init='',
        layer=None,
        transposed=False,
        layer_reps=1,
        use_pos_embed=False,
        use_cls_token=False,
        track_norms=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            d_model (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            dropout (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.d_model = d_model  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.use_pos_embed = use_pos_embed
        self.use_cls_token = use_cls_token

        self.track_norms = track_norms

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=d_model,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = None
        self.dist_token = None
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model)) if distilled else None
        else:
            assert not distilled, 'Distillation token not supported without class token'

        self.pos_embed = None
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, d_model))
            self.pos_drop = nn.Dropout(p=dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.transposed = transposed

        layer = to_list(layer, recursive=False) * layer_reps

        # Some special arguments are passed into each layer
        for _layer in layer:
            # If layers don't specify dropout, add it
            if _layer.get('dropout', None) is None:
                _layer['dropout'] = dropout
            # Ensure all layers are shaped the same way
            _layer['transposed'] = transposed

        # Config for the inverted bottleneck
        ff_cfg = {
            '_name_': 'ff',
            'expand': int(expand),
            'transposed': self.transposed,
            'activation': 'gelu',
            'initializer': None,
            'dropout': dropout,
        }

        blocks = []
        for i in range(depth):
            for _layer in layer:
                blocks.append(
                    SequenceResidualBlock(
                        d_input=d_model,
                        i_layer=i,
                        prenorm=True,
                        dropout=dropout,
                        layer=_layer,
                        residual='R',
                        norm=norm,
                        pool=None,
                        drop_path=dpr[i],
                    )
                )
            if expand > 0:
                blocks.append(
                    SequenceResidualBlock(
                        d_input=d_model,
                        i_layer=i,
                        prenorm=True,
                        dropout=dropout,
                        layer=ff_cfg,
                        residual='R',
                        norm=norm,
                        pool=None,
                        drop_path=dpr[i],
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = Normalization(d_model, transposed=self.transposed, _name_=norm)
        else:
            self.norm = Normalization(d_model, transposed=self.transposed, **norm)

        # Representation layer: generally defaults to nn.Identity()
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(d_model, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s): TODO: move to decoder
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.d_model, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            if self.cls_token is not None:
                trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward_features(self, x):
        # TODO: move to encoder
        x = self.patch_embed(x)

        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        if self.use_pos_embed:
            x = self.pos_drop(x + self.pos_embed)

        if self.track_norms: output_norms = [torch.mean(x.detach() ** 2)]

        for block in self.blocks:
            x, _ = block(x)
            if self.track_norms: output_norms.append(torch.mean(x.detach() ** 2))
        x = self.norm(x)

        if self.track_norms:
            metrics = to_dict(output_norms, recursive=False)
            self.metrics = {f'norm/{i}': v for i, v in metrics.items()}

        if self.dist_token is None:
            if self.use_cls_token:
                return self.pre_logits(x[:, 0])
            else:
                # pooling: TODO move to decoder
                return self.pre_logits(x.mean(1))
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, rate=1.0, resolution=None, state=None):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x, None


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(m, (nn.Linear)):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                dense_init_fn_ = partial(trunc_normal_, std=.02)
                if isinstance(m, nn.Linear):
                    dense_init_fn_(m.weight)

    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed, getattr(model, 'num_tokens', 1),
                                                model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model


def vit_small_patch16_224(pretrained=False, **kwargs):
    """ Tri's custom 'small' ViT model. d_model=768, depth=8, num_heads=8, mlp_ratio=3.
    NOTE:
        * this differs from the DeiT based 'small' definitions with d_model=384, depth=12, num_heads=6
        * this model does not have a bias for QKV (unlike the official ViT and DeiT models)
    """
    print(kwargs)
    model_kwargs = dict(
        patch_size=16,
        d_model=768,
        depth=8,
        expand=3,
        norm='layer',
    )
    model_kwargs = {
        **model_kwargs,
        **kwargs,
    }
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        model_kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16,
        d_model=768,
        depth=12,
        # num_heads=12,
    )
    model_kwargs = {
        **model_kwargs,
        **kwargs,
    }
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
