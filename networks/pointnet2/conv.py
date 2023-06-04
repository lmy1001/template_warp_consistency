
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction
from itertools import repeat
import numpy as np
import shutil
import tqdm
from scipy.stats import t as student_t
import statistics as stats

class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args,
        bn=False,
        activation=nn.ReLU(inplace=True),
        preact=False,
        first=False,
        name="",
    ):
        # type: (SharedMLP, List[int], bool, Any, bool, bool, AnyStr) -> None
        super(SharedMLP, self).__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0))
                    else None,
                    preact=preact,
                ),
            )


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super(_BNBase, self).__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):
    def __init__(self, in_size, name=""):
        # type: (BatchNorm1d, int, AnyStr) -> None
        super(BatchNorm1d, self).__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size, name=""):
        # type: (BatchNorm2d, int, AnyStr) -> None
        super(BatchNorm2d, self).__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):
    def __init__(self, in_size, name=""):
        # type: (BatchNorm3d, int, AnyStr) -> None
        super(BatchNorm3d, self).__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class _ConvBase(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        dilation,
        activation,
        bn,
        init,
        conv=None,
        norm_layer=None,
        bias=True,
        preact=False,
        name="",
    ):
        super(_ConvBase, self).__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = norm_layer(out_size)
            else:
                bn_unit = norm_layer(in_size)

        if preact:
            if bn:
                self.add_module(name + "normlayer", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "conv", conv_unit)

        if not preact:
            if bn:
                self.add_module(name + "normlayer", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)


class Conv1d(_ConvBase):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm1d,
    ):
        # type: (Conv1d, int, int, int, int, int, int, Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(Conv1d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            name=name,
        )


class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm2d,
    ):
        # type: (Conv2d, int, int, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(Conv2d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            name=name,
        )


class Conv3d(_ConvBase):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm3d,
    ):
        # type: (Conv3d, int, int, Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(Conv3d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.Conv3d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            name=name,
        )


class FC(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=None,
        preact=False,
        name="",
    ):
        # type: (FC, int, int, Any, bool, Any, bool, AnyStr) -> None
        super(FC, self).__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant_(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "fc", fc)

        if not preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + "activation", activation)


class _DropoutNoScaling(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @staticmethod
    def symbolic(g, input, p=0.5, train=False, inplace=False):
        if inplace:
            return None
        n = g.appendNode(
            g.create("Dropout", [input]).f_("ratio", p).i_("is_test", not train)
        )
        real = g.appendNode(g.createSelect(n, 0))
        g.appendNode(g.createSelect(n, 1))
        return real

    @classmethod
    def forward(cls, ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = cls._make_noise(input)
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise.expand_as(input)
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(ctx.noise), None, None, None
        else:
            return grad_output, None, None, None


dropout_no_scaling = _DropoutNoScaling.apply


class _FeatureDropoutNoScaling(_DropoutNoScaling):
    @staticmethod
    def symbolic(input, p=0.5, train=False, inplace=False):
        return None

    @staticmethod
    def _make_noise(input):
        return input.new().resize_(
            input.size(0), input.size(1), *repeat(1, input.dim() - 2)
        )


feature_dropout_no_scaling = _FeatureDropoutNoScaling.apply