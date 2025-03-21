# -- coding=utf-8 -- #
import torch
import torch.nn as nn
from numpy import prod

def forward_channel_last(module, x, keep_shape=True):
    """
    Use this when you need to temporarily put channels to the last dimension to fit the module
    (e.g. applying LayerNorm on [B, C, *x_shape] input)
    """
    if isinstance(module, nn.Identity):
        return x
    
    B, C = x.shape[:2]
    x_size = x.shape[2:]
    ndim = len(x.shape)
    if keep_shape:
        if ndim == 3:
            x = x.transpose(1, 2)
            x = module(x.contiguous())
            x = x.transpose(1, 2).contiguous()
        elif ndim == 4:
            x = x.permute(0, 2, 3, 1)
            x = module(x.contiguous())
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = module(x.contiguous())
            x = x.permute(0, 4, 1, 2, 3).contiguous()
    else:
        x = x.flatten(2).transpose(1, 2)  # [B, C, x_size] -> [B, -1, C]
        x = module(x)
        x = x.transpose(1, 2).reshape(B, C, *x_size).contiguous()  # [B, -1, C] -> [B, C, x_size]
    return x

def forward_channel_second(module, x, x_size):
    """
    Use this when you need to temporarily put channels to the second dimension to fit the module
    (e.g. applying Convolution on [B, N, C] input)
    """
    if isinstance(module, nn.Identity):
        return x
    
    assert x.ndim == 3
    B, N, C = x.shape
    assert N == prod(x_size), f'The length of sequence ({N}) doesn\'t match the size of x ({prod(x_size)}).'

    x = x.transpose(1, 2).reshape(B, C, *x_size)
    x = module(x)
    x = x.flatten(2).transpose(1, 2)
    return x


class LayerNorm(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)  # [B, 1, ...]
        s = (x - u).pow(2).mean(1, keepdim=True)  # [B, 1, ...]
        x = (x - u) / torch.sqrt(s + self.eps)  # [B, C, ...]
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        #     x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x



def get_conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                   bias=True, is_transposed=False, output_padding=0, padding_mode='zeros', adn_order='CNA', 
                   act_type='RELU', act_kwargs=None, norm_type='BATCH', norm_kwargs=None, 
                   drop_type='DROPOUT', drop_rate=0., dropout_dims=1, spatial_dims=2, **kwargs):
    
    assert isinstance(adn_order, str)
    adn_order = adn_order.upper()
    adn_order = 'C' + adn_order if 'C' not in adn_order else adn_order  # default first layer as convolution
    norm_idx = adn_order.find('N')
    conv_idx = adn_order.find('C')
    norm_channels = out_channels if conv_idx < norm_idx else in_channels
    
    layer_dict = {
        'C': get_conv_layer(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
            padding_mode=padding_mode, dilation=dilation, groups=groups, bias=bias, 
            is_transposed=is_transposed, output_padding=output_padding, spatial_dims=spatial_dims),
        'N': get_norm_layer(norm_type, [norm_channels], kwargs=norm_kwargs, spatial_dims=spatial_dims) ,
        'A': get_act_layer(act_type, kwargs=act_kwargs),
        'D': get_drop_layer(drop_type=drop_type, drop_rate=drop_rate, spatial_dims=dropout_dims)}

    layers = []
    for name in adn_order:
        layers.append(layer_dict[name])
    return nn.Sequential(*layers)


def get_norm_act_drop_block(dims, adn_order='NAD', act_type='RELU', act_kwargs=None, norm_type='BATCH',
                            drop_type='DROPOUT', drop_rate=0., dropout_dims=1, spatial_dims=2, **kwargs):

    layer_dict = {
        'N': get_norm_layer(norm_type, [dims], spatial_dims=spatial_dims) if norm_type is not None else nn.Identity(),
        'A': get_act_layer(act_type, kwargs=act_kwargs) if act_type is not None else nn.Identity(),
        'D': get_drop_layer(drop_type=drop_type, drop_rate=drop_rate, spatial_dims=dropout_dims)}

    layers = []
    for name in adn_order.upper():
        layers.append(layer_dict[name])
    return nn.Sequential(*layers)


def get_linear_block(in_dims, out_dims, bias=True, adn_order='NAD', act_type='RELU', act_kwargs=None,
                     norm_type='BATCH', drop_type='DROPOUT', drop_rate=0.1, dropout_dims=1, spatial_dims=1, **kwargs):
    linear_layer = nn.Linear(in_dims, out_dims, bias=bias)
    layer_dict = {
        'N': get_norm_layer(norm_type, [out_dims], spatial_dims=spatial_dims) if norm_type is not None else nn.Identity(),
        'A': get_act_layer(act_type, kwargs=act_kwargs) if act_type is not None else nn.Identity(),
        'D': get_drop_layer(drop_type=drop_type, drop_rate=drop_rate, spatial_dims=dropout_dims)}

    layers = [linear_layer]
    for name in adn_order.upper():
        layers.append(layer_dict[name])
    return nn.Sequential(*layers)


def get_conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1,
                   groups=1, bias=True, padding_mode='zeros', is_transposed=False, output_padding=0, 
                   conv_type=None, spatial_dims=2, **kwargs):
    conv_kwargs = {
            'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size,
            'stride': stride, 'padding': padding, 'padding_mode': padding_mode, 'dilation': dilation, 
            'groups': groups, 'bias': bias}
    if conv_type is not None:
        assert conv_type in (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                             nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d), \
            f'Since conv_type ({conv_type}) is not None, it should be an nn.Module like nn.ConvXd or ' \
            f'nn.ConvTransposedXd (X=1,2,3). If you would like to call convolution operation in another way, ' \
            f'then simply set conv_type=None and specify other arguments.'
        if conv_type in (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d):
            conv_kwargs['output_padding'] = output_padding
        layer = conv_type(**conv_kwargs)
    else:
        if is_transposed:
            # Ho = s * Hi - s - 2 * p + d * (k - 1) + op + 1
            # Ho = 2 * Hi - 2 - 2 * p + 2 + op + 1 
            #    = 2 * Hi - 2 * p + op + 1
            #    = 2 * Hi - 2 + 1 + 1  # s==2, k==3, p==1, op==1
            
            # Ho = 2 * Hi - 2 - 2 * p + 1 + op + 1 
            #    = 2 * Hi - 2 * p + op  # s==2, k==2, p==0, op==0
            conv_classes = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
            layer = conv_classes[spatial_dims - 1]
            conv_kwargs['output_padding'] = output_padding
        else:
            # Ho = (Hi - k + 2 * p) / s + 1
            # strided conv:
            # Ho = (Hi - 2 + 0) / 2 + 1  # s==2, k==2, p==0
            conv_classes = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
            layer = conv_classes[spatial_dims - 1]
        layer = layer(**conv_kwargs)
    return layer


def get_pool_layer(pool_type, spatial_dims=2, args=None, kwargs=None):
    args = [] if args is None else args
    args = [args] if not isinstance(args, (tuple, list)) else args
    kwargs = {} if kwargs is None else kwargs
    if not isinstance(pool_type, str):
        assert hasattr(pool_type, '__base__'), f'Since pool_type is not str, it is expected to be a class, got {pool_type}.'
        layer = pool_type(*args, **kwargs)
    else:
        pool_type = pool_type.upper().replace('_', '').replace('POOL', '')
        if pool_type == 'MAX':
            pool_classes = (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)
        elif pool_type == 'AVG':
            pool_classes = (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)
        elif pool_type == 'ADAPTIVEMAX':
            pool_classes = (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)
        elif pool_type == 'ADAPTIVEAVG':
            pool_classes = (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)
        else:
            raise NotImplementedError('Unsupported pooling method.')
        layer = pool_classes[spatial_dims - 1](*args, **kwargs)
    return layer


def get_norm_layer(norm_type, args=None, kwargs=None, spatial_dims=2, return_instance=True):
    args = [] if args is None else args
    args = [args] if not isinstance(args, (tuple, list)) else args
    kwargs = {} if kwargs is None else kwargs
    
    if norm_type is None:
        #print('Normalization type: None, returning identity map.')
        return nn.Identity() if return_instance else nn.Identity
    
    if not isinstance(norm_type, str):
        assert hasattr(norm_type, '__base__'), f'Since norm_type is not str, it is expected to be a class, got {norm_type}.'
        layer = norm_type(*args, **kwargs)
    else:
        norm_type = norm_type.upper().replace('_', '').replace('NORM', '')
        
        if norm_type == 'INSTANCE':
            norm_classes = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
            # `affine` default to False in pytorch, for unknown reasons...
            # Ref: https://github.com/pytorch/pytorch/issues/22755
            if 'affine' not in kwargs.keys():
                kwargs.update({'affine':True})
        elif norm_type == 'BATCH':
            norm_classes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        elif norm_type == 'LAYER':
            norm_classes = (nn.LayerNorm, nn.LayerNorm, nn.LayerNorm)
        elif norm_type == 'LAYERCV':
            norm_classes = (LayerNorm, LayerNorm, LayerNorm)
        elif norm_type == 'GROUP':
            # `num_channels` should be in `args`, but GroupNorm takes `num_groups` first.
            # We need to deal with this.
            assert args is not None, f'{norm_type} chosen with no args.'
            keys = kwargs.keys()
            if len(args) == 1 and 'num_groups' in keys and 'num_channels' not in keys:
                # Input: `args` == [num_channels], `kwargs` == dict(num_groups=num_groups)
                args = (kwargs['num_groups'], args[0])
                kwargs = {}
            elif len(args) == 1 and 'num_channels' in keys and 'num_groups' not in keys:
                # Input: `args` == [num_groups], `kwargs` == dict(num_channels=num_channels)
                args = (args[0], kwargs['num_channels'])
            elif len(args) == 2:
                # Input: `args` == [num_groups, num_channels]
                assert 'num_channels' not in keys and 'num_groups' not in keys, f'Got 2 args ({args}), contradict with kwargs ({kwargs})'
            norm_classes = (nn.GroupNorm, nn.GroupNorm, nn.GroupNorm)
        elif norm_type == 'LOCALRESPONSE':
            norm_classes = (nn.LocalResponseNorm, nn.LocalResponseNorm, nn.LocalResponseNorm)
        elif norm_type == 'SYNCBATCH':
            norm_classes = (nn.SyncBatchNorm, nn.SyncBatchNorm, nn.SyncBatchNorm)
        else:
            raise NotImplementedError('Unsupported normalization method.')
        layer = norm_classes[spatial_dims - 1](*args, **kwargs) if return_instance else norm_classes[spatial_dims - 1]
    return layer


def get_act_layer(act_type, args=None, kwargs=None):
    args = [] if args is None else args
    args = [args] if not isinstance(args, (tuple, list)) else args
    kwargs = {} if kwargs is None else kwargs
    
    if act_type is None:
        #print('Activation type: None, returning identity map.')
        return nn.Identity()
        
    if not isinstance(act_type, str):
        assert hasattr(act_type, '__base__'), f'Since act_type is not str, it is expected to be a class, got {act_type}.'
        layer = act_type(**kwargs)
    else:
        #assert isinstance(act_type, str), f'Expect act_type to be a str, got {act_type}.'
        act_type = act_type.upper().replace('_', '')
        if act_type == 'RELU':
            layer = nn.ReLU(**kwargs)
        elif act_type == 'RELU6':
            layer = nn.ReLU6(**kwargs)
        elif act_type == 'PRELU':
            layer = nn.PReLU(*args, **kwargs)
        elif act_type == 'GELU':
            layer = nn.GELU()
        elif act_type in ['LEAKYRELU', 'LEAKY', 'LRELU']:
            layer = nn.LeakyReLU(*args, **kwargs)
        elif act_type == 'ELU':
            layer = nn.ELU(*args, **kwargs)
        elif act_type == 'SELU':
            layer = nn.SELU(**kwargs)
        elif act_type == 'CELU':
            layer = nn.CELU(*args, **kwargs)
        elif act_type == 'SOFTMAX':
            layer = nn.Softmax(*args, **kwargs)
        elif act_type == 'LOGSOFTMAX':
            layer = nn.LogSoftmax(*args, **kwargs)
        elif act_type == 'SIGMOID':
            layer = nn.Sigmoid()
        elif act_type == 'TANH':
            layer = nn.Tanh()
        else:
            raise NotImplementedError('Unsupported activation function.')
    return layer


def get_drop_layer(drop_type='DROPOUT', drop_rate=0., spatial_dims=1):
    assert isinstance(drop_type, str), f'Expect drop_type to be a str, got {drop_type}.'
    drop_type = drop_type.upper().replace('_', '')
    if drop_type == 'DROPOUT':
        dropout_classes = (nn.Dropout, nn.Dropout2d, nn.Dropout3d)
        layer = dropout_classes[spatial_dims - 1](drop_rate)
    elif drop_type == 'PATH':
        layer = DropPath(drop_rate)
    elif drop_type == 'ALPHA':
        layer = nn.AlphaDropout(drop_rate)
    elif drop_type == 'FEATURE':
        layer = nn.FeatureAlphaDropout(drop_rate)
    else:
        raise NotImplementedError(f"Unsupported dropout operation, try {['DROPOUT', 'PATH', 'ALPHA', 'FEATURE']}.")
    return layer


class DropPath(nn.Module):
    # modified from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    @staticmethod
    def drop_path(x, drop_prob=0., training=False, scale_by_keep=True):
        if drop_prob == 0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor=4):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        N, C, H, W = x.size()
        S = self.downscale_factor
        x = x.view(N, C, H // S, S, W // S, S)  # (B, C, H//S, S, W//S, S)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (B, S, S, C, H//S, W//S)
        x = x.view(N, C * S * S, H // S, W // S)  # (N, C*S^2, H//S, W//S)
        return x

    def extra_repr(self):
        return f"downscale_factor={self.downscale_factor}"

# def pixel_unshuffle(input, downscale_factor):
#     '''
#     input: batchSize * c * k*w * k*h
#     kdownscale_factor: k

#     batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
#     '''
#     c = input.shape[1]

#     kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
#                                1, downscale_factor, downscale_factor],
#                          device=input.device)
#     for y in range(downscale_factor):
#         for x in range(downscale_factor):
#             kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
#     return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

# class PixelUnshuffle(nn.Module):
#     def __init__(self, downscale_factor):
#         super(PixelUnshuffle, self).__init__()
#         self.downscale_factor = downscale_factor
#     def forward(self, input):
#         '''
#         input: batchSize * c * k*w * k*h
#         kdownscale_factor: k

#         batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
#         '''

#         return pixel_unshuffle(input, self.downscale_factor)


if __name__ == '__main__':
    norm_args = [64, ]
    norm_kwargs = {'num_groups':4}
    layer = get_norm_layer('GROUP', norm_args, norm_kwargs)
    print(layer)