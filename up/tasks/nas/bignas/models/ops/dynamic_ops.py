import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
import torch.nn as nn
import torch
import copy

try:
    import spring.linklink as link
except:  # noqa
    from up.utils.general.fake_linklink import link

from up.utils.model.bn_helper import GroupSyncBatchNorm, PyTorchSyncBN
from up.tasks.nas.bignas.models.ops.dynamic_utils import sub_filter_start_end, get_same_padding, int2list
from up.tasks.nas.bignas.utils.registry_factory import DYNAMIC_OPS_REGISTRY
from up.utils.general.global_flag import DIST_BACKEND


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    @property
    def module_str(self):
        return 'Identity'


@DYNAMIC_OPS_REGISTRY.register('DynamicConv2d')
class DynamicConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 same_padding=True, padding=1, KERNEL_TRANSFORM_MODE=False):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.same_padding = same_padding
        self.padding = padding
        self.sample_active_weights = False

        # if len(self.kernel_size_list) == 1 normal conv without changing kernel
        self.kernel_size_list = int2list(kernel_size)
        kernel_size = max(self.kernel_size_list)
        if self.same_padding:
            padding = get_same_padding(kernel_size)
        # if len(self.groups_list) == 1 normal conv without changing groups
        self.groups_list = int2list(groups)
        groups = min(self.groups_list)
        super(DynamicConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)

        # if False, the weights are center croped; if True, the transformation matrix is used to transform kernels
        self.KERNEL_TRANSFORM_MODE = KERNEL_TRANSFORM_MODE
        self.depthwise = True if groups == in_channels else False

        if len(self.kernel_size_list) > 1:
            self._ks_set = list(set(self.kernel_size_list))
            self._ks_set.sort()  # e.g., [3, 5, 7]
            if self.KERNEL_TRANSFORM_MODE:
                # register scaling parameters
                # 7to5_matrix, 5to3_matrix
                scale_params = {}
                for i in range(len(self._ks_set) - 1):
                    ks_small = self._ks_set[i]
                    ks_larger = self._ks_set[i + 1]
                    param_name = '%dto%d' % (ks_larger, ks_small)
                    scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** 2))
                for name, param in scale_params.items():
                    self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_out_channel = self.out_channels
        self.active_in_channel = self.in_channels
        self.active_groups = min(self.groups_list)

    def get_active_filter(self, out_channel, in_channel, kernel_size):
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE and kernel_size < max_kernel_size:
            start_filter = self.weight[:out_channel, :in_channel, :, :]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter, self.__getattr__('%dto%d_matrix' % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** 2)
                _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks, target_ks)
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None, out_channel=None, groups=None):
        # Used in Depth Conv, Group Conv and Normal Conv
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        if out_channel is None:
            out_channel = self.active_out_channel
        if groups is None:
            groups = self.active_groups

        in_channel = x.size(1)
        self.active_in_channel = in_channel
        if self.depthwise or groups > in_channel:
            groups = in_channel

        filters = self.get_active_filter(out_channel, in_channel // groups, kernel_size).contiguous()
        if self.same_padding:
            padding = get_same_padding(kernel_size)
        else:
            padding = self.padding

        bias = self.bias[:out_channel].contiguous() if self.bias is not None else None

        if self.sample_active_weights:
            self.weight.data = filters
            self.bias.data = bias

        y = F.conv2d(
            x, filters, bias, self.stride, padding, self.dilation, groups,
        )
        return y

    @property
    def module_str(self):
        return 'DyConv'


@DYNAMIC_OPS_REGISTRY.register('DynamicLinear')
class DynamicLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(DynamicLinear, self).__init__(in_features, out_features, bias)
        self.out_features_list = int2list(out_features)
        self.in_features_list = int2list(in_features)
        self.active_in_channel = max(self.in_features_list)
        self.active_out_channel = max(self.out_features_list)
        self.sample_active_weights = False

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel

        if x.dim() == 3:
            in_features = x.size(2)
        elif x.dim() == 2:
            in_features = x.size(1)
        else:
            raise ValueError('input to linear must be in 2 or 3 dimension')

        self.active_in_channel = in_features
        weight = self.weight[:out_channel, :in_features].contiguous()
        bias = self.bias[:out_channel].contiguous() if self.bias is not None else None

        if self.sample_active_weights:
            self.weight.data = weight
            self.bias = bias

        y = F.linear(x, weight, bias)
        return y

    @property
    def module_str(self):
        return 'DyLinear(I_%d, O_%d)' % (self.active_in_channel, self.active_out_channel)


@DYNAMIC_OPS_REGISTRY.register('DynamicBatchNorm2d')
class DynamicBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(DynamicBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.sample_active_weights = False

    def forward(self, x):
        self._check_input_dim(x)
        feature_dim = x.size(1)
        # Normal BN
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        active_running_mean = self.running_mean[:feature_dim]
        active_running_var = self.running_var[:feature_dim]
        active_weight = self.weight[:feature_dim]
        active_bias = self.bias[:feature_dim]
        if self.sample_active_weights:
            self.running_mean.data = active_running_mean
            self.running_var.data = active_running_var
            self.weight.data = active_weight
            self.bias.data = active_bias

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        return F.batch_norm(
            x, active_running_mean, active_running_var, active_weight,
            active_bias, self.training or not self.track_running_stats,
            exponential_average_factor, self.eps, )

    @property
    def module_str(self):
        return 'DyBN'


@DYNAMIC_OPS_REGISTRY.register('DynamicLayerNorm')
class DynamicLayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(DynamicLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)
        self.sample_active_weights = False

    def forward(self, x):
        feature_dim = x.size(-1)
        return F.layer_norm(x, (feature_dim, ), self.weight[:feature_dim], self.bias[:feature_dim])

    @property
    def module_str(self):
        return 'DyLN'


@DYNAMIC_OPS_REGISTRY.register('SwitchableBatchNorm2d')
class SwitchableBatchNorm2d(DynamicBatchNorm2d):
    def __init__(self, num_features, width_list=None):
        super(SwitchableBatchNorm2d, self).__init__(
            num_features, affine=True)  # TODO: check if need to track running stats
        self.num_features_max = num_features
        assert num_features == max(width_list)
        self.width_list = copy.deepcopy(width_list)
        self.width_list.pop()
        # for tracking performance during training
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(i, affine=False) for i in self.width_list])
        self.sample_active_weights = False

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        feature_dim = input.size(1)

        if feature_dim in self.width_list:
            idx = self.width_list.index(feature_dim)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean,
                self.bn[idx].running_var,
                weight[:feature_dim],
                bias[:feature_dim],
                self.training,
                self.momentum,
                self.eps)
        else:
            y = super().forward(input)
        return y


@DYNAMIC_OPS_REGISTRY.register('DynamicSyncBatchNorm2d')
class DynamicSyncBatchNorm2d(link.nn.SyncBatchNorm2d):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 sync_stats=False,
                 var_mode=link.syncbnVarMode_t.L2,
                 group=None,
                 async_ar_stats=False):
        super(DynamicSyncBatchNorm2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine, sync_stats=sync_stats,
            var_mode=var_mode, group=group, async_ar_stats=async_ar_stats)
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.group = group
        self.sync_stats = sync_stats
        self.var_mode = var_mode
        self.async_ar_stats = async_ar_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()
        self.sample_active_weights = False

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine}, var_mode={var_mode})'
                .format(name=self.__class__.__name__, **self.__dict__))

    def forward(self, input):
        feature_dim = input.size(1)

        active_running_mean = self.running_mean[:feature_dim]
        active_running_var = self.running_var[:feature_dim]
        active_weight = self.weight[:feature_dim]
        active_bias = self.bias[:feature_dim]

        if self.sample_active_weights:
            self.running_mean.data = active_running_mean
            self.running_var.data = active_running_var
            self.weight.data = active_weight
            self.bias.data = active_bias

        return link.nn.SyncBNFunc.apply(
            input,
            active_weight,
            active_bias,
            active_running_mean,
            active_running_var,
            self.eps,
            self.momentum,
            self.group,
            self.sync_stats,
            self.var_mode,
            self.training,
            self.async_ar_stats)


@DYNAMIC_OPS_REGISTRY.register('SwitchableSyncBatchNorm2d')
class SwitchableSyncBatchNorm2d(DynamicSyncBatchNorm2d):
    def __init__(self, num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 sync_stats=False,
                 var_mode=link.syncbnVarMode_t.L2,
                 group=None,
                 async_ar_stats=False,
                 width_list=None):
        super(SwitchableSyncBatchNorm2d, self).__init__(
            num_features, group=group, momentum=momentum, affine=affine,
            eps=eps, async_ar_stats=async_ar_stats,
            sync_stats=sync_stats, var_mode=var_mode)

        assert num_features == max(width_list), "num_features is the maximum width."
        self.num_features_max = num_features
        self.width_list = copy.deepcopy(width_list)
        self.width_list.pop()
        self.bn = nn.ModuleList([link.nn.SyncBatchNorm2d(i,
                                                         affine=False,
                                                         group=group,
                                                         momentum=momentum,
                                                         sync_stats=sync_stats,
                                                         var_mode=var_mode) for i in self.width_list])

    def forward(self, input):
        feature_dim = input.size(1)

        if feature_dim in self.width_list:
            idx = self.width_list.index(feature_dim)
            return link.nn.SyncBNFunc.apply(
                input,
                self.weight[:feature_dim],
                self.bias[:feature_dim],
                self.bn[idx].running_mean,
                self.bn[idx].running_var,
                self.eps,
                self.momentum,
                self.group,
                self.sync_stats,
                self.var_mode,
                self.training,
                self.async_ar_stats)
        else:
            return super().forward(input)


@DYNAMIC_OPS_REGISTRY.register('DynamicGroupSyncBatchNorm2d')
class DynamicGroupSyncBatchNorm2d(GroupSyncBatchNorm):
    """
    Dynamic sync-BN with elastic output channels, note that running_stats are not tracked in this type of BN
    """
    def __init__(self, num_features, group_size=None, momentum=0.1):
        super(DynamicGroupSyncBatchNorm2d, self).__init__(num_features,
                                                          group_size=group_size,
                                                          momentum=momentum)
        self.max_num_features = num_features
        self.sample_active_weights = False

    def forward(self, x):
        feature_dim = x.size(1)

        active_running_mean = self.running_mean[:feature_dim]
        active_running_var = self.running_var[:feature_dim]
        active_weight = self.weight[:feature_dim]
        active_bias = self.bias[:feature_dim]

        if self.sample_active_weights:
            self.running_mean.data = active_running_mean
            self.running_var.data = active_running_var
            self.weight.data = active_weight
            self.bias.data = active_bias

        return link.nn.SyncBNFunc.apply(x,
                                        active_weight,
                                        active_bias,
                                        active_running_mean,
                                        active_running_var,
                                        self.eps,
                                        self.momentum,
                                        self.group,
                                        self.sync_stats,
                                        self.var_mode,
                                        self.training,
                                        self.async_ar_stats)


@DYNAMIC_OPS_REGISTRY.register('DynamicSyncBN_DDP')
class DynamicSyncBN_DDP(PyTorchSyncBN):
    def __init__(self, num_features, group_size=None, **kwargs):
        super(DynamicSyncBN_DDP, self).__init__(
            num_features,
            group_size=None,
            **kwargs
        )

    def deploy_forward(self, input):
        active_num_features = input.size(1)
        self.running_mean.data = self.running_mean[:active_num_features]
        self.running_var.data = self.running_var[:active_num_features]
        self.weight.data = self.weight[:active_num_features]
        self.bias.data = self.bias[:active_num_features]

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            False,
            exponential_average_factor,
            self.eps,
        )

    def forward(self, input):
        if getattr(self, '_deploying', False):
            return self.deploy_forward(input)

        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError('DynamicSyncBatchNorm expected \
                 input tensor to be on GPU')
        self._check_input_dim(input)
        active_num_features = input.size(1)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        # currently tracking running stats with dynamic input sizes
        # is meaningless.
        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum

        r"""
        This decides whether the mini-batch stats should be used for
        normalization rather than the buffers. Mini-batch stats
        are used in training mode, and in eval mode when buffers
        are None.
        """
        need_sync = self.training or not self.track_running_stats
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to vanilla BN when synchronization is not necessary
        if not need_sync:
            if self.running_mean is not None and self.running_var is not None:
                return F.batch_norm(
                    input,
                    self.running_mean[:active_num_features],
                    self.running_var[:active_num_features],
                    self.weight[:active_num_features],
                    self.bias[:active_num_features],
                    self.training or not self.track_running_stats,
                    exponential_average_factor,
                    self.eps,
                )
            else:
                return F.batch_norm(
                    input,
                    None,
                    None,
                    self.weight[:active_num_features],
                    self.bias[:active_num_features],
                    self.training or not self.track_running_stats,
                    exponential_average_factor,
                    self.eps,
                )
        else:
            if not self.ddp_gpu_size:
                raise AttributeError('DynamicSyncBatchNorm is only supported \
                     torch.nn.parallel.DistributedDataParallel')

            if self.track_running_stats:
                return sync_batch_norm.apply(
                    input,
                    self.weight[:active_num_features],
                    self.bias[:active_num_features],
                    self.running_mean[:active_num_features],
                    self.running_var[:active_num_features],
                    self.eps,
                    exponential_average_factor,
                    process_group,
                    world_size)
            else:
                return sync_batch_norm.apply(
                    input,
                    self.weight[:active_num_features],
                    self.bias[:active_num_features],
                    None,
                    None,
                    self.eps,
                    exponential_average_factor,
                    process_group,
                    world_size)

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = DynamicSyncBN_DDP(
                module.num_features,
                module.eps, module.momentum,
                module.affine,
                module.track_running_stats,
                process_group)

            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias

            if module.track_running_stats:
                module_output.running_mean = module.running_mean
                module_output.running_var = module.running_var
                module_output.num_batches_tracked = module.num_batches_tracked

            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))

        del module
        return module_output


_norm_cfg = {
    'dynamic_solo_bn': DynamicBatchNorm2d,
    'dynamic_switch_solo_bn': SwitchableBatchNorm2d,
    'dynamic_layer_norm': DynamicLayerNorm,
    'dynamic_switch_sync_bn': SwitchableSyncBatchNorm2d,
    'dynamic_sync_bn': DynamicGroupSyncBatchNorm2d}


def build_dynamic_norm_layer(config):
    """
    Build normalization layer according to configurations.

    solo_bn: Dynamic Version of torch.nn.BatchNorm2d
    sync_bn: Dynamic Version of nn.SyncBatchNorm2d
    switch_solo_bn: Switchable torch.nn.BatchNorm2d
    switch_sync_bn: Switchable nn.SyncBatchNorm2d
    """

    assert isinstance(config, dict) and 'type' in config
    config = copy.deepcopy(config)
    norm_type = config.pop('type')
    config_kwargs = config.get('kwargs', {})

    if DIST_BACKEND.backend == 'dist':
        _norm_cfg['dynamic_sync_bn'] = DynamicSyncBN_DDP

    def NormLayer(*args, **kwargs):
        return _norm_cfg[norm_type](*args, **kwargs, **config_kwargs)

    return NormLayer
