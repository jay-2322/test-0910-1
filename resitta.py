import random
from copy import deepcopy
import PIL
import torch
from torch.cuda.amp import GradScaler, autocast

import math
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import my_transforms
import torch.nn.functional as F
from typing import Tuple, TypeVar
from torch import Tensor
from collections import OrderedDict


def get_tta_transforms(gaussian_std: float = 0.005, soft=False, clip_inputs=False, dataset='cifar'):
    img_shape = (32, 32, 3) if 'cifar' in dataset else (224, 224, 3)
    print('img_shape in cotta transform', img_shape)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
            fill=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


class ResiTTA(nn.Module):
    def __init__(self, model, paras_optim, capacity, num_class, dataset, ema_nu, update_frequency,
                 steps, bn_alpha, lambda_bn_d, lambda_bn_w, e_margin=0.4, class_balance=True):
        super().__init__()

        self.bn_alpha = bn_alpha
        # self.prior = prior
        self.lambda_bn_d = lambda_bn_d
        self.lambda_bn_w = lambda_bn_w
        self.lambda_bn_d_ema = lambda_bn_d
        self.lr = paras_optim.lr

        self.bn_modules = []
        self.model = self.configure_model(model)
        params, param_names = self.collect_params(self.model)
        self.optimizer = setup_optimizer(params, paras_optim)

        threshold = e_margin * math.log(num_class)
        self.mem = LowEntropyMemoryBankV2(capacity=capacity, num_class=num_class, threshold=threshold,
                                          class_balance=class_balance)

        model_ema = deepcopy(model)
        for param in model_ema.parameters():
            param.detach_()
        self.model_ema = model_ema
        self.bn_modules_ema = get_batch_norm_modules(self.model_ema)
        for bn in self.bn_modules_ema:
            bn.lambda_bn_d = self.lambda_bn_d_ema

        self.transform = get_tta_transforms(dataset=dataset)
        self.ema_nu = ema_nu
        self.update_frequency = update_frequency  # actually the same as the size of memory bank
        self.num_instance = 0
        self.steps = steps

        self.scaler = GradScaler()

    def forward(self, batch):
        if isinstance(batch, dict):
            x = batch['img']
        else:
            x = batch

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return dict(logits=outputs)

    def collect_params(self, model: nn.Module):
        names = []
        params = []

        for n, p in model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        return params, names

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        with torch.no_grad():
            model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(batch_data)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        # add into memory
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.num_instance += 1

            if self.num_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)

        return ema_out

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()
        # get memory data
        sup_data, _ = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            # print('sup_data', len(sup_data))
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                sup_data = torch.stack(sup_data)
                strong_sup_aug = self.transform(sup_data)
                ema_sup_out = self.model_ema(sup_data)
                stu_sup_out = model(strong_sup_aug)
                # instance_weight = timeliness_reweighting(ages)
                instance_weight = 1  # no time reweighting
                l_sup = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

                # soft-alignment on bn weights

                if self.lambda_bn_w > 0:
                    l_soft_alignment = []
                    for bn_module in self.bn_modules:
                        l_soft_alignment.append(bn_module.get_soft_alignment_loss_weight())
                    l_soft_alignment = torch.stack(l_soft_alignment).sum()
                else:
                    l_soft_alignment = torch.tensor(0.0).cuda()

                l = l_sup + l_soft_alignment * self.lambda_bn_w
            optimizer.zero_grad()

            self.scaler.scale(l).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            # l.backward()
            # optimizer.step()

            # update for bn alignment
            for bn_module in self.bn_modules:
                bn_module.regularize_statistics()

            for bn_module in self.bn_modules_ema:
                bn_module.regularize_statistics()

            self.update_ema_variables(self.model_ema, self.model, self.ema_nu)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def configure_model(self, model: nn.Module):

        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = SoftAlignmentBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = SoftAlignmentBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer,
                                self.bn_alpha,
                                self.lambda_bn_d,
                                self.lambda_bn_w)
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
            self.bn_modules.append(momentum_bn)
        self.bn_module_names = normlayer_names
        return model

    def reset(self):
        print('reset is omitted in MoTTAClassificationV2')


def setup_optimizer(params, paras_optim):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if paras_optim.name == 'Adam':
        return optim.Adam(params,
                          lr=paras_optim.lr,
                          betas=(paras_optim.beta, 0.999),
                          weight_decay=paras_optim.wd,
                          foreach=True)
    elif (paras_optim.name == 'SGD'):
        return optim.SGD(params,
                         lr=paras_optim.lr,
                         momentum=paras_optim.momentum,
                         dampening=paras_optim.dampening,
                         weight_decay=paras_optim.weight_decay,
                         nesterov=paras_optim.nesterov)
    elif paras_optim.name == 'AdamW':
        return optim.AdamW(params,
                           lr=paras_optim.lr,
                           weight_decay=paras_optim.weight_decay, )
    else:
        raise NotImplementedError


class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"


class LowEntropyMemoryBankV2:
    def __init__(self, capacity, num_class, threshold, class_balance=True):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = max(self.capacity / self.num_class, 1)

        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]
        self.threshold = threshold
        self.class_balance = class_balance

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls)
        return occupancy

    def per_class_dist(self):
        per_class_occupied = [0] * self.num_class
        for cls, class_list in enumerate(self.data):
            per_class_occupied[cls] = len(class_list)

        return per_class_occupied

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if occupied == max_occupied:
                classes.append(i)
        return classes

    def add_instance(self, instance):
        assert (len(instance) == 3)
        self.add_age()
        x, prediction, uncertainty = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=1)

        if uncertainty < self.threshold:
            if self.get_occupancy() >= self.capacity:
                # remove the oldest instance from majority classes

                if self.class_balance:
                    majority_classes = self.get_majority_classes()
                    random_cls = random.choice(majority_classes)
                    self.data[random_cls].pop(random.randint(0, len(self.data[random_cls]) - 1))
                else:
                    class_occupied = self.per_class_dist()
                    non_empty_classes = [i for i, occupied in enumerate(class_occupied) if occupied > 0]
                    random_cls = random.choice(non_empty_classes)
                    self.data[random_cls].pop(random.randint(0, len(self.data[random_cls]) - 1))

            self.data[prediction].append(new_item)

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()
        return

    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age)

        tmp_age = [x / self.capacity for x in tmp_age]

        return tmp_data, tmp_age


class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum, lambda_bn_d, lambda_bn_w):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.source_mean = deepcopy(bn_layer.running_mean)
            self.source_var = deepcopy(bn_layer.running_var)
            self.target_mean = deepcopy(bn_layer.running_mean)
            self.target_var = deepcopy(bn_layer.running_var)

            self.source_num = bn_layer.num_batches_tracked

        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)

        self.source_weight = deepcopy(bn_layer.weight).detach()
        self.source_bias = deepcopy(bn_layer.bias).detach()

        self.eps = bn_layer.eps

        self.lambda_bn_d = lambda_bn_d
        self.lambda_bn_w = lambda_bn_w

    def forward(self, x):
        raise NotImplementedError

    def get_soft_alignment_loss_weight(self):
        # return F.mse_loss(self.weight, self.source_weight) + F.mse_loss(self.bias, self.source_bias)
        return torch.sum((self.weight - self.source_weight) ** 2) + torch.sum((self.bias - self.source_bias) ** 2)

    @torch.no_grad()
    def regularize_statistics(self):
        gradient_mean = 2 * (self.target_mean - self.source_mean)

        target_std = torch.sqrt(self.target_var + self.eps)
        source_std = torch.sqrt(self.source_var + self.eps)
        gradient_std = 2 * target_std - 2 * source_std

        target_std = target_std - self.lambda_bn_d * gradient_std

        self.target_mean.copy_(self.target_mean - self.lambda_bn_d * gradient_mean)
        self.target_var.copy_(target_std ** 2)


class SoftAlignmentBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=0, unbiased=False, keepdim=False)  # (C,)
            mean = (1 - self.momentum) * self.target_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.target_var + self.momentum * b_var
            self.target_mean, self.target_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1), var.view(1, -1)
        else:
            mean, var = self.target_mean.view(1, -1), self.target_var.view(1, -1)

        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias = self.bias.view(1, -1)

        return x * weight + bias


class SoftAlignmentBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            with torch.no_grad():
                b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False, keepdim=False)  # (C,)
                mean = (1 - self.momentum) * self.target_mean + self.momentum * b_mean
                var = (1 - self.momentum) * self.target_var + self.momentum * b_var
                self.target_mean.copy_(mean)
                self.target_var.copy_(var)
        else:
            mean, var = self.target_mean, self.target_var

        return F.batch_norm(x, mean, var, self.weight, self.bias, False, 0, self.eps)


# This is a sample Python script.
class ImageNormalizer(nn.Module):

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore


def normalize_model(model: nn.Module, mean: Tuple[float, float, float],
                    std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([('normalize', ImageNormalizer(mean, std)),
                          ('model', model)])
    return nn.Sequential(layers)

def get_batch_norm_modules(model):
    bn_modules = []
    for name, sub_module in model.named_modules():
        if isinstance(sub_module, SoftAlignmentBN1d) or isinstance(sub_module, SoftAlignmentBN2d):
            bn_modules.append(sub_module)
    return bn_modules


@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])
        else:
            setattr(module, names[i], value)
