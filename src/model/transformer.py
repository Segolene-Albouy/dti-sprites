from abc import ABCMeta, abstractmethod
from copy import deepcopy

from kornia.geometry import homography_warp
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, RMSprop
from torchvision.models import vgg16_bn
from torchvision.transforms.functional import resize
import torchvision
from .mini_resnet import get_resnet_model as get_mini_resnet_model
from .resnet import get_resnet_model
from .tools import copy_with_noise, get_output_size, TPSGrid, create_mlp, get_clamp_func
from ..utils.logger import print_warning

import omegaconf

N_HIDDEN_UNITS = 128
N_LAYERS = 2


class PrototypeTransformationNetwork(nn.Module):
    def __init__(
        self, in_channels, img_size, n_prototypes, transformation_sequence, **kwargs
    ):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.sequence_name = transformation_sequence
        # if self.sequence_name in ["id", "identity"]:
        #    return None

        encoder = kwargs.get("encoder", None)
        if encoder is not None:
            self.shared_enc = True
            self.encoder = encoder
            self.enc_out_channels = self.encoder.out_ch
        else:
            self.shared_enc = False
            encoder_kwargs = {
                "in_channels": in_channels,
                "encoder_name": kwargs.get("encoder_name", "resnet20"),
                "img_size": img_size,
                "with_pool": kwargs.get("with_pool", True),
            }
            encoder_name = encoder_kwargs["encoder_name"]
            self.encoder = Encoder(**encoder_kwargs)
            self.enc_out_channels = get_output_size(
                in_channels, img_size, self.encoder, encoder_name
            )
        tsf_kwargs = {
            "freeze_frg": kwargs.get("freeze_frg", False),
            "in_channels": self.enc_out_channels,
            "img_size": img_size,
            "layer_size": kwargs.get("layer_size", img_size),
            "color_channels": kwargs.get("color_channels", 3),
            "sequence_name": self.sequence_name,
            "grid_size": kwargs.get("grid_size", 4),
            "kernel_size": kwargs.get("kernel_size", 3),
            "padding_mode": kwargs.get("padding_mode", "zeros"),
            "curriculum_learning": kwargs.get("curriculum_learning", False),
            "shared_t": kwargs.get("shared_t", False),
            "use_clamp": kwargs.get("use_clamp", "soft"),
            "n_hidden_layers": kwargs.get("n_hidden_layers", N_LAYERS),
        }
        self.shared_t = tsf_kwargs["shared_t"]
        if self.shared_t:
            self.tsf_sequences = TransformationSequence(**deepcopy(tsf_kwargs))
        else:
            self.tsf_sequences = nn.ModuleList(
                [
                    TransformationSequence(**deepcopy(tsf_kwargs))
                    for i in range(n_prototypes)
                ]
            )

    def get_parameters(self):
        if self.shared_enc:
            return self.tsf_sequences.parameters()
        return self.parameters()

    @property
    def is_identity(self):
        return self.sequence_name in ["id", "identity"]

    def forward(self, x, prototypes, features=None):
        # x shape is BCHW, prototypes list of K elements of size BCHW
        features = self.encoder(x) if features is None else features
        if self.is_identity:
            inp = x.unsqueeze(1).expand(-1, self.n_prototypes, -1, -1, -1)
            target = prototypes.permute(1, 0, 2, 3, 4)
        else:
            inp = x.unsqueeze(1).expand(-1, self.n_prototypes, -1, -1, -1)
            if self.shared_t:
                target = [self.tsf_sequences(proto, features) for proto in prototypes]
            else:
                target = [
                    tsf_seq(proto, features)
                    for tsf_seq, proto in zip(self.tsf_sequences, prototypes)
                ]
            target = torch.stack(target, dim=1)
        return inp, target, features

    @torch.no_grad()
    def inverse_transform(self, x, features=None):
        """Apply inverse transformation to the inputs (for visual purposes)"""
        if self.is_identity:
            return x.unsqueeze(1).expand(-1, self.n_prototypes, -1, -1, -1)
        else:
            features = self.encoder(x[:, :3, :, :]) if features is None else features
            if self.shared_t:
                y = self.tsf_sequences(x, features, inverse=True)
            else:
                y = torch.stack(
                    [
                        tsf_seq(x, features, inverse=True)
                        for tsf_seq in self.tsf_sequences
                    ],
                    1,
                )
            return y

    def predict_parameters(self, x=None, features=None):
        features = self.encoder(x) if features is None else features
        if self.shared_t:
            return self.tsf_sequences.predict_parameters(features)
        return torch.stack(
            [tsf_seq.predict_parameters(features) for tsf_seq in self.tsf_sequences],
            dim=0,
        )

    def apply_parameters(self, prototypes, betas, is_var=False):
        if self.is_identity:
            return prototypes
        else:
            target = (
                [
                    self.tsf_sequences.apply_parameters(proto, beta, is_var=is_var)
                    for proto, beta in zip(prototypes, betas)
                ]
                if self.shared_t
                else [
                    tsf_seq.apply_parameters(proto, beta, is_var=is_var)
                    for tsf_seq, proto, beta in zip(
                        self.tsf_sequences, prototypes, betas
                    )
                ]
            )
            return torch.stack(target, dim=1)

    def restart_branch_from(self, i, j, noise_scale=0.001):
        if self.is_identity or self.shared_t:
            return None

        self.tsf_sequences[i].load_with_noise(
            self.tsf_sequences[j], noise_scale=noise_scale
        )
        if hasattr(self, "optimizer"):
            opt = self.optimizer
            if isinstance(opt, (Adam,)):
                for param_i, param_j in zip(
                    self.tsf_sequences[i].parameters(),
                    self.tsf_sequences[j].parameters(),
                ):
                    if param_i in opt.state:
                        opt.state[param_i]["exp_avg"] = opt.state[param_j]["exp_avg"]
                        opt.state[param_i]["exp_avg_sq"] = opt.state[param_j][
                            "exp_avg_sq"
                        ]
            elif isinstance(opt, (RMSprop,)):
                for param_i, param_j in zip(
                    self.tsf_sequences[i].parameters(),
                    self.tsf_sequences[j].parameters(),
                ):
                    if param_i in opt.state:
                        opt.state[param_i]["square_avg"] = opt.state[param_j][
                            "square_avg"
                        ]
            else:
                raise NotImplementedError(
                    "unknown optimizer: you should define how to reinstanciate statistics if any"
                )

    def add_noise(self, noise_scale=0.001):
        for i in range(len(self.tsf_sequences)):
            self.tsf_sequences[i].load_with_noise(
                self.tsf_sequences[i], noise_scale=noise_scale
            )

    def step(self):
        if not self.is_identity:
            if self.shared_t:
                self.tsf_sequences.step()
            else:
                [tsf_seq.step() for tsf_seq in self.tsf_sequences]

    def activate_all(self):
        if not self.is_identity:
            if self.shared_t:
                self.tsf_sequences.activate_all()
            else:
                [tsf_seq.activate_all() for tsf_seq in self.tsf_sequences]

    @property
    def only_id_activated(self):
        if self.shared_t:
            return self.tsf_sequences.only_id_activated
        return self.tsf_sequences[0].only_id_activated

    def set_optimizer(self, opt):
        self.optimizer = opt


class Encoder(nn.Module):
    def __init__(self, in_channels, encoder_name="default", **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.with_pool = kwargs.get("with_pool", True)
        if encoder_name == "default":
            seq = [
                nn.Conv2d(in_channels, 8, kernel_size=7),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.BatchNorm2d(10),
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),
            ]
        elif encoder_name == "vgg16":
            seq = [vgg16_bn(pretrained=False).features]
        elif encoder_name == "dinov2":
            dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
            seq = [dino]
        else:
            try:
                resnet = get_resnet_model(encoder_name)(
                    pretrained=False, progress=False
                )
                seq = [
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
                    resnet.layer2,
                    resnet.layer3,
                    resnet.layer4,
                ]
            except KeyError:
                resnet = get_mini_resnet_model(encoder_name)(in_channels=in_channels)
                seq = [
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.layer1,
                    resnet.layer2,
                    resnet.layer3,
                ]
        if self.with_pool:
            size = (
                self.with_pool if isinstance(self.with_pool, (tuple, list)) else (1, 1)
            )
            seq.append(torch.nn.AdaptiveAvgPool2d(output_size=size))
        self.encoder = nn.Sequential(*seq)
        img_size = kwargs.get("img_size", None)
        if img_size is not None:
            self.out_ch = get_output_size(
                in_channels, img_size, self.encoder, encoder_name
            )
        else:
            None
        self.encoder_name = encoder_name

    def forward(self, x):
        if self.encoder_name == "dinov2":
            x = resize(x, (28, 28))
        return self.encoder(x).flatten(1)


class TransformationSequence(nn.Module):
    def __init__(self, in_channels, sequence_name, **kwargs):
        super().__init__()
        self.tsf_names = sequence_name.split("_")
        self.n_tsf = len(self.tsf_names)

        tsf_modules = []
        for name in self.tsf_names:
            tsf_modules.append(self.get_module(name)(in_channels, **kwargs))
        self.tsf_modules = nn.ModuleList(tsf_modules)

        curriculum_learning = kwargs.get("curriculum_learning", False)
        if curriculum_learning:
            assert (
                isinstance(curriculum_learning, (list, tuple, omegaconf.listconfig.ListConfig))
                and len(curriculum_learning) == self.n_tsf - 1
            )
            self.act_milestones = curriculum_learning
            n_act = 1 + (np.asarray(curriculum_learning) == 0).sum()
            self.next_act_idx = n_act
            self.register_buffer(
                "activations",
                torch.Tensor([True] * n_act + [False] * (self.n_tsf - n_act)).bool(),
            )
        else:
            self.act_milestones = [-1] * self.n_tsf
            self.next_act_idx = self.n_tsf
            self.register_buffer(
                "activations", torch.Tensor([True] * (self.n_tsf)).bool()
            )
        self.cur_milestone = 0

    @staticmethod
    def get_module(name):
        return {
            # standard
            "id": IdentityModule,
            "identity": IdentityModule,
            "col": ColorModule,
            "color": ColorModule,
            "linearcolor": LinearColorModule,
            # spatial
            "aff": AffineModule,
            "affine": AffineModule,
            "pos": PositionModule,
            "position": PositionModule,
            "proj": ProjectiveModule,
            "projective": ProjectiveModule,
            "homography": ProjectiveModule,
            "sim": SimilarityModule,
            "similarity": SimilarityModule,
            "tps": TPSModule,
            "thinplatespline": TPSModule,
            "translation": TranslationModule,
            # morphological
            "morpho": MorphologicalModule,
            "morphological": MorphologicalModule,
        }[name]

    def forward(self, x, features, inverse=False):
        for module, activated in zip(self.tsf_modules, self.activations):
            if activated:
                x = module(x, features, inverse)
        return x

    def predict_parameters(self, features):
        betas = []
        for module, activated in zip(self.tsf_modules, self.activations):
            if activated:
                betas.append(module.regressor(features))
        return torch.cat(betas, dim=1)

    def apply_parameters(self, x, beta, is_var=False):
        betas = torch.split(
            beta,
            [
                d.dim_parameters
                for d, act in zip(self.tsf_modules, self.activations)
                if act
            ],
            dim=1,
        )
        for module, activated, beta in zip(self.tsf_modules, self.activations, betas):
            if activated and (
                not is_var
                or isinstance(module, (AffineModule, ProjectiveModule, TPSModule))
            ):
                x = module.transform(x, beta)
        return x

    def load_with_noise(self, tsf_seq, noise_scale):
        for k in range(self.n_tsf):
            self.tsf_modules[k].load_with_noise(tsf_seq.tsf_modules[k], noise_scale)

    def step(self):
        self.cur_milestone += 1
        while (
            self.next_act_idx < self.n_tsf
            and self.act_milestones[int(self.next_act_idx - 1)] == self.cur_milestone
        ):
            self.activations[self.next_act_idx] = True
            self.next_act_idx += 1

    def activate_all(self):
        for k in range(self.n_tsf):
            self.activations[k] = True
        self.next_act_idx = self.n_tsf

    @property
    def only_id_activated(self):
        for m, act in zip(self.tsf_modules, self.activations):
            if not isinstance(m, (IdentityModule,)) and act:
                return False
        return True


class _AbstractTransformationModule(nn.Module):
    __metaclass__ = ABCMeta

    def forward(self, x, features, inverse=False):
        beta = self.regressor(features)
        return self.transform(x, beta, inverse)

    def transform(self, x, beta, inverse=False):
        return self._transform(x, beta, inverse)

    @abstractmethod
    def _transform(self, x, beta, inverse=False):
        pass

    def load_with_noise(self, module, noise_scale):
        self.load_state_dict(module.state_dict())
        self.regressor[-1].bias.data.copy_(
            copy_with_noise(module.regressor[-1].bias, noise_scale)
        )

    @property
    def dim_parameters(self):
        return self.regressor[-1].out_features


########################
#    Standard Modules
########################


class IdentityModule(_AbstractTransformationModule):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(in_channels, 0))
        self.register_buffer("identity", torch.zeros(0))

    def forward(self, x, *args, **kwargs):
        return x

    def _transform(self, x, *args, **kwargs):
        return x

    def load_with_noise(self, module, noise_scale):
        pass


class ColorModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.color_ch = kwargs.get("color_channels", 3)
        clamp_name = kwargs.get("use_clamp", False)
        self.clamp_func = get_clamp_func(clamp_name)
        n_layers = kwargs.get("n_hidden_layers", N_LAYERS)
        self.regressor = create_mlp(
            in_channels, self.color_ch * 2, N_HIDDEN_UNITS, n_layers
        )

        # Identity transformation parameters and regressor initialization
        self.register_buffer("identity", torch.eye(self.color_ch, self.color_ch))
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.zero_()

    def _transform(self, x, beta, inverse=False):
        if inverse:
            return x
        else:
            if x.size(1) == 2 or x.size(1) > 3:
                x, mask = torch.split(
                    x, [self.color_ch, x.size(1) - self.color_ch], dim=1
                )
            else:
                mask = None
            if x.size(1) == 1:
                x = x.expand(-1, 3, -1, -1)

            weight, bias = torch.split(beta.view(-1, self.color_ch, 2), [1, 1], dim=2)
            weight = (
                weight.expand(-1, -1, self.color_ch) * self.identity + self.identity
            )
            bias = bias.unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))

            output = torch.einsum("bij, bjkl -> bikl", weight, x) + bias
            output = self.clamp_func(output)
        if mask is not None:
            output = torch.cat([output, mask], dim=1)
        return output

class LinearColorModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.color_ch = 3 #kwargs.get("color_channels", 3)
        clamp_name = kwargs.get("use_clamp", False)
        self.clamp_func = get_clamp_func(clamp_name)
        n_layers = kwargs.get("n_hidden_layers", N_LAYERS)
        self.regressor = create_mlp(
            in_channels, self.color_ch, N_HIDDEN_UNITS, n_layers
        )

        # Identity transformation parameters and regressor initialization
        self.register_buffer("identity", torch.eye(self.color_ch, self.color_ch))
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.zero_()

    def _transform(self, x, beta, inverse=False):
        if inverse:
            return x
        else:
            if x.size(1) == 2 or x.size(1) > 3:
                x, mask = torch.split(
                    x, [self.color_ch, x.size(1) - self.color_ch], dim=1
                )
            else:
                mask = None
            if x.size(1) == 1:
                x = x.expand(-1, 3, -1, -1)

            weight = beta.view(-1, self.color_ch, 1)
            weight = (
                weight.expand(-1, -1, self.color_ch) * self.identity + self.identity
            )
            output = torch.einsum("bij, bjkl -> bikl", weight, x)
            output = self.clamp_func(output)
        if mask is not None:
            output = torch.cat([output, mask], dim=1)
        return output

########################
#    Spatial Modules
########################


class AffineModule(_AbstractTransformationModule):
    def __init__(self, in_channels, img_size, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.padding_mode = kwargs.get("padding_mode", "border")
        self.freeze_frg = kwargs.get("freeze_frg", False)
        n_layers = kwargs.get("n_hidden_layers", N_LAYERS)
        self.regressor = create_mlp(in_channels, 6, N_HIDDEN_UNITS, n_layers)

        # Identity transformation parameters and regressor initialization
        self.register_buffer(
            "identity", torch.cat([torch.eye(2, 2), torch.zeros(2, 1)], dim=1)
        )
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.zero_()

    def _transform(self, x, beta, inverse=False):
        beta = beta.view(-1, 2, 3) + self.identity
        if inverse:
            row = torch.tensor(
                [[[0, 0, 1]]] * x.size(0), dtype=torch.float, device=beta.device
            )
            beta = torch.cat([beta, row], dim=1)
            beta = torch.inverse(beta)[:, :2, :]

            if x.shape[1] == 4:
                grid = F.affine_grid(
                    beta,
                    (x.size(0), 1, self.img_size[0], self.img_size[1]),
                    align_corners=False,
                )
                out = F.grid_sample(
                    x[:, -1, :, :].unsqueeze(1),
                    grid,
                    mode="bilinear",
                    padding_mode=self.padding_mode,
                    align_corners=False,
                )
                return torch.cat([x[:, :3, ...], out], dim=1)

        if self.freeze_frg:
            grid = F.affine_grid(
                beta,
                (x.size(0), 1, self.img_size[0], self.img_size[1]),
                align_corners=False,
            )
            out = F.grid_sample(
                x[:, -1, ...].unsqueeze(1),
                grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=False,
            )
            return torch.cat([x[:, : x.size(1) - 1, ...], out], dim=1)
        else:
            grid = F.affine_grid(
                beta,
                (x.size(0), x.size(1), self.img_size[0], self.img_size[1]),
                align_corners=False,
            )
            return F.grid_sample(
                x,
                grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=False,
            )


class TranslationModule(_AbstractTransformationModule):
    def __init__(self, in_channels, img_size, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.padding_mode = kwargs.get("padding_mode", "border")
        n_layers = kwargs.get("n_hidden_layers", N_LAYERS)
        self.regressor = create_mlp(in_channels, 2, N_HIDDEN_UNITS, n_layers)

        # Identity transformation parameters and regressor initialization
        self.register_buffer(
            "identity", torch.cat([torch.eye(2, 2), torch.zeros(2, 1)], dim=1)
        )
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.zero_()

    def _transform(self, x, beta, inverse=False):
        if inverse:
            print_warning("Inverse transform for PositionModule is not implemented.")
            return x
        t = beta
        scale = torch.ones(t.shape[0]).to(t.device)
        scale = scale[..., None, None].expand(-1, 2, 2) * torch.eye(2, 2).to(t.device)
        beta = torch.cat([scale, t.unsqueeze(2)], dim=2) + self.identity
        grid = F.affine_grid(
            beta,
            (x.size(0), x.size(1), self.img_size[0], self.img_size[1]),
            align_corners=False,
        )
        out = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )   
        return out

class PositionModule(_AbstractTransformationModule):
    def __init__(self, in_channels, img_size, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.padding_mode = kwargs.get("padding_mode", "border")
        n_layers = kwargs.get("n_hidden_layers", N_LAYERS)
        self.regressor = create_mlp(in_channels, 3, N_HIDDEN_UNITS, n_layers)

        # Identity transformation parameters and regressor initialization
        self.register_buffer(
            "identity", torch.cat([torch.eye(2, 2), torch.zeros(2, 1)], dim=1)
        )
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.zero_()

    def _transform(self, x, beta, inverse=False):
        if inverse:
            print_warning("Inverse transform for PositionModule is not implemented.")
            return x
        s, t = beta.split([1, 2], dim=1)
        s = torch.exp(s)
        scale = s[..., None].expand(-1, 2, 2) * torch.eye(2, 2).to(s.device)
        beta = torch.cat([scale, t.unsqueeze(2)], dim=2) + self.identity
        grid = F.affine_grid(
            beta,
            (x.size(0), x.size(1), self.img_size[0], self.img_size[1]),
            align_corners=False,
        )
        out = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )   
        return out

class SimilarityModule(_AbstractTransformationModule):
    def __init__(self, in_channels, img_size, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.padding_mode = kwargs.get("padding_mode", "border")
        n_layers = kwargs.get("n_hidden_layers", N_LAYERS)
        # self.regressor = create_mlp(in_channels, 4, N_HIDDEN_UNITS, n_layers)
        self.regressor = create_mlp(in_channels, 1, N_HIDDEN_UNITS, n_layers)
        self.layer_size = kwargs.get("layer_size")
        # Identity transformation parameters and regressor initialization
        self.register_buffer(
            "identity", torch.cat([torch.eye(2, 2), torch.zeros(2, 1)], dim=1)
        )
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.zero_()

    def _transform(self, x, beta, inverse=False):
        if inverse:
            print_warning("Inverse transform for SimilarityModule is not implemented.")
            return x
        # a, b, t = beta.split([1, 1, 2], dim=1)
        b = beta
        t = torch.zeros(b.shape[0], 2).to(b.device)
        # a_eye = torch.eye(2, 2).to(a.device)
        b_eye = torch.Tensor([[0, -1], [1, 0]]).to(b.device)
        # scaled_rot = (
        #     a[..., None].expand(-1, 2, 2) * a_eye
        #     + b[..., None].expand(-1, 2, 2) * b_eye
        # )
        scaled_rot = b[..., None].expand(-1, 2, 2) * b_eye 
        beta = torch.cat([scaled_rot, t.unsqueeze(2)], dim=2) + self.identity
        grid = F.affine_grid(
            beta,
            (x.size(0), x.size(1), self.layer_size[0], self.layer_size[1]),
            align_corners=False,
        )
        out = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        )
        return out


class ProjectiveModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.padding_mode = kwargs.get("padding_mode", "border")
        n_layers = kwargs.get("n_hidden_layers", N_LAYERS)
        self.regressor = create_mlp(in_channels, 9, N_HIDDEN_UNITS, n_layers)

        # Identity transformation parameters and regressor initialization
        self.register_buffer("identity", torch.eye(3, 3))
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.zero_()

    def _transform(self, x, beta, inverse=False):
        beta = beta.view(-1, 3, 3) + self.identity
        if inverse:
            beta = torch.inverse(beta)
        return homography_warp(
            x,
            beta,
            dsize=(x.size(2), x.size(3)),
            mode="bilinear",
            padding_mode=self.padding_mode,
        )


class TPSModule(_AbstractTransformationModule):
    def __init__(self, in_channels, img_size, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.padding_mode = kwargs.get("padding_mode", "border")
        self.freeze_frg = kwargs.get("freeze_frg", False)
        self.grid_size = kwargs.get("grid_size", 4)
        n_layers = kwargs.get("n_hidden_layers", N_LAYERS)
        self.regressor = create_mlp(
            in_channels, self.grid_size**2 * 2, N_HIDDEN_UNITS, n_layers
        )
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.grid_size), torch.linspace(-1, 1, self.grid_size)
        )
        target_control_points = torch.stack([x.flatten(), y.flatten()], dim=1)
        self.tps_grid = TPSGrid(img_size, target_control_points)

        # Identity transformation parameters and regressor initialization
        self.register_buffer("identity", target_control_points)
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.zero_()

    def _transform(self, x, beta, inverse=False):
        if inverse:
            print_warning("Inverse transform for TPSModule is not implemented.")
            return x
        source_control_points = self.identity + beta.view(x.size(0), -1, 2)
        grid = self.tps_grid(source_control_points).view(x.size(0), *self.img_size, 2)
        if self.freeze_frg:
            out = F.grid_sample(
                x[:, -1, ...].unsqueeze(1),
                grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=False,
            )
            return torch.cat([x[:, : x.size(1) - 1, ...], out], dim=1)
        else:
            return F.grid_sample(
                x,
                grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=False,
            )


########################
#    Morphological Modules
########################


class MorphologicalModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.kernel_size = kwargs.get("kernel_size", 3)
        self.freeze_frg = kwargs.get("freeze_frg", False)
        assert isinstance(self.kernel_size, (int, float))
        self.padding = self.kernel_size // 2
        n_layers = kwargs.get("n_hidden_layers", N_LAYERS)
        self.regressor = create_mlp(
            in_channels, self.kernel_size**2 + 1, N_HIDDEN_UNITS, n_layers
        )

        # Identity transformation parameters and regressor initialization
        weights = torch.full(
            (self.kernel_size, self.kernel_size), fill_value=-5, dtype=torch.float
        )
        center = self.kernel_size // 2
        weights[center, center] = 5
        self.register_buffer("identity", torch.cat([torch.zeros(1), weights.flatten()]))
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.zero_()

    def _transform(self, x, beta, inverse=False):
        if inverse:
            print_warning(
                "Inverse transform for MorphologicalModule is not implemented."
            )
            return x
        beta = beta + self.identity
        alpha, weights = torch.split(beta, [1, self.kernel_size**2], dim=1)
        if self.freeze_frg:
            out = self.smoothmax_kernel(
                x[:, -1, ...].unsqueeze(1), alpha, torch.sigmoid(weights)
            )
            return torch.cat([x[:, : x.size(1) - 1, ...], out], dim=1)
        else:
            assert x.shape[1] == 1
            return self.smoothmax_kernel(x, alpha, torch.sigmoid(weights))

    def smoothmax_kernel(self, x, alpha, kernel):
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.flatten()[:, None, None]

        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        x_unf = F.unfold(x, self.kernel_size, padding=self.padding).transpose(1, 2)
        w = torch.exp(alpha * x_unf) * kernel.unsqueeze(1).expand(-1, x_unf.size(1), -1)
        return ((x_unf * w).sum(2) / w.sum(2)).view(B, C, H, W)
