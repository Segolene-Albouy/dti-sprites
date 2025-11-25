from abc import ABCMeta, abstractmethod, ABC
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


def normalize_tsf_name(name):
    return {
        'id': 'identity', 'identity': 'identity',
        'col': 'color', 'color': 'color', 'linearcolor': 'linearcolor',
        'aff': 'affine', 'affine': 'affine',
        'pos': 'isotropictranslation', 'position': 'isotropictranslation',
        'proj': 'projective', 'projective': 'projective', 'homography': 'projective',
        'sim': 'similarity', 'similarity': 'similarity',
        'rotation': 'rotation', 'translation': 'translation',
        'tps': 'tps', 'thinplatespline': 'tps',
        'morpho': 'morphological', 'morphological': 'morphological',
        'isoscaling': 'isotropicscaling', 'isotropicscaling': 'isotropicscaling',
        'anisoscaling': 'anisotropicscaling', 'anisotropicscaling': 'anisotropicscaling',
        'isotranslation': 'isotropictranslation', 'isotropictranslation': 'isotropictranslation',
        'anisotranslation': 'anisotropictranslation', 'anisotropictranslation': 'anisotropictranslation',
    }.get(name.lower(), name.lower())


def normalize_argmin_idx(argmin_idx, batch_size):
    if isinstance(argmin_idx, (int, np.integer)):
        return [argmin_idx] * batch_size
    elif isinstance(argmin_idx, torch.Tensor):
        return argmin_idx.cpu().tolist()
    elif isinstance(argmin_idx, np.ndarray):
        return argmin_idx.tolist()
    return argmin_idx


class PrototypeTransformationNetwork(nn.Module):
    def __init__(
        self, in_channels, img_size, n_prototypes, transformation_sequence, **kwargs
    ):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.sequence_name = transformation_sequence

        encoder = kwargs.get("encoder", None)
        self.shared_enc = encoder is not None

        if self.shared_enc:
            self.encoder = encoder
            self.enc_out_channels = self.encoder.out_ch
        else:
            self.encoder = Encoder(**{
                "in_channels": in_channels,
                "encoder_name": kwargs.get("encoder_name", "resnet20"),
                "img_size": img_size,
                "with_pool": kwargs.get("with_pool", True),
            })
            self.enc_out_channels = get_output_size(
                in_channels, img_size, self.encoder, kwargs.get("encoder_name", "resnet20")
            )

        self.tsf_kwargs = {
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
        self.shared_t = self.tsf_kwargs["shared_t"]
        if self.shared_t:
            self.tsf_sequences = TransformationSequence(**deepcopy(self.tsf_kwargs))
        else:
            self.tsf_sequences = nn.ModuleList(
                [
                    TransformationSequence(**deepcopy(self.tsf_kwargs))
                    for _ in range(n_prototypes)
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
                    for tsf_seq, proto, beta in zip(self.tsf_sequences, prototypes, betas)
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
                        opt.state[param_i]["exp_avg_sq"] = opt.state[param_j]["exp_avg_sq"]
            elif isinstance(opt, (RMSprop,)):
                for param_i, param_j in zip(
                    self.tsf_sequences[i].parameters(),
                    self.tsf_sequences[j].parameters(),
                ):
                    if param_i in opt.state:
                        opt.state[param_i]["square_avg"] = opt.state[param_j]["square_avg"]
            else:
                raise NotImplementedError(
                    "unknown optimizer: you should define how to reinstanciate statistics if any"
                )
        return None


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

    def is_tsf_in_seq(self, tsf_name):
        """Check if transformation name is in sequence, handling aliases."""
        tsf_name = normalize_tsf_name(tsf_name)
        seq_tsf = [normalize_tsf_name(t) for t in self.sequence_name.split('_')]
        return tsf_name in seq_tsf

    @torch.no_grad()
    def get_tsf_matrix(self, tsf_name, x, argmin_idx=0, feats=None):
        """
        Get transformation matrix for a specific transformation by name.

        Args:
            tsf_name: Name of transformation ('affine', 'color', etc.)
            x: Input images [B, C, H, W]
            argmin_idx: Prototype index - scalar (same for all batch) OR tensor/list [B] (per-image)
            feats: Precomputed encoder features [B, enc_out_channels] (optional)

        Returns:
            torch.Tensor: Transformation matrix or identity if not active/present
                Spatial transforms: [B, 3, 3] homogeneous matrices
                Color transforms: [B, 4, 4] homogeneous matrices
                TPS: [B, grid_size^2, 2] control points
                Morphological: [B, kernel_size, kernel_size] structuring element
        """
        batch_size = x.size(0)

        tsf_name = normalize_tsf_name(tsf_name)
        argmin_idx = normalize_argmin_idx(argmin_idx, batch_size)

        if len(argmin_idx) != batch_size:
            raise ValueError(
                f"argmin_idx length ({len(argmin_idx)}) must match batch size ({batch_size})"
            )
        if self.is_identity or not self.is_tsf_in_seq(tsf_name):
            module = TransformationSequence.get_module(tsf_name)(self.enc_out_channels, **self.tsf_kwargs)
            return module.get_matrix_representation(params=None, batch_size=batch_size, device=x.device)

        feats = self.encoder(x) if feats is None else feats

        # Process each image with its corresponding prototype
        all_matrices = []
        for b, idx in enumerate(argmin_idx):
            seq = self.tsf_sequences if self.shared_t else self.tsf_sequences[int(idx)]
            params = None
            for module, name, activated in zip(seq.tsf_modules, seq.tsf_names, seq.activations):
                if name == tsf_name and activated:
                    params = module.regressor(feats[b:b + 1])
                    break

            matrix = module.get_matrix_representation(params=params, batch_size=1, device=x.device)
            all_matrices.append(matrix[0])

        return torch.stack(all_matrices)

    @torch.no_grad()
    def get_batch_tsf_matrices(self, x, argmin_idx, tsf_names=None):
        """
        Get all transformation matrices for a batch efficiently.

        Args:
            x: Input images [B, C, H, W]
            argmin_idx: Cluster index for each image [B] (array-like)
            tsf_names: List of transformation names. If None, extracts all non-identity.

        Returns:
            List[List[Tensor]]: tsf_matrices[b][t] for image b and transform t
        """
        if tsf_names is None:
            tsf_names = [name for name in self.sequence_name.split('_') if name not in ['id', 'identity']]

        batch_size = x.size(0)
        if self.is_identity or len(tsf_names) == 0:
            return [[torch.tensor([]) for _ in tsf_names] for _ in range(batch_size)]

        feats = self.encoder(x)
        all_tsf_matrices = [
            self.get_tsf_matrix(tsf_name, x, argmin_idx=argmin_idx, feats=feats)
            for tsf_name in tsf_names
        ]

        return [
            [matrices[b] for matrices in all_tsf_matrices]
            for b in range(batch_size)
        ]


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
                self.with_pool if isinstance(self.with_pool, (tuple, list, omegaconf.listconfig.ListConfig)) else (1, 1)
            )
            seq.append(torch.nn.AdaptiveAvgPool2d(output_size=size))
        self.encoder = nn.Sequential(*seq)
        img_size = kwargs.get("img_size", None)
        if img_size is not None:
            self.out_ch = get_output_size(
                in_channels, img_size, self.encoder, encoder_name
            )
        else:
            pass
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

        curriculum_learning = kwargs.get("curriculum_learning", [])
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
                "activations", torch.Tensor([True] * self.n_tsf).bool()
            )
        self.cur_milestone = 0

    @staticmethod
    def get_module(name):
        tsf_modules = {
            # standard
            "id": IdentityModule,
            "identity": IdentityModule,
            "col": ColorModule,
            "color": ColorModule,
            "linearcolor": LinearColorModule,
            # spatial
            "aff": AffineModule,
            "affine": AffineModule,
            "proj": ProjectiveModule,
            "projective": ProjectiveModule,
            "homography": ProjectiveModule,
            "sim": SimilarityModule,
            "similarity": SimilarityModule,
            "rotation": RotationModule,
            "tps": TPSModule,
            "thinplatespline": TPSModule,
            "translation": TranslationModule,
            "isoscaling": IsotropicScalingModule,
            "isotropicscaling": IsotropicScalingModule,
            "anisoscaling": AnisotropicScalingModule,
            "anisotropicscaling": AnisotropicScalingModule,
            "pos": IsotropicTranslationModule,
            "position": IsotropicTranslationModule,
            "isotranslation": IsotropicTranslationModule, # PositionModule
            "isotropictranslation": IsotropicTranslationModule, # PositionModule
            "anisotranslation": AnisotropicTranslationModule,
            "anisotropictranslation": AnisotropicTranslationModule,
            # morphological
            "morpho": MorphologicalModule,
            "morphological": MorphologicalModule,
        }
        if name not in tsf_modules:
            raise ValueError(f"Unknown transformation '{name}'. Available: {sorted(set(tsf_modules.keys()))}")
        return tsf_modules[name]

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
            if activated and (not is_var or isinstance(module, (AffineModule, ProjectiveModule, TPSModule))):
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

    def __init__(self):
        super().__init__()
        self.out_channels = 1
        self.tensor = None
        self.img_size = None
        self.color_ch = 3
        self.n_layers = N_LAYERS
        self.padding_mode = "border"
        self.regressor = None

    def init(self, in_channels, **kwargs):
        self.img_size = kwargs.get("img_size", None)
        self.color_ch = kwargs.get("color_channels", 3)
        self.n_layers = kwargs.get("n_hidden_layers", N_LAYERS)
        self.padding_mode = kwargs.get("padding_mode", "border")

        if self.tensor is None:
            self.tensor = torch.cat([torch.eye(2, 2), torch.zeros(2, 1)], dim=1)

        self.regressor = create_mlp(in_channels, self.out_channels, N_HIDDEN_UNITS, self.n_layers)
        self.register_buffer("identity", self.tensor)
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.zero_()

    @staticmethod
    def get_out_channels(**kwargs):
        return 1

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

    def get_matrix_representation(self, params=None, batch_size=1, device=None):
        """
        Returns matrix representation of transformation.

        Args:
            params: Transformation parameters. If None, returns identity.
            batch_size: Number of samples in batch
            device: Target device for tensors

        Matrix formats by transformation type:
        - Spatial (affine, similarity, rotation, translation, position): [B, 3, 3]
        - Projective: [B, 3, 3]
        - Color: [B, 4, 4]
        - LinearColor: [B, 4, 4]
        - TPS: [B, grid_size^2, 2] (control points, not a matrix)
        - Morphological: [B, kernel_size, kernel_size] (structuring element)
        """
        if device is None:
            device = self.identity.device if hasattr(self, 'identity') else torch.device('cpu')

        if params is None:
            return self.get_id_matrix(batch_size, device)

        return self.param_2_matrix(params, batch_size)

    def get_id_matrix(self, batch_size, device):
        """Returns identity transformation matrix."""
        if not hasattr(self, 'identity'):
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement get_id_matrix or register 'identity' buffer"
            )

        identity = self.identity
        while len(identity.shape) < 3:
            identity = identity.unsqueeze(0)
        return identity.expand(batch_size, *identity.shape[1:]).to(device)

    @abstractmethod
    def param_2_matrix(self, params, batch_size):
        """Converts parameters to matrix representation."""
        pass

    @staticmethod
    def add_bottom_row(matrix, batch_size):
        """Adds bottom row [0 ... 0 1] to make homogeneous matrix."""
        # matrix = torch.cat([matrix, t], dim=2) + self.identity.unsqueeze(0)
        bottom_row = torch.zeros(batch_size, 1, matrix.size(2), device=matrix.device)
        bottom_row[:, 0, -1] = 1.0
        return torch.cat([matrix, bottom_row], dim=1)

    def _compose_into(self, params, scale_matrix, translation_vector):
        """Compose this transformation into existing matrix components.
        called during composition to update the cumulative transformation matrix.
        Each module can implement its own composition logic.
        Args:
            params: Parameters for this transformation [B, n_params]
            scale_matrix: Current scale matrix [B, 2, 2]
            translation_vector: Current translation vector [B, 2, 1]
        Returns:
            tuple: (new_scale_matrix [B, 2, 2], new_translation_vector [B, 2, 1])
        """
        this_scale, this_translation = self._to_matrix_components(params)
        # Default: matrix multiplication
        new_scale = torch.bmm(scale_matrix, this_scale)
        new_translation = torch.bmm(scale_matrix, this_translation) + translation_vector
        return new_scale, new_translation

    def _to_matrix_components(self, params):
        """Convert parameters to matrix components (scale and translation).
        Args:
            params: Transformation parameters [B, n_params]
        Returns:
            tuple: (scale_matrix [B, 2, 2], translation_vector [B, 2, 1])
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _to_matrix_components() "
            "or override _compose_into()"
        )


########################
#   Standard Modules   #
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

    def param_2_matrix(self, params, batch_size):
        return torch.zeros(batch_size, 0, device=params.device)

#########################
#     Color Modules     #
#########################

class ColorModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        color_ch = kwargs.get("color_channels", 3)
        self.clamp_func = get_clamp_func(kwargs.get("use_clamp", False))
        self.tensor = torch.eye(color_ch, color_ch)
        self.out_channels = color_ch * 2
        self.init(in_channels, **kwargs)

    def _transform(self, x, beta, inverse=False):
        if inverse:
            return x

        mask = None
        if x.size(1) == 2 or x.size(1) > 3:
            x, mask = torch.split(
                x, [self.color_ch, x.size(1) - self.color_ch], dim=1
            )

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

    def get_id_matrix(self, batch_size, device):
        """Returns [B, 4, 4] homogeneous color matrix: [[I_3x3 | 0], [0 0 0 1]]."""
        identity = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        return identity

    def param_2_matrix(self, params, batch_size):
        """Converts to [B, 4, 4]: [[weight | bias], [0 0 0 1]]."""
        weight, bias = torch.split(params.view(batch_size, self.color_ch, 2), [1, 1], dim=2)
        weight_matrix = weight.expand(-1, -1, self.color_ch) * self.identity + self.identity
        top = torch.cat([weight_matrix, bias], dim=2)
        return self.add_bottom_row(top, batch_size)

class LinearColorModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        color_ch = 3
        kwargs["color_channels"] = color_ch
        self.tensor = torch.eye(color_ch, color_ch)
        self.out_channels = color_ch
        self.clamp_func = get_clamp_func(kwargs.get("use_clamp", False))
        self.init(in_channels, **kwargs)

    def _transform(self, x, beta, inverse=False):
        if inverse:
            return x

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

    def get_id_matrix(self, batch_size, device):
        """Returns [B, 4, 4] homogeneous diagonal matrix."""
        return torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    def param_2_matrix(self, params, batch_size):
        """Converts to [B, 4, 4]: [[diag(w) | 0], [0 0 0 1]]."""
        weights = params.view(batch_size, self.color_ch)
        diag = torch.diag_embed(weights) + self.identity
        top = torch.cat([diag, torch.zeros(batch_size, 3, 1, device=params.device)], dim=2)
        return self.add_bottom_row(top, batch_size)

########################
#    Spatial Modules   #
########################

class AffineModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.freeze_frg = kwargs.get("freeze_frg", False)
        self.out_channels = self.get_out_channels()
        self.init(in_channels, **kwargs)

    @staticmethod
    def get_out_channels(**kwargs):
        return 6

    def _transform(self, x, beta, inverse=False):
        beta = beta.view(-1, 2, 3) + self.identity
        size_1 = 1 if (self.freeze_frg or (inverse and x.shape[1] == 4)) else x.size(1)

        if inverse:
            row = torch.tensor(
                [[[0, 0, 1]]] * x.size(0), dtype=torch.float, device=beta.device
            )
            beta = torch.cat([beta, row], dim=1)
            beta = torch.inverse(beta)[:, :2, :]

        grid = F.affine_grid(
            beta,
            (x.size(0), size_1, self.img_size[0], self.img_size[1]),
            align_corners=False,
        )

        if inverse and x.shape[1] == 4:
            out = F.grid_sample(
                x[:, -1, :, :].unsqueeze(1),
                grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=False,
            )
            return torch.cat([x[:, :3, ...], out], dim=1)

        if self.freeze_frg:
            out = F.grid_sample(
                x[:, -1, ...].unsqueeze(1),
                grid,
                mode="bilinear",
                padding_mode=self.padding_mode,
                align_corners=False,
            )
            return torch.cat([x[:, : x.size(1) - 1, ...], out], dim=1)

        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        )

    def param_2_matrix(self, params, batch_size):
        """Converts to [B, 3, 3]: [[A | t], [0 0 1]]."""
        matrix_2x3 = params.view(batch_size, 2, 3) + self.identity
        return self.add_bottom_row(matrix_2x3, batch_size)


class TranslationModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = self.get_out_channels()
        self.init(in_channels, **kwargs)

    @staticmethod
    def get_out_channels(**kwargs):
        return 2

    def _to_matrix_components(self, params):
        """Convert translation parameters to matrix components."""
        batch_size = params.shape[0]
        scale = torch.eye(2, 2, device=params.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        translation = params.unsqueeze(2)
        return scale, translation

    def _compose_into(self, params, scale_matrix, translation_vector):
        """Compose translation by adding to translation vector."""
        this_translation = params.unsqueeze(2)
        new_translation = translation_vector + this_translation
        return scale_matrix, new_translation

    def _transform(self, x, beta, inverse=False):
        if inverse:
            print_warning("Inverse transform for TranslationModule is not implemented.")
            return x

        scale, translation = self._to_matrix_components(beta)
        beta_matrix = torch.cat([scale, translation], dim=2) + self.identity

        grid = F.affine_grid(
            beta_matrix,
            (x.size(0), x.size(1), self.img_size[0], self.img_size[1]),
            align_corners=False,
        )
        return F.grid_sample(
            x, grid, mode="bilinear",
            padding_mode=self.padding_mode, align_corners=False
        )

    def param_2_matrix(self, params, batch_size):
        scale, translation = self._to_matrix_components(params)
        matrix_2x3 = torch.cat([scale, translation], dim=2) + self.identity.unsqueeze(0)
        return self.add_bottom_row(matrix_2x3, batch_size)


class ScalingModule(_AbstractTransformationModule):
    def __init__(self, in_channels, isotropic=True, **kwargs):
        super().__init__()
        self.isotropic = isotropic
        self.out_channels = self.get_out_channels(isotropic)
        self.init(in_channels, **kwargs)

    @staticmethod
    def get_out_channels(isotropic=True):
        return 1 if isotropic else 2

    def _compute_scale_matrix(self, params):
        """Compute scale matrix from parameters."""
        if self.isotropic:
            s = torch.exp(params)
            return s.unsqueeze(-1) * torch.eye(2, 2, device=params.device).unsqueeze(0)
        sx, sy = params.split([1, 1], dim=1)
        sx, sy = torch.exp(sx), torch.exp(sy)
        batch_size = params.shape[0]
        scale = torch.zeros(batch_size, 2, 2, device=params.device)
        scale[:, 0, 0] = sx.squeeze(1)
        scale[:, 1, 1] = sy.squeeze(1)
        return scale

    def _to_matrix_components(self, params):
        """Convert scale parameters to matrix components."""
        scale = self._compute_scale_matrix(params)
        translation = torch.zeros(params.shape[0], 2, 1, device=params.device)
        return scale, translation

    def _compose_into(self, params, scale_matrix, translation_vector):
        """Compose scaling by multiplying scale matrices."""
        this_scale = self._compute_scale_matrix(params)
        new_scale = torch.bmm(scale_matrix, this_scale)
        return new_scale, translation_vector

    def _transform(self, x, beta, inverse=False):
        scale, translation = self._to_matrix_components(beta)
        beta_matrix = torch.cat([scale, translation], dim=2) + self.identity

        grid = F.affine_grid(
            beta_matrix,
            (x.size(0), x.size(1), self.img_size[0], self.img_size[1]),
            align_corners=False,
        )
        return F.grid_sample(
            x, grid, mode="bilinear",
            padding_mode=self.padding_mode, align_corners=False
        )

    def param_2_matrix(self, params, batch_size):
        scale, translation = self._to_matrix_components(params)
        matrix_2x3 = torch.cat([scale, translation], dim=2) + self.identity.unsqueeze(0)
        return self.add_bottom_row(matrix_2x3, batch_size)


class IsotropicScalingModule(ScalingModule):
    isotropic = True
    def __init__(self, in_channels, **kwargs):
        super().__init__(in_channels, isotropic=True, **kwargs)


class AnisotropicScalingModule(ScalingModule):
    isotropic = False
    def __init__(self, in_channels, **kwargs):
        super().__init__(in_channels, isotropic=False, **kwargs)


# class PositionModule(_AbstractTransformationModule):
#     # NOTE same as IsotropicTranslationModule, to delete?
#     def __init__(self, in_channels, **kwargs):
#         super().__init__()
#         self.out_channels = self.get_out_channels()
#         self.init(in_channels, **kwargs)
#
#     @staticmethod
#     def get_out_channels(**kwargs):
#         return 3
#
#     def _transform(self, x, beta, inverse=False):
#         if inverse:
#             print_warning("Inverse transform for PositionModule is not implemented.")
#             return x
#         s, t = beta.split([1, 2], dim=1)
#         s = torch.exp(s)
#         scale = s[..., None].expand(-1, 2, 2) * torch.eye(2, 2).to(s.device)
#         beta = torch.cat([scale, t.unsqueeze(2)], dim=2) + self.identity
#         grid = F.affine_grid(
#             beta,
#             (x.size(0), x.size(1), self.img_size[0], self.img_size[1]),
#             align_corners=False,
#         )
#         out = F.grid_sample(
#             x,
#             grid,
#             mode="bilinear",
#             padding_mode="zeros",
#             align_corners=False,
#         )
#         return out
#
#     def param_2_matrix(self, params, batch_size):
#         s, t = params.split([1, 2], dim=1)
#         s = torch.exp(s)
#         scale = s.unsqueeze(-1) * torch.eye(2, 2, device=params.device).unsqueeze(0)
#         matrix_2x3 = torch.cat([scale, t.unsqueeze(2)], dim=2) + self.identity.unsqueeze(0)
#         return self.add_bottom_row(matrix_2x3, batch_size)

class RotationModule(_AbstractTransformationModule):
    def __init__(self, in_channels, exact_rotation=True, **kwargs):
        super().__init__()
        self.exact_rotation = exact_rotation
        self.layer_size = kwargs.get("layer_size")
        self.init(in_channels, **kwargs)

    def _compute_rotation_matrix(self, params):
        if self.exact_rotation:
            theta = params.squeeze(1)
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            row1 = torch.stack([cos_theta, -sin_theta], dim=1)
            row2 = torch.stack([sin_theta, cos_theta], dim=1)
            return torch.stack([row1, row2], dim=1)

        # First-order approximation (legacy)
        b_eye = torch.tensor([[0, -1], [1, 0]], device=params.device, dtype=params.dtype)
        return params.squeeze(1).unsqueeze(-1).unsqueeze(-1) * b_eye.unsqueeze(0)

    def _to_matrix_components(self, params):
        rotation = self._compute_rotation_matrix(params)
        translation = torch.zeros(params.shape[0], 2, 1, device=params.device)
        return rotation, translation

    def _compose_into(self, params, scale_matrix, translation_vector):
        rotation = self._compute_rotation_matrix(params)
        new_scale = torch.bmm(scale_matrix, rotation)
        return new_scale, translation_vector

    def _transform(self, x, beta, inverse=False):
        if inverse:
            print_warning("Inverse transform for RotationModule is not implemented.")
            return x

        rotation, translation = self._to_matrix_components(beta)
        if self.exact_rotation:
            beta_matrix = torch.cat([rotation, translation], dim=2) + self.identity
        else:
            beta_matrix = torch.cat([rotation, translation], dim=2) + torch.eye(2, 3, device=beta.device).unsqueeze(0)

        grid_size = self.img_size if hasattr(self, 'img_size') and self.img_size is not None else self.layer_size
        grid = F.affine_grid(
            beta_matrix,
            (x.size(0), x.size(1), grid_size[0], grid_size[1]),
            align_corners=False,
        )
        return F.grid_sample(
            x, grid, mode="bilinear",
            padding_mode=self.padding_mode, align_corners=False
        )

    def param_2_matrix(self, params, batch_size):
        rotation, translation = self._to_matrix_components(params)
        id_tensor = self.identity.unsqueeze(0) if self.use_exact_rotation else torch.eye(2, 3, device=params.device).unsqueeze(0)
        matrix_2x3 = torch.cat([rotation, translation], dim=2) + id_tensor
        return self.add_bottom_row(matrix_2x3, batch_size)


class SimilarityModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = self.get_out_channels()
        self.init(in_channels, **kwargs)

    @staticmethod
    def get_out_channels(**kwargs):
        return 4

    def _transform(self, x, beta, inverse=False):
        a, b, t = beta.split([1, 1, 2], dim=1)
        a_eye = torch.eye(2, 2).to(a.device)
        b_eye = torch.Tensor([[0, -1], [1, 0]]).to(b.device)
        scaled_rot = a[..., None].expand(-1, 2, 2) * a_eye + b[..., None].expand(-1, 2, 2) * b_eye
        beta = torch.cat([scaled_rot, t.unsqueeze(2)], dim=2) + self.identity
        grid = F.affine_grid(beta, (x.size(0), x.size(1), self.img_size[0], self.img_size[1]), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode=self.padding_mode, align_corners=False)

    def param_2_matrix(self, params, batch_size):
        a, b, t = params.split([1, 1, 2], dim=1)
        a_eye = torch.eye(2, 2, device=params.device)
        b_eye = torch.tensor([[0, -1], [1, 0]], device=params.device, dtype=params.dtype)
        scaled_rot = a.unsqueeze(-1) * a_eye + b.unsqueeze(-1) * b_eye
        matrix_2x3 = torch.cat([scaled_rot, t.unsqueeze(2)], dim=2) + self.identity.unsqueeze(0)
        return self.add_bottom_row(matrix_2x3, batch_size)

class ProjectiveModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.out_channels = self.get_out_channels()
        self.tensor = torch.eye(3, 3)
        self.init(in_channels, **kwargs)

    @staticmethod
    def get_out_channels(**kwargs):
        return 9

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

    def param_2_matrix(self, params, batch_size):
        """Converts projective parameters to 3x3 matrix [B, 3, 3]."""
        return params.view(batch_size, 3, 3) + self.identity


class TPSModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.freeze_frg = kwargs.get("freeze_frg", False)
        self.grid_size = kwargs.get("grid_size", 4)
        self.out_channels = self.grid_size ** 2 * 2
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.grid_size), torch.linspace(-1, 1, self.grid_size)
        )
        self.tensor = torch.stack([x.flatten(), y.flatten()], dim=1)
        self.init(in_channels, **kwargs)
        self.tps_grid = TPSGrid(self.img_size, self.tensor)

    def _transform(self, x, beta, inverse=False):
        if inverse:
            print_warning("Inverse transform for TPSModule is not implemented.")
            return x
        source_control_points = self.identity + beta.view(x.size(0), -1, 2)
        grid = self.tps_grid(source_control_points).view(x.size(0), *self.img_size, 2)
        grid_x = x[:, -1, ...].unsqueeze(1) if self.freeze_frg else x

        out = F.grid_sample(
            grid_x,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=False,
        )
        if self.freeze_frg:
            return torch.cat([x[:, : x.size(1) - 1, ...], out], dim=1)
        return out

    def param_2_matrix(self, params, batch_size):
        """Get control points from parameters for a specific module instance."""
        # Return control points [B, grid_size^2, 2]
        return self.identity + params.view(batch_size, -1, 2)


###########################
#  Morphological Modules  #
###########################

class MorphologicalModule(_AbstractTransformationModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.freeze_frg = kwargs.get("freeze_frg", False)
        self.kernel_size = kwargs.get("kernel_size", 3)
        assert isinstance(self.kernel_size, (int, float))
        self.padding = self.kernel_size // 2
        self.out_channels = self.kernel_size ** 2 + 1
        weights = torch.full((self.kernel_size, self.kernel_size), fill_value=-5, dtype=torch.float)
        center = self.kernel_size // 2
        weights[center, center] = 5
        self.tensor = torch.cat([torch.zeros(1), weights.flatten()])
        self.init(in_channels, **kwargs)

    def _transform(self, x, beta, inverse=False):
        if inverse:
            print_warning("Inverse transform for MorphologicalModule is not implemented.")
            return x
        beta = beta + self.identity
        alpha, weights = torch.split(beta, [1, self.kernel_size ** 2], dim=1)

        if self.freeze_frg:
            out = self.smoothmax_kernel(x[:, -1, ...].unsqueeze(1), alpha, torch.sigmoid(weights))
            return torch.cat([x[:, : x.size(1) - 1, ...], out], dim=1)

        if x.shape[1] > 1:
            # Multi-channel input
            morph_channel = x[:, 0:1, ...]  # Use first channel
            transformed = self.smoothmax_kernel(morph_channel, alpha, torch.sigmoid(weights))
            return torch.cat([transformed, x[:, 1:, ...]], dim=1)

        return self.smoothmax_kernel(x, alpha, torch.sigmoid(weights))

    def smoothmax_kernel(self, x, alpha, kernel):
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.flatten()[:, None, None]

        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        x_unf = F.unfold(x, self.kernel_size, padding=self.padding).transpose(1, 2)
        w = torch.exp(alpha * x_unf) * kernel.unsqueeze(1).expand(-1, x_unf.size(1), -1)
        return ((x_unf * w).sum(2) / w.sum(2)).view(B, C, H, W)

    def get_id_matrix(self, batch_size, device):
        kernel_identity = self.identity[1:].view(self.kernel_size, self.kernel_size)
        return kernel_identity.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def param_2_matrix(self, params, batch_size):
        """Get kernel from parameters for a specific module instance."""
        # Return morphological kernel [B, kernel_size, kernel_size]
        kernel_params = params[:, 1:] + self.identity[1:]
        return kernel_params.view(batch_size, self.kernel_size, self.kernel_size)


###############################
#      Composite Modules      #
###############################

class ComposedTransformModule(_AbstractTransformationModule, ABC):
    """Generic module that composes multiple transformations.
    This module can compose Affine modules that implement the _compose_into() protocol.
    """

    def __init__(self, in_channels, component_modules, **kwargs):
        """
        in_channels: Number of input feature channels
        component_modules: List of (ModuleClass, kwargs_dict) tuples
            Example: [
                (IsotropicScalingModule, {}),
                (RotationModule, {}),
                (TranslationModule, {})
            ]
        """
        super().__init__()

        self.components = []
        for module_class, module_kwargs in component_modules:
            if not hasattr(module_class, '_compose_into'):
                raise TypeError(f"{module_class.__name__} must implement _compose_into()")

            component = module_class.__new__(module_class)
            component.__dict__.update(module_kwargs)
            component.out_channels = module_class.get_out_channels(**module_kwargs)
            self.components.append(component)

        self.out_channels = sum(comp.out_channels for comp in self.components)
        self.init(in_channels, **kwargs)

    def _build_matrix(self, params):
        """Build composed transformation matrix from parameters.

        This method splits parameters and calls each component's
        _compose_into() method to build the final transformation.
        """
        param_splits = [comp.out_channels for comp in self.components]
        param_list = torch.split(params, param_splits, dim=1)

        # initialize to identity
        batch_size = params.shape[0]
        scale = torch.eye(2, 2, device=params.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        translation = torch.zeros(batch_size, 2, 1, device=params.device)

        for component, param in zip(self.components, param_list):
            scale, translation = component._compose_into(param, scale, translation)

        return torch.cat([scale, translation], dim=2)

    def _transform(self, x, beta, inverse=False):
        """Apply composed transformation."""
        beta_matrix = self._build_matrix(beta) + self.identity.unsqueeze(0)

        grid = F.affine_grid(
            beta_matrix,
            (x.size(0), x.size(1), self.img_size[0], self.img_size[1]),
            align_corners=False,
        )
        return F.grid_sample(
            x, grid, mode="bilinear",
            padding_mode=self.padding_mode, align_corners=False
        )

    def param_2_matrix(self, params, batch_size):
        """Convert parameters to transformation matrix."""
        matrix_2x3 = self._build_matrix(params) + self.identity.unsqueeze(0)
        return self.add_bottom_row(matrix_2x3, batch_size)


class IsotropicTranslationModule(ComposedTransformModule):
    """Isotropic scaling + translation."""

    def __init__(self, in_channels, **kwargs):
        component_modules = [
            # (ScalingModule, {'isotropic': True}),
            (IsotropicScalingModule, {}),
            (TranslationModule, {})
        ]
        super().__init__(in_channels, component_modules, **kwargs)


class AnisotropicTranslationModule(ComposedTransformModule):
    """Anisotropic scaling + translation."""

    def __init__(self, in_channels, **kwargs):
        component_modules = [
            # (ScalingModule, {'isotropic': False}),
            (AnisotropicScalingModule, {}),
            (TranslationModule, {})
        ]
        super().__init__(in_channels, component_modules, **kwargs)


class SimilarityTransformModule(ComposedTransformModule):
    """Isotropic scaling + rotation + translation."""

    def __init__(self, in_channels, **kwargs):
        component_modules = [
            # (ScalingModule, {'isotropic': True}),
            (IsotropicScalingModule, {}),
            (RotationModule, {}),
            (TranslationModule, {})
        ]
        super().__init__(in_channels, component_modules, **kwargs)


class RotationTranslationModule(ComposedTransformModule):
    """Rotation + translation."""

    def __init__(self, in_channels, **kwargs):
        component_modules = [
            (RotationModule, {}),
            (TranslationModule, {})
        ]
        super().__init__(in_channels, component_modules, **kwargs)
