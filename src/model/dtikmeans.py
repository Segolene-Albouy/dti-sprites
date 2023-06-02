from itertools import chain
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import numpy as np

from .transformer import (
    PrototypeTransformationNetwork as Transformer,
    N_HIDDEN_UNITS,
    N_LAYERS,
)
from .tools import (
    copy_with_noise,
    create_gaussian_weights,
    generate_data,
    get_clamp_func,
)
from utils.logger import print_warning
from .u_net import UNet

NOISE_SCALE = 0.0001
EMPTY_CLUSTER_THRESHOLD = 0.2


class DTIKmeans(nn.Module):
    name = "dtikmeans"

    def __init__(self, dataset=None, n_prototypes=10, **kwargs):
        super().__init__()
        if dataset is None:
            raise NotImplementedError
        self.n_prototypes = n_prototypes
        self.n_objects = 1
        self.img_size = dataset.img_size
        noise = kwargs.get("noise", 0.5)
        value = kwargs.get("value", None)
        self.ms = kwargs.get("ms", 0)
        proto_args = kwargs.get("prototype")
        self.proto_source = proto_args.get("source", "data")
        if self.proto_source == "generator":
            gen_name = proto_args.get("generator", "unet")
            latent_size = (
                (128,)
                if gen_name == "marionette"
                else (1, self.img_size[0], self.img_size[1])
            )
            self.generator = self.init_generator(
                gen_name,
                latent_dim=128,
                out_channel=self.img_size[0] * self.img_size[1],
            )
            assert proto_args.get("proba")
            self.latent_params = (
                nn.Parameter(  # TODO: Check during optimization when shared.
                    torch.stack(
                        [
                            torch.normal(mean=0.0, std=1.0, size=latent_size)
                            for k in range(n_prototypes)
                        ]
                    )
                )
            )
            self.latent_shared = proto_args.get("shared_latent", False)
            if self.latent_shared:
                self.proba_latent_params = self.latent_params
            else:
                self.proba_latent_params = nn.Parameter(
                    torch.stack(
                        [
                            torch.normal(mean=0.0, std=1.0, size=latent_size)
                            for k in range(n_prototypes)
                        ]
                    )
                )
            clamp_name = kwargs.get("use_clamp", "soft")
            self.clamp_func = get_clamp_func(clamp_name)

        else:
            data_args = proto_args.get("data")
            proto_init = data_args.get("init", "sample")
            std = data_args.get("gaussian_weights_std", 25)

            samples = torch.stack(
                generate_data(
                    dataset,
                    n_prototypes,
                    proto_init,
                    std=std,
                    size=self.img_size,
                    value=0.9,
                )
            )
            self.prototype_params = nn.Parameter(samples)
            clamp_name = kwargs.get("use_clamp", "soft")
            self.clamp_func = get_clamp_func(clamp_name)

        self.transformer = Transformer(
            dataset.n_channels, dataset.img_size, n_prototypes, **kwargs
        )
        self.encoder = self.transformer.encoder
        if proto_args.get("proba"):
            self.projector = self.init_projector(self.encoder.out_ch)
            self.proba_estimator = self.init_proba(self.proba_latent_params.shape[1])

        self.empty_cluster_threshold = kwargs.get(
            "empty_cluster_threshold", EMPTY_CLUSTER_THRESHOLD / n_prototypes
        )
        self._reassign_cluster = kwargs.get("reassign_cluster", True)
        use_gaussian_weights = kwargs.get("gaussian_weights", False)
        if use_gaussian_weights:
            std = kwargs["gaussian_weights_std"]
            self.register_buffer(
                "loss_weights",
                create_gaussian_weights(dataset.img_size, dataset.n_channels, std),
            )
        else:
            self.loss_weights = None
        self.scale_t = kwargs.get("scale", "minmax")

    @staticmethod
    def init_generator(name, latent_dim, out_channel):
        if name == "unet":
            return UNet(1, 1)
        elif name == "marionette":
            return nn.Sequential(
                nn.Linear(latent_dim, 8 * latent_dim),
                nn.GroupNorm(8, 8 * latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(8 * latent_dim, out_channel),
                nn.Sigmoid(),
            )
        else:
            NotImplementedError("Generator not implemented.")

    @staticmethod
    def init_proba(in_channel):
        return nn.Sequential(
            nn.Linear(in_channel, N_HIDDEN_UNITS),
            nn.LayerNorm(N_HIDDEN_UNITS),
        )

    @staticmethod
    def init_projector(in_channel):
        return nn.Sequential(
            nn.Linear(in_channel, N_HIDDEN_UNITS),
            nn.LayerNorm(N_HIDDEN_UNITS),
        )

    @torch.no_grad()
    def proto_scale(self):
        if self.scale_t == "minmax":
            b, c, h, w = self.prototypes.shape
            x = self.prototypes.view(b, c, h * w)
            min_ = x.min(-1, keepdim=True)[0]
            max_ = x.max(-1, keepdim=True)[0]

            x = (x - min_) / (max_ - min_)
            return x.view(b, c, h, w)
        else:
            return NotImplementedError("Scale type is not implemented.")

    @property
    def prototypes(self):
        if self.proto_source == "generator":
            gen_theta = self.generator(self.latent_params)
            if self.gen_name == "marionette":
                params = gen_theta.reshape(-1, 1, self.img_size[0], self.img_size[1])
        else:
            params = self.prototype_params

        return self.clamp_func(params)

    def cluster_parameters(self):
        params = []
        if hasattr(self, "generator"):
            params += list(chain(*[self.generator.parameters()])) + [self.latent_params]
        else:
            params += [self.prototype_params]
        if hasattr(self, "proba_estimator"):
            params += list(chain(*[self.proba_estimator.parameters()])) + list(
                chain(*[self.projector.parameters()])
            )
            if not self.latent_shared:
                params += [self.proba_latent_params]

        return iter(params)

    def transformer_parameters(self):
        return self.transformer.parameters()

    def forward(self, x, epoch=None):
        B, _, _, _ = x.size()
        if epoch == None or epoch >= self.ms:
            self.prototypes.data.copy_(self.proto_scale())

        features = self.encoder(x)

        if hasattr(self, "proba_estimator"):
            proba_theta = self.proba_estimator(self.proba_latent_params)
            probas = (1.0 / np.sqrt(self.encoder.out_ch)) * torch.matmul(
                proba_theta, self.projector(features).T
            )

            prototypes = self.prototypes.unsqueeze(1).expand(
                -1, probas.shape[1], -1, -1, -1
            ) * probas.reshape(probas.shape[0], probas.shape[1], 1, 1, 1)
        else:
            prototypes = self.prototypes.unsqueeze(1).expand(
                self.n_prototypes, B, -1, -1, -1
            )

        inp, target = self.transformer(x, prototypes)
        distances = (inp - target) ** 2
        if self.loss_weights is not None:
            distances = distances * self.loss_weights
        distances = distances.flatten(2).mean(2)
        dist_min = distances.min(1)[0]
        return dist_min.mean(), distances

    @torch.no_grad()
    def transform(self, x, inverse=False):
        if inverse:
            return self.transformer.inverse_transform(x)
        else:
            prototypes = self.prototypes.unsqueeze(1).expand(-1, x.size(0), -1, -1, -1)
            return self.transformer(x, prototypes)[1]

    def step(self):
        self.transformer.step()

    def set_optimizer(self, opt):
        self.optimizer = opt
        self.transformer.set_optimizer(opt)

    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in state_dict.items():
            if name in state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                if "activations" in name and state[name].shape != param.shape:
                    state[name].copy_(
                        torch.Tensor([True] * state[name].size(0)).to(param.device)
                    )
                else:
                    state[name].copy_(param)
            elif name == "prototypes":
                # TODO: Check if prototype_params is already copied.
                if not hasattr(self, "generator"):
                    state["prototype_params"].copy_(param)
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f"load_state_dict: {unloaded_params} not found")

    def reassign_empty_clusters(self, proportions):
        if not self._reassign_cluster:
            return [], 0

        idx = np.argmax(proportions)
        reassigned = []
        for i in range(self.n_prototypes):
            if proportions[i] < self.empty_cluster_threshold:
                self.restart_branch_from(i, idx)
                reassigned.append(i)
        if len(reassigned) > 0:
            self.restart_branch_from(idx, idx)
        return reassigned, idx

    def restart_branch_from(self, i, j):
        if hasattr(self, "generator"):
            self.latent_params[i].data.copy_(self.latent_params[j])
        else:
            self.prototype_params[i].data.copy_(
                copy_with_noise(self.prototypes[j], NOISE_SCALE)
            )
        if hasattr(self, "proba_estimator") and not self.latent_shared:
            self.proba_latent_params[i].data.copy_(self.proba_latent_params[j])
        self.transformer.restart_branch_from(i, j, noise_scale=0)

        if hasattr(self, "optimizer"):
            opt = self.optimizer
            params = [self.prototype_params]
            if isinstance(opt, (Adam,)):
                for param in params:
                    opt.state[param]["exp_avg"][i] = opt.state[param]["exp_avg"][j]
                    opt.state[param]["exp_avg_sq"][i] = opt.state[param]["exp_avg_sq"][
                        j
                    ]
            elif isinstance(opt, (RMSprop,)):
                for param in params:
                    opt.state[param]["square_avg"][i] = opt.state[param]["square_avg"][
                        j
                    ]
            else:
                raise NotImplementedError(
                    "unknown optimizer: you should define how to reinstanciate statistics if any"
                )
