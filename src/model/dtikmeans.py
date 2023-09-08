from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, RMSprop, AdamW
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
EPSILON = 0.5


class DTIKmeans(nn.Module):
    name = "dtikmeans"

    def __init__(self, dataset=None, n_prototypes=10, **kwargs):
        super().__init__()
        if dataset is None:
            raise NotImplementedError
        self.n_prototypes = n_prototypes
        self.n_objects = 1
        self.img_size = dataset.img_size
        self.ms = kwargs.get("ms", 0)
        self.color_channels = kwargs.get("color_channels", 3)
        proto_args = kwargs.get("prototype")
        self.proto_source = proto_args.get("source", "data")
        if self.proto_source == "generator":
            self.gen_name = proto_args.get("generator", "unet")
            latent_size = (
                (128,)
                if self.gen_name == "mlp"
                else (1, self.img_size[0], self.img_size[1])  # unet
            )
            self.generator = self.init_generator(
                self.gen_name,
                latent_dim=128,
                color_channel=self.color_channels,
                out_channel=self.color_channels * self.img_size[0] * self.img_size[1],
            )
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

        if proto_args.get("proba", False):
            self.latent_shared = proto_args.get("shared_latent", False)
            if self.latent_shared:
                assert hasattr(self, "latent_params")
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

        self.transformer = Transformer(
            dataset.n_channels, dataset.img_size, n_prototypes, **kwargs
        )
        self.encoder = self.transformer.encoder

        self.proba_type = None
        if proto_args.get("proba", False):
            self.proba_type = proto_args.get("proba_type", "")
            self.lambda_freq = proto_args.get("lambda_freq", 0.1)
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

    @staticmethod
    def init_generator(name, latent_dim, color_channel, out_channel):
        if name == "unet":
            return UNet(1, color_channel)
        elif name == "mlp":
            return nn.Sequential(
                nn.Linear(latent_dim, 8 * latent_dim),
                nn.GroupNorm(8, 8 * latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(8 * latent_dim, out_channel),
                nn.Sigmoid(),
            )
        else:
            NotImplementedError("Generator not implemented.")

    def init_proba(self, in_channel):
        return nn.Sequential(
            nn.Linear(in_channel, self.encoder.out_ch),
            nn.LayerNorm(self.encoder.out_ch, elementwise_affine=False),
        )

    @property
    def prototypes(self):
        if self.proto_source == "generator":
            params = self.generator(self.latent_params)
            if self.gen_name == "mlp":
                params = params.reshape(
                    -1, self.color_channels, self.img_size[0], self.img_size[1]
                )
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
            params += list(chain(*[self.proba_estimator.parameters()]))
            if not self.latent_shared:
                print("Latent parameters of decision modules will be updated.")
                params += [self.proba_latent_params]

        return iter(params)

    def transformer_parameters(self):
        return self.transformer.parameters()

    def forward(self, x, epoch=None):
        B, _, _, _ = x.size()
        features = self.encoder(x)

        prototypes = self.prototypes.unsqueeze(1).expand(
            self.n_prototypes, B, -1, -1, -1
        )

        inp, target = self.transformer(x, prototypes)

        if hasattr(self, "proba_estimator"):
            proba_theta = self.proba_estimator(self.proba_latent_params)
            logits = (1.0 / np.sqrt(self.encoder.out_ch)) * (features @ proba_theta.T)
            probas = F.softmax(logits, dim=-1)
            if self.proba_type == "weight_sprite":
                distances = (
                    inp[:, 0, ...] - (probas[..., None, None, None] * target).sum(1)
                ) ** 2
                dist = distances.flatten(1).mean(1)
                freqs = probas.sum(dim=0)
                freqs = freqs / freqs.sum()
                freq_loss = freqs.clamp(max=EPSILON / self.n_prototypes)
                return (
                    dist.mean() + self.lambda_freq * (1 - freq_loss.sum()),
                    dist,
                    probas.max(1)[1],
                )
            elif self.proba_type == "weight_diff":
                distances = (probas[..., None, None, None] * ((inp - target) ** 2)).sum(
                    1
                )
                dist = distances.flatten(1).mean(1)
                freqs = probas.sum(dim=0)
                freqs = freqs / freqs.sum()
                freq_loss = freqs.clamp(max=EPSILON / self.n_prototypes)
                return (
                    dist.mean() + self.lambda_freq * (1 - freq_loss.sum()),
                    dist,
                    probas.max(1)[1],
                )
            else:
                NotImplementedError("Reconstruction loss not implemented.")
        else:
            distances = (inp - target) ** 2
            if self.loss_weights is not None:
                distances = distances * self.loss_weights
            if hasattr(self, "sprite_optimizer"):
                distances = distances.flatten(2).sum(2)
            else:
                distances = distances.flatten(2).mean(2)

            dist = distances.min(1)[0]
            dist_min_by_sample, argmin_idx = distances.min(1)
            return dist.mean(), dist_min_by_sample, argmin_idx

    @torch.no_grad()
    def transform(self, x, inverse=False):
        if inverse:
            return self.transformer.inverse_transform(x)
        else:
            prototypes = self.prototypes.unsqueeze(1).expand(-1, x.size(0), -1, -1, -1)
            return self.transformer(x, prototypes)[1]

    def step(self):
        self.transformer.step()

    def set_optimizer(self, opt, sprite_opt=None):
        self.optimizer = opt
        if sprite_opt:
            self.sprite_optimizer = sprite_opt
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
            self.latent_params[i].data.copy_(self.latent_params[j].detach().clone())
        else:
            self.prototype_params[i].data.copy_(
                copy_with_noise(self.prototypes[j], NOISE_SCALE)
            )
        if hasattr(self, "proba_estimator") and not self.latent_shared:
            self.proba_latent_params[i].data.copy_(
                self.proba_latent_params[j].detach().clone()
            )
        self.transformer.restart_branch_from(i, j, noise_scale=0)

        if hasattr(
            self, "sprite_optimizer"
        ):  # NOTE: This is the case for SGD experiments only.
            pass
        else:
            if hasattr(self, "optimizer"):
                opt = self.optimizer
                params = (
                    [self.latent_params]
                    if hasattr(self, "generator")
                    else [self.prototype_params]
                )
                if hasattr(self, "proba_estimator") and not self.latent_shared:
                    params += [self.proba_latent_params]
                if isinstance(opt, (Adam, AdamW)):
                    for param in params:
                        opt.state[param]["exp_avg"][i] = opt.state[param]["exp_avg"][j]
                        opt.state[param]["exp_avg_sq"][i] = opt.state[param][
                            "exp_avg_sq"
                        ][j]
                elif isinstance(opt, (RMSprop,)):
                    for param in params:
                        opt.state[param]["square_avg"][i] = opt.state[param][
                            "square_avg"
                        ][j]
                else:
                    raise NotImplementedError(
                        "unknown optimizer: you should define how to reinstanciate statistics if any"
                    )
