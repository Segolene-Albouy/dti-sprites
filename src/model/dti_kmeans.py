import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from itertools import chain

from .abstract_dti import AbstractDTI
from .transformer import PrototypeTransformationNetwork
from .tools import copy_with_noise, create_gaussian_weights, generate_data
from ..utils.logger import print_warning

from .u_net import UNet

NOISE_SCALE = 0.0001
EMPTY_CLUSTER_THRESHOLD = 0.2
LATENT_SIZE = 128
EPSILON = 1e-6


class DTIKmeans(AbstractDTI):
    name = "dtikmeans"

    def __init__(self, dataset=None, n_prototypes=10, **kwargs):
        super().__init__()
        if dataset is None:
            raise NotImplementedError
        self.n_prototypes = n_prototypes
        self.img_size = dataset.img_size
        # Prototypes
        self.color_channels = kwargs.get("color_channels", 3)
        assert kwargs.get("prototype")
        proto_args = kwargs.get("prototype")
        proto_source = proto_args.get("source", "data")
        assert proto_source in ["data", "generator"]
        self.proto_source = proto_source
        if proto_source == "data":
            data_args = proto_args.get("data")
            init_type = data_args.get("init", "sample")
            std = data_args.get("gaussian_weights_std", 25)
            self.prototype_params = nn.Parameter(
                torch.stack(generate_data(dataset, n_prototypes, init_type, std=std))
            )
        else:
            gen_name = proto_args.get("generator", "mlp")
            print_warning("Sprites will be generated from latent variables.")
            assert gen_name in ["mlp", "unet"]
            latent_dims = (
                (LATENT_SIZE,)
                if gen_name == "mlp"
                else (1, self.img_size[0], self.img_size[1])
            )
            self.latent_params = nn.Parameter(
                torch.stack(
                    [
                        torch.normal(mean=0.0, std=1.0, size=latent_dims)
                        for k in range(n_prototypes)
                    ],
                    dim=0,
                )
            )
            self.generator = self.init_generator(
                gen_name,
                self.color_channels,
                self.color_channels * self.img_size[0] * self.img_size[1],
            )

        self.transformer = PrototypeTransformationNetwork(
            dataset.n_channels, dataset.img_size, n_prototypes, **kwargs
        )
        self.empty_cluster_threshold = kwargs.get(
            "empty_cluster_threshold", EMPTY_CLUSTER_THRESHOLD / n_prototypes
        )
        self._reassign_cluster = kwargs.get("reassign_cluster", True)

        proba = kwargs.get("proba", False)
        if proba:
            # proba regularization loss weight
            self.freq_weight = kwargs.get("freq_weight", 0)
            self.bin_weight = kwargs.get("bin_weight", 0)
            self.beta_dist = torch.distributions.Beta(
                torch.Tensor([2.0]).to("cuda"), torch.Tensor([2.0]).to("cuda")
            )
            # what to weight with probabilities
            self.weighting = kwargs.get("proba_weighting", "tr_sprite")
            if self.weighting == "sprite":
                assert kwargs.get("shared_t", False)

            # how to generate probabilities
            self.proba_type = kwargs.get("proba_type", "marionette")
            self.out_ch = self.transformer.enc_out_channels
            if self.proba_type == "linear":  # linear mapping
                self.proba = nn.Linear(self.out_ch, n_prototypes)
            else:  # marionette-like
                self.proba = nn.Sequential(
                    nn.Linear(self.out_ch, LATENT_SIZE),
                    nn.LayerNorm(LATENT_SIZE, elementwise_affine=False),
                )
                self.shared_l = kwargs.get("shared_l", True)
                if not self.shared_l:
                    self.proba_latent_params = nn.Parameter(
                        torch.stack(
                            [
                                torch.normal(mean=0.0, std=1.0, size=latent_dims)
                                for k in range(n_prototypes)
                            ],
                            dim=0,
                        )
                    )

        use_gaussian_weights = kwargs.get("gaussian_weights", False)
        if use_gaussian_weights:
            std = kwargs.get("gaussian_weights_std")
            self.register_buffer(
                "loss_weights",
                create_gaussian_weights(dataset.img_size, dataset.n_channels, std),
            )
        else:
            self.loss_weights = None

    def cluster_parameters(self):
        out = []
        if hasattr(self, "generator"):
            out += list(chain(*[self.generator.parameters()])) + [self.latent_params]
        else:
            out += [self.prototype_params]
        if hasattr(self, "proba"):
            out += list(chain(*[self.proba.parameters()]))
            if hasattr(self, "shared_l") and (not self.shared_l):
                out += [self.proba_latent_params]
        return out

    def transformer_parameters(self):
        return self.transformer.parameters()

    @staticmethod
    def init_generator(name, color_channel, out_channel, latent_dim=LATENT_SIZE):
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
            raise NotImplementedError("Generator not implemented.")

    @property
    def prototypes(self):
        if self.proto_source == "generator":
            with torch.no_grad():
                params = self.generator(self.latent_params)
                if len(params.size()) != 4:
                    params = params.reshape(
                        -1, self.color_channels, self.img_size[0], self.img_size[1]
                    )
                return params
        return self.prototype_params

    def estimate_logits(self, features):
        if self.proba_type == "marionette":
            latent_params = self.proba_latent_params if not self.shared_l else self.latent_params
            latent_params = torch.nn.functional.layer_norm(latent_params, (latent_params.shape[-1],))
            proba_theta = self.proba(features).permute(1,0) # DB
            # D, B = proba_theta.shape
            temp = torch.matmul(latent_params, proba_theta).permute(1,0)
            logits = (1.0 / np.sqrt(self.out_ch)) * (temp) # BK
            return logits
        elif self.proba_type == "linear":
            logits = self.proba(features)
            return logits
        else:
            raise ValueError(f"Unknown proba type: {self.proba_type}")

    def reg_func(self, probas, type="freq"):
        if type == "freq":
            freqs = probas.mean(dim=0)
            freqs = freqs / freqs.sum()
            return freqs.clamp(max=(self.empty_cluster_threshold))
        elif type == "bin":
            p = probas.clamp(min=EPSILON, max=1 - EPSILON)
            dist = self.beta_dist.log_prob(p)
            return torch.exp(dist)
        else:
            raise ValueError(f"undefined regularizer: {type}")

    def forward(self, x, img_masks=None):
        if self.proto_source == "generator":
            params = self.generator(self.latent_params)
            if len(params.size()) != 4:
                params = params.reshape(
                    -1, self.color_channels, self.img_size[0], self.img_size[1]
                )
        else:
            params = self.prototype_params

        prototypes = params.unsqueeze(1).expand(-1, x.size(0), x.size(1), -1, -1)

        if hasattr(self, "proba"):
            inp, target, features = self.transformer(x, prototypes)
            logits = self.estimate_logits(features)

            if self.training:
                probas = F.gumbel_softmax(logits, dim=-1)
            else:
                probas = F.softmax(logits, dim=-1)

            freq_loss = self.reg_func(probas, type="freq")

            if self.weighting == "tr_sprite":
                # Weight transformed sprites and sum
                weighted_target = (probas[..., None, None, None] * target).sum(1)
                distances = self.criterion(inp[:, 0, ...], weighted_target, alpha_masks=img_masks)

                # Distances of input from transformed sprites
                samplewise_distances = self.criterion(inp, target, alpha_masks=img_masks)

                bin_loss = self.reg_func(probas, type="bin")
                a = distances.mean()
                b = self.freq_weight * (1 - freq_loss.sum())
                c = self.bin_weight * (bin_loss.mean())

                return (
                    a + b + c,
                    samplewise_distances,
                    probas,
                )

            elif self.weighting == "diff":
                distances = self.criterion(inp, target, alpha_masks=img_masks)
                dist_min = (distances * probas).sum(1)

                return (
                    dist_min.mean() + self.freq_weight * (1 - freq_loss.sum()),
                    distances,
                    probas,
                )
            raise ValueError(f"Probability weighting is not implemented: {self.weighting}")

        # no proba
        inp, target, features = self.transformer(x, prototypes)
        distances = self.criterion(inp, target, alpha_masks=img_masks)
        dist_min = distances.min(1)[0]
        return dist_min.mean(), distances, None

    @torch.no_grad()
    def transform(self, x, inverse=False):
        if inverse:
            return self.transformer.inverse_transform(x)
        prototypes = self.prototypes.unsqueeze(1).expand(
            -1, x.size(0), x.size(1), -1, -1
        )
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
                state[name].copy_(param)
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
        # to add noise to reassigned class with max number of samples
        if len(reassigned) > 0:
            self.restart_branch_from(idx, idx)
        return reassigned, idx

    def restart_branch_from(self, i, j):
        if hasattr(self, "generator"):
            self.latent_params[i].data.copy_(
                copy_with_noise(self.latent_params[j], NOISE_SCALE)
            )
            param = self.latent_params
        else:
            self.prototype_params[i].data.copy_(
                copy_with_noise(self.prototype_params[j], NOISE_SCALE)
            )
            param = self.prototype_params

        if hasattr(self, "proba"):
            if self.proba_type == "linear":
                self.proba.weight[i].data.copy_(
                    copy_with_noise(self.proba.weight[j], noise_scale=0)
                )
                self.proba.bias[i].data.copy_(
                    copy_with_noise(self.proba.bias[j], noise_scale=0)
                )
                proba_params = [self.proba.weight, self.proba.bias]
            elif not self.shared_l:
                self.proba_latent_params[i].data.copy_(
                    copy_with_noise(self.proba_latent_params[j], NOISE_SCALE)
                )
                proba_params = [self.proba_latent_params]
            else:
                proba_params = []
        self.transformer.restart_branch_from(i, j, noise_scale=0)

        if hasattr(self, "optimizer"):
            opt = self.optimizer
            if isinstance(opt, (Adam,)):
                opt.state[param]["exp_avg"][i] = opt.state[param]["exp_avg"][j]
                opt.state[param]["exp_avg_sq"][i] = opt.state[param]["exp_avg_sq"][j]
                if hasattr(self, "proba"):
                    for p_ in proba_params:
                        opt.state[p_]["exp_avg"][i] = opt.state[p_]["exp_avg"][j]
                        opt.state[p_]["exp_avg_sq"][i] = opt.state[p_]["exp_avg_sq"][j]
            else:
                raise NotImplementedError(
                    "unknown optimizer: you should define how to reinstanciate statistics if any"
                )
