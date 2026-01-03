import numpy as np
import torch
import torch.nn as nn



def build_field(config):
    layer_dims = [config["field_hidden_size"]] * config["field_hidden_layers"] + [
        (config["backbone_link_dim"] + 1) * (config["depth"] + 2)
    ]
    gps_net = GPSNet(
        pos_enc_dim=512,
        hidden_dim=2048,
        output_dim=config["field_posenc_dim"],
        norm_mode=config["gps_norm_mode"],
        mini=config["gps_mini"],
        maxi=config["gps_maxi"],
        eps=float(config["gps_eps"]),
    )
    return GPSField(gps_net, layer_dims=layer_dims, config=config)


class GPSField(nn.Module):
    def __init__(self, gps_net, layer_dims, config):
        super().__init__()
        self.posenc_dim = config["field_posenc_dim"]
        self.gps_net = gps_net

        # TODO: the first hidden layer's weights should be regularized
        self.pred_mlp = nn.Sequential()
        self.pred_mlp.append(nn.Linear(config["field_posenc_dim"], layer_dims[0]))
        self.pred_mlp.append(nn.GELU())
        for i in range(1, len(layer_dims) - 1):
            self.pred_mlp.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            self.pred_mlp.append(nn.GELU())
        self.pred_mlp.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.r_sqrt_eigva = nn.Buffer(
            torch.rsqrt(
                torch.ones(config["field_posenc_dim"], dtype=torch.float32)
            ),
            # This buffer is trained/serialized (present in checkpoints as
            # "heatmap_head.weight_field.r_sqrt_eigva") so it must be persistent.
            persistent=True,
        )

    def forward(self, inp):
        lbo = self.gps_net(inp.reshape(-1, 3))[..., : self.posenc_dim]
        lbo = torch.reshape(lbo, inp.shape[:-1] + (self.posenc_dim,))
        lbo = lbo * self.r_sqrt_eigva[: self.posenc_dim] * 0.1
        return self.pred_mlp(lbo)


class GPSNet(nn.Module):
    def __init__(
        self,
        pos_enc_dim=512,
        hidden_dim=2048,
        output_dim=1024,
        norm_mode: str = 'dynamic',
        mini=None,
        maxi=None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.factor = 1 / np.sqrt(np.float32(pos_enc_dim))
        if norm_mode not in ('static', 'dynamic'):
            raise ValueError(f'norm_mode must be "static" or "dynamic", got: {norm_mode!r}')
        self.norm_mode = norm_mode
        self.eps = float(eps)

        # Static bounds are optional unless norm_mode="static"
        if mini is None or maxi is None:
            if self.norm_mode == 'static':
                raise ValueError(
                    'GPSNet norm_mode="static" requires `mini` and `maxi` (3 floats each). '
                    'Set them via config: gps_mini/gps_maxi, or use gps_norm_mode="dynamic".'
                )
            mini_t = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
            maxi_t = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        else:
            mini_t = torch.tensor(mini, dtype=torch.float32)
            maxi_t = torch.tensor(maxi, dtype=torch.float32)

        # These buffers are expected in checkpoints under
        # "heatmap_head.weight_field.gps_net.{mini,maxi,center}".
        self.mini = nn.Buffer(mini_t, persistent=True)
        self.maxi = nn.Buffer(maxi_t, persistent=True)
        self.center = nn.Buffer((self.mini + self.maxi) / 2, persistent=True)

        self.learnable_fourier = LearnableFourierFeatures(3, pos_enc_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inp):
        if self.norm_mode == 'dynamic':
            mini = torch.amin(inp, dim=0)
            maxi = torch.amax(inp, dim=0)
            center = (mini + maxi) / 2
            denom = torch.clamp(maxi - mini, min=self.eps)
            x = (inp - center) / denom
        else:
            denom = torch.clamp(self.maxi - self.mini, min=self.eps)
            x = (inp - self.center) / denom
        x = self.learnable_fourier(x) * self.factor
        return self.mlp(x)


class LearnableFourierFeatures(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError('out_features must be even (sin and cos in pairs)')
        self.linear = nn.Linear(in_features, out_features // 2, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=12)

    def forward(self, inp):
        x = self.linear(inp)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
