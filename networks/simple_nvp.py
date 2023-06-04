import torch
from torch import nn

class ProjectionLayer(nn.Module):
    def __init__(self, input_dims=3, proj_dims=128, proj_=True):
        super().__init__()
        self.proj_dims = proj_dims

        self.proj = nn.Sequential(
            nn.Linear(input_dims, 2*proj_dims),
            nn.LeakyReLU(),
            nn.Linear(2*proj_dims, proj_dims)
        )
        self.proj_ = proj_

    def forward(self, x):
        if self.proj_:
            return self.proj(x)
        else:
            return x

class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, projection, mask):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.projection = projection
        self.register_buffer("mask", mask)

    def forward(self, F, y):
        y1 = y * self.mask

        F_y1 = torch.cat([F, self.projection(y1)], dim=-1)
        s = self.map_s(F_y1)
        t = self.map_t(F_y1)

        x = y1 + (1-self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj

    def inverse(self, F, x):
        x1 = x * self.mask
        F_x1 = torch.cat([F, self.projection(x1)], dim=-1)
        s = self.map_s(F_x1)
        t = self.map_t(F_x1)

        y = x1 + (1-self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)
        return y, ldj


class SimpleNVP(nn.Module):
    def __init__(self, n_layers, feature_dims, hidden_size, input_dims=3,
                 normalize=False, proj=True):
        super().__init__()
        self._normalize = normalize
        proj_dims = 128 if proj else 3
        self._projection = ProjectionLayer(input_dims=input_dims, proj_dims=proj_dims, proj_=proj)
        self._create_layers(n_layers, feature_dims, hidden_size)

    def _create_layers(self, n_layers, feature_dims, hidden_size):
        input_dims = 3
        proj_dims = self._projection.proj_dims

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(input_dims)
            # mask[torch.arange(input_dims) % 2 == (i%2)] = 1
            mask[torch.randperm(input_dims)[:2]] = 1

            map_s = nn.Sequential(
                nn.Linear(proj_dims+feature_dims, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, input_dims),
                nn.Hardtanh(min_val=-10, max_val=10)
            )
            map_t = nn.Sequential(
                nn.Linear(proj_dims+feature_dims, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, input_dims)
            )
            self.layers.append(CouplingLayer(
                map_s,
                map_t,
                self._projection,
                mask
            ))

        if self._normalize:
            self.scales = nn.Sequential(
                nn.Linear(feature_dims, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, 3)
            )

    def _check_shapes(self, F, x):
        B1, M1 = F.shape
        B2, M2, D2 = x.shape
        assert B1 == B2 and D2 == 3

    def _expand_features(self, F, x):
        B, M2, D = x.shape
        return F[:, None].expand(-1, M2, -1)

    def _call(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def _normalize_input(self, F, y):
        if not self._normalize:
            return 0, 1

        sigma = torch.nn.functional.elu(self.scales(F)) + 1
        sigma = sigma[:, None]

        return 0, sigma

    def forward(self, x, F, mode='direct'):
        self._check_shapes(F, x)
        mu, sigma = self._normalize_input(F, x)
        F = self._expand_features(F, x)
        y_seq = []

        if mode == 'direct':
            y = x
            y = (y - mu) * sigma
            ldj = 0
            for l in self.layers:
                y, ldji = self._call(l, F, y)
                ldj = ldj + ldji
                y_n = y / sigma + mu
                y_seq.append(y_n)

        elif mode == 'inverse':
            y = x
            y = (y - mu) * sigma
            ldj = 0
            for l in reversed(self.layers):
                y, ldji = self._call(l.inverse, F, y)
                ldj = ldj + ldji
                y_n = y / sigma + mu
                #y_n = y
                y_seq.append(y_n)

        return y_seq, ldj

    def inverse(self, F, y):
        self._check_shapes(F, y)
        mu, sigma = self._normalize_input(F, y)
        F = self._expand_features(F, y)

        x = y
        x = (x - mu) * sigma
        ldj = 0
        for l in reversed(self.layers):
            x, ldji = self._call(l.inverse, F, x)
            ldj = ldj + ldji
        return x, ldj

if __name__=="__main__":
    net = SimpleNVP(5, 256, 256)
    print("Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in net.parameters())
        ))
    input = torch.rand(2, 5000, 3)
    F = torch.rand(2, 256)
    out, ldj = net(input, F, mode="inverse")
