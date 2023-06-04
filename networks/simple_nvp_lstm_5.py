import torch
from torch import nn

class ProjectionLayer(nn.Module):
    def __init__(self, input_dims=3, proj_dims=128, proj_=True):
        super().__init__()
        self.proj_dims = proj_dims

        self.proj = nn.Sequential(
            nn.Linear(input_dims, 2*proj_dims),
            nn.ReLU(),
            nn.Linear(2*proj_dims, proj_dims)
        )
        self.proj_ = proj_

    def forward(self, x):
        if self.proj_:
            return self.proj(x)
        else:
            return x

class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, projection, mask, rnn_F):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.projection = projection
        self.register_buffer("mask", mask)
        self.rnn_F = rnn_F

    def forward(self, F, y, state_F):
        y1 = y * self.mask
        F_y1 = torch.cat([F, self.projection(y1)], dim=-1)
        B, N, D = F_y1.shape
        F_ = F_y1.reshape(-1, D)
        state_F = self.rnn_F(F_, state_F)
        F_y1_s = state_F[0].reshape(B, N, -1)
        s = self.map_s(F_y1_s)
        t = self.map_t(F_y1_s)

        x = y1 + (1-self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj, state_F

    def inverse(self, F, x, state_F):
        x1 = x * self.mask
        F_x1 = torch.cat([F, self.projection(x1)], dim=-1)
        B, N, D = F_x1.shape
        F_ = F_x1.reshape(-1, D)
        state_F = self.rnn_F(F_, state_F)
        F_x1_s = state_F[0].reshape(B, N, -1)
        s = self.map_s(F_x1_s)
        t = self.map_t(F_x1_s)

        y = x1 + (1-self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)
        return y, ldj, state_F

def init_out_weights(self):
    for m in self.modules():
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param.data, -1e-5, 1e-5)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)

class SimpleNVP(nn.Module):
    def __init__(self, n_layers, feature_dims, hidden_size, input_dims=3,
                 normalize=False, proj=True):
        super().__init__()
        self._normalize = normalize
        proj_dims = 128 if proj else 3
        self._projection = ProjectionLayer(input_dims=input_dims, proj_dims=proj_dims, proj_=proj)

        self.rnn_F = nn.LSTMCell(feature_dims+proj_dims, hidden_size)
        self.rnn_F.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.rnn_F)

        self._create_layers(n_layers, feature_dims, hidden_size)

    def _create_layers(self, n_layers, feature_dims, hidden_size):
        input_dims = 3
        self.layers = nn.ModuleList()
        proj_dims = self._projection.proj_dims
        for i in range(n_layers):
            map_s = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims),
                nn.Hardtanh(min_val=-10, max_val=10)
            )
            map_t = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims)
            )

            map_s[-2].apply(init_out_weights)
            map_t[-1].apply(init_out_weights)
            mask = torch.zeros(input_dims)
            # mask[torch.arange(input_dims) % 2 == (i%2)] = 1
            mask[torch.randperm(input_dims)[:2]] = 1
            self.layers.append(CouplingLayer(
                map_s,
                map_t,
                self._projection,
                mask,
                self.rnn_F
            ))

        if self._normalize:
            self.scales = nn.Sequential(
                nn.Linear(feature_dims, hidden_size),
                nn.ReLU(),
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

    def forward(self, x, F, mode='direct', local_code=None, state_F=None):
        self._check_shapes(F, x)
        mu, sigma = self._normalize_input(F, x)
        F = self._expand_features(F, x)
        if local_code is not None:
            F = torch.cat([F, local_code], dim=-1)
        y_seq = []

        if mode == 'direct':
            y = x
            ldj = 0

            for l in self.layers:
                y, ldji, state_F = self._call(l, F, y, state_F)
                ldj = ldj + ldji
                y_n = y / sigma + mu
                y_seq.append(y_n)

        elif mode == 'inverse':
            y = x
            y = (y - mu) * sigma
            ldj = 0
            for l in reversed(self.layers):
                y, ldji, state_F = self._call(l.inverse, F, y, state_F)
                ldj = ldj + ldji
                y_n = y
                y_seq.append(y_n)

        return y_seq, ldj

    def inverse(self, F, y):
        self._check_shapes(F, y)
        mu, sigma = self._normalize_input(F, y)
        F = self._expand_features(F, y)

        state_s, state_t = None, None

        x = y
        x = (x - mu) * sigma
        ldj = 0
        for l in reversed(self.layers):
            x, ldji, state_s, state_t = self._call(l.inverse, F, x, state_s, state_t)
            ldj = ldj + ldji
        return x, ldj

if __name__=='__main__':
    net = SimpleNVP(5, 256, 256)
    print("Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in net.parameters())
        ))
    input = torch.rand(2, 5000, 3)
    F = torch.rand(2, 256)
    out, ldj = net(input, F, mode="inverse")
    p_final = out[-1]
    out_2, ldj_2 = net(p_final, F, mode='direct')
    p_final_inv = out_2[-1]
    err = torch.sum(torch.norm(p_final_inv-input, dim=-1))
    print(err)
