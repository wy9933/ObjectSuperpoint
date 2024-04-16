import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channel=3, output_channel=1024, hidden_channel=None, use_global=True):
        """
        init of encoder
        :param input_channel:
        :param output_channel:
        :param hidden_channel: list, the output channel of hidden layers
        :param use_global: bool, True: add global feature to point-wise feature and then do one more conv
        """
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.use_global = use_global
        self.hidden_channel = [64, 256] if hidden_channel is None else hidden_channel

        assert len(self.hidden_channel) > 0

        self.input_layer = nn.Sequential(
            nn.Conv1d(self.input_channel, self.hidden_channel[0], 1),
            nn.BatchNorm1d(self.hidden_channel[0]),
            nn.ReLU(inplace=True)
        )

        self.hidden_layers = nn.Sequential()
        hidden_in = self.hidden_channel[0]
        for c in self.hidden_channel[1:]:
            self.hidden_layers.append(nn.Sequential(
                nn.Conv1d(hidden_in, c, 1),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True)
            ))
            hidden_in = c

        if self.use_global:
            self.final_layer = nn.Sequential(
                nn.Conv1d(self.hidden_channel[-1], self.output_channel, 1),
                nn.BatchNorm1d(self.output_channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.output_channel * 2, self.output_channel, 1),
                nn.BatchNorm1d(self.output_channel),
                nn.ReLU(inplace=True)
            )
        else:
            self.final_layer = nn.Sequential(
                nn.Conv1d(self.hidden_channel[-1], self.output_channel, 1),
                nn.BatchNorm1d(self.output_channel),
                nn.ReLU(inplace=True)
            )
    def forward(self, points):
        """
        forward of encoder
        :param points: input data, B(batch) N(num) 3(xyz input)
        :return: point-wise feature, B N C
        """
        b, n, c = points.shape
        points = points.transpose(2, 1)  # B 3 N

        x = self.input_layer(points)  # B 64(or other) N

        for layer in self.hidden_layers:
            x = layer(x)  # B _ N

        if self.use_global:
            x = self.final_layer[:3](x)  # B 1024(or other) N
            x_global = torch.max(x, dim=2, keepdim=True)[0]  # B 1024(or other) N
            x = torch.cat([x_global.expand(-1, -1, n), x], dim=1)  # BG 2*1024(or other) N
            x = self.final_layer[3:](x)  # B 1024(or other) N
        else:
            x = self.final_layer(x)  # B 1024(or other) N

        x = x.transpose(2, 1)  # B N 1024(or other)
        return x


class SuperPoint(nn.Module):
    def __init__(self, config=None):
        """

        :param config:
        """
        super().__init__()
        if config is None:
            self.encoder_input_channel = 3
            self.encoder_output_channel = 1024
            self.encoder_hidden_channel = [64, 256]
            self.encoder_use_global = True

            self.superpoint_num = 50
            self.mlp_hidden_channel = [512, 256]
            self.param_num = 14
        else:
            raise NotImplementedError

        assert len(self.mlp_hidden_channel) > 0

        self.encoder = Encoder(input_channel=self.encoder_input_channel,
                               output_channel=self.encoder_output_channel,
                               hidden_channel=self.encoder_hidden_channel,
                               use_global=self.encoder_use_global)

        self.attention_layer = nn.Sequential(
            nn.Conv1d(self.encoder_output_channel, self.superpoint_num, 1)
        )

        self.param_mlp = nn.Sequential()
        mlp_in = self.encoder_output_channel
        for c in self.mlp_hidden_channel:
            self.param_mlp.append(nn.Sequential(
                nn.Conv1d(mlp_in, c, 1),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True)
            ))
            mlp_in = c
        self.param_mlp.append(nn.Conv1d(mlp_in, self.param_num, 1))

    def forward(self, points):
        """
        forward of SuperPoint
        :param points: input data, B(batch) N(num) 3(xyz input)
        :return: p_feat: point-wise features
                 sp_atten: superpoint attention map of points
                 sp_feat: superpoint-wise features
                 sp_param: superpoint-wise parameters
        """
        # get point-wise features O
        p_feat = self.encoder(points)  # B N C

        # get superpoint attention map A
        sp_atten = self.attention_layer(p_feat.transpose(2, 1))  # B 50(sp num) N
        sp_atten = F.softmax(sp_atten, dim=1)  # B 50(sp num) N, softmax on superpoint dim: dim-1

        # get superpoint features S
        sp_feat = torch.bmm(F.normalize(sp_atten, p=1, dim=2), p_feat)  # B 50(sp num) C, l1-norm on attention map last dim: dim-2

        # get superpoint parameters θ
        sp_param = self.param_mlp(sp_feat.transpose(2, 1)).transpose(2, 1)  # B 50(sp num) 14

        return p_feat, sp_atten, sp_feat, sp_param


if __name__ == "__main__":
    from torchsummary import summary
    net = SuperPoint().cuda()
    summary(net, input_size=(8192, 3), batch_size=8)