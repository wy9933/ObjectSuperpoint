from models.superpoint import SuperPoint
import torch

net = SuperPoint().cuda()
net.train()
points = torch.rand(1, 200, 3).cuda()  # 1 batch, 2 points, 3 coords
p_feat, sp_atten, sp_feat, sp_param = net(points)
loss = net.get_loss(points, p_feat, sp_atten, sp_feat, sp_param)
print(loss)
