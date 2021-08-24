from torchsummary import summary
from resmodel import resnet18
import torch
from torch_train import *

# dealDataset = DealDataset()
# for i in range(100):
#     a = dealDataset.__getitem__(100)

# x = torch.tensor([1,1],dtype=torch.float64)
# y = torch.tensor([-1,-1],dtype=torch.float64)
#
# c = (torch.cosine_similarity(x, y,dim=0)+1)/2
# print(c)



# T1 = torch.FloatTensor(32,128,1)
# T2 = torch.FloatTensor(32,128,1)
#
# T3 = torch.stack((T1,T2),dim=-1)
# T4 = torch.stack((T3,T3),dim=-1)
#
# print(T4.shape)

net = resnet18().to('cuda')
summary(net, (3, 256, 256))

# ts = []
# x = torch.rand(10,128,64,64)
# y = torch.chunk(x,8,dim=2)
# for i in y :
#     z = torch.split(i,8,dim=3)
#     for d in z:
#         ts.append(d)
#
# print(ts[0].size())
#
# cow = [torch.rand(10,128,64,64)]
#
# cow = torch.stack(cow, dim=-1)