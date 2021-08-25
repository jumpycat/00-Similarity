from resmodel import resnet18
from torchsummary import summary
from torch_train import DealDataset
net = resnet18().to('cuda')
summary(net, (3, 256, 256))

# dealDataset = DealDataset()
# for i in range(100):
#     a = dealDataset.__getitem__(100)
#     print(a[1][0])
#     print(a[1][1])