import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from utils import *
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Deepfakes Face2Face FaceSwap NeuralTextures
TRAIN_FAKE_ROOT = r'/data2/Jianwei-Fei/00-Dataset/01-Images/00-FF++/Face2Face/raw/train/'
TRAIN_REAL_ROOT = r'/data2/Jianwei-Fei/00-Dataset/01-Images/00-FF++/Real/raw/train/'

VAL_FAKE_ROOT = r'/data2/Jianwei-Fei/00-Dataset/01-Images/00-FF++/Deepfakes/raw/val/'
VAL_REAL_ROOT = r'/data2/Jianwei-Fei/00-Dataset/01-Images/00-FF++/Real/raw/val/'

TYPE = 'raw'
EPOCH = 50
BATCH_SIZE = 64
LENGTH = BATCH_SIZE * 100


net = resnet18().to(device)
pretext_model = torch.load(r'/data2/Jianwei-Fei/01-Projects/02-Similarity/00-final/resnet18-5c106cde.pth')
model2_dict = net.state_dict()
state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
state_dict.pop('fc.weight')
state_dict.pop('fc.bias')
model2_dict.update(state_dict)
net.load_state_dict(model2_dict)

net.to(device)
dealDataset = DealDataset(TRAIN_FAKE_ROOT=TRAIN_FAKE_ROOT, TRAIN_REAL_ROOT=TRAIN_REAL_ROOT, LENGTH=LENGTH,TYPE=TYPE)
train_loader = DataLoader(dataset=dealDataset, batch_size=BATCH_SIZE, shuffle=True)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

if __name__ == '__main__':
    for epoch in range(EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        print('-'*80)
        net.train()
        step = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, label1, label2, label3, label4 = inputs.to(device), labels[0].to(device), labels[1].to(device), \
                                                     labels[2].to(device), labels[3].to(device)
            optimizer.zero_grad()
            output1, output2, output3, output4 = net(inputs)

            loss1 = criterion(output1, label1)
            loss2 = criterion(output2, label2)
            loss3 = criterion(output3, label3)
            loss4 = criterion(output4, label4)


            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            optimizer.step()

            data = '[epoch:%03d, iter:%03d] Loss: %.03f' % (epoch + 1, i, loss.item())
            with open('runs/log.txt', 'a', encoding='utf-8') as f:
                f.write(data)
                f.write('\n')

            if step % 10 == 0:
                print(data)
            step += 1
        acc1,th1,auc1 = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT, VAL_REAL_ROOT=VAL_REAL_ROOT)
        acc2,th2,auc2 = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT.replace('Deepfakes','Face2Face'), VAL_REAL_ROOT=VAL_REAL_ROOT)
        acc3,th3,auc3 = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT.replace('Deepfakes','FaceSwap'), VAL_REAL_ROOT=VAL_REAL_ROOT)
        acc4,th4,auc4 = Val(net, VAL_FAKE_ROOT=VAL_FAKE_ROOT.replace('Deepfakes','NeuralTextures'), VAL_REAL_ROOT=VAL_REAL_ROOT)

        # Deepfakes Face2Face FaceSwap NeuralTextures

        tag = 'epoch-%03d-loss-%.03f' \
              '--DFAcc-%.03f-AUC-%.04f--' \
              '--F2FAcc-%.03f-AUC-%.04f--' \
              '--FSAcc-%.03f-AUC-%.04f--' \
              '--NTAcc-%.03f-AUC-%.04f' % \
               (epoch + 1, loss.item(), acc1, auc1, acc2, auc2, acc3, auc3,acc4, auc4)
        print(tag)
        print('Average AUC:',round((auc1+auc2+auc3+auc4)/3,4))
        print('-'*50)
        with open('runs/log.txt', 'a', encoding='utf-8') as f:
            f.write(tag)
            f.write('\n')
        torch.save(net, r'models/' + tag + '.pkl')
