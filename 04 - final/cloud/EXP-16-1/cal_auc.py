import matplotlib as mpl
from utils import *
mpl.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':
    # Deepfakes Face2Face FaceSwap NeuralTextures
    VAL_FAKE_ROOT = r'/data2/Jianwei-Fei/00-Dataset/01-Images/00-FF++/Deepfakes/raw/val'
    VAL_REAL_ROOT = r'/data2/Jianwei-Fei/00-Dataset/01-Images/00-FF++/Real/raw/val'

    net = torch.load(r'models/epoch-044-loss-0.456-F2FAcc-0.959-AUC-0.985--FSAcc-0.670-AUC-0.715--NTAcc-0.956-AUC-0.987.pkl')
    Val(net, VAL_REAL_ROOT, VAL_FAKE_ROOT)
