import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torchvision.models.resnet as resnet
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from PIL import Image
from torch.autograd import Variable

def main():
    model=resnet.resnet101()
    model.load_state_dict(torch.load('/home/winshare/premodel/resnet/resnet101-5d3b4d8f.pth'))
    model.cuda()
    model.eval()
    imagedir='/home/winshare/WSNet/timg.jpeg'
    image=Image.open(imagedir)
    img = tv_F.to_tensor(tv_F.resize(image, (224, 224)))
    img = tv_F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_input = Variable(torch.unsqueeze(img, 0))

    img_input = img_input.cuda()
    output = model(img_input)
    output1=output.cpu()
    # print(output.cpu())
    output2=output1.detach().numpy()
    print(output2.shape)

    
    score = F.softmax(output1, dim=1)
    _, prediction = torch.max(score.data, dim=1)
    print('prediction ',prediction)


if __name__ == '__main__':
    main()
    