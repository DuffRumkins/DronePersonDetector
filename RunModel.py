import torch
from PIL import Image
from torchvision import models, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((720,1280)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    ])


img = Image.open("LOCATION/OF/IMAGE/HERE")
img = transform(img).unsqueeze(0).to(device)



#state_dict = torch.load("Model/state_dict.pth")

#model = models.mobilenet_v2(pretrained=True)

#num_ftrs = model.classifier[1].in_features
#model.classifier[1] = torch.nn.Linear(num_ftrs,2)#len(class_names))

#model.to(device)

#model.load_state_dict(state_dict)


model = torch.load("Model/model.pth")
model.to(device)

model.eval()

outputs = model(img)
_, pred = torch.max(outputs,1)
print (["False","True"][pred])
