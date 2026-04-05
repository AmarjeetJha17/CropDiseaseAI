# Run this once to get full architecture printout for your report
import torch
from torchvision import models
import torch.nn as nn

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.classifier[1].in_features, 13)
)
model.load_state_dict(torch.load("models/best_crop_disease_model.pth", map_location=torch.device('cpu')))

print(model)
# OR for cleaner summary:
# from torchsummary import summary
# summary(model, (3, 224, 224))