import torch
from torchvision import datasets, transforms
from model import Autoencoder
from torch.autograd import Variable
from torch.nn.functional import mse_loss

# Hyperparameters
batch_size = 8
epochs = 50
learning_rate = 0.001

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Data loader
train_dataset = datasets.ImageFolder(root='/home/smithtape/Desktop/phd_ncu_ubuntu/20230917_lab1/code3/data/non_fault', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Model, criterion, optimizer
model = Autoencoder().to(device)
criterion = mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Train model
for epoch in range(epochs):
    for data in train_loader:
        img, _ = data
        img = Variable(img).to(device)
        output = model(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

print('Training complete.')
torch.save(model.state_dict(), '/home/smithtape/Desktop/phd_ncu_ubuntu/20230917_lab1/code3/autoencoder_mse.pth')
