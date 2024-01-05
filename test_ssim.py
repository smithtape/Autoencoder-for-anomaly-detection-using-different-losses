import torch
from torchvision import datasets, transforms
from model import Autoencoder
from pytorch_msssim import ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
test_dataset = datasets.ImageFolder(root='/home/smithtape/Desktop/phd_ncu_ubuntu/20230917_lab1/code3/data/test', transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

model = Autoencoder().to(device)
model.load_state_dict(torch.load('/home/smithtape/Desktop/phd_ncu_ubuntu/20230917_lab1/code3/autoencoder_mse.pth'))
model.eval()

with open('ssim_test_results.txt', 'w') as f:
    f.write("Filename\tSSIM Loss\tAnomaly Status\tCategory\n")

    for i, (img, label) in enumerate(test_loader):
        filepath, _ = test_dataset.samples[i]
        filename = filepath.split('/')[-1]

        img = img.to(device)
        output = model(img)
        loss = 1 - ssim(img, output)  # SSIM returns similarity; 1 - SSIM gives the loss
        category = "fault" if label.item() == 0 else "non_fault"
        anomaly_status = "Normal" if loss < 0.004 else "Anomaly"
        f.write(f"{filename}\t{loss.item():.4f}\t{anomaly_status}\t{category}\n")
