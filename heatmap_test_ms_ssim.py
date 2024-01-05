import torch
import numpy as np
from torchvision import datasets, transforms
from model import Autoencoder
import cv2
from pytorch_msssim import ms_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
test_dataset = datasets.ImageFolder(root='/home/smithtape/Desktop/phd_ncu_ubuntu/20230917_lab1/code3/data/test', transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

model = Autoencoder().to(device)
model.load_state_dict(torch.load('/home/smithtape/Desktop/phd_ncu_ubuntu/20230917_lab1/code3/autoencoder_ms_ssim.pth', map_location=device))
model.eval()

with open('ms_ssim_heatmap_results.txt', 'w') as f:
    f.write("Filename\tMS-SSIM Loss\tAnomaly Status\tCategory\n")
    for idx, (img, label) in enumerate(test_loader):
        img = img.to(device)
        output = model(img)
        loss = 1 - ms_ssim(img, output)  # Using 1-ms_ssim as the loss
        
        residual = (img - output).squeeze().detach().cpu().numpy()
        heatmap = np.sum(np.abs(residual), axis=0)
        heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        img_path, _ = test_loader.dataset.samples[idx]
        img_name = img_path.split('/')[-1].split('.')[0]

        cv2.imwrite(f'heatmap_images_ms_ssim/{img_name}_heatmap.png', heatmap)

        category = "fault" if label.item() == 0 else "non_fault"
        anomaly_status = "Normal" if loss < 0.04 else "Anomaly"  # Adjust the threshold as needed
        f.write(f"{img_name}.png\t{loss.item():.4f}\t{anomaly_status}\t{category}\n")
