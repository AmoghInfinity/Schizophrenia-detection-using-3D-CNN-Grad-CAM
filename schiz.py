!pip install nibabel
!pip install torch torchvision torchcam
!pip install scikit-learn

import os
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

#Base directory
BASE_DIR = "/content/drive/MyDrive/coinstac-regression-vbm-data-master"

df1 = pd.read_csv(os.path.join(BASE_DIR, "site_01+02", "site_01+02_large.csv"))
df2 = pd.read_csv(os.path.join(BASE_DIR, "site_03+04", "site_03+04_large.csv"))

#labels: 0 = Control, 1 = Schizophrenia
def convert_label(x):
    return 0 if str(x).upper() == "TRUE" else 1

for df, site in zip([df1, df2], ["site_01+02", "site_03+04"]):
    df["label"] = df["isControl"].apply(convert_label)
    df["path"] = df["niftifile"].apply(lambda x: os.path.join(BASE_DIR, site, x))

#Merge both sites
df_all = pd.concat([df1, df2], ignore_index=True)

#Checking for existing files dropping the rest
df_all = df_all[df_all["path"].apply(os.path.exists)].reset_index(drop=True)
print(f"Total subjects with valid files: {len(df_all)}")

class MRIDataset(Dataset):
    def __init__(self, dataframe, target_shape=(64,64,64)):
        self.df = dataframe
        self.target_shape = target_shape

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = nib.load(row["path"]).get_fdata()
        img = np.nan_to_num(img)

        #Normalize
        img = (img - np.mean(img)) / (np.std(img) + 1e-5)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  #(1, D,H,W)

        #Resizing if needed
        if self.target_shape is not None:
            img = F.interpolate(img.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False)
            img = img.squeeze(0)

        label = torch.tensor(row["label"], dtype=torch.long)
        return img, label

train_df, test_df = train_test_split(df_all, test_size=0.2, stratify=df_all["label"], random_state=42)

train_dataset = MRIDataset(train_df, target_shape=(64,64,64))
test_dataset = MRIDataset(test_df, target_shape=(64,64,64))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

class CNN3D(nn.Module):
    def __init__(self, input_shape=(1,64,64,64)):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2),
        )
        #Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.conv_layers(dummy)
            n_features = dummy.numel() // dummy.shape[0]

        self.fc_layers = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN3D(input_shape=(1,64,64,64)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
EPOCHS = 40

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct = 0.0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    train_acc = correct / len(train_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {train_acc:.4f}")

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Total: {total}")
print(f"Correct: {correct}")

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.device = next(model.parameters()).device
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        #register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        input_tensor = input_tensor.to(self.device)
        self.model.zero_grad()

        outputs = self.model(input_tensor)
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        score = outputs[0, target_class]
        score.backward(retain_graph=True)

        #compute weights
        weights = self.gradients.mean(dim=(2,3,4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        #normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        #upsample to input size
        upsample_size = input_tensor.shape[2:]
        cam_torch = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
        cam_up = F.interpolate(cam_torch, size=upsample_size, mode='trilinear', align_corners=False)
        cam_up = cam_up.squeeze().numpy()

        return cam_up, target_class

  def central_slices(volume):
    D, H, W = volume.shape
    return volume[D//2,:,:], volume[:,H//2,:], volume[:,:,W//2]

def show_gradcam(volume, cam, title="Grad-CAM"):
    #volume and cam are (D,H,W)
    ax_img, cor_img, sag_img = central_slices(volume)
    ax_cam, cor_cam, sag_cam = central_slices(cam)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.ravel(): ax.axis('off')

    #Original slices
    axes[0,0].imshow(ax_img.T, cmap='gray', origin='lower'); axes[0,0].set_title('Axial Slice')
    axes[0,1].imshow(cor_img.T, cmap='gray', origin='lower'); axes[0,1].set_title('Coronal Slice')
    axes[0,2].imshow(sag_img.T, cmap='gray', origin='lower'); axes[0,2].set_title('Sagittal Slice')

    #Overlay heatmap
    axes[1,0].imshow(ax_img.T, cmap='gray', origin='lower')
    axes[1,0].imshow(ax_cam.T, cmap='hot', alpha=0.5, origin='lower')
    axes[1,0].set_title('Axial + CAM')

    axes[1,1].imshow(cor_img.T, cmap='gray', origin='lower')
    axes[1,1].imshow(cor_cam.T, cmap='hot', alpha=0.5, origin='lower')
    axes[1,1].set_title('Coronal + CAM')

    axes[1,2].imshow(sag_img.T, cmap='gray', origin='lower')
    axes[1,2].imshow(sag_cam.T, cmap='hot', alpha=0.5, origin='lower')
    axes[1,2].set_title('Sagittal + CAM')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

sample_indices = [0, 1, 2]  #indices of test samples

#Choosing the last conv layer
target_layer = model.conv_layers[-4]
gcam = GradCAM3D(model, target_layer)

for idx in sample_indices:
    sample_img, sample_label = test_dataset[idx]
    input_tensor = sample_img.unsqueeze(0).to(device)  #(1,1,D,H,W)

    #generate Grad-CAM
    cam3d, pred_class = gcam.generate_cam(input_tensor)

    #visualize
    volume_np = input_tensor.cpu().squeeze().numpy()
    show_gradcam(volume_np, cam3d, title=f"Grad-CAM | Sample={idx} | True={sample_label.item()} | Pred={pred_class}")
