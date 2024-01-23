import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random

# UNet modeli tanımı
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3

# Özel veri seti sınıfı
class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".jpg")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Resmi ikiye bölelim
        img_width = img.shape[2]
        img_half_width = img_width // 2
        img_normal = img[:, :, :img_half_width]
        img_segmented = img[:, :, img_half_width:]

        return img_normal, img_segmented

# Veri seti ve DataLoader oluşturma
transform = transforms.Compose([
    transforms.Resize((256, 512)),  # Dikkat: boyutu 256x512 olarak ayarlayın
    transforms.ToTensor()
])

dataset = CustomDataset("cityscapes_data/train", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss ve optimizer tanımlama
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Eğitim
num_epochs = 5

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for step, (images, targets) in enumerate(train_loader, 1):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        if outputs.shape[-2:] != targets.shape[-2:]:
            outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Training accuracy calculation
        correct_train += torch.sum((outputs - targets).abs() < 0.5)  # Modify as needed
        total_train += np.prod(targets.shape)

        # Print metrics
        if step % 10== 0:  # Print every 10 steps
            train_accuracy = correct_train.item() / total_train  # Calculate accuracy here
            print(f"Epoch {epoch + 1}/{num_epochs}, Step {step}/{len(train_loader)}, Train Loss: {loss.item()}")
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_accuracy}")

    average_train_loss = running_loss / len(train_loader)
    train_losses.append(average_train_loss)

    train_accuracy = correct_train.item() / total_train
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for step, (val_images, val_targets) in enumerate(val_loader, 1):
            val_images, val_targets = val_images.to(device), val_targets.to(device)
            val_outputs = model(val_images)

            if val_outputs.shape[-2:] != val_targets.shape[-2:]:
                val_outputs = F.interpolate(val_outputs, size=val_targets.shape[-2:], mode='bilinear', align_corners=False)

            val_loss = criterion(val_outputs, val_targets)
            running_val_loss += val_loss.item()

            # Validation accuracy calculation
            correct_val += torch.sum((val_outputs - val_targets).abs() < 0.5)  # Modify as needed
            total_val += np.prod(val_targets.shape)

            # Print metrics
            if step % 10 == 0:  # Print every 10 steps
                print(f"Epoch {epoch + 1}/{num_epochs}, Validation Step {step}/{len(val_loader)}, Validation Loss: {val_loss.item()}")

    average_val_loss = running_val_loss / len(val_loader)
    val_losses.append(average_val_loss)

    val_accuracy = correct_val.item() / total_val
    val_accuracies.append(val_accuracy)

    # Print and/or log the metrics
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_train_loss}, Train Accuracy: {train_accuracy}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {average_val_loss}, Validation Accuracy: {val_accuracy}")

# Plotting the results
plt.figure(figsize=(12, 4))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("Eğitim tamamlandı.")

# Select a random image from the validation dataset
random_index = random.randint(0, len(val_dataset) - 1)
sample_image_tuple = val_dataset[random_index]
sample_image_actual, sample_image_segmented = sample_image_tuple

# Add batch dimension and move to the device
sample_image_actual = sample_image_actual.unsqueeze(0).to(device)
sample_image_segmented = sample_image_segmented.unsqueeze(0).to(device)

# Use the trained model to make a prediction
model.eval()
with torch.no_grad():
    predicted_image = model(sample_image_actual)

    # Resize the predicted image to match the actual size
    predicted_image = F.interpolate(predicted_image, size=sample_image_actual.shape[-2:], mode='bilinear', align_corners=False)
# Görselleştirmek için tensörleri numpy dizilerine dönüştürme
actual_image_np = sample_image_actual.cpu().numpy().squeeze().transpose((1, 2, 0))
predicted_image_np = predicted_image.cpu().numpy().squeeze().transpose((1, 2, 0))
actual_image_segmented_np = sample_image_segmented.cpu().numpy().squeeze().transpose((1, 2, 0))

# Piksel değerlerini [0, 1] aralığında olacak şekilde kırpma
actual_image_np = np.clip(actual_image_np, 0, 1)
predicted_image_np = np.clip(predicted_image_np, 0, 1)
actual_image_segmented_np = np.clip(actual_image_segmented_np, 0, 1)

# Gerçek, tahmin edilen ve zemin gerçekliği segmente edilmiş görüntüleri göstermek için
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(actual_image_np)
plt.title('Gerçek Görüntü')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(predicted_image_np)
plt.title('Tahmin Edilen Görüntü')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(actual_image_segmented_np)
plt.title('Orjinal Segmente Hali')
plt.axis('off')

plt.show()
