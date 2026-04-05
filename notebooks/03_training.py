import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import torch.multiprocessing as mp

# Import our custom dataset
import sys
sys.path.insert(0, str(Path(".")))
from custom_dataset import CropDiseaseDataset, get_transforms

# ====================== CONFIG ======================
device = torch.device("cpu")
print(f"🚀 Using device: {device}")

root = Path(".")
processed = root / "data" / "processed"
batch_size = 32
num_epochs = 15
patience = 5

# ====================== MAIN TRAINING BLOCK ======================
if __name__ == "__main__":
    mp.freeze_support()   # Required for Windows multiprocessing
    
    print("🔥 Starting Training...\n")
    
    # ====================== LOAD DATASETS ======================
    train_tf, val_tf = get_transforms()
    
    train_dataset = CropDiseaseDataset(processed / "train.csv", transform=train_tf)
    val_dataset   = CropDiseaseDataset(processed / "val.csv",   transform=val_tf)
    
    # num_workers=0 → fixes the Windows error (safe & reliable)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    num_classes = len(train_dataset.classes)
    print(f"✅ Training {num_classes} classes")
    
    # ====================== MODEL ======================
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze early layers
    for param in model.features[:6].parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    
    model = model.to(device)
    
    # ====================== LOSS & OPTIMIZER ======================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # ====================== TRAINING LOOP ======================
    best_val_acc = 0.0
    early_stop_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # --- Train ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Time: {epoch_time:.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_crop_disease_model.pth")
            early_stop_counter = 0
            print("   💾 New best model saved!")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("⏹️ Early stopping triggered!")
                break
    
    print(f"\n🎉 Training finished! Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), "models/final_crop_disease_model.pth")
    print("💾 Final model saved!")
# ====================== PLOTS (for your report) ======================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title('Accuracy Curve')
plt.legend()
plt.tight_layout()
plt.savefig("reports/training_curves.png", dpi=300)
plt.show()

# ====================== FINAL METRICS ======================
print("\n📊 Final Classification Report on Validation Set:")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Confusion matrix (saved for report)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=300)
plt.show()