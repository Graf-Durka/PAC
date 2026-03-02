import os
import cv2
import torch
import random
import natsort
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE


# Данные
def get_data_splits(root, limit=2400):
    folders = natsort.natsorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    class_map = {folder: i for i, folder in enumerate(folders)}
    
    train_files, test_files = [], []
    for folder in folders:
        f_path = os.path.join(root, folder)
        files = natsort.natsorted([os.path.join(f_path, f) for f in os.listdir(f_path)])
        
        train_files.extend(files[:limit])
        test_files.extend(files[:limit])
        
    random.shuffle(train_files)
    random.shuffle(test_files)
    return train_files, test_files, class_map

# Архитектура
class SatelliteEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

class CosineSimilarityHead(nn.Module):
    def __init__(self, emb_size, num_classes):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(emb_size, num_classes))
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x):
        x_n = F.normalize(x)
        w_n = F.normalize(self.W, dim=0)
        return x_n @ w_n

# Функция потерь
def arcface_loss(cosine, target, num_classes, m=0.4):
    cosine = cosine.clip(-1+1e-7, 1-1e-7)
    arcosine = cosine.arccos()

    arcosine += F.one_hot(target, num_classes=num_classes) * m
    return F.cross_entropy(arcosine.cos(), target)

# Датасет
class EuroDataset(Dataset):
    def __init__(self, file_paths, mapping, transform):
        self.file_paths = file_paths
        self.mapping = mapping
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        label = self.mapping[os.path.basename(os.path.dirname(path))]
        return img, label

def main():
    train_paths, test_paths, class_to_idx = get_data_splits("EuroSAT_RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(EuroDataset(train_paths, class_to_idx, preprocess), 
                              batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(EuroDataset(test_paths, class_to_idx, preprocess), 
                             batch_size=32, shuffle=False)

    encoder = SatelliteEncoder(128).to(device)

    if((input("Do you wnat to load preptrained model? (y/n) ")).lower() == "y"):
        encoder.load_state_dict(torch.load("best_model.pth", map_location=device))
    else:
        cosine_head = CosineSimilarityHead(128, len(class_to_idx)).to(device)
        
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(cosine_head.parameters()), lr=0.001)

        # Обучение
        for epoch in range(3):
            encoder.train()
            total_loss = 0
            for i, (imgs, lbls) in enumerate(train_loader):
                imgs, lbls = imgs.to(device), lbls.to(device)
                
                optimizer.zero_grad()
                embs = encoder(imgs)
                scores = cosine_head(embs)
                loss = arcface_loss(scores, lbls, len(class_to_idx))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                if i % 50 == 0:
                    print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            print(f"== Epoch {epoch+1} Done. Avg Loss: {total_loss/len(train_loader):.4f}")

        torch.save(encoder.state_dict(), "best_model.pth")

    # Визуализация
    encoder.eval()
    all_embs, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            e = encoder(imgs.to(device))
            all_embs.append(e.cpu().numpy())
            all_lbls.extend(lbls.numpy())
            if len(all_lbls) >= 1000: break

    embs_2d = TSNE(n_components=2, random_state=42).fit_transform(np.concatenate(all_embs))
    plt.figure(figsize=(10, 8))
    plt.scatter(embs_2d[:, 0], embs_2d[:, 1], c=all_lbls[:len(embs_2d)], cmap='tab10', s=10)
    plt.show()

    encoder.eval()
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    for i in range(100):

        idx1 = random.randint(0, len(test_paths) - 1)
        idx2 = random.randint(0, len(test_paths) - 1)
        
        p1, p2 = test_paths[idx1], test_paths[idx2]
        
        img1_raw = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        img2_raw = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            t1 = preprocess(img1_raw).unsqueeze(0).to(device)
            t2 = preprocess(img2_raw).unsqueeze(0).to(device)
            
            e1 = F.normalize(encoder(t1))
            e2 = F.normalize(encoder(t2))
            
            distance = 1 - (e1 @ e2.T).item()


        plt.figure(f"Pair {i+1}", figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img1_raw)
        plt.title(f"Class: {idx_to_class[class_to_idx[os.path.basename(os.path.dirname(p1))]]}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img2_raw)
        plt.title(f"Class: {idx_to_class[class_to_idx[os.path.basename(os.path.dirname(p2))]]}")
        plt.axis('off')

        is_same = os.path.dirname(p1) == os.path.dirname(p2)
        plt.suptitle(f"Pair {i+1} | Distance: {distance:.4f} | Same Class: {is_same}")
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()