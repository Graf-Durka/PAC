import torch
import numpy as np
import os
import random as rn
from PIL import Image
from torchvision import transforms
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset

def create_pairs(path, num):
    folders = [os.path.join(path, f) for f in (os.listdir(path)) if os.path.isdir(os.path.join(path, f))]
    rn.shuffle(folders)
    
    persons_train = {}
    persons_test = {}
    tup_persons_train = []
    tup_persons_test = []
    pers = 0
    

    for f in folders:
        if pers < num:
            persons_train[pers] = [os.path.join(f, i) for i in (os.listdir(f))]
        else:
            persons_test[pers] = [os.path.join(f, i) for i in (os.listdir(f))]
        pers += 1
    
    for i in range(0, num):
        for k in range(0, num):
            if i == k:

                max_img = min(10, len(persons_train[i]))
                for p1 in range(max_img):
                    for p2 in range(p1+1, max_img):
                        tup_persons_train.append([
                            persons_train[i][p1], 
                            persons_train[k][p2], 
                            1
                        ])
            else:
                if i < k:
                    max_img1 = min(10, len(persons_train[i]))
                    max_img2 = min(10, len(persons_train[k]))

                    if max_img1 > 0 and max_img2 > 0:
                        tup_persons_train.append([
                            persons_train[i][0], 
                            persons_train[k][0], 
                            0
                        ])
    
    rn.shuffle(tup_persons_train)
    
    test_indices = list(persons_test.keys())
    
    for idx_i, i in enumerate(test_indices):
        for idx_k, k in enumerate(test_indices):
            if i == k:
                max_img = min(10, len(persons_test[i]))
                for p1 in range(max_img):
                    for p2 in range(p1+1, max_img):
                        tup_persons_test.append([
                            persons_test[i][p1], 
                            persons_test[k][p2], 
                            1
                        ])
            else:
                if idx_i < idx_k:
                    max_img1 = min(10, len(persons_test[i]))
                    max_img2 = min(10, len(persons_test[k]))
                    if max_img1 > 0 and max_img2 > 0:
                        tup_persons_test.append([
                            persons_test[i][0], 
                            persons_test[k][0], 
                            0
                        ])
    
    rn.shuffle(tup_persons_test)
    
    return tup_persons_train, tup_persons_test


class Dataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = self.transform(Image.open(img1_path).convert('RGB'))
        img2 = self.transform(Image.open(img2_path).convert('RGB'))
        return img1, img2, torch.tensor(label, dtype=torch.float32)
    

class Layers(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 128)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = Layers()
        self.margin = 1

    def forward_sup(self, x):
        return self.layers(x)
    
    def forward(self, input1, input2):
        output1 = self.forward_sup(input1)
        output2 = self.forward_sup(input2)
        return output1, output2

    def loss(self, output1, output2, Y):
        dist = torch.nn.functional.pairwise_distance(output1, output2)
        loss = Y * 0.5 * (dist**2) + (1 - Y) * 0.5 * (torch.clamp(self.margin - dist, min=0.0)**2)
        return torch.mean(loss)
    

def test_model(model, test_loader, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            output1, output2 = model(img1, img2)
            dist = torch.nn.functional.pairwise_distance(output1, output2).cpu().numpy()
            pred = (dist < threshold).astype(int)
            labels_np = labels.cpu().numpy()
            correct += np.sum(pred == labels_np)
            total += len(labels_np)
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    return accuracy


def visualize_test_pairs(model, pairs_list, transform, num_pairs=100, threshold=0.5):

    model.eval()
    indices = random.sample(range(len(pairs_list)), min(num_pairs, len(pairs_list)))
    
    for idx in indices:
        path1, path2, true_label = pairs_list[idx]

        img1 = transform(Image.open(path1).convert('RGB')).unsqueeze(0)
        img2 = transform(Image.open(path2).convert('RGB')).unsqueeze(0)
        
        with torch.no_grad():
            out1, out2 = model(img1, img2)
            distance = torch.nn.functional.pairwise_distance(out1, out2).item()
        
        pred_label = 1 if distance < threshold else 0
        status = "Совпадают" if pred_label == 1 else "Разные"
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        img1_pil = Image.open(path1)
        img2_pil = Image.open(path2)
        
        axes[0].imshow(img1_pil)
        axes[0].set_title(f"Истинная метка: {true_label}")
        axes[0].axis('off')
        
        axes[1].imshow(img2_pil)
        axes[1].set_title(f"Предсказание: {status}\nРасстояние: {distance:.4f}")
        axes[1].axis('off')
        
        plt.suptitle(f"Тестовая пара #{idx}")
        plt.tight_layout()
        plt.show()


def main():

    train, test = create_pairs('/media/denis/C/NSU_rutina/2_Course/PAC/Sem2/Face/archive', 27)
    train_dataset = Dataset(train)
    test_dataset = Dataset(test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epoch = 10
    model.train()

    for epoch in range(num_epoch):
        running_loss = 0.0
        for img1, img2, labels in train_loader:
            
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = model.loss(output1, output2, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
    
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {running_loss/len(train_loader):.4f}')

    test_model(model, test_loader)
    visualize_test_pairs(model, test, test_dataset.transform)


if __name__ == "__main__":
    main()