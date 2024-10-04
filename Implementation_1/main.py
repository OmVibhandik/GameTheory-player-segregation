import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

def extract_features(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(image)
    
    return features.cpu().numpy().flatten()

def process_folder(folder_path, model, transform, device, n_clusters=2):
    features = []
    file_paths = []
    
    for filename in tqdm(os.listdir(folder_path), desc=f"Processing {os.path.basename(folder_path)}"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            feature = extract_features(file_path, model, transform, device)
            features.append(feature)
            file_paths.append(file_path)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    return file_paths, labels

def segregate_players(input_folders, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor().to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    os.makedirs(output_folder, exist_ok=True)
    player_count = 0

    for folder in input_folders:
        file_paths, labels = process_folder(folder, model, transform, device)
        
        for file_path, label in tqdm(zip(file_paths, labels), desc=f"Segregating {os.path.basename(folder)}", total=len(file_paths)):
            player_folder = os.path.join(output_folder, f"player{player_count + label}")
            os.makedirs(player_folder, exist_ok=True)
            shutil.copy(file_path, os.path.join(player_folder, os.path.basename(file_path)))
        
        player_count += 2

if __name__ == "__main__":
    input_folders = ['two_players_bot', 'two_players_top']
    output_folder = 'output'
    segregate_players(input_folders, output_folder)