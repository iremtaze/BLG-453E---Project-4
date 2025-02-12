## QUESTION 4-) PCA / Feature Extraction

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Remove the last FC layer
        self.features = nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class TransferLearning:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_save_path = "best_model.pth"
        
        # Data transforms
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        self.train_dataset = datasets.ImageFolder(root='train', transform=self.data_transforms)
        self.val_dataset = datasets.ImageFolder(root='val', transform=self.data_transforms)
        self.test_dataset = datasets.ImageFolder(root='test', transform=self.data_transforms)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        self.model = models.resnet50(pretrained=True)
        
        # Freeze all layers except last few
        for name, param in self.model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        
        # Modify final layer
        num_classes = len(self.train_dataset.classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def train_model(self, num_epochs=10):
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * correct / total
            
            # Validation phase
            val_loss, val_acc = self.validate()
            
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train Loss: {running_loss/len(self.train_loader):.3f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.model_save_path)
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return val_loss/len(self.val_loader), 100.*correct/total
    
    def extract_features(self):
        # Create feature extractor
        feature_extractor = FeatureExtractor(self.model).to(self.device)
        feature_extractor.eval()
        
        features = []
        labels = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                batch_features = feature_extractor(inputs)
                features.append(batch_features.cpu().numpy())
                labels.extend(targets.numpy())
        
        features = np.concatenate(features, axis=0)
        return features, np.array(labels)
    
    def basic_feature_extraction(self):
        features = []
        labels = []
        
        for inputs, targets in self.test_loader:
            batch_features = inputs.view(inputs.size(0), -1).numpy()
            features.append(batch_features)
            labels.extend(targets.numpy())
        
        return np.concatenate(features, axis=0), np.array(labels)
    
    def visualize_pca(self, features, labels, title):
        # Apply PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)
        
        # Plot results
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels, cmap='tab10')
        plt.title(f'PCA Visualization - {title}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter)
        plt.show()
        
        # Print explained variance ratio
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    def visualize_cluster_samples(self, features_pca, labels, original_images, title, n_samples=3):
        """Visualize sample images from each cluster"""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Create a figure with subplots for each cluster
        fig = plt.figure(figsize=(15, 3*n_clusters))
        plt.suptitle(f'Sample Images from Each Cluster - {title}', size=16)
        
        for i, label in enumerate(unique_labels):
            # Get indices for this cluster
            cluster_indices = np.where(labels == label)[0]
            
            # Get sample images from this cluster
            sample_indices = np.random.choice(cluster_indices, min(n_samples, len(cluster_indices)), replace=False)
            
            for j, idx in enumerate(sample_indices):
                ax = fig.add_subplot(n_clusters, n_samples, i*n_samples + j + 1)
                img = original_images[idx]
                ax.imshow(img)
                ax.axis('off')
                if j == 0:
                    ax.set_title(f'Cluster {label}')
        
        plt.tight_layout()
        plt.show()

    def get_original_images(self):
        """Get original images from test dataset"""
        original_images = []
        for inputs, _ in self.test_loader:
            # Convert tensor to numpy and denormalize
            imgs = inputs.numpy()
            imgs = np.transpose(imgs, (0, 2, 3, 1))  # Change from BCHW to BHWC
            imgs = imgs * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            imgs = np.clip(imgs, 0, 1)
            original_images.extend(imgs)
        return np.array(original_images)

def main():
    transfer_learning = TransferLearning()
    
    # Train the model
    transfer_learning.train_model()
    
    # Load best model
    transfer_learning.model.load_state_dict(torch.load(transfer_learning.model_save_path))
    
    # Get original images for visualization
    original_images = transfer_learning.get_original_images()
    
    # Basic feature extraction and visualization
    print("Extracting basic features...")
    basic_features, basic_labels = transfer_learning.basic_feature_extraction()
    basic_pca = transfer_learning.visualize_pca(basic_features, basic_labels, "Basic Features")
    transfer_learning.visualize_cluster_samples(basic_pca, basic_labels, original_images, "Basic Features")
    
    # Learned feature extraction and visualization
    print("Extracting learned features...")
    learned_features, learned_labels = transfer_learning.extract_features()
    learned_pca = transfer_learning.visualize_pca(learned_features, learned_labels, "Learned Features")
    transfer_learning.visualize_cluster_samples(learned_pca, learned_labels, original_images, "Learned Features")
    
    # Analysis
    print("\nAnalysis of Results:")
    print("1. Basic Feature Extraction (Raw Pixels):")
    print("   - Clusters based on raw pixel values")
    print("   - More sensitive to lighting and background variations")
    print("   - Less semantic meaning in groupings")
    
    print("\n2. Learned Feature Extraction (ResNet):")
    print("   - Clusters based on high-level features")
    print("   - More robust to variations in lighting and pose")
    print("   - Better semantic grouping of similar objects")

if __name__ == "__main__":
    main()
