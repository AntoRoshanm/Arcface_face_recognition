import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import time
from flask import Flask, render_template, Response

IMAGE_SIZE = 112

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Flask app
app = Flask(__name__)

# Haar Cascade Classifier for Face and Feature Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Data Augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Dataset class with augmentation
def load_images(data_directory):
    image_data, labels = [], []
    for label_name in os.listdir(data_directory):
        label_path = os.path.join(data_directory, label_name)
        if os.path.isdir(label_path):
            for file_name in os.listdir(label_path):
                image_path = os.path.join(label_path, file_name)
                image = cv2.imread(image_path)
                if image is not None:
                    resized_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                    normalized_image = resized_image / 255.0
                    image_data.append(normalized_image)
                    labels.append(label_name)
    return np.array(image_data), np.array(labels)

data_directory = "E:/Ecrio/Task/arcface/arcface/dataset/train"
images, labels = load_images(data_directory)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
class FaceDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms.ToPILImage()(img)
        img = transform(img)
        label = self.labels[idx]
        return img, label

train_dataset = FaceDataset(X_train, y_train)
test_dataset = FaceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ArcFace Loss Definition
class ArcFaceLoss(nn.Module):
    def __init__(self, scale=30.0, margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, logits, labels):
        cos_theta = logits.clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)
        target_logits = torch.cos(theta + self.margin)
        logits = logits * (1 - F.one_hot(labels, logits.size(1)).float()) + target_logits * F.one_hot(labels, logits.size(1)).float()
        logits = logits * self.scale
        return F.cross_entropy(logits, labels)

# ArcFace Model Definition
class ArcFaceModel(nn.Module):
    def __init__(self, input_shape, embedding_size, num_classes):
        super(ArcFaceModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * (input_shape[0] // 8) * (input_shape[1] // 8), 1024)
        self.fc2 = nn.Linear(1024, embedding_size)
        self.arcface_dense = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        embeddings = F.normalize(self.fc2(x), p=2, dim=1)
        logits = self.arcface_dense(embeddings)
        return embeddings, logits

input_shape = (IMAGE_SIZE, IMAGE_SIZE)
num_classes = len(np.unique(y_train))
embedding_size = 512
model = ArcFaceModel(input_shape, embedding_size, num_classes).to(device)

criterion = ArcFaceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def get_training_embeddings(model, train_loader):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, lbls in train_loader:
            images = images.to(device)
            emb, _ = model(images)
            embeddings.append(emb.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    return np.vstack(embeddings), labels

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    train_embeddings, train_labels = get_training_embeddings(model, train_loader)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if len(eye_cascade.detectMultiScale(face)) > 0 and len(smile_cascade.detectMultiScale(face)) > 0:
                face_resized = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
                frame_tensor = torch.tensor(face_resized.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding, _ = model(frame_tensor)
                    similarities = cosine_similarity(embedding.cpu().numpy(), train_embeddings)
                    max_similarity = np.max(similarities)
                    best_match_idx = np.argmax(similarities)
                    label = label_encoder.inverse_transform([train_labels[best_match_idx]])[0] if max_similarity > 0.75 else "NOT VERIFIED"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0) if max_similarity > 0.75 else (0, 0, 255), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
