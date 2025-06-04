# CNNforAnimalDetection


# 🧠 CNN-Based Animal Image Classifier

This project demonstrates a **Convolutional Neural Network (CNN)** implemented using PyTorch to classify images into two categories: `animal` and `non-animal`. The CNN is trained on a labeled dataset structured in folders and is capable of predicting the class of new images.

---

## 🔍 Objective

Build a **from-scratch CNN classifier** using PyTorch that learns visual patterns from images and distinguishes between the two classes based on extracted features like edges, shapes, and textures.

---

## 🛠️ Technologies Used

- **Python 3**
- **PyTorch**
- **Torchvision**
- **Google Colab / Jupyter Notebook**

---

## 🧠 CNN Architecture Used

```python
class AnimalDetectionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AnimalDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)    # Output: [B, 16, 224, 224]
        self.pool = nn.MaxPool2d(2, 2)                             # Output: [B, 16, 112, 112]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # Output: [B, 32, 112, 112] → Pool → [B, 32, 56, 56]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # Output: [B, 64, 56, 56] → Pool → [B, 64, 28, 28]
        
        self.fc1 = nn.Linear(64 * 28 * 28, 128)                    # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes)                     # Output: 2 classes (animal / non-animal)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 + ReLU + Pool

        x = x.view(-1, 64 * 28 * 28)          # Flatten the tensor
        x = F.relu(self.fc1(x))               # Fully connected ReLU
        x = self.fc2(x)                       # Final output layer
        return x


dataset/
├── animal/
│   ├── image1.jpg
│   └── ...
└── non_animal/
    ├── image2.jpg
    └── ...




