
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchquantum as tq
from torchquantum.device.devices import QuantumDevice

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Trainable Quantum Classifier Design
class QuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "rx", "wires": [0]},
            {"input_idx": [1], "func": "rx", "wires": [1]},
            {"input_idx": [2], "func": "rx", "wires": [2]},
            {"input_idx": [3], "func": "rx", "wires": [3]},
        ])

        self.rot_params0 = nn.Parameter(torch.zeros(4, 3))
        self.rot_params1 = nn.Parameter(torch.zeros(4, 3))

        self.rx = tq.RX()
        self.ry = tq.RY()
        self.rz = tq.RZ()
        self.cnot = tq.CNOT()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        x = x.view(x.shape[0], -1)
        self.q_device.reset_states(x.shape[0])
        self.encoder(self.q_device, x)

        # Layer 0
        for i in range(self.n_wires):
            self.rx(self.q_device, wires=i, params=self.rot_params0[i, 0].unsqueeze(0))
            self.ry(self.q_device, wires=i, params=self.rot_params0[i, 1].unsqueeze(0))
            self.rz(self.q_device, wires=i, params=self.rot_params0[i, 2].unsqueeze(0))
        for i in range(self.n_wires):
            self.cnot(self.q_device, wires=[i, (i + 1) % self.n_wires])

        # Layer 1
        for i in range(self.n_wires):
            self.rx(self.q_device, wires=i, params=self.rot_params1[i, 0].unsqueeze(0))
            self.ry(self.q_device, wires=i, params=self.rot_params1[i, 1].unsqueeze(0))
            self.rz(self.q_device, wires=i, params=self.rot_params1[i, 2].unsqueeze(0))
        for i in range(self.n_wires):
            self.cnot(self.q_device, wires=[i, (i + 1) % self.n_wires])

        out = self.measure(self.q_device)
        return out[:, :2]

# 2. Data Preparation
transform = transforms.Compose([
    transforms.Resize((2, 2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST Dataloader
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
mnist_train_idx = (mnist_train.targets == 0) | (mnist_train.targets == 1)
mnist_test_idx = (mnist_test.targets == 0) | (mnist_test.targets == 1)
mnist_train.data, mnist_train.targets = mnist_train.data[mnist_train_idx], mnist_train.targets[mnist_train_idx]
mnist_test.data, mnist_test.targets = mnist_test.data[mnist_test_idx], mnist_test.targets[mnist_test_idx]

mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True)
mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=128, shuffle=False)

# CIFAR-10 Dataloader
cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
cifar_train_targets = torch.tensor(cifar_train.targets)
cifar_test_targets = torch.tensor(cifar_test.targets)
cifar_train_idx = (cifar_train_targets == 0) | (cifar_train_targets == 1)
cifar_test_idx = (cifar_test_targets == 0) | (cifar_test_targets == 1)
cifar_train.data = cifar_train.data[cifar_train_idx.numpy()]
cifar_train.targets = cifar_train_targets[cifar_train_idx].tolist()
cifar_test.data = cifar_test.data[cifar_test_idx.numpy()]
cifar_test.targets = cifar_test_targets[cifar_test_idx].tolist()

cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=128, shuffle=True)
cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=128, shuffle=False)

# Helper function to train and evaluate
def train_and_eval(train_loader, test_loader, epochs=50):
    model = QuantumClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1

        scheduler.step()
        loss_history.append(running_loss / count)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        acc = (correct / total) * 100
        accuracy_history.append(acc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs}: Loss = {loss_history[-1]:.4f}, Accuracy = {acc:.2f}%")

    return loss_history, accuracy_history

# 3. Running Trainings
epochs = 50
print("=== Training QNN on MNIST (Binary) ===")
mnist_losses, mnist_accs = train_and_eval(mnist_train_loader, mnist_test_loader, epochs=epochs)

print("\n=== Training QNN on CIFAR-10 (Binary) ===")
cifar_losses, cifar_accs = train_and_eval(cifar_train_loader, cifar_test_loader, epochs=epochs)

# 4. Plot four graphs (MNIST loss/accuracy and CIFAR loss/accuracy)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Graph 1: MNIST Accuracy (Raw Only)
axs[0, 0].plot(range(1, epochs + 1), mnist_accs, color='blue', linewidth=2, label='MNIST Val Acc')
axs[0, 0].set_title('MNIST Validation Accuracy')
axs[0, 0].set_xlabel('Epochs')
axs[0, 0].set_ylabel('Accuracy (%)')
axs[0, 0].grid(True)
axs[0, 0].legend()

# Graph 2: MNIST Loss
axs[0, 1].plot(range(1, epochs + 1), mnist_losses, label='MNIST Train Loss', color='cyan', linewidth=2)
axs[0, 1].set_title('MNIST Training Loss')
axs[0, 1].set_xlabel('Epochs')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].grid(True)
axs[0, 1].legend()

# Graph 3: CIFAR-10 Accuracy (Raw Only)
axs[1, 0].plot(range(1, epochs + 1), cifar_accs, color='red', linewidth=2, label='CIFAR-10 Val Acc')
axs[1, 0].set_title('CIFAR-10 Validation Accuracy')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Accuracy (%)')
axs[1, 0].grid(True)
axs[1, 0].legend()

# Graph 4: CIFAR-10 Loss
axs[1, 1].plot(range(1, epochs + 1), cifar_losses, label='CIFAR-10 Train Loss', color='orange', linewidth=2)
axs[1, 1].set_title('CIFAR-10 Training Loss')
axs[1, 1].set_xlabel('Epochs')
axs[1, 1].set_ylabel('Loss')
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.savefig('comparison_curve.png')
plt.close()
print("Four comparison graphs generated and saved to comparison_curve.png")
