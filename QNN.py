import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import torchquantum as tq
from torchquantum import QuantumModule
from torchquantum import QuantumDevice
import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plttrainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True)

# Define the quantum-classical model, inheriting from TorchQuantum's QuantumModule
class QFCModel(tq.QuantumModule):
    # Nested class defining the quantum layer
    class QLayer(tq.QuantumModule):
        def __init__(self, op_list=None):
            super().__init__()
            self.n_wires = 4  # Number of qubits/wires
            self.random_layer = tq.RandomLayer(n_ops=6, wires=list(range(self.n_wires)))
            # Use predefined quantum operations if provided
            if op_list:
                self.random_layer.op_list = op_list  # Set the same operations for all models
            # Define quantum gates with parameters that can be trained
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            # Apply random quantum operations and then apply custom gates
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            # Apply additional quantum gates
            qdev.h(wires=3)
            qdev.sx(wires=2)
            qdev.cnot(wires=[3, 0])

    # Initialize the quantum-classical hybrid model
    def __init__(self, op_list=None):
        super().__init__()
        self.n_wires = 4  # Number of qubits
        # Encoder to convert classical data into quantum states
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_u3_h_rx"])
        # Quantum layer defined above
        self.q_layer = self.QLayer(op_list=op_list)
        # Measurement to convert quantum state back to classical information
        self.measure = tq.MeasureAll(tq.PauliZ)

    # Define forward pass through the quantum model
    def forward(self, x, use_qiskit=False):
        # Create a quantum device with the given number of qubits
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True)
        bsz = x.shape[0]
        # Perform average pooling and reshape data for quantum encoding
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode classical data into quantum states
        self.encoder(qdev, x)
        qdev.reset_op_history()  # Reset operation history after encoding
        # Apply quantum operations through the quantum layer
        self.q_layer(qdev)
        x = self.measure(qdev)
        # Reshape and apply softmax for classification output
        x = x.view(bsz, -1)[:, :2]
        x = F.log_softmax(x, dim=1)
        return x

# Custom dataset class to remap certain labels in the dataset
class RemapLabels(Dataset):
    def __init__(self, base_dataset, class_map):
        self.base_dataset = base_dataset  # Original dataset
        self.class_map = class_map  # Map from original labels to new ones
        # Filter dataset to include only the desired labels
        self.filtered_indices = [
            i for i, (_, label) in enumerate(self.base_dataset) if label in self.class_map
        ]

    # Retrieve an image and its remapped label
    def __getitem__(self, index):
        img, label = self.base_dataset[self.filtered_indices[index]]
        return img, self.class_map[label]

    # Return the length of the filtered dataset
    def __len__(self):
        return len(self.filtered_indices)

# Function to load the MNIST dataset and apply label remapping
def load_dataset():
    # Transformation pipeline for preprocessing the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize images to have zero mean and unit variance
    ])
    
    # Load the full training and testing datasets
    full_train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)

    # Remap labels 3 to 0 and 6 to 1 for a binary classification task
    class_map = {3: 0, 6: 1}  # Remapping 3 to 0 and 6 to 1
    train_dataset = RemapLabels(full_train_dataset, class_map)
    test_dataset = RemapLabels(full_test_dataset, class_map)

    return train_dataset, test_dataset

# Function to perform standard centralized QNN training
def qnn_training(model, train_dataset, device, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        total_correct = 0

        # Iterate over batches of dataset
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Accumulate loss, total samples, and accuracy metrics
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

# Main function to set up and run standard QNN training
def main():
    train_dataset, test_dataset = load_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate random quantum operations for the quantum model
    random_layer = tq.RandomLayer(n_ops=50, wires=list(range(4)))
    op_list = random_layer.op_list

    # Initialize the quantum neural network model
    model = QFCModel(op_list=op_list).to(device)

    # Run standard centralized QNN training
    qnn_training(model, train_dataset, device, epochs=10)
  # 13. EXPECTED COST VS ITERATIONS (OBJECTIVE PLOT)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, STEPS + 1), cost_history, linewidth=2, label="QNN Expected Cost")
    plt.xlabel("Optimization Iterations")
    plt.ylabel("Expected Cost")
    plt.title("Expected Cost vs Iterations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()  

if __name__ == "__main__":
    main()