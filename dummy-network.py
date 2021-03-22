import os
import numpy as np

# libs for creating the torch based network
import torch
from torch_geometric.data import Data, DataLoader, Dataset
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class LoadData:
    def __init__(self, data_folder, transform):
        self.data_folder = data_folder
        self.files = os.listdir(self.data_folder)
        self.data_objects = self._get_data_objects()

    def _get_data_objects(self):
        data_objs = []
        for file in self.files:
            filename = os.path.join(self.data_folder, file)
            data_obj = torch.load(filename)
            data_objs.append(data_obj)
        return data_objs

    def load_data(self, batch_size=1):
        return DataLoader(self.data_objects, batch_size=batch_size)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(9, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0., training=self.training)
        x = self.lin(x)

        return self.sigmoid(x)


if __name__ == "__main__":
    print("hello")
    dataloader = LoadData("cascades")
    dataset = dataloader.load_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')

    model = GCN(hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    def train():
        model.train()
        for data in dataset:  # Iterate in batches over the training dataset.
            optimizer.zero_grad()  # Clear gradients.
            print(data.x)
            data = data.to(device)
            out = model(
                data.x, data.edge_index, data.batch
            )  # Perform a single forward pass.
            print(out)

            target = torch.unsqueeze(data.y, 1).to(torch.float)
            loss = criterion(out, target)  # Compute the loss.
            # print(loss)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

    def test(loader):
        model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            target = torch.unsqueeze(data.y, 1)
            # print(out, target)
            correct += np.count_nonzero(
                (out == target).cpu()
            )  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    for epoch in range(1, 201):
        train()
        train_acc = test(dataset)
        test_acc = test(dataset)
        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )
