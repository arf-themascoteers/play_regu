import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_file = "data_l1.csv"
epochs_file = "epochs_l1.csv"
if not pd.io.common.file_exists(data_file):
    x1 = np.random.uniform(-1, 1, 10000)
    data = pd.DataFrame({"x1": x1, "x2": x1, "y": 3 * x1})
    data.to_csv(data_file, index=False)

dataset = pd.read_csv(data_file)
x_data = torch.tensor(dataset[['x1', 'x2']].values, dtype=torch.float32)
y_data = torch.tensor(dataset['y'].values, dtype=torch.float32).view(-1, 1)

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.w1 * x[:, 0].view(-1, 1) + self.w2 * x[:, 1].view(-1, 1)

model = LinearModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

lambda_reg = 0.01

epochs_data = []
mse_losses = []
l1_norms = []
total_losses = []

for epoch in range(500):
    y_pred = model(x_data)
    mse_loss = criterion(y_pred, y_data)
    l1_norm = torch.abs(model.w1) + torch.abs(model.w2)
    loss = mse_loss + lambda_reg * l1_norm
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epochs_data.append([model.w1.item(), model.w2.item()])
    mse_losses.append(mse_loss.item())
    l1_norms.append(l1_norm.item())
    total_losses.append(loss.item())

epochs_df = pd.DataFrame(epochs_data, columns=["w1", "w2"])
epochs_df['mse_loss'] = mse_losses
epochs_df['l1_norm'] = l1_norms
epochs_df['total_loss'] = total_losses

epochs_df.to_csv(epochs_file, index=False)

fig, ax1 = plt.subplots()

ax1.plot(epochs_df['w1'], label="w1")
ax1.plot(epochs_df['w2'], label="w2")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Weights")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(epochs_df['mse_loss'], label="MSE Loss", color='blue')
ax2.plot(epochs_df['l1_norm'], label="L1 norm", color='green')
ax2.plot(epochs_df['total_loss'], label="Total Loss: MSE Loss + 0.01 * L1 norm", color='red')
ax2.set_ylabel("Loss")
ax2.legend(loc="upper right")
fig.suptitle("y = w1*x1 + w2*x2 [Data: y = 3*x1 = 3*x2 ]")

plt.savefig("l1.png")
plt.show()
