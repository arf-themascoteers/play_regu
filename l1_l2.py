import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_file = "data_l1.csv"
epochs_file = "epochs_l1_l2.csv"
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

lambda_reg = 1

epochs_data = []
mse_losses = []
l1_norms = []
l2_norms = []
regs = []
total_losses = []

for epoch in range(500):
    y_pred = model(x_data)
    mse_loss = criterion(y_pred, y_data)
    l1_norm = torch.abs(model.w1) + torch.abs(model.w2)
    l2_norm = torch.sqrt(torch.square(model.w1) + torch.square(model.w2))
    reg = (l1_norm/l2_norm)
    loss = mse_loss + lambda_reg * reg
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epochs_data.append([model.w1.item(), model.w2.item()])
    mse_losses.append(mse_loss.item())
    l1_norms.append(l1_norm.item())
    l2_norms.append(l2_norm.item())
    regs.append(reg.item())
    total_losses.append(loss.item())

epochs_df = pd.DataFrame(epochs_data, columns=["w1", "w2"])
epochs_df['mse_loss'] = mse_losses
epochs_df['l1_norm'] = l1_norms
epochs_df['l2_norm'] = l2_norms
epochs_df['reg'] = regs
epochs_df['total_loss'] = total_losses

epochs_df.to_csv(epochs_file, index=False)

fig, ax1 = plt.subplots()

ax1.plot(epochs_df['w1'], label="w1", linewidth=4, linestyle="-", color='black')
ax1.plot(epochs_df['w2'], label="w2", linewidth=4, linestyle="-", color='grey')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Weights")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(epochs_df['mse_loss'], label="MSE Loss", color='blue', linestyle="--")
ax2.plot(epochs_df['l1_norm'], label="L1 norm", color='green', linestyle="--")
ax2.plot(epochs_df['l2_norm'], label="L2 norm", color='yellow', linestyle="--")
ax2.plot(epochs_df['reg'], label="L1/L2", color='magenta', linestyle="--")
ax2.plot(epochs_df['total_loss'], label="Total Loss = MSE Loss + 1* (L1/L2)", color='red', linestyle="--")
ax2.set_ylabel("Loss")
ax2.legend(loc="upper right")

fig.suptitle("y = w1*x1 + w2*x2 [Data: y = 3*x1 = 3*x2 ]")

plt.savefig("l1_l2.png")
plt.show()