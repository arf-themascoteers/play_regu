import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_file = "data.csv"
epochs_file = "epochs.csv"
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

# Update code to record loss at each epoch and plot with twin y-axis for loss

epochs_data = []
losses = []

for epoch in range(500):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epochs_data.append([model.w1.item(), model.w2.item()])
    losses.append(loss.item())

epochs_df = pd.DataFrame(epochs_data, columns=["w1", "w2"])
epochs_df['loss'] = losses
epochs_df.to_csv(epochs_file, index=False)

fig, ax1 = plt.subplots()

ax1.plot(epochs_df['w1'], label="w1")
ax1.plot(epochs_df['w2'], label="w2")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Weights")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(epochs_df['loss'], color='red', label="Loss")
ax2.set_ylabel("Loss")
ax2.legend(loc="upper right")
fig.suptitle("y = w1*x1 + w2*x2 [Data: y = 3*x1 = 3*x2 ]")

plt.savefig("main.png")
plt.show()



