from models.autoencoders import Autoencoder
import torch.optim as optim

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 100

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
