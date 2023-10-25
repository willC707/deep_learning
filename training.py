import torch

from model import UNET
from data_loader import DataLoader
from data_loader import SegmentationDataset
import torch.nn as nn
import torch.optim as optim


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_loader = DataLoader()
    train_loader, train_size = d_loader.get_data_loader('./data/Data/train/image/', './data/Data/train/mask/')
    val_loader, val_size = d_loader.get_data_loader('./data/Data/test/image/', './data/Data/test/mask/')

    model = UNET(encChannels=(3,8,16), decChannels=(16,8))
    model = model.to(device)

    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_loss = []

    for epoch in range(500):

        model.train()
        t_loss = 0
        v_loss = 0

        for (i, (x,y)) in enumerate(train_loader):
            (x,y) = (x.to(device), y.to(device))
            pred = model(x)
            _loss = loss(pred, y)

            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()

            t_loss += _loss

        with torch.no_grad():
            model.eval()

            for (x,y) in val_loader:
                (x, y) = (x.to(device), y.to(device))
                pred = model(x)

                v_loss += loss(pred, y)

        print(f'Epoch: {epoch},Train Loss {t_loss/train_size}')
        print(f'Epoch: {epoch}, Validation Loss {v_loss/val_size}')


if __name__ == '__main__':
    train()