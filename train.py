from models.vit import ViT
import torch
import torch.nn as nn

from data.loader import get_dataset


# train model
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    epoch = 0
    for inputs, labels in train_loader:
        if epoch == 1:
            break
        epoch += 1
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    return model


model = ViT(image_size=224,
            patch_size=16,
            num_classes=2,
            dim=256,
            depth=2,
            heads=2,
            mlp_dim=512)
train_loader = torch.utils.data.DataLoader(get_dataset(),
                                           batch_size=4,
                                           shuffle=True,
                                           num_workers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train(model, train_loader, optimizer, criterion, device)
