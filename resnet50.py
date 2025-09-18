import torch
print(torch.cuda.is_available())
import os
from torchvision.models import convnext_small, convnext_base, efficientnet_v2_s, resnet50
from torchvision.models import resnext101_32x8d, densenet161, resnet101
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def train_model(model, epochs):

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    def train_one_epoch(epoch_index):
        running_loss = 0.
        last_loss = 0.
        correct = 0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in tqdm(enumerate(training_loader)):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.to('cuda')
            # labels = torch.Tensor([validation_set.classes.index(classes[a]) for a in labels])
            # labels = labels.type(torch.IntTensor)
            labels = labels.to('cuda')

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)
            out = torch.argmax(outputs, dim=1)
            correct += (out == labels).float().sum()

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                running_loss = 0.

        return last_loss, correct


    epoch_number = 0
    EPOCHS = epochs
    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, correct = train_one_epoch(epoch_number)

        running_vloss = 0.0
        model.eval()

        val_correct=0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to('cuda')
                # vlabels = torch.Tensor([int(validation_set.classes.index(classes[a])) for a in vlabels]).int()
                vlabels = vlabels.to('cuda')
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

                vout = torch.argmax(voutputs, dim=1)
                val_correct += (vout == vlabels).float().sum()

        avg_vloss = running_vloss / (i + 1)
        accuracy = 100 * correct / len(training_set)
        vaccuracy = 100 * val_correct / len(validation_set)
        print(f'Training: {avg_loss}, Validation: {avg_vloss}')
        print(f'Training: {accuracy}, Validation: {vaccuracy}')
        if not os.path.exists(f"models/resnet50"):
            os.makedirs(f"models/resnet50")
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'models/resnet50/model_{}'.format(epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

transform = transforms.Compose([transforms.ToTensor()])

# Create datasets for training & validation, download if necessary
training_set = ImageFolder(root='train', transform=transform)
validation_set = ImageFolder(root='test', transform=transform)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=8, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=8, shuffle=False)

# Class labels
classes = ('Healthy', 'Cancer', 'Cardiomegaly', 'Effusion', 'Pneumonia', 'Pneumothorax')

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))   

model = resnet50(num_classes=6)
model.to('cuda')
train_model(model, 50)


