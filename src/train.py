import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import os
from datetime import datetime
from rich.progress import Progress
import matplotlib.pyplot as plt

from earlyStopper import EarlyStopper
from UCF_CC_50 import UCF_CC_50_Dataset
from model import SimpleUNet
from loss import balanced_cross_entropy


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

BATCH_SIZE = 8
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 2e-4
EPOCHS = 50
NORM = 10000
KERNEL = 9


def train():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Initialized with: " + str(device), end='\n\n')

    dataset = UCF_CC_50_Dataset("./dataset/UCF_CC_50/", "./temp/dataset/UCF_CC_50/", norm=NORM, kernel=KERNEL)

    training_set = dataset["train"]
    validation_set = dataset["valid"]

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    model: nn.Module = SimpleUNet()
    model.to(device)

    lossFN = nn.BCELoss() # TODO balance it
    #lossFN = balanced_cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    lr_schd = lr_scheduler.StepLR(optimizer, step_size=1e4, gamma=0.1)
    early_stopper = EarlyStopper(5, 500)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = 1000000.0
    optimizer.zero_grad()

    progress = Progress()

    with progress:

        epoch_progress = progress.add_task("Epoch", True, EPOCHS)
        train_progress = progress.add_task("Train progress: ", False, len(training_loader))
        validation_progress = progress.add_task("Validation progress: ", False, len(validation_loader))

        for epoch in range(EPOCHS):

            running_loss = 0
            i = 0

            progress.reset(train_progress)
            progress.reset(validation_progress)

            model.train(True)

            for data in training_loader:

                inputs, targets, count = data

                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)

                loss = lossFN(outputs, targets)
                running_loss += loss.item()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                lr_schd.step()
                    
                i += 1

                progress.advance(train_progress, 1)

            model.train(False)

            torch.cuda.empty_cache()

            avg_loss = running_loss / i
            running_loss = 0

            running_vloss = 0.0
            with torch.no_grad():
                for i, vdata in enumerate(validation_loader):
                    vinputs, vtargets, vcount = vdata
                    vinputs = vinputs.to(device)
                    vtargets = vtargets.to(device)
                    voutputs = model(vinputs)
                    vloss = lossFN(voutputs, vtargets)
                    running_vloss += vloss.item()

                    print(f'count: {torch.sum(voutputs[0])/NORM} | {vcount[0]}')

                    progress.advance(validation_progress, 1)

            avg_vloss = running_vloss / len(validation_loader)
            print('Epoch {} LOSS train {} valid {}'.format(epoch+1, avg_loss, avg_vloss))

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = './temp/model/model_{}.pt'.format(timestamp)
                torch.save(model.state_dict(), model_path)

            progress.advance(epoch_progress, 1)

            if early_stopper(avg_vloss):
                print("Train stopped")
                break
            pass
        pass
    pass

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model: nn.Module = SimpleUNet()
    model.to(device)

    files = [file for file in os.listdir("./temp/model/")]
    files.sort(reverse=True)
    
    model.load_state_dict(torch.load(os.path.join("./temp/model/", files[0])))

    dataset = UCF_CC_50_Dataset("./dataset/UCF_CC_50/", "./temp/dataset/UCF_CC_50/", norm=NORM)

    tset = dataset["test"]

    d, gt, p = tset[0]

    input = d.unsqueeze(0).to(device)
    target = gt

    output = model(input)

    print(f'count: {torch.sum(output)/NORM} | {p}')

    plt.imshow(output.to('cpu').detach()[0].permute(1, 2, 0))
    plt.savefig("./temp/testImg/test1.png")
    plt.imshow(target.permute(1, 2, 0))
    plt.savefig("./temp/testImg/test2.png")
    plt.imshow(input.to('cpu').detach()[0].permute(1, 2, 0))
    plt.savefig("./temp/testImg/test3.png")

    pass


if __name__ == "__main__":
    train()
    test()

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model: nn.Module = SimpleUNet()
    model.to(device)

    dataset = UCF_CC_50_Dataset("./dataset/UCF_CC_50/", "./temp/dataset/UCF_CC_50/")
    tset = dataset["test"]

    d, gt, p = tset[0]

    input = d.unsqueeze(0).to(device)
    target = gt.unsqueeze(0).to(device)

    output = model(input)

    print(f'count: {torch.sum(output)/10000} = {p}')

    plt.imshow(output.to('cpu').detach()[0].permute(1, 2, 0))
    plt.savefig("./temp/testImg/test1.png")
    plt.imshow(target.to('cpu').detach()[0].permute(1, 2, 0))
    plt.savefig("./temp/testImg/test2.png")
    plt.imshow(input.to('cpu').detach()[0].permute(1, 2, 0))
    plt.savefig("./temp/testImg/test3.png")
    """

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model: nn.Module = SimpleUNet()
    model.to(device)

    inputs = torch.ones(16, 3, 200, 200).to(device)
    targets = torch.ones(16, 3, 200, 200).to(device)

    lossFN = nn.MSELoss()

    model.train(True)

    running_loss = 0

    for n in range(10):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = model(inputs)

        loss = lossFN(outputs, targets)
        running_loss += loss.item()
        loss.backward()
        print(outputs)

    model.train(False)
    """
    