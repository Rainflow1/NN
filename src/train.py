import os
from datetime import datetime
import matplotlib.pyplot as plt

from UCF_CC_50 import UCF_CC_50_Dataset
from model import *
from dataset import *
from torch.utils.data import DataLoader


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

BATCH_SIZE = 1
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 2e-4
EPOCHS = 300
NORM = 10000
KERNEL = 9
THREADS = 2


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initialized with: " + str(device), end='\n\n')

    dataset = UCF_CC_50_Dataset("../UCF_CC_50", "../temp/UCF_CC_50", norm=NORM, kernel=KERNEL)

    training_set = dataset["train"]
    validation_set = dataset["valid"]

    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=THREADS)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=THREADS)

    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    model = SASNet()
    model.to(device)

    lossFN = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_mae = 1000000.0

    for epoch in range(EPOCHS):

        loss_avg = 0

        model.train()
        for i, (inputs, targets, count) in enumerate(training_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
                
            outputs = model(inputs)

            if epoch % 10 == 0 and i == 0:
               plt.figure(1)
               plt.imshow(targets.cpu().detach().squeeze().numpy())
               plt.figure(2)
               plt.imshow(outputs.cpu().detach().squeeze().numpy())
               plt.show()

            loss = lossFN(outputs, targets).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()

            print(f"Epoch: {epoch}, Pred {outputs.sum()/10000*255.0}, "
                  f"Count: {count.sum()}")

        mae, mse = 0., 0.
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets, count) in enumerate(validation_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                mae += abs(outputs.data.sum()-targets.sum()).to(device)

        mae = mae / len(validation_loader)

        print(f'(-)   Epoch {epoch}, MAE: {mae}    (-)')

        if epoch % 10 == 0 and epoch != 0:
            best_mae = mae
            model_path = '../checkpoint/model_{}.pt'.format(timestamp+"_"+str(epoch))
            torch.save(model.state_dict(), model_path)


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SASNet()
    model.to(device)

    files = [file for file in os.listdir("../checkpoint")]
    files.sort(reverse=True)
    
    model.load_state_dict(torch.load("../checkpoint/model_20240406_232349 160.pt"))

    dataset = UCF_CC_50_Dataset("../UCF_CC_50/", "../temp/UCF_CC_50/")

    tset = dataset["test"]

    d, gt, p = tset[2]

    input = d.unsqueeze(0).to(device)
    target = gt

    output = model(input)

    print(f'count: {output.sum() * 255.0 / 10000} | {p}')

    plt.imshow(output.to('cpu').detach()[0].permute(1, 2, 0))
    plt.savefig("../temp/testImg/test1.png")
    plt.imshow(target.permute(1, 2, 0))
    plt.savefig("../temp/testImg/test2.png")
    plt.imshow(input.to('cpu').detach()[0].permute(1, 2, 0))
    plt.savefig("../temp/testImg/test3.png")

    pass


if __name__ == "__main__":
    train()
    #test()

    