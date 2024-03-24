import os
from datetime import datetime
import matplotlib.pyplot as plt

from earlyStopper import EarlyStopper
from UCF_CC_50 import UCF_CC_50_Dataset
from model import *
from torch.utils.data import DataLoader


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

    dataset = UCF_CC_50_Dataset("../UCF_CC_50", "../temp/UCF_CC_50", norm=NORM, kernel=KERNEL)

    training_set = dataset["train"]
    validation_set = dataset["valid"]

    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    print('Training set has {} instances'.format(len(training_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    model = SASNet().to(device)

    lossFN = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopper = EarlyStopper(5, 500)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_mae = 1000000.0

    for epoch in range(EPOCHS):

        loss_avg = 0

        model.train()
        for i, (inputs, targets, count) in enumerate(training_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
                
            outputs = model(inputs)

            loss = lossFN(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()

            print("Epoch:{}, Batch:{}, Loss:{:.4f}".format(epoch, i, loss_avg / BATCH_SIZE / (i + 1)))
            print(f"Pred {outputs.sum()}, Count: {count.sum()}")

        mae, mse = 0., 0.
        model.eval()
        with torch.no_grad():
            for i, (img, gt, count) in enumerate(validation_loader):
                img = img.to(device)

                pred = model(img)

                mae += torch.abs(pred.sum() - gt.sum()).item()
                mse += ((pred.sum() - gt.sum()) ** 2).item()

        mae = mae / len(validation_loader)
        mse = mse / len(validation_loader)

        print(f'Epoch {epoch}, MAE: {mae}, MSE: {mse}')

        if mae < best_mae:
            best_mae = mae
            model_path = '../checkpoint/model_{}.pt'.format(timestamp+" "+str(epoch))
            torch.save(model.state_dict(), model_path)


        #if early_stopper(mae):
        #    print("Train stopped")
        #    break

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model: nn.Module = UNet()
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
    