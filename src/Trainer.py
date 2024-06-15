import os
from datetime import datetime
import time
from model import *
from losses import *
from vgg19 import *
from dataset import UCFDataset
from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader


class Avg:
    def __init__(self):
        self.n = 0
        self.sum = 0

    def add_sample(self, value):
        self.sum += value
        self.n += 1

    def get_avg(self):
        return 1.0 * self.sum / self.n


def gettime():
    return datetime.now().strftime('%y.%m.%d %H:%M:%S')


class Trainer:
    def __init__(self, args):
        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_save_path = args.model
        start_time = datetime.now().strftime("%y-%m-%d_%H-%M")
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        if not os.path.exists(os.path.join(model_save_path, start_time)):
            os.makedirs(os.path.join(model_save_path, start_time))

        self.start_time = start_time
        self.model_save_path = model_save_path
        self.epochs = args.epochs

        train_set = UCFDataset(dataset_path=os.path.join(args.dataset, "train"), mode="train")
        val_set = UCFDataset(dataset_path=os.path.join(args.dataset, "val"), mode="val")

        self.training_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, pin_memory=True,
                                          num_workers=args.workers)
        self.validation_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, pin_memory=True,
                                            num_workers=args.workers)

        self.model = vgg19()
        self.model.to(device)
        self.lossFN = BayLoss(True, device)
        self.postProb = PostProb(8.0, 512, 8, 1.0, True, device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.best = np.inf

    def train(self):
        for epoch in range(self.epochs):
            print("{} Epoch: {}".format(gettime(), epoch))
            self.train_epoch()
            self.validate(epoch)
        print("{} Training Finished".format(gettime()))

    def train_epoch(self):
        mae = Avg()
        mse = Avg()
        avg_loss = Avg()
        epoch_time = time.time()
        self.model.train()

        for i, (inputs, points, target, count) in enumerate(self.training_loader):
            inputs = inputs.to(self.device)
            gt_count = np.array([len(p) for p in points])
            points = [p.to(self.device) for p in points]
            target = [t.to(self.device) for t in target]
            count = count.to(self.device)

            with torch.enable_grad():
                outputs = self.model(inputs)
                prob_list = self.postProb(points, count)
                loss = self.lossFN(prob_list, target, outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pre_count = outputs.sum().detach().cpu().numpy()
                diff = pre_count - gt_count
                avg_loss.add_sample(loss.item())
                mae.add_sample(np.mean(abs(diff)))
                mse.add_sample(np.mean(diff * diff))

        print("{} Stage: Train, Loss: {:.2f}, MAE: {:.2f}, MSE: {:.2f}, Time: {:.2f}".
              format(gettime(), avg_loss.get_avg(), mae.get_avg(), np.sqrt(mse.get_avg()), time.time() - epoch_time))

    def validate(self, epoch):
        diff_list = []
        epoch_time = time.time()
        self.model.eval()

        for i, (inputs, count, name) in enumerate(self.validation_loader):
            inputs = inputs.to(self.device)
            count = count.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                diff = count.item() - outputs.sum().item()

                diff_list.append(diff)

        diff_list = np.array(diff_list)
        mse = np.sqrt(np.mean(np.square(diff_list)))
        mae = np.mean(np.abs(diff_list))
        print("{} Stage: Val,                 MAE: {:.2f}, MSE: {:.2f}, Time: {:.2f}".
              format(gettime(), mae, mse, time.time() - epoch_time))

        if mae * 2 + mse < self.best:
            self.best = mae * 2 + mse
            save_path = os.path.join(self.model_save_path, self.start_time, "best_model.pth")
            torch.save(self.model.state_dict(), save_path)
            print("{} New best model - Epoch {}, MAE: {:.2f}, MSE: {:.2f}".
                  format(gettime(), epoch, mae, mse))

    @staticmethod
    def testOneImage(modelPath, imgPath):
        torch.backends.cudnn.benchmark = True
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = os.path.join(modelPath, "best_model.pth")

        img = F.to_tensor(Image.open(imgPath).convert("RGB"))
        img = img.unsqueeze(0)

        model = vgg19()
        model.to(device)
        model.load_state_dict(torch.load(model_path), device)

        img = img.to(device)
        output = model(img)
        print("\nObliczona ilośc osób")
        print("{:.2f}".format(output.sum().item()))
