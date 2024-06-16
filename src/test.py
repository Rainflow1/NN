from Trainer import *
from argparse import *
import torch
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = ArgumentParser(description="Training")
    parser.add_argument("--dataset", default="./dataset", help="preprocessed dataset path")
    parser.add_argument("--model", default="./checkpoint/24-04-29_22-03", help="model path")

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    for k, v in args.__dict__.items():
        print("Argument: {} -> {}".format(k, v))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(args.model, "best_model.pth")

    test_set = UCFDataset(dataset_path=os.path.join(args.dataset, "test"), mode="test")

    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, pin_memory=True,
                             num_workers=args.workers)

    model = vgg19()
    model.to(device)
    model.load_state_dict(torch.load(model_path), device)
    diff_list = []

    for i, (inputs, count, gt) in enumerate(test_loader):
        inputs = inputs.to(device)
        count = count.to(device)
        gt = gt.to(device)
        with torch.no_grad():
            output = model(inputs)
            diff = count[0].item() - output.sum().item()

            diff_list.append(diff)

            if i == 11: break

            if i < 10:
                plt.figure(0)
                plt.imshow(gt.detach().cpu().squeeze().numpy())
                plt.title("GT")
                plt.figure(1)
                plt.imshow(output.detach().cpu().squeeze().numpy())
                plt.title("Generated")
                plt.show()
                print("IMG: {}, GT: {}, PRED: {}, DIFF: {}".format(i, count[0].item(), output.sum().item(), count[0].item() - output.sum().item()))

    diff_list = np.array(diff_list)
    mse = np.sqrt(np.mean(np.square(diff_list)))
    mae = np.mean(np.abs(diff_list))
    print("Test Results  ->  MAE: {:.2f}, MSE: {:.2f}".format(mae, mse))
