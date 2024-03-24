import csv
import random
from itertools import islice

import scipy
from torch.utils.data import Dataset
import torchvision.transforms.functional as T
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from PIL import Image

class UCF_CC_50_Dataset:

            # rozmiar moznabyc zmienic na 192x192 zeby bylo podzielne przez 16 co moze czasami sie przydac
    def __init__(self, datasetPath, tempPath, maxSize = (200, 200), seed = 123, norm = 10000, kernel = 9):
        self.norm = norm
        self.kernel = kernel
        self.rng = random.Random(seed)
        self.maxSize = maxSize

        num = 0
        files = []

        for file in os.listdir(datasetPath):
            if os.path.isfile(os.path.join(datasetPath, file)) and file.endswith('.jpg'):
                num += 1
                files.append(file)


        trainNum = round(0.4 * num)
        testNum = round(0.4 * num)
        validNum = round(0.2 * num)

        self.rng.shuffle(files)

        dirs = ['train', 'valid', 'test']
        dirsNum = [trainNum, validNum, testNum]
        iterFiles = iter(files)
        files = [list(islice(iterFiles, i)) for i in dirsNum]

        for n, dir in enumerate(dirs):
            dirPath = os.path.join(tempPath, dir)

            if not os.path.exists(dirPath) and not os.path.isdir(dirPath):
                os.makedirs(dirPath)

            for file in os.listdir(dirPath):
                os.remove(os.path.join(dirPath, file))

            if dir == 'test':
                for file in files[n]:
                    with Image.open(os.path.join(datasetPath, file)) as im:
                        im.save(os.path.join(dirPath, f'{os.path.splitext(file)[0]}_d.jpg'))

                        points = loadmat(os.path.join(datasetPath, f'{os.path.splitext(file)[0]}_ann.mat'))['annPoints']
                        points = [(int(x), int(y)) for x, y in points if int(x) < im.size[0] and int(y) < im.size[1]]

                        target = np.zeros(im.size[::-1], dtype=np.float32)
                        for x, y in points:
                            target[y][x] = self.norm

                        target = gaussian_filter(target, sigma=(self.kernel, self.kernel), order=0)
                        assert round(np.sum(target)/self.norm) == len(points)
                        target = Image.fromarray(target, 'F').convert("L")
                        target.save(os.path.join(dirPath, f'{os.path.splitext(file)[0]}_gt.jpg'))

                        with open(os.path.join(dirPath, f'{os.path.splitext(file)[0]}_p.csv'), 'x', newline='') as csvFile:
                            writer = csv.writer(csvFile)
                            writer.writerows(points)

            else:
                for file in files[n]:
                    self.processFile(os.path.join(datasetPath, file), dirPath)

        pass

    def processFile(self, filePath, dirPath):

        file = os.path.basename(filePath)
        path = os.path.dirname(filePath)
        name = os.path.splitext(file)[0]

        with Image.open(filePath) as im:

            points = loadmat(os.path.join(path, f'{name}_ann.mat'))['annPoints']
            points = [(int(x), int(y)) for x, y in points if int(x) < im.size[0] and int(y) < im.size[1]]
            
            fragments: list(tuple(int, int, int, int)) = []

            ratio = (im.size[0] // self.maxSize[0]) * (im.size[1] // self.maxSize[1])
            tries = ratio * 10
            for _ in range(tries):
                boxX: int = self.rng.randrange(0, im.size[0] - self.maxSize[0])
                boxY: int = self.rng.randrange(0, im.size[1] - self.maxSize[1])
                box: tuple(int, int, int, int) = (boxX, boxY, boxX + self.maxSize[0], boxY + self.maxSize[1])
                
                if self.intersectionAreaRatio(box, fragments) < 0.3 and len(self.pointsInBox(box, points))/len(points) > 0.5 * (1/ratio):
                    fragments.append(box)

            for i, frag in enumerate(fragments):
                reg = im.crop(frag)
                reg.save(os.path.join(dirPath, f'{name}_{i+1}_d.jpg'))

                target = np.zeros(reg.size[::-1], dtype=np.float32)
                targetPoints = self.pointsInBox(frag, points)
                for x, y in targetPoints:
                    target[y - frag[1]][x - frag[0]] = self.norm

                #target = self.gaussian_filter_density(reg, target)
                target = gaussian_filter(target, sigma=(self.kernel, self.kernel), order=0)
                assert round(np.sum(target)/self.norm) == len(targetPoints)
                target = Image.fromarray(target, 'F').convert("L")
                target.save(os.path.join(dirPath, f'{name}_{i+1}_gt.jpg'))

                with open(os.path.join(dirPath, f'{name}_{i+1}_p.csv'), 'x', newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(targetPoints)
        pass

    def gaussian_filter_density(self, img, points):
        '''
        This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

        points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
        img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

        return:
        density: the density-map we want. Same shape as input image but only has one channel.

        example:
        points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
        img_shape: (768,1024) 768 is row and 1024 is column.
        '''
        img_shape = [img.size[0], img.size[1]]
        print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
        density = np.zeros(img_shape, dtype=np.float32)
        gt_count = len(points)
        if gt_count == 0:
            return density

        leafsize = 2048
        # build kdtree
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(points, k=4)

        print('generate density...')
        for i, pt in enumerate(points):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
                pt2d[int(pt[1]), int(pt[0])] = 1.
            else:
                continue
            if gt_count > 1:
                sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            else:
                sigma = np.average(np.array(img.size)) / 2. / 2.  # case: 1 point
            density += gaussian_filter(pt2d, sigma, mode='constant')
        print('done.')
        return density

    def intersectionAreaRatio(self, box, fragments):
        ret = 0

        for frag in fragments:
            intersectionArea = max(0, min(box[2], frag[2]) - max(box[0], frag[0])) * max(0, min(box[3], frag[3]) - max(box[1], frag[1]))
            fragArea = (frag[2] - frag[0]) * (frag[3] - frag[1])
            boxArea = (box[2] - box[0]) * (box[3] - box[1])
            area = boxArea + fragArea - intersectionArea
            ret = max(ret, intersectionArea / area)

        return ret
    
    def pointsInBox(self, box, points):
        ret = []

        for x, y in points:
            if box[0] <= x < box[2] and box[1] <= y < box[3]:
                ret.append((x, y))

        return ret

    def __getitem__(self, i):
        
        match i:
            case 'test':
                return CCDataset("../temp/UCF_CC_50/test", self.rng)
            case 'train':
                return CCDataset("../temp/UCF_CC_50/train", self.rng)
            case 'valid':
                return CCDataset("../temp/UCF_CC_50/valid", self.rng)
            case _:
                return CCDataset("../temp/UCF_CC_50/train", self.rng)

    pass


class CCDataset(Dataset):

    def __init__(self, datasetPath, rng):

        self.dir = datasetPath
        self.num = 0
        self.files = []

        for file in os.listdir(datasetPath):
            if os.path.isfile(os.path.join(datasetPath, file)) and file.endswith('_d.jpg'):
                self.num += 1
                self.files.append(file[:-6])

        for file in self.files:
            assert os.path.exists(os.path.join(datasetPath, f'{file}_p.csv'))
            assert os.path.exists(os.path.join(datasetPath, f'{file}_d.jpg'))
            assert os.path.exists(os.path.join(datasetPath, f'{file}_gt.jpg'))

        rng.shuffle(self.files)

        pass

    def __len__(self):
        return self.num

    def __getitem__(self, i):

        inputName = f'{self.files[i]}_d.jpg'
        targetName = f'{self.files[i]}_gt.jpg'
        pointsName = f'{self.files[i]}_p.csv'

        inputPath = os.path.join(self.dir, inputName)
        targetPath = os.path.join(self.dir, targetName)
        pointsPath = os.path.join(self.dir, pointsName)

        image = Image.open(inputPath)
        target = Image.open(targetPath).convert("L")
        points = None
        with open(pointsPath, 'r', newline='') as csvFile:
            reader = csv.reader(csvFile, quoting=csv.QUOTE_NONNUMERIC)
            points = [(int(x), int(y)) for x, y in reader]

        #plt.figure(1)
        #plt.imshow(image)
        #plt.figure(2)
        #plt.imshow(target)
        #plt.show()
        #print(f'{len(points)}, {np.sum(np.array(target))}')

        image = T.to_tensor(T.to_grayscale(image, 3))
        target = T.to_tensor(target)

        return image, target, len(points)
    pass


if __name__ == "__main__":
    
    dataset = UCF_CC_50_Dataset("./dataset/UCF_CC_50/", "./temp/dataset/UCF_CC_50/")
    ds = dataset["test"]
    print(ds[0])

    """
    gt_dir = "./dataset/UCF_CC_50/2_ann.mat"
    img_dir = "./dataset/UCF_CC_50/2.jpg"

    img = Image.open(img_dir)

    gt = loadmat(gt_dir)['annPoints']

    label = np.zeros(img.size[::-1], dtype=np.float32)

    n = 0

    for x, y in gt:
        if int(x) < img.size[0] and int(y) < img.size[1]:
            if label[int(y)][int(x)] != 0:
                print("lalalla")
            label[int(y)][int(x)] = 1000
            n += 1

    label = gaussian_filter(label, sigma=(4, 4), order=0)

    gt_img = Image.fromarray(label, 'F').convert("L")

    gt_img.save("./test.png")

    print(n)
    print(np.sum(label)/1000)

    plt.imsave("test.png", img, cmap='gray');
    plt.imsave("test1.png", gt_img, cmap='inferno');
    """

    pass