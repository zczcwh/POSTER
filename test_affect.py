import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torch
import os
import argparse
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class

from utils import *
from models.emotion_hyp_affect import pyramid_trans_expr
from sklearn.metrics import confusion_matrix
from data_preprocessing.plot_confusion_matrix import plot_confusion_matrix



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='affectnet', help='dataset')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='Pytorch checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--modeltype', type=str, default='large', help='small or base or large')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--gpu', type=str, default='0', help='assign multi-gpus by comma concat')
    parser.add_argument('-p', '--plot_cm', action="store_true", help="Ploting confusion matrix.")
    return parser.parse_args()

def test():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    data_transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    num_classes = 7
    if args.dataset == "rafdb":
        datapath = './data/raf-basic/'
        num_classes = 7
        test_dataset = RafDataSet(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet":
        datapath = './data/AffectNet/'
        num_classes = 7
        test_dataset = Affectdataset(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    elif args.dataset == "affectnet8class":
        datapath = './data/AffectNet/'
        num_classes = 8
        test_dataset = Affectdataset_8class(datapath, train=False, transform=data_transforms_test)
        model = pyramid_trans_expr(img_size=224, num_classes=num_classes, type=args.modeltype)

    else:
        return print('dataset name is not correct')


    print("Loading pretrained weights...", args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    checkpoint = checkpoint["model_state_dict"]
    model = load_pretrained_weights(model, checkpoint)

    test_size = test_dataset.__len__()
    print('Test set size:', test_size)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    model = model.cuda()


    pre_labels = []
    gt_labels = []
    with torch.no_grad():
        bingo_cnt = 0
        model.eval()
        for batch_i, (imgs, targets) in enumerate(test_loader):
            outputs, features = model(imgs.cuda())
            targets = targets.cuda()
            _, predicts = torch.max(outputs, 1)
            _, predicts = torch.max(outputs, 1)
            correct_or_not = torch.eq(predicts, targets)
            bingo_cnt += correct_or_not.sum().cpu()
            pre_labels += predicts.cpu().tolist()
            gt_labels += targets.cpu().tolist()


        acc = bingo_cnt.float() / float(test_size)
        acc = np.around(acc.numpy(), 4)
        print(f"Test accuracy: {acc:.4f}.")
        cm = confusion_matrix(gt_labels, pre_labels)
        # print(cm)

    if args.plot_cm:
        cm = confusion_matrix(gt_labels, pre_labels)
        cm = np.array(cm)
        if args.dataset == "rafdb":
            labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]  #
            plot_confusion_matrix(cm, labels_name, 'RAF-DB', acc)

        if args.dataset == "affectnet":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN"]  #
            plot_confusion_matrix(cm, labels_name, 'AffectNet', acc)

        if args.dataset == "affectnet8class":
            labels_name = ['NE', 'HA', 'SA', 'SU', 'FE', 'DI', "AN", "CO"]  #
            # 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,
            # 7: Contempt,
            plot_confusion_matrix(cm, labels_name, 'AffectNet_8class', acc)




if __name__ == "__main__":                    
    test()

