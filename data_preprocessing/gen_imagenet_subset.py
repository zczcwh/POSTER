import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

data_dir = '/data/ImageNet/CLS-LOC/'

# Data loading code
traindir = os.path.join(data_dir, 'train')
train_dataset = datasets.ImageFolder(traindir, None)
classes = train_dataset.classes
print("the number of total classes: {}".format(len(classes)))

seed = 1993
np.random.seed(seed)
subset_num = 200
subset_classes = np.random.choice(classes, subset_num, replace=False)
print("the number of subset classes: {}".format(len(subset_classes)))
print(subset_classes)

des_root_dir = '/data/seed_{}_subset_{}_imagenet_500/data/'.format(seed, subset_num)
if not os.path.exists(des_root_dir):
    os.makedirs(des_root_dir)
phase_list = ['train', 'val']
for phase in phase_list:
    if not os.path.exists(os.path.join(des_root_dir, phase)):
        os.mkdir(os.path.join(des_root_dir, phase))
    for idx, sc in enumerate(subset_classes):
        print('{}/{} class: {}'.format(idx, subset_num, sc))
        if not os.path.exists(os.path.join(des_root_dir, phase, sc)):
            os.mkdir(os.path.join(des_root_dir, phase,sc))
        src_dir = os.path.join(data_dir, phase, sc)
        imgs = os.listdir(os.path.join(data_dir, phase, sc))
        if phase=='train':
            imgs = np.random.choice(imgs, 500, replace=False)
        for im in imgs:
            src_img = os.path.join(data_dir, phase, sc, im)
            des_dir = os.path.join(des_root_dir, phase, sc, im)
            cmd = "cp -r {} {}".format(src_img, des_dir)
            # print(cmd)
            os.system(cmd)

print("DONE")

# the number of total classes: 1000
# the number of subset classes: 100
# ['n01729322' 'n01514668' 'n04550184' 'n03109150' 'n01990800' 'n02363005'
#  'n12267677' 'n01531178' 'n02123597' 'n03016953' 'n02011460' 'n04153751'
#  'n02096177' 'n09468604' 'n03657121' 'n02101388' 'n02107908' 'n03290653'
#  'n02107574' 'n02786058' 'n02115913' 'n01914609' 'n07614500' 'n04428191'
#  'n03017168' 'n04355933' 'n02951585' 'n03759954' 'n01806567' 'n02112706'
#  'n03085013' 'n03485407' 'n02526121' 'n02091467' 'n03710193' 'n02128385'
#  'n03775071' 'n02102318' 'n04604644' 'n02033041' 'n04409515' 'n02088094'
#  'n03891332' 'n02128757' 'n01984695' 'n02279972' 'n03532672' 'n03250847'
#  'n15075141' 'n03697007' 'n02536864' 'n04074963' 'n03000247' 'n04371430'
#  'n02002556' 'n04399382' 'n03785016' 'n12768682' 'n03527444' 'n03942813'
#  'n04026417' 'n01537544' 'n01775062' 'n02814860' 'n02640242' 'n04442312'
#  'n02120079' 'n03223299' 'n01774384' 'n02917067' 'n04525305' 'n02229544'
#  'n02168699' 'n01744401' 'n07892512' 'n02749479' 'n07831146' 'n04152593'
#  'n03089624' 'n02669723' 'n02950826' 'n03691459' 'n02116738' 'n03967562'
#  'n01882714' 'n03141823' 'n02105056' 'n02094258' 'n02971356' 'n03075370'
#  'n02948072' 'n04591157' 'n03884397' 'n12620546' 'n02280649' 'n04266014'
#  'n02112137' 'n04599235' 'n02840245' 'n01689811']

# 200 classes
# ['n01729322' 'n01514668' 'n04550184' 'n03109150' 'n01990800' 'n02363005'
#  'n12267677' 'n01531178' 'n02123597' 'n03016953' 'n02011460' 'n04153751'
#  'n02096177' 'n09468604' 'n03657121' 'n02101388' 'n02107908' 'n03290653'
#  'n02107574' 'n02786058' 'n02115913' 'n01914609' 'n07614500' 'n04428191'
#  'n03017168' 'n04355933' 'n02951585' 'n03759954' 'n01806567' 'n02112706'
#  'n03085013' 'n03485407' 'n02526121' 'n02091467' 'n03710193' 'n02128385'
#  'n03775071' 'n02102318' 'n04604644' 'n02033041' 'n04409515' 'n02088094'
#  'n03891332' 'n02128757' 'n01984695' 'n02279972' 'n03532672' 'n03250847'
#  'n15075141' 'n03697007' 'n02536864' 'n04074963' 'n03000247' 'n04371430'
#  'n02002556' 'n04399382' 'n03785016' 'n12768682' 'n03527444' 'n03942813'
#  'n04026417' 'n01537544' 'n01775062' 'n02814860' 'n02640242' 'n04442312'
#  'n02120079' 'n03223299' 'n01774384' 'n02917067' 'n04525305' 'n02229544'
#  'n02168699' 'n01744401' 'n07892512' 'n02749479' 'n07831146' 'n04152593'
#  'n03089624' 'n02669723' 'n02950826' 'n03691459' 'n02116738' 'n03967562'
#  'n01882714' 'n03141823' 'n02105056' 'n02094258' 'n02971356' 'n03075370'
#  'n02948072' 'n04591157' 'n03884397' 'n12620546' 'n02280649' 'n04266014'
#  'n02112137' 'n04599235' 'n02840245' 'n01689811' 'n04099969' 'n04069434'
#  'n01755581' 'n01667778' 'n01496331' 'n01641577' 'n02093991' 'n04235860'
#  'n02087394' 'n03998194' 'n07753592' 'n02916936' 'n03207743' 'n03476991'
#  'n02077923' 'n01871265' 'n04554684' 'n02454379' 'n04286575' 'n03642806'
#  'n04325704' 'n02086079' 'n07760859' 'n02091831' 'n03482405' 'n12057211'
#  'n02091134' 'n02104365' 'n02895154' 'n01833805' 'n02028035' 'n03649909'
#  'n03775546' 'n02444819' 'n02017213' 'n03450230' 'n02088466' 'n02795169'
#  'n01770081' 'n02105251' 'n07584110' 'n03498962' 'n02361337' 'n02747177'
#  'n03970156' 'n04044716' 'n02493793' 'n04264628' 'n01768244' 'n04557648'
#  'n02219486' 'n01980166' 'n04039381' 'n01829413' 'n01677366' 'n02281787'
#  'n03218198' 'n04590129' 'n02843684' 'n02410509' 'n01704323' 'n03623198'
#  'n03534580' 'n02105641' 'n07734744' 'n02412080' 'n02102480' 'n07745940'
#  'n09246464' 'n01773549' 'n04606251' 'n03134739' 'n02096294' 'n04040759'
#  'n07248320' 'n02111500' 'n07768694' 'n03868863' 'n02086240' 'n04310018'
#  'n03476684' 'n07742313' 'n01828970' 'n03461385' 'n01819313' 'n02492660'
#  'n02606052' 'n01749939' 'n04589890' 'n02090622' 'n02101556' 'n03874293'
#  'n01873310' 'n02727426' 'n02488291' 'n03124170' 'n02981792' 'n03661043'
#  'n02708093' 'n02086646']
