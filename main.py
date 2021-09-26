import PIL.ImageChops
import numpy as np

from engine import train_one_epoch, evaluate
import utils
import torch
from dataset import PennFudanDataset, get_model_instance_segmentation
from transformation import get_transform
from PIL import Image
from torchvision import transforms
import os
import cv2
import numpy

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print('GPU')
    else:
        print('netko nekog ovdje levati')
    num_classes = 2  # 3 + background
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 30
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
        torch.save(model, './models/SSIP/model' + str(epoch) + '.pt')

    print("That's it!")

    img, _ = dataset_test[0]
    print(dataset_test)
    model.eval()

    torch.cuda.empty_cache()


def koristi_postojeci():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print('GPU')
    else:
        print('netko nekog ovdje leva')

    model = torch.load('./models/testing/model44.pt')
    model.to(device)
    model.eval()

    br = 0
    pic_source = os.path.join(os.getcwd(), "PennFudanPed", "NotUsed")
    for sp in os.listdir(pic_source):
        im = Image.open(os.path.join(pic_source, sp))
        i = transforms.ToTensor()(im).unsqueeze_(0)
        with torch.no_grad():
            prediction = model([i[0].to(device)])
        img = Image.fromarray(i[0].mul(255).permute(1, 2, 0).byte().numpy()).convert('RGB')
        open_cv_image = numpy.array(img)
        img = img.convert('1')
        pilmulti = PIL.Image.new('1', img.size, 0)

        open_cv_image = open_cv_image[:, :, ::-1].copy()

        multi = np.zeros(open_cv_image.shape, np.uint8)
        cv2.imwrite("./output/aCV" + str(br) + ".png", open_cv_image)
        img.save("./output/aPIL" + str(br) + ".png", 'png')

        for j in range(prediction[0]['masks'].shape[0]):
            pilslika = Image.fromarray(prediction[0]['masks'][j, 0].mul(255).byte().cpu().numpy())
            maskica = numpy.array(pilslika)
            pilslika = pilslika.convert('1')
            ret, maskica = cv2.threshold(maskica, 150, 255, cv2.THRESH_BINARY)

            nemaska = cv2.bitwise_not(maskica)
            #ret, nemaska = cv2.threshold(nemaska, 254, 255, cv2.THRESH_BINARY)
            multi = cv2.bitwise_and(multi, multi, mask=nemaska)

            maska = cv2.bitwise_and(open_cv_image, open_cv_image, mask=maskica)
            multi = cv2.add(multi, maska)

            pilmulti = PIL.ImageChops.lighter(pilmulti, PIL.ImageChops.logical_and(pilslika, img))
        cv2.imwrite("./output/cv" + str(br) + ".png", multi)
        pilmulti.save("./output/pil" + str(br) + ".png", 'png')
        br += 1

    torch.cuda.empty_cache()


if __name__ == '__main__':
    #main()
    koristi_postojeci()
