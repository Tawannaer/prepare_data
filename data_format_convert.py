
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File  : datafomat_convert.py
@Author: Lijing
@Date  : 2021/8/6 16:57
@Desc  :  convert 3D nifti data format to 2d rgb image

'''
'''
function:将.nii.gz格式的三维mask数据转换为二维图像和带标注的检测框的.xml格式（voc数据集格式）
'''

import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import SimpleITK as sitk
from tqdm import  tqdm
import cv2
from  skimage import measure
import copy


def indent(elem, level=0):
    '''
    add enter to xml file
    '''
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def write_xml(imgname, filepath, labeldicts):
    root = ET.Element('Annotation')
    ET.SubElement(root, 'filename').text = str(imgname)
    sizes = ET.SubElement(root, 'size')

    for labeldict in labeldicts:
        ET.SubElement(sizes, 'width').text = str(int(labeldict['weight']))
        ET.SubElement(sizes, 'height').text = str(int(labeldict['height']))
        ET.SubElement(sizes, 'depth').text = '3'
        objects = ET.SubElement(root, 'object')  # 创建object子节点
        ET.SubElement(objects, 'name').text = labeldict['name']  # BDD100K_10.names文件中 # 的类别名
        ET.SubElement(objects, 'pose').text = 'Unspecified'
        ET.SubElement(objects, 'truncated').text = '0'
        ET.SubElement(objects, 'difficult').text = '0'
        bndbox = ET.SubElement(objects, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(labeldict['xmin']))
        ET.SubElement(bndbox, 'ymin').text = str(int(labeldict['ymin']))
        ET.SubElement(bndbox, 'xmax').text = str(int(labeldict['xmax']))
        ET.SubElement(bndbox, 'ymax').text = str(int(labeldict['ymax']))
    indent(root,0)
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding='utf-8')


# param: bounding box,image_array,img_name
def save_xml(img_name, img_array, bbox,label_name_list):
    labeldicts = []
    for index,bounding_box in  enumerate(bbox):
        sh, sw = img_array.shape[0], img_array.shape[1]
        new_dict = {'name': label_name_list[index],
                    'difficult': '0',
                    'height': sh,
                    'weight': sw,
                    'xmin': str(bounding_box[1]-1),
                    'ymin': str(bounding_box[0]-1),
                    'xmax': str(bounding_box[3]+1),
                    'ymax': str(bounding_box[2]+1),

                    }
        labeldicts.append(new_dict)
    write_xml(img_name, img_name, labeldicts)


def generator_channel(image_slice,lower,upper):
    image_slice_copy = copy.deepcopy(image_slice)
    image_slice_copy[image_slice_copy > upper] = upper
    image_slice_copy[image_slice_copy< lower] = lower
    img_normalize = (((image_slice_copy - lower) / (upper - lower)) * 255).astype(np.uint8)
    return img_normalize

def main():
    image_data = './val'
    image_label = './val'
    save_dir = './example/val_pngimage'
    save_xml_dir = './example/val_annotation'

    label_list = [
    "background",
    "nodule",
    "patches",
    "strip",
    "grid",
    "spherical",
    "empty",
    "cavity",
    "pleuralEffusion"]


    label_count = {"background":0,
    "patches":0,
    "strip":0,
    "grid":0,
    "spherical":0,
    "empty":0,
    "cavity":0,
    "pleuralEffusion":0}
    _COLORS = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
        ]
    ).astype(np.float32).reshape(-1, 3)



    for file in tqdm(os.listdir(image_data)):
        label_dict = []
        if file.startswith('img-'):
            image = sitk.ReadImage(os.path.join(image_data,file))
            image_array = sitk.GetArrayFromImage(image)
            label = sitk.ReadImage(os.path.join(image_label,file.replace('img-','label-')))
            label_array = sitk.GetArrayFromImage(label)
            label_array[label_array == 1] = 0
            # if np.max(label_array) ==1:
            #     print(file)

            for index in range(image_array.shape[0]):
                if np.max(label_array[index]) > 0:
                    bounding_box = []
                    label_name_list = []

                    image_slice = image_array[index]
                    img_channel2 = generator_channel(image_slice, -1000, 400)  # 肺窗
                    img_channel3 = generator_channel(image_slice, -160, 240)  # 高衰减
                    img_channel1 = generator_channel(image_slice, -1400, -600)  # 低衰减
                    img_normalize = np.dstack((img_channel1,img_channel2,img_channel3))
                    if (img_channel1 == img_channel3).all():
                        print('true')
                    label_slice = label_array[index]
                    label = measure.label(label_slice,connectivity=2)
                    region = measure.regionprops(label)

                    for i in range(len(region)):
                        bounding = region[i].bbox
                        bounding_box.append(bounding)
                        center = region[i].coords[0]
                        label_name_list.append(label_list[label_array[index,int(center[0]),int(center[1])]])
                        label_count[label_list[label_array[index,int(center[0]),int(center[1])]]] = \
                            label_count[label_list[label_array[index,int(center[0]),int(center[1])]]] + 1
                        # color = (_COLORS[label_array[index,int(center[0]),int(center[1])]] * 255).astype(np.uint8).tolist()
                        # cv2.rectangle(img_normalize,(bounding[1],bounding[0]),(bounding[3],bounding[2]),color =color,thickness=2)
                    cv2.imwrite(os.path.join(save_dir,file.split('.')[0]+str('-')+str(index)+'.png'),img_normalize)

                    # cv2.imshow('xx',img_normalize)
                    # cv2.waitKey(0)
                    save_xml(os.path.join(save_xml_dir,file.split('.')[0]+str('-')+str(index)+'.xml'),img_normalize,bounding_box,label_name_list)


if __name__ =='__main__':
    main()