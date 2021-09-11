
import SimpleITK as sitk
import os
import numpy as np
import math
from config  import  cfg




# fucntion : Z轴的spacing固定到2，阈值是-1000-400 ,normalization 0-255
def resample(image, origin_sapcing, target_spacing, resamplemethod=sitk.sitkNearestNeighbor):
    '''

    :param image:
    :param origin_sapcing:
    :param target_spacing:
    :param resamplemethod:
    :return:
    '''
    newsize = []
    imagesize = image.GetSize()
  #  imagesize = imagesize[2],imagesize[1],imagesize[0]
    for i in range(3):
        newsize.append(int(math.ceil(imagesize[i] * origin_sapcing[i] / target_spacing[i])))
    resample_filter = sitk.ResampleImageFilter()
    if resamplemethod == sitk.sitkNearestNeighbor:
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    if resamplemethod == sitk.sitkLinear:
        resample_filter.SetInterpolator(sitk.sitkLinear)

    resample_filter.SetSize(tuple(newsize))
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetOutputOrigin(image.GetOrigin())
    image = resample_filter.Execute(image)

    return image

def resample_new(image, origin_sapcing, target_spacing, resamplemethod=sitk.sitkNearestNeighbor):
    '''

    :param image:
    :param origin_sapcing:
    :param target_spacing:
    :param resamplemethod:
    :return:
    '''
    newsize = []
    imagesize = image.GetSize()
   # imagesize = imagesize[2], imagesize[1], imagesize[0]
    origin_sapcing = origin_sapcing[2],origin_sapcing[1],origin_sapcing[0]
    target_spacing = target_spacing[2],target_spacing[1],target_spacing[0]

    for i in range(3):
        newsize.append(int(np.round((imagesize[i] * origin_sapcing[i] / target_spacing[i]))))
    resample_filter = sitk.ResampleImageFilter()
    newsize = [512,512,299]
    if resamplemethod == sitk.sitkNearestNeighbor:
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    if resamplemethod == sitk.sitkLinear:
        resample_filter.SetInterpolator(sitk.sitkLinear)

    resample_filter.SetSize(tuple(newsize))
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetOutputOrigin(image.GetOrigin())
    image = resample_filter.Execute(image)
    # image_array = sitk.GetArrayFromImage(image)
    # image_array = image_array.transpose([2,1,0])
    #image_array_4d = image_array[np.newaxis, :]

    return image

def image_threshold(srcitkimage, lower, upper):
        '''
        对图像阈值化处理，设置upper，lower，大于upper的设置为uppper,小于lower的设置为lower


        '''
        # srcitkimage = sitk.ReadImage(filename)
        sitkimagearray = sitk.GetArrayFromImage(srcitkimage)
        sitkimagearray[sitkimagearray > upper] = upper
        sitkimagearray[sitkimagearray < lower] = lower
        sitkstructimage = sitk.GetImageFromArray(sitkimagearray)
        origin = srcitkimage.GetOrigin()
        spacing = srcitkimage.GetSpacing()
        direction = srcitkimage.GetDirection()
        sitkstructimage.SetOrigin(origin)
        sitkstructimage.SetSpacing(spacing)
        sitkstructimage.SetDirection(direction)

        return sitkstructimage


def normalization(ct):
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    image = resacleFilter.Execute(ct)
    return image


if __name__ == '__main__':
    file_path  = 'C:/Users/Administrator/Desktop/PneumoniaDataPre/wzz'
    save_path = 'C:/Users/Administrator/Desktop/PneumoniaDataPre/resample_data'
    for file in os.listdir(file_path):
        data = sitk.ReadImage(os.path.join(file_path,file))
        origin_spacing = data.GetSpacing()
        target_spcing= [origin_spacing[0],origin_spacing[1],2]
        image_ = resample(image = data,origin_sapcing=origin_spacing,target_spacing=target_spcing,resamplemethod=sitk.sitkLinear)
        image_th = image_threshold(image_,lower=-1000,upper = 400)
        #image_th = image_threshold(data,lower= -1000,upper = 400)
        image_n = normalization(image_th)
        sitk.WriteImage(image_n,os.path.join(save_path,file.split('_')[0] + '_0000.nii.gz'))