import os
import shutil


target_dir = "../imageset/Cityscapes/image"


def copy_dir(data_dir):
    sub_dir = os.listdir(data_dir)
    for dir1 in sub_dir:  # 次级目录遍历
        data_dir_1 = data_dir + '/' + dir1
        for dir2 in os.listdir(data_dir_1):  # dir3 is the name for image
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            # if "color" in dir2:
            shutil.copy(data_dir_1 + '/' + dir2, target_dir + f'/{dir2}')


if __name__ == '__main__':
    copy_dir("../imageset/Cityscapes/val")
