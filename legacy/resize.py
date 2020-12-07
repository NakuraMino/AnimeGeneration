import cv2
import os

def resize_images(path, save_path):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path,filename))
        if img is not None:
            resized = cv2.resize(img, (64,64))
            cv2.imwrite(save_path + '/' + filename, resized)


base_path = 'C:\\Users\\minon\\Desktop\\AnimeGeneration\\dataset\\'

train_path = base_path + 'train_photo'
save_train_path = base_path + 'train_photo_small'
shinkai_smooth_path = base_path + 'Shinkai\\smooth'
save_shinkai_smooth_path = base_path + 'Shinkai_small\\smooth'
shinkai_style_path = base_path + 'Shinkai\\style'
save_shinkai_style_path = base_path + 'Shinkai_small\\style'

resize_images(train_path, save_train_path)
print("done train")
resize_images(shinkai_smooth_path, save_shinkai_smooth_path)
print("done smooth")
resize_images(shinkai_style_path, save_shinkai_style_path)
print("done shinkai")