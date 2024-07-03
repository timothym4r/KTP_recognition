import os
import shutil
from sklearn.model_selection import train_test_split 

source_folders = ['data\\non_ktp\\Aadhaar', 'data\\non_ktp\\Driving Licence', 'data\\non_ktp\\PAN', "data\\non_ktp\\Passport", "data\\non_ktp\\Utility", "data\\non_ktp\\Voter ID"]

output_dir = 'non_ktp'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

def main():
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for folder in source_folders:

        images = os.listdir(folder)
        train_images, test_images = train_test_split(images, test_size = 0.25, random_state=42)
        test_images, val_images = train_test_split(test_images, test_size=0.4, random_state=42)

        for image in train_images:
            file_name = image
            if os.path.exists(os.path.join(train_dir, image)):
                base, ext = os.path.splitext(image)
                counter = 1
                file_name = f"{base}_{counter}{ext}"
                while os.path.exists(file_name):
                    counter+=1
                    file_name = f"{base}_{counter}{ext}"
            shutil.copy(os.path.join(folder, image), os.path.join(train_dir, file_name))
        
        for image in val_images:
            file_name = image
            if os.path.exists(os.path.join(val_dir, image)):
                base, ext = os.path.splitext(image)
                counter = 1
                file_name = f"{base}_{counter}{ext}"
                while os.path.exists(file_name):
                    counter+=1
                    file_name = f"{base}_{counter}{ext}"
            shutil.copy(os.path.join(folder, image), os.path.join(val_dir, file_name))
        
        for image in test_images:
            file_name = image
            if os.path.exists(os.path.join(test_dir, image)):
                base, ext = os.path.splitext(image)
                counter = 1
                file_name = f"{base}_{counter}{ext}"
                while os.path.exists(file_name):
                    counter+=1
                    file_name = f"{base}_{counter}{ext}"
            shutil.copy(os.path.join(folder, image), os.path.join(test_dir, file_name))

if __name__ == "__main__":
    main()

