import os
from PIL import Image

def main():
    for path_1, folders_1, files_1 in os.walk('data'):
        for path_2, folders_2, files_2 in os.walk(path_1):
            for filename in files_2:
                if filename.endswith('.jpg'):
                    path = os.path.join(path_2, filename)
                    try:
                        img = Image.open(path) # open the image file
                        img.verify() # verify that it is, in fact an image
                    except (IOError) as e:
                        print('Bad file:', path)

if __name__ == "__main__":
    main()