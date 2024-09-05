import os
import shutil


# Function to copy images from source to destination
def copy_images(source_folder, destination_folder):
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        if os.path.isfile(source_file):  # Ensure it's a file
            shutil.copy(source_file, destination_folder)
            # print(f'Copied {source_file} to {destination_folder}')

def copyfolder(fake,real,new):
    
    # Define the paths
    source_folder_fake = 'D:\Deep_Fake\DataSet\Train\Fake'
    source_folder_real = 'D:\Deep_Fake\DataSet\Train\Real'
    destination_folder = 'D:\Deep_Fake\DataSet\Train_Mix'

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)



    # Copy images from both "fake" and "real" folders
    copy_images(source_folder_fake, destination_folder)
    print('Successfully Copied Fake Images')
    copy_images(source_folder_real, destination_folder)
    print('Successfully Copied Real Images')

    print('All images have been copied to the "abc" folder.')



