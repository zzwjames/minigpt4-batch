import zipfile

def unzip_file(zip_filepath, dest_path):
    # Open the zip file in read mode
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        # Extract all the contents of the zip file in the destination directory
        zip_ref.extractall(dest_path)

# Call the function
unzip_file('./models.zip', './')