import requests
import shutil
import zipfile
import os


def download_new_gbs_dataset():

    # Define the path to the zip file and the destination directory
    zip_file_path = 'vgbs.zip'
    extract_to_directory = './vgbs'

    # remove files if they exist
    try:
        os.remove(zip_file_path)
    except FileNotFoundError:
        pass
    try:
        for fp in os.listdir("./vgbs"):
            os.remove("./vgbs/" + fp)
        os.rmdir("./vgbs")
    except OSError:
        pass

    # download zip
    url = "https://gitlab.in2p3.fr/korol/lisa-verification-binaries/-/jobs/artifacts/master/download?job=make+par+files"
    response = requests.get(url)

    filename = "vgbs.zip"
    with open(filename, 'wb') as f:
        f.write(response.content)

    
    # Ensure the destination directory exists
    os.makedirs(extract_to_directory, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            zf.extractall(extract_to_directory)
            print(f"Successfully unzipped '{zip_file_path}' to '{extract_to_directory}'")
    except FileNotFoundError:
        print(f"Error: Zip file not found at '{zip_file_path}'")
    except zipfile.BadZipFile:
        print(f"Error: '{zip_file_path}' is not a valid ZIP file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    os.remove(zip_file_path)


if __name__ == "__main__":
    download_new_gbs_dataset()