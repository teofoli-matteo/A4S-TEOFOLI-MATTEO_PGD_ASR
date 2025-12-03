import os
import urllib.request
import zipfile

def download_and_extract_tinyimagenet(dest_dir="tiny-imagenet-200"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = "tiny-imagenet-200.zip"

    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("The zip file already exists, download skipped.")

    if not os.path.exists(dest_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete.")
    else:
        print("The extracted file already exists, extraction skipped.")

if __name__ == "__main__":
    download_and_extract_tinyimagenet()
