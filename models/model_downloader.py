import os
import tarfile
import wget

def model_download(url, output_dir):
    """
    Downloads a model from the specified URL and extracts it to the given output directory.

    Parameters:
    - url (str): The URL of the model to download.
    - output_dir (str): The directory where the model will be saved and extracted.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the filename from the URL
    filename = url.split("/")[-1]
    filepath = os.path.join(output_dir, filename)  # Full path for the downloaded model
    
    # Check if the model already exists
    if not os.path.exists(filepath):
        print(f"Downloading {filename} ...")
        wget.download(url, out=filepath)  # Download the model
        print(f"\nExtracting {filename} to {output_dir}")
        
        # Extract the downloaded tar file
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=output_dir)  # Extract to output directory
            
        print("The model is READY!")
    else:
        print(f"The model {filename} is already in dir {output_dir}")

if __name__ == '__main__':
    # URL of the model to be downloaded
    model_url = "http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz"
    # Call the model_download function
    model_download(model_url, "./MyModelHub")
