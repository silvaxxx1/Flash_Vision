import os
import tqdm
import argparse
import requests

def load_images(url_file, output_dir):
    """
    Download images from URLs listed in a file and save them to a specified directory.

    Args:
        url_file (str): Path to the text file containing image URLs.
        output_dir (str): Directory where images will be saved.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Read URLs from the specified file
    with open(url_file, 'r') as file:
        urls = file.readlines()
    
    # Iterate over each URL to download the image
    for url in tqdm.tqdm(urls, desc="Loading images..."):
        url = url.strip()  # Remove leading/trailing whitespace
        if not url:
            continue  # Skip empty lines

        try:
            # Send a GET request to fetch the image
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            
            # Construct the image file path
            image_name = os.path.join(output_dir, url.split('/')[-1])
            
            # Save the image to the specified directory
            with open(image_name, 'wb') as img:
                img.write(response.content)

        except Exception as e:
            # Print an error message if the download fails
            print(f"Error loading URL {url}: {e}")    

def main():
    """
    Main function to parse command line arguments and initiate the image download process.
    """
    parser = argparse.ArgumentParser(description="Download images from a list of URLs.")
    parser.add_argument('-u', '--url_file', type=str, required=True,
                        help='Path to the text file containing image URLs.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory where images will be saved.')
    
    args = parser.parse_args()  # Parse the command line arguments
    load_images(args.url_file, args.output_dir)  # Call the download function

if __name__ == '__main__':
    main()  # Execute the main function
