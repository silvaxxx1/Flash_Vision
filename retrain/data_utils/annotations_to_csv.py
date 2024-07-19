import os
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

def convert_annotations_to_csv(xml_folder, output_csv):
    """
    Convert XML annotations to a CSV file.

    Args:
        xml_folder (str): Path to the folder containing XML files.
        output_csv (str): Path to the output CSV file.
    """
    data = []
    
    print(f"Reading XML files from {xml_folder}")
    
    for xml_file in os.listdir(xml_folder):
        # Process only XML files
        if xml_file.endswith('.xml'):
            try:
                # Parse the XML file
                tree = ET.parse(os.path.join(xml_folder, xml_file))
                root = tree.getroot()
                
                # Extract relevant information from each object element
                for member in root.findall('object'):
                    value = (
                        root.find('filename').text,
                        member.find('name').text,
                        int(member.find('bndbox/xmin').text),
                        int(member.find('bndbox/ymin').text),
                        int(member.find('bndbox/xmax').text),
                        int(member.find('bndbox/ymax').text)
                    )
                    data.append(value)
            except Exception as e:
                print(f"Error processing file {xml_file}: {e}")

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

    # Debugging statement to check the output path and file
    print(f"Attempting to save CSV to {output_csv}")
    
    try:
        # Save the DataFrame to a CSV file
        df.to_csv(output_csv, index=False)
        print(f"CSV file saved to {output_csv}")
    except Exception as e:
        print(f"Error saving CSV file to {output_csv}: {e}")

def main():
    """
    Main function to parse command-line arguments and call the conversion function.
    """
    parser = argparse.ArgumentParser(description='Convert XML annotations to CSV.')
    parser.add_argument('xml_folder', type=str, help='Folder containing XML files.')
    parser.add_argument('output_csv', type=str, help='Output CSV file path.')

    args = parser.parse_args()
    convert_annotations_to_csv(args.xml_folder, args.output_csv)

if __name__ == '__main__':
    main()
