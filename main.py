import pandas as pd
import argparse

def get_file_path():
    parser = argparse.ArgumentParser(description="Enter the path to the csv file to read", add_help=False)

    #File path
    parser.add_argument("--path", type=str, required=True, help="The csv file path")
    return parser
    
def main(arguments):
    file = pd.read_csv(arguments.file_path)
    print(file.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AutoFactory", parents=[get_file_path()])
    arguments = parser.parse_args()
    main(arguments)