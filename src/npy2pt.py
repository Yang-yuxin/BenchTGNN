import numpy as np
import torch
import argparse

def npy_to_pt(npy_file_path, pt_file_path):
    # Load the .npy file
    npy_data = np.load(npy_file_path)

    # Convert the NumPy array to a PyTorch tensor
    tensor_data = torch.from_numpy(npy_data).float()

    # Save the tensor as a .pt file
    torch.save(tensor_data, pt_file_path)

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Convert a .npy file to a .pt file")
    parser.add_argument('--npy', type=str, help="Path to the input .npy file")
    parser.add_argument('--pt', type=str, help="Path for the output .pt file")

    # Parse the arguments
    args = parser.parse_args()

    # Convert the file
    npy_to_pt(args.npy, args.pt)

if __name__ == "__main__":
    main()
