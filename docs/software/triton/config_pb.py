# Package the following code in a Python script to generate a model configuration file for PyTorch.
# This script generates a model configuration file for a PyTorch model using the Triton Inference Server format.
# Save this script as `config_pb.py` and run it to create the `config.pbtxt` file.
import os

def save_config(model_path: str):
    """
    This function generates a configuration file for a PyTorch model.
    The configuration is saved in the Triton Inference Server format.
    """
    # Define the path for the configuration file
    config_file_path = os.path.join(os.path.dirname(model_path), 'config.pbtxt')
    
    # Create the configuration file with the necessary content
    with open(config_file_path, 'w') as f:
        f.write("# Configuration file for PyTorch model\n")
        f.write("name: \"lr\"\n")
        f.write("platform: \"pytorch_libtorch\"\n")
        f.write("max_batch_size: 0  # Set to 0 for no batching, or a positive integer for max batch size\n")
        f.write("version_policy: { all { }}\n")
        f.write("input [\n")
        f.write("  {\n")
        f.write("    name: \"input_name\"\n")
        f.write("    data_type: TYPE_FP32 # Or other data type\n")
        f.write("    dims: [1] # Specify input dimensions\n")
        f.write("  }\n")
        f.write("]\n")
        f.write("output [\n")
        f.write("  {\n")
        f.write("    name: \"output_name\"\n")
        f.write("    data_type: TYPE_FP32 # Or other data type\n")
        f.write("    dims: [1] # Specify output dimensions\n")
        f.write("  }\n")
        f.write("]\n")
    f.close()

if __name__ == "__main__":
    # Example usage
    save_config("/var/tmp/")
    print(f"Configuration file saved at: {os.path.join(os.path.dirname(model_path), 'config.pbtxt')}")
    
