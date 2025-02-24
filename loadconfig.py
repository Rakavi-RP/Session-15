import os
import yaml

# Correct path to the YAML file
config_path = "/kaggle/input/configfile/config_smollm2_135M.yaml"

# Load the YAML configuration file from the correct path.
with open(config_path, "r") as f:
    config_dict = yaml.safe_load(f)

# Print the keys at the top level to see how the YAML is structured
print("Top-Level Keys in Config:", config_dict.keys())

# If 'model' is a key, print its subkeys
if "model" in config_dict:
    print("Keys inside 'model':", config_dict["model"].keys())
    if "model_config" in config_dict["model"]:
        print("Keys inside 'model_config':", config_dict["model"]["model_config"].keys())
