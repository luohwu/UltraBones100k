import yaml
from pyhocon import ConfigFactory

def read_anatomy_map_from_yaml(filename):
    """Reads a YAML file and returns the data as a dictionary."""
    try:
        with open(filename, 'r') as file:
            anatomy_map = yaml.safe_load(file)
            return anatomy_map
    except FileNotFoundError:
        print("The specified file was not found.")
    except yaml.YAMLError as exc:
        print("Error while parsing YAML file:", exc)
    except Exception as e:
        print("An error occurred while reading the YAML file:", e)




def read_confs(conf_path):
    f = open(conf_path)
    conf_text = f.read()
    f.close()
    return ConfigFactory.parse_string(conf_text)