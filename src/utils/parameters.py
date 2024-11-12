import os, yaml, logging 

def write_params_to_file(params, out_dir):
    filename = os.path.join(out_dir, "parameters.txt")
    with open(filename, "w") as file:
        for arg, value in params.items():
            file.write(f"{arg}: {value}\n")


def load_parameters(path):
    try:
        stream_file = open(path, "r")
        parameters = yaml.load(stream_file, Loader=yaml.FullLoader)

        logging.info(
            "[AROB-2024-KAPTIOS::main] Success: Loaded parameter file at: {}".format(
                path
            )
        )
    except:
        logging.error(
            "[AROB-2024-KAPTIOS::main] ERROR: Invalid parameter file at: {}, exiting...".format(
                path
            )
        )
        exit()

    return parameters
