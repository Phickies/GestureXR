import numpy as np
import math


def get_function_sample(samples: int = 2000):
    """
    Generates a synthetic dataset which follows the function f(x) = cos(x) + sin(2x)

    Parameters
    ----------
    samples : int
        Amount of samples to generate
        Default value is 2000 samples

    Returns
    -------
    x, y : tuple[np.ndarray, np.ndarray]
        x- and y- coordinates (features and labels)
    """

    # Generate random x coordinates between 0 and 2pi, shuffle their order
    x = np.random.uniform(0, 2 * math.pi, size=samples)
    np.random.shuffle(x)

    # Generate y values with noise
    y = np.cos(x) + np.sin(2 * x)
    y += 0.1 * np.random.randn(*y.shape)

    return x, y


def convert_tflite_to_c(tflite_path: str = 'model.tflite', model_name: str = 'model') -> str:
    """
    Converts TFLite models into C-compatible header files for Arduino etc.

    Parameters
    ----------
    tflite_path : str
        Path to the TFLite model file
        Default path is 'model.tflite'
    model_name : str
        Output model name
        Default value is 'model'

    Returns
    -------
    model_path : str
        Path to the converted C-compatible model

    Raises
    ------
    ValueError
        If the provided model is not a TFLite model
    """

    # Check if file is a tflite model. If not, raise ValueError.
    if not tflite_path.endswith('.tflite'):
        raise ValueError("The provided file is not a TFLite model.")

    # Open the TFLite model file in binary mode and read its content into 'tflite_content'.
    with open(tflite_path, 'rb') as tflite_file:
        tflite_content = tflite_file.read()

    # Calculate the length of 'tflite_content' (i.e., the size of the TFLite model in bytes).
    array_length = len(tflite_content)

    # Split 'tflite_content' into chunks of 12 bytes each and convert each chunk to a hexadecimal string.
    # This is done so that the TFLite model can be represented as an array in C-compatible format.
    hex_lines = [', '.join([f'0x{byte:02x}' for byte in tflite_content[i:i + 12]]) for i in
                 range(0, len(tflite_content), 12)]

    # Join the chunks of hexadecimal strings with newlines to format them neatly.
    hex_array = ',\n     '.join(hex_lines)

    # Open a header file in write mode and write out the TFLite model as an array.
    with open(model_name + '.h', 'w') as header_file:
        # Write the length of the TFLite model to the header file.
        header_file.write(f'unsigned int {model_name}_len = {array_length};\n\n')

        # Write out the TFLite model as an array in C-compatible format.
        header_file.write(f'alignas(8) const unsigned char {model_name}[] = {{\n     ')
        header_file.write(f'{hex_array}\n')
        header_file.write(f'}};\n')

    # Return the name of the generated header file.
    return model_name + '.h'
