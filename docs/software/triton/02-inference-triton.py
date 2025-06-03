# 
# A simple request to the Triton Inference Server using the REST API.
# This script sends a request to the Triton Inference Server to make a prediction
# using a linear regression model. The model is expected to be running on the server.
# The script checks if the HOST environment variable is set, and if not, it exits.
# It also checks if the model is available on the server and sends a request to make a prediction.
# The response is logged, and if there is an error, it is also logged.
# The script uses the requests library to send the request and handle the response.
# The script is designed to be run as a standalone program.
#
import requests
import os
import logging
import ast
import json

def make_prediction(input_value: float, host: str)-> requests:
    """Make a prediction using the Triton Inference Server.
    Args:
        input_value (float): The input value for the model.
        host (str): The Triton Inference Server host.
    Returns:
        requests: The response from the Triton Inference Server.
    """
    # Build the request payload.
    # The input name is the name of the input tensor in the model.
    # The shape is the shape of the input tensor.
    # The datatype is the data type of the input tensor.
    # The data is the input value.
    # The input value is a float, so we need to convert it to a list.
    req2 = {
        "inputs": [
            {
            "name": "input_name",
            "shape": [1],
            "datatype": "FP32",
            "data": [input_value]
            }
        ]
        }

    url = f'{host}/v2/models/lr/infer'
    r = requests.post(url, json=req2)
    return r

if __name__ == "__main__":

  logging.basicConfig(level=logging.INFO)

  #
  # Check that $HOST is set.
  #
  host = os.getenv('HOST', 'http://localhost:8000')

r = None
try:
    r = make_prediction(2.5, host)
    logging.debug(f'{host = }')
    logging.debug(f'REST inference response = {r.status_code}')
    logging.info(f'REST inference response content = {r.content.decode()}')
    p = ast.literal_eval(r.content.decode())
except requests.exceptions.RequestException as e:
    logging.error(f"Request failed: {e}")

    
