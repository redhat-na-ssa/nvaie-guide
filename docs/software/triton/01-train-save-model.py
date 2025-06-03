#
# Create a simple PyTorch linear regression model
# and save it to a file for use with nvidia triton.
#
import torch
from torch.autograd import Variable
import logging
import os
import config_pb 

# Set up logging
logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def create_model_repo(model_repository: str = "models"):
    """
    Create a model repository for Triton Inference Server.
    """
    cwd = os.getcwd()
    target_dir = cwd + "/" + model_repository

    if os.path.isdir(target_dir):
        logger.info(f"Directory '{target_dir}' exists.")
    else:
        logger.info(f"Directory '{target_dir}' does not exist.")
        os.makedirs(target_dir, exist_ok=True)
        logger.info(f"Directory '{target_dir}' created.")


def save_model(repo_name: str, name: str, version: int, model: torch.nn.Module):
    """
    Save the PyTorch model to a file. This function creates a directory structure for the model
    and saves the model in the appropriate format.
    :param name: Name of the model.
    :param version: Version of the model.
    :param model: The PyTorch model to save.
    """
    cwd = os.getcwd()
    model_path = f'{cwd}/{repo_name}/{name}/{version}/model.pt'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.jit.save(torch.jit.script(model), model_path)
    logger.info(f'Model saved to {model_path}')
    # Save the model protobuf config file
    config_file_path = f'{cwd}/{repo_name}/{name}/'
    config_pb.save_config(config_file_path)
    logger.info(f'Saved Triton Model Config to {config_file_path}')


class LinearRegressionModel(torch.nn.Module):
    """
    A simple linear regression model.
    This model takes a single input and produces a single output w/o a bias term.
    """
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)

    def forward(self, x):
        """
        Forward pass through the model.
        :param x: Input tensor.
        :return: Output tensor.
        """
        y_pred = self.linear(x)
        return y_pred

# Create the model instance.
our_model = LinearRegressionModel()

# The model is a linear regression model, so we can use
# the Mean Squared Error loss function
# and stochastic gradient descent as the optimizer.
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)

# Training data

# In this example, we will use a simple linear function y = 2x
# x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
# y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

# In this example, we will use a simple linear function y = 3x
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[3.0], [6.0], [9.0]]))

# Tell PyTorch to track the gradients of the input data
# so we can compute the gradients of the loss with respect to the input data.
x_data.requires_grad = True

# Training loop
epochs = 100
for epoch in range(epochs):

    # Forward pass: Compute predicted y by passing 
    # x to the model
    pred_y = our_model(x_data)

    # Compute and print loss
    loss = criterion(pred_y, y_data)

    # Zero gradients, perform a backward pass, 
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    logger.debug('epoch {}, loss {}'.format(epoch, loss.item()))

# Test the model
new_var = Variable(torch.Tensor([[2.5]]))
pred_y = our_model(new_var)

logger.info(f"{new_var = } {pred_y = }")

repo_name = "models"
model_name = "lr"
model_version = 2
create_model_repo(repo_name)
save_model(repo_name, model_name, model_version, our_model)

