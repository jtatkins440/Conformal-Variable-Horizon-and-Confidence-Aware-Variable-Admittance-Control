import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MaxAbsScaler
import torch.optim as optim
import yaml

import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels, num_classes, window_size, same_padding,out_channels1,out_channels2, kernel_size1, kernel_size2):
        super(TimeSeriesCNN, self).__init__() 
        kernel_size1 = kernel_size1
        kernel_size2 = kernel_size2
        if same_padding:
            padding1 = (kernel_size1-1)//2
            padding2 = (kernel_size2-1)//2
        else:
            padding1 = 0
            padding2 = 0
        # Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=out_channels1, kernel_size=kernel_size1, stride=1, padding=padding1)
        self.conv2 = nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size2, stride=1, padding=padding2)
        
        conv1_output_size = (window_size + 2*padding1 - (kernel_size1-1) - 1) + 1  # Conv1D Layer 1
        conv2_output_size = (conv1_output_size + 2*padding2 - (kernel_size2-1) - 1) + 1
        self.fc = nn.Linear(out_channels2 * int(conv2_output_size) , num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def predict_output(self, x):
        x = F.relu(self.conv1(x))  # [Batch, 16, Sequence Length]
        x = self.dropout(x)
        x = F.relu(self.conv2(x))  # [Batch, 32, Sequence Length // 2]
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten: [Batch, Flattened Features]
        return self.fc(x)

    def predict(self, input):
        with torch.no_grad():
            return self.predict_output(input)


class TaskClassModelWrapper():
    def __init__(self, model_name, model_dir_path = None, weights_extension = ".pt", config_extension = "_config.yaml", compile_model = True, use_floats = True):
        self.use_floats = use_floats
        

        # 1.setting path for model and config
        if model_dir_path is not None:
            model_path = os.path.join(model_dir_path, model_name + weights_extension)
            config_path = os.path.join(model_dir_path, model_name + config_extension)
        else:
            model_path = model_name + weights_extension
            config_path = model_name + config_extension


        # 2.import model configuration  
        with open(config_path) as file:
            config_dict = yaml.safe_load(file)
        
        # 3. Initialize the model with the same architecture used during training
        base_model = TimeSeriesCNN(config_dict["model"]["input_channels"],
                                        config_dict["model"]["num_classes"],
                                        config_dict["model"]["window_size"],
                                        config_dict["model"]["same_padding"],
                                        config_dict["model"]["out_channels1"],
                                        config_dict["model"]["out_channels2"],
                                        config_dict["model"]["kernel_size1"],
                                        config_dict["model"]["kernel_size2"])
        
        self.maxabs_weights = config_dict["model"]["MaxAbsScale"]
        # 4. Load the saved weights into the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model.load_state_dict(torch.load(model_path,map_location=device))
        
        # 5. Give initial value to the model (To eliminate the delay when inputting the first value into the model)
        temp_input = np.zeros(shape=(config_dict["model"]["window_size"], config_dict["model"]["input_channels"])) # input shape: (window_size, input_channels)
        if compile_model:
            self.base_model = torch.compile(base_model.eval()).eval()
            temp_output,temp_prob,temp_label = self.predict(temp_input)
            print("Compiled model: " + model_name + "!")
        else:
            self.base_model = base_model
            print("Did not compile model!")
        
    
    def preprocess_input(self, input):
        # assume input is a numpy array and have size (window_size, input_channels)
        # process input should have tensor size = (batch_size, input_channels, window_sizes) as input for model
        input = input.T # need to transpose change size (window_size, input_channels) to (input_channels, window_size)
        if self.use_floats:
            input_t = torch.tensor(input, dtype=torch.float)
        else:
            input_t = torch.tensor(input, dtype=torch.double)
        
        input_t = torch.unsqueeze(input_t, 0) # add batch dim
        return input_t
    
    
    def postprocess_output(self, output_t):
        prob_out = F.softmax(output_t,dim=1)
        label_out = torch.argmax(prob_out,dim=1)
        return output_t.numpy(force=True),prob_out.numpy(force=True),label_out.numpy(force=True)
    
    def predict(self, input):
        input_t = self.preprocess_input(input)
        output_t = self.base_model.predict(input_t)
        return self.postprocess_output(output_t)