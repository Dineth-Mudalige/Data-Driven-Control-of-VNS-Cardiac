# Data Driven Control of VNS for the Cardiac System
 
 This is the read me for the open source code provided as supplement to the Frontiers in Physiology paper: 
 
 *Data Driven Control of Vagus Nerve Stimulation for the Cardiovascular System: an in silico Computational Study*
 
 Authors: A. Branen, Y. Yao, M. Kothare, B. Mahmoudi, and G. Kumar, 2022

Correspondence: gautam.kumar@sjsu.edu 

DOI: <https://doi.org/10.3389/fphys.2022.798157>

## Folders 

MATLAB - contains 7 files for the physiological model, intra-patient modified physiological model, and the sympathetic modified physiological model (diseased state), and associated files regarding obtaining the constants for each model. 

CONTROLLER - contains 7 files for the Python controller code, and the trained LSTM model used for control. 

See below for a description of the files. 

## Running Simulations 

To run a simulation with this code, place the MATLAB files into the current working MATLAB directory. Then update the path variables in the Python Controller codes with the MATLAB Path: 

`matlab_path = 'path to the current working MATLAB directory'` 

An example for Linux/Mac users may look like: `'/Users/Your-Name/Documents/MATLAB'` 

Also update the path variable to the LSTM model: 

`model_path = 'path-to-the-model/LSTM_10_tanh.h5'` 

Then run the controller file to obtain the simulation results of interest with a traditional call to the Python environment from the command line: 

`python "path-to-controller-file/simulation-of-interest.py"`

More information regarding how to install the MATLAB Engine API for Python can be found on the MathWorks website: 
<https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html> 

## Packages and Versions Used 

`Python == version 3.7.9` 

`MATLAB == version 2019b` 

`numpy == version 1.19.5`

`tensorflow == version 2.5.0`

`matplotlib == version 3.4.2`

`scipy == version 1.6.0`

### MATLAB - Files 

**plant.m** - the physiological model of the rat  cardiovascular system with vagal nerve stimulation. 

**Get_Model_Constants.m** - a function that returns the values of the model parameters for the rat physiological model. 

**par_greyid.mat** - model parameters for the physiological model 

**myplant.m** - a function that takes the ODE initial condition vector, and vagal nerve stimulation parameters, and uses those to call the physiological model, and returns the final ODE state vector, heart rate, and mean arterial blood pressure values. 

**myplant_intra_patient.m** - a function that takes the ODE initial condition vector, and vagal nerve stimulation parameters, and uses those to call the intra-patient variability modified physiological model, and returns the final ODE state vector, heart rate, and mean arterial blood pressure values

**par_greyid_sympathetic.mat** - model parameters for the overactive sympathetic diseased pathological model 

**myplant_sympathetic.m** - function that takes the ODE initial condition state vector, and vagal nerve stimulation parameters, calls the overactive sympathetic modified physiological model, and returns the final ODE state vector, heart rate, and mean arterial blood pressure values 

### CONTROLLER - Files 

**LSTM_10_tanh.h5** - the trained LSTM from the paper that was used in the controller simulations 

**Controller-Code-L1.py** - The controller code that runs closed-loop simulations of the sparsity promoted controller design (section 2.2.1) with the physiological model. This will generate the results shown in *Figure 3*. 

**Controller-Code-L2.py** - The controller code that runs closed-loop simulations of the minimum energy based controller design (section 2.2.2) with the physiological model. This will generate the results shown in *Figure 4*. 

**Controller-Code-DeltaU.py** - The controller code that runs the closed-loop simulations of the minimum overshoot based controller design (section 2.2.3) with the physiological model. This will generate the results shown in *Figure 5*. 

**Controller-Code-Intra-No-OffsetFree.py** - The controller code that runs the closed-loop simulations of the sparsity promoted controller design with the intra-patient modified physiological model (section 2.3). This will generate the results shown in *Figure 7*. 

**Controller-Code-Intra-Patient.py** - The controller code that runs the closed-loop simulations of the offset-free controller design with the intra-patient modified physiological model (section 2.3). This will generate the results shown in *Figure 8*. 

**Controller-Code-Sympathetic.py** - The controller code that runs the closed-loop simulations of the offset-free controller design with the sympathetic modified physiolgoical model (section 2.4). This will generate the results shown in *Figure 10*. 
