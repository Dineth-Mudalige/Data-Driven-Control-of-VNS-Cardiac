# -*- coding: utf-8 -*-
"""
Created on 7-28-2021

Runs a closed-loop simulation of the sparsity promoted controller design (section 2.2.1 and section 2.3) with the intra-patient modified physiological model. The results from this simulation are shown in Figure 7.

@author: Andrew Branen

"""
# Import required packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
# Additional packages for matlab
import matlab.engine

# Setting up Matlab to work 
eng = matlab.engine.start_matlab()
matlab_path = 'MATLAB'
eng.addpath(matlab_path)

# Load the LSTM
model_path = 'LSTM_10_tanh.h5'
model = tf.keras.models.load_model(model_path)

# Normalization constants for the LSTM 
# HR and MAP Normed values
var_range = np.asarray([99.576448, 86.495630])
var_mean = np.asarray([385.841893, 134.356465])

# VNS Normalization constants
VNS_range = np.array([0.000499, 49.978558, 0.000500, 49.903930, 0.000500, 49.996492])
VNS_mean = np.array([0.000128, 12.432705, 0.000131, 12.715696, 0.000122, 12.571199])

##########################################################################################################
# loss function
def loss(y_sp, y, u):
    """
    This function calculates the loss value using the following equation: 
    
    Loss = Sum(y_sp - y)^2 + Sum(|u|) 
    
    Where the first sum represents a quadratic cost, and the second sum represents a L1 cost on the inputs. 
    
    Arguments: 
    
    y_sp: [HR set point, MAP set point], the target set points for the current objective. Shape = (2,1) 
    
    y: [HR values, MAP values], the LSTM's predicted values for HR and MAP over Np (predictive horizon). Shape =(Np,2) i.e. Np = 10, shape = (10, 2) 
    
    u: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], the inputs parameters for all 3 locations over Nc (control horizon). Shape = (Nc, 6). i.e. Nc = 2, shape = (2, 6)  
    
    Returns: 
    
    J: scalar quantity of the loss (i.e. "20") 
    
    """
    # Finding Nc  
    t_steps = u.shape[1] 
    
    # Each input has a different zeroing constant
    z_const = tf.constant([0.2565130260521042, 0.24876077857228296, 0.262, 0.25480349944383135, 0.244, 0.2514416211441395])
    
    # L1 cost (second sum) 
    L1 = 0 
    for i in range(t_steps):
        L1 += tf.abs(u[:,i,:] + z_const) 

    # Full cost is quadratic and L1, with L1 weight W 
    J = tf.reduce_sum((y_sp - y)**2) + W*tf.reduce_sum(L1)
    return J
    
##########################################################################################################    

# Gradient
def jac(up, x, n, sp):
    """
    This function calculates the Jacobian using Tensorflow and supplies the output to the Scipy solver. This allows for faster optimization.
    
    Arguments: 
    
    up: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], the selected inputs over Nc (control horizon). Shape will be (Nc, 6) i.e. Nc = 2, up = (12,) 
    
    x: [HR, MAP], the current values of HR and MAP, the "initial conditions". Shape = (2,1) 
    
    n: scalar, the number of cycles to be run, equal to Np (predictive horizon) in this case
    
    sp: [HR_sp, MAP_sp], the target set points of the HR and MAP. Shape = (2,1)
    
    Returns: 
    
    dl_du1: [dl_u1 PW, dl_u1 Freq, dl_du2 PW, dl_u2 Freq, dl_du3 PW, dl_du3 Freq], the derivative of the loss function with respect to all of the inputs over Nc (control horizon). Shape will be some multiple of 6 depending on the selection of Nc. i.e. Nc = 2, dl_du1 = (2*6,) = (12,) 
    
    """
    # make sure everything coming in is float32
    u1 = tf.cast((up), tf.float32) 
    
    # autosizing, based on Nc
    t_steps = int(len(up)/6)
    u1 = tf.reshape((u1), (1, t_steps, 6)) 
    
    # convert to variable tensors for tensorflow tracking
    u1 = tf.Variable(u1)
    
    # current input values of HR and MAP
    x = tf.constant([[x]], dtype=tf.float32)
    
    # target set point values of HR and MAP 
    ysp = tf.constant([[sp]], dtype=tf.float32)
    
    # Now to get to the gradient 
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(u1)
        for i in range(0, n):
            VNS_step = tf.reshape(u1[:,i,:], (1,1,6)) if i < t_steps else tf.reshape(u1[:,-1,:], (1,1,6))
            g = tf.concat((VNS_step, x), axis=2)
            # run the model
            y = model(g) if i == 0 else tf.concat((y, model(g)), axis=1)
            x = tf.reshape((y[:,-1,:]), (1,1,2))
            
        # Compute the loss value by calling loss_L1
        loss_value = loss(ysp, y, u1)
    # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to loss
    dl_du1 = tape.gradient(loss_value, [u1])
    
    # Convert to numpy for scipy
    dl_du1 = np.reshape((np.array(dl_du1)), (6*t_steps,)) 
    
    return dl_du1

##########################################################################################################

# Loss function
def Loss(up, x, n, sp):
    """
    This is the same function as jac, however it will only return the loss value, instead of the gradient of the loss with respect to the inputs.
    
    Arguments: 
    
    up: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], the selected inputs over Nc (control horizon). Shape will be (Nc, 6) i.e. Nc = 2, up = (12,) 
    
    x: [HR, MAP], the current values of HR and MAP, the "initial conditions". Shape = (2,1) 
    
    n: integer, the number of cycles to be run, equal to Np (predictive horizon) in this case
    
    sp: [HR_sp, MAP_sp], the target set points of the HR and MAP. Shape = (2,1)
    
    Returns: 
    
    J: scalar, the loss value of the cost function specified by loss_L1
    
    """
    
    # make sure everything coming in is float32
    u1 = tf.cast((up), tf.float32) 
    
    # autosizing 
    t_steps = int(len(up)/6)
    u1 = tf.reshape((u1), (1, t_steps, 6)) 
    
    # convert to variable tensors for tensorflow
    u1 = tf.Variable(u1)
    
    # then we have current HR and MAP values 
    x = tf.constant([[x]], dtype=tf.float32)
    
    # and the target set point 
    ysp = tf.constant([[sp]], dtype=tf.float32)
    
    # Now to get to the gradient part of things
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(u1)
        for i in range(0, n):
            VNS_step = tf.reshape(u1[:,i,:], (1,1,6)) if i < t_steps else tf.reshape(u1[:,-1,:], (1,1,6))
            g = tf.concat((VNS_step, x), axis=2)
            # run the model
            y = model(g) if i == 0 else tf.concat((y, model(g)), axis=1)
            x = tf.reshape((y[:,-1,:]), (1,1,2))
            
        # Compute the loss value
        loss_value = loss(ysp, y, u1)
    
    return loss_value.numpy().reshape((1,)) 

##########################################################################################################
# Weight of the L1 cost 
W = 0.001
# Prediction Horizon (number of cycles)
Np = 10
# Control Horizon (number of cycles)
Nc = 5
# Plot the output from the simulation? 
plotflag = True

##########################################################################################################
# Controller Function
def controller(IC, Np, set_point, initial_guess):
    """
    This function specifies the controller by optimizing the VNS parameters (inputs)
    
    Arguments: 
    
    IC: [HR, MAP], initial condition vector of size (2,) for the current values of the HR and MAP 
    
    Np: integer, the number of cycles for the predictive horizon
    
    set_point: [HR_sp, MAP_sp], the target values of the HR and MAP of shape (2,) 
    
    initial_guess: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], an initial guess for the solver to start its search from, of shape (Nc*6,) i.e. Nc = 2, shape = (12,) 
    
    Returns: 
    
    controller_actions: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], the optimized VNS parameters for the optimization problem specified. The shape of this output will depend on Nc. i.e. Nc = 2, shape = (12,) 
    
    """

    # Bounds of the variables 
    bnds = ((-0.2565130260521042 ,  0.7442447729893789), (-0.24876077857228296 ,  0.7512392141651547),
       (-0.262,  0.7374489713901999), (-0.25480349944383135 ,  0.7451964954883866),
       (-0.244 ,  0.7557925644446), (-0.2514416211441395 ,  0.7485583789001786))

    # Solving the optimization problem using the Sequential Least Squares Programming algorithm from Scipy
    controller_actions = scipy.optimize.minimize(Loss, Nc*initial_guess,
                                         args=(IC, Np, set_point), method='SLSQP', bounds=Nc*bnds,
                                         jac=jac,
                          options = {'ftol': 1e-5, 'disp': False, 'maxiter':50}).x

    # Finally, we return the control action selection
    return controller_actions
##########################################################################################################

# Start the simulation with the controller and the PM 
##########################################################################################################
# initial conditions for the physiological model
state_IC = [217.77, 4.42, 108.84, 92.08, 0.17, 0.04, 0.13, 0.15, 0.12, 1.94, 0]

# Prep for Matlab
state_IC = matlab.double(state_IC)

# Declare a matrix to store the physiological variables of interest
ICs = np.zeros((451,2))

# create a matrix for storing the VNS actions applied 
applied_VNS = np.zeros((450,6))

# Get the initial value from the PM model
# No VNS input
VNS_initial = np.array([[0, 0, 0, 0, 0, 0]])
# Send to matlab
VNS_i = matlab.double(VNS_initial.T.tolist()) 
# Run the matlab file for one cycle
new_ini, HR_tot, MAP_tot = eng.myplant_intra_patient(state_IC, VNS_i, nargout=3)
# Store the output in the physiological variables matrix
ICs[0,:] = np.asarray([HR_tot, MAP_tot])

# A matrix of our set points 
SPs = np.asarray([[356.0, 150.0], [393.0, 129.0],[377.0,143.0]])

# An initial guess, NOTE: this will be multiplied by Nc to give the correct dimensions, check the controller
# in the controller, note "Nc*initial_guess"
ig = [0.24448898, 0.25145373, 0.238, 0.24615905, 0.256, 0.24859346]
##########################################################################################################

# Running the feedback simulation
##########################################################################################################
for i in range(450):
    
    # Determine the set point based on cycle 
    if i < 150:
        # Normalize the setpoint
        SP = (SPs[0,:] - var_mean) / var_range
    elif i < 300:
        # Normalize the setpoint
        SP = (SPs[1,:] - var_mean) / var_range
    else : 
        # Normalize the setpoint
        SP = (SPs[2,:] - var_mean) / var_range
        
    # The initial guess supplied to the solver depends on the optimized paramters 
    # if the solver has already optimized parameters, we start looking from those optimized parameters
    guess_i = ig if i ==0 else ((applied_VNS[i-1,:] - VNS_mean)/VNS_range).tolist()
    
    # Normalize the initial conditions for the controller 
    controller_IC =( ICs[i,:] - var_mean )/ var_range
    
    # Run the controller to determine the optimal VNS parameters 
    fin_VNS = controller(controller_IC, Np, SP, guess_i)
    
    # un normalize the optimized VNS paramters 
    fin_VNS = fin_VNS.reshape((Nc,6))*VNS_range + VNS_mean
    
    # store the selected action, NOTE: only stores the first action (i.e. Nc=1) if Nc > 1 
    applied_VNS[i,:] = fin_VNS[0,:]
    
    # implement the controller action and run the PM for one cycle 
    VNS_action = matlab.double(fin_VNS[0].reshape((6,1)).tolist())
    new_ini, HR, MAP = eng.myplant_intra_patient(new_ini, VNS_action, nargout=3)
    
    # store the PM output 
    ICs[i+1,:] = np.asarray([HR, MAP]) 
    
    # Give updates on where the simulation is at 
    print("We are on step: ", i, "/449")
    
##########################################################################################################

# Plotting
if plotflag: 
    xts = np.arange(0, len(ICs[:,0]))
    
    # Plots the Heart Rate with cardiac cycle
    plt.figure(figsize=(8,12))
    plt.subplot(211)
    plt.scatter(xts, ICs[:,0], c='b')
    plt.plot([0, 150, 150, 300, 300, 450], [356, 356, 393, 393, 377, 377], c='k')
    plt.ylabel('HR (bpm)')
    plt.xlabel('Cardiac Cycle')

    # Plots the blood pressure with cardiac cycle
    plt.subplot(212)
    plt.scatter(xts, ICs[:,1], c='b')
    plt.plot([0,150,150,300,300,450], [150, 150, 129, 129, 143, 143], c='k')
    plt.ylabel('MAP (mmHg)')
    plt.xlabel('Cardiac Cycle')
    plt.show()

    # Plots the VNS Pulse Widths applied by the controller
    plt.figure(figsize=(8,12))
    plt.subplot(211)
    plt.plot(applied_VNS[:,0], label='Loc 1', c='k')
    plt.plot(applied_VNS[:,2], label='Loc 2', c='b')
    plt.plot(applied_VNS[:,4], label='Loc 3', c='r')
    plt.xlabel('Cardiac Cycle')
    plt.ylabel('Pulse Width (s)')
    plt.legend()
 
    # Plots the VNS Frequencies applied by the controller
    plt.subplot(212)
    plt.plot(applied_VNS[:,1], label='Loc 1', c='k')
    plt.plot(applied_VNS[:,3], label='Loc 2', c='b')
    plt.plot(applied_VNS[:,5], label='Loc 3', c='r')
    plt.xlabel('Cardiac Cycle')
    plt.ylabel('Pulse Frequency (Hz)')
    plt.legend()
    plt.show() 
