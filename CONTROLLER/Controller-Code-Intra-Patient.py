# -*- coding: utf-8 -*-
"""
7 - 13 - 21

Runs a closed-loop simulation of the offset-free controller design (section 2.3) with the intra-patient modified physiological model. The results from this simulation are shown in Figure 8.

@author: Andrew Branen
"""

# required packages
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
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

####################################################################################
# loss function
def loss(y_sp, y, u, t_steps):
    """
    This function calculates the loss value using the following equation:
    
    Loss = Sum((ybar - y)^2) + Sum((ubar - u)^2)
    
    Where the first sum represents a quadratic cost in the disturbance altered model predictions and the optimization variable of the disturbance evaluated targets.
    
    Arguments:
    
    y_sp: [HR set point, MAP set point], the target set points for the current objective. Shape = (2,1)
    
    y: [HR values, MAP values], the LSTM's predicted values for HR and MAP over Np (predictive horizon). Shape =(Np,2) i.e. Np = 10, shape = (10, 2)
    
    u: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], the inputs parameters for all 3 locations over Nc (control horizon). Additionally ybar ubar, where ubar is the disturbance evaluated control action that leads to ybar (new target evaluated through the disturbance model). Shape = (Nc+8, 6). i.e. Nc = 2, shape = (10, 6)
    
    t_steps: integer, the size of the control horizon (Nc)
    
    Returns:
    
    J: scalar quantity of the loss (i.e. "20")
    """
    
    # new set point evaluated from disturbance
    xbar = u[0:2]
    
    # control action that leads to the new set point
    ubar = u[2:8]
        
    # delta ubar
    del_ubar = 0
    for i in range(t_steps-1):
        del_ubar += tf.square(u[8+6*i:8+6*(i+1)] - ubar)
        
    # Full cost
    J = tf.reduce_sum((y - xbar)**2) + tf.reduce_sum(del_ubar)
    
    return J

####################################################################################
# Gradient
def jac(up, x, n, sp, d):
    """
    This function calculates the Jacobian using Tensorflow and supplies the output to the Scipy solver. This allows for faster optimization.
    
    Arguments:
    
    up: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], the inputs parameters for all 3 locations over Nc (control horizon). Additionally ybar ubar, where ubar is the disturbance evaluated control action that leads to ybar (new target evaluated through the disturbance model). Shape = (Nc+8, 6). i.e. Nc = 2, shape = (10, 6)
    
    x: [HR, MAP], the current values of HR and MAP, the "initial conditions". Shape = (2,1)
    
    n: integer, the number of cycles to be run, equal to Np (predictive horizon) in this case
    
    sp: [HR_sp, MAP_sp], the target set points of the HR and MAP. Shape = (2,1)
    
    d: [HR_dhat, MAP_dhat], the disturbance values for each variable. Shape = (2,1)
    
    Returns:
    
    dl_du1: [dl_u1 PW, dl_u1 Freq, dl_du2 PW, dl_u2 Freq, dl_du3 PW, dl_du3 Freq], the derivative of the loss function with respect to all of the inputs over Nc (control horizon). Additionally, the 8 other optimized variables from ubar and ybar are included. Shape will be some multiple of 6 plus 8, depending on the selection of Nc. i.e. Nc = 2, dl_du1 = (2*6+8,) = (20,)
    
    """
    # make sure everything coming in is float32
    u1 = tf.cast((up), tf.float32)
    
    # autosizing, based on Nc
    t_steps = int((len(up) - 8)/6)
    u1 = tf.Variable(u1)
    
    # current input values of HR and MAP
    x = tf.constant([[x]], dtype=tf.float32)
    
    # disturbance values
    d = tf.constant([[d]], dtype=tf.float32)
    
    # target set point values of HR and MAP
    ysp = tf.constant([[sp]], dtype=tf.float32)
    
    
    # Now to get to the gradient
    with tf.GradientTape(watch_accessed_variables=True) as tape:
        tape.watch(u1)
        for i in range(0, n):
            # xbar values
            xbar = u1[0:2]
            # ubar values
            ubar = u1[2:8]
            VNS_step = tf.reshape(u1[8+6*i:8+6*(i+1)], (1,1,6)) if i < t_steps else tf.reshape(u1[-6:], (1,1,6))
            g = tf.concat((VNS_step, x), axis=2)
            # run the model
            y = model(g) + d if i == 0 else tf.concat((y, model(g)+d), axis=1)
            x = tf.reshape((y[:,-1,:]), (1,1,2))
        
        # Compute the loss value by calling loss
        loss_value = loss(ysp, y, u1, t_steps)
    # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to loss
    dl_dother = tape.gradient(loss_value, [u1, model.trainable_weights])
    
    # Convert to numpy for scipy
    dl_du1 = np.reshape((np.array(dl_dother[0])), (8 + 6*t_steps,))
    
    return dl_du1

####################################################################################
# Loss function
def Loss(up, x, n, sp, d):
    """
    This is the same function as jac, however it will only return the loss value, instead of the gradient of the loss with respect to the inputs.
    
    Arguments:
    
    up: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], the inputs parameters for all 3 locations over Nc (control horizon). Additionally ybar ubar, where ubar is the disturbance evaluated control action that leads to ybar (new target evaluated through the disturbance model). Shape = (Nc+8, 6). i.e. Nc = 2, shape = (10, 6)
    
    x: [HR, MAP], the current values of HR and MAP, the "initial conditions". Shape = (2,1)
    
    n: integer, the number of cycles to be run, equal to Np (predictive horizon) in this case
    
    sp: [HR_sp, MAP_sp], the target set points of the HR and MAP. Shape = (2,1)
    
    d: [HR_dhat, MAP_dhat], the disturbance values for each variable. Shape = (2,1)
    
    Returns:
    
    J: scalar, the loss value of the cost function specified by loss
    
    
    """
   # make sure everything coming in is float32
    u1 = tf.cast((up), tf.float32)
    
    # autosizing, based on Nc
    t_steps = int((len(up) - 8)/6)
    
    # convert to variable tensors for tensorflow tracking
    u1 = tf.Variable(u1)
    
    # current input values of HR and MAP
    x = tf.constant([[x]], dtype=tf.float32)
    
    # disturbance values
    d = tf.constant([[d]], dtype=tf.float32)
    
    # target set point values of HR and MAP
    ysp = tf.constant([[sp]], dtype=tf.float32)
    
    # Now to get to the gradient
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(u1)
        for i in range(0, n):
            # xbar values
            xbar = u1[0:2]
            # ubar values
            ubar = u1[2:8]
            VNS_step = tf.reshape(u1[8+6*i:8+6*(i+1)], (1,1,6)) if i < t_steps else tf.reshape(u1[-6:], (1,1,6))
            g = tf.concat((VNS_step, x), axis=2)
            # run the model with the disturbance
            y = model(g) + d if i == 0 else tf.concat((y, model(g)+d), axis=1)
            x = tf.reshape((y[:,-1,:]), (1,1,2))
        
        # Compute the loss value by calling loss
        loss_value = loss(ysp, y, u1, t_steps)
    
    return loss_value.numpy().reshape((1,))

####################################################################################
# Constraint function

def rcons(u, d0, r):
    """
    The nonlinear constraint that allows for optimization of the extra optimization variables, ybar and ubar.
    
    Arguments:
    
    u: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], the inputs parameters for all 3 locations over Nc (control horizon). Additionally ybar ubar, where ubar is the disturbance evaluated control action that leads to ybar (new target evaluated through the disturbance model). Shape = (Nc+8, 6). i.e. Nc = 2, shape = (10, 6)
    
    d0: [HR_dhat, MAP_dhat], the disturbance values at the current time, shape = (2,1)
    
    r: [HR_sp, MAP_sp], the target set points for the HR and MAP, shape = (2,1)
    
    Returns:
    
    Difference between the target set point and the disturbance evaluated LSTM model, shape = (2,1)
    """
    # cast to tensors
    u = tf.cast((u), tf.float32)
    xbar = u[0:2]
    ubar = u[2:8]
    
    xbar = tf.reshape((xbar), (1,1,2))
    d0 = tf.cast((d0), tf.float32) # (2,)
    d0 = tf.reshape((d0), (1,1,2))
    ubar = tf.reshape((ubar), (1,1,6))
    r = tf.cast((r), tf.float32)
    r = tf.reshape((r), (1,1,2))
    # assess the model output
    feed = tf.concat((ubar, xbar), axis=2)
    xout = model(feed) + d0
    # return the difference
    return (xout - r).numpy().reshape((2,))

# Constraint function gradient
@tf.function(experimental_relax_shapes=True)
def drcons1(u, d0, r):
    """
    The gradient of the nonlinear constraint function, that speeds up the optimization routine.
    
    Arguments:
    
    u: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], the inputs parameters for all 3 locations over Nc (control horizon). Additionally ybar ubar, where ubar is the disturbance evaluated control action that leads to ybar (new target evaluated through the disturbance model). Shape = (Nc+8, 6). i.e. Nc = 2, shape = (10, 6)
    
    d0: [HR_dhat, MAP_dhat], the disturbance values at the current time, shape = (2,1)
    
    r: [HR_sp, MAP_sp], the target set points for the HR and MAP, shape = (2,1)
    
    Returns:
    
    dl_du0: Gradient of the nonlinear constraint function w.r.t. the inputs, shape = (2,6*Nc+8)
    
    """

    steps = len(u)
    u = tf.cast((u), tf.float32)
    
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(u)
    # cast to tensors
    
        xbar = u[0:2]
    
        ubar = u[2:8]
    
        xbar = tf.reshape((xbar), (1,1,2))
        d0 = tf.cast((d0), tf.float32) # (2,)
        d0 = tf.reshape((d0), (1,1,2))
        ubar = tf.reshape((ubar), (1,1,6))
        r = tf.cast((r), tf.float32)
        r = tf.reshape((r), (1,1,2))
        # assess the model output
        
        feed = tf.concat((ubar, xbar), axis=2)
        xout = model(feed) + d0
        
        loss_value = xout - r
    
    dl_d0 = tape.jacobian(loss_value, u)
    
    
    dl_du0 = tnp.asarray(dl_d0)
    
    return dl_du0.reshape((2,steps))

####################################################################################
# Augmented function

def g_aug(x,u):
    """
    A function to evaluate the LSTM model predictions
    
    Arguments:
    
    x: [HR_t, MAP_t], the current values of the HR and MAP
    
    u: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], the VNS parameters to be applied.
    
    Returns
    
    output: [HR_t+1, MAP_t+1], the LSTM predicted values for the HR and MAP at the next cardiac cycle
    """

    x = tf.cast((x), tf.float32)
    x = tf.reshape((x), (1,1,2))
    
    u = tf.cast((u), tf.float32)
    u = tf.reshape((u), (1,1,6))
    
    fedd = tf.concat((u,x), axis=2)
    output =model(fedd)
    return output.numpy().reshape((1,2))

####################################################################################
# Prediction Horizon (number of cycles)
Np = 20
# Control Horizon (number of cycles)
Nc = 10
# Plot the output from the simulation?
plotflag = True
####################################################################################
# Controller

# declare the bounds as scipy class
lb = (Nc+1)*[-0.2565130260521042, -0.24876077857228296, -0.262, -0.25480349944383135, -0.244, -0.2514416211441395]
ub = (Nc+1)*[0.7442447729893789, 0.7512392141651547, 0.7374489713901999, 0.7451964954883866, 0.7557925644446, 0.7485583789001786]
lb.insert(0, -np.inf)
lb.insert(0, -np.inf)
ub.insert(0, np.inf)
ub.insert(0, np.inf)
bnds = scipy.optimize.Bounds(lb, ub)

# similarly for the initial guess
initial_guess = (Nc+1)*[0.24448898, 0.25145373, 0.238, 0.24615905, 0.256, 0.24859346]
initial_guess.insert(0, 0.1) # initial guess for HR bar
initial_guess.insert(0, 0.2) # initial guess for MAP bar

def controller(IC, Np, set_point, initial_guess, D):
    """
    This function specifies the controller by optimizing the VNS parameters (inputs)
    
    Arguments:
    
    IC: [HR, MAP], initial condition vector of size (2,) for the current values of the HR and MAP
    
    Np: integer, the number of cycles for the predictive horizon
    
    set_point: [HR_sp, MAP_sp], the target values of the HR and MAP of shape (2,)
    
    initial_guess: [u1 PW, u1 Freq, u2 PW, u2 Freq, u3 PW, u3 Freq], an initial guess for the solver to start its search from, and includes the initial guesses of the additional optimization variables (ybar and ubar) of shape (Nc*6+8,) i.e. Nc = 2, shape = (20,)
    
    D: [HR_dhat, MAP_dhat], the current disturbance values for the HR and MAP of shape (2,)
    
    Returns:
    
    controller_actions: solver object that contains the optimized parameters, along with solver status
    """
    # Define the constraint arguments
    args = (D, set_point)
    
    # Constraints, from the constrain function above
    cons = {'type':'eq', 'fun':rcons, 'args':args, 'jac':drcons1}
    
    
    # Solving the optimization problem using the Sequential Least Squares Programming algorithm from Scipy
    controller_actions = scipy.optimize.minimize(Loss, initial_guess,
                                         args=(IC, Np, set_point, D), method='SLSQP', bounds=bnds,
                                         jac=jac, constraints=cons,
                          options = {'ftol': 1e-4, 'disp': False, 'maxiter':500})

    # Finally, we return the solver class
    return controller_actions
    
####################################################################################
# state for the intra-patient model
state_IC = [217.77, 4.42, 108.84, 92.08, 0.17, 0.04, 0.13, 0.15, 0.12, 1.94, 0]

# Prep for Matlab
state_IC = matlab.double(state_IC)

# cycles of simulation
N = 450

# Declare a matrix to store the physiological variables of interest
ICs = np.zeros((N+1,2))

# create a matrix for storing the VNS actions applied and other metrics
applied_VNS = np.zeros((N,6))
Dhat = np.zeros((N+1,2))
Errorhold = np.zeros((N,2))
lossvals = np.zeros((N,))

# Disturbance gain
Ld = np.array([[0.06, 0.0], [0.0, 0.05]])

# No VNS input
VNS_initial = np.array([[0, 0, 0, 0, 0, 0]])
# Send to matlab
VNS_i = matlab.double(VNS_initial.T.tolist())
# Run the matlab file for one cycle
new_ini, HR_tot, MAP_tot = eng.myplant_intra_patient(state_IC, VNS_i, nargout=3)
# Store the output in the physiological variables matrix
ICs[0,:] = np.asarray([HR_tot, MAP_tot])

# A matrix of our set points
SPs = np.asarray([[356., 150.], [393., 129.], [377., 143.]])


# In case the solver fails, kill the simulation
soltag = True
##########################################################################################################

# Running the feedback simulation
##########################################################################################################
for i in range(N):
    
    # Checking for solver convergence
    if soltag:
    
        # Determine the set point based on cycle
        if i < 150:
            # Normalize the setpoint
            SP = (SPs[0,:] - var_mean) / var_range
        elif i < 300:
            # Normalize the setpoint
            SP = (SPs[1,:] - var_mean) / var_range
        else:
            # Normalize the setpoint
            SP = (SPs[2,:] - var_mean) / var_range
        
        # The initial guess supplied to the solver depends on the optimized paramters
        # if the solver has already optimized parameters, we start looking from those optimized parameters
        guess_i = initial_guess if i ==0 else fin_VNS
    
        # Normalize the initial conditions for the controller
        prev_control_IC = (ICs[0,:] - var_mean) / var_range if i == 0 else (ICs[i-1,:] - var_mean) / var_range
        controller_IC =( ICs[i,:] - var_mean )/ var_range
    
        # Run the controller to determine the optimal VNS parameters
        prev_VNS = ([0.,0.,0.,0.,0.,0.] - VNS_mean)/VNS_range if i == 0 else (applied_VNS[i,:] - VNS_mean) / VNS_range
        
        # Update the states of the controller
        sol_out = controller(controller_IC, Np, SP, guess_i, Dhat[i,:])
        
        # The optimized parameters
        fin_VNS = sol_out.x
        # Updating the solver convergence
        soltag = sol_out.success
        # Logging the loss values
        lossvals[i] = sol_out.fun
    
        # un normalize the optimized VNS paramters, only the first action
        applied_VNS[i,:] = fin_VNS[8:8+6].reshape((1,6))*VNS_range + VNS_mean
    
        # implement the controller action and run the PM for one cycle
        VNS_action = matlab.double(applied_VNS[i,:].reshape((6,1)).tolist())
        new_ini, HR, MAP = eng.myplant_intra_patient(new_ini, VNS_action, nargout=3)
    
        # store the PM output
        ICs[i+1,:] = np.asarray([HR, MAP])
    
        # update the disturbance, Dhat
        # Calculate the rror
        error = ((ICs[i+1,:] - var_mean)/var_range).reshape((1,2)) - g_aug(((ICs[i,:]-var_mean)/var_range), fin_VNS[8:8+6]) - Dhat[i,:].reshape((1,2))
        # Update Dhat variable
        Dhat[i+1,:] = (Dhat[i,:].reshape((2,1)) + Ld@error.reshape((2,1))).reshape((2,))
        # Track the error
        Errorhold[i,:] = error
        
        # Give updates on where the simulation is at
        print("We are on step: ", i, "/", N-1)
    
    else:
        # If the solver fails
        print("We ENDED on step: ", i-1)
        break

####################################################################################
# Plotting

if plotflag:
    xts = np.arange(0, len(ICs[:,0]))
    
    # Plots the Heart Rate with cardiac cycle
    plt.figure(figsize=(12,8))
    plt.subplot(121)
    plt.scatter(xts, ICs[:,0], c='b')
    plt.plot([0, 150, 150, 300, 300, N], [356, 356, 393, 393, 377, 377], c='k')
    plt.ylabel('HR (bpm)')
    plt.xlabel('Cardiac Cycle')

    # Plots the blood pressure with cardiac cycle
    plt.subplot(122)
    plt.scatter(xts, ICs[:,1], c='b')
    plt.plot([0, 150, 150, 300, 300, N], [150, 150, 129, 129, 143, 143], c='k')
    plt.ylabel('MAP (mmHg)')
    plt.xlabel('Cardiac Cycle')
    plt.show()
    
    # Plots the VNS Pulse Widths applied by the controller
    plt.figure(figsize=(12,8))
    plt.subplot(121)
    plt.plot(applied_VNS[:,0], label='Loc 1', c='k')
    plt.plot(applied_VNS[:,2], label='Loc 2', c='b')
    plt.plot(applied_VNS[:,4], label='Loc 3', c='r')
    plt.xlabel('Cardiac Cycle')
    plt.ylabel('Pulse Amplitude (mA)')
    plt.legend()
 
    # Plots the VNS Frequencies applied by the controller
    plt.subplot(122)
    plt.plot(applied_VNS[:,1], label='Loc 1', c='k')
    plt.plot(applied_VNS[:,3], label='Loc 2', c='b')
    plt.plot(applied_VNS[:,5], label='Loc 3', c='r')
    plt.xlabel('Cardiac Cycle')
    plt.ylabel('Pulse Frequency (Hz)')
    plt.legend()
    plt.show()
    
    # Plot the disturbance variable
    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.plot(Dhat[:,0], label='D-HR')
    plt.plot(Dhat[:,1], label='D-MAP')
    plt.legend()
    plt.xlabel('Cardiac Cycle')
    plt.ylabel('Dhat')
    
    # Plot the error
    plt.subplot(212)
    plt.plot(Errorhold[:,0], label='E-HR')
    plt.plot(Errorhold[:,1], label='E-MAP')
    plt.legend()
    plt.xlabel('Cardiac Cycle')
    plt.ylabel('Error')
    plt.show()
    
