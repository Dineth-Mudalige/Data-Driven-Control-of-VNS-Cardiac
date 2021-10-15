% The diseased dynamics model, with overactive sympathetic system
% 7-27-2021
% Andrew Branen

function [ini, HR, MAP] = myplant_sympathetic(ini, usim)
% Get the parameters 
load('par_greyid_sympathetic.mat', 'par_ini') 
par = struct(); 
par = Get_Model_Constants(par, par_ini); 
% All locations are active (non-active will be set to 0) 
psim = ones(size(usim)); 
cycles = size(usim,2); 
% Run the simulation for specified cycles 
[ini, ysim(:, 1:cycles)] = plant(usim(:, 1:cycles), psim(:,1:cycles), cycles, ini, par); 
MAP = ysim(1, 1:cycles); 
HR = ysim(2, 1:cycles); 

end

% Get Model Constants
function [par] = Get_Model_Constants(par, par_ini)
    % Parameters of averaged model
    par.Mean.Cvc = par_ini{1};
    par.Mean.Cao = par_ini{2};
    par.Mean.cT = par_ini{3};
    par.Mean.cR = par_ini{4};
    par.Mean.cEm = par_ini{5};
    par.Mean.cEM = par_ini{6};
    par.Mean.tauT = par_ini{7};
    par.Mean.tauR = par_ini{8};
    par.Mean.tauEm = par_ini{9};
    par.Mean.tauEM = par_ini{10};
    par.Mean.aT = par_ini{11};
    par.Mean.aR = par_ini{12};
    par.Mean.aEm = par_ini{13};
    par.Mean.aEM = par_ini{14};
    par.Mean.bT = par_ini{15};
    par.Mean.Psp = par_ini{16};
    par.Mean.k = par_ini{17};
    
    % Constant of CVS model
    par.CVS.R1 = 0.007;                   % Systemic resistance
    par.CVS.R2 = 0.0002;               % Mitral Valve Resistance
    par.CVS.R3 = 0.006;                 % Aortic Valve Resistance
    par.CVS.C2 = 25;                    % Veneous compliance
    par.CVS.C3 = 1.4;                   % Systemic compliance
    par.CVS.E_min = 0.01;                % End-diastolic Elastance
    par.CVS.E_max = 1.1;                  % End-systolic Elastance
    par.CVS.L = 1e-6;                   % Inertance
    par.CVS.T0 = 60/480;                % Baseline HR

    % Constant of baroreceptor 
    % Parameters for arterial wall deformation
    par.BR.Am = 15.71;                 % Unstressed aortic cross-sectional area
    par.BR.A0 = 3.14;                  % Maximum aortic cross-sectional area
    par.BR.a = 150;                     % Saturation pressure
    par.BR.k = 5;                       % Steepness constant
    % Parameters for mechanoreceptor model
    par.BR.a1 = 0.4;                    % Nerve ending const
    par.BR.a2 = 0.5;                    
    par.BR.b1 = 0.5;                    % Nerve ending relaxation rate
    par.BR.b2 = 2;
    % Parameters for BR firing model
    par.BR.s1 = 2.947e-10;              % Firing constant
    par.BR.s2 = 3.473e-12;
    par.BR.Cm = 37.5e-11;               % Membrane capacitance 
    par.BR.gl = 5.019e-8;               % Membrane conductance
    par.BR.Vth = 0.00116;               % Voltage threshold
    par.BR.Tr = 0.0062;                 % refractory period

    % Parameters of CNS 
    par.CNS.fes_inf = 3;                % Sympathetic rate with maximum afferent inputs
    par.CNS.fes0 = 16;                  % Sympathetic rate with minimum afferent inputs
    par.CNS.kes = 0.07;                 % Steepness constant for sympathetic pathway
    par.CNS.fev_inf = 6;                % Sympathetic rate with maximum afferent inputs
    par.CNS.fev0 = 3;                   % Sympathetic rate with minimum afferent inputs
    par.CNS.kev = 7;                    % Steepness constant for sympathetic pathway
    par.CNS.fas0 = 30;                  % Central Frequency of Afferent Baroreceptor

    % Parameters of efferent pathways
    par.efrt.tau_E = 8/5;               % Time Constant for Elastance Change
    par.efrt.tau_R = 6/5;               % Time Constant for Systemic Resistance Change
    par.efrt.tau_Ts = 2/5;              % Time Constant for Heart Period Change due to Sympathetic Stimulation
    par.efrt.tau_Tv = 1.5/5;            % Time Constant for Heart Period Change due to Vagal Stimulation
    par.efrt.tau_P = 0.5;               % Time Constant for BR sensed pressure
    par.efrt.tau_Z = 1.5;               % TIme constant for BR rate change
    par.efrt.G_E = 0.3;                 % Gain of Systolic Left Ventricular Elastance Change 
    par.efrt.G_R = 0.07;                 % Gain of Systemic Resistance  Change
    par.efrt.G_Ts = -0.015;              % Gain of Heart Period Change by Sympathetic Stimulation
    par.efrt.G_Tv = 0.011;               % Gain of Heart Period Chance by Vagal Stimulation
    par.efrt.fes_min = 3;               % Minimum Sympathetic Frequency
    par.efrt.d_Ts = 0.4; 
    par.efrt.d_Tv = 0.04; 
    par.efrt.d_E = 0.4; 
    par.efrt.d_R = 0.4;  

    % Constant of Device Model
    par.stm.Gas = 178;
    par.stm.Ges = 33;
    par.stm.Gev = 27;
    par.stm.kw = 2.5e-4;
    par.stm.kf = 25;
    par.stm.loc = [1.3  0.1  0.1; ...
                   0.1  1.4  0.1; ...
                   0.1  0.1  1.3];
    par.stm.fsp_as = 38;
    par.stm.fsp_es = 8;
    par.stm.fsp_ev = 8;
end
