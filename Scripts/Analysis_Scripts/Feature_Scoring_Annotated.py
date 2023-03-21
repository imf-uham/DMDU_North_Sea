#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 18:32:43 2021

@author: jan
"""

"""
1) Import the required packages
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence

mpl.rc('image', cmap='plasma')
plasma = mpl.cm.get_cmap('plasma', 17)

"""
2) Set up a function that generates boosted-regression tree models for computing partial 
interacting effects of policies and uncertainties. The purpose of using this function is 
to ascertain that the maximum and minimum number of sustainable years are the same for
all combinations of exploitation metrics (catch, harvest rate) and SR models (Ricker, 
Beverton-Holt)
"""
def make_model(results_file):
    dat = pd.read_csv(results_file) # load stock metrics
    nms = dat.columns
    
    dat = np.array(dat)
    
    n_both = dat[:,nms == 'n_both'] # select objective metric (number of years with sustainable
    # management (i.e., SSB exceeding MSY Btrigger and F equaling or being lower than FMSY))
    
    return([np.min(n_both), np.max(n_both)])

"""
3) Apply the above function on the output files for every combination of policy
metric and SR model. The maximum- and minimum effect strengths over all combinations are
determined, in order to check whether an adjustment to the color scales for effects-
interaction plots must be made in the next step
"""
val_range = [make_model('/home/jan/PhD/EMA/North_Sea_Paper/Catch_Bevertonh.csv'),
        make_model('/home/jan/PhD/EMA/North_Sea_Paper/Harvest_Rate_Bevertonh_Focus.csv'),
        make_model('/home/jan/PhD/EMA/North_Sea_Paper/Catch_Ricker.csv'),
        make_model('/home/jan/PhD/EMA/North_Sea_Paper/Harvest_Rate_Ricker_Focus.csv')]
# retrieve minimum and maximum number of sustainable years for all combinations of exploitation metrics 
# and SR models
  
print(val_range) # since minimum and maximum values are all the same, no adjustment must be made to the
# color scale in effects interaction plots

"""
4) Set up a function that generates boosted-regression tree models for computing partial 
interacting effects and feature importances of policies and uncertainties. This function 
will be used also to plot these effects
"""    
def feature_importance(results_file, params_file, SR_mod, manage):
    interactions = [[[0,1]], [[1,2]], [[2,3]], [[3,4]], [[0,2]], [[0,3]], [[0,4]], [[1,3]], [[1,4]], [[2,4]]]
    # set possible interactions between parameters and policies: catch / harvest rate, log_alpha,
    # log_beta, gamma
    
    var_names = ['Catch', r'log($\alpha$)', r'$\beta$', r'$\gamma$', 'RCP']
        
    if manage == 'harvest':
        var_names[0] = 'Harvest \nrate'
    
    dat = pd.read_csv(results_file) # load stock metrics
    
    nms = dat.columns
    dat = np.array(dat)
    
    params = pd.read_csv(params_file) # load policy levels and values of uncertain parameters
    
    params = np.array(params)[:,[1,2,3,8]] # select parameters used in partial-effects
    # calculation: the level of catch or harvest rate, and the values of the three SR
    # parameters
    
    if manage == 'catch':
        catch = dat[:,1] * 1000**-1
    else:
        catch = dat[:,1] * 100
    
    n_both = dat[:,nms == 'n_both'] # select objective metric (number of years with sustainable
    # management (i.e., SSB exceeding MSY Btrigger and F equaling or being lower than FMSY))
      
    X = np.concatenate([catch.reshape([len(catch),1]), params], axis = 1) # predictor variables
    # (catch or harvest rate) and the three SR parameters
    if 'Bevertonh' in results_file:
        X[:,3] = np.abs(X[:,3]) # sign of the "gamma" parameter is reversed for the Beverton-
    # Holt model to make the SR models better comparable 
    X[X[:,4] == 'sst45',4] = 0
    X[X[:,4] == 'sst85',4] = 1 # choice of climate scenario is transformed into a binary 
    # variable
    X = X.astype('float64')
    
    y = n_both # predicted variable: the number of sustainable years
    
    model = GradientBoostingRegressor() # setting up the
    # boosted regression-tree model
    model.fit(X, y) # fitting the model on the data defined above
        
    importance = model.feature_importances_ # extracting the importances of policy and 
    # uncertain factors estimated by the model
        
    plt.bar(var_names, importance * 100, color = [plasma(3), plasma(15), plasma(15), plasma(15), plasma(15)])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Feature', fontsize=20)
    plt.ylabel('Feature importance [%]', fontsize=20)
    plt.ylim([0,100])
    plt.savefig('/home/jan/PhD/EMA/North_Sea_Paper/feature_'+SR_mod+'_'+manage+'.tiff', bbox_inches='tight', dpi = 600)# plotting the feature importances as a barplot
    
    var_names = ['Catch [kt]', r'log($\alpha$)', r'$\beta$', r'$\gamma$ [$10^{-2}$]', 'RCP']
    if manage == 'harvest':
        var_names[0] = 'Harvest rate [%]'
        
    def plot_part(i): # setting up a function for computing and plotting partial interacting 
    # effects for pairs of policy and uncertain factors
        plt.figure(figsize=(8, 10))
        a = partial_dependence(model, interactions[i], X = X, grid_resolution=50) # computing 
        # partial interacting effects using the boosted-regression tree model
        a_arr = a[0].reshape([len(a[1][0]),len(a[1][1])]) # extracting the effect strengths
        im = plt.imshow(np.transpose(np.flip(a_arr, axis = 1)), aspect = 'auto')
        plt.contour(np.transpose(np.flip(a_arr, axis = 1)), colors = 'k')
        plt.xticks(np.linspace(0,50,num = 5).tolist(), labels = np.round(np.linspace(np.min(catch), np.max(catch),num = 5)).astype('int').tolist(), fontsize=24)
        if i == 0:
            plt.yticks(np.linspace(0,50,num = 5).tolist(), labels = np.flip(np.round(np.linspace(np.min(X[:,1]), np.max(X[:,1]),num = 5)).astype('int')).tolist(), fontsize=24)
        else:
            plt.yticks(np.linspace(0,50,num = 5).tolist(), labels = np.flip(np.round(100 * np.linspace(np.min(X[:,3]), np.max(X[:,3]),num = 5)).astype('int')).tolist(), fontsize=24)
        plt.xlabel(var_names[interactions[i][0][0]], fontsize=34)
        plt.ylabel(var_names[interactions[i][0][1]], fontsize=34)
        cbar = plt.colorbar(im, orientation = 'horizontal', pad = 0.2)
        cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, num = 5))
        cbar.set_ticklabels(np.round(np.linspace(np.min(y), np.max(y), num = 5)).astype('int'))
        cbar.set_label('Sustainable years [N]', fontsize = 34)
        cbar.ax.tick_params(labelsize = 24)
        # plt.show() 
        plt.savefig('/home/jan/PhD/EMA/North_Sea_Paper/interact_'+SR_mod+'_'+str(i)+'_'+manage+'.tiff', bbox_inches='tight', dpi = 600) # plotting partial interactive effects of policy and uncertain parameter
        # as a heatmap
    
    [plot_part(i) for i in [0,5]] # plotting partial interactive effects for catch / harvest 
    # rate, and the parameters log_alpha and gamma

feature_importance('/home/jan/PhD/EMA/North_Sea_Paper/Catch_Bevertonh.csv',
                                '/home/jan/PhD/EMA/North_Sea_Paper/Catch_Bevertonh_Params.csv', 'bevertonh', 'catch')
feature_importance('/home/jan/PhD/EMA/North_Sea_Paper/Harvest_Rate_Bevertonh_Focus.csv',
                        '/home/jan/PhD/EMA/North_Sea_Paper/Harvest_Rate_Bevertonh_Focus_Params.csv', 'bevertonh', 'harvest')
feature_importance('/home/jan/PhD/EMA/North_Sea_Paper/Catch_Ricker.csv',
                        '/home/jan/PhD/EMA/North_Sea_Paper/Catch_Ricker_Params.csv', 'ricker', 'catch')
feature_importance('/home/jan/PhD/EMA/North_Sea_Paper/Harvest_Rate_Ricker_Focus.csv',
                        '/home/jan/PhD/EMA/North_Sea_Paper/Harvest_Rate_Ricker_Focus_Params.csv', 'ricker', 'harvest')
# plot feature importance and partial interacting effects for all combinations of catch, harvest
# rate, Ricker SR model and Beverton-Holt SR model



