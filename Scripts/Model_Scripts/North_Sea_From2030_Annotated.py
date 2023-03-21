#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:56:46 2020

@author: u/u300881
"""

import warnings
warnings.filterwarnings("ignore")

"""
1) Importing packages
"""
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import os
import copy
from ema_workbench import (Model, RealParameter, CategoricalParameter, ScalarOutcome, Constant)
from ema_workbench import (ema_logging, MultiprocessingEvaluator)
from ema_workbench.em_framework import samplers, util
from multiprocessing import Pool

"""
2) Importing the temperature data

Importing NOAA reconstructions (Huang et al., 2017) (i.e. "observed data") and projection- and hindcast data from 
CERES physical regional ocean model (Peck et al., 2020). We calculate the median difference between reconstructions 
and hindcast  data, and add the offset to the model projections, as a means of bias correction
"""
robjects.r['load']("/home/u/u300881/PhD/Data/Physical_Data/tot_sst.RData")
sst_obs_ref = np.array(robjects.r['tot_sst'])[4,23:][7:]
# Reconstructed Sea Surface Temperature, Version 5 (ERSSTv5) (Huang et al., 2017)
# 4th row = North Sea, columns = 1963-2019 (including 2019); start from 1963+7 = 1970 to match start year of 
# hindcast of physical ocean model

files = os.listdir('/home/u/u300881/PhD/Data/Physical_Data/Projections/Henrike_Thuenen_Historical_Extracted_AllVars/')
files = np.sort(files).tolist()
temp_list = []
for i in range(len(files)):
    robjects.r['load']('/home/u/u300881/PhD/Data/Physical_Data/Projections/Henrike_Thuenen_Historical_Extracted_AllVars/'+files[i])
    temp_list.append(np.nanmean(np.array(robjects.r['outp'])[0,:]))
# NC files (= hindcast of physical ocean model) cover period 1970-2005 (including 2005)
    
sst45 = pd.read_csv('/home/u/u300881/PhD/Data/Physical_Data/Projections/Henrike_Thuenen_RCP45_Extracted/EnvData_Henrike_NS_RCP45.csv') 
# projection data for RCP4.5 (Peck et al., 2020)
sst45_mod_ref = np.concatenate([temp_list, np.array(sst45)[0,:14]]) # initial projection data that can be compared with
# reconstructed data (2006-2019) are concatenated with model hindcast

sst45_corr = -np.median(sst45_mod_ref - sst_obs_ref) # correction offset between model hindcast plus initial projection
# data and NOAA reconstructions; entire period: 1970-2019

sst45 = (np.array(sst45)[0,12:94]).astype('float') + sst45_corr # bias-corrected temperature projection
# period: 2018-2099 (last column = variable names)

sst45 = sst45[int(2029-2018):] # limit time series to 2029-2099, since projection is supposed to start in 2030

sst85 = pd.read_csv('/home/u/u300881/PhD/Data/Physical_Data/Projections/Henrike_Thuenen_RCP85_Extracted/EnvData_Henrike_NS_RCP85.csv')
 # projection data for RCP8.5 (Peck et al., 2020)
sst85_mod_ref = np.concatenate([temp_list, np.array(sst85)[0,:14]]) # initial projection data that can be compared with
# reconstructed data (2006-2019) are concatenated with model hindcast

sst85_corr = -np.median(sst85_mod_ref - sst_obs_ref) # correction offset between model hindcast plus initial projection
# data and NOAA reconstructions; entire period: 1970-2019

sst85 = (np.array(sst85)[0,12:94]).astype('float') + sst85_corr # add offset to bias-correct temperature projection
# period: 2018-2099 (last column = variable names)

sst85 = sst85[int(2029-2018):] # limit time series to 2029-2099, since projection is supposed to start in 2029 (initial population projection: 
# 2030; initial catch projection: 2029)

"""
3) Importing the stock-related data. These are the fishing- and natural mortalities, weight in the stock, weight in the catch, 
maturity rates and numbers of fish in the stock. Data are derived from ICES Stock Assessment Report (ICES, 2021)
"""     
f_old = pd.read_csv('/home/u/u300881/PhD/Data/Lindegren_Papers/Cod_North_Sea/FatAge.txt', sep = '	') # natural mortality per age-class and year
f_old = np.transpose(np.array(f_old))
f_old = f_old[1:,:]
f_ly = np.median(f_old[:,-5:], axis = 1) # median of fishing mortalities from final five assessment years

f_frac = np.median(f_old[:,-5:], axis = 1) / np.max(np.median(f_old[:,-5:], axis = 1), axis = 0) # calculating age-specific catchability for later
# application in the calculation of exploitation pattern
   
m_old = pd.read_csv('/home/u/u300881/PhD/Data/Lindegren_Papers/Cod_North_Sea/MatAge.txt', sep = '	') # natural mortality per age-class and year
m_old = np.transpose(np.array(m_old))
m_old = m_old[1:,:]
m_ly = np.median(m_old[:,-5:], axis = 1) # median of natural mortalities from final five assessment years taken as constant
m = np.zeros([len(m_old), len(sst45)])
m[:,:] = np.transpose(np.array([np.array([m_ly[i] for i in range(len(m_ly))]) for j in range(71)])) # applying these values for all
# future years

wt_old = pd.read_csv('/home/u/u300881/PhD/Data/Lindegren_Papers/Cod_North_Sea/WatAge.txt', sep = '	') # weight in the stock per age-class and year
wt_old = np.transpose(np.array(wt_old))
wt_old = wt_old[1:,:]
wt_ly = np.median(wt_old[:,-5:], axis = 1) # median of weights from final five assessment years taken as constant
wt = np.zeros([len(wt_old), len(sst45)])
wt[:,:] = np.transpose(np.array([np.array([wt_ly[i] for i in range(len(wt_ly))]) for j in range(71)])) # applying these values for all
# future years

wt_ct_old = pd.read_csv('/home/u/u300881/PhD/Data/Lindegren_Papers/Cod_North_Sea/WatAge_Catch.txt', sep = '	') # weight in the catch  
# per age-class and year
wt_ct_old = np.transpose(np.array(wt_ct_old))
wt_ct_old = wt_ct_old[1:,:]

wt_ct_old_full = copy.deepcopy(wt_ct_old)
wt_ct_old[5,:] = np.median(wt_ct_old[5:,:], axis = 0) # weights are provided for each age-class in the plus group; since the
# population model uses the plus group, not the single age classes, the median weight of these age classes is calculated (a
# copy with values for all age classes is kept for calculation of past profit)
wt_ct_old = wt_ct_old[:6,:]

wt_ct_ly = np.median(wt_ct_old[:,-5:], axis = 1) # median of weights from final five assessment years taken as constant
wt_ct = np.zeros([len(wt_ct_old), len(sst45)])
wt_ct[:,:] = np.transpose(np.array([np.array([wt_ct_ly[i] for i in range(len(wt_ct_ly))]) for j in range(71)])) # applying these values for all
# future years

mt_old = pd.read_csv('/home/u/u300881/PhD/Data/Lindegren_Papers/Cod_North_Sea/PatAge.txt', sep = '	') # maturity rates per age-class and year
mt_old = np.transpose(np.array(mt_old))
mt_old = mt_old[1:,:]
mt_ly = np.median(mt_old[:,-5:], axis = 1) # median of maturity rates from final five assessment years taken as constant
mt = np.zeros([len(mt_old), len(sst45)])
mt[:,:] = np.transpose(np.array([np.array([mt_ly[i] for i in range(len(mt_ly))]) for j in range(71)])) # applying these values for all
# future years

numbers = pd.read_csv('/home/u/u300881/PhD/Data/Lindegren_Papers/Cod_North_Sea/NatAge.txt', sep = '	') # numbers of fish in the stock per age-class 
# and year
numbers = np.transpose(np.array(numbers))
numbers = numbers[1:,:]
 
Btrigger = 97777 # sustainability reference value #1 (MSYBtrigger)
Fmsy = 0.28 # sustainability reference value #2 (Fmsy)

inissb = Btrigger * ((wt_ly * mt_ly * np.median(numbers[:,-5:], axis = 1)) / np.sum(wt_ly * mt_ly * np.median(numbers[:,-5:], axis = 1)))
inipop = inissb / (wt_ly * mt_ly) # calculating initial population numbers from initial SSB values, assuming that the stock is at MSYBtrigger in 
# that year: age-specific SSB is calculated from MSYBtrigger, and initial population size is calculated from that (for details see manuscript)

"""
5) Importing the catch data. These are the full reported catch (including discards and recreational catch) in tonnes,
as well as catch numbers at age, as estimated by the ICES stock-assessment model (ICES, 2021). These are used in the
calculation of past profits
"""
catch_old_full = pd.read_csv('/home/u/u300881/PhD/Data/Lindegren_Papers/Cod_North_Sea/Catch_AY.txt', sep = '	') # catch in numbers per age-class 
# and year
catch_old_full = np.transpose(np.array(catch_old_full))
catch_old_full = catch_old_full[1:,:]
    
catch_tot = pd.read_csv('/home/u/u300881/PhD/Data/Lindegren_Papers/Cod_North_Sea/Catch.txt', sep = '	') # total catches (in weight)
catch_tot = np.transpose(np.array(catch_tot))
catch_tot = catch_tot[1,:]

"""
6) Calculating economic price classes for the projection and past profit based on an economic model presented in 
Schenk et al. (in prep.). Weight- and price classes and parameter values were also taken from Schenk et al. (in prep.)
"""
eco_weight_class = np.array([0.3, 1, 2, 4, 7, np.inf, np.nan]) # economic weight classes
eco_price_class = np.array([1.500, 2.050, 2.653, 3.133, 1.606, np.inf, np.nan]) # corresponding price classes

compmat = np.array([[(wt_ct_ly[i] >= eco_weight_class[j]) & (wt_ct_ly[i] < eco_weight_class[j+1]) for j in range(len(wt_ct_ly))] for i in range(len(wt_ct_ly))])
weight_class = [eco_weight_class[:6][compmat[i,:]] for i in range(len(wt_ct_ly))] # assigning current weight in the catch to economic weight classes

price_class = np.array([eco_price_class[:6][compmat[i,:]][0] for i in range(len(wt_ct_ly))]) # determining price classes for the projections

co = 6.29
teta = 0.0106
chi = 0.24 # parameters of the cost function (see Schenk et al., in prep.)
    
def make_past_profit(): # function for calculating past profit
    years = np.arange(1963,2019)
    
    compmat = np.array([[[(wt_ct_old_full[i,h] >= eco_weight_class[j]) & (wt_ct_old_full[i,h] < eco_weight_class[j+1]) 
                         for j in range(len(eco_weight_class)-1)] for i in range(len(wt_ct_old_full))] for h in range(len(years))])
    weight_class_past = [[eco_weight_class[:6][compmat[i,j]] for j in range(len(wt_ct_old_full))] for i in range(len(years))] # assigning past weight in the
    # catch to economic weight classes
    
    price_class_past = [[eco_price_class[:6][compmat[i,j]] for j in range(len(wt_ct_old_full))] for i in range(len(years))] # determining past price classes
    price_class_past = np.array(price_class_past)
    
    check = np.array([[price_class_past[i,j].size == 0 for j in range(11)] for i in range(56)])
    price_class_past[check] = np.nanmin(eco_price_class)
    price_class_past = np.array([[(price_class_past[i,j][0] if type(price_class_past[i,j]) == np.ndarray else price_class_past[i,j]) 
                                  for j in range(11)] for i in range(56)]) # where no price class could be determined (i.e. fish have too low weight to fall
    # into the lowest economic weight class), the lowest price class is applied
            
    catchability = f_old / np.max(f_old, axis = 0) # calculating past catchabilities
    
    catchable_biom = numbers * wt_old * catchability # determining past catchable biomass
    
    revenue = np.array([[catch_old_full[i,j] * wt_ct_old_full[i,j] * price_class_past[j,i] * 1000 for j in range(56)] for i in range(11)])
    # calculating past revenue
    
    ct = co * np.exp(teta * (years - 2019))
    cost = ct * (np.sum(catchable_biom, axis = 0)**-chi) * catch_tot # calculating past cost
    
    profit = (np.sum(revenue, axis = 0) - cost) # calculating past profit
    
    return(profit)

profit_past = make_past_profit() # calculating past profits
past_profit = np.mean(profit_past[np.arange(1963,2019) >= 2000]) # determining the average profit over years 2000 to 2019 (used as custom economic reference
# point for calculation of profitability risk)

"""
7) Function to solve the catch equation for fishing mortality. In the population model, an estimate of fishing mortality is 
# calculated using the Pope (1972) equation, which assumes that all catch is taken in the middle of the year (this is a simplification).
# The "nlminb" optimizer function (R Core Team, 2020) is used to better estimate the fishing mortality using the Pope estimate as start value
"""
def f_opt(f_est, catch, numbers, mmort):
    robjects.r('calc_f <- function(f_est, catch, numbers, mmort){catch_est <- (f_est / (f_est + mmort)) * numbers * (1 - exp(-(f_est + mmort))); div <- (catch_est - catch)^2; return(div)}')
    # function calculates catch from fishing-mortality estimate (derived using Pope's (1972) equation)
    
    robjects.r('solved <- nlminb(start = '+str(f_est)+', objective = calc_f, catch = '+str(catch)+', numbers = '+str(numbers)+', mmort = '+str(mmort)+')')
    # optimizer changes fishing mortality such that squared difference between estimated and calculated catch is minimized
    
    solved = robjects.r('solved$par')[0]
    solved = solved * (robjects.r('solved$convergence')[0] == 0) + f_est * (1 - (robjects.r('solved$convergence')[0] == 0))
    # if optimization did not converge (convergence code not "0"), the Pope estimate is returned instead
    
    return(solved)

"""
8) The population model. Projects future population numbers, SSB and fishing mortality, as well as profit
"""
def cod_problem(
         a_rk = 10.3426067714013,
         b_rk = -12.2713114051505,
         c_rk = 0.795182416822305, # the Ricker-stock-recruitment-model parameters
         
         a_bh = 11.287307498387,
         b_bh = -11.1600090984174,
         c_bh = -0.850163740861109, # the Beverton-Holt-stock-recruitment model parameters
         # (note that "a_rk" / "a_bh" and "b_rk" / "b_bh" are actually the logarithms of these SR parameters; for details see manuscript)
         
         catch_mtp = 0, # the level of catch tested
         hrv_rate = 0, # the level of harvest rate tested
         
         metric_choice = 'harvest', # the exploitation metric applied (catch or harvest rate)
         sr_choice = 'bh', # the stock-recruitment model used (Beverton-Holt or Ricker)
         
         sst = 'sst45', # the climate scenario used (RCP4.5 ("sst45") or RCP8.5 ("sst85"))
         
         stop_year = 2100, # final year of the projection
         
         full_output = 'no' # should the function return the number of sustainable years (for exploration) or all stock metrics?
         ):
    
    sst = eval(sst)
    sst = sst[:(len(sst) - (2100 - stop_year))]
    
    pop = np.zeros([6,len(sst)]) # setting up empty array for projected population numbers
    pop[:,0] = inipop # first column is the initial population size
    ssb = np.zeros([6,len(sst)]) # setting up empty array for projected age-specific SSB
    ssb[:,0] = inissb # first column is the initial age-specific SSB
    sbsum = np.zeros((len(sst),)) # setting up empty vector for projected total SSB
    sbsum[0] = np.sum(ssb[:,0]) # first element is initial total SSB
    fs = np.zeros([6,len(sst)]) # setting up empty array for projected fishing mortality
    fs[:,:] = 999 # fishing mortalities all initially set to 999, for cases of projection breakdown / stock collapse
    
    rl_catch_n = np.zeros((6,len(sst))) # setting up empty array for projected age-specific realized catch
    rl_catch_w = np.zeros((len(sst),)) # setting up empty vector for projected total realized catch in weight
    rl_hr = np.zeros((len(sst),)) # setting up empty vector for projected harvest rate
    
    profit = np.zeros((len(sst),)) # setting up empty vector for projected profit
    
    for t in range(1,len(sst)): # looping over the projection yeats
    
        pred_ricker = np.sum(ssb[:,t-1]) * np.exp(a_rk - np.exp(b_rk) * np.sum(ssb[:,t-1]) - c_rk * sst[t-1])
        # predicted recruitment (Ricker function (Ricker, 1975); note that a re-arranged equation is used here, for details see manuscript)
        pred_bh = np.exp(c_bh * sst[t-1] + (a_bh + np.log(np.sum(ssb[:,t-1]))) - np.log(1 + np.exp(b_bh + np.log(np.sum(ssb[:,t-1])))))
        # predicted recruitment (Beverton-Holt function (Hilborn & Walters, 1992); note that a re-arranged equation is used here, for details see manuscript)
        
        if sr_choice == 'rk':
            pop[0,t] = pred_ricker
        elif sr_choice == 'bh':
            pop[0,t] = pred_bh # recruitment prediction is inserted into population matrix (first entry for year t)
        
        if (np.sum(pop[0,t]) > 0) == False:
            pop[0,t] = np.min(numbers[0,:]) * 0.1 # if recruitment is zero, it is set to 10 % of minimum observed recruitment
        
        pot_catch = f_frac * pop[:,t-1] * np.exp(-m_ly * 0.5)
        pot_catch = pot_catch * wt_ct_ly
        catch_frac_ct = pot_catch / np.sum(pot_catch) # calculating exploitation pattern for distributing catch in weight over age classes.
        # A reference catch (population size reduced by half a year of natural mortality and multiplied by age-specific catchability), is 
        # used to determine the pattern
            
        if metric_choice == 'catch':
            rl_catch_n[:,t-1] = (catch_frac_ct*catch_mtp)/wt_ct_ly
            rl_catch_n[:,t-1] = np.min([(catch_frac_ct*catch_mtp)/wt_ct_ly, pop[:,t-1]*np.exp(-0.5*m_ly)], axis = 0)
            rl_catch_w[t-1] = np.sum([rl_catch_n[i,t-1] * wt_ct_ly[i] for i in range(6)]) 
            # calculate realized catch. First, total catch in kilo-tonnes is converted to catch in numbers for the specific age-class by 
            # multiplying the catch with the weight-based exploitation pattern (note that weight in the catch was used in calculating the
            # exploitation pattern). This catch in weight per age-class is then divided by weight in the catch per age-class to calculate 
            # catch in numbers per age-class. This is then compared to the numbers of fish in the stock per age-class (after 0.5 years of 
            # natural mortality only); for age-classes in  which catch > numbers in the stock, the catch is reduced to this number. Realized 
            # catch in weight is realized catch in numbers multiplied with weight in the catch, summed over all age-classes
            
            for i in range(len(pop)-1): # loop over age classes
                f_est = -((np.log(((pop[i,t-1]*np.exp(-0.5*m_ly[i])) - rl_catch_n[i,t-1]) * np.exp(-0.5*m_ly[i])) - np.log(pop[i,t-1])) + m_ly[i])
                # a fishing-mortality estimate is calculated using the Pope equation: 
                # Nt+1 = (Nt * exp(-M * 0.5) - Ct) * exp(-M * 0.5)
                # F = -((log(Nt) - log(Nt-1)) + Mt)
                # (where N = numbers, M = natural mortality, F = fishing mortality, C = catch, t = time (1 year))
                
                if (np.isnan(f_est) == False) & (np.isinf(f_est) == False): # fishing-mortality estimate can turn to NA or Inf when stock is close
                # to collapse or is collapsed (e.g. when population size is smaller than or equal to realized catch)
                    f = f_opt(f_est, rl_catch_n[i,t-1], pop[i,t-1], m_ly[i]) # estimate fishing mortality, using the Pope estimate as a start guess, 
                    # via optimization algorithm
                    pop[i+1,t] = np.max([0, pop[i,t-1] * np.exp(-(m_ly[i] + f))]) # calculating population numbers of subsequent age-class in subsequent 
                    # year. This results in population numbers for age-classes 2-6 (age-class 1 = recruits, see above)
                    fs[i,t-1] = f # recording the fishing mortalities calculated above
            
            f_est = -((np.log(((pop[5,t-1]*np.exp(-0.5*m_ly[5])) - rl_catch_n[5,t-1]) * np.exp(-0.5*m_ly[5])) - np.log(pop[5,t-1])) + m_ly[5])
            
            if (np.isnan(f_est) == False) & (np.isinf(f_est) == False):
                f = f_opt(f_est, rl_catch_n[5,t-1], pop[5,t-1], m_ly[5])
                pgroup_add = np.max([0, pop[5,t-1] * np.exp(-(m_ly[5] + f))])
                pop[5,t] = pop[5,t] + pgroup_add
                fs[5,t-1] = f
            # procedures of calculating realized catch and fishing mortality are repeated for the plus group. Plus-group survivors are added to those
            # of age-class 5
        
            rl_hr[t-1] = rl_catch_w[t-1] / np.sum(wt_ct_ly*pop[:,t-1]*np.exp(-0.5*m_ly))
        else:
            catch_mtp = (hrv_rate * np.sum(pop[:,t-1]*np.exp(-0.5*m_ly)*wt_ct_ly)) # calculating total catch in weight from harvest rate (in % stock
            # biomass) and stock size after 0.5 years of natural mortality only
            
            rl_catch_n[:,t-1] = (catch_frac_ct*catch_mtp)/wt_ct_ly
            rl_catch_n[:,t-1] = np.min([(catch_frac_ct*catch_mtp)/wt_ct_ly, pop[:,t-1]*np.exp(-0.5*m_ly)], axis = 0)
            rl_catch_w[t-1] = np.sum([rl_catch_n[i,t-1] * wt_ct_ly[i] for i in range(6)]) 
            
            for i in range(len(pop)-1):
                f_est = -((np.log(((pop[i,t-1]*np.exp(-0.5*m_ly[i])) - rl_catch_n[i,t-1]) * np.exp(-0.5*m_ly[i])) - np.log(pop[i,t-1])) + m_ly[i])
                if (np.isnan(f_est) == False) & (np.isinf(f_est) == False):
                    f = f_opt(f_est, rl_catch_n[i,t-1], pop[i,t-1], m_ly[i])
                    pop[i+1,t] = np.max([0, pop[i,t-1] * np.exp(-(m_ly[i] + f))])
                    fs[i,t-1] = f
            
            f_est = -((np.log(((pop[5,t-1]*np.exp(-0.5*m_ly[5])) - rl_catch_n[5,t-1]) * np.exp(-0.5*m_ly[5])) - np.log(pop[5,t-1])) + m_ly[5])
            if (np.isnan(f_est) == False) & (np.isinf(f_est) == False):
                f = f_opt(f_est, rl_catch_n[5,t-1], pop[5,t-1], m_ly[5])
                pgroup_add = np.max([0, pop[5,t-1] * np.exp(-(m_ly[5] + f))])
                pop[5,t] = pop[5,t] + pgroup_add
                fs[5,t-1] = f
            # procedures for calculating realized catch, fishing mortality and stock numbers (including plus group) are identical to those above
            
            rl_hr[t-1] = rl_catch_w[t-1] / np.sum(wt_ct_ly*pop[:,t-1]*np.exp(-0.5*m_ly)) # calculating realized harvest rate as the ratio between
            # realized catch in weight and (hypothetical) stock size after 0.5 years of natural mortality only
        
        for i in range(len(pop)):
            ssb[i,t] = pop[i,t] * wt[i,t] * mt[i,t] # calculating age-specific SSB of the survivors and recruits
            
        sbsum[t] = np.sum(ssb[:,t]) # summing the age-specific SSBs over age classes
        
        ## economic calcuations (for details see manuscript and Schenk et al. (in prep.) ##
    
        catchable_biomass = pop[:,t-1] * wt_ly * fs[:,t-1] / np.max(fs[:,t-1]) # estimation of catchable biomass, i.e. amount of fish biomass that 
        # is available for harvesting given current gear capacity
        revenue = (rl_catch_n[:,t-1] * wt_ct_ly) * price_class * 1000 # calculation of revenue based on age-specific weight in the catch and 
        # market prices (see data import in step 6))
        ct = co * np.exp(teta * (((2029-2018) + t-1) - (2019-2018))) # calculating cost of fishing, which depends on realized catch, catchable biomass 
        # and temporal dynamics (e.g. increasing labour costs (Schenk et al., in prep.)). Cost function is parameterized with 2019 as baseline year, 
        # hence the factor multiplied with teta must be 10 for the first projection year (initially 11 = 2029-2019)
        cost = ct * (np.sum(catchable_biomass)**-chi) * (rl_catch_w[t-1])
        profit[t-1] = (np.sum(revenue) - cost) / 1e+6 # calculating profit as the difference of revenue and loss
             
        # end loop over years
        
    n_overbmsy = np.sum(sbsum > Btrigger) # determining the number of projection years in which SSB is > MSYBtrigger
    n_underfmsy = np.sum(np.mean(fs[1:4,:], axis = 0) <= Fmsy) # determining the number of years in which Fbar (mean fishing mortality over ages 2-4)
    # is <= Fmsy
    n_both = np.sum((sbsum > Btrigger) & (np.mean(fs[1:4,:], axis = 0) <= Fmsy)) # determining the number of years in which both SSB is > MSYBrigger
    # and Fbar is <= Fmsy
    
    if full_output == 'no':
        return(n_overbmsy, n_underfmsy, n_both) # output returned when performing the exploratory modeling (used by Monte-Carlo sampler)
    else:
        return(pop, sbsum, fs, rl_catch_w, rl_catch_n, n_overbmsy, n_underfmsy, n_both, profit, rl_hr) # output returned when exploration
    # has been completed and stock metrics are to be saved for later analysis

"""
9) Preparation of the exploratory modeling. Here, the model is initialized in the context of the "EMA workbench" package: 
Uncertainties and model constants, as well as policies are set
"""
cod_model = Model('codproblem', function=cod_problem) # initializing the population model for the exploration process

cod_model.time_horizon = len(sst45)-(2100-2100)

# cod_model.uncertainties = np.concatenate([np.array([RealParameter('a_bh', 11.287307498387-1.93861002201588, 11.287307498387+1.93861002201588),
#                             RealParameter('b_bh', -11.1600090984174-0.795395269535202, -11.1600090984174+0.795395269535202),
#                             RealParameter('c_bh', -0.850163740861109-0.161641890707372, -0.850163740861109+0.161641890707372),
#                             CategoricalParameter('sst', ['sst45', 'sst85'])])]).tolist() 

cod_model.uncertainties = np.concatenate([np.array([RealParameter('a_rk', 10.3426067714013-1.67420856232342, 10.3426067714013+1.67420856232342),
                            RealParameter('b_rk', -12.2713114051505-0.369325767802175, -12.2713114051505+0.369325767802175),
                            RealParameter('c_rk', 0.795182416822305-0.154264200660738, 0.795182416822305+0.154264200660738),
                            CategoricalParameter('sst', ['sst45', 'sst85'])])]).tolist()
# setting the uncertainties sampled in the exploration process. These include the stock-recruitment-model parameters (log_alpha, log_beta and gamma),
# as well as the climate scenario. Two separates runs of the script are performed for the two stock-recruitment models, since the meaning of the
# stock-recruitment parameters is not entirely equivalent between the two models. For the stock-recruitment parameters, the range to be sampled is
# given (mean estimate minus standard error --> mean estimate plus standard error)

# cod_model.constants = [Constant('metric_choice', 'harvest'), Constant('sr_choice', 'bh'), Constant('full_output', 'no'), 
#                         Constant('stop_year', 2100)]
# cod_model.constants = [Constant('metric_choice', 'catch'), Constant('sr_choice', 'bh'), Constant('full_output', 'no'), 
#                         Constant('stop_year', 2100)]

# cod_model.constants = [Constant('metric_choice', 'harvest'), Constant('sr_choice', 'rk'), Constant('full_output', 'no'), 
#                         Constant('stop_year', 2100)]
cod_model.constants = [Constant('metric_choice', 'catch'), Constant('sr_choice', 'rk'), Constant('full_output', 'no'), 
                        Constant('stop_year', 2100)]
# setting the function constants. These include the choice of the exploitation metric and the choice of the stock-recruitment model (it is treated
# as an uncertainty by running the script twice with the constant changed)

# cod_model.levers = [RealParameter('hrv_rate', 0, 1.0)]
cod_model.levers = [RealParameter('catch_mtp', 0, np.sum(inipop * wt_ct_ly * np.exp(-m_ly * 0.5)) * 1)]
# setting the policies to be explored. This is either the level of catch or the level of harvest rate that is maintained constant over all projection
# years. For harvest rate, we initially set a maximum of 100 %, i.e. complete removal of the stock. That was later reduced to 25 %, since no marked
# changes to sustainability were noted above that level. For catch, the maximum was set as the initial population size after 0.5 years of natural 
# mortality, multiplied with weight in the catch and summed over all age classes. This approximates the full initial stock biomass

cod_model.outcomes = [ScalarOutcome('n_overbmsy', kind=ScalarOutcome.MAXIMIZE),
                       ScalarOutcome('n_underfmsy', kind=ScalarOutcome.MAXIMIZE),
                       ScalarOutcome('n_both', kind=ScalarOutcome.MAXIMIZE)] # setting the model outcomes. These are required by the monte-Carlo sampler

ema_logging.log_to_stderr(ema_logging.INFO) # technical requirement

n_scenarios = 200 # setting the number of uncertain scenarios (stock-recruitment-model parameterizations and climate scenarios) to be tried
n_policies = 100 # setting the number of policies (catch- or harvest-rate levels) to be tried

fn = './{} scenarios {} policies.tar.gz'.format(n_scenarios, n_policies) # technical requirement
counter = util.Counter() # technical requirement

policies = samplers.sample_levers(cod_model, n_policies, 
                                  sampler=samplers.MonteCarloSampler())

"""
10) Performing the exploratory modeling. The model is first run "internally" by the EMA workbench, and only the objective
values (number of sustinable years) and parameters and policies are returned. A second pass is then necessary to save the
stock metrics
"""
with MultiprocessingEvaluator(cod_model) as evaluator:
    results = evaluator.perform_experiments(n_scenarios, policies) # performing the exploration of uncertainties and policies. The total number of
    # model runs equals the product of n_scenarios and n_policies

params = results[0]
objectives = results[1]

def make_preds(i): # function to perform the model runs a second time, in order to save the generated stock metrics
    # results = cod_problem(a_bh = params['a_bh'][i], b_bh = params['b_bh'][i], c_bh = params['c_bh'][i], sst = params['sst'][i],
    #         hrv_rate = params['hrv_rate'][i], 
    #         metric_choice = 'harvest', sr_choice = 'bh', full_output = 'yes')
    # results = cod_problem(a_bh = params['a_bh'][i], b_bh = params['b_bh'][i], c_bh = params['c_bh'][i], sst = params['sst'][i],
    #         catch_mtp = params['catch_mtp'][i], 
    #         metric_choice = 'catch', sr_choice = 'bh', full_output = 'yes')
    
    # results = cod_problem(a_rk = params['a_rk'][i], b_rk = params['b_rk'][i], c_rk = params['c_rk'][i], sst = params['sst'][i],
    #         hrv_rate = params['hrv_rate'][i], 
    #         metric_choice = 'harvest', sr_choice = 'rk', full_output = 'yes')
    results = cod_problem(a_rk = params['a_rk'][i], b_rk = params['b_rk'][i], c_rk = params['c_rk'][i], sst = params['sst'][i],
            catch_mtp = params['catch_mtp'][i], 
            metric_choice = 'catch', sr_choice = 'rk', full_output = 'yes')
    # generating the output of one model run ("i"). Note that the parameters returned by the "internal" exploration of EMA workbench
    # are used as input here
    
    # results = np.concatenate([params['hrv_rate'][i].reshape([1,]), 
    #                           results[0][0,:], results[0][1,:], results[0][2,:], results[0][3,:], results[0][4,:], results[0][5,:],
    #                           results[2][0,:], results[2][1,:], results[2][2,:], results[2][3,:], results[2][4,:], results[2][5,:],
    #                           results[4][0,:], results[4][1,:], results[4][2,:], results[4][3,:], results[4][4,:], results[4][5,:],
    #                    results[1], results[3], results[5].reshape([1,]), results[6].reshape([1,]), results[7].reshape([1,]), results[8], results[9]])
    
    results = np.concatenate([params['catch_mtp'][i].reshape([1,]), 
                              results[0][0,:], results[0][1,:], results[0][2,:], results[0][3,:], results[0][4,:], results[0][5,:],
                              results[2][0,:], results[2][1,:], results[2][2,:], results[2][3,:], results[2][4,:], results[2][5,:],
                              results[4][0,:], results[4][1,:], results[4][2,:], results[4][3,:], results[4][4,:], results[4][5,:],
                        results[1], results[3], results[5].reshape([1,]), results[6].reshape([1,]), results[7].reshape([1,]), results[8], results[9]])
    # results are concatenated into a large vector
    
    if i in np.arange(1000, len(params), 100):
        print(i)
    return(results)

with Pool() as p:
    time_series = p.map(make_preds, range(len(params))) # performing the second round of model runs via parallel processing

time_series = np.array(time_series)
time_series = pd.DataFrame(time_series)

# time_series.columns = ['harvest']+['n_age_1_year_'+str(2029+i) for i in range(71)]+['n_age_2_year_'+str(2029+i) for i in range(71)]+['n_age_3_year_'+str(2029+i) for i in range(71)]+['n_age_4_year_'+str(2029+i) for i in range(71)]+['n_age_5_year_'+str(2029+i) for i in range(71)]+['n_age_6_year_'+str(2029+i) for i in range(71)]+['f_age_1_year_'+str(2029+i) for i in range(71)]+['f_age_2_year_'+str(2029+i) for i in range(71)]+['f_age_3_year_'+str(2029+i) for i in range(71)]+['f_age_4_year_'+str(2029+i) for i in range(71)]+['f_age_5_year_'+str(2029+i) for i in range(71)]+['f_age_6_year_'+str(2029+i) for i in range(71)]+['crn_age_1_year_'+str(2029+i) for i in range(71)]+['crn_age_2_year_'+str(2029+i) for i in range(71)]+['crn_age_3_year_'+str(2029+i) for i in range(71)]+['crn_age_4_year_'+str(2029+i) for i in range(71)]+['crn_age_5_year_'+str(2029+i) for i in range(71)]+['crn_age_6_year_'+str(2029+i) for i in range(71)]+['ssb_year_'+str(2029+i) for i in range(71)]+['crw_year_'+str(2029+i) for i in range(71)]+['n_overbmsy']+['n_underfmsy']+['n_both']+['profit_'+str(2029+i) for i in range(71)]+['hrr_year_'+str(2029+i) for i in range(71)]
time_series.columns = ['catch']+['n_age_1_year_'+str(2029+i) for i in range(71)]+['n_age_2_year_'+str(2029+i) for i in range(71)]+['n_age_3_year_'+str(2029+i) for i in range(71)]+['n_age_4_year_'+str(2029+i) for i in range(71)]+['n_age_5_year_'+str(2029+i) for i in range(71)]+['n_age_6_year_'+str(2029+i) for i in range(71)]+['f_age_1_year_'+str(2029+i) for i in range(71)]+['f_age_2_year_'+str(2029+i) for i in range(71)]+['f_age_3_year_'+str(2029+i) for i in range(71)]+['f_age_4_year_'+str(2029+i) for i in range(71)]+['f_age_5_year_'+str(2029+i) for i in range(71)]+['f_age_6_year_'+str(2029+i) for i in range(71)]+['crn_age_1_year_'+str(2029+i) for i in range(71)]+['crn_age_2_year_'+str(2029+i) for i in range(71)]+['crn_age_3_year_'+str(2029+i) for i in range(71)]+['crn_age_4_year_'+str(2029+i) for i in range(71)]+['crn_age_5_year_'+str(2029+i) for i in range(71)]+['crn_age_6_year_'+str(2029+i) for i in range(71)]+['ssb_year_'+str(2029+i) for i in range(71)]+['crw_year_'+str(2029+i) for i in range(71)]+['n_overbmsy']+['n_underfmsy']+['n_both']+['profit_'+str(2029+i) for i in range(71)]+['hrr_year_'+str(2029+i) for i in range(71)]
# naming the columns of the output data frame

time_series.to_csv('/home/u/u300881/PhD/EMA/Catch_Ricker.csv') # saving the stock metrics to disk

params = pd.DataFrame(params)

params.to_csv('/home/u/u300881/PhD/EMA/Catch_Ricker_Params.csv') # saving the uncertain parameters and policies to disk

"""
11) References
"""
# - Hilborn, R. & Walters, C. J. (1992). Quantitative Fisheries Stock Assessment. Choice, Dynamics and Uncertainty. Chapman and Hall, London, 
# 570 pp.
# - Huang, B., Thorne, P. W., Banzon, V. F., Boyer, T., Chepurin, G., Lawrimore, J. H. et al. (2017). Extended Reconstructed Sea Surface
# Temperature, Version 5 (ERSSTv5): upgrades, validations, and intercomparisons. Journal of Climate, 30, 8179-8205
# - Peck, M. A., Catalán, I. A., Damalas, D., Elliott, M., Ferreira, J. G., Hamon, K. G. et al. (2020). Cllimate change and European fisheries 
# and aquaculture. CERES Project Synthesis Report, Universität Hamburg, Hamburg (DE), 110 pp.
# - Pope, J. G. (1972). An investigation of the accuracy of virtual population analysis using cohort analysis. ICNAF Research Bulletin, 9, 65-74
# - R Core Team (2020). R: an environment for statistical computing. R Foundation for Statistical Computing, Vienna / AT
# - Ricker, W. E. (1975). Computation and interpretation of biological statistics of fish populations. Bulletin - Fisheries Research Board of
# Canada, 191, 382

