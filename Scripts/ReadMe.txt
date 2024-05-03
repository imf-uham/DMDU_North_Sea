The script "SR_Model_Fitting_Annotated.Rmd" fits the Ricker and Beverton-Holt SR models on the 
assessment estimates of SSB and recruitment as well as on NOAA SST reconstructions for the North 
Sea [1]. The script should be run in interactive mode (e.g. in IDE RStudio).

"North_Sea_From2030_NewEconModel_Annotated.py" is the core model script which conducts the sampling
of uncertain factors and runs the model with the various policies and uncertainties. The script is
run four times to cover the two different harvest strategies (constant catch, constant harvest rate) 
and the two different functional types of the SR relationship (Ricker and Beverton-Holt); to this
end, some sections of the script need to be commented or uncommented.

"Feature_Scoring_Annotated.py" conducts a feature-scoring analysis, i.e. step (i) of the analyses of 
model output noted above. This script should be run in interactive mode (e.g. in IDE Spyder).

"ScenarioDiscovery_RiskAnalysis_From2030_Annotated.Rmd" conducts the remaining analyses of the model 
output. The script should be run in interactive mode (e.g. in IDE RStudio).
There is now an additional file that contains code for plotting risk dynamics over projection years,
and an additional file that plots such risk time series for model runs conducted with zero catch or
zero harvest rate.

The folder "input_data" contains all data required to fit the SR models and to initialize and run the
population model. The model output data are available on https://zenodo.org/records/10617806.

The Python environment required for running the population model is available as the yaml file "EMAenv.yml".
It can be unpacked using the Anaconda distribution for Python. We used R version 3.6.3 for running the 
R scripts, though they may work also under newer versions of R. Required packages are listed in the beginning 
of each script.

References:

[1] Huang, B., Thorne, P. W., Banzon, V. F., Boyer, T., Chepurin, G., Lawrimore, J. H., Menne, M. J., Smith, T. M., Vose, R. S. & Zhang, H.-M. (2017). Extended Reconstructed Sea Surface Temperature, Version 5 (ERSSTv5): upgrades, validations, and intercomparisons. Journal of Climate, 30, 8179-8205. doi: 10.1175/JCLI-D-16-0836.1


