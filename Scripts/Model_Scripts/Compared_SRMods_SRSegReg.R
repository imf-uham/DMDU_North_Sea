# 1) Load required packages

library('tidyverse')
library('minpack.lm')
library('viridis')
# library('segmented')

options(scipen = 999)
rm(list = ls())

# 2) set constants

start_year <- 1963 # first year of stock data
stop_year <- 2018 # final year of stock data
stock_code <- 'nsTmean' # code word required for subsetting past SST data

# 3) Load SSB and recruitment data, and historic SST data

dat <- read.table("/home/jan/PhD/Data/Lindegren_Papers/Cod_North_Sea/2021_Assessment/R_SSB.txt",header=T) # SSB- and recruitment data

assign('SST_NOAA', get(load('/home/jan/PhD/Data/Physical_Data/tot_sst.RData'))) # SST data (multiple areas) (Huang et al., 2017)

sst <- as.data.frame(cbind(SST_NOAA$Year, eval(parse(text = paste0('SST_NOAA$', stock_code))))) # subsetting SST data for the North Sea
sst <- sst[sst$V1 %in% seq(start_year, stop_year, 1),] # limiting SST data to the time frame covered by the stock data

names(sst) <- c('year', 'mean_sst')

x <- dat$SSB[1:55] # SSB data (lag of 1 year on recruitment)
x2 <- sst$mean_sst[1:55] # SST data (lag of 1 year on recruitment)
y <- dat$Recruits[2:56] # recruitment data

# 4) Fit hockey-stick function (in line with the "segreg3" function from the ICES assessment,
# [https://rdrr.io/github/flr/diags/src/R/EqSim.R]). One model is fitted on the full time
# series of SR data, and one on the limited time series from 1998 onwards as recognized by
# ICES.
# 69841 = Blim

dat_lim <- dat[dat$Year >= 1998,]
x_lim <- dat_lim$SSB[1:20]
x2_lim <- sst$mean_sst[(length(x2)-length(x_lim)+1):length(x2)]
y_lim <- dat_lim$Recruits[2:21]

m_seg_blim <- nlsLM(log(y) ~ ifelse(x < 69841, b * x, b * 69841), start = list('b' = 1))
m_seg_blim_lim <- nlsLM(log(y_lim) ~ ifelse(x_lim < min(x_lim), b * x_lim, b * min(x_lim)), start = list('b' = 1))

# m_seg_nocc <- nlsLM(log(y) ~ ifelse(x < a, b * x, d), start = list('a' = median(x), 'b' = 1, 'd' = 1))
# m_seg <- nlsLM(log(y) ~ ifelse(log(x) < a, b * log(x) + c * x2, d), start = list('a' = median(log(x)), 'b' = 1, 'c' = 1, 'd' = 1))
# m_seg_alt <- nlsLM(log(y) ~ ifelse(b * x + e * x2 <= d, b * x + e * x2, d), start = list('b' = 0.0001, 'd' = median(log(y)), 'e' = 0.001))
# seg_1 <- segmented::segmented(lm(log(y) ~ 0 + x + x2), seg.Z = ~x)
# seg_2 <- segmented::segmented(lm(log(y) ~ 0 + x), seg.Z = ~x)

# 5) Fit Ricker models with and without SST as covariate (Ricker, 1954; Ricker, 1975)

rk <- mod <- nlsLM(log(y/x) ~ log_ac - exp(log_bc) * x - cc * x2, control = list(maxiter = Inf))
rk_nocc <- mod <- nlsLM(log(y/x) ~ log_ac - exp(log_bc) * x, control = list(maxiter = Inf))

# 6) Fit Beverton-Holt models with and without covariate (Beverton & Holt, 1957; Hilborn & Walters, 1992)

bh <- nlsLM(log(y) ~ cc * x2 + (log_ac + log(x)) - log(1 + exp(log_bc + log(x))), control = list(maxiter = Inf))
bh_nocc <- nlsLM(log(y) ~ (log_ac + log(x)) - log(1 + exp(log_bc + log(x))), control = list(maxiter = Inf))

# 7) Compare deviance explained between the models. Comparision is done once for the full time series of SR data, and once 
# for the limited time series from 1998 onwards as recognized by ICES (2021)

y_pred_seg_blim <- predict(m_seg_blim)
1 - sum((y - exp(y_pred_seg_blim))**2) / sum((y - mean(y))**2)

y_pred_seg_blim_lim <- predict(m_seg_blim_lim)
1 - sum((y_lim - exp(y_pred_seg_blim_lim))**2) / sum((y_lim - mean(y_lim))**2)

y_pred_rk <- predict(rk)
1 - sum((y - x*exp(y_pred_rk))**2) / sum((y - mean(y))**2)
y_pred_rk <- predict(rk_nocc)
1 - sum((y - x*exp(y_pred_rk))**2) / sum((y - mean(y))**2)

y_pred_rk_lim <- predict(rk, newdata = list('x' = x_lim, 'x2' = x2_lim))
1 - sum((y_lim - x_lim*exp(y_pred_rk_lim))**2) / sum((y_lim - mean(y_lim))**2)
y_pred_rk_lim <- predict(rk_nocc, newdata = list('x' = x_lim))
1 - sum((y_lim - x_lim*exp(y_pred_rk_lim))**2) / sum((y_lim - mean(y_lim))**2)

y_pred_bh <- predict(bh)
1 - sum((y - exp(y_pred_bh))**2) / sum((y - mean(y))**2)
y_pred_bh <- predict(bh_nocc)
1 - sum((y - exp(y_pred_bh))**2) / sum((y - mean(y))**2)

y_pred_bh_lim <- predict(bh, newdata = list('x' = x_lim, 'x2' = x2_lim))
1 - sum((y_lim - exp(y_pred_bh_lim))**2) / sum((y_lim - mean(y_lim))**2)
y_pred_bh_lim <- predict(bh_nocc, newdata = list('x' = x_lim))
1 - sum((y_lim - exp(y_pred_bh_lim))**2) / sum((y_lim - mean(y_lim))**2)

# 8) Compare AIC between the models

AIC(m_seg_blim)
AIC(rk)
AIC(rk_nocc)
AIC(bh)
AIC(bh_nocc)

# 9) Plot (partial) effect of SSB on recruitment. Plotting is done once for the full time series of SR data, and once 
# for the limited time series from 1998 onwards as recognized by ICES (2021)

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_segreg_blim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(x/1000, y/1000, xlab = 'SSB [kt]', ylab = expression(paste('Recruitment [',10^{6},']')))
lines(seq(min(x), max(x), length.out = 100)/1000, 
      exp(predict(m_seg_blim, newdata = list('x' = seq(min(x), max(x), length.out = 100))))/1000, 
      col = 'red')
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_segreg_blim_lim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(x_lim/1000, y_lim/1000, xlab = 'SSB [kt]', ylab = expression(paste('Recruitment [',10^{6},']')),
     xlim = c(0,max(x_lim/1000)), ylim = c(0,max(y_lim/1000)))
lines(seq(min(x_lim), max(x_lim), length.out = 100)/1000, 
      exp(predict(m_seg_blim_lim, newdata = list('x_lim' = seq(min(x_lim), max(x_lim), length.out = 100))))/1000, 
      col = 'red')
lines(c(0,min(x_lim)/1000), c(0,exp(predict(m_seg_blim_lim))[1])/1000, col='red')
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_ricker.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(x/1000, y/1000, xlab = 'SSB [kt]', ylab = expression(paste('Recruitment [',10^{6},']')))
lines(seq(min(x), max(x), length.out = 100)/1000, 
      seq(min(x), max(x), length.out = 100) * exp(predict(rk_nocc, newdata = list('x' = seq(min(x), max(x), length.out = 100))))/1000, 
      col = 'red')
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_ricker_lim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(x_lim/1000, y_lim/1000, xlab = 'SSB [kt]', ylab = expression(paste('Recruitment [',10^{6},']')),
     xlim = c(0,max(x_lim/1000)), ylim = c(0,max(y_lim/1000)))
lines(seq(0, max(x_lim), length.out = 100)/1000, 
      seq(0, max(x_lim), length.out = 100) * exp(predict(rk_nocc, newdata = list('x' = seq(0, max(x_lim), length.out = 100))))/1000, 
      col = 'red')
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_ricker_clim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(x/1000, y/1000, xlab = 'SSB [kt]', ylab = expression(paste('Recruitment [',10^{6},']')))
lines(seq(min(x), max(x), length.out = 100)/1000, 
      seq(min(x), max(x), length.out = 100) * exp(predict(rk, newdata = list('x' = seq(min(x), max(x), length.out = 100), 'x2' = median(x2))))/1000, 
      col = plasma(7)[4])
lines(seq(min(x), max(x), length.out = 100)/1000, 
      seq(min(x), max(x), length.out = 100) * exp(predict(rk, newdata = list('x' = seq(min(x), max(x), length.out = 100), 'x2' = quantile(x2,0.05))))/1000, 
      col = plasma(7)[2])
lines(seq(min(x), max(x), length.out = 100)/1000, 
      seq(min(x), max(x), length.out = 100) * exp(predict(rk, newdata = list('x' = seq(min(x), max(x), length.out = 100), 'x2' = quantile(x2,0.95))))/1000, 
      col = plasma(7)[6])
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_ricker_clim_lim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(x_lim/1000, y_lim/1000, xlab = 'SSB [kt]', ylab = expression(paste('Recruitment [',10^{6},']')),
     xlim = c(0,max(x_lim/1000)), ylim = c(0,max(y_lim/1000)))
lines(seq(0, max(x_lim), length.out = 100)/1000, 
      seq(0, max(x_lim), length.out = 100) * exp(predict(rk, newdata = list('x' = seq(0, max(x_lim), length.out = 100), 'x2' = median(x2_lim))))/1000, 
      col = plasma(7)[4])
lines(seq(0, max(x_lim), length.out = 100)/1000, 
      seq(0, max(x_lim), length.out = 100) * exp(predict(rk, newdata = list('x' = seq(0, max(x_lim), length.out = 100), 'x2' = quantile(x2_lim,0.05))))/1000, 
      col = plasma(7)[2])
lines(seq(0, max(x_lim), length.out = 100)/1000, 
      seq(0, max(x_lim), length.out = 100) * exp(predict(rk, newdata = list('x' = seq(0, max(x_lim), length.out = 100), 'x2' = quantile(x2_lim,0.95))))/1000, 
      col = plasma(7)[6])
dev.off()

# tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_ricker_clim_sst.tiff",
#      width = 5,height = 5,compression = "lzw",res = 600, units = "in")
# plot(x2, y/1000, xlab = 'SST [°C]', ylab = expression(paste('Recruitment [',10^{6},']')))
# lines(seq(min(x2), max(x2), length.out = 100),
#       median(x) * exp(predict(rk, newdata = list('x' = median(x), 'x2' = seq(min(x2), max(x2), length.out = 100))))/1000,
#       col = 'red')
# dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_bevertonh.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(x/1000, y/1000, xlab = 'SSB [kt]', ylab = expression(paste('Recruitment [',10^{6},']')))
lines(seq(min(x), max(x), length.out = 100)/1000, 
      exp(predict(bh_nocc, newdata = list('x' = seq(min(x), max(x), length.out = 100))))/1000, 
      col = 'red')
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_bevertonh_lim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(x_lim/1000, y_lim/1000, xlab = 'SSB [kt]', ylab = expression(paste('Recruitment [',10^{6},']')),
     xlim = c(0,max(x_lim/1000)), ylim = c(0,max(y_lim/1000)))
lines(seq(0, max(x_lim), length.out = 100)/1000, 
      exp(predict(bh_nocc, newdata = list('x' = seq(0, max(x_lim), length.out = 100))))/1000, 
      col = 'red')
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_bevertonh_clim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(x/1000, y/1000, xlab = 'SSB [kt]', ylab = expression(paste('Recruitment [',10^{6},']')))
lines(seq(min(x), max(x), length.out = 100)/1000, 
      exp(predict(bh, newdata = list('x' = seq(min(x), max(x), length.out = 100), 'x2' = median(x2))))/1000, 
      col = plasma(7)[4])
lines(seq(min(x), max(x), length.out = 100)/1000, 
      exp(predict(bh, newdata = list('x' = seq(min(x), max(x), length.out = 100), 'x2' = quantile(x2,0.05))))/1000, 
      col = plasma(7)[2])
lines(seq(min(x), max(x), length.out = 100)/1000, 
      exp(predict(bh, newdata = list('x' = seq(min(x), max(x), length.out = 100), 'x2' = quantile(x2,0.95))))/1000, 
      col = plasma(7)[6])
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_bevertonh_clim_lim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(x_lim/1000, y_lim/1000, xlab = 'SSB [kt]', ylab = expression(paste('Recruitment [',10^{6},']')),
     xlim = c(0,max(x_lim/1000)), ylim = c(0,max(y_lim/1000)))
lines(seq(0, max(x_lim), length.out = 100)/1000, 
      exp(predict(bh, newdata = list('x' = seq(0, max(x_lim), length.out = 100), 'x2' = median(x2_lim))))/1000, 
      col = plasma(7)[4])
lines(seq(0, max(x_lim), length.out = 100)/1000, 
      exp(predict(bh, newdata = list('x' = seq(0, max(x_lim), length.out = 100), 'x2' = quantile(x2_lim,0.05))))/1000, 
      col = plasma(7)[2])
lines(seq(0, max(x_lim), length.out = 100)/1000, 
      exp(predict(bh, newdata = list('x' = seq(0, max(x_lim), length.out = 100), 'x2' = quantile(x2_lim,0.95))))/1000, 
      col = plasma(7)[6])
dev.off()

# tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/srcompar_bevertonh_clim_sst.tiff",
#      width = 5,height = 5,compression = "lzw",res = 600, units = "in")
# plot(x2, y/1000, xlab = 'SST [°C]', ylab = expression(paste('Recruitment [',10^{6},']')))
# lines(seq(min(x2), max(x2), length.out = 100),
#       exp(predict(bh, newdata = list('x' = median(x), 'x2' = seq(min(x2), max(x2), length.out = 100))))/1000,
#       col = 'red')
# dev.off()

# 10) Plot predicted recruitment vs "observed" recruitment for the different SR models. Plotting is done once
# for the full time series of SR data, and once for the limited time series from 1998 onwards as recognized
# by ICES (2021)

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/obspred_segreg_blim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(y/1000, exp(predict(m_seg_blim))/1000, 
     xlab = expression(paste('Recruitment [',10^{6},']')), ylab = expression(paste('Predicted recruitment [',10^{6},']')),
     xlim = c(min(y/1000), max(y/1000)), ylim = c(min(y/1000), max(y/1000)))
abline(c(0,1))
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/obspred_segreg_blim_lim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(y_lim/1000, exp(predict(m_seg_blim_lim))/1000, 
     xlab = expression(paste('Recruitment [',10^{6},']')), ylab = expression(paste('Predicted recruitment [',10^{6},']')),
     xlim = c(min(y_lim/1000), max(y_lim/1000)), ylim = c(min(y_lim/1000), max(y_lim/1000)))
abline(c(0,1))
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/obspred_ricker.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(y/1000, x*exp(predict(rk_nocc))/1000, 
     xlab = expression(paste('Recruitment [',10^{6},']')), ylab = expression(paste('Predicted recruitment [',10^{6},']')),
     xlim = c(min(y/1000), max(y/1000)), ylim = c(min(y/1000), max(y/1000)))
abline(c(0,1))
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/obspred_ricker_lim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(y_lim/1000, x_lim*exp(predict(rk_nocc, newdata = list('x' = x_lim)))/1000, 
     xlab = expression(paste('Recruitment [',10^{6},']')), ylab = expression(paste('Predicted recruitment [',10^{6},']')),
     xlim = c(min(y_lim/1000), max(y_lim/1000)), ylim = c(min(y_lim/1000), max(y_lim/1000)))
abline(c(0,1))
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/obspred_ricker_clim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(y/1000, x*exp(predict(rk))/1000, 
     xlab = expression(paste('Recruitment [',10^{6},']')), ylab = expression(paste('Predicted recruitment [',10^{6},']')),
     xlim = c(min(y/1000), max(y/1000)), ylim = c(min(y/1000), max(y/1000)))
abline(c(0,1))
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/obspred_ricker_clim_lim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(y_lim/1000, x_lim*exp(predict(rk, newdata = list('x' = x_lim, 'x2' = x2_lim)))/1000, 
     xlab = expression(paste('Recruitment [',10^{6},']')), ylab = expression(paste('Predicted recruitment [',10^{6},']')),
     xlim = c(min(y_lim/1000), max(y_lim/1000)), ylim = c(min(y_lim/1000), max(y_lim/1000)))
abline(c(0,1))
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/obspred_bevertonh.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(y/1000, exp(predict(bh_nocc))/1000, 
     xlab = expression(paste('Recruitment [',10^{6},']')), ylab = expression(paste('Predicted recruitment [',10^{6},']')),
     xlim = c(min(y/1000), max(y/1000)), ylim = c(min(y/1000), max(y/1000)))
abline(c(0,1))
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/obspred_bevertonh_lim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(y_lim/1000, exp(predict(bh_nocc, newdata = list('x' = x_lim)))/1000, 
     xlab = expression(paste('Recruitment [',10^{6},']')), ylab = expression(paste('Predicted recruitment [',10^{6},']')),
     xlim = c(min(y_lim/1000), max(y_lim/1000)), ylim = c(min(y_lim/1000), max(y_lim/1000)))
abline(c(0,1))
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/obspred_bevertonh_clim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(y/1000, exp(predict(bh))/1000, 
     xlab = expression(paste('Recruitment [',10^{6},']')), ylab = expression(paste('Predicted recruitment [',10^{6},']')),
     xlim = c(min(y/1000), max(y/1000)), ylim = c(min(y/1000), max(y/1000)))
abline(c(0,1))
dev.off()

tiff(filename = "/home/jan/PhD/EMA/North_Sea_Paper/obspred_bevertonh_clim_lim.tiff",
     width = 5,height = 5,compression = "lzw",res = 600, units = "in")
par(mar=c(5,6,4,1)+.1)
plot(y_lim/1000, exp(predict(bh, newdata = list('x' = x_lim, 'x2' = x2_lim)))/1000, 
     xlab = expression(paste('Recruitment [',10^{6},']')), ylab = expression(paste('Predicted recruitment [',10^{6},']')),
     xlim = c(min(y_lim/1000), max(y_lim/1000)), ylim = c(min(y_lim/1000), max(y_lim/1000)))
abline(c(0,1))
dev.off()

# 11) References

# 5) References

# - Beverton, R. J. H. & Holt, S. J. (1957). On the Dynamics of Exploited Fish Populations. Chapman & Hall
# - Hilborn, R. & Walters, C. J. (1992). Quantitative Fisheries Stock Assessment. Choice, Dynamics and Uncertainty. Chapman and Hall, London, 
# 570 pp.
# - Huang, B., Thorne, P. W., Banzon, V. F., Boyer, T., Chepurin, G., Lawrimore, J. H. et al. (2017). Extended Reconstructed Sea Surface
# Temperature, Version 5 (ERSSTv5): upgrades, validations, and intercomparisons. Journal of Climate, 30, 8179-8205
# - ICES (2021). Benchmark Workshop on North Sea Stocks (WKNSEA). ICES Scientific Reports, 3 (25), 756 pp
# - Ricker, W. E. (1954). Stock and Recruitment. J. Fish. Res. Board Can., 11, 559-623
# - Ricker, W. E. (1975). Computation and interpretation of biological statistics of fish populations. Bulletin - Fisheries Research Board of
# Canada, 191, 382



