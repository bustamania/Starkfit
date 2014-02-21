import scipy.io
import pipes
import os
import numpy as np
import matplotlib.pyplot as plt
#import numdifftools as nd
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import mpl_toolkits.mplot3d.axes3d as p3
import pylab as p

from fittingSG import fittingSG
import MathFunctions as function

pi = 3.14159
c= 3.0*10**10  # c in cm/s
h= 6.626 * 10**(-34)

# import data -------------------------------------------------------------------------------------------------------------

rev_wavenumber, rev_absorption = np.loadtxt("sample_absorbance.dpt", dtype=float, unpack=True)
rev_wavenumber_diff, rev_difference = np.loadtxt("sample_difference.dpt", dtype=float, unpack=True)


# reverse data array ------------------------------------------------------------------------------------------------------

wavenumber = rev_wavenumber[::-1]
absorption = rev_absorption[::-1]-0.0025

wavenumber_diff = rev_wavenumber_diff[::-1]
difference = rev_difference[::-1]-0.000342298251

# Gaussian fitting of absorbance ------------------------------------------------------------------------------------------


GaussFitParam, GaussFitCovariances = curve_fit(function.Gauss, wavenumber, absorption, p0 = [0, 0.0012, 2230, 20])


# Voigtian fitting of absorbance ------------------------------------------------------------------------------------------


#VoigtFitParam, VoigtFitCovariances = curve_fit(Voigt, wavenumber, absorption, p0 = [0, 0.0012, 2156, 20])
#print VoigtFitParam

# Savitzky-Golay Smoothing of Data ----------------------------------------------------------------------------------------

window_size = 17
order = 6

(ysg, dydxsg, ddydxxsg, diffSG) = fittingSG(wavenumber, absorption, difference, window_size, order)
#(ysg2, dydxsg2, ddydxxsg2, diffSG) = fittingSG(wavenumber, absorption, difference, window_size, order)

# Cubic Spline of absorption ----------------------------------------------------------------------------------------------

spline_abs = interpolate.splrep(wavenumber, absorption, s=0)			# spline
spline_absy = interpolate.splev(wavenumber, spline_abs, der=0)			# 0th derivative (to show it)
#spline_absy = function.derivative0(wavenumber, absorption)

# first derivative of cubic spline ----------------------------------------------------------------------------------------

dy = interpolate.splev(wavenumber, spline_abs, der=1)				# 1st derivative
spline_dy = interpolate.splrep(wavenumber, dy, s=0.0)				# spline of 1st derivative
#spline_dy= function.derivative1(wavenumber, spline_absy)
#spline_dyy = interpolate.splev(wavenumber, spline_dy, der=0)			# 0th derivative (of first derivative to show it)

# second derivative of cubic spline ---------------------------------------------------------------------------------------

ddy = interpolate.splev(wavenumber, spline_dy, der=1)				# 2nd derivative
spline_ddy = interpolate.splrep(wavenumber, ddy, s=0)				# spline of 2nd derivative
spline_ddyy = interpolate.splev(wavenumber, spline_ddy, der=0)			# 0th derivative (of second derivative to show it)
#spline_ddy = function.derivative1(wavenumber, spline_dy)

# cubic spline of difference ----------------------------------------------------------------------------------------------


spline_diff = interpolate.splrep(wavenumber_diff, difference, s=0.0000000)	# spline
spline_diffy = interpolate.splev(wavenumber, spline_diff, der=0)		# 0th derivative (to show it)


# definition of fitted function -------------------------------------------------------------------------------------------



#def sum_of_abs(x, A2, B2, C2):
#    return A2 * spline_absy + B2 * spline_dyy + C2 * spline_ddyy
A1= A3= 1.68*10**(-19)
B1= B3= 1.1*10**(-39)
C1= C3= 5.2*10**(-62)

def sum_of_Gauss(x,y0,ElectricField):
    return y0+ElectricField**2*(A1*10**16 * function.Gauss(wavenumber, *GaussFitParam)  +    B1*10**16/(15.0*h*c)* x * function.Gauss1der(wavenumber, *GaussFitParam) +    C1*10**16/(30.0*h**2*c**2)* x * function.Gauss2der(wavenumber, *GaussFitParam))

def sum_of_Voigt(x, A2, B2, C2, ElectricField):
    return ElectricField**2*(A2*10**16 * UnifiedVoigt(wavenumber, *VoigtFitParam)  +    B2*10**16/(15.0*h*c)* x * UnifiedVoigt1der(wavenumber, *VoigtFitParam) +    C2*10**16/(30.0*h**2*c**2)* x * UnifiedVoigt2der(wavenumber, *VoigtFitParam))
def sum_of_SG(x,A3, B3, C3, F):
    return F**2*(A3*10**16 * ysg    +    B3*10**16/(15.0*c*h)* x * dydxsg    +    C3*10**16/(30.0*c**2*h**2)* x * ddydxxsg)


# Fitting of difference data as sum of 1st and 2nd order derivative of absoption data ---------------------------------



FitParamSG, FitCovariancesSG = curve_fit(sum_of_SG, wavenumber, diffSG, p0 = [2*10**(-19), 10**(-39), 10**(-62), 10])
#FitParamData, FitCovariancesData = curve_fit(sum_of_abs, wavenumber, difference, p0 = [1, 10000, 100000])
FitParamGauss, FitCovariancesGauss = curve_fit(sum_of_Gauss, wavenumber, spline_diffy, p0 = [0.0005, 10])


# find Voigtian that fits the best ---------------------------------------------------------------------------------------

DataOut = []
Out = open("Voigtian.dat", "w")
VoigtR2_max = 0
BestFit = []
BestFitSumParam = []

#for xc in xrange(2150, 2170):
#    for w in xrange(1, 100):
#        for A in xrange(1, 10):
#            VoigtFitParam = [1.28123677e-03, A/100, xc, w]
#            FitParamVoigt, FitCovariancesVoigt = curve_fit(sum_of_Voigt, wavenumber, spline_diffy, p0 = [1.8*10**(-20), 0.22*10**(-40), 5.56*10**(-63), 1])
#            VoigtR2 = 1 - np.sum((spline_diffy - sum_of_Voigt(wavenumber, *FitParamVoigt)) ** 2) / np.sum((spline_diffy - np.sum(spline_diffy)/ len(spline_diffy))**2)
#            if VoigtR2 >= VoigtR2_max:
#                VoigtR2_max = VoigtR2
#                BestFit = [xc, w, VoigtR2_max]
#                BestFitSumParam = FitParamVoigt
#            print >> Out, VoigtR2,
#            DataOut.append((A, xc, w, VoigtR2))
#    print >> Out, " "

print BestFit

#np.savetxt("VoigtianData.dat", DataOut)
   
VoigtFitParam = [1.28123677e-03, 3.03572676e-03, 2157, 25]


#R2_old= 0
GaussR2_new = 1 - np.sum((spline_diffy - sum_of_Gauss(wavenumber, *FitParamGauss)) ** 2) / np.sum((spline_diffy - np.sum(spline_diffy)/ len(spline_diffy))**2)
#VoigtR2_new = 1 - np.sum((spline_diffy - sum_of_Voigt(wavenumber, *FitParamVoigt)) ** 2) / np.sum((spline_diffy - np.sum(spline_diffy)/ len(spline_diffy))**2)
SGR2_new = 1 - np.sum((diffSG - sum_of_SG(wavenumber, *FitParamSG)) ** 2) / np.sum((diffSG - np.sum(diffSG)/ len(diffSG))**2)

#while R2_new-R2_old > 0.01:
#    R2_old = R2_new
#    #print 'R^2 =', R2_new
#    window_size += 2
#    (ysg, dydxsg, ddydxxsg, diffSG2) = fittingSG(wavenumber, absorption, difference, window_size, order)
#    (ysg2, dydxsg2, ddydxxsg2, diffSG) = fittingSG(wavenumber, absorption, difference, window_size, order)
#    FitParamSG, FitCovariancesSG = curve_fit(sum_of_SG, wavenumber, diffSG, p0 = [1.8*10**(-20), 0.22*10**(-40), 5.56*10**(-63), 1])
#    R2_new = 1 - np.sum((diffSG - sum_of_SG(wavenumber, *FitParamSG)) ** 2) / np.sum((diffSG - np.sum(diffSG)/ len(diffSG))**2)
    

print 'Gauss fit parameter\n', FitParamGauss
print 'Gauss fit covariances\n', FitCovariancesGauss

#print 'Voigt fit parameter\n', BestFitSumParam
#print 'Voigt fit covariances\n', FitCovariancesVoigt

print 'Savitzky-Golay fit parameter\n', FitParamSG
print 'Savitzky-Golay covariances\n', FitCovariancesSG

print 'Gauss R^2', GaussR2_new
#print 'Voigt R^2', VoigtR2_new
print 'SG    R^2', SGR2_new

# check highest and lowest value for plot ---------------------------------------------------------------------------------

abs_lowerlimit = -max(absorption)*1.3
abs_upperlimit = max(absorption)*1.3

#min_diff = [min(difference)*0.7, min(diffSG)*0.7, min(sum_of_SG)*0.7]
diff_lowerlimit = min(difference)- (max(difference) - min(difference))/10
diff_upperlimit = max(difference)+ (max(difference) - min(difference))/10


# plotting figures --------------------------------------------------------------------------------------------------------


plt.figure('Absorption spectra Gaussian fit')

plt.plot(wavenumber, absorption, 'x', label='absorption Data')
plt.plot(wavenumber, spline_absy, '-', label='cubic spline')

plt.plot(wavenumber, dy*10, '-', label= 'first derivative')
plt.plot(wavenumber, ddy*100, '-', label= 'second derivative')
#plt.plot(wavenumber, spline_dyy*100, '-', label= 'second derivative spline')

plt.plot(wavenumber, function.Gauss(wavenumber, *GaussFitParam), '-', label='Gaussian fit')
plt.plot(wavenumber, function.Gauss1der(wavenumber, *GaussFitParam)*10000, '-', label= 'first derivative Gauss')
plt.plot(wavenumber, function.Gauss2der(wavenumber, *GaussFitParam)*100000, '-', label= 'second derivative Gauss')

#plt.plot(wavenumber, Voigt(wavenumber, *VoigtFitParam), '-', label='Voigtian fit')
#plt.plot(wavenumber, Voigt1der(wavenumber, *VoigtFitParam)*100, '-', label= 'first derivative Voigt')
#plt.plot(wavenumber, Voigt2der(wavenumber, *VoigtFitParam)*1000, '-', label= 'second derivative Voigt')

plt.legend(loc= 'lower right', bbox_to_anchor=(1.05, 1.05))
plt.axis([2190,2130,abs_lowerlimit,abs_upperlimit])
plt.grid



plt.figure('Difference spectra Gaussian fit')

plt.plot(wavenumber, difference, 'x', label='difference Data')
plt.plot(wavenumber, spline_diffy, '-', label='difference spline')

#plt.plot(wavenumber, sum_of_abs(wavenumber, *FitParamData), '-', label = 'Fit as sum of 0th, 1st and 2nd derivative of absorption')
plt.plot(wavenumber, sum_of_Gauss(wavenumber, *FitParamGauss), '-', label = 'Fit as sum of 0th, 1st and 2nd derivative of Gauss')
#plt.plot(wavenumber, sum_of_Voigt(wavenumber, *BestFitSumParam), '-', label = 'Fit as sum of 0th, 1st and 2nd derivative of Voigt')

plt.legend(loc= 'lower right', bbox_to_anchor=(1.05, 1.05))
plt.axis([2190,2130,diff_lowerlimit, diff_upperlimit])
plt.grid


#----------------------------------------------------------------------------------------------------------------------------------


plt.figure('Absorption spectra Savitzky-Golay smoothed fit')

plt.plot(wavenumber, absorption, 'x', label='absorption Data')
plt.plot(wavenumber, spline_absy, '-', label='cubic spline')

# plots of derivatives of cubic spline

plt.plot(wavenumber, dy*10, '-', label= 'first derivative')
plt.plot(wavenumber, ddy*100, '-', label= 'second derivative')
#plt.plot(wavenumber, spline_dyy*100, '-', label= 'second derivative spline')

# plots of derivatives of Savitzky-Golay smoothed data

plt.plot(wavenumber, ysg, '-', label= 'savitzky golay ')
plt.plot(wavenumber, dydxsg*10000, '-', label= 'SG first derivative')
plt.plot(wavenumber, ddydxxsg*80000, '-', label= 'SG second derivative')

plt.legend(loc= 'lower right', bbox_to_anchor=(1.05, 1.05))
plt.axis([2190,2130,abs_lowerlimit,abs_upperlimit])




plt.figure('Difference spectra Savitzky-Golay smoothed fit')

plt.plot(wavenumber, difference, 'x', label='difference Data')
plt.plot(wavenumber, spline_diffy, '-', label='spline')
plt.plot(wavenumber, diffSG, '-', label='SG %i, %i' % (window_size, order))
#plt.plot(wavenumber, sum_of_abs(wavenumber, *FitParamData), '-', label = 'Fit as sum of 0th, 1st and 2nd derivative of absorption')
plt.plot(wavenumber, sum_of_Gauss(wavenumber, *FitParamGauss), '-', label = 'Fit as sum of 0th, 1st and 2nd derivative of Gauss')

plt.plot(wavenumber, sum_of_SG(wavenumber, *FitParamSG), '-', label = 'Fit as sum of SG')
plt.legend(loc= 'lower right', bbox_to_anchor=(1.05, 1.05))
plt.axis([2190,2130,diff_lowerlimit, diff_upperlimit])


plt.show()





