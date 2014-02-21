import numpy as np

def Gauss(x, y0, A, xc, w):
    return y0+A*np.exp(-(x-xc)**2/(2*w))
def Gauss1der(x, y0, A, xc, w):
    return -(A*y0*(xc-x)*np.exp(-(x-xc)**2/(2*w)))/w
def Gauss2der(x, y0, A, xc, w):
    return  (A*y0*np.exp(-(x-xc)**2/(2*w))*(xc**2-2*xc*x-w+x**2))/w**2

def UnifiedVoigt(x, y0, A, xc, w):
    return y0+A*np.exp(-(x-xc)**2/(2*w))+2*A/pi*w/(4*(x-xc)**2+w**2)
def UnifiedVoigt1der(x, y0, A, xc, w):
    return -(A*y0*(xc-x)*np.exp(-(x-xc)**2/(2*w)))/w -(16*A*w*(x-xc))/(pi*(4*(xc-x)**2+w**2)**2)
def UnifiedVoigt2der(x, y0, A, xc, w):
    return  (A*y0*np.exp(-(x-xc)**2/(2*w))*(xc**2-2*xc*x-w+x**2))/w**2  -(16*A*w*(-12*xc**2+24*xc*x+w**2-12*x**2))/(pi*(4*xc**2-8*xc*x+w**2+4*x**2)**3)

def sum_of_Gauss(x,y0,ElectricField):
    return y0+ElectricField**2*(A1*10**16 * Gauss(wavenumber, *GaussFitParam)  +    B1*10**16/(15.0*h*c)* x * Gauss1der(wavenumber, *GaussFitParam) +    C1*10**16/(30.0*h**2*c**2)* x * Gauss2der(wavenumber, *GaussFitParam))

def sum_of_Voigt(x, A2, B2, C2, ElectricField):
    return ElectricField**2*(A2*10**16 * UnifiedVoigt(wavenumber, *VoigtFitParam)  +    B2*10**16/(15.0*h*c)* x * UnifiedVoigt1der(wavenumber, *VoigtFitParam) +    C2*10**16/(30.0*h**2*c**2)* x * UnifiedVoigt2der(wavenumber, *VoigtFitParam))

