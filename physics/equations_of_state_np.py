
import numpy as np

##########################################################################
#                                                                        #
#  Equations of state translated directly from the Pflotran source code  #
#  https://bitbucket.org/pflotran/pflotran/                              #
#                                                                        #
##########################################################################

H2O_CRITICAL_TEMPERATURE = 647.3  # [K]
H2O_CRITICAL_PRESSURE = 22.064e6 # [Pa]

def eos_water_saturation_pressure_IFC67(T):
    """
    Calculates the saturation pressure of water as a function of temperature.

    Translated from the Pflotran function "EOSWaterSaturationPressureIFC67".

    Parameters
    ----------
    T: Temperature [°C]

    Returns
    -------
    Saturation pressure
    """
    A = [ -7.691234564, -2.608023696e1, -1.681706546e2, 6.423285504e1, -1.189646225e2, 4.167117320, 2.097506760e1, 1.0e9, 6.0 ]

    TC = ( T+273.15 ) / H2O_CRITICAL_TEMPERATURE
    one_m_tc = 1.0 - TC
    one_m_tc_sq = one_m_tc*one_m_tc
    SC = A[0]*one_m_tc + A[1]*one_m_tc_sq + A[2]*one_m_tc**3 + A[3]*one_m_tc**4 + A[4]*one_m_tc**5
    E1 = TC * (1.0 + A[5]*one_m_tc + A[6]*one_m_tc_sq)
    E2_bottom = A[7]*one_m_tc_sq+A[8]
    E2 = one_m_tc / E2_bottom
    PCAP = np.exp(SC / E1-E2)

    PS = PCAP*H2O_CRITICAL_PRESSURE
    
    return PS

def eos_water_viscosity_1(T, P, PS):
    """
    Calculates the viscosity of water as a function of
    temperature, pressure, and saturation pressure.

    Translated from the Pflotran function "EOSWaterViscosity1".

    Parameters
    ----------
    T: Temperature [°C]
    P: Pressure [Pa]
    PS: Saturation pressure

    Returns
    -------
    Viscosity
    """

    EX  = 247.8 / (T+133.15)
    PHI = 1.0467 * (T-31.85)
    AM  = 1.0 + PHI*(np.maximum(P,PS) - PS) * 1.0e-11
    pwr = np.power(10, EX)
    VW = 1.0e-7 * AM * 241.4 * pwr
    return VW

FMWH2O = 18.01534  # [kg/kmol] h2o

def eos_water_enthalphy(T, p):
    """
    Calculates the water enthalphy as a function of
    temperature and pressure.

    Translated from the Pflotran function "EOSWaterEnthalpyIFC67".

    Ref.: International Formulation Committee of the Sixth International
          Conference on Properties of Steam (1967).

    Parameters
    ----------
    T: Temperature [°C]
    p: Pressure [Pa]

    Returns
    -------
    Enthalphy
    """

    zero = 0.0
    one = 1.0
    two = 2.0
    three = 3.0
    five = 5.0
    six = 6.0
    nine = 9.0
    ten = 10.0

    aa = [
#-----data aa0,aa1,aa2,aa3/
        6.824687741e3,-5.422063673e2,-2.096666205e4, 3.941286787e4,
#-----data aa4,aa5,aa6,aa7/
        -6.733277739e4, 9.902381028e4,-1.093911774e5, 8.590841667e4,
#-----data aa8,aa9,aa10,aa11/
        -4.511168742e4, 1.418138926e4,-2.017271113e3, 7.982692717,
#-----data aa12,aa13,aa14,aa15/
        -2.616571843e-2, 1.522411790e-3, 2.284279054e-2, 2.421647003e2,
#-----data aa16,aa17,aa18,aa19/
        1.269716088e-10,2.074838328e-7, 2.174020350e-8, 1.105710498e-9,
#-----data aa20,aa21,aa22/
        1.293441934e1, 1.308119072e-5, 6.047626338e-14
    ]

    a1,a2,a3,a4 = 8.438375405e-1, 5.362162162e-4, 1.720000000, 7.342278489e-2
    a5,a6,a7,a8 = 4.975858870e-2, 6.537154300e-1, 1.150000000e-6, 1.510800000e-5
    a9,a10,a11,a12 = 1.418800000e-1, 7.002753165, 2.995284926e-4, 2.040000000e-1

    tc1 = H2O_CRITICAL_TEMPERATURE
    pc1 = H2O_CRITICAL_PRESSURE
    vc1 = 0.00317  # [m^3/kg]
    utc1 = one / tc1 # [1/C]
    upc1 = one / pc1 # [1/Pa]
    vc1mol = vc1*FMWH2O # [m^3/kmol]

    theta = (T+273.15)*utc1
    theta2x = theta*theta
    theta18 = np.power(theta,18.0)
    theta20 = theta18*theta2x

    beta = p*upc1
    beta2x = beta*beta
    beta4  = beta2x*beta2x

    yy = one-a1*theta2x-a2*np.power(theta, -6.0)
    xx = a3*yy*yy-two*(a4*theta-a5*beta)
    xx = np.sqrt(xx)

    zz = yy + xx
    u0 = -five/17.0
    u1 = aa[11]*a5*np.power(zz, u0)

    ypt = six*a2*np.power(theta, -7.0)-two*a1*theta

    #---compute enthalpy internal energy and derivatives for water
    utheta = one/theta
    term1 = aa[0]*theta
    term2 = -aa[1]
    # term2t is part of the derivative calc., but left here to avoid
    # recomputing the expensive do loop
    term2t = zero
    #do i = 3,10
    for i in range(3,11):
        tempreal = (i-2)*aa[i]*np.power(theta, i-1)
        term2t = term2t+tempreal*utheta*(i-1)
        term2 = term2+tempreal

    # "v" section 1
    v0_1 = u1/a5
    v2_1 = 17.0*(zz/29.0-yy/12.0)+five*theta*ypt/12.0
    v3_1 = a4*theta-(a3-one)*theta*yy*ypt
    v1_1 = zz*v2_1+v3_1
    term3 = v0_1*v1_1

    # "v" section 2
    v1_2 = nine*theta+a6
    v20_2 = (a6-theta)
    v2_2 = np.power(v20_2,9.0)
    v3_2 = a7+20.0*np.power(theta,19.0)
    v40_2 = a7+np.power(theta, 19.0)
    v4_2 = one/(v40_2*v40_2)
    # term4p is a derivative, but left due to dependency in term4
    term4p = aa[12]-aa[14]*theta2x+aa[15]*v1_2*v2_2+aa[16]*v3_2*v4_2
    term4 = term4p*beta

    # "v" section 3
    v1_3 = beta*(aa[17]+aa[18]*beta+aa[19]*beta2x)
    v2_3 = 12.0*np.power(theta,11.0)+a8
    v4_3 = one/(a8+np.power(theta,11.0))
    v3_3 = v4_3*v4_3
    term5 = v1_3*v2_3*v3_3

    # "v" section 4
    v1_4 = np.power(a10+beta,-3.0) + a11*beta
    v3_4 = (17.0*a9+19.0*theta2x)
    v2_4 = aa[20]*theta18*v3_4
    term6 = v1_4*v2_4

    # "v" section 5
    v1_5 = 21.0*aa[22]/theta20*beta4
    v2_5 = aa[21]*a12*beta2x*beta
    term7 = v1_5+v2_5

    # "v" section 6
    v1_6 = pc1*vc1mol
    hw = (term1-term2+term3+term4-term5+term6+term7)*v1_6
    
    return hw


def eos_water_density_IFC67(T, p):
    """
    Calculates the water density and molar density as a function of
    temperature and pressure.

    Translated from the Pflotran function "EOSWaterDensityIFC67".

    Ref.: International Formulation Committee of the Sixth International
          Conference on Properties of Steam (1967).

    Parameters
    ----------
    T: Temperature [°C]
    p: Pressure [Pa]

    Returns
    -------
    Density [kg/m^3]
    Molar density [kmol/m^3]
    """

    zero = 0.0
    one = 1.0
    two = 2.0
    three = 3.0
    four = 4.0
    five = 5.0
    six = 6.0
    nine = 9.0
    ten = 10.0

    aa = [
#-----data aa0,aa1,aa2,aa3/
        6.824687741e3,-5.422063673e2,-2.096666205e4, 3.941286787e4,
#-----data aa4,aa5,aa6,aa7/
        -6.733277739e4, 9.902381028e4,-1.093911774e5, 8.590841667e4,
#-----data aa8,aa9,aa10,aa11/
        -4.511168742e4, 1.418138926e4,-2.017271113e3, 7.982692717,
#-----data aa12,aa13,aa14,aa15/
        -2.616571843e-2, 1.522411790e-3, 2.284279054e-2, 2.421647003e2,
#-----data aa16,aa17,aa18,aa19/
        1.269716088e-10,2.074838328e-7, 2.174020350e-8, 1.105710498e-9,
#-----data aa20,aa21,aa22/
        1.293441934e1, 1.308119072e-5, 6.047626338e-14
    ]

    a1,a2,a3,a4 = 8.438375405e-1, 5.362162162e-4, 1.720000000, 7.342278489e-2
    a5,a6,a7,a8 = 4.975858870e-2, 6.537154300e-1, 1.150000000e-6, 1.510800000e-5
    a9,a10,a11,a12 = 1.418800000e-1, 7.002753165, 2.995284926e-4, 2.040000000e-1

    tc1 = H2O_CRITICAL_TEMPERATURE
    pc1 = H2O_CRITICAL_PRESSURE
    vc1 = 0.00317  # [m^3/kg]
    utc1 = one / tc1 # [1/C]
    upc1 = one / pc1 # [1/Pa]
    vc1mol = vc1*FMWH2O # [m^3/kmol]

    theta = (T+273.15)*utc1
    theta2x = theta*theta
    theta18 = np.power(theta,18.0)
    theta20 = theta18*theta2x

    beta = p*upc1
    beta2x = beta*beta
    beta4  = beta2x*beta2x
    
    yy = one-a1*theta2x-a2*np.power(theta, -6.0)
    xx = a3*yy*yy-two*(a4*theta-a5*beta)
    xx = np.sqrt(xx)

    zz = yy + xx
    u0 = -five/17.0
    u1 = aa[11]*a5*np.power(zz, u0)
    u2 = one/(a8+np.power(theta, 11.0))
    u3 = aa[17]+(two*aa[18]+three*aa[19]*beta)*beta
    u4 = one/(a7+theta18*theta)
    u5 = np.power(a10+beta, -4.0)
    u6 = a11-three*u5
    u7 = aa[20]*theta18*(a9+theta2x)
    u8 = aa[15]*np.power(a6-theta,9.0)

    vr = u1+aa[12]+theta*(aa[13]+aa[14]*theta)+u8*(a6-theta) \
        +aa[16]*u4-u2*u3-u6*u7+(three*aa[21]*(a12-theta) \
        +four*aa[22]*beta/theta20)*beta2x

    dwmol = one/(vr*vc1mol) # kmol/m^3
    dw = one/(vr*vc1) # kg/m^3
    
    return dw, dwmol

def thermal_conductivity(T):
    """
    Placeholder for thermal conductivity as a function of 
    temperature and saturation.
    Not used because saturation s is always 1 in our case
    so the thermal conductivity is also 1 based on Somerton.

    Translated from the Pflotran function "TCFDefaultConductivity".
    """
    kT_dry = 0.7 # [W / K m]
    kT_wet = 1.0 # [W / K m]

    #based on Somerton et al., 1974:
    #k_eff = k_dry + sqrt(s_l)*(k_wet-k_dry)
    
    tempreal = (kT_wet - kT_dry)
    thermal_cond = kT_dry + tempreal
    
    return thermal_cond