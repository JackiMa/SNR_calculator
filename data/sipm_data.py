# sipm_data.py

import numpy as np
import scipy.constants as CONST


class Conditions:
    def __init__(self, temp=None, dose=None, Vsipm=None):
        self.temp = temp
        self.dose = dose
        self.Vsipm = Vsipm

    def __str__(self):
        return f"Conditions(temp={self.temp}, dose={self.dose}, Vsipm={self.Vsipm})"

    def __repr__(self):
        return self.__str__()

class SiPMType:
    def __init__(self, name):
        self.name = name
        self.type = None
        self.conditions = Conditions()
        self.area = None
        self.nums_pix = None
        self.fill_factor = None
        self.k_Vbr = None
        self.Vbr0 = None  # Reference breakdown voltage at temp0
        self.temp0 = None  # Reference temperature
        self.C_SiPM = None
        self.PDE = None
        self.gain = None
        self.DCR_mm2 = None
        self.I_dark = None
        self.ECF = None
        self.t_rise = None
        self.t_fall = None
        self.p_crosstalk = None
        self.p_afterpulsing = None
        self.PDE_wl = None  # Placeholder for PDE wavelength-dependent function

    def check_lengths(self):
        # Check if the lengths of Vsipm-dependent parameters match the length of Vsipm
        Vsipm_len = len(self.conditions.Vsipm)
        params = [self.PDE, self.gain, self.DCR_mm2, self.I_dark,
                  self.t_rise, self.t_fall, self.p_crosstalk, self.p_afterpulsing]
        param_names = ['PDE', 'gain', 'DCR_mm2', 'I_dark',
                       't_rise', 't_fall', 'p_crosstalk', 'p_afterpulsing']

        for param, name in zip(params, param_names):
            if param is not None and len(param) != Vsipm_len:
                raise ValueError(f"Length of {name} does not match length of Vsipm in conditions for model {self.name}.")

    def check_DCR_I_dark(self):
        # Ensure at least one of DCR_mm2 or I_dark is provided
        if self.DCR_mm2 is None and self.I_dark is None:
            raise ValueError(f"Either DCR_mm2 or I_dark must be provided for model {self.name}.")

        # If both are provided, calculate ECF
        if self.DCR_mm2 is not None and self.I_dark is not None:
            self.calculate_ECF()
        else:
            # If only one is provided, calculate the other
            if self.DCR_mm2 is not None:
                self.calculate_I_dark()
            elif self.I_dark is not None:
                self.calculate_DCR_mm2()
            else:
                raise ValueError("DCR_mm2 and I_dark must be fully provided or empty.")

            # Set ECF to default value 1.2 if not provided
            if self.ECF is None:
                self.ECF = [1.2] * len(self.conditions.Vsipm)

    def calculate_ECF(self):
        # I_dark = DCR_mm2 * area * gain * e * ECF
        # ECF = I_dark / (DCR_mm2 * area * gain * e)
        area_mm2 = self.area * 1e6  # Convert area from m^2 to mm^2
        self.ECF = []
        Vsipm_len = len(self.conditions.Vsipm)
        for i in range(Vsipm_len):
            DCR = self.DCR_mm2[i] * area_mm2  # Hz
            denominator = DCR * self.gain[i] * CONST.e
            if denominator == 0:
                self.ECF.append(0)
            else:
                ECF_value = self.I_dark[i] / denominator
                self.ECF.append(ECF_value)

    def calculate_I_dark(self):
        # I_dark = DCR_mm2 * area * gain * e * ECF
        area_mm2 = self.area * 1e6  # Convert area from m^2 to mm^2
        if self.ECF is None:
            self.ECF = [1.2] * len(self.conditions.Vsipm)
        self.I_dark = []
        Vsipm_len = len(self.conditions.Vsipm)
        for i in range(Vsipm_len):
            DCR = self.DCR_mm2[i] * area_mm2  # Hz
            I_dark_value = DCR * self.gain[i] * CONST.e * self.ECF[i]
            self.I_dark.append(I_dark_value)

    def calculate_DCR_mm2(self):
        # DCR_mm2 = I_dark / (area * gain * e * ECF)
        area_mm2 = self.area * 1e6  # Convert area from m^2 to mm^2
        if self.ECF is None:
            self.ECF = [1.2] * len(self.conditions.Vsipm)
        self.DCR_mm2 = []
        Vsipm_len = len(self.conditions.Vsipm)
        for i in range(Vsipm_len):
            denominator = area_mm2 * self.gain[i] * CONST.e * self.ECF[i]
            if denominator == 0:
                self.DCR_mm2.append(0)
            else:
                DCR_mm2_value = self.I_dark[i] / denominator
                self.DCR_mm2.append(DCR_mm2_value)

# Instantiate SiPM types
def create_sipm_types():
    sipm_types = {}

    # Define J60035
    sipm = SiPMType('J60035')
    sipm.type = '60035'
    sipm.conditions = Conditions(
        temp=21,
        dose=0,
        Vsipm=[24.45 + 2.5, 24.45 + 6.0]  # Vsipm = Vbr0 + Vov
    )
    sipm.area = (6.07e-3) * (6.07e-3)  # m^2
    sipm.nums_pix = 22292
    sipm.fill_factor = 0.75
    sipm.k_Vbr = 0.0215  # V/degree
    sipm.Vbr0 = 0.5 * (24.2 + 24.7)  # V
    sipm.temp0 = 21  # Reference temperature in degrees Celsius
    sipm.C_SiPM = 4140e-12  # F
    sipm.PDE = [0.38, 0.5]
    sipm.gain = [2.9e6, 6.3e6]
    sipm.DCR_mm2 = [50e3, 150e3]  # Hz/mm^2
    sipm.I_dark = [0.9e-6, 7.5e-6]  # A
    sipm.t_rise = [180e-12, 250e-12]  # s
    sipm.t_fall = [50e-9, 50e-9]  # s
    sipm.p_crosstalk = [0.08, 0.25]
    sipm.p_afterpulsing = [0.0075, 0.05]
    # Perform checks and calculations
    sipm.check_lengths()
    sipm.check_DCR_I_dark()
    sipm_types['J60035'] = sipm

    # Define J30035
    sipm = SiPMType('J30035')
    sipm.type = '30035'
    sipm.conditions = Conditions(
        temp=21,
        dose=0,
        Vsipm=[24.45 + 2.5, 24.45 + 6.0]
    )
    sipm.area = (3.07e-3) * (3.07e-3)  # m^2
    sipm.nums_pix = 5676
    sipm.fill_factor = 0.75
    sipm.k_Vbr = 0.0215  # V/degree
    sipm.Vbr0 = 0.5 * (24.2 + 24.7)  # V
    sipm.temp0 = 21  # Reference temperature in degrees Celsius
    sipm.C_SiPM = 1070e-12  # F
    sipm.PDE = [0.38, 0.5]
    sipm.gain = [2.9e6, 6.3e6]
    sipm.DCR_mm2 = [50e3, 150e3]  # Hz/mm^2
    sipm.I_dark = [0.23e-6, 1.9e-6]  # A
    sipm.t_rise = [90e-12, 110e-12]  # s
    sipm.t_fall = [45e-9, 45e-9]  # s
    sipm.p_crosstalk = [0.08, 0.25]
    sipm.p_afterpulsing = [0.0075, 0.05]
    # Perform checks and calculations
    sipm.check_lengths()
    sipm.check_DCR_I_dark()
    sipm_types['J30035'] = sipm

    # Define S14161_6050
    # 不靠谱，还需要改
    sipm = SiPMType('S14161_6050')
    sipm.type = 'S14161_6050'
    sipm.conditions = Conditions(
        temp=25,
        dose=0,
        Vsipm=[38 + 2.0, 38 + 2.7, 38 + 3.0]
    )
    sipm.area = (6e-3) * (6e-3)  # m^2
    sipm.nums_pix = 14331
    sipm.fill_factor = 0.74
    sipm.k_Vbr = 0.034  # V/degree
    sipm.Vbr0 = 38  # V
    sipm.temp0 = 25  # Reference temperature in degrees Celsius
    sipm.C_SiPM = 2000e-12  # F
    sipm.PDE = [0.38, 0.5, 0.525]
    sipm.gain = [1.84e6, 2.5e6, 2.8e6]
    sipm.I_dark = [2.5e-6, 2.5e-6, 2.5e-6]  # A
    sipm.p_crosstalk = [0.045, 0.07, 0.1]
    # Perform checks and calculations
    sipm.check_lengths()
    sipm.check_DCR_I_dark()
    sipm_types['S14161_6050'] = sipm

    return sipm_types

# Instantiate the data
sipm_types = create_sipm_types()
