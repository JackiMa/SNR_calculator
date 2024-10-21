# SiPM.py

# BUG & TODO
# DCR等参数 改成论文中的非线性插值
# 目前还没有把温度关联到所有参数，只有Vbr和Vov关联了温度
# 还没有把辐照剂量关联到所有参数


"""
This module provides the SiPM class, which models the behavior of a Silicon Photomultiplier (SiPM)
based on manufacturer data and user-defined conditions. The class allows calculation of various
parameters such as gain, PDE, DCR, and others, under different operating conditions like temperature,
bias voltage, and radiation dose.

Usage:
    condition = {'temp': 21, 'dose': 0, 'Vsipm': 24.45 + 5}
    sipm = SiPM("J60035", condition)
    sipm.print_parameters()

Attributes:
    - temp: Temperature in °C (modifiable)
    - Vsipm: Bias voltage applied to the SiPM (modifiable)
    - Vov: Overvoltage (modifiable via Vsipm or directly with a warning)
    - PDE_wl: Wavelength-dependent PDE function (modifiable)
    - dose: Radiation dose in Gy (modifiable)

Parameters that cannot be modified externally:
    - gain, Vbr, PDE, etc.

Methods:
    - calculate_parameters(): Calculates SiPM parameters based on current conditions.
    - print_parameters(): Prints the current SiPM parameters.
    - func_xxx(): Functions to calculate specific parameters like PDE, gain, etc.
"""

import numpy as np
import scipy.constants as CONST
import sys
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Add the directory containing sipm_data.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + '/..')
import config

global sipm_types, Conditions
exec(open(current_dir + r'/../data/sipm_data.py').read(), globals())

class SiPM:
    """
    The SiPM class models the behavior of a Silicon Photomultiplier based on
    manufacturer data and user-defined conditions.

    Parameters that can be modified externally:
        - temp
        - Vsipm
        - Vov
        - PDE_wl
        - dose

    Parameters that cannot be modified externally:
        - gain
        - Vbr
        - PDE
        - etc.
    """

    def __init__(self, model_name, condition):
        # Load data for the specified model
        self.model_name = model_name
        if model_name in sipm_types:
            self.data = sipm_types[model_name]
        else:
            raise ValueError(f"Model {model_name} not found in sipm_types.")

        # Initialize conditions
        self.condition = Conditions()
        self.set_conditions(condition)

        # PDE_wl attribute
        self.PDE_wl = None  # Placeholder for the PDE vs. wavelength function
        self.PDE_wl_scaled = None

        # Updating flag to prevent recursion in setters
        self._updating = False

        # Initialize parameters
        self.calculate_parameters()
        if config.verbose == 1:
            self.print_parameters()

    # ----------------------------------------------------------------------------------------
    # Condition Methods
    # ----------------------------------------------------------------------------------------

    def set_conditions(self, condition_dict):
        """
        Set the operating conditions for the SiPM.

        Args:
            condition_dict (dict): Dictionary with keys 'temp', 'dose', 'Vsipm'.

        Raises:
            ValueError: If conditions are outside the manual data range.
        """
        self.condition.temp = condition_dict.get('temp', self.data.conditions.temp)
        self.condition.dose = condition_dict.get('dose', self.data.conditions.dose)
        self.condition.Vsipm = condition_dict.get('Vsipm', self.data.conditions.Vsipm[0])

        # Validate conditions
        self.validate_conditions()

    def validate_conditions(self):
        """
        Validate the current conditions against the manufacturer's manual data.
        """
        self.exact_match = True
        manual_conditions = self.data.conditions

        if self.condition.temp != manual_conditions.temp:
            self.exact_match = False
            print(f"Warning: Condition temp={self.condition.temp} is not in the manual data. Parameters will be calculated based on formulas.")
        if self.condition.dose != manual_conditions.dose:
            self.exact_match = False
            print(f"Warning: Condition dose={self.condition.dose} is not in the manual data. Parameters will be calculated based on formulas.")
        if self.condition.Vsipm not in manual_conditions.Vsipm:
            self.exact_match = False
            if self.condition.Vsipm < min(manual_conditions.Vsipm) or self.condition.Vsipm > max(manual_conditions.Vsipm):
                print(f"Warning: Vsipm={self.condition.Vsipm} is outside the manual Vsipm range [{min(manual_conditions.Vsipm)}, {max(manual_conditions.Vsipm)}].")
            else:
                print(f"Warning: Condition Vsipm={self.condition.Vsipm} is not in the manual data. Parameters will be calculated based on formulas.")

    # ----------------------------------------------------------------------------------------
    # Calculation Methods
    # ----------------------------------------------------------------------------------------

    def calculate_parameters(self):
        """
        Calculate all SiPM parameters based on current conditions.
        """
        # First, calculate Vbr and Vov
        self._Vbr = self.calculate_Vbr()
        self._Vov = self.calculate_Vov()

        Vov_input = self._Vov

        # Prepare Vov_list based on data conditions
        Vsipm_list = self.data.conditions.Vsipm
        self.Vov_list = [Vsipm - self.data.Vbr0 for Vsipm in Vsipm_list]
        Vov_array = np.array(self.Vov_list)

        # Convert Vov-dependent parameters to arrays
        params = ['PDE', 'gain', 'DCR_mm2', 'I_dark', 'ECF',
                  't_rise', 't_fall', 'p_crosstalk', 'p_afterpulsing']
        param_arrays = {}
        for param in params:
            values = getattr(self.data, param)
            if values is not None:
                param_arrays[param] = np.array(values)
            else:
                param_arrays[param] = None

        # Fixed parameters
        self._area = self.data.area  # m^2
        self._nums_pix = self.data.nums_pix
        self._fill_factor = self.data.fill_factor
        self._k_Vbr = self.data.k_Vbr
        self._Vbr0 = self.data.Vbr0  # V
        self._C_SiPM = self.data.C_SiPM  # F

        # Calculate parameters
        if self.exact_match:
            # Directly use the manual data
            index = self.Vov_list.index(Vov_input)
            for param in params:
                setattr(self, f'_{param}', param_arrays[param][index] if param_arrays[param] is not None else None)
        else:
            # Perform linear fitting for Vov-dependent parameters
            for param in params:
                if param_arrays[param] is not None:
                    setattr(self, f'_{param}', self.linear_fit(Vov_array, param_arrays[param], Vov_input, param))
                else:
                    setattr(self, f'_{param}', None)
            # Check if Vov is outside the manual data range
            if Vov_input < min(self.Vov_list) or Vov_input > max(self.Vov_list):
                print(f"Warning: Vov={Vov_input:.2f} V is outside the manual Vov range [{min(self.Vov_list):.2f}, {max(self.Vov_list):.2f}]. Extrapolation may not be accurate.")

        # Calculated parameters
        self.calculate_additional_parameters()

        # Handle PDE_wl scaling if provided
        if self.PDE_wl is not None:
            self.scale_PDE_wl()

    def calculate_Vbr(self, temp=None):
        """
        Calculate the breakdown voltage Vbr at a given temperature.

        Args:
            temp (float): Temperature in °C. If None, uses current condition.

        Returns:
            float: Calculated breakdown voltage Vbr.
        """
        if temp is None:
            temp = self.condition.temp
        Vbr = self.data.Vbr0 + (temp - self.data.temp0) * self.data.k_Vbr
        return Vbr

    def calculate_Vov(self, temp=None, Vsipm=None):
        """
        Calculate the overvoltage Vov.

        Args:
            temp (float): Temperature in °C. If None, uses current condition.
            Vsipm (float): Bias voltage. If None, uses current condition.

        Returns:
            float: Calculated overvoltage Vov.
        """
        if temp is None:
            temp = self.condition.temp
        if Vsipm is None:
            Vsipm = self.condition.Vsipm
        Vbr = self.calculate_Vbr(temp)
        Vov = Vsipm - Vbr
        return Vov

    def calculate_additional_parameters(self):
        """
        Calculate additional parameters based on the current values.
        """
        area_mm2 = self._area * 1e6  # Convert area from m^2 to mm^2
        if self._DCR_mm2 is not None:
            self._DCR = self._DCR_mm2 * area_mm2  # Hz
        else:
            self._DCR = None
        if self._gain is not None and self._Vov is not None:
            self._C_pix_cal = self._gain * CONST.e / self._Vov  # F
            self._C_SiPM_cal = self._C_pix_cal * self._nums_pix  # F
            self._C_pix = self._C_SiPM / self._nums_pix  # F
        else:
            self._C_pix_cal = None
            self._C_SiPM_cal = None
            self._C_pix = None

    def linear_fit(self, x, y, x_input, param_name):
        """
        Perform linear fitting for a parameter and evaluate at x_input.

        Args:
            x (array): Independent variable data.
            y (array): Dependent variable data.
            x_input (float): Value at which to evaluate the fit.
            param_name (str): Name of the parameter (for error messages).

        Returns:
            float: Interpolated or extrapolated parameter value.
        """
        if len(x) < 2:
            return y[0]  # Not enough data points, return the first value
        m, c = np.polyfit(x, y, 1)
        y_input = m * x_input + c
        return y_input

    # ----------------------------------------------------------------------------------------
    # Parameter Functions
    # ----------------------------------------------------------------------------------------

    def scale_PDE_wl(self):
        """
        Scale the wavelength-dependent PDE function according to the current Vov.
        """
        if 'Vov0' in self.PDE_wl and 'func' in self.PDE_wl:
            Vov0 = self.PDE_wl['Vov0']
            func = self.PDE_wl['func']
            PDE_Vov = self.func_PDE(Vov=self._Vov)
            PDE_Vov0 = self.func_PDE(Vov=Vov0)
            scaling_factor = PDE_Vov / PDE_Vov0
            def scaled_func(wavelength):
                return func(wavelength) * scaling_factor
            self.PDE_wl_scaled = scaled_func
        else:
            print("PDE_wl must have 'Vov0' and 'func' keys.")

    def func_PDE(self, Vov=None, temp=None, Vsipm=None):
        """
        Return the Photon Detection Efficiency (PDE) for given conditions.

        Args:
            Vov (float): Overvoltage. If None, uses current Vov.
            temp (float): Temperature in °C.
            Vsipm (float): Bias voltage.

        Returns:
            float: PDE value.
        """
        if Vov is None:
            Vov = self.calculate_Vov(temp=temp, Vsipm=Vsipm)
        Vov_array = np.array(self.Vov_list)
        PDE_array = np.array(self.data.PDE)
        PDE_value = self.linear_fit(Vov_array, PDE_array, Vov, 'PDE')
        return PDE_value

    def func_gain(self, Vov=None, temp=None, Vsipm=None):
        """
        Return the gain for given conditions.

        Args:
            Vov (float): Overvoltage. If None, uses current Vov.
            temp (float): Temperature in °C.
            Vsipm (float): Bias voltage.

        Returns:
            float: Gain value.
        """
        if Vov is None:
            Vov = self.calculate_Vov(temp=temp, Vsipm=Vsipm)
        Vov_array = np.array(self.Vov_list)
        gain_array = np.array(self.data.gain)
        gain_value = self.linear_fit(Vov_array, gain_array, Vov, 'gain')
        return gain_value

    def func_Vov(self, temp=None, Vsipm=None):
        """
        Return the overvoltage Vov for given conditions.

        Args:
            temp (float): Temperature in °C.
            Vsipm (float): Bias voltage.

        Returns:
            float: Vov value.
        """
        return self.calculate_Vov(temp=temp, Vsipm=Vsipm)

    def func_DCR(self, Vov=None, temp=None, Vsipm=None):
        if Vov is None:
            Vov = self.calculate_Vov(temp=temp, Vsipm=Vsipm)
        Vov_array = np.array(self.Vov_list)
        DCR_mm2_array = np.array(self.data.DCR_mm2)
        DCR_mm2_value = self.linear_fit(Vov_array, DCR_mm2_array, Vov, 'DCR_mm2')
        area_mm2 = self.area * 1e6
        DCR_value = DCR_mm2_value * area_mm2
        return DCR_value

    def func_I_dark(self, Vov=None, temp=None, Vsipm=None):
        if Vov is None:
            Vov = self.calculate_Vov(temp=temp, Vsipm=Vsipm)
        Vov_array = np.array(self.Vov_list)
        I_dark_array = np.array(self.data.I_dark)
        I_dark_value = self.linear_fit(Vov_array, I_dark_array, Vov, 'I_dark')
        return I_dark_value

    def func_C_SiPM_cal(self, Vov=None, temp=None, Vsipm=None):
        if Vov is None:
            Vov = self.calculate_Vov(temp=temp, Vsipm=Vsipm)
        gain_value = self.func_gain(Vov=Vov)
        C_pix_cal = gain_value * CONST.e / Vov
        C_SiPM_cal = C_pix_cal * self.nums_pix
        return C_SiPM_cal

    def func_C_pix_cal(self, Vov=None, temp=None, Vsipm=None):
        if Vov is None:
            Vov = self.calculate_Vov(temp=temp, Vsipm=Vsipm)
        gain_value = self.func_gain(Vov=Vov)
        C_pix_cal = gain_value * CONST.e / Vov
        return C_pix_cal

    def func_t_rise(self, Vov=None, temp=None, Vsipm=None):
        if Vov is None:
            Vov = self.calculate_Vov(temp=temp, Vsipm=Vsipm)
        Vov_array = np.array(self.Vov_list)
        t_rise_array = np.array(self.data.t_rise)
        t_rise_value = self.linear_fit(Vov_array, t_rise_array, Vov, 't_rise')
        return t_rise_value

    def func_t_fall(self, Vov=None, temp=None, Vsipm=None):
        if Vov is None:
            Vov = self.calculate_Vov(temp=temp, Vsipm=Vsipm)
        Vov_array = np.array(self.Vov_list)
        t_fall_array = np.array(self.data.t_fall)
        t_fall_value = self.linear_fit(Vov_array, t_fall_array, Vov, 't_fall')
        return t_fall_value

    def func_p_crosstalk(self, Vov=None, temp=None, Vsipm=None):
        if Vov is None:
            Vov = self.calculate_Vov(temp=temp, Vsipm=Vsipm)
        Vov_array = np.array(self.Vov_list)
        p_crosstalk_array = np.array(self.data.p_crosstalk)
        p_crosstalk_value = self.linear_fit(Vov_array, p_crosstalk_array, Vov, 'p_crosstalk')
        return p_crosstalk_value

    def func_p_afterpulsing(self, Vov=None, temp=None, Vsipm=None):
        if Vov is None:
            Vov = self.calculate_Vov(temp=temp, Vsipm=Vsipm)
        Vov_array = np.array(self.Vov_list)
        p_afterpulsing_array = np.array(self.data.p_afterpulsing)
        p_afterpulsing_value = self.linear_fit(Vov_array, p_afterpulsing_array, Vov, 'p_afterpulsing')
        return p_afterpulsing_value

    def func_PDE_wl(self, temp=None, Vsipm=None):
        """
        Return the scaled wavelength-dependent PDE function.

        Args:
            temp (float): Temperature in °C.
            Vsipm (float): Bias voltage.

        Returns:
            function: Scaled PDE function or None if not defined.
        """
        if self.PDE_wl_scaled is not None:
            # If temp or Vsipm are specified, need to rescale PDE_wl
            if temp is not None or Vsipm is not None:
                # Update Vov and PDE
                Vov = self.calculate_Vov(temp=temp, Vsipm=Vsipm)
                PDE_Vov = self.func_PDE(Vov=Vov)
                Vov0 = self.PDE_wl['Vov0']
                PDE_Vov0 = self.func_PDE(Vov=Vov0)
                scaling_factor = PDE_Vov / PDE_Vov0
                func = self.PDE_wl['func']
                def scaled_func(wavelength):
                    return func(wavelength) * scaling_factor
                return scaled_func
            else:
                return self.PDE_wl_scaled
        else:
            print("PDE_wl is not defined or has not been scaled.")
            return None

    # ----------------------------------------------------------------------------------------
    # Utility Methods
    # ----------------------------------------------------------------------------------------

    def print_parameters(self):
        """
        Print the current SiPM parameters.
        """
        print('===============================================')
        print(f"SiPM Model: {self.model_name}")
        print(f"Conditions:")
        print(f"  Temperature: {self.condition.temp} °C")
        print(f"  Dose: {self.condition.dose} Gy")
        print(f"  Vsipm: {self.condition.Vsipm:.2f} V")
        print(f"  Breakdown Voltage (Vbr): {self.Vbr:.2f} V")
        print(f"  Overvoltage (Vov): {self.Vov:.2f} V")
        print(f"Parameters:")
        if self.area is not None:
            print(f"  Active Area: {self.area * 1e6:.2f} mm^2, Number of Pixels: {self.nums_pix}")
        if self.C_SiPM is not None:
            print(f"  Anode Capacitance (C_SiPM): {self.C_SiPM * 1e12:.2f} pF, Pixel Capacitance (C_pix): {self.C_pix * 1e15:.2f} fF")
        if self.C_SiPM_cal is not None:
            print(f"  Calculated Anode Capacitance (C_SiPM_cal): {self.C_SiPM_cal * 1e12:.2f} pF, Calculated Pixel Capacitance (C_pix_cal): {self.C_pix_cal * 1e15:.2f} fF")
        if self.gain is not None:
            print(f"  Gain: {self.gain / 1e6:.2f}E6")
        if self.PDE is not None:
            print(f"  Photon Detection Efficiency (PDE): {self.PDE:.3f}")
        if self.fill_factor is not None:
            print(f"  Fill Factor: {self.fill_factor:.3f}")
        if self.k_Vbr is not None:
            print(f"  Temperature Coefficient (k_Vbr): {self.k_Vbr:.4f} V/°C")
        if self.DCR_mm2 is not None:
            print(f"  Dark Count Rate (DCR_mm2): {self.DCR_mm2 / 1e6:.2f} MHz/mm^2, (DCR): {self.DCR / 1e6:.2f} MHz")
        if self.I_dark is not None:
            print(f"  Dark Current (I_dark): {self.I_dark * 1e6:.3f} μA")
        if self.t_rise is not None:
            print(f"  Rise Time (t_rise): {self.t_rise * 1e12:.2f} ps")
        if self.t_fall is not None:
            print(f"  Recharge Time Constant (t_fall): {self.t_fall * 1e9:.2f} ns")
        if self.p_crosstalk is not None:
            print(f"  Crosstalk Probability: {self.p_crosstalk:.3f}")
        if self.p_afterpulsing is not None:
            print(f"  Afterpulsing Probability: {self.p_afterpulsing:.3f}")
        print('===============================================\n')

    def __str__(self):
        """
        Return a string representation of the SiPM object.
        """
        info = f"SiPM Model: {self.model_name}\n"
        info += f"Conditions: temp={self.condition.temp} °C, dose={self.condition.dose} Gy, Vsipm={self.condition.Vsipm:.2f} V\n"
        info += f"Vbr={self.Vbr:.2f} V, Vov={self.Vov:.2f} V\n"
        return info

    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------------------------
    # Property Methods
    # ----------------------------------------------------------------------------------------

    # Modifiable properties
    @property
    def Vsipm(self):
        return self.condition.Vsipm

    @Vsipm.setter
    def Vsipm(self, value):
        self.condition.Vsipm = value
        self.validate_conditions()
        self.calculate_parameters()

    @property
    def temp(self):
        return self.condition.temp

    @temp.setter
    def temp(self, value):
        self.condition.temp = value
        self.validate_conditions()
        self.calculate_parameters()

    @property
    def dose(self):
        return self.condition.dose

    @dose.setter
    def dose(self, value):
        self.condition.dose = value
        self.validate_conditions()
        self.calculate_parameters()

    @property
    def Vov(self):
        return self._Vov

    @Vov.setter
    def Vov(self, value):
        if self._updating:
            self._Vov = value
        else:
            print("Warning: It is not recommended to modify Vov directly. Please set Vsipm instead.")
            print(f"Current conditions: temp={self.condition.temp}, Vsipm={self.condition.Vsipm}")
            # Update Vsipm
            self._Vbr = self.calculate_Vbr()
            self.condition.Vsipm = self._Vbr + value
            print(f"Updated Vsipm: {self.condition.Vsipm}")
            self.validate_conditions()
            self.calculate_parameters()

    # Properties for parameters that cannot be modified externally
    def _create_readonly_property(name):
        private_name = f"_{name}"
        def getter(self):
            return getattr(self, private_name)
        def setter(self, value):
            raise AttributeError(f"External modification of '{name}' is not allowed.")
        return property(getter, setter)

    # Generate read-only properties for parameters
    for param in ['gain', 'Vbr', 'PDE', 'DCR', 'DCR_mm2', 'area', 'nums_pix', 'C_SiPM', 'C_pix',
                  'C_SiPM_cal', 'C_pix_cal', 'I_dark', 't_rise', 't_fall', 'p_crosstalk',
                  'p_afterpulsing', 'fill_factor', 'k_Vbr', 'Vbr0']:
        locals()[param] = _create_readonly_property(param)

    # Setter for PDE_wl (allowed to modify externally)
    @property
    def PDE_wl(self):
        return self.__dict__['PDE_wl']

    @PDE_wl.setter
    def PDE_wl(self, value):
        """
        Set the wavelength-dependent PDE function.

        Args:
            value (dict): Dictionary with keys 'Vov' and 'func'.

        Raises:
            ValueError: If value is not in the correct format.
        """
        if value is None:
            self.__dict__['PDE_wl'] = None
            self.PDE_wl_scaled = None
        elif isinstance(value, dict) and 'Vov' in value and 'func' in value:
            self.__dict__['PDE_wl'] = value
            self.PDE_wl['Vov0'] = value['Vov']  # Save the Vov0 value
            self.scale_PDE_wl()
        else:
            raise ValueError("PDE_wl must be a dictionary with keys 'Vov' and 'func'.")

# ----------------------------------------------------------------------------------------
# Main Section for Unit Testing
# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Unit tests
    # Note: The following code assumes that sipm_types and required data are available.
    # You should adjust the code according to your data source.

    # Example usage
    condition = {'temp': 21, 'dose': 0, 'Vsipm': 24.45 + 5}
    sipm = SiPM("J60035", condition)
    print(sipm)
    sipm.print_parameters()

    # Modify Vsipm after creation
    print("\nModifying Vsipm to 24.45 + 2.5 V")
    sipm.Vsipm = 24.45 + 2.5
    sipm.print_parameters()

    # Modify Vov directly
    print("\nAttempting to modify Vov directly to 3.0 V")
    sipm.Vov = 3.0
    sipm.print_parameters()

    # Attempt to modify gain (should raise an error)
    try:
        print("\nAttempting to modify gain")
        sipm.gain = 6.0e6
    except AttributeError as e:
        print(e)

    # Set PDE_wl
    wavelengths = np.array([400, 500, 600, 700])  # nm
    pde_values = np.array([0.2, 0.3, 0.25, 0.1])  # Example PDE values
    interp_func = interp1d(wavelengths, pde_values, kind='linear', fill_value="extrapolate")
    sipm.PDE_wl = {'Vov': 2.5, 'func': interp_func}

    # Get scaled PDE_wl function
    fPDE_wl = sipm.func_PDE_wl()
    if fPDE_wl:
        print(f"\nScaled PDE_wl at 550 nm: {fPDE_wl(550):.3f}")

    # Get PDE function
    fPDE = sipm.func_PDE
    print(f"\nPDE at Vov=3.5 V: {fPDE(Vov=3.5):.3f}")
    # Modify Vsipm and get updated PDE
    sipm.Vsipm = 24.45 + 5.0
    print(f"PDE at Vov={sipm.Vov:.2f} V: {fPDE():.3f}")

    # Plot Vov-dependent parameters
    Vov_values = np.linspace(min(sipm.Vov_list), max(sipm.Vov_list), 100)
    PDE_values = [sipm.func_PDE(Vov=Vov) for Vov in Vov_values]
    gain_values = [sipm.func_gain(Vov=Vov) for Vov in Vov_values]
    DCR_mm2_values = [sipm.func_DCR(Vov=Vov)/1e6/(sipm.area*1e6) for Vov in Vov_values]
    I_dark_values = [sipm.func_I_dark(Vov=Vov)*1e6 for Vov in Vov_values]
    t_rise_values = [sipm.func_t_rise(Vov=Vov)*1e12 for Vov in Vov_values]
    t_fall_values = [sipm.func_t_fall(Vov=Vov)*1e9 for Vov in Vov_values]
    p_crosstalk_values = [sipm.func_p_crosstalk(Vov=Vov) for Vov in Vov_values]
    p_afterpulsing_values = [sipm.func_p_afterpulsing(Vov=Vov) for Vov in Vov_values]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(Vov_values, PDE_values, label='PDE')
    plt.xlabel('Vov (V)')
    plt.ylabel('PDE')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(Vov_values, np.array(gain_values)/1e6, label='Gain')
    plt.xlabel('Vov (V)')
    plt.ylabel('Gain (E6)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(Vov_values, DCR_mm2_values, label='DCR_mm2')
    plt.xlabel('Vov (V)')
    plt.ylabel('DCR_mm2 (MHz/mm^2)')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(Vov_values, I_dark_values, label='I_dark')
    plt.xlabel('Vov (V)')
    plt.ylabel('I_dark (μA)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(Vov_values, t_rise_values, label='t_rise')
    plt.xlabel('Vov (V)')
    plt.ylabel('t_rise (ps)')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(Vov_values, t_fall_values, label='t_fall')
    plt.xlabel('Vov (V)')
    plt.ylabel('t_fall (ns)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(Vov_values, p_crosstalk_values, label='p_crosstalk')
    plt.xlabel('Vov (V)')
    plt.ylabel('Crosstalk Probability')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(Vov_values, p_afterpulsing_values, label='p_afterpulsing')
    plt.xlabel('Vov (V)')
    plt.ylabel('Afterpulsing Probability')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Test another model
    print("\nTesting model J30035")
    condition = {'temp': 21, 'dose': 0, 'Vsipm': 24.45 + 5.0}
    sipm2 = SiPM("J30035", condition)
    sipm2.print_parameters()

    # Print object to see basic info
    print(sipm2)
