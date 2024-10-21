import numpy as np
import sys
import os

# Add the directory containing config.py to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir+'/..')
import config

class OPAMP:
    def __init__(self, type_OPAMP):
        self.type_OPAMP = type_OPAMP
        self.set_parameters()
        
    def set_parameters(self):
        # Execute the data file to load parameters
        global opamp_parameters 
        exec(open(current_dir+r'/../data/opamp_data.py').read(), globals())
        
        if self.type_OPAMP in opamp_parameters:
            params = opamp_parameters[self.type_OPAMP]
            self.C_CM = params["C_CM"]
            self.C_DIFF = params["C_DIFF"]
            self.GBP = params["GBP"]
            self.I_N = params["I_N"]
            self.E_N = params["E_N"]
            self.A_OL = params["A_OL"]
        else:
            raise ValueError(f'No performance parameters for OPAMP type {self.type_OPAMP}.')
        
        self.omega_A = 2 * np.pi * self.GBP / self.A_OL
        
        if config.verbose == 1:
            self.print_parameters()
    
    def print_parameters(self):
        print('===============================================')
        print(f'OPAMP Type {self.type_OPAMP}')
        print(f'Common-Mode Input Capacitance = {self.C_CM} F')
        print(f'Differential Input Capacitance = {self.C_DIFF} F')
        print(f'Gain Bandwidth Product = {self.GBP} Hz')
        print(f'Input Current Noise = {self.I_N} A/sqrt(Hz)')
        print(f'Input Voltage Noise = {self.E_N} V/sqrt(Hz)')
        print(f'Open Loop Gain = {self.A_OL}')
        print(f'Open Loop Gain Bandwidth = {self.omega_A} rad/s')
        print('===============================================\n')

def main():
    config.verbose = 1
    
    type_OPAMP = "opa818"  # Example OPAMP type
    opamp = OPAMP(type_OPAMP)

if __name__ == "__main__":
    main()