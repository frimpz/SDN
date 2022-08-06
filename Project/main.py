import SwarmPackagePy
import testFunctions
from animation import animation
from CaLF import CaLF
from OriginalCa import OriginalCa
from OriginalPSO import OriginalPSO
import numpy as np
import matplotlib.pyplot as plt

function = testFunctions.sphere_function
#alh = SwarmPackagePy.ca(50, function, -10, 10, 2, 200)
alh = CaLF(50, function, -10, 10, 2, 200)
#alh = OriginalCa(50, function, -10, 10, 2, 200)
#alh = OriginalPSO(50, function, -10, 10, 2, 200)
#animation(alh.get_agents(), function, -10, 10)
print(function(alh.get_Gbest()))









