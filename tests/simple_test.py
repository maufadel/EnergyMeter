from energymeter import EnergyMeter
import time
import numpy as np

meter = EnergyMeter(disk_avg_speed=1600*1e6, 
                            disk_active_power=6, 
                            disk_idle_power=1.42, 
                            label="Matrix Multiplication", include_idle=False)
meter.begin()
(np.random.rand(300,300)**5)*1/np.random.rand(300,300)
meter.end()
print(meter.get_total_joules_per_component())