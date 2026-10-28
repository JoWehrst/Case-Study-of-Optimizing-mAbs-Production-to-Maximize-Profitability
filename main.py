# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================

import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()
from src.classes import Batch  # This allows you to import individual Classes from classes.py
from src.constants import *  # This imports all constants from constants.py

ph_optimal = sf.calculate_average(PH_LIMITS)
temperature_optimal = sf.calculate_average(TEMPERATURE_LIMITS)

batch = Batch(batch_id=1, ph_opt=ph_optimal, temperature_opt=temperature_optimal)
batch.display_info()
