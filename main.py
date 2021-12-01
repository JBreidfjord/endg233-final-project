# JONATHAN BREIDFJORD (30154027)
# Pandas used to import data

import numpy as np
import pandas as pd

plant_data = np.asarray(pd.read_csv("plants.csv", dtype=str))
animal_data = np.asarray(pd.read_csv("animals.csv", dtype=str))
park_data = np.asarray(pd.read_csv("parks.csv", dtype=str))


print(plant_data.shape)
print(animal_data.shape)
print(park_data.shape)


def compute_unique_species(data: np.ndarray, species):
    ...
