import pandas as pd

df: pd.DataFrame = pd.read_csv("species.csv", dtype=str)

unique_species = df["Category"].unique()

plants = ["Vascular Plant", "Fungi", "Nonvascular Plant", "Algae"]
animals = [
    "Mammal",
    "Bird",
    "Reptile",
    "Amphibian",
    "Fish",
    "Spider/Scorpion",
    "Insect",
    "Invertebrate",
    "Crab/Lobster/Shrimp",
    "Slug/Snail",
]

animal_df = df.loc[df["Category"].isin(animals)]
animal_df.to_csv("animals.csv", index=False)

plant_df = df.loc[df["Category"].isin(plants)]
plant_df.to_csv("plants.csv", index=False)
