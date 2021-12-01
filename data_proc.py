import pandas as pd

df: pd.DataFrame = pd.read_csv("species.csv", dtype=str)
park_df: pd.DataFrame = pd.read_csv("parks.csv")

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
# animal_df.to_csv("animals.csv", index=False)

plant_df = df.loc[df["Category"].isin(plants)]
# plant_df.to_csv("plants.csv", index=False)

animal_df = animal_df.join(park_df.set_index("Park Name"), on="Park Name")
plant_df = plant_df.join(park_df.set_index("Park Name"), on="Park Name")

# Calculate unique number of animals and plants in each park
park_df.set_index("Park Code", inplace=True)
park_df["Animal Diversity"] = animal_df.groupby("Park Code")["Scientific Name"].nunique()
park_df["Plant Diversity"] = plant_df.groupby("Park Code")["Scientific Name"].nunique()

# Calculate density of unique animals and plants in each park
park_df["AD Density"] = park_df["Animal Diversity"] / park_df["Acres"]
park_df["PD Density"] = park_df["Plant Diversity"] / park_df["Acres"]

print(park_df)

print("\nHighest Density of Unique Animals:")
print(park_df.loc[park_df["AD Density"] == park_df.max("rows")["AD Density"]])
print("\nHighest Density of Unique Plants:")
print(park_df.loc[park_df["PD Density"] == park_df.max("rows")["PD Density"]])
