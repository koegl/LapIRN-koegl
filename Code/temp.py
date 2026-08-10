path = "/home/iml/fryderyk.koegl/code/LapIRN-koegl/tumour_analysis/tumour_bone_overlap_per_image_nonan.csv"


# remove all rows that contain NaN values in any column and save to the same file
import pandas as pd

df = pd.read_csv(path)
df = df.dropna()
df.to_csv(path, index=False)
