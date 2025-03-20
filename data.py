# Import necessary libraries
import seaborn as sns

# Load the penguins dataset. It requires package 'seaborn'
penguins = sns.load_dataset("penguins").dropna()

# Display dataset information
penguins.info()



