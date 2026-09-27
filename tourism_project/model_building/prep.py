

# Import pandas for data manipulation
import pandas as pd

# Import train_test_split to divide the dataset into training and testing sets
from sklearn.model_selection import train_test_split


# Load the tourism dataset
df = pd.read_csv("tourism_project/data/tourism.csv")


# Remove CustomerID because it is only an identifier and does not help in prediction
df.drop(columns=["CustomerID"], inplace=True)


# Remove the unnecessary index column created while saving the CSV
df.drop(columns="Unnamed: 0", inplace=True)


# Correct inconsistent spelling in the Gender column
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')


# Combine "Unmarried" with "Single" to maintain consistent marital-status categories
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")


# Remove duplicate records from the dataset
df = df.drop_duplicates()


# Separate independent features (X) from the target variable (y)
# ProdTaken is the target variable indicating whether the customer purchased the product
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]


# Split the data into training and testing sets
# test_size=0.2 means 80% of the data is used for training
# and 20% is used for testing
# random_state=42 ensures the same split every time
# stratify=y keeps the target-class ratio consistent across both splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Save the training features to a CSV file
Xtrain.to_csv("Xtrain.csv", index=False)

# Save the testing features to a CSV file
Xtest.to_csv("Xtest.csv", index=False)

# Save the training target values to a CSV file
ytrain.to_csv("ytrain.csv", index=False)

# Save the testing target values to a CSV file
ytest.to_csv("ytest.csv", index=False)


# Display a confirmation message after successful data preparation
print("Data prepared: train/test splits written.")
