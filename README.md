Housing Price Prediction using Decision Tree Regressor

This project builds a Machine Learning model using Decision Tree Regression to predict housing prices based on various property features such as area, number of bedrooms, bathrooms, parking, furnishing status, and amenities.

The project includes data preprocessing, model training, hyperparameter tuning, and post-pruning to reduce overfitting.

Project Structure
housing-price-decision-tree/
│
├── decision tree regressor.ipynb   # Main Jupyter notebook
├── Housing Price.csv               # Dataset
├── README.md                       # Project documentation
Dataset

The dataset contains housing information including:

Area

Bedrooms

Bathrooms

Stories

Parking

Main road access

Guest room

Basement

Hot water heating

Air conditioning

Preferred area

Furnishing status

Price (target variable)

The target variable is:

price
Technologies Used

Python

Pandas

NumPy

Matplotlib

Scikit-learn

Jupyter Notebook

Machine Learning Workflow
1. Data Loading

The dataset is loaded using Pandas.

df = pd.read_csv("Housing Price.csv")
2. Data Preprocessing
Binary Encoding

Columns with yes/no values are converted to 1/0.

binary_cols=["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]

for col in binary_cols:
    df[col]=df[col].map({"yes":1,"no":0})
One-Hot Encoding

Categorical column furnishingstatus is converted into dummy variables.

df = pd.get_dummies(df, columns=["furnishingstatus"])
3. Feature and Target Split
X = df.drop("price", axis=1)
y = df["price"]
4. Train-Test Split

The dataset is split into 80% training and 20% testing.

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
5. Model Training

A DecisionTreeRegressor is trained on the training dataset.

dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
6. Model Evaluation

Model performance is evaluated using:

Mean Squared Error

R² Score

mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)
7. Hyperparameter Tuning

RandomizedSearchCV is used to find optimal hyperparameters.

Parameters tuned include:

max_depth

min_samples_leaf

min_samples_split

max_features

RandomizedSearchCV(
    DecisionTreeRegressor(),
    param_distributions=param_grid,
    n_iter=20,
    cv=5
)
8. Post Pruning (Cost Complexity Pruning)

To reduce overfitting, cost complexity pruning is applied.

Steps:

Compute ccp_alpha values

Train models with different alpha values

Select alpha with best test score

path = dtr.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = path.ccp_alphas
9. Decision Tree Visualization

The final pruned decision tree is visualized using:

plot_tree(model, feature_names=X.columns, filled=True)
Evaluation Metrics

The model is evaluated using:

R² Score

Mean Squared Error (MSE)

These metrics help measure how well the model predicts housing prices.

How to Run the Project

Clone the repository

git clone https://github.com/yourusername/housing-price-decision-tree.git

Install dependencies

pip install pandas numpy matplotlib scikit-learn

Run the notebook

jupyter notebook

Open:

decision tree regressor.ipynb
