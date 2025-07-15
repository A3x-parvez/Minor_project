#!/usr/bin/env python
# coding: utf-8

# In[30]:


#Missing value counts 
def missing_percentage(df):
    for i in df.columns:
        if df[i].isnull().sum() != 0:
            print(i ,' : ',df[i].isnull().sum(),' : [',df[i].isnull().sum()*100/1460,'%]',)


# In[32]:


#detect the outlier columns and plot them 
def outlier(X):
    # Loop through each column in the DataFrame
    for column in X.columns:
        if np.issubdtype(X[column].dtype, np.number):  # Check if the column is numeric
            plt.figure(figsize=(7, 4))

            # KDE plot
            sns.boxplot(X[column])
            plt.title(f"boxplot Plot for {column}", fontsize=14)

            # plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()


# In[34]:


#value count of catagorical column in dataframe
def cat_value_count(df_new):
    import pandas as pd
    categorical_cols = df_new.select_dtypes(include=['object', 'category']).columns  # Select categorical columns
    for col in categorical_cols:
        print(f"Value counts for column: {col}\n")
        print(df_new[col].value_counts(), "\n" + "-"*40 + "\n")  # Display counts for each unique value
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns  # Select categorical columns
    return categorical_cols


# In[36]:


# Function to cap outliers using IQR
def cap_outliers_iqr(df):
    df_capped = df.copy()  # Make a copy to avoid modifying the original DataFrame

    # Loop through numeric columns and cap outliers based on IQR
    for column in df_capped.select_dtypes(include=[np.number]).columns:
        Q1 = df_capped[column].quantile(0.25)
        Q3 = df_capped[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap values outside the bounds to the lower or upper bound
        df_capped[column] = np.clip(df_capped[column], lower_bound, upper_bound)

    return df_capped


# In[38]:


#plot multiple plot in one functions
def All_plot(X):
    import numpy as np
    col = X.columns
    for column in col:
        if np.issubdtype(X[column].dtype, np.number):  # Check if the column is numeric
            fig, axes = plt.subplots(1, 3, figsize=(21, 6))  # 3 plots in a row

            skewness = X[column].skew()

            # Determine skewness type
            if skewness > 0:
                skew_type = "Positive Skewness"
            elif skewness < 0:
                skew_type = "Negative Skewness"
            else:
                skew_type = "Approximately Symmetrical"

            print(f"{column}: Skewness = {skewness:.2f} ({skew_type})")

            # KDE plot
            sns.kdeplot(X[column].dropna(), fill=True, color='blue', alpha=0.5, ax=axes[0])
            axes[0].set_title(f"KDE Plot for {column}\n(Skewness: {skewness:.2f}) ({skew_type})", fontsize=12)
            axes[0].set_xlabel(column)
            axes[0].set_ylabel('Density')
            axes[0].grid(alpha=0.3)

            # Q-Q plot
            stats.probplot(X[column].dropna(), dist="norm", plot=axes[1])
            axes[1].set_title(f"Q-Q Plot for {column}", fontsize=12)

            # Boxplot
            sns.boxplot(x=X[column], ax=axes[2])
            axes[2].set_title(f"Boxplot for {column}", fontsize=12)
            axes[2].set_xlabel(column)

            plt.tight_layout()
            plt.show()


# In[11]:


#plot the confusion matrix
def binary_confusion_plot(y_test,y_pred_r_test):
    from sklearn.metrics import confusion_matrix
    # Plot confusion matrix using seaborn heatmap
    cm = confusion_matrix(y_test, y_pred_r_test)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Purples",linewidths=1, linecolor='black',
                xticklabels=['0', '1'], 
                yticklabels=['0', '1'])
    plt.title('Confusion Matrix for Test')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# In[13]:


#confusion matrix for multiple class
def multi_confusion_plot(y_test,y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,y_pred)
    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm,cmap="YlGnBu",fmt='d',annot=True,linecolor='black',linewidths=0.6)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# In[15]:


#calculate F1 score 
def All_F1_score(y_test, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Compute F1 Scores
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Print Results
    print(f"Micro F1 Score: {f1_micro}")
    print(f"Macro F1 Score: {f1_macro}")
    print(f"Weighted F1 Score: {f1_weighted}")


# In[17]:


#KNN check with multiple neigbour value
def Dynamic_KNN(X_train,y_train,range=30,weights='distance',p=2,algorithm='ball_tree',leaf_size=30,metric='manhattan'):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    accuracy_list_test = []
    accuracy_list_train = []

    for i in range(1,range+1,2):
        knn = KNeighborsClassifier(n_neighbors=i,weights=weights, p=p, algorithm=algorithm ,leaf_size=leaf_size,metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        y_predict_test = knn.predict(X_train)
        accuracy_test = accuracy_score(y_test, y_pred)
        accuracy_train = accuracy_score(y_train, y_predict_test)
        accuracy_list_test.append(float(accuracy_test))
        accuracy_list_train.append(float(accuracy_train))
        print("K =",i," ,Test accuracy =",accuracy_test," ,Train accuracy =",accuracy_train)

    avg_accuracy_test = np.mean(accuracy_list_test)
    avg_accuracy_train = np.mean(accuracy_list_train)
    max_acc_test =np.max(accuracy_list_test) 
    max_acc_train =np.max(accuracy_list_train) 
    print("Average Accuracy Test =", avg_accuracy_test,"Average Accuracy Train =", avg_accuracy_train," ,Max Test Accuracy =",max_acc_test," ,Max Train Accuracy =",max_acc_train)


# In[19]:


#Nomial and Odinal column transformer
def Nominal_Odinal_Transformer(category_orders,ordinal_columns,nominal_columns):

    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import ColumnTransformer

    transformer = ColumnTransformer(
        transformers=[
            ('tnf1',OrdinalEncoder(categories=category_orders),ordinal_columns),
            ('tnf2',OneHotEncoder(sparse_output=False,drop='first'),nominal_columns)
        ],remainder='passthrough'
    )

    return transformer


# In[21]:


# Cross validation for Regression and Classification
def cross_validation(model, X, y, problem_type='classification', scoring='accuracy', n_splits=10, state_range=50):
    """
    Perform cross-validation across multiple random states and find the best one.

    Parameters:
    - model: scikit-learn model (classifier or regressor)
    - X: features (DataFrame or ndarray)
    - y: labels/target
    - problem_type: 'classification' or 'regression'
    - scoring: scoring metric (e.g., 'accuracy', 'neg_mean_squared_error')
    - n_splits: number of cross-validation folds
    - state_range: number of random states to try (0 to state_range-1)

    Returns:
    - best_random_state: the random_state that gave the highest mean score
    - best_score: corresponding score
    """
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
    import numpy as np

    best_score = -np.inf
    best_random_state = None

    for state in range(state_range):
        # Choose cross-validation strategy
        if problem_type == 'classification':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=state)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=state)

        # Cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        mean_score = np.mean(scores)
        print(f"Random state {state}: Mean {scoring} = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_random_state = state

    print(f"\n✅ Best random_state: {best_random_state} with mean {scoring} = {best_score:.4f}")
    return best_random_state, best_score


# In[23]:


#Apply column transformer in desire columns of a data set
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
import pandas as pd

def Advance_column_transformer(df, columns, method='yeo-johnson', standardize=False):
    """
    Applies PowerTransformer to specific columns using ColumnTransformer,
    keeps other columns unchanged.

    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to transform
    - method: 'yeo-johnson' or 'box-cox'
    - standardize: whether to standardize the output

    Returns:
    - transformed_df: full DataFrame with transformed columns
    - lambdas_df: DataFrame with lambdas for transformed columns
    """
    df_copy = df.copy()

    # Validate columns list
    columns = [col for col in columns if col in df_copy.columns]
    if len(columns) == 0:
        raise ValueError("No valid columns provided for transformation.")

    # Check for missing values in selected columns
    if df_copy[columns].isnull().any().any():
        raise ValueError("Input data contains NaN values. Please impute or drop missing values before transformation.")

    # Box-Cox requires strictly positive values
    if method == 'box-cox':
        for col in columns:
            if (df_copy[col] <= 0).any():
                raise ValueError(f"Box-Cox transformation requires all positive values in column '{col}'")

    # Build ColumnTransformer
    col_transformer = ColumnTransformer(
        transformers=[
            ('power', PowerTransformer(method=method, standardize=standardize), columns)
        ],
        remainder='passthrough'
    )

    # Fit and transform the data
    transformed_array = col_transformer.fit_transform(df_copy)

    # Retrieve columns after transformation
    transformed_feature_names = columns
    passthrough_columns = [col for col in df_copy.columns if col not in columns]
    all_columns_order = transformed_feature_names + passthrough_columns

    # Rebuild DataFrame with correct column order
    transformed_df = pd.DataFrame(transformed_array, columns=all_columns_order, index=df.index)

    # Get lambdas for transformed columns
    fitted_power_transformer = col_transformer.named_transformers_['power']
    lambdas_df = pd.DataFrame({
        'columns': columns,
        'lambdas': fitted_power_transformer.lambdas_
    })

    return transformed_df, lambdas_df


# In[25]:


# Function to clean text (keep only words)
def clean_text(text):
    text = re.sub(r'[^A-Za-z]', ' ', text)  # Remove everything except letters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()  # Convert to lowercase


# In[27]:


# Function to remove stopwords
def remove_stopwords(text):
    words = text.split()  # Tokenize (split into words)
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)  # Join back into a single string


# In[1]:


# Plots class distribution of target variable y using bar chart and pie chart side by side.
import matplotlib.pyplot as plt

def plot_class_balance(y, figsize=(12,5), bar_color='skyblue', pie_colors=None):
    """
    Plots class distribution of target variable y using bar chart and pie chart side by side.

    Parameters:
    - y: pandas Series or list/array of target labels
    - figsize: tuple, size of the matplotlib figure
    - bar_color: color of bars in bar chart
    - pie_colors: list of colors for pie chart slices, optional

    Returns:
    - None (displays plots)
    """
    # Convert y to pandas Series if not already
    import pandas as pd
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    counts = y.value_counts().sort_index()
    labels = counts.index.astype(str)
    sizes = counts.values

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Bar Chart
    axes[0].bar(labels, sizes, color=bar_color)
    axes[0].set_title('Class Distribution - Bar Chart')
    axes[0].set_xlabel('Classes')
    axes[0].set_ylabel('Counts')

    for i, count in enumerate(sizes):
        axes[0].text(i, count + max(sizes)*0.01, str(count), ha='center')

    # Pie Chart
    axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=pie_colors, startangle=90, counterclock=False)
    axes[1].set_title('Class Distribution - Pie Chart')

    plt.tight_layout()
    plt.show()


# In[ ]:




