import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

df_train = pd.read_csv('train.csv', index_col='PassengerId')
df_test = pd.read_csv('test.csv', index_col='PassengerId')

def take_name_part_from_df(df, col_name='Name', want_to_add_parts=['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.'], missing_name_part='no_name_part'):
    dict_name_parts = {}
    for name in df[col_name]:
        name_parts = name.split(' ')
        for name_part in name_parts:
            if name_part not in dict_name_parts.keys():
                dict_name_parts[name_part] = 1
            else:
                dict_name_parts[name_part] += 1
    
    df_name_parts = pd.DataFrame.from_dict(dict_name_parts, orient='index')
    df_name_parts = df_name_parts.reset_index()
    df_name_parts.columns = ['name_part', 'cnt']
    
    name_part_to_df = []

    for p_name in df[col_name]:
        counter = 0
        for name_part in want_to_add_parts:
            if name_part in p_name:
                name_part_to_df.append(name_part)
                continue
            else:
                counter += 1
                if counter == len(want_to_add_parts):
                    name_part_to_df.append(missing_name_part)
                    
    return name_part_to_df

def adding_data_to_df(df, list_of_cols_to_add, list_of_names_of_cols_to_add):
    for col_index in range(len(list_of_cols_to_add)):
        col = list_of_cols_to_add[col_index]
        col_name = list_of_names_of_cols_to_add[col_index]
        df[col_name] = col
    return df

def transform_data_for_model(df):

    list_of_cols_to_add = []
    list_of_names_of_cols_to_add = []

    name_parts = take_name_part_from_df(df, col_name='Name', want_to_add_parts=['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.'], missing_name_part='no_name_part')
    list_of_cols_to_add.append(name_parts)
    list_of_names_of_cols_to_add.append('name_part')

    cab_num_for_df = []
    for cab_num in df.Cabin:
        if pd.isna(cab_num):
            cab_num_for_df.append('N')
        else:
            cab_num_for_df.append(cab_num.split(' ')[0][0])
    list_of_cols_to_add.append(cab_num_for_df)
    list_of_names_of_cols_to_add.append('cabin_letter')

    list_of_cols_to_add.append(df['Sex'].map({'male':0, 'female':1}))
    list_of_names_of_cols_to_add.append('sex_binary')
    
    df = adding_data_to_df(df, list_of_cols_to_add, list_of_names_of_cols_to_add)
    list_of_cols_to_add = []
    list_of_names_of_cols_to_add = []
    
    df['Embarked'] = df.Embarked.fillna('S')
    mean_age_dict = df[['name_part', 'Age']].groupby('name_part').agg({'Age':'median'}).to_dict(orient='dict')['Age']
    list_of_cols_to_add.append(df.apply(lambda row: mean_age_dict[row['name_part']] if np.isnan(row['Age']) else row['Age'], axis=1))
    list_of_names_of_cols_to_add.append('age_no_nan')

    df_for_return = adding_data_to_df(df, list_of_cols_to_add, list_of_names_of_cols_to_add)

    return df_for_return

df_train_for_model = transform_data_for_model(df_train)
df_train_for_model = df_train_for_model[['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked', 'name_part', 'cabin_letter', 'sex_binary', 'age_no_nan']]
df_train_for_model = pd.get_dummies(df_train_for_model, prefix=['Embarked','name_part','cabin_letter'], columns=['Embarked','name_part','cabin_letter'])

feature_cols = df_train_for_model.columns[1:]
y = df_train_for_model['Survived']
X = df_train_for_model[feature_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7) # 70% training and 30% test

clf_dt = DecisionTreeClassifier(max_depth=4,  min_samples_leaf=20)
clf_rf = RandomForestClassifier(max_depth=4)
clf_xgb = XGBClassifier()
cat_features = [0, 1]
clf_cb = CatBoostClassifier(iterations=100, learning_rate=1, depth=4)

eclf2 = VotingClassifier(estimators=[('dt', clf_dt), ('rf', clf_rf), ('xgb', clf_xgb), ('catb', clf_cb)], voting='soft')
eclf2 = eclf2.fit(X_train, y_train)

clf_xgb = clf_xgb.fit(X_train, y_train)
clf_rf = RandomForestClassifier(max_depth=4)
clf_rf = clf_rf.fit(X_train, y_train)

































































