import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(37)

#CUSTOM PATH
file_path = ""

train_data = pd.read_csv(file_path)
test_data = pd.read_csv(file_path)

train_input = train_data.drop(columns=['PRODUCT_ID', 'TIMESTAMP', 'Y_Class', 'Y_Quality'])
train_target = train_data['Y_Class']

test_input = test_data.drop(columns=['PRODUCT_ID', 'TIMESTAMP'])

train_input = train_input.fillna(0)
test_input = test_input.fillna(0)

qual_col = ['LINE', 'PRODUCT_CODE']

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train_input[i])
    train_input[i] = le.transform(train_input[i])
    
    for label in np.unique(test_input[i]): 
        if label not in le.classes_: 
            le.classes_ = np.append(le.classes_, label)
    test_input[i] = le.transform(test_input[i]) 
print('Done.')

model = RandomForestClassifier(n_jobs=-1, random_state=37)

model.fit(train_input, train_target)

scores = cross_validate(model, train_input, train_target, return_train_score=True, n_jobs=-1)

#Checkout
print(np.mean(scores["train_score"]), np.mean(scores["test_score"]))