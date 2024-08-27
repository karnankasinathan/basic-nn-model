# Developing a Neural Network Regression Model
## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Developing a neural network regression model entails a structured process, encompassing phases such as data acquisition, preprocessing, feature selection, model architecture determination, training, hyperparameter optimization, performance evaluation, and deployment, followed by ongoing monitoring for refinement.

## Neural Network Model

![nn model](https://github.com/user-attachments/assets/d655cb86-f7c1-409e-ba7b-420adeff3144)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Karnan.K
### Register Number:212222230062
```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('DL').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'int':'float'})
dataset1 = dataset1.astype({'out':'float'})

dataset1.head()

x = dataset1[['int']].values
y = dataset1[['out']].values

X_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)

Ai_Brain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])

Ai_Brain.compile(optimizer = 'rmsprop', loss = 'mse')

Ai_Brain.fit(X_train1,y_train,epochs = 2000)

loss_df = pd.DataFrame(Ai_Brain.history.history)

loss_df.plot()

x_test1 = Scaler.transform(x_test)

Ai_Brain.evaluate(x_test1,y_test)

x_n1 = [[7]]

x_n1_1 = Scaler.transform(x_n1)

Ai_Brain.predict(x_n1_1)

```
## Dataset Information

![dl](https://github.com/user-attachments/assets/a2667578-a556-4671-b2ec-be09d7359aeb)

## OUTPUT

### Training Loss Vs Iteration Plot
![dl2](https://github.com/user-attachments/assets/b305cd57-4432-4a2b-9a1c-e8d95d041f14)


### Test Data Root Mean Squared Error

![dl3](https://github.com/user-attachments/assets/485858c2-1560-46d9-9ab1-cf5d49fcc91b)

### New Sample Data Prediction
![dl4](https://github.com/user-attachments/assets/8c34ac89-13ae-4f22-b1f0-be36e93a6d59)


## RESULT

Thus the program executed successfully
