#%% Packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#%% Data Prep
df = pd.read_csv('https://raw.githubusercontent.com/RiccardoBrioschi/Heart-attack-prediction/refs/heads/main/heart.csv')
df.head()


# %% Independent & Dependent Features
X = np.array(df.loc[:, df.columns !='output'])
y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")

# %% Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# %% Scale the data
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.fit_transform(X_test)

# %% Network Class
class NeuralNetworkFromScratch:
    def __init__(self, LR, X_train, y_train, X_test, y_test):
        self.w = np.random.randn(X_train.shape[1])
        self.b = np.random.randn()
        self.LR = LR
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.L_train = []
        self.L_test = []
    
    def activation(self, x):
        #Sigmoid
        return 1 / (1 + np.exp(-x))
    
    def dactivation(self, x):
        # Derivative of sigmoid
        return self.activation(x) * (1 - self.activation(x))
    
    def forward(self, X):
        hidden_1 = np.dot(X, self.w) + self.b
        activate_1 = self.activation(hidden_1)
        return activate_1
    
    def backward(self, X, y_true): # y_true to calculate losses
        # Calculate Gradients
        hidden_1 = np.dot(X, self.w) + self.b
        # predictions
        y_pred = self.forward(X)
        # Derivative of Losses to the predictions
        dL_dpred = 2 * (y_pred - y_true)
        # Derivative predictions to hidden layer
        dpred_dhidden1 = self.dactivation(hidden_1)
        dhidden1_db = 1
        dhidden1_dw = X

        # Derivative with respect to Weight vector and bias vector
        dL_db = dL_dpred * dpred_dhidden1 * dhidden1_db
        dL_dw = dL_dpred * dpred_dhidden1 * dhidden1_dw
        return dL_db, dL_dw
    
    def optimizer(self, dL_db, dL_dw):
        # update weights
        self.b = self.b - dL_db * self.LR
        self.w = self.w - dL_dw * self.LR

    def train(self, ITERATIONS):
        for i in range(ITERATIONS):
            # random position
            random_pos = np.random.randint(len(self.X_train))

            # forward pass
            y_train_true = self.y_train[random_pos]
            y_train_pred = self.forward(self.X_train[random_pos])

            # Calculate training losses
            L = np.sum(np.square(y_train_pred - y_train_true))
            self.L_train.append(L)

            # Calculate gradients
            dL_db, dL_dw = self.backward(self.X_train[random_pos], self.y_train[random_pos])

            # update weights
            self.optimizer(dL_db, dL_dw)

            # Calculate error for test data
            L_sum = 0
            for j in range(len(self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])
                L_sum += np.square(y_pred - y_true)
            self.L_test.append(L_sum)

        return "Training successful"
# %% Hyper parameters
LR = 0.1
ITERATIONS = 1000

# %% Model instance and training
nn = NeuralNetworkFromScratch(LR=LR, X_train=X_train_scale, y_train=y_train, X_test=X_test_scale, y_test=y_test)
nn.train(ITERATIONS=ITERATIONS)

# %% Check losses
sns.lineplot(x = list(range(len(nn.L_test))), y = nn.L_test)
# we can see losses reducing over time and stabalizing

# %% Iterate over test data
total = X_test_scale.shape[0] # rows
correct = 0  # Correct predictions
y_preds = []

for i in range(total):
    y_true = y_test[i]
    y_pred = np.round(nn.forward(X_test_scale[i]))
    y_preds.append(y_pred)
    correct += 1 if y_true == y_pred else 0

# %%  Calculate Accuracy
acc = correct / total

# %% Baseline Classifier
from collections import Counter

Counter(y_test)

# %% Confution Matrix
confusion_matrix(y_true=y_test, y_pred=y_preds)

# %%
