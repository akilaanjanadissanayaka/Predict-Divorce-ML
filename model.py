import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, plot_roc_curve, confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv('divorce_data.csv', sep = ';')
# print(df.head())

x_train, x_test, y_train, y_test = train_test_split(df.drop('Divorce', axis=1),
                                                   df['Divorce'],
                                                   test_size = 0.25,
                                                   random_state=42)

model_lr = LogisticRegression().fit(x_train, y_train)
y_pred_lr = model_lr.predict(x_test)

print('Accuracy score LR:  {:.4f}' .format(accuracy_score(y_test, y_pred_lr)))




# Saving model to disk
pickle.dump(model_lr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,2,4,1,0,0,0,0,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,2,1,2,0,1,2,1,3,3,2,1,1,2,3,2,1,3,3,3,2,3,2,1]]))


