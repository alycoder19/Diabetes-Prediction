import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 

df= pd.read_csv(r"C:\Users\org\Desktop\NEW\Diabetes_prediction.csv")
print(df.head(7))
print(df.shape)

print("\n DATA CLEANING")

print(df.info())
print(df.isnull().sum())

if df.isnull().sum().sum() > 0:
    df = df.fillna(df.median())

duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()

print(df.describe())



sns.heatmap(df.corr(),annot=True,fmt='0.2f',linewidths=.5)
plt.show()

sns.countplot(x='Diagnosis',data=df,palette=['r','k'])
plt.show()

sns.boxenplot(df.Age)
plt.show()




df["y"] = df["Diagnosis"]     
df = df.drop("Diagnosis", axis=1)
X = df.drop("y", axis=1)
y = df["y"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "KNN Classifier": KNeighborsClassifier(n_neighbors=7),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
} 

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    accuracies[name] = acc

    print("\n\n\n")
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, pred))

   
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


print("\n\nModel Accuracies \n \n")
for name, acc in accuracies.items():
    print(f"{name}: {acc:.4f}")

plt.figure(figsize=(7, 4))
plt.bar(list(accuracies.keys()), list(accuracies.values()) ,color=['red','black'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()