import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import shap
import warnings
warnings.filterwarnings('ignore')

# Load Data
df = pd.read_csv('cleaned_analysis_data.csv')

# Preprocessing
df[df.columns.drop('state')] = df[df.columns.drop('state')].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# Derive severity
def derive_severity(row):
    if row['number_killed'] >= 50:
        return 'High'
    elif row['number_killed'] >= 10:
        return 'Medium'
    else:
        return 'Low'

df['severity'] = df.apply(derive_severity, axis=1)

# Encode labels
le = LabelEncoder()
df['severity_encoded'] = le.fit_transform(df['severity'])

# Select features
numeric_columns = df.select_dtypes(include=[np.number]).columns.drop('severity_encoded')
X = df[numeric_columns]
y = df['severity_encoded']

# 🔧 Define the scaler before using it
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_categorical = to_categorical(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Begin PDF report
pdf = PdfPages('crash_analysis_report.pdf')

# 1. Distribution Plots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
features = ['total_cases', 'number_killed', 'number_injured', 'total_casualty', 'people_involved']
axs = axs.flatten()
for i, col in enumerate(features):
    sns.histplot(df[col], bins=30, kde=True, ax=axs[i], color='steelblue')
    axs[i].set_title(f'Distribution of {col}')
axs[-1].axis('off')
pdf.savefig(fig)
plt.close(fig)

# 2. Causative Factor Frequencies
causative_factors = ['spv', 'upd', 'tbt', 'mdv', 'bfl', 'ovl', 'dot', 'wot']
causative_sums = df[causative_factors].sum().sort_values(ascending=False)
fig = plt.figure(figsize=(10, 6))
sns.barplot(x=causative_sums.index, y=causative_sums.values, palette="muted")
plt.title("Frequency of Causative Factors")
pdf.savefig(fig)
plt.close(fig)

# 3. Crash Frequency by State
fig = plt.figure(figsize=(14, 6))
sns.countplot(data=df, x='state', order=df['state'].value_counts().index, palette="deep")
plt.xticks(rotation=90)
plt.title("Crash Records by State")
pdf.savefig(fig)
plt.close(fig)

# 4. Correlation Matrix
fig = plt.figure(figsize=(12, 10))
corr = df[numeric_columns].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
pdf.savefig(fig)
plt.close(fig)

# 5. Deep Learning Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(y_categorical.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=0)

# 6. Accuracy and Loss Curves
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(history.history['accuracy'], label='Train')
axs[0].plot(history.history['val_accuracy'], label='Val')
axs[0].set_title('Model Accuracy')
axs[0].legend()
axs[1].plot(history.history['loss'], label='Train')
axs[1].plot(history.history['val_loss'], label='Val')
axs[1].set_title('Model Loss')
axs[1].legend()
pdf.savefig(fig)
plt.close(fig)

# 7. Evaluation
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_labels)
fig = plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
pdf.savefig(fig)
plt.close(fig)

# Classification Report
report = classification_report(y_true, y_pred_labels, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv")

# 8. Logistic Regression Feature Importance
clf = LogisticRegression(max_iter=1000)
clf.fit(X_scaled, y)
result = permutation_importance(clf, X_scaled, y, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()[::-1]
fig = plt.figure(figsize=(10, 6))
plt.barh(X.columns[sorted_idx], result.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance (Logistic Regression)")
pdf.savefig(fig)
plt.close(fig)

# 9. SHAP Explainability (if possible)
try:
    explainer = shap.Explainer(model, X_train[:100])
    shap_values = explainer(X_test[:50])
    fig = shap.summary_plot(shap_values, X_test[:50], feature_names=X.columns, show=False)
    pdf.savefig(bbox_inches='tight')
    plt.close()
except Exception as e:
    print("SHAP could not be generated:", str(e))

# Save PDF
pdf.close()
print("✅ All analysis completed. PDF saved as crash_analysis_report.pdf")
