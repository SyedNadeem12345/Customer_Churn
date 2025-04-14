import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# App Title
st.title("📊 Customer Churn Prediction App")

# Upload Data
uploaded_file = st.file_uploader("Upload Customer Churn CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")
    st.write(df.head())

    st.subheader("🔍 Data Visualizations")
    # Bar plot
    st.markdown("**Internet Service Distribution**")
    fig, ax = plt.subplots()
    df['InternetService'].value_counts().plot(kind='bar', color='orange', ax=ax)
    plt.xlabel("Internet Service")
    plt.ylabel("Count")
    st.pyplot(fig)

    # Histogram
    st.markdown("**Tenure Distribution**")
    fig2, ax2 = plt.subplots()
    df['tenure'].hist(bins=30, color='green', ax=ax2)
    st.pyplot(fig2)

    # Scatter plot
    st.markdown("**Tenure vs Monthly Charges**")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x='tenure', y='MonthlyCharges', data=df, color='brown', ax=ax3)
    st.pyplot(fig3)

    # Select model
    model_choice = st.selectbox("Choose a model for prediction:", ["Logistic Regression", "Decision Tree", "Random Forest"])

    # Feature selection
    features = ['tenure', 'MonthlyCharges']
    X = df[features]
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert Yes/No to 1/0

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        model = RandomForestClassifier(n_estimators=100)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Results
    st.subheader("✅ Model Evaluation")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Predict on uploaded data (optional)
    if st.checkbox("🔮 Predict Churn for Full Dataset"):
        churn_predictions = model.predict(X)
        df['Predicted_Churn'] = np.where(churn_predictions == 1, 'Yes', 'No')
        st.write(df[['tenure', 'MonthlyCharges', 'Predicted_Churn']].head(20))
