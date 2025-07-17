{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18edc693-cc8b-42c2-b373-e8c3e32b395a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "27be9ce5-5a1c-419b-ac5e-ae7c44d522c2",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv(r\"C:\\Users\\Akshay\\Downloads\\Telegram Desktop\\adult 3.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "753963bb-b6c1-46fd-9fa8-8ec499e88d19",
   "metadata": {},
   "outputs": [],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3bb65d43-67cd-465c-96a1-43b632a07a84",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a2b7169c-733b-4d8a-98c8-eaeb5692b951",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07b1d815-1fb6-47bb-b96e-7bc4497eb7ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null values (if false there is no any null values)\n",
    "data.isna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "222b7afb-2ae3-48e0-bb74-04592f66cdd4",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12975389-d259-4be0-8b59-eca76dc166a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(data.occupation.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b328dfe-8989-446c-af7e-0e52714733e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(data.gender.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a31757af-6417-47e4-bc84-c722c8c7ffa1",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(data.race.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0549694b-347d-4563-b304-a80573560d57",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(data['workclass'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "11b79cc3-5e2f-4597-ac98-8b8811c15720",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.occupation.replace({'?':'others'},inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c2644622-fd67-4333-a570-d5c9300ec471",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(data.occupation.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5453ad83-0202-4bf5-bb4d-d2a160c021a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.workclass.replace({'?':'Notlisted'},inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f27dda9-14ca-4f46-a42e-a672bfd70cca",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(data.workclass.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1572b0f4-e5fc-415a-a465-4fcb5e432ece",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7a69af6d-682d-46de-807d-49f7120333c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#redundency\n",
    "data.drop(columns='education',inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "03a2d7c5-5599-4bb1-8b5d-5fd946df8d6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ba35888e-83a5-483c-8cda-3b9cba7cec58",
   "metadata": {},
   "outputs": [],
   "source": [
    "#outlier\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "74253c8d-3937-4dcd-afdf-077cc11fe6fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.boxplot(data['age'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2a38c3e7-fab8-49f9-96bb-db8faedb1d9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=data[(data['age']<=75)&(data['age']>=17)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12bb31ba-dbb3-4d13-8b35-17694cb333e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.boxplot(data['age'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ef1a2c6d-60bf-4360-911c-c8f2c7c7d7f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c084911-28d6-4da6-9273-32172ba56ed2",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.boxplot(data['capital-gain'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "07ee9b43-027c-4f8f-ba1d-cad1b42a52fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.boxplot(data['capital-gain'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5668b26-878a-43d4-879b-5d147ce9c66f",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.boxplot(data['educational-num'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f223506f-c6ef-4c7f-80fc-d605490546c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=data[(data['educational-num']<=16)&(data['educational-num']>=5)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "666b5644-f15e-4a7d-96da-340be6aedee0",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.boxplot(data['educational-num'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e990410e-3db1-4aec-a553-877eaa0e2d09",
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.boxplot(data['hours-per-week'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "86a9202b-6d0b-4c03-a368-599d64307fb4",
   "metadata": {},
   "outputs": [],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b8b098b7-e278-489a-b8a4-c2aefae2f170",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=data.drop(columns=['education']) #redundant features removal"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a2313432-fc78-40a0-aa8e-c855bc955268",
   "metadata": {},
   "outputs": [],
   "source": [
    "data\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0c395a1c-f82a-4f23-b698-b063b16d1325",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder   #import libarary\n",
    "encoder=LabelEncoder()                       #create object\n",
    "data['workclass']=encoder.fit_transform(data['workclass']) #7 categories   0,1, 2, 3, 4, 5, 6,\n",
    "data['marital-status']=encoder.fit_transform(data['marital-status'])   #3 categories 0, 1, 2\n",
    "data['occupation']=encoder.fit_transform(data['occupation'])\n",
    "data['relationship']=encoder.fit_transform(data['relationship'])      #5 categories  0, 1, 2, 3, 4\n",
    "data['race']=encoder.fit_transform(data['race'])  \n",
    "data['gender']=encoder.fit_transform(data['gender'])    #2 catogories     0, 1\n",
    "data['native-country']=encoder.fit_transform(data['native-country'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd79bc50-e0e9-4698-82c5-13b14f50ee6f",
   "metadata": {},
   "outputs": [],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e5677273-4b31-4ad5-9cf7-4f468958671e",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=data.drop(columns=['income'])\n",
    "y=data['income']\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aa25c884-4592-4563-94b3-f48f199e4a0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)\n",
    "\n",
    "models = {\n",
    "    \"LogisticRegression\": LogisticRegression(),\n",
    "    \"RandomForest\": RandomForestClassifier(),\n",
    "    \"KNN\": KNeighborsClassifier(),\n",
    "    \"SVM\": SVC(),\n",
    "    \"GradientBoosting\": GradientBoostingClassifier()\n",
    "}\n",
    "results = {}\n",
    "\n",
    "for name, model in models.items():\n",
    "    pipe = Pipeline([\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('model', model)\n",
    "    ])\n",
    "    \n",
    "    pipe.fit(X_train, y_train)\n",
    "    y_pred = pipe.predict(X_test)\n",
    "    acc = accuracy_score(y_test, y_pred)\n",
    "    results[name] = acc\n",
    "    print(f\"{name} Accuracy: {acc:.4f}\")\n",
    "    print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a413c60-e678-43c1-b3c9-9bf6e11a054f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.bar(results.keys(), results.values(), color='skyblue')\n",
    "plt.ylabel('Accuracy Score')\n",
    "plt.title('Model Comparison')\n",
    "plt.xticks(rotation=45)\n",
    "plt.grid(True)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b374ef4-f5ea-421c-9744-03856fa94e49",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "import joblib\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Define models\n",
    "models = {\n",
    "    \"LogisticRegression\": LogisticRegression(max_iter=1000),\n",
    "    \"RandomForest\": RandomForestClassifier(),\n",
    "    \"KNN\": KNeighborsClassifier(),\n",
    "    \"SVM\": SVC(),\n",
    "    \"GradientBoosting\": GradientBoostingClassifier()\n",
    "}\n",
    "\n",
    "results = {}\n",
    "# Train and evaluate\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    preds = model.predict(X_test)\n",
    "    acc = accuracy_score(y_test, preds)\n",
    "    results[name] = acc\n",
    "    print(f\"{name}: {acc:.4f}\")\n",
    "\n",
    "# Get best model\n",
    "best_model_name = max(results, key=results.get)\n",
    "best_model = models[best_model_name]\n",
    "print(f\"\\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}\")\n",
    "\n",
    "# Save the best model\n",
    "joblib.dump(best_model, \"best_model.pkl\")\n",
    "print(\"✅ Saved best model as best_model.pkl\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0aed03f7-837e-409e-b18d-fab87a78154e",
   "metadata": {},
   "outputs": [],
   "source": [
    "%%writefile app.py\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "# Load the trained model\n",
    "model = joblib.load(\"best_model.pkl\")\n",
    "\n",
    "st.set_page_config(page_title=\"Employee Salary Classification\", page_icon=\"💼\", layout=\"centered\")\n",
    "\n",
    "st.title(\"💼 Employee Salary Classification App\")\n",
    "st.markdown(\"Predict whether an employee earns >50K or ≤50K based on input features.\")\n",
    "\n",
    "# Sidebar inputs (these must match your training feature columns)\n",
    "st.sidebar.header(\"Input Employee Details\")\n",
    "\n",
    "# ✨ Replace these fields with your dataset's actual input columns\n",
    "age = st.sidebar.slider(\"Age\", 18, 65, 30)\n",
    "education = st.sidebar.selectbox(\"Education Level\", [\n",
    "    \"Bachelors\", \"Masters\", \"PhD\", \"HS-grad\", \"Assoc\", \"Some-college\"\n",
    "])\n",
    "occupation = st.sidebar.selectbox(\"Job Role\", [\n",
    "    \"Tech-support\", \"Craft-repair\", \"Other-service\", \"Sales\",\n",
    "    \"Exec-managerial\", \"Prof-specialty\", \"Handlers-cleaners\", \"Machine-op-inspct\",\n",
    "    \"Adm-clerical\", \"Farming-fishing\", \"Transport-moving\", \"Priv-house-serv\",\n",
    "    \"Protective-serv\", \"Armed-Forces\"\n",
    "])\n",
    "hours_per_week = st.sidebar.slider(\"Hours per week\", 1, 80, 40)\n",
    "experience = st.sidebar.slider(\"Years of Experience\", 0, 40, 5)\n",
    "\n",
    "# Build input DataFrame (⚠️ must match preprocessing of your training data)\n",
    "input_df = pd.DataFrame({\n",
    "    'age': [age],\n",
    "    'education': [education],\n",
    "    'occupation': [occupation],\n",
    "    'hours-per-week': [hours_per_week],\n",
    "    'experience': [experience]\n",
    "})\n",
    "\n",
    "st.write(\"### 🔎 Input Data\")\n",
    "st.write(input_df)\n",
    "\n",
    "# Predict button\n",
    "if st.button(\"Predict Salary Class\"):\n",
    "    prediction = model.predict(input_df)\n",
    "    st.success(f\"✅ Prediction: {prediction[0]}\")\n",
    "\n",
    "# Batch prediction\n",
    "st.markdown(\"---\")\n",
    "st.markdown(\"#### 📂 Batch Prediction\")\n",
    "uploaded_file = st.file_uploader(\"Upload a CSV file for batch prediction\", type=\"csv\")\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    batch_data = pd.read_csv(uploaded_file)\n",
    "    st.write(\"Uploaded data preview:\", batch_data.head())\n",
    "    batch_preds = model.predict(batch_data)\n",
    "    batch_data['PredictedClass'] = batch_preds\n",
    "    st.write(\"✅ Predictions:\")\n",
    "    st.write(batch_data.head())\n",
    "    csv = batch_data.to_csv(index=False).encode('utf-8')\n",
    "    st.download_button(\"Download Predictions CSV\", csv, file_name='predicted_classes.csv', mime='text/csv')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44534496-a5a1-49d6-aafb-d900d82d319d",
   "metadata": {},
   "outputs": [],
   "source": [
    "!streamlit run app.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f2992ff-cb3e-46bd-bc8c-ac490614a554",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
