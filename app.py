{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "68d1fa4d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import streamlit as st\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3c6d780",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Chargement du dataset\n",
    "df = pd.read_csv('mobile_prices.csv')\n",
    "print(df)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c29ec0ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Affichage d'un aperçu des données\n",
    "st.write(\"Aperçu des données :\")\n",
    "st.dataframe(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f4fc4089",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Informations sur le dataset\n",
    "st.write(\"Informations sur le dataset :\",df.info())\n",
    "\n",
    "# Vérification des valeurs manquantes\n",
    "st.write(\"Valeurs manquantes :\",df.isnull().sum())\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "782d6b8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualisation de la distribution des gammes de prix\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.countplot(x='price_range', data=df)\n",
    "plt.title('Distribution des Gammes de Prix')\n",
    "plt.xlabel('Gamme de Prix')\n",
    "plt.ylabel('Nombre de Téléphones')\n",
    "st.pyplot(plt)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3d34317",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Matrice de corrélation\n",
    "plt.figure(figsize=(12, 8))\n",
    "correlation_matrix = df.corr()\n",
    "sns.heatmap(correlation_matrix, annot=True, fmt=\".2f\", cmap='coolwarm')\n",
    "plt.title('Matrice de Corrélation')\n",
    "st.pyplot(plt)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "cc4e6e5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Séparation des features et de la cible\n",
    "X = df.drop('price_range', axis=1)\n",
    "y = df['price_range']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "c2944324",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Normalisation des données\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "9e2627a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Division en ensembles d'entraînement et de test\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4ae127a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Création et entraînement du modèle Random Forest\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(X_train, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "91fa6d51",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prédictions sur l'ensemble de test\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Évaluation du modèle\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "st.write(f\"Précision du modèle : {accuracy:.2f}\")\n",
    "st.write(\"Rapport de classification :\")\n",
    "st.text(classification_report(y_test, y_pred))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "9518de67",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-09 15:06:52.347 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.347 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.355 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.357 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.359 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.359 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.364 Session state does not function when running a script without `streamlit run`\n",
      "2025-04-09 15:06:52.366 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.368 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.368 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.371 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.373 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.373 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.373 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.377 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.380 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.381 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.384 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.384 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.384 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.388 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.388 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.388 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.388 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.396 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.397 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:06:52.399 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "st.title('Prédiction des Gammes de Prix de Téléphones Mobiles')\n",
    "\n",
    "# Entrée des caractéristiques\n",
    "battery_power = st.number_input('Capacité de la batterie (mAh)', min_value=500, max_value=2000)\n",
    "ram = st.number_input('RAM (MB)', min_value=256, max_value=4000)\n",
    "px_width = st.number_input('Largeur en pixels', min_value=500, max_value=2000)\n",
    "px_height = st.number_input('Hauteur en pixels', min_value=0, max_value=2000)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "20bf8d4d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_price_range(features):\n",
    "    # Prétraitement\n",
    "    features_scaled = scaler.transform([features])\n",
    "    # Prédiction\n",
    "    prediction = model.predict(features_scaled)\n",
    "    return prediction[0]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "05a5b3ab",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-04-09 15:07:38.547 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:07:38.555 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:07:38.558 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:07:38.559 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-04-09 15:07:38.561 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "if st.button('Prédire la gamme de prix'):\n",
    "    prediction = predict_price_range([battery_power, ram, px_width, px_height])\n",
    "    price_ranges = {0: \"Entrée de gamme\", 1: \"Milieu de gamme\", 2: \"Haut de gamme\", 3: \"Premium\"}\n",
    "    st.write(f'Gamme de prix prédite : {price_ranges[prediction]}')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
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
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
