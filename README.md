# Hygdra Forecasting 🚀

**Hygdra Forecasting** est un algorithme rapide et optimisé pour la prévision des tendances boursières, conçu pour aider les traders à prendre des décisions plus sûres et à maximiser leurs opportunités sur les marchés financiers.

---

## 📌 Fonctionnalités  

- 📈 **Chargement et traitement avancé des données** via `yfinance`  
- 🧠 **Modèles de deep learning** pour la prédiction des tendances  
- 📊 **Extraction automatique de caractéristiques techniques** (Bollinger Bands, RSI, ROC, etc.)  
- 🔥 **Optimisation dynamique du taux d'apprentissage** (scheduler Cosine Warmup)  
- 🏗️ **Architecture modulaire et extensible** pour différents horizons temporels (journaliers, horaires, minutes)  
- ⚡ **Compatibilité GPU pour un entraînement rapide**  

---

## ⚙️ Installation  

### 📋 Prérequis  

- **Python** `>=3.8`  
- **GPU compatible CUDA** (optionnel mais recommandé)  
- **Minimum** : 2 cœurs CPU, 2 Go RAM  

### 🏗️ Installation locale  

Il est recommandé d'exécuter le projet dans un environnement virtuel.  

**Sur Linux/macOS :**  
```bash
python3 -m venv .hygdra_forecasting_env
source .hygdra_forecasting_env/bin/activate
```

**Sur Windows (PowerShell) :**  
```powershell
python -m venv .hygdra_forecasting_env
.hygdra_forecasting_env\Scripts\Activate
```

Ensuite, installez les dépendances :  
```bash
pip install -r requirements.txt
pip install .
```
---

## 🚀 Utilisation  

### 🔥 Entraîner un modèle  
```bash
python3 hygdra_forecasting/model/train.py
```

### 🎯 Affiner un modèle sur un ETF sélectionné  
1. **Sélectionnez les poids** du modèle entraîné  
2. **Chargez-les dans** `app/api/finetune.py`  
3. **Lancez l'entraînement finetune :**  
   ```bash
   python finetune.py
   ```

### 📊 Effectuer une inférence  
```bash
python inference.py
```

### 🌍 Lancer l'API avec FastAPI  
```bash
uvicorn main:app --reload
```

---

## 🤖 Automatisation  
Pour automatiser l'exécution du modèle à intervalles réguliers :  
```bash
python3 scheduler.py
```

---
## 📜 Licence  
Ce projet est sous licence **GNU**.  

📧 **Contact** : Bucamp Axle - axle.bucamp@gmail.com  

🚀 **Profitez du trading assisté par IA avec Hygdra Forecasting !** 🚀  
  
## TODO
- Dockerised app
- frontend
- big data capable app
- live mode kraken api