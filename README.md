# Hygdra Forecasting 🚀

**Hygdra Forecasting** est un algorithme rapide et optimisé pour la prévision des tendances boursières, conçu pour aider les traders à prendre des décisions plus sûres et à maximiser leurs opportunités sur les marchés financiers.

---

## 📌 Fonctionnalités

- 📈 **Chargement et traitement avancé des données** via `yfinance`
- 🧠 **Modèles de deep learning** pour la prédiction des tendances
- 📊 **Extraction automatique de caractéristiques techniques** (Bollinger Bands, RSI, ROC, etc.)
- 🔥 **Optimisation dynamique du taux d'apprentissage** (scheduler Cosine Warmup)
- 🏠 **Architecture modulaire et extensible** pour différents horizons temporels (journaliers, horaires, minutes)
- ⚡ **Compatibilité GPU** pour un entraînement rapide

---

## ⚙️ Installation

### 👋 Prérequis

- **Python** `>=3.8`
- **GPU compatible CUDA** (optionnel, mais recommandé)
- **Minimum** : 2 cœurs CPU, 2 Go RAM

### 🏠 Installation via Docker

Utilisez Docker pour une configuration rapide et reproductible :

```bash
docker-compose up -d
```

> **Note :** Assurez-vous d'avoir installé Docker et Docker Compose sur votre machine.

### 🏠 Installation Locale

Il est recommandé d'exécuter le projet dans un environnement virtuel.

**Sur Linux/macOS :**

```bash
python3 -m venv .hygdra_forecasting_env
source .hygdra_forecasting_env/bin/activate
```

**Sur Windows (PowerShell) :**

```bash
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

### 🔥 Entraîner un Modèle

Pour entraîner un modèle, exécutez :

```bash
python3 hygdra_forecasting/model/train.py
```

### 🎯 Affiner un Modèle sur un ETF Sélectionné

1. **Sélectionnez les poids** du modèle entraîné.
2. **Chargez-les dans** `app/api/finetune.py`.
3. **Lancez l'entraînement de fine-tuning :**

```bash
python app/scheduler/finetune.py
```

### 📊 Effectuer une Inférence

Pour lancer une inférence, exécutez :

```bash
python app/scheduler/inference.py
```

### 🌍 Lancer l'API avec FastAPI

Pour démarrer l'API FastAPI :

```bash
uvicorn main:app --reload
```

---

## 🤖 Automatisation

Pour automatiser l'exécution du modèle à intervalles réguliers, utilisez :

```bash
python app/scheduler/scheduler.py
```

---

## 🐟 Sélection du Modèle et du Mode d'Exécution

Le script principal vous permet de choisir dynamiquement :

- Le modèle (ex. `ConvCausalLTSM`, `LtsmAttentionforecastPred`, `VisionLiquidNet`)
- Le type de chargeur de données (`StockDataset`, `StockGraphDataset`)
- Le mode d'exécution (`inférence`, `évaluation`, `entraînement`)

Utilisation :

```bash
python main.py --model ConvCausalLTSM --dataloader StockDataset --mode inference
```

---

## 🌟 Améliorations Futures

- Intégration de nouveaux modèles (Liquid Neural Networks, Transformers, etc.)
- Tests unitaires et d'intégration
- Mode en direct via l'API Kraken

---

## 📄 Licence

Ce projet est sous licence **GNU**.

---

## 📧 Contact

Bucamp Axle - [axle.bucamp@gmail.com](mailto:axle.bucamp@gmail.com)

---

Profitez du trading assisté par IA avec **Hygdra Forecasting** ! 🚀

---

