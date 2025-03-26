# Setup

- Créer et activer l'environement virtuel

`py -m venv .venv`

`.venv\Scripts\activate`

- Installer les packages du requirements.txt

`pip install -r requirements.txt`

source: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

- Créer le kernel et choisir l'environement virtuel depuis jupyter notebook

source: https://www.geeksforgeeks.org/using-jupyter-notebook-in-virtual-environment/

NB: J'ai du installer _ipython_ et _ipykernel_ avec ces commandes:

`pip install ipython`

`pip install ipykernel`

avant de rouler la commande de l'article:

`ipython kernel install --user --name=.venv`




