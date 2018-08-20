# DemoMLMicrosoft
Un projet simple qui sert de démonstration de certaines capacité des outils Microsoft en machine learning : créer un modèle d'apprentissage automatique capable de reconnaitre des chiffres manuscrits utilisable dans une application UWP.
Pour une introduction et une explication de ce projet, lire [ce billet](https://blogs.msdn.microsoft.com/mlfrance/2018/08/09/une-mise-en-perspective/) du blog Microsoft Machine Learning.

Tout l'aspect ML.NET (chargement des données, entrainement et exportation d'un modèle prédictif) est géré dans le dossier [mldotnetMNIST](mldotnetMNIST), l'aspect Windows ML (chargement du modèle, application UWP et prediction) est géré dans le dossier [windowsmlMNIST](windowsmlMNIST). Le dossier [dataHelper](dataHelper) lui contient des scripts pythons qui générent des fichiers plus facilement utilisables à partir des données d'entrée sources (disponibles sur [ce site](http://yann.lecun.com/exdb/mnist/)).
