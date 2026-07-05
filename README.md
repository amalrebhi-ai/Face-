Ce projet est un système complet de reconnaissance faciale en temps réel développé en Python. Il combine OpenCV, face_recognition et SQLite pour permettre l’enregistrement, la reconnaissance et la gestion de personnes via la caméra.

Le système capture les visages en direct, les compare à une base de données locale d’encodages faciaux, puis affiche le nom et les informations associées à chaque personne reconnue.

Le projet inclut également un module d’enregistrement qui permet d’ajouter de nouvelles personnes automatiquement avec leur encodage facial et leurs informations personnelles stockées en base de données.

⚙️ Fonctionnalités principales
📸 Capture de visages via webcam en temps réel
🧠 Reconnaissance faciale basée sur les encodages (face_recognition)
💾 Stockage des données dans une base SQLite
➕ Ajout automatique de nouvelles personnes
🗂️ Association de données supplémentaires (infos personnelles)
🖥️ Interface vidéo avec affichage des résultats en direct
🎯 Affichage intelligent des informations selon la personne détectée
🛠️ Technologies utilisées
Python
OpenCV
face_recognition
NumPy
SQLite
PIL (Pillow)
📌 Objectif du projet

Ce projet vise à démontrer l’utilisation de la vision par ordinateur et des systèmes biométriques pour créer une application simple mais complète de reconnaissance faciale avec stockage persistant des données.
