import cv2
import face_recognition
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# =========================================================
# ÉTAPE 1 : CHARGEMENT DES VISAGES DE RÉFÉRENCE
# =========================================================

# Dossier contenant les images des personnes connues
images_path = r'C:\Users\ASUS\miniconda3\envs\myenv\projet_face_reco'

known_encodings = []   # encodages faciaux (vecteurs numériques des visages)
known_names = []       # noms associés aux visages
known_infos = {}       # infos textuelles associées à chaque personne

# Parcourir toutes les images du dossier
for filename in os.listdir(images_path):
    name, ext = os.path.splitext(filename)

    # Vérifier si le fichier est une image
    if ext.lower() in ['.jpg', '.jpeg', '.png']:

        # Charger l'image
        img = cv2.imread(os.path.join(images_path, filename))
        if img is None:
            continue

        # Conversion BGR -> RGB (nécessaire pour face_recognition)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extraction des encodages faciaux
        encodings = face_recognition.face_encodings(rgb)

        # Si un visage est détecté
        if len(encodings) > 0:
            known_encodings.append(encodings[0])  # stockage encodage
            known_names.append(name)              # stockage nom

            # Lecture des infos associées (fichier .txt)
            txt_path = os.path.join(images_path, name + '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                known_infos[name] = lines
            else:
                known_infos[name] = ["Pas d'infos disponibles"]

print("Visages chargés :", known_names)

# =========================================================
# POLICES POUR AFFICHAGE DU TEXTE (support accents)
# =========================================================

title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)
text_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)

# =========================================================
# FONCTION : AFFICHER TEXTE AVEC PIL (plus propre que OpenCV)
# =========================================================

def draw_text_pil(frame, text, position, font, color=(255, 255, 255)):
    # Conversion image OpenCV -> PIL
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Dessin du texte
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))

    # Retour vers OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# =========================================================
# FONCTION : CALCUL DE LA LARGEUR DU TEXTE
# =========================================================

def text_width(text, font):
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]

# =========================================================
# ÉTAPE 2 : CAPTURE VIDÉO (WEBCAM)
# =========================================================

cap = cv2.VideoCapture(0)
process_this_frame = True  # optimisation : traitement 1 frame sur 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Réduction de la frame pour accélérer le traitement
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # =====================================================
    # DÉTECTION ET RECONNAISSANCE FACIALE
    # =====================================================
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:

            # Calcul des distances avec les visages connus
            distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Inconnu"

            if len(distances) > 0:
                best_match_index = np.argmin(distances)

                # seuil de reconnaissance
                if distances[best_match_index] < 0.5:
                    name = known_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # =====================================================
    # AFFICHAGE DES RÉSULTATS
    # =====================================================
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # remettre à l’échelle originale
        top *= 4; right *= 4; bottom *= 4; left *= 4

        # couleur selon reconnaissance
        color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)

        # rectangle autour du visage
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # =================================================
        # CAS : PERSONNE CONNUE
        # =================================================
        if name != "Inconnu" and name in known_infos:

            lines = known_infos[name]
            line_height = 30

            # calcul taille du panneau selon texte
            longest_line = max(lines + [name], key=lambda l: text_width(l, text_font))
            panel_width = text_width(longest_line, text_font) + 30
            panel_height = line_height * (len(lines) + 1) + 20

            panel_x = right + 10
            panel_y = top

            frame_h, frame_w = frame.shape[:2]

            # ajustement si dépasse écran
            if panel_x + panel_width > frame_w:
                panel_x = max(0, left - panel_width - 10)

            if panel_y + panel_height > frame_h:
                panel_y = frame_h - panel_height - 10

            # fond semi-transparent
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y),
                          (panel_x + panel_width, panel_y + panel_height),
                          (20, 20, 20), cv2.FILLED)

            frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

            # bordure
            cv2.rectangle(frame, (panel_x, panel_y),
                          (panel_x + panel_width, panel_y + panel_height),
                          color, 2)

            # nom (titre)
            frame = draw_text_pil(frame, name.capitalize(),
                                   (panel_x + 12, panel_y + 8),
                                   title_font, color=(255, 255, 0))

            # ligne séparation
            cv2.line(frame, (panel_x + 10, panel_y + 38),
                     (panel_x + panel_width - 10, panel_y + 38), color, 1)

            # infos détaillées
            for i, line in enumerate(lines):
                y = panel_y + 45 + i * line_height
                frame = draw_text_pil(frame, line, (panel_x + 12, y),
                                       text_font, color=(255, 255, 255))

        # =================================================
        # CAS : PERSONNE INCONNUE
        # =================================================
        else:
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            frame = draw_text_pil(frame, name, (left + 6, bottom - 30),
                                   text_font, color=(255, 255, 255))

    # affichage fenêtre
    cv2.imshow('Face Recognition', frame)

    # quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#*******************************************************************

# import cv2
# import face_recognition
# import os
# import numpy as np

# # ----- ÉTAPE 1 : Chargement des visages de référence -----
# # 📂 Dossier contenant amal.jpg, amal.txt, ameni.jpg, ameni.txt
# images_path = r'C:\Users\ASUS\miniconda3\envs\myenv\projet_face_reco'

# known_encodings = []
# known_names = []
# known_infos = {}   # dictionnaire {nom: [liste de lignes du txt]}

# for filename in os.listdir(images_path):
#     name, ext = os.path.splitext(filename)

#     # Charger les images (jpg/png)
#     if ext.lower() in ['.jpg', '.jpeg', '.png']:
#         img = cv2.imread(os.path.join(images_path, filename))
#         if img is None:
#             continue
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encodings = face_recognition.face_encodings(rgb)
#         if len(encodings) > 0:
#             known_encodings.append(encodings[0])
#             known_names.append(name)

#             # Charger le .txt correspondant (ex: amal.jpg -> amal.txt)
#             txt_path = os.path.join(images_path, name + '.txt')
#             if os.path.exists(txt_path):
#                 with open(txt_path, 'r', encoding='utf-8') as f:
#                     lines = [line.strip() for line in f.readlines() if line.strip()]
#                 known_infos[name] = lines
#             else:
#                 known_infos[name] = ["Pas d'infos disponibles"]

# print("Visages chargés :", known_names)

# # ----- ÉTAPE 2 : Ouverture de la caméra -----
# cap = cv2.VideoCapture(0)
# process_this_frame = True

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#     if process_this_frame:
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []
#         for face_encoding in face_encodings:
#             distances = face_recognition.face_distance(known_encodings, face_encoding)
#             name = "Inconnu"
#             if len(distances) > 0:
#                 best_match_index = np.argmin(distances)
#                 if distances[best_match_index] < 0.5:
#                     name = known_names[best_match_index]
#             face_names.append(name)

#     process_this_frame = not process_this_frame

#     # ----- ÉTAPE 3 : Affichage des résultats + panneau d'infos -----
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         top *= 4; right *= 4; bottom *= 4; left *= 4

#         color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
#         cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
#         cv2.putText(frame, name, (left + 6, bottom - 6),
#                     cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

#         # ----- Panneau d'infos (seulement si personne connue) -----
#         if name != "Inconnu" and name in known_infos:
#             lines = known_infos[name]

#             panel_x = right + 10                  # à droite du visage
#             panel_y = top
#             line_height = 25
#             panel_width = 220
#             panel_height = line_height * len(lines) + 20

#             # Empêcher le panneau de sortir de l'écran
#             frame_h, frame_w = frame.shape[:2]
#             if panel_x + panel_width > frame_w:
#                 panel_x = left - panel_width - 10  # bascule à gauche si pas de place à droite

#             # Fond semi-transparent
#             overlay = frame.copy()
#             cv2.rectangle(overlay, (panel_x, panel_y),
#                           (panel_x + panel_width, panel_y + panel_height),
#                           (0, 0, 0), cv2.FILLED)
#             frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

#             # Bordure du panneau
#             cv2.rectangle(frame, (panel_x, panel_y),
#                           (panel_x + panel_width, panel_y + panel_height),
#                           color, 1)

#             # Écriture des lignes
#             for i, line in enumerate(lines):
#                 y = panel_y + 25 + i * line_height
#                 cv2.putText(frame, line, (panel_x + 10, y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

#     cv2.imshow('Face Recognition', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import face_recognition
# import numpy as np
# import sqlite3
# import pickle

# # =====================================================
# # 🔵 CHARGEMENT DE LA BASE DE DONNÉES
# # =====================================================

# conn = sqlite3.connect("face_recognition.db")
# cursor = conn.cursor()

# known_names = []
# known_encodings = []
# known_infos = {}

# # Charger infos personnes
# cursor.execute("SELECT name, info FROM persons")
# persons = cursor.fetchall()

# for name, info in persons:
#     known_names.append(name)
#     known_infos[name] = info.split("\n") if info else []

# # Charger encodings
# cursor.execute("SELECT person_name, encoding FROM encodings")
# encodings = cursor.fetchall()

# for name, enc in encodings:
#     known_encodings.append(pickle.loads(enc))

# conn.close()

# print("✔ Base chargée :", known_names)

# # =====================================================
# # 🔵 WEBCAM
# # =====================================================

# cap = cv2.VideoCapture(0)
# process_this_frame = True

# face_locations = []
# face_names = []

# # =====================================================
# # 🔴 LOOP PRINCIPAL
# # =====================================================

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Réduction pour accélérer
#     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#     if process_this_frame:
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []

#         for face_encoding in face_encodings:

#             name = "Inconnu"

#             if len(known_encodings) > 0:
#                 distances = face_recognition.face_distance(known_encodings, face_encoding)
#                 best_match_index = np.argmin(distances)

#                 if distances[best_match_index] < 0.5:
#                     name = known_names[best_match_index]

#             face_names.append(name)

#     process_this_frame = not process_this_frame

#     # =====================================================
#     # 🟡 AFFICHAGE
#     # =====================================================

#     for (top, right, bottom, left), name in zip(face_locations, face_names):

#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4

#         color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)

#         cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

#         # Nom
#         cv2.putText(frame, name, (left, top - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#         # Infos si connu
#         if name != "Inconnu" and name in known_infos:
#             y = top + 25
#             for line in known_infos[name]:
#                 cv2.putText(frame, line, (left, y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#                 y += 20

#     cv2.imshow("Face Recognition + SQLite", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()