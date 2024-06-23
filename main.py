import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components

# Charger les données
data2023 = pd.read_csv('parcoursup_data_2023.csv')
data2022 = pd.read_csv('parcoursup_data_2022.csv')
data2021 = pd.read_csv('parcoursup_data_2021.csv')
data2020 = pd.read_csv('parcoursup_data_2020.csv')
data2019 = pd.read_csv('parcoursup_data_2019.csv')

# Ajouter une colonne pour l'année
data2023['Année'] = 2023
data2022['Année'] = 2022
data2021['Année'] = 2021
data2020['Année'] = 2020
data2019['Année'] = 2019

# Combiner toutes les données dans un seul DataFrame
data_combined = pd.concat([data2023, data2022, data2021, data2020, data2019])


# Fonction pour mapper chaque institution à sa catégorie respective  
def map_type_filiere(filiere, year):
    filiere_lower = filiere.lower()
    if year == 2020 and "dut" in filiere_lower:
        return "DUT"
    elif year == 2020 and "but" in filiere_lower:
        return "DUT"
    elif "but" in filiere_lower:
        return "BUT"
    elif "bts" in filiere_lower:
        return "BTS"
    elif "licence" in filiere_lower:
        return "Licence"
    elif any(term in filiere_lower for term in ["classe préparatoire", "cycle préparatoire", "cpge", "cupge", "formation d'ingénieur"]):
        return "CPGE"
    else:
        return "Autres"

# Appliquer la fonction de mapping pour toutes les années
data2023['Filière'] = data2023.apply(lambda row: map_type_filiere(row['Filière de formation'], row['Année']), axis=1)
data2022['Filière'] = data2022.apply(lambda row: map_type_filiere(row['Filière de formation'], row['Année']), axis=1)
data2021['Filière'] = data2021.apply(lambda row: map_type_filiere(row['Filière de formation'], row['Année']), axis=1)
data2020['Filière'] = data2020.apply(lambda row: map_type_filiere(row['Filière de formation'], row['Année']), axis=1)
data2019['Filière'] = data2019.apply(lambda row: map_type_filiere(row['Filière de formation'], row['Année']), axis=1)
data_combined['Filière'] = data_combined.apply(lambda row: map_type_filiere(row['Filière de formation'], row['Année']), axis=1)

# Fonction pour préparer les données et générer un graphique à courbes avec Plotly
def generate_line_chart(data, selected_types_formation):
    # Filtrer les données pour exclure l'année 2019
    data_filtered = data[data['Année'] != 2019]

    # Compter le nombre de formations pour chaque filière et chaque année
    filiere_counts = data_filtered.groupby(['Année', 'Filière']).size().unstack(fill_value=0)
    
    # Calculer le pourcentage de chaque filière par année
    filiere_percentage = filiere_counts.div(filiere_counts.sum(axis=1), axis=0) * 100
    
    # Convertir les données en format long pour Plotly
    filiere_percentage_reset = filiere_percentage.reset_index()
    filiere_percentage_melted = pd.melt(filiere_percentage_reset, id_vars=['Année'], value_vars=selected_types_formation, var_name='Filière', value_name='Pourcentage')
    
    # Créer le graphique à courbes avec Plotly
    fig = px.line(filiere_percentage_melted, x='Année', y='Pourcentage', color='Filière', markers=True,
                  title="Répartition des types de formations en informatique (2019-2023)")
    
    # Définir la plage de l'axe y de 0 à 100
    fig.update_yaxes(range=[0, 100])
    
    # Convertir les étiquettes de l'axe x en années entières
    fig.update_xaxes(tickvals=np.arange(min(filiere_percentage_reset['Année']), max(filiere_percentage_reset['Année']) + 1, step=1))
    
    fig.update_layout(
        title={
            'text': "Répartition des types de formations en informatique (2019-2023)",
            'y':0.9,  
            'x':0.5,  
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 24  
            }
        }
    )
    
    return fig

# Liste des filières disponibles
types_formation = ["DUT", "BUT", "BTS", "Licence", "CPGE", "Autres"]

st.write("# Écart de présence entre les genres dans les études supérieures en informatique")


# Cases à cocher pour sélectionner les types de formations à afficher
st.write("##### Sélectionnez les types de formations à afficher pour le graphique ci-dessous")
selected_types_formation = [filiere for filiere in types_formation if st.checkbox(filiere, value=True, key=f"checkbox_{filiere}")]

# Vérifier si au moins une filière est sélectionnée
if selected_types_formation:
    # Générer et afficher le graphique à courbes pour les filières sélectionnées
    st.plotly_chart(generate_line_chart(data_combined, selected_types_formation))
else:
    st.write("Veuillez sélectionner au moins un type de formation pour afficher le graphique.")


# Fonction pour générer un graphique à courbes montrant le pourcentage d'admises filles par année, filtré par type de filière
def generate_admises_filles_chart(data, selected_types_formation):
    # Filtrer les données par filière
    data_filtered = data[data['Filière'].isin(selected_types_formation)]
    
    # Calculer le nombre total d'admis et le nombre d'admises filles par année et par filière
    total_admis_par_annee = data_filtered.groupby(['Année'])['Effectif total des candidats ayant accepté la proposition de l’établissement (admis)'].sum()
    admises_filles_par_annee = data_filtered.groupby(['Année'])['Dont effectif des candidates admises'].sum()

    # Calculer le pourcentage d'admises filles
    pourcentage_admises_filles = (admises_filles_par_annee / total_admis_par_annee) * 100

    # Préparer les données pour Plotly
    pourcentage_admises_filles_df = pourcentage_admises_filles.reset_index()
    pourcentage_admises_filles_df.columns = ['Année', 'Pourcentage d\'admises filles']

    # Créer le graphique à courbes avec Plotly
    fig = px.line(pourcentage_admises_filles_df, x='Année', y='Pourcentage d\'admises filles',
                  title="Pourcentage d'admises filles par année", markers=True,
                  labels={'Pourcentage d\'admises filles': 'Pourcentage d\'admises filles (%)'})

    # Définir la plage de l'axe y de 0 à 100
    fig.update_yaxes(range=[0, 100])
    
    # Convertir les étiquettes de l'axe x en années entières
    fig.update_xaxes(tickvals=np.arange(min(pourcentage_admises_filles_df['Année']), max(pourcentage_admises_filles_df['Année']) + 1, step=1))

    fig.update_layout(
        title={
            'text': "Pourcentage d'admises filles par année",
            'y':0.9,  
            'x':0.5,  
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 24  
            }
        }
    )

    return fig

# Liste des filières disponibles
types_formation = ["DUT", "BUT", "BTS", "Licence", "CPGE", "Autres"]

st.write("##### Sélectionnez les types de formations à afficher pour le graphique ci-dessous")
selected_types_formation_admises = [filiere for filiere in types_formation if st.checkbox(filiere, value=True)]

# Vérifier si au moins une filière est sélectionnée
if selected_types_formation_admises:
    # Générer et afficher le graphique pour le pourcentage d'admises filles
    st.plotly_chart(generate_admises_filles_chart(data_combined, selected_types_formation_admises))
else:
    st.write("Veuillez sélectionner au moins une filière pour afficher le graphique.")

# Fonction pour préparer les données et générer une matrice de corrélation avec Plotly
def generate_corr_matrix(data, year):
    data_selected = data[[
        "Effectif total des candidats ayant accepté la proposition de l’établissement (admis)",
        "Dont effectif des candidates pour une formation",
        "Dont effectif des candidats garçons pour une formation",
        "Dont effectif des candidates admises",
        "Dont effectif des candidats garçons admis",
        "Dont effectif des admis boursiers néo bacheliers",
        "Dont effectif des admis issus de la même académie",
        "Dont effectif des admis néo bacheliers sans mention au bac",
        "Dont effectif des admis néo bacheliers avec mention Assez Bien au bac",
        "Dont effectif des admis néo bacheliers avec mention Bien au bac",
        "Dont effectif des admis néo bacheliers avec mention Très Bien au bac"
    ]]

    # Renommer les colonnes
    data_selected = data_selected.rename(columns={
        "Effectif total des candidats ayant accepté la proposition de l’établissement (admis)": "Admis totaux",
        "Dont effectif des candidates pour une formation": "Candidates filles",
        "Dont effectif des candidats garçons pour une formation": "Candidats garçons",
        "Dont effectif des candidates admises": "Admises filles",
        "Dont effectif des candidats garçons admis": "Admis garçons",
        "Dont effectif des admis boursiers néo bacheliers": "Admis boursiers",
        "Dont effectif des admis issus de la même académie": "Admis même académie",
        "Dont effectif des admis néo bacheliers sans mention au bac": "Admis sans mention",
        "Dont effectif des admis néo bacheliers avec mention Assez Bien au bac": "Admis mention Assez Bien",
        "Dont effectif des admis néo bacheliers avec mention Bien au bac": "Admis mention Bien",
        "Dont effectif des admis néo bacheliers avec mention Très Bien au bac": "Admis mention Très Bien"
    })

    # Centrer et réduire les données
    data_selected_centered = (data_selected - data_selected.mean()) / data_selected.std()

    # Calculer la matrice de corrélation
    corr_matrix = data_selected_centered.corr()

    # Créer le graphique de la matrice de corrélation avec Plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdYlBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Corrélation')
    ))

    # Ajouter des annotations pour les valeurs de la matrice de corrélation
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(
                x=corr_matrix.columns[j],
                y=corr_matrix.columns[i],
                text=f'{corr_matrix.iloc[i, j]:.2f}',
                showarrow=False,
                font=dict(color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
            )

    # Mettre à jour les titres et les étiquettes
    fig.update_layout(
        title=f'Matrice de corrélation {year}',
        xaxis=dict(title='Variables'),
        yaxis=dict(title='Variables')
    )

    fig.update_layout(
        title={
            'text': f'Matrice de corrélation {year}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 24
            }
        },
        xaxis=dict(title='Variables'),
        yaxis=dict(title='Variables')
    )

    return fig

# Liste des années disponibles
years = ["2023", "2022", "2021", "2020", "2019"]

st.write("## Matrice de corrélation des attributs des étudiants")

# Sélecteur pour choisir l'année pour la matrice de corrélation
st.write("##### Sélectionnez une année pour la matrice de corrélation")
selected_year_corr = st.selectbox("", years, key='corr_matrix')

# Charger les données de l'année sélectionnée pour la matrice de corrélation
if selected_year_corr == "2023":
    data_corr = data2023
elif selected_year_corr == "2022":
    data_corr = data2022
elif selected_year_corr == "2021":
    data_corr = data2021
elif selected_year_corr == "2020":
    data_corr = data2020
elif selected_year_corr == "2019":
    data_corr = data2019

# Générer et afficher la matrice de corrélation
st.plotly_chart(generate_corr_matrix(data_corr, selected_year_corr))





# Fonction pour générer la carte
def generate_map(data, selected_types, min_perc, max_perc):
    # Filtrer les données par type de formation et pourcentage de filles admises
    data_filtered = data[(data['Filière'].isin(selected_types)) & 
                         (data["% d’admis dont filles"] >= min_perc) & 
                         (data["% d’admis dont filles"] <= max_perc)]

    # Carte centrée sur la France
    carte = folium.Map(location=[46.603354, 1.888334], zoom_start=6)
    
    # Cluster de marqueurs
    marker_cluster = MarkerCluster().add_to(carte)
    
    # Ajouter un marqueur pour chaque formation
    for index, row in data_filtered.iterrows():
        # Vérifier que les coordonnées sont disponibles
        if not pd.isnull(row['Coordonnées GPS de la formation']):
            coords = row['Coordonnées GPS de la formation'].split(',')
            lat = float(coords[0])
            lon = float(coords[1])
            
            # Ajouter le contenu du graphique dans la popup
            popup_content = f'<div style="width: 450px;"><b>{row["Établissement"]}</b><br>' \
                            f'{row["Filière de formation"]}<br>'
                            

            if 'Taux d’accès' in row:
                popup_content += f'Taux d’accès: {row["Taux d’accès"]}%<br>'

            popup_content += f'Effectif total des candidats pour la formation: {row["Effectif total des candidats pour une formation"]}<br>' \
                            f'Dont effectif des candidats garçons pour la formation: {row["Dont effectif des candidats garçons pour une formation"]}<br>' \
                            f'Dont effectif des candidates pour la formation: {row["Dont effectif des candidates pour une formation"]}<br>' \
                            f'% des candidates pour la formation: {row["% d’admis dont filles"]}%<br>' \
                            f'Effectif total des candidats ayant accepté la proposition de l’établissement (admis): {row["Effectif total des candidats ayant accepté la proposition de l’établissement (admis)"]}<br>' \
                            f'Dont effectif des candidats garçons admis: {row["Dont effectif des candidats garçons admis"]}<br>' \
                            f'Dont effectif des candidates admises: {row["Dont effectif des candidates admises"]}<br>' \
                            f'% d’admis dont filles: {row["% d’admis dont filles"]}%<br>' \
                            f'<a href="{row["Lien de la formation sur la plateforme Parcoursup"]}">Lien de la formation sur la plateforme Parcoursup</a></div>'
            
            # Ajouter le marqueur à la carte
            folium.Marker(location=[lat, lon], popup=popup_content).add_to(marker_cluster)
    
    return carte

# Fonction pour afficher la carte dans Streamlit
def display_map(carte):
    map_html = carte._repr_html_()
    components.html(map_html, height=600)

# Sélecteur pour choisir l'année
st.write("## Carte interactive des formations en informatique")
selected_year = st.selectbox("Sélectionnez l'année", [2023, 2022, 2021, 2020, 2019])

# Sélection des types de formations
st.write("##### Sélectionnez les types de formations à afficher sur la carte")
#selected_types_formation_map = st.multiselect("Types de formations", types_formation, default=types_formation)
selected_types_formation_map = [filiere for filiere in types_formation if st.checkbox(filiere, value=True, key=f"checkbox_{filiere}_map")]


# Sélection du pourcentage de filles admises
st.write("##### Sélectionnez la plage de pourcentage de filles admises")
min_percentage, max_percentage = st.slider("Pourcentage de filles admises", 0, 100, (0, 100))



# Charger les données de l'année sélectionnée
if selected_year == 2023:
    data = data2023
elif selected_year == 2022:
    data = data2022
elif selected_year == 2021:
    data = data2021
elif selected_year == 2020:
    data = data2020
elif selected_year == 2019:
    data = data2019

# Générer et afficher la carte
if selected_types_formation_map:
    carte = generate_map(data, selected_types_formation_map, min_percentage, max_percentage)
    display_map(carte)
else:
    st.write("Veuillez sélectionner au moins un type de formation pour afficher la carte.")


# Fonction pour afficher les informations supplémentaires pour chaque dataset
def display_dataset_info(data, year):
    num_formations = data.shape[0]
    num_candidates_garcons = data["Dont effectif des candidats garçons pour une formation"].sum()
    num_candidates_filles = data["Dont effectif des candidates pour une formation"].sum()
    num_admis_garcons = data["Dont effectif des candidats garçons admis"].sum()
    num_admises_filles = data["Dont effectif des candidates admises"].sum()
    
    st.write(f"#### Statistiques générales pour {year}")
    st.write(f"Nombre de formations : {num_formations}")
    st.write(f"Nombre de candidats garçons : {num_candidates_garcons}")
    st.write(f"Nombre de candidates filles : {num_candidates_filles}")
    st.write(f"Nombre d'admis garçons : {num_admis_garcons}")
    st.write(f"Nombre d'admises filles : {num_admises_filles}")
    st.write("Source des données : [Parcoursup](https://data.enseignementsup-recherche.gouv.fr/pages/parcoursupdata/?disjunctive.fili)")

st.write("## Explorer les données")

# Afficher chaque dataset avec un titre et les informations supplémentaires
st.write("### Données Parcoursup 2023 filière informatique")
st.write(data2023)
display_dataset_info(data2023, 2023)

st.write("### Données Parcoursup 2022 filière informatique")
st.write(data2022)
display_dataset_info(data2022, 2022)

st.write("### Données Parcoursup 2021 filière informatique")
st.write(data2021)
display_dataset_info(data2021, 2021)

st.write("### Données Parcoursup 2020 filière informatique")
st.write(data2020)
display_dataset_info(data2020, 2020)

st.write("### Données Parcoursup 2019 filière informatique")
st.write(data2019)
display_dataset_info(data2019, 2019)