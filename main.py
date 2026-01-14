import numpy as np
import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from statsmodels.formula.api import ols 
from sklearn.cluster import KMeans
from shutil import make_archive, unpack_archive
from bin import util as util 
from bin.student import Student

"""
Questions de recherche:

1. Y a-t-il une différence significative entre les étudiants qui prévoient ou 
non poursuivre aux études supérieures au niveau de leur résultat moyen?
H1: Les étudiants qui ne prévoient pas poursuivre aux études supérieures ont 
résultat plus faible que ceux qui prévoient poursuivre aux études supérieures. 
H0: Il n'y a pas de différence significative entre les groupes
-> test t à 2 échantillons 

2. Est-ce que le sexe, être en couple, avoir accès à internet, faire des cours 
supplémentaires, et des activités extracurriculaires 
peuvent prédire le résultat moyen?
H1: Les variables être en couple, avoir accès à internet et faire des cours 
supplémentaires permettent de prédire le résultat moyen
H0: Aucune de ces variables ne permet de prédire le résultat moyen
-> régression linéaire 
(VD: G_moy, VI: sex, romantic, internet, paid, activities)

3. Dans quelle mesure les variables disponibles permettent de prédire la 
volonté de faire des études supérieures?
-> Algorithme d'apprentissage supercisé (y: higher, features: les varibles 
disponibles dans la banque de données initiale) 
-> Algorithmes: PCA pour réduction de dimensionalité, random forest classifier
Exploratoire, pas d'hypothèse 

4. Comment les données sont-elles organisées? Y a-t-il des clusters?
-> Algorithme d'apprentissage non-supervisé: kmeans
Exploratoire, pas d'hypothèse 
"""

# Téléchargement du cadre de données 
try: 
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'students.csv'))
    df.head()
except FileNotFoundError:
    print('File not found')

# Path pour sauvegarder figures
save_dir = os.path.join(os.path.dirname(__file__), 'figures') 

if not os.path.exists(save_dir): # Si n'existe pas, créer dossier 'figures' 
    os.makedirs(save_dir)

"""
PREPARATION DES DONNEES
"""
# Rejet valeurs manquantes 
df = util.rejet_valeurs_manquantes(df) # Rejet des nan

pre_rejet2 = len(df)

# Rejet des valeurs manquantes/mauvaises spécifiques au dataframe 
df = df.drop(df.romantic[df.romantic == 'none'].index[0], axis = 0) 
df = df.drop(df.schoolsup[df.schoolsup == '1'].index[0], axis = 0)

post_rejet2 = len(df)

print("\nRejet de", pre_rejet2 - post_rejet2, 
    "autres participant(s) en raison de valeur(s) manquante(s).\n")


# Création d'une variable moyenne de G1, G2, G3 et enlever ces colonnes 
df['G_moy'] = df[['G1', 'G2', 'G3']].mean(axis = 1).round(2) 
df = df.drop(labels = ['G1', 'G2', 'G3'], axis = 1).reset_index(drop=True)

# Encodage des labels 
enc = LabelEncoder()
for col in df.columns:
    if type(df[col][0]) is str:      # Sélection des colonnes contenant des str
        label_encoder = enc.fit(df[col])
        df[col] = label_encoder.transform(df[col])

"""
1. Y a-t-il une différence significative entre les étudiants qui prévoient 
ou non poursuivre aux études supérieures au niveau du résultat moyen?
"""
# H1: Étudiants qui ne prévoient pas poursuivre aux études supérieures ont 
# résultat plus faible que ceux qui prévoient poursuivre études supérieures. 
alternative = 'less'
res_g_moy = util.testt_2_gr_ind(df, 'higher', 'G_moy', alternative, save_dir)

"""
2. Est-ce que sexe, être en couple, accès à internet, faire cours supp. et 
activités extracurriculaires peuvent prédire résultat moyen?
"""
VI = ['sex', 'romantic', 'internet', 'paid', 'activities'] # VI
model = ols("G_moy ~ C(sex) + C(romantic) + C(internet)+ C(paid) + \
            C(activities)", data = df).fit()

# Variables indépendantes significatives
VI_sign = list(filter(
    lambda x: util.significatif(model.pvalues[VI.index(x)+1]) and x!=0, VI)) 

print(model.summary(), "\n")
print("\nVariables ayant effet significatif sur résultat : ", VI_sign, "\n")

for vi in VI_sign:
    sns.displot(df,x = 'G_moy', hue = vi).set(title = 
        "Histogramme résultat moyen pour chaque niveau de {0} ".format(vi)) 
    plt.savefig(os.path.join(save_dir, 'hist_{0}_gmoy'.format(vi))) 
    # plt.show()
    plt.clf()
    plt.cla()
    plt.close()

"""
3. Dans quelle mesure les variables disponibles permettent de prédire la 
volonté de faire des études supérieures?
Ici, le plus important est de maximiser la précision puisque les faux positifs 
sont plus graves que les faux négatifs, ie identifier un étudiant comme voulant 
poursuivre des études supérieures alors qu'il ne le veut pas fait qu'on ne l'aidera 
pas s'il vit des difficultés qui l'empêchent de se voir poursuivre ses études supérieures. 
VS il n'est pas grave de mettre des efforts sur un étudiant identifier comme ne 
voulant pas poursuivre des études supérieures alors qu'il le veut. 
"""
y = df['higher']
X = df.loc[:, df.columns != 'higher']

# Séparation en train et test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.33, 
                                                random_state = 42, stratify = y) 

# Standardisation des données
scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

# Nombre de composantes optimales selon ratio variance expliquée vs biais 
n_comp = util.optimal_n_comp(Xtrain, threshold = 0.023) 

# Random forest classifier
model_rf = RandomForestClassifier(random_state = 42).fit(Xtrain, ytrain)

pred_test = model_rf.predict(Xtest)

# Métriques de la classification
util.metrics(ytest, pred_test)

util.plot_classif_test('random_forest', Xtest, ytest, pred_test, save_dir) 

"""
4. Comment les données sont-elles organisées? Y a-t-il des clusters?
"""
# Pour déterminer le nombre optimal de clusters ->elbow method
X = df.loc[:, df.columns!='higher']


# Appliquer PCA (même transformation que dans le pipeline)
pca_for_elbow = PCA(random_state=42, n_components=2)
X_pca_elbow = pca_for_elbow.fit_transform(X)

# Elbow method sur les données PCA (2D) - même espace où le clustering sera fait
inertias = []
k_range = range(1, 11) # Test k from 1 to 10

for k in k_range:
    # Initialize and fit the KMeans model sur données PCA
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') 
    kmeans.fit(X_pca_elbow)
    # Append the inertia (WCSS) to the list
    inertias.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, marker='o', linestyle='--')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method to determine optimal k (sur données PCA 2D)")
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'optimal_clusters_number'))
plt.show()

optimal_k = int(input("Nombre optimal de clusters: "))

# Pipeline d'AA non-supervisé: PCA et kmeans clustering
model_aa_unsupervised = make_pipeline(PCA(random_state = 42, n_components = 2),  
                        KMeans(random_state = 42, n_clusters = optimal_k, n_init='auto')).fit(X)

ypred = model_aa_unsupervised.predict(X)

pca = model_aa_unsupervised.named_steps['pca'] # Extraire l'ACP du pipeline
X_pca = pca.transform(X)

kmeans = model_aa_unsupervised.named_steps['kmeans'] # Extraire kmeans
centers = kmeans.cluster_centers_                    # centres des clusters 

# Nombre de valeurs dans chaque cluster (trier par numéro de cluster)
num, sizes = np.unique(ypred, return_counts = True)
sorted_indices = np.argsort(num)  # Trier pour avoir clusters dans l'ordre
num = num[sorted_indices]
sizes = sizes[sorted_indices]

# Créer la colormap et mapper les couleurs aux clusters
cmap = plt.cm.get_cmap('viridis')
norm = plt.Normalize(vmin=ypred.min(), vmax=ypred.max())

# Scatter plot avec couleurs selon clusters
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c = ypred, s = 50, cmap = 'viridis', 
                      alpha = 0.6, edgecolors='k', linewidths=0.5)
plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 200, alpha = 0.8, 
            marker='X', edgecolors='black', linewidths=2, label='Centroids', zorder=10)

# Créer la légende avec les couleurs correspondantes
legend_elements = []
for i, (cluster_num, size) in enumerate(zip(num, sizes)):
    color = cmap(norm(cluster_num))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10,
                                     label=f'Cluster {cluster_num} (n={size})'))

plt.legend(handles=legend_elements, loc='best', title='Clusters')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Clustering par kmeans (visualisation 2D PCA)')
# plt.show()
plt.savefig(os.path.join(save_dir, 'kmeans_clustering'))
plt.clf()
plt.cla()
plt.close()




"""
OTHER
"""
"""
CODAGE RECURSIF ET ALGORITHME D'AUTOMATISATION
"""
print("\nÂges différents présents, en ordre d'apparition: ", df.age.unique())
print("\n...Mise en ordre des âges par codage récursif...\n")
util.quick_sort(df.age.unique()) 

print("\n...Algorithme d'automatisation ordonne les résultats moyens...\n")
print("Résultats moyens ordonnés (valeurs uniques): ")
gmoy_ordonne = util.insertion_sort(list(df['G_moy']))
print(np.unique(gmoy_ordonne))

"""
RENCONTRE D'UN ÉTUDIANT 
"""
num = int(input('\nQuel étudiant voulez-vous rencontrer? (nombre [0-1000]) '))
student = Student(df['age'][num], df['G_moy'][num], df['absences'][num], 
                  df['higher'][num]) 

student.presentation()

"""
CODE POUR BASE DE DONNÉES SQLITE
"""

table_name = "student" + str(num) # Nom du tableau 

data_student = tuple(df.iloc[num-1, :3]) # Seulement age, school, sex

path = os.path.dirname(__file__)

# Crée un tableau contenant age, school et sex de l'étudiant choisi
conn = util.__connect_db(path, table_name)
util.__creer_tableau_(path, table_name)
util.__definir_les_donnees_(path, table_name, data_student) 

# aller chercher les données dans le tableau créé
res = conn.execute('''SELECT * FROM {0}'''.format(table_name)).fetchall()
conn.commit()
conn.close()
print('\nDonnées du tableau sqlite: ', res)

"""
ARCHIVAGE DES DONNÉES
"""
# Archiver les graphiques produits 
base_dir = os.path.join(os.path.dirname(__file__), 'figures')
make_archive(base_name = os.path.join(os.path.dirname(__file__), 
                                      'figures'), 
                                      format = 'gztar', 
                                      base_dir=base_dir)

# # Extraire les graphiques archivés
# extract_dir = os.path.join(os.path.dirname(__file__), 
#                            'figures_extracted')
# unpack_archive(filename = os.path.join(os.path.dirname(__file__), 
#                                        'figures.tar.gz'), 
#                                        format = 'gztar', 
#                                        extract_dir = extract_dir)

print("\nLes figures sont disponibles dans le dossier 'Figures'")
print("Les résultats sont rapportés et interprétés dans results.txt")

"""
INFORMATIONS SUPPLÉMENTAIRES
"""
print("\nÉtudiants sans relation amoureuse :", np.sum(df['romantic'] == 0), 
      "Étudiants avec relation amoureuse:", np.sum(df['romantic'] == 1))
print("Étudiants sans internet :", np.sum(df['internet'] == 0), 
      "Étudiants avec internet:", np.sum(df['internet'] == 1))




