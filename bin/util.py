import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 as sq
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from scipy import stats
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler


# Au moins 5 « def » sont présents (2%)
def preprocessing(x, y, balance, standardize):
    """
    Fait les étapes demandées du prétraitement des données 
    x (array): variables indépendantes
    y (array): variable dépendante binaire (0 et 1)
    balance (bool): True-> balancement par oversampling (False non)
    standardize (bool): True-> standardisation des données (False non)
    return:
        - x et y prétraitées 
    """

    if balance:
        ros = RandomOverSampler(random_state = 0)
        x, y = ros.fit_resample(x, y)

    if standardize:
        scaler = StandardScaler() # Pour standardiser les données en score z
        x = scaler.fit_transform(x)

    return x, y


def metrics(ytest, pred):
    """
    Appelle les fonctions classification_report, confusion_matrix et 
    accuracy_score et imprime leurs outputs. 
    Arg:
    - ytest: valeurs test de y (labels réels)
    - pred: valeurs prédites de y (labels prédits)
    
    Print:
    - Classification report:
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN) (for positive class = sensitivity, 
        for negative class = specificity)
        - F1-score: Harmonic mean of precision and recall (weighted average 
        of the precision and recall)
        - Support: number of samples of the true response that lies in 
        each class of target values.
        - Macro avr (averaging the unweighted mean per label)
        - Weighted avr (averaging the support-weighted mean per label)
        - Accuracy
        
    - Confusion matrix
        - Row : true labels
        - Column : predicted labels

    - Accuracy score: 
        - Fraction of correctly classified samples
        
    """
    print('\n---Classification report---')
    print(classification_report(ytest, pred))
    
    print('\n---Confusion matrix---')
    print(confusion_matrix(ytest, pred))
    
    print(f"\nAccuracy: {accuracy_score(ytest, pred):.2f}")
    print(f"Precision: {precision_score(ytest, pred):.2f}")
    print(f"Recall: {recall_score(ytest, pred):.2f}")
    print(f"F1-score: {f1_score(ytest, pred):.2f}")


def rejet_valeurs_manquantes(df):
    """
    Rejet des lignes avec des valeurs manquantes 
    (Code pour vérifier la présence des valeurs manquantes implémenté)
    df: cadre de données  
    """
    
    pre_rejet = len(df)

    # Si au moins une valeur manquante dans le dataframe
    if (df.isna()).sum().sum():         
        df = df.dropna(axis = 'index') # Rejet rangées des valeurs manquantes

    post_rejet = len(df)

    print("\nRejet de", pre_rejet - post_rejet, 
          "participant(s) en raison de valeur(s) manquante(s).\n")


    return df


def plot_classif_test(algo, Xtest, ytest, ypred, save_dir):
    """
    Imprime et sauvegarde un graphique des features du test set identifiés 
    selon leur classe en identifiant erreurs de classification de l'algorithme.
    Graphique 2d selon les deux premières dimensions du pca
    Algo: algorithme d'apprentissage automatique supervisé utilisé
    Xtest: test set des features
    ytest: test set des labels
    ypred: labels prédits
    save_dir: path où sauvegarder la figure 
    """
    for l, c, m in zip(range(0, 2), ('blue', 'green'), ('^', 's')):
        plt.scatter(Xtest[ytest == l, 0],
                    Xtest[ytest == l, 1],
                    color = c,
                    label = '{0} : {1}'.format(l, 'no' if l == 0 else 'yes'),
                    alpha = 0.5,
                    marker = m)
    errors = Xtest[np.where(ypred != ytest)] # Erreurs prédiction 

    plt.scatter(errors[:, 0], errors[:, 1], c = 'r', label = 'erreur')
    plt.legend()
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    # code pour le formattage de champ est utilisé (1%)
    plt.title('Classification par {0}'.format(algo)) 
    # plt.show()
    # code pour le formattage de champ est utilisé (1%)
    plt.savefig(os.path.join(save_dir, '{0}_plot_2d'.format(algo))) 
    plt.clf()
    plt.cla()
    plt.close()

def optimal_n_comp(Xtrain, threshold):
    """
    Trouve le nombre de composantes optimal selon le seuil du ratio variance 
    vs biais voulu
    Xtrain (array): train set des features
    threshold: seuil du ratio variance vs biais voulu
    return:
        - nombre de composantes optimal (int)
    """
    pca_test = PCA().fit(Xtrain)
    cumul = np.cumsum(pca_test.explained_variance_ratio_)
    n_comp = np.sum([(cumul[i] - cumul[i-1])/cumul[i-1] >= threshold 
                     for i in range(len(cumul)) if i!=0]) + 1 
                     # +1 pour ajouter la première composante  

    return n_comp

def testt_2_gr_ind(df, var_pop, var_diff, alternative, save_dir):
    """
    Évalue s'il y a une différence significative entre 2 groupes indépendants 
    au niveau d'une variable par un test t à 2 échantillons
    df (pandas dataframe): cadre de données 
    var_pop (str): variable séparant les échantillons  
    var_diff (str): variable pour laquelle la différence est évaluée
    alternative: hypothèse alternative
    source: https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.stats.ttest_ind.html)
        'two-sided': means of distributions underlying the samples are unequal.
        'less': mean of the distribution underlying the first sample is less 
        than the mean of the distribution underlying the second sample.
        'greater': mean of the distribution underlying the first sample is 
        greater than the mean of the distribution underlying the second sample.
    return:
        - res: liste contenant la valeur de la statistique t et la valeur p 
    """
    # Valeurs possibles de la variable séparant les groupes 
    val1, val2 = np.sort(df[var_pop].unique()) 

    res = stats.ttest_ind(df[df[var_pop] == val1][var_diff], 
                          df[df[var_pop] == val2][var_diff], 
                          alternative = alternative, equal_var=False)

    if significatif(res.pvalue):
        print("\nIl y a une différence significative entre les groupes")
        print("Test t: {:.2g}, valeur p: {:.2g}".format(res.statistic, 
                                                        res.pvalue) + "\n")

    else:
        print("\nIl n'y a pas de différence entre les groupes")
        print("Test t: {:.2g}, valeur p: {:.2g}".format(res.statistic, 
                                                        res.pvalue) + "\n")

    # Graphique pour les stats (statsmodels, SciPy) présent (2%)
    sns.displot(data = df, x = var_diff, hue = var_pop, kind = "kde", 
                fill=True).set(title = 'Fonction de distribution de densité') 
    # plt.show()
    # code pour le formattage de champ est utilisé (1%)
    plt.savefig(os.path.join(save_dir, 'testt_{0}_{1}'.format(var_pop, 
                                                              var_diff)))
    plt.clf()
    plt.cla()
    plt.close() 

    return res


def significatif(value, threshold = 0.05):
    """
    Évalue la significativité d'une valeur p selon seuil de significativité 
    value: valeur p
    threshold: seuil de significativité (default = 0.05)
    return:
        - True si significatif
        - False si non significatif 
    """
    return True if value <= threshold else False


def quick_sort(liste): 
    """
    liste (list or pandas series): liste à ordonner 
    return: 
        - liste ordonnée (list)
    source: https://stackoverflow.com/questions/26858358/a-recursive-function-
    to-sort-a-list-of-ints
    """
    if len(liste) <= 1: # cas minimal
        return liste
    else:
        print(liste)
        # Fonction appliquée sur éléments inférieurs au pivot, pivot, 
        # fonction appliquée sur éléments supérieurs au pivot
        # code pour codage récursif, présent (3%)
        return quick_sort([e for e in liste[1:] if e <= liste[0]]) + \
            [liste[0]] + quick_sort([e for e in liste[1:] if e > liste[0]])
    

def insertion_sort(liste):
    """
    Algorithme de tri par insertion
    liste (list): liste qui sera triée
    return:
        - liste triée
    Code pris dans le cours 8c du cours PSY-3019 (Alexandru Hanganu)
    """

    for i in range(1, len(liste)):
 
        key = liste[i]
 
        # Move elements of arr[0..i-1], that are greater than key, to one 
        # position ahead of their current position
        j = i-1
        while j >=0 and key < liste[j] :
                liste[j+1] = liste[j]
                j -= 1
        liste[j+1] = key
        
    return liste

# Fonctions liées à sqlite (source: cours 12b par Alexandru Hanganu)
def __connect_db(path, table_name):
    """
    Crée cadre de données vides et retourne la connection
    path: path où créer connection
    table_name: nom du tableau
    connect: activation de la base de données sqlite
    """
    conn = sq.connect(os.path.join(path, '{}.db'.format(table_name)))
    return conn

def __creer_tableau_(path, table_name):
    """
    Crée tableau si n'existe pas 
    table_name: nom du tableau 
    """
    conn = __connect_db(path, table_name)
    conn.execute('''CREATE TABLE IF NOT EXISTS {0} (age, school, sex)'''
                 .format(table_name,))
    conn.commit()
    conn.close() 

def __definir_les_donnees_(path, table_name, donnees):
    """
    Insérer données dans tablea
    table_name: nom du tableau 
    donnees: données à insérer
    """
    conn = __connect_db(path, table_name)
    conn.execute('''INSERT INTO {0} VALUES {1}'''.format(table_name, donnees))
    conn.commit()
    conn.close()