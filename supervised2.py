from sklearn.neighbors import KNeighborsClassifier  #import de KNeighborsClassifier
from sklearn.model_selection import GridSearchCV    #import de GridSearchCV


#Initialisation du modèle
knc = KNeighborsClassifier()

#Parametrage de grid search
grid_params = {
    'n_neighbors': [3, 5, 11, 19],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

#Initialisation de GridSearch 
gs = GridSearchCV(
        knc,
        grid_params,
        cv = 3,
        n_jobs = -1
    )

#Fonction pour l'échantillonnage, l'entrainement et la prédiction. Retourne les resultats de l'entrainement
def my_knc(gs):
    # divisez votre dataset en utilise la stratégie en 5-fold
    folds = KFold(n_splits=5)
    folds.get_n_splits(X_C)

    results = [] # stocker les résultats intermédiaires pour la validation
    gs_results = [] # stocker les résultats de l'entrainement
    i=0
    for train_index, test_index in folds.split(X_C):
    
        # recuperer les données utilisées d'entrainement
        X_train = X_C.iloc[train_index]
        # récupérer les labels de références pour l'entrainement
        y_train = y_C.iloc[train_index]
    
        # récupérer les données utilisées pour le test
        X_test = X_C.iloc[test_index]
        # récupérer les labels pour la validation
        y_test = y_C.iloc[test_index]
    
        # entrainer le model (fit)
        gs_results = gs.fit(X_train, y_train)
    
        # collecter les résultats du model sur le set d'échantillons de test
        p_test = gs_results.predict(X_test)
    
        # Effectuer une evaluation locale pour cette itération
        print("Accuracy score for fold {:0}: {:1.4%}".format(i, accuracy_score(y_test, p_test)))
        i+=1
        # concatener les résultats dans une liste
        results.append(p_test)

    # merger toutes les predictions dans un seul tableau
    predictions = np.concatenate(results, axis=0)

    # effectuer une evaluation globale pour le dataset
    print()
    print("Global accuracy score: {:0.4%}".format(accuracy_score(y_C, predictions)))
    
    return gs_results

# Appel de la fonction my_knc() avec un GridSearchCV comme paramètre
gs_results = my_knc(gs)

print()
print("Meilleur score: ", gs_results.best_score_)
print("Meilleurs paramètres: ", gs_results.best_params_)
#print("Meilleur estimateur: ", gs_results.best_estimator_)

#Initialisation d'un nouveau modèle avec le paramètre n_neighbors=19
knc2 = KNeighborsClassifier(n_neighbors=19)

# Appel de la fonction my_knc() notre nouveau modèle comme paramètre
print()
print()
print("Resultats avec la meilleure valeur pour le parametre")
print()
my_knc(knc2)