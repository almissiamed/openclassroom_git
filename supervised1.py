from sklearn import tree                      #import de la bibliothèque tree de sklearn

#Initialisation du modèle
clf = tree.DecisionTreeClassifier()
# divisez votre dataset en utilise la stratégie en 5-fold (cf. derniere page du chapitre)
folds = KFold(n_splits=5)
folds.get_n_splits(X_C)

# stocker les résultats intermédiaires pour la validation
results = []
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
    clf.fit(X_train, y_train)
    
    # collecter les résultats du model sur le set d'échantillons de test
    p_test = clf.predict(X_test)
    
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