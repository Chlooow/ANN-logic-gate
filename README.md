# Projet RNA Porte logique en C

## Objectif du problème

L’objectif est d’implémenter un Réseau de Neurones Artificiel (RNA) capable d’apprendre les fonctions logiques de base (AND, OR, XOR) en langage C, et d’explorer l’utilisation du multithreading pour entraîner plusieurs réseaux en parallèle.

Ce projet sert à appliquer la notion de Deep Learning avec le réseau de neurone sur un petit exemple et voir au dela de ce que fait le perceptron.

![projet-rna](https://hackmd.io/_uploads/ryO99r_hxg.png)


## Qu'est-ce que c'est le Réseau de Neurone Artificiel (RNA)?

Un RNA est un modèle inspiré du cerveau humain, utilisé pour l’apprentissage automatique. Il est capable d’apprendre des relations complexes entre des entrées et des sorties grâce à un processus itératif de propagation avant et de rétropropagation.

## Les données utilisés
Les réseaux ont été entraînés sur les tables de vérité suivantes :

- AND : 0,0 → 0 ; 0,1 → 0 ; 1,0 → 0 ; 1,1 → 1
- OR : 0,0 → 0 ; 0,1 → 1 ; 1,0 → 1 ; 1,1 → 1
- XOR : 0,0 → 0 ; 0,1 → 1 ; 1,0 → 1 ; 1,1 → 0

## Cahier des Charges de l'architecture

![PorteLogique_RNA](https://hackmd.io/_uploads/r1znqSu3gl.png)

- **Couche entrée** : 2 entrées (E1, E2)
- **Couche cachée** : 1 couche cachée de 2/3 neurones activation sigmoïde
- **Couche sortie** : 1 sortie qui donne le résultat de (E1, E2) activation sigmoïde;
- **fonction d'activation** : Sigmoid = 1 / (1 + exp(-x)) sinon RelU
- **La dérivée de la fonction d'activation** : y*(1-y)
- **Thread** sur les 3 portes de sortie
- **Mutex**: pour protéger l’affichage et éviter la confusion des résultats
- **Initialisation des paramètres** : poids et biais aléatoires(matrice de dimension *W_ec[CE, CJ] [2,3], W_cs[CC,CJ][3,1], B_ec[CC][3] B_cs[CS][1]*  )
- **Fonction coût** : Erreur Quadratique Moyenne (MSE)
- **Optimisation** : Descente de gradient stochastique (SGD)

## Methodologie

 ### Définir le réseaux et ses paramètres
 - Création d’une structure RNA pour stocker les poids et biais
`w_ce[CACHEE][ENTREE]` : poids entre la couche d’entrée et la couche cachée.
`b_c[CACHEE]` : biais pour chaque neurone de la couche cachée.
`w_cs[CACHEE]` : poids entre la couche cachée et la sortie.
`b_s `: biais du neurone de sortie.

 - Initialisation des paramètres : Les poids et biais sont initialisés aléatoirement dans l’intervalle [-0,5, 0,5] pour éviter la symétrie et permettre au réseau de commencer l’apprentissage de manière différente pour chaque neurone.


 ### Fonction d'activation Sigmoid -> Sigmoid'
 - Sigmoid = 1 / (1 + exp(-x))
 transforme la somme pondérée des entrées d’un neurone en une valeur comprise entre 0 et 1
 - Sigmoid' = y*(1-y)
 utilisée pour la rétropropagation afin de calculer comment ajuster les poids en fonction de l’erreur.

 ### Propagation avant (Forward)
 - Pour chaque entrée `(E1, E2)` : 
     - on calcul la somme pondérée pour chaque neurone de la couche cachée
     - on applique Sigmoid pour avoir la sortie de chaque neurone dans la couche cachée
     - Finamalement on calcul la sortie finale donc la somme pondérée des sorties cachées + biais sortie puis on l'a met dans Sigmoid
 - et on obtiens y_pred du reseau pour une entrée donnée

 ### Propagation arrière (backward)
 - Calculer l’erreur sur la sortie : différence entre la sortie prédite et la sortie cible `(y_pred - target)`
 - Calculer les deltas Propagation de l’erreur     vers la couche cachée : applique la dérivée de sigmoid à l’erreur pour déterminer comment corriger la sortie.
 - Propagation vers la couche cachée : chaque neurone caché reçoit une part de l’erreur de sortie proportionnelle à son poids vers la sortie.
 - Mettre à jour les poids et biais
```c
w_cs[i] = w_cs[i] - eta * delta_s * cs[i]
w_ce[i][j] = w_ce[i][j] - eta * delta_c[i] * input[j]
b_c[i] = b_c[i] - eta * delta_c[i]
b_s = b_s - eta * delta_s
```

où `eta` est le taux d’apprentissage

 ### Multi-threading 
 - 1 thread = porte logique(AND, OR, XOR) Chaque thread entraîne un réseau pour une porte logique différente (AND, OR, XOR)
 - Chaque thread a ses propres poids et biais
 - Utilisation d’un mutex pour l'affichage (pour synchroniser l’affichage)

------

### Technologie utilisé

- **Environnement de développement :**VSCode
- Terminal powershell avec un wsl(linux Ubuntu)
- **Langage** : C
- **Bibliothèques** : 
``` 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
```
### Comment lancer ?

```shell=
gcc -o rna rna.c -lpthread -lm
./rna
```

### Ressources utilisées

- Cours de Monsieur Touati
- Youtube
- Cours de Francois Landes L3 (IAS)

### Bug et problèmes rencontrés

- Compréhension du fonctionnement d’une couche cachée au début: Passer d’un perceptron simple à un réseau avec une couche cachée a nécessité de bien comprendre comment les neurones cachés calculent leurs sorties et propagent l’erreur lors de la rétropropagation.

- Implémentation en C : Structurer le code pour un RNA multi-couches en C a été plus complexe que pour un perceptron en Python, car il faut gérer manuellement les tableaux de poids, les biais, et la mémoire, sans bénéficier des abstractions et bibliothèques Python.

- Gestion du multithreading : S’assurer que chaque thread utilise ses propres poids et biais tout en protégeant l’affichage via un mutex pour éviter que les résultats se mélangent.

- Ce n'est pas un bug, mais il est vrai qu'a un certain eta, ou epoch, le réseau peux avoir appris très bien ou pas du tout, donc il faut jouer avec les valeurs
pour bien voir certains résultats...

### Conclusion

Les réseaux apprennent correctement les fonctions AND, OR, XOR. Un MSE faible et précision maximale (4/4) pour chaque porte à epoch 10000.
Exécution parallèle efficace grâce au multithreading

Lorsque le réseau obtient une précision parfaite (4/4) pour les portes logiques simples (AND, OR, XOR) avec un MSE très faible. C’est assez logique pour ce type de problème pour plusieurs raisons : 
1. Nature des portes logiques (AND, OR)
Ce sont des problèmes linéairement séparables : on peut séparer les sorties 0 et 1 avec une seule ligne (hyperplan).
2. Taille du dataset (4 exemples)
Le jeu de données est minuscule, donc le réseau peut “mémoriser” parfaitement tous les exemples.
3. Nombre d’époques et taux d’apprentissage
10 000 époques avec un taux de 0.5 permettent au réseau de converger complètement, ajustant tous les poids et biais pour minimiser l’erreur.
4. Rétropropagation efficace
La combinaison de la descente de gradient et de la rétropropagation ajuste rapidement les poids pour atteindre la solution optimale.

La vraie validation est la capacité à résoudre XOR, ce qui montre la puissance de l’architecture multi-couches. L’overfitting n’est pas un problème, car il n’y a pas d’autres données à généraliser.

Un réseau avec une couche cachée peut facilement créer cette frontière grâce aux neurones.

Le projet a permis de comprendre les concepts fondamentaux des réseaux de neurones artificiels et l’intérêt du multithreading pour paralléliser l’entraînement. Les résultats montrent que le réseau est capable d’apprendre les fonctions logiques de base.

![resultat1](https://hackmd.io/_uploads/SJPaqBdnxg.png)
![resultat2](https://hackmd.io/_uploads/ryPaqBu2xl.png)


### Ouverture du projet et améliorations

Extension avec des données concrètes. Peut-être vers un chatbot : Ce projet de RNA pour portes logiques peut être la base pour un chatbot, comme celui que je développe sur le thème de Bambam un artiste de Kpop d'origine Thailandaise, où l’on récupère des tweets, interviews et expressions typiques pour entraîner le réseau. Cela montre comment un RNA peut être appliqué à un contexte réel de traitement de texte et génération de réponses.

**Améliorations du code actuel** :

- Permettre d’afficher l’ID du thread lors de l’entraînement pour mieux suivre quel thread s’occupe de quelle porte.
- Ajouter la fonctionnalité où l’utilisateur peut choisir dynamiquement la porte logique à entraîner via le terminal, ce qui rend l’application plus interactive.
- Optimiser la convergence et la stabilité du réseau en adaptant dynamiquement le learning rate ou en essayant d’autres fonctions d’activation (ReLU, tanh).
- Sauvegarder et charger les poids/biais après l’entraînement pour éviter de réentraîner à chaque exécution.
- Tester avec des valeurs différentes
- tester avec la porte `NOR`, `NOT` et des portes dîtes de combinaison `(A AND (B OR C))` ...etc

**Perspectives futures** :

- Étendre le réseau à plusieurs couches cachées pour traiter des problèmes plus complexes.
- Développer une interface graphique simple pour visualiser les sorties du réseau et la progression de l’apprentissage en temps réel.

### Crédits et droits

Projet et rédaction créé par Makoundou Chloé
dans le cadre du Master 1 Cursus IBD Informatique et Big Data pour le cours Programmation Concurrente avec monsieur Touati.

Email : Chloe.makpro@gmail.com
Lien du Github : https://github.com/Chlooow/ANN-logic-gate
Lien du compte rendu sur Hackmd : https://hackmd.io/@CholorSplash/B1ROo673el

### Feedback du professeur (Le client) (23/10/2025)

- Attention il y avait une erreur dans la fonction du MSE j'ai inversé output et target => une précision catastrophique (ex : 0.000003...etc).

- Essayer de faire des Batches ?
