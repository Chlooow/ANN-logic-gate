# Projet RNA Porte logique en C

## Objectif du problème

L’objectif est d’implémenter un Réseau de Neurones Artificiel (RNA) capable d’apprendre les fonctions logiques de base (AND, OR, XOR) en langage C, et d’explorer l’utilisation du multithreading pour entraîner plusieurs réseaux en parallèle.

Ce projet sert à appliquer la notion de Deep Learning avec le réseau de neurone sur un petit exemple et voir au dela de ce que fait le perceptron.

![projet-rna](https://hackmd.io/_uploads/ryO99r_hxg.png)


## Qu'est-ce que c'est le Réseau de Neurone Artificiel (RNA)?

Un RNA est un modèle inspiré du cerveau humain, utilisé pour l’apprentissage automatique. Il est capable d’apprendre des relations complexes entre des entrées et des sorties grâce à un processus itératif de propagation avant et de rétropropagation.

## Les données
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

## Comment faire ?

 ### Définir le réseaux et ses paramètres
 - Création d’une structure RNA pour stocker les   poids et biais
 - Initialisation aléatoire des poids et biais
 ### Fonction d'activation Sigmoid -> Sigmoid'
 - Sigmoid = 1 / (1 + exp(-x))
 - Sigmoid' = y*(1-y)

 ### Propagation avant (Forward)
 - Calcul des sorties de la couche cachée
 - Calcul de la sortie finale du réseau

 ### Propagation arrière (backward)
 - Calculer l’erreur sur la sortie 
 - Calculer les deltas Propagation de l’erreur     vers la couche cachée
 - Mettre à jour les poids et biais

 ### Multi-threading 
 - 1 thread = porte logique(AND, OR, XOR) Chaque   thread entraîne un réseau pour une porte             logique différente (AND, OR, XOR)
 - Chaque thread a ses propres poids et biais
 - Utilisation d’un mutex pour l'affichage (pour   synchroniser l’affichage)

------

### Technologie utilisé

- **Environnement de développement :**VSCode
- Terminal (linux Ubuntu)
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

### Ressources utilisées

- Cours de Monsieur Touati
- Youtube
- Cours de Francois Landes L3 (IAS)

### Bug et problèmes rencontrés

- Compréhension du fonctionnement d’une couche cachée : Passer d’un perceptron simple à un réseau avec une couche cachée a nécessité de bien comprendre comment les neurones cachés calculent leurs sorties et propagent l’erreur lors de la rétropropagation.

- Implémentation en C : Structurer le code pour un RNA multi-couches en C a été plus complexe que pour un perceptron en Python, car il faut gérer manuellement les tableaux de poids, les biais, et la mémoire, sans bénéficier des abstractions et bibliothèques Python.

- Gestion du multithreading : S’assurer que chaque thread utilise ses propres poids et biais tout en protégeant l’affichage via un mutex pour éviter que les résultats se mélangent.

### Conclusion

Les réseaux apprennent correctement les fonctions AND, OR, XOR. Un MSE faible et précision maximale (4/4) pour chaque porte.
Exécution parallèle efficace grâce au multithreading

Le réseau obtient une précision parfaite (4/4) pour les portes logiques simples (AND, OR, XOR) avec un MSE très faible. C’est assez logique pour ce type de problème pour plusieurs raisons : 
1. Nature des portes logiques (AND, OR)
Ce sont des problèmes linéairement séparables : on peut séparer les sorties 0 et 1 avec une seule ligne (hyperplan).
2. Taille du dataset (4 exemples)
Le jeu de données est minuscule, donc le réseau peut “mémoriser” parfaitement tous les exemples.
3. Nombre d’époques et taux d’apprentissage
10 000 époques avec un taux de 0.5 permettent au réseau de converger complètement, ajustant tous les poids et biais pour minimiser l’erreur.
4. Rétropropagation efficace
La combinaison de la descente de gradient et de la rétropropagation ajuste rapidement les poids pour atteindre la solution optimale.

La vraie validation est la capacité à résoudre XOR, ce qui montre la puissance de l’architecture multi-couches. Ici, l’overfitting n’est pas un problème, car il n’y a pas d’autres données à généraliser.

Un réseau avec une couche cachée peut facilement créer cette frontière grâce aux neurones.

Le projet a permis de comprendre les concepts fondamentaux des réseaux de neurones artificiels et l’intérêt du multithreading pour paralléliser l’entraînement. Les résultats montrent que le réseau est capable d’apprendre les fonctions logiques de base.

![resultat1](https://hackmd.io/_uploads/SJPaqBdnxg.png)
![resultat2](https://hackmd.io/_uploads/ryPaqBu2xl.png)


### Ouverture du projet et améliorations

Extension vers un chatbot : Ce projet de RNA pour portes logiques peut être la base pour un chatbot, comme celui que je développe sur le thème de Bambam, où l’on récupère des tweets, interviews et expressions typiques pour entraîner le réseau. Cela montre comment un RNA peut être appliqué à un contexte réel de traitement de texte et génération de réponses.

**Améliorations du code actuel** :

- Permettre d’afficher l’ID du thread lors de l’entraînement pour mieux suivre quel thread s’occupe de quelle porte.
- Ajouter la fonctionnalité où l’utilisateur peut choisir dynamiquement la porte logique à entraîner via le terminal, ce qui rend l’application plus interactive.
- Optimiser la convergence et la stabilité du réseau en adaptant dynamiquement le learning rate ou en essayant d’autres fonctions d’activation (ReLU, tanh).
- Sauvegarder et charger les poids/biais après l’entraînement pour éviter de réentraîner à chaque exécution.

**Perspectives futures** :

- Étendre le réseau à plusieurs couches cachées pour traiter des problèmes plus complexes.
- Développer une interface graphique simple pour visualiser les sorties du réseau et la progression de l’apprentissage en temps réel.

### Crédits et droits

Projet créé par Makoundou Chloé (Cholor)
dans le cadre du Master 1 Cursus IBD Informatique et Big Data pour le cours Programmation Concurrente avec monsieur Touati.

Email : Chloe.makpro@gmail.com
