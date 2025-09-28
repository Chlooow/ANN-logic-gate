# Projet RNA Porte logique en C

## Objectif du problème

On cherche à implémenter un "Artificial Neuronal Network" (Réseau de Neurone Artificiel) avec pour exemple les porte logiques (XOR, OR, AND, NOR...etc) en langage C.

## Qu'est-ce que c'est le Réseau de Neurone Artificiel (RNA)?

Un réseau de neurone artificiel est une technique venant du deep learning pour l'apprentissage de la machine 

## Les données

- AND 
- OR
- XOR
- NOR
- ...etc.

## Cahier des Charges de l'architecture

- Couche entrée : 2 entrées (E1, E2)
- Couche cachée : 1 couche cachée de 2/3 neurones
- Couche sortie : 1 sortie qui donne le résulat de (E1, E2);
- fonction d'activation : Sigmoid = 1 / (1 + exp(-x)) sinon RelU
- La dérivée de la fonction d'activation
- Initialiser alétoirement les parametres en random (matrice de dimension W_ec[CE, CJ] [2,3], W_cs[CC,CJ][3,1], B_ec[CC][3] B_cs[CS][1]  )
- Thread sur les 3 portes de sortie
- poid et biais init randomly
- fonction coût

## Comment faire ?

### Définir le réseaux et ses paramètres

### Fonction d'activation Sigmoid -> Sigmoid'

### Propagation avant (Forward)
- Pour chaque entrée, calculer :
Les sorties de la couche cachée
La sortie finale

### Propagation arrière (backward)
- Calculer l’erreur sur la sortie 
- Calculer les deltas sur chaque neurone caché
- Mettre à jour les poids et biais
- Mettre à jour les poids et biais

### Multi-threading 
- 1 thread = porte logique(AND, OR, XOR)
- chaque thread a ses propres poids et biais
- Mutex pour l'affichage pour l'affichage pour ne pas mélanger les résultats

### Pourquoi ce projet ?
Ce projet sert à appliquer la notion de Deep Learning avec le réseaux de neurone sur un petit exemple en utilisant également les threads

### Pourquoi utiliser des Threads ?

Car nous voulons déléguer les taches lorsque nous faisons un réseau de neurone. On veut pouvoir laisser un thread faire un réseau 1, un autre thread faire un réseau 2, ...etc.

------

### Méthodologie de travail

### Ce que j'ai fais


### Technologie utilisé

- VSCode
- Terminal (linux Ubuntu)
- Langage : C
- Bibliotheques : 

### Ressources utilisés

- Cours de Monsieur Touati

### Bug et problèmes rencontrés

### Conclusion 

### Ouverture du projet et améliorations

### Crédits et droit

Projet crée par Makoundou Chloé (Cholor)
dans le cadre du Master 1 Cursus IBD Informatique et Big Data pour le cours Programmation Concurrente avec monsieur Touati.

Email : Chloe.makpro@gmail.com
