/**
 * Réalisation d'une architecture orientée IA basé sur le 
 * RNA. Mettre en oeuvre des threads pour executer plusiers RNA
 * et le faire sur un application sur un process : 2 entrées/ 1 sortie
 * sur les portes logiques (AND)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>

// ---  INITIALISATION DU RN  ---

#define ENTREE 2
#define CACHEE 3 // nombre de neurones
#define SORTIE 1
#define EPOCHS 10000

// base de la structure du réseau neuronal
typedef struct {
    float w_ce[CACHEE][ENTREE]; // poids d'entrée cachée
    float w_cs[CACHEE]; // poids caché de sortie
    float b_c[CACHEE]; // biais caché
    float b_s; // biais de sortie 
    // w_cc n'est pas ici car on a un réseau peu profond
} RNA;

// ---  INITIALISATION DU DATA  ---

// La table d'entrées
float inputs[4][ENTREE] = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

// la table de sortie AND
static const float sortie_AND[4] = {0,0,0,1};
// la table de sortie OR
float const targets_OR[4] = {0,1,1,1};
// la table de sortie XOR
float const targets_XOR[4] = {0,1,1,0};


// ---  INITIALISATION DES THREADS  ---

typedef struct {
    RNA reseau;
    float (*inputs)[ENTREE]; // nos datas E1 et E2
    const float *sortie; // target la réponse après l'apprentissage
    char *pl; // pour porte logique
    unsigned int seed;// pour le random
    float eta; // pas d'apprentissage lr
    // int epochs; 
    int thread_id; // les threads 
} ThreadPL;

// ------------------------------------------

// --- FONCTION D'ACTIVATION  ---

// TODO : La fonction a dérivée
static inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Fonction D'activation "Comment allumer un neurone"
// double fonctionActivation(double x) {
//     sigmoid(x);
// }

static inline double deriveSigmoid(double y) {
    return y * (1.0 - y);
}

// --- ETAPE DE PROPAGATION AVANT (FORWARD)

double initialisationDesPoids(unsigned int *seed) {
    double r = (double) rand_r(seed) / (double) RAND_MAX;
    return (r*1.0) - 0.5;
}

void init_reseau(RNA *reseau, unsigned int *seed) {
    for (int i = 0; i < CACHEE; i++) {
        for (int j = 0; j < ENTREE; j++) {
            reseau->w_ce[i][j] = initialisationDesPoids(seed);
        }
        reseau->b_c[i] = initialisationDesPoids(seed);
        reseau->w_cs[i] = initialisationDesPoids(seed);
    }
    reseau->b_s = initialisationDesPoids(seed);
}

double propAvant(RNA *reseau, float entree[ENTREE], double cs[CACHEE]) {
    for (int i = 0; i < CACHEE; i++) {
        double somme = 0.0;
        for (int j = 0; j < ENTREE; j++) {
            somme += reseau->w_ce[i][j] * entree[j];
        }
        somme += reseau->b_c[i];
        cs[i] = sigmoid(somme);
    }

    double somme_prime = 0.0;
    for (int i = 0; i < CACHEE; i++) {
        somme_prime += reseau->w_cs[i] * cs[i];
    }
    somme_prime += reseau->b_s;
    return sigmoid(somme_prime);
}

// --- ETAPE DE PROPAGATION ARRIERE (BACKWARD)

// Fonction cout (Loss): il faut qu'on sorte le minimale (erreur décroissante tout
// au long de l'apprentissage)
// fonction quadratique

static inline double MSE(double target, double output) {
    return 0.5 * pow((target-output), 2);
}

static inline double GradientMSE(double target, double y_pred) {
    return (y_pred - target);
}

void propArriere(RNA *reseau, float entree[ENTREE], double cs[CACHEE], 
    double s_pred, float target, float eta) {

    double delta_s = GradientMSE(target, s_pred) * deriveSigmoid(s_pred);
    double delta_c[CACHEE];
    for (int i = 0; i < CACHEE; i++) {
        delta_c[i] = delta_s * reseau->w_cs[i] * deriveSigmoid(cs[i]);
    }

    for (int i = 0; i < CACHEE; i++) {
        reseau->w_cs[i] -= eta* delta_s * cs[i];

        for (int j = 0; j < ENTREE; j++) {
            reseau->w_ce[i][j] -= eta * delta_c[i] * entree[j];
        }
        reseau->b_c[i] -= eta * delta_c[i];
    }
    reseau->b_s -= eta * delta_s;
}

// ---

int main() {

srand(time(NULL));

return EXIT_SUCCESS;
}