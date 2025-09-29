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
#include <string.h>
#include <unistd.h>

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
float const sortie_OR[4] = {0,1,1,1};
// la table de sortie XOR
float const sortie_XOR[4] = {0,1,1,0};

// pour choisir la porte logique

int choisir_porte(const float **targets, char **porte){
    char choix[10];
    printf("Choisis une porte logique que tu veux (AND/OR/XOR) : ");
    scanf("%9s", choix);
    if (strcasecmp(choix, "AND") == 0) {
        *targets = sortie_AND;
        *porte = "AND";
    } else if (strcasecmp(choix, "OR") == 0) {
        *targets = sortie_OR;
        *porte = "OR";
    } else if (strcasecmp(choix, "XOR") == 0) {
        *targets = sortie_XOR;
        *porte = "XOR";
    } else {
        printf("Porte non reconnue !\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

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

// mutex
pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;

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

void train(RNA *reseau, const float targets[4], int epochs, float eta) {
    double cs[CACHEE];
    for(int e = 0; e < epochs; e++) {
        double mse_epoch = 0.0;
        for(int i = 0; i < 4; i++) {
            double y_pred = propAvant(reseau, inputs[i], cs);
            propArriere(reseau, inputs[i], cs, y_pred, targets[i], eta);
            double diff = targets[i] - y_pred;
            mse_epoch += diff * diff;
        }

        mse_epoch /= 4.0;
        if ((e % 1000) == 0) {
            pthread_mutex_lock(&print_mutex);
            printf("[Train] epoch %5d  MSE = %.6f\n", e, mse_epoch);
            pthread_mutex_unlock(&print_mutex);
            usleep(50000); // 50 ms
        } 
    }
}

void test_network(const RNA *reseau, const float targets[4], const char *porte) {
    double cs[CACHEE];
    double mse = 0.0;
    int correct = 0;
    pthread_mutex_lock(&print_mutex);
    printf("---- Test %s ----\n", porte);
    for (int s = 0; s < 4; ++s) {
        double y = propAvant((RNA *)reseau, inputs[s], cs);
        printf("in=(%.1f, %.1f) -> y=%.6f target=%.1f\n",
               inputs[s][0], inputs[s][1], y, targets[s]);
        double d = targets[s] - y;
        mse += d * d;
        int pred = (y >= 0.5) ? 1 : 0;
        if (pred == (int)targets[s]) correct++;
    }
    mse = 0.5 * (mse / 4.0);
    printf("MSE(avg, 0.5 factor)=%.6f  accuracy=%d/4\n\n", mse, correct);
    pthread_mutex_unlock(&print_mutex);
}

void *thread_func(void *arg) {
    ThreadPL *tp = (ThreadPL *)arg;

    /* initialisation du réseau avec graine locale (rand_r thread-safe) */
    init_reseau(&tp->reseau, &tp->seed);

    pthread_mutex_lock(&print_mutex);
    printf("Thread %d: porte=%s seed=%u eta=%.3f\n",
           tp->thread_id, tp->pl, tp->seed, tp->eta);
    pthread_mutex_unlock(&print_mutex);

    /* entraîner */
    train(&tp->reseau, tp->sortie, EPOCHS, tp->eta);

    /* tester */
    test_network(&tp->reseau, tp->sortie, tp->pl);

    return NULL;
}

// ---

int main() {

    // pour le random
    srand(time(NULL));

    // les threads pour chaque porte
    pthread_t threads[3];
    ThreadPL params[3];

    //code pour faire les portes d'un coup
    const float *targets[3] = {sortie_AND, sortie_OR, sortie_XOR};
    char *porte[3] = {"AND", "OR", "XOR"};

    for(int i = 0; i < 3; i++) {
        params[i].inputs = inputs;
        params[i].sortie = targets[i];
        params[i].pl = porte[i];
        params[i].seed = rand();
        params[i].eta = 0.5;
        params[i].thread_id = i;

        if(pthread_create(&threads[i], NULL, thread_func, &params[i])) {
            fprintf(stderr, "Erreur création thread %d\n", i);
            perror("pthread_create");
            return EXIT_FAILURE;
        }
    }

    for(int th = 0; th < 3; th++){
        pthread_join(threads[th], NULL);
    }

    // code pour demander une seul porte logique
    // pthread_t thread;
    // ThreadPL param;

    // const float *targets;
    // char *porte;

    // // choisir la porte
    // if (choisir_porte(&targets, &porte) != 0) {
    //     return EXIT_FAILURE;
    // }

    // // remplir les paramètres du thread
    // param.inputs = inputs;
    // param.sortie = targets;
    // param.pl = porte;
    // param.seed = rand();
    // param.eta = 0.5;
    // param.thread_id = 0;

    // time_t start = time(NULL);

    // // créer le thread
    // if (pthread_create(&thread, NULL, thread_func, &param)) {
    //     fprintf(stderr, "Erreur création thread\n");
    //     perror("pthread_create");
    //     return EXIT_FAILURE;
    // }

    // // attendre la fin
    // pthread_join(thread, NULL);

    // time_t end = time(NULL);

    // printf("Temps total d'exécution : %ld secondes\n", end - start);

    return EXIT_SUCCESS;
}