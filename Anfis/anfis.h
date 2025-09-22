#ifndef ANFIS_H
#define ANFIS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Constantes do modelo
#define NUM_RULES 5
#define MAX_EPOCHS 100
#define ALPHA 0.001
#define NUM_FEATURES 5
#define MAX_LINE_LENGTH 1024
#define MAX_SAMPLES 10000

// Limites para normalização
#define MAX_SPEED 120.0
#define MIN_SPEED 0.0
#define MAX_ACC_NORM 5.0
#define MIN_ACC_NORM 0.0
#define MAX_ENGINE_SPEED 10000.0
#define MIN_ENGINE_SPEED 0.0
#define MAX_THROTTLE_POSITION 100.0
#define MIN_THROTTLE_POSITION 0.0
#define MAX_DELTA_ACC_LAT 3.0
#define MIN_DELTA_ACC_LAT 0.0

// Estrutura para armazenar os dados
typedef struct {
    double speed;
    double acc_norm;
    double engine_speed;
    double throttle_position;
    double delta_acc_lat;
    int cluster_id;
} DataPoint;

// Estrutura para os parâmetros do ANFIS
typedef struct {
    double c[NUM_FEATURES][NUM_RULES];  // Centros das funções de pertinência
    double s[NUM_FEATURES][NUM_RULES];  // Larguras das funções de pertinência
    double p[NUM_FEATURES][NUM_RULES];  // Coeficientes lineares das consequências
    double q[NUM_RULES];                // Termos constantes das consequências
} ANFISParams;

// Estrutura para os dados normalizados
typedef struct {
    double inputs[MAX_SAMPLES][NUM_FEATURES];
    int outputs[MAX_SAMPLES];
    int num_samples;
} Dataset;

// Protótipos das funções
int load_data(const char* filename, DataPoint* data);
void normalize_data(DataPoint* data, int num_samples, double normalized_inputs[][NUM_FEATURES]);
void randomize_matrix();
void split_data(double inputs[][NUM_FEATURES], int* outputs, int num_samples, 
                Dataset* train_data, Dataset* val_data, double train_ratio);
void initialize_params(ANFISParams* params, double inputs[][NUM_FEATURES], int num_samples);
double random_double(double min, double max);
double calys(double* x, ANFISParams* params, double* w, double* y, double* b_out);
void train_anfis(Dataset* train_data, ANFISParams* params, double* mse_history);
void evaluate_anfis(Dataset* val_data, ANFISParams* params, double* accuracy, double* error_percent);
void save_params(ANFISParams* params);
void save_results(double* mse_history, double accuracy, double error_percent);

#endif // ANFIS_H