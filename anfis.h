#ifndef ANFIS_H
#define ANFIS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Constantes do modelo
#define NUM_RULES 5
#define MAX_EPOCHS 200
#define ALPHA 0.006
#define NUM_FEATURES 5
#define MAX_LINE_LENGTH 1024
#define MAX_SAMPLES 1816

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

// Caminho dos Arquivos .csv

#define PATH_DATA_CSV "arquivos_csv/data.csv"
#define PATH_TRAINING_CSV "arquivos_csv/training.csv"
#define PATH_VALIDATION_CSV "arquivos_csv/validation.csv"
#define PATH_Q_CSV "arquivos_csv/q.csv"
#define PATH_P_CSV "arquivos_csv/p.csv"
#define PATH_S_CSV "arquivos_csv/s.csv"
#define PATH_C_CSV "arquivos_csv/c.csv"
#define PATH_TRAINING_RESULTS_CSV "arquivos_csv/training_results.csv"

// Estrutura para armazenar os dados
typedef struct
{
    float speed;
    float acc_norm;
    float engine_speed;
    float throttle_position;
    float delta_acc_lat;
    int cluster_id;
} DataPoint;

// Estrutura para os parâmetros do ANFIS
typedef struct
{
    float c[NUM_FEATURES][NUM_RULES]; // Centros das funções de pertinência
    float s[NUM_FEATURES][NUM_RULES]; // Larguras das funções de pertinência
    float p[NUM_FEATURES][NUM_RULES]; // Coeficientes lineares das consequências
    float q[NUM_RULES];               // Termos constantes das consequências
} ANFISParams;

// Estrutura para os dados normalizados
typedef struct
{
    float inputs[MAX_SAMPLES][NUM_FEATURES];
    int outputs[MAX_SAMPLES];
    int num_samples;
} Dataset;

// Protótipos das funções
int load_data(const char *filename, DataPoint *data);
void normalize_data(DataPoint *data, int num_samples, float normalized_inputs[][NUM_FEATURES]);
void randomize_matrix(float inputs[][NUM_FEATURES], int *outputs, int num_samples, Dataset *train_data, Dataset *val_data);
void write_csv_train_val(const char *train_file, const char *val_file, Dataset *train_data, Dataset *val_data);
void split_data(float inputs[][NUM_FEATURES], int *outputs, int num_samples,
                Dataset *train_data, Dataset *val_data, float train_ratio);
void initialize_params(ANFISParams *params, float inputs[][NUM_FEATURES], int num_samples);
float random_double(float min, float max);
float calys(float *x, ANFISParams *params, float *w, float *y, float *b_out);
void train_anfis(Dataset *train_data, ANFISParams *params, float *mse_history);
void evaluate_anfis(Dataset *val_data, ANFISParams *params, float *accuracy, float *error_percent);
void save_params(ANFISParams *params);
void save_results(float *mse_history, float accuracy, float error_percent);

#endif // ANFIS_H