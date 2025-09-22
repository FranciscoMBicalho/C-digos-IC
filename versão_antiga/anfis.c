#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CSV_PATH "teste1.csv"
#define MAX_LINES 10
#define MAX_COLUMNS 6

void read_csv(const char* filename, int matrix[MAX_LINES][MAX_COLUMNS]) {

    int col_count = 0;
    int line_count = 0;
    char line[256];
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Erro ao abrir o arquivo %s\n", filename);
        return;
    }

    // Lê cada linha do arquivo
    while (fgets(line, sizeof(line), file) && line_count < MAX_LINES) {
        col_count = 0;  // Reseta a contagem de colunas a cada nova linha

        // Usa strtok para dividir a linha com base nos espaços ou vírgulas
        char *token = strtok(line, ",");  // Pode ser espaço ou vírgula

        // Enquanto houver tokens, converte e armazena os valores
        while (token != NULL && col_count < MAX_COLUMNS) {
            matrix[line_count][col_count] = atoi(token);  // Converte o token para inteiro e armazena
            col_count++;  // Avança para a próxima coluna
            token = strtok(NULL, ", \n");  // Continua a busca de tokens
        }

        line_count++;  // Avança para a próxima linha
    }
    fclose(file);
}

void write_vectors(int vec1[MAX_LINES], int vec2[MAX_LINES], int vec3[MAX_LINES], int vec4[MAX_LINES], int vec5[MAX_LINES], int vec6[MAX_LINES], int matrix_in[MAX_LINES][MAX_COLUMNS])
{
    for(int i=0; i<MAX_LINES; i++) {
    vec1[i] = matrix_in[i][0]; // Retira os valores de data_speed da matriz principal e coloca no vetor
    vec2[i] = matrix_in[i][1]; // Retira os valores de data_dist da matriz principal e coloca no vetor
    vec3[i] = matrix_in[i][2]; // Retira os valores de data_vel da matriz principal e coloca no vetor
    vec4[i] = matrix_in[i][3]; // Retira os valores de data_break da matriz principal e coloca no vetor
    vec5[i] = matrix_in[i][4]; // Retira os valores de data_steer da matriz principal e coloca no vetor
    vec6[i] = matrix_in[i][5]; // Retira os valores de data_cluster da matriz principal e coloca no vetor
    }
}

void show_vectors(int vec1[MAX_LINES], int vec2[MAX_LINES], int vec3[MAX_LINES], int vec4[MAX_LINES], int vec5[MAX_LINES], int vec6[MAX_LINES], int matrix_in[MAX_LINES][MAX_COLUMNS])
{
    for(int i=0; i<MAX_LINES; i++)
    {
        printf("Speed: %d, Dist: %d, Vel: %d, Brake: %d, Steer: %d, Cluster: %d \n", vec1[i], vec2[i], vec3[i], vec4[i], vec5[i],vec6[i]);
    }
}

void normalize_inputs(float inputs[5], float normalized_inputs[5], int max_inputs[5], int min_inputs[5])
{
    for (int i = 0; i < 5; i++) {
        normalized_inputs[i] = ((inputs[i] - min_inputs[i]) / (max_inputs[i] - min_inputs[i]));
    }
}

void show_normalized_inputs(float normalized_inputs[5])
{
    for (int i = 0; i < 5; i++) {
        printf("\nNormalized input %d: %f\n", i, normalized_inputs[i]);
    }
}

int main()
{
    int matrix_csv[MAX_LINES][MAX_COLUMNS];
    int data_speed[MAX_LINES], data_dist[MAX_LINES], data_vel[MAX_LINES], data_brake[MAX_LINES], data_steer[MAX_LINES], data_cluster[MAX_LINES];

    int max_speed = 120; // não é necessário declarar esses valores,estou colocando eles so para ficar parecido com o código do matlab
    int min_speed = 0;  // para melhorar o desempenho,talvez seja melhor colocar os vaores diretos nos vetores max e min_inputs
    int max_acc_norm = 5;
    int min_acc_norm = 0;
    int max_engine_speed = 10000;
    int min_engine_speed = 0;
    int max_throttle_position = 100;
    int min_throttle_position = 0;
    int max_delta_acc_lat = 3;
    int min_delta_acc_lat = 0;
    float normalized_inputs[5], inputs[5] = {131,155,72,414,125};

    int max_inputs[5] = {max_speed, max_acc_norm, max_engine_speed, max_throttle_position, max_delta_acc_lat};
    int min_inputs[5] = {min_speed, min_acc_norm, min_engine_speed, min_throttle_position, min_delta_acc_lat};


    read_csv(CSV_PATH, matrix_csv);
    write_vectors(data_speed, data_dist, data_vel, data_brake, data_steer, data_cluster, matrix_csv);
    show_vectors(data_speed, data_dist, data_vel, data_brake, data_steer, data_cluster, matrix_csv);
    normalize_inputs(inputs, normalized_inputs, max_inputs, min_inputs);
    show_normalized_inputs(normalized_inputs);

    /* Funções matlab que ainda precisam se implementadas em c:
        cvpartition()
        training()
        test()
        x_train()
        y_train()
    */

    return 0;
}