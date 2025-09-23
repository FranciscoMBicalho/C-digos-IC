#include "anfis.h"

// Função para gerar número aleatório entre min e max
double random_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

// Função para carregar dados do CSV
int load_data(const char* filename, DataPoint* data) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Erro ao abrir arquivo: %s\n", filename);
        return -1;
    }
    
    char line[MAX_LINE_LENGTH];
    int count = 0;
    
    // Pular cabeçalho
    if (fgets(line, sizeof(line), file)) {
        // Cabeçalho lido
    }
    
    // Ler dados
    while (fgets(line, sizeof(line), file) && count < MAX_SAMPLES) {
        char* token = strtok(line, ",");
        if (!token) continue;
        
        // Assumindo ordem: speed, acc_norm, engine_speed, throttle_position, delta_acc_lat, cluster_id
        data[count].speed = atof(token);
        
        token = strtok(NULL, ",");
        if (!token) continue;
        data[count].acc_norm = atof(token);
        
        token = strtok(NULL, ",");
        if (!token) continue;
        data[count].engine_speed = atof(token);
        
        token = strtok(NULL, ",");
        if (!token) continue;
        data[count].throttle_position = atof(token);
        
        token = strtok(NULL, ",");
        if (!token) continue;
        data[count].delta_acc_lat = atof(token);
        
        token = strtok(NULL, ",");
        if (!token) continue;
        data[count].cluster_id = atoi(token);
        
        count++;
    }
    
    fclose(file);
    return count;
}

// Função para normalizar os dados
void normalize_data(DataPoint* data, int num_samples, double normalized_inputs[][NUM_FEATURES]) {
    double max_inputs[NUM_FEATURES] = {MAX_SPEED, MAX_ACC_NORM, MAX_ENGINE_SPEED, 
                                       MAX_THROTTLE_POSITION, MAX_DELTA_ACC_LAT};
    double min_inputs[NUM_FEATURES] = {MIN_SPEED, MIN_ACC_NORM, MIN_ENGINE_SPEED, 
                                       MIN_THROTTLE_POSITION, MIN_DELTA_ACC_LAT};
    
    for (int i = 0; i < num_samples; i++) {
        normalized_inputs[i][0] = (data[i].speed - min_inputs[0]) / (max_inputs[0] - min_inputs[0]);
        normalized_inputs[i][1] = (data[i].acc_norm - min_inputs[1]) / (max_inputs[1] - min_inputs[1]);
        normalized_inputs[i][2] = (data[i].engine_speed - min_inputs[2]) / (max_inputs[2] - min_inputs[2]);
        normalized_inputs[i][3] = (data[i].throttle_position - min_inputs[3]) / (max_inputs[3] - min_inputs[3]);
        normalized_inputs[i][4] = (data[i].delta_acc_lat - min_inputs[4]) / (max_inputs[4] - min_inputs[4]);
    }
}

// Função para embaralhar um array de inteiros (algoritmo de Fisher-Yates)
void shuffle_int_array(int *array, int n) {
    for (int i = n - 1; i > 0; i--) { // Vai da última posição até a segunda, trocando o elemento atual por um aleatório antes dele
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// Função para dividir dados de forma estratificada (cvpartition do MATLAB)
void split_data_stratified(double inputs[][NUM_FEATURES], int* outputs, int num_samples, 
                Dataset* train_data, Dataset* val_data, double train_ratio) {

    // Descobrir as classes presentes e contar exemplos de cada
        int class_counts[3] = {0}; // guarda quantos exemplos há de cada classe, suporta até 3 classes
        int class_labels[3]; // guarda os rótulos das classes
        int num_classes = 0;
    for (int i = 0; i < num_samples; i++) {
        int found = 0;
        for (int c = 0; c < num_classes; c++) {
            if (outputs[i] == class_labels[c]) {
                class_counts[c]++;
                found = 1;
                break;
            }
        }
        if (!found) {
            class_labels[num_classes] = outputs[i];
            class_counts[num_classes] = 1;
            num_classes++;
        }
    }

    // Para cada classe, armazene os índices dos exemplos
        int *class_indices[3];
        int class_pos[3] = {0};
    for (int c = 0; c < num_classes; c++) {
        class_indices[c] = (int*)malloc(class_counts[c] * sizeof(int));
    }
    for (int i = 0; i < num_samples; i++) {
        for (int c = 0; c < num_classes; c++) {
            if (outputs[i] == class_labels[c]) {
                class_indices[c][class_pos[c]++] = i;
                break;
            }
        }
    }

    // Embaralhar índices de cada classe
    // Garante que a seleção de treino/validação seja aleatória dentro de cada classe
    for (int c = 0; c < num_classes; c++) {
        shuffle_int_array(class_indices[c], class_counts[c]);
    }

    // Montar arrays finais de índices de treino e validação
    int *train_indices = (int*)malloc(num_samples * sizeof(int));
    int *val_indices = (int*)malloc(num_samples * sizeof(int));
    int train_count = 0, val_count = 0;
    for (int c = 0; c < num_classes; c++) {
        int n_train = (int)(class_counts[c] * train_ratio + 0.5); // arredonda para cima
        if (n_train >= class_counts[c]) n_train = class_counts[c] - 1; // garante pelo menos 1 para validação
        for (int i = 0; i < class_counts[c]; i++) {
            if (i < n_train) {
                train_indices[train_count++] = class_indices[c][i];
            } else {
                val_indices[val_count++] = class_indices[c][i];
            }
        }
    }

    // // Embaralhar os arrays finais (opcional, mas deixa mais aleatório)
    // shuffle_int_array(train_indices, train_count);
    // shuffle_int_array(val_indices, val_count);

    // Preencher os datasets
    // Garante que cada conjunto tenha exemplos de todas as classes, na mesma proporção
    train_data->num_samples = train_count;
    for (int i = 0; i < train_count; i++) {
        int idx = train_indices[i];
        for (int j = 0; j < NUM_FEATURES; j++) {
            train_data->inputs[i][j] = inputs[idx][j];
        }
        train_data->outputs[i] = outputs[idx];
    }
    val_data->num_samples = val_count;
    for (int i = 0; i < val_count; i++) {
        int idx = val_indices[i];
        for (int j = 0; j < NUM_FEATURES; j++) {
            val_data->inputs[i][j] = inputs[idx][j];
        }
        val_data->outputs[i] = outputs[idx];
    }

    // Liberar memória
    for (int c = 0; c < num_classes; c++) {
        free(class_indices[c]);
    }
    free(train_indices);
    free(val_indices);
}

// Função para inicializar parâmetros do ANFIS
void initialize_params(ANFISParams* params, double inputs[][NUM_FEATURES], int num_samples) {
    // Encontrar min e max dos dados de treino
    double xmin[NUM_FEATURES], xmax[NUM_FEATURES];
    
    for (int i = 0; i < NUM_FEATURES; i++) {
        xmin[i] = inputs[0][i];
        xmax[i] = inputs[0][i];
        
        for (int k = 1; k < num_samples; k++) {
            if (inputs[k][i] < xmin[i]) xmin[i] = inputs[k][i];
            if (inputs[k][i] > xmax[i]) xmax[i] = inputs[k][i];
        }
    }
    
    // Inicializar parâmetros aleatoriamente
    for (int j = 0; j < NUM_RULES; j++) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            params->c[i][j] = random_double(xmin[i], xmax[i]);
            params->s[i][j] = random_double(0.0, 1.0);
            params->p[i][j] = random_double(0.0, 1.0);
        }
        params->q[j] = random_double(0.0, 1.0);
    }
}

// Função principal do ANFIS (equivalente à função calys do MATLAB)
double calys(double* x, ANFISParams* params, double* w, double* y, double* b_out) {
    double a = 0.0, b = 0.0;
    
    // Calcular saída e peso para cada regra
    for (int j = 0; j < NUM_RULES; j++) {
        y[j] = params->q[j];
        w[j] = 1.0;
        
        for (int i = 0; i < NUM_FEATURES; i++) {
            y[j] += params->p[i][j] * x[i];
            double diff = x[i] - params->c[i][j];
            w[j] *= exp(-0.5 * (diff * diff) / (params->s[i][j] * params->s[i][j]));
        }
        
        a += w[j] * y[j];
        b += w[j];
    }
    
    *b_out = b;
    return a / (b + EPS);  // Igual ao MATLAB: ys = a / (b + eps)
}

// Função de treinamento do ANFIS
void train_anfis(Dataset* train_data, ANFISParams* params, double* mse_history) {
    double w[NUM_RULES], y[NUM_RULES];
    
    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double total_error = 0.0;
        
        for (int k = 0; k < train_data->num_samples; k++) {
            double* x = train_data->inputs[k];
            int target = train_data->outputs[k];
            double b;
            
            double ys = calys(x, params, w, y, &b);
            double error = ys - target;
            total_error += error * error;
            
            // Backpropagation - atualizar parâmetros
            for (int j = 0; j < NUM_RULES; j++) {
                double dys_dw = (y[j] - ys) / (b + EPS);
                double dys_dy = w[j] / (b + EPS);
                
                for (int i = 0; i < NUM_FEATURES; i++) {
                    double diff = x[i] - params->c[i][j];
                    double s_sq = params->s[i][j] * params->s[i][j];
                    double s_cu = s_sq * params->s[i][j];
                    
                    double dw_dc = w[j] * diff / s_sq;
                    double dw_ds = w[j] * diff * diff / s_cu;
                    double dy_dp = x[i];
                    
                    // Atualizar parâmetros
                    params->c[i][j] -= ALPHA * error * dys_dw * dw_dc;
                    params->s[i][j] -= ALPHA * error * dys_dw * dw_ds;
                    params->p[i][j] -= ALPHA * error * dys_dy * dy_dp;
                }
                params->q[j] -= ALPHA * error * dys_dy;
            }
        }
        
        mse_history[epoch] = total_error / train_data->num_samples;
        
        // Mostrar progresso a cada 10 épocas
        if ((epoch + 1) % 10 == 0) {
            printf("Época %d: MSE = %.6f\n", epoch + 1, mse_history[epoch]);
        }
    }
}

// Função para avaliar o modelo
void evaluate_anfis(Dataset* val_data, ANFISParams* params, double* accuracy, double* error_percent) {
    double w[NUM_RULES], y[NUM_RULES];
    int correct_predictions = 0;
    double total_error_percent = 0.0;
    
    for (int k = 0; k < val_data->num_samples; k++) {
        double* x = val_data->inputs[k];
        int target = val_data->outputs[k];
        double b;
        
        double y_pred = calys(x, params, w, y, &b);
        
        // Classificação (arredondamento e limitação)
        int y_pred_class = (int)round(y_pred);
        if (y_pred_class > 3) y_pred_class = 3;
        if (y_pred_class < 1) y_pred_class = 1;
        
        if (y_pred_class == target) {
            correct_predictions++;
        }
        
        // Erro percentual
        total_error_percent += fabs((target - y_pred) / (target + EPS));
    }
    
    *accuracy = (double)correct_predictions / val_data->num_samples * 100.0;
    *error_percent = total_error_percent / val_data->num_samples * 100.0;
}

// Função para salvar parâmetros em arquivos CSV
void save_params(ANFISParams* params) {
    FILE* file;
    
    // Salvar centros (c)
    file = fopen("results/c.csv", "w");
    if (file) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            for (int j = 0; j < NUM_RULES; j++) {
                fprintf(file, "%.6f", params->c[i][j]);
                if (j < NUM_RULES - 1) fprintf(file, ",");
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }
    
    // Salvar larguras (s)
    file = fopen("results/s.csv", "w");
    if (file) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            for (int j = 0; j < NUM_RULES; j++) {
                fprintf(file, "%.6f", params->s[i][j]);
                if (j < NUM_RULES - 1) fprintf(file, ",");
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }
    
    // Salvar coeficientes (p)
    file = fopen("results/p.csv", "w");
    if (file) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            for (int j = 0; j < NUM_RULES; j++) {
                fprintf(file, "%.6f", params->p[i][j]);
                if (j < NUM_RULES - 1) fprintf(file, ",");
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }
    
    // Salvar constantes (q)
    file = fopen("results/q.csv", "w");
    if (file) {
        for (int j = 0; j < NUM_RULES; j++) {
            fprintf(file, "%.6f", params->q[j]);
            if (j < NUM_RULES - 1) fprintf(file, ",");
        }
        fprintf(file, "\n");
        fclose(file);
    }
}

// Função para salvar resultados
void save_results(double* mse_history, double accuracy, double error_percent) {
    FILE* file = fopen("results/training_results.csv", "w");
    if (file) {
        fprintf(file, "Epoch,MSE\n");
        for (int i = 0; i < MAX_EPOCHS; i++) {
            fprintf(file, "%d,%.6f\n", i + 1, mse_history[i]);
        }
        fclose(file);
    }
    
    printf("\n=== RESULTADOS ===\n");
    printf("Acurácia no conjunto de validação: %.2f%%\n", accuracy);
    printf("Erro Percentual Médio: %.2f%%\n", error_percent);
    printf("Parâmetros salvos em: results/c.csv, results/s.csv, results/p.csv, results/q.csv\n");
    printf("Histórico de treinamento salvo em: results/training_results.csv\n");
}