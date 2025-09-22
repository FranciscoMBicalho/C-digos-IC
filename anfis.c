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

// Função para dividir dados em treino e validação
void split_data(double inputs[][NUM_FEATURES], int* outputs, int num_samples, 
                Dataset* train_data, Dataset* val_data, double train_ratio) {
    int train_size = (int)(num_samples * train_ratio);
    
    // Copiar dados de treino
    train_data->num_samples = train_size;
    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            train_data->inputs[i][j] = inputs[i][j];
        }
        train_data->outputs[i] = outputs[i];
    }
    
    // Copiar dados de validação
    val_data->num_samples = num_samples - train_size;
    for (int i = train_size; i < num_samples; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            val_data->inputs[i - train_size][j] = inputs[i][j];
        }
        val_data->outputs[i - train_size] = outputs[i];
    }
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
    srand(42); // Semente para reprodutibilidade
    
    for (int j = 0; j < NUM_RULES; j++) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            params->c[i][j] = random_double(xmin[i], xmax[i]);
            params->s[i][j] = random_double(0.1, 1.0);
            params->p[i][j] = random_double(-1.0, 1.0);
        }
        params->q[j] = random_double(-1.0, 1.0);
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
    return (b > 1e-10) ? a / b : 0.0;  // Evitar divisão por zero
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
                double dys_dw = (y[j] - ys) / (b + 1e-10);
                double dys_dy = w[j] / (b + 1e-10);
                
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
        total_error_percent += fabs((target - y_pred) / (target + 1e-10));
    }
    
    *accuracy = ((double)correct_predictions / val_data->num_samples) * 100.0;
    *error_percent = (total_error_percent / val_data->num_samples) * 100.0;
}

// Função para salvar parâmetros em arquivos CSV
void save_params(ANFISParams* params) {
    FILE* file;
    
    // Salvar centros (c)
    file = fopen("c.csv", "w");
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
    file = fopen("s.csv", "w");
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
    file = fopen("p.csv", "w");
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
    file = fopen("q.csv", "w");
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
    FILE* file = fopen("training_results.csv", "w");
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
    printf("Parâmetros salvos em: c.csv, s.csv, p.csv, q.csv\n");
    printf("Histórico de treinamento salvo em: training_results.csv\n");
}

void randomize_matrix(double inputs[][NUM_FEATURES], int* outputs,int num_samples,Dataset* train_data, Dataset* val_data)
{
    //DataPoint data[MAX_SAMPLES];
    //int total_samples = load_data(input_file, data);

    //if (total_samples <= 0) {
        //printf("Erro ao carregar dados de %s\n", input_file);
        //return;
    //}

    // Embaralhar os índices
    int indices[MAX_SAMPLES];
    for (int i = 0; i < num_samples; i++) indices[i] = i;
    //srand((unsigned int)time(NULL));
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    int train_count = (int)(num_samples * 0.7);
    train_data->num_samples = train_count;
    val_data->num_samples = num_samples - train_count;

    // Escrever em train_data

    for (int i = 0; i < train_data->num_samples; i++) {
        int idx = indices[i];
        for (int j = 0; j < NUM_FEATURES; j++) {
            train_data->inputs[i][j] = inputs[idx][j];
        }
        train_data->outputs[i] = outputs[idx];
    }

    // Escrever em val_data

    for (int i = 0; i < val_data->num_samples; i++) {
        int idx = indices[i];
        for (int j = 0; j < NUM_FEATURES; j++) {
            val_data->inputs[i][j] = inputs[idx][j];
        }
        val_data->outputs[i] = outputs[idx];
    }

    // Escrever training_csv.csv
    /*FILE* ftrain = fopen(train_file, "w");
    if (!ftrain) {
        printf("Erro ao criar %s\n", train_file);
        return;
    }
    // Cabeçalho igual ao arquivos.csv/data.csv
    fprintf(ftrain, "speed,acc_norm,engine_speed,throttle_position,delta_acc_lat,cluster_id\n");
    for (int i = 0; i < train_count; i++) {
        int idx = indices[i];
        fprintf(ftrain, "%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
            data[idx].speed,
            data[idx].acc_norm,
            data[idx].engine_speed,
            data[idx].throttle_position,
            data[idx].delta_acc_lat,
            data[idx].cluster_id);
    }
    fclose(ftrain);

    // Escrever validation.csv
    FILE* fval = fopen(val_file, "w");
    if (!fval) {
        printf("Erro ao criar %s\n", val_file);
        return;
    }
    fprintf(fval, "speed,acc_norm,engine_speed,throttle_position,delta_acc_lat,cluster_id\n");
    for (int i = train_count; i < total_samples; i++) {
        int idx = indices[i];
        fprintf(fval, "%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
            data[idx].speed,
            data[idx].acc_norm,
            data[idx].engine_speed,
            data[idx].throttle_position,
            data[idx].delta_acc_lat,
            data[idx].cluster_id);
    }
    fclose(fval);

    printf("Dados embaralhados e divididos em %s (%d linhas) e %s (%d linhas)\n",
        train_file, train_count, val_file, total_samples - train_count);*/
}

void write_csv_train_val(const char* train_file, const char* val_file,Dataset* train_data, Dataset* val_data)
{
    // Escrever training.csv
    FILE* ftrain = fopen(train_file, "w");
    if (!ftrain) {
        printf("Erro ao criar %s\n", train_file);
        return;
    }
    // Cabeçalho igual ao arquivos.csv/data.csv
    fprintf(ftrain, "speed,acc_norm,engine_speed,throttle_position,delta_acc_lat,cluster_id\n");
    for (int i = 0; i < train_data->num_samples; i++) {
        fprintf(ftrain, "%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
            train_data->inputs[i][0],
            train_data->inputs[i][1],
            train_data->inputs[i][2],
            train_data->inputs[i][3],
            train_data->inputs[i][4],
            train_data->outputs[i]);
    }
    fclose(ftrain);

    // Escrever validation.csv

    FILE* fval = fopen(val_file, "w");
    if (!fval) {
        printf("Erro ao criar %s\n", val_file);
        return;
    }
    fprintf(fval, "speed,acc_norm,engine_speed,throttle_position,delta_acc_lat,cluster_id\n");
    for (int i = 0; i < val_data->num_samples; i++) {
        fprintf(fval, "%.6f,%.6f,%.6f,%.6f,%.6f,%d\n",
            val_data->inputs[i][0],
            val_data->inputs[i][1],
            val_data->inputs[i][2],
            val_data->inputs[i][3],
            val_data->inputs[i][4],
            val_data->outputs[i]);
    }
    fclose(fval);

}

/*void load_train_val_csv(const char* train_file, const char* val_file,Dataset* train_data, Dataset* val_data) {
    FILE* ftrain = fopen(train_file, "r");
    FILE* fval = fopen(val_file, "r");
    char line[MAX_LINE_LENGTH];

    // Carregar training.csv
    train_data->num_samples = 0;
    if (ftrain) {
        fgets(line, sizeof(line), ftrain); // Pular cabeçalho
        while (fgets(line, sizeof(line), ftrain)) {
            double speed, acc_norm, engine_speed, throttle_position, delta_acc_lat;
            int cluster_id;
            if (sscanf(line, "%lf,%lf,%lf,%lf,%lf,%d",
                       &speed, &acc_norm, &engine_speed, &throttle_position, &delta_acc_lat, &cluster_id) == 6) {
                int idx = train_data->num_samples;
                train_data->inputs[idx][0] = speed;
                train_data->inputs[idx][1] = acc_norm;
                train_data->inputs[idx][2] = engine_speed;
                train_data->inputs[idx][3] = throttle_position;
                train_data->inputs[idx][4] = delta_acc_lat;
                train_data->outputs[idx] = cluster_id;
                train_data->num_samples++;
            }
        }
        fclose(ftrain);
    }

    // Carregar validation.csv
    val_data->num_samples = 0;
    if (fval) {
        fgets(line, sizeof(line), fval); // Pular cabeçalho
        while (fgets(line, sizeof(line), fval)) {
            double speed, acc_norm, engine_speed, throttle_position, delta_acc_lat;
            int cluster_id;
            if (sscanf(line, "%lf,%lf,%lf,%lf,%lf,%d",
                       &speed, &acc_norm, &engine_speed, &throttle_position, &delta_acc_lat, &cluster_id) == 6) {
                int idx = val_data->num_samples;
                val_data->inputs[idx][0] = speed;
                val_data->inputs[idx][1] = acc_norm;
                val_data->inputs[idx][2] = engine_speed;
                val_data->inputs[idx][3] = throttle_position;
                val_data->inputs[idx][4] = delta_acc_lat;
                val_data->outputs[idx] = cluster_id;
                val_data->num_samples++;
            }
        }
        fclose(fval);
    }
}*/