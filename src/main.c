#include "anfis.h"

int main() {
    printf("=== ANFIS em C ===\n");
    printf("Inicializando sistema...\n\n");
    
    // Inicializar gerador de números aleatórios
    srand(46);
    
    // Alocar memória para os dados
    DataPoint* raw_data = malloc(MAX_SAMPLES * sizeof(DataPoint));
    if (!raw_data) {
        printf("Erro ao alocar memória para dados brutos\n");
        return -1;
    }
    
    // Carregar dados do CSV
    printf("Carregando dados...\n");
    int num_samples = load_data("data/data.csv", raw_data);
    if (num_samples <= 0) {
        printf("Erro ao carregar dados ou arquivo vazio\n");
        free(raw_data);
        return -1;
    }
    printf("Dados carregados: %d amostras\n", num_samples);
    
    // Alocar memória para dados normalizados
    double (*normalized_inputs)[NUM_FEATURES] = malloc(num_samples * sizeof(*normalized_inputs));
    int* outputs = malloc(num_samples * sizeof(int));
    if (!normalized_inputs || !outputs) {
        printf("Erro ao alocar memória para dados normalizados\n");
        free(raw_data);
        free(normalized_inputs);
        free(outputs);
        return -1;
    }
    
    // Normalizar dados
    printf("Normalizando dados...\n");
    normalize_data(raw_data, num_samples, normalized_inputs);
    
    // Extrair saídas
    for (int i = 0; i < num_samples; i++) {
        outputs[i] = raw_data[i].cluster_id;
    }
    
    // Dividir dados em treino e validação (70% treino, 30% validação) - estratificado
    Dataset train_data, val_data;
    printf("Dividindo dados em treino e validação (estratificado)...\n");
    split_data_stratified(normalized_inputs, outputs, num_samples, &train_data, &val_data, 0.7);
    printf("Dados de treino: %d amostras\n", train_data.num_samples);
    printf("Dados de validação: %d amostras\n", val_data.num_samples);
    
    // Inicializar parâmetros do ANFIS
    ANFISParams params;
    printf("Inicializando parâmetros do ANFIS...\n");
    initialize_params(&params, train_data.inputs, train_data.num_samples);
    
    // Alocar memória para histórico de MSE
    double* mse_history = malloc(MAX_EPOCHS * sizeof(double));
    if (!mse_history) {
        printf("Erro ao alocar memória para histórico MSE\n");
        free(raw_data);
        free(normalized_inputs);
        free(outputs);
        return -1;
    }
    
    // Treinar o modelo
    printf("\nIniciando treinamento do ANFIS...\n");
    printf("Parâmetros: %d regras, %d épocas, taxa de aprendizado = %.4f\n", 
           NUM_RULES, MAX_EPOCHS, ALPHA);
    printf("----------------------------------------\n");
    
    clock_t start_time = clock();
    train_anfis(&train_data, &params, mse_history);
    clock_t end_time = clock();
    
    double training_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("----------------------------------------\n");
    printf("Treinamento concluído em %.2f segundos\n\n", training_time);
    
    // Avaliar o modelo
    printf("Avaliando modelo no conjunto de validação...\n");
    double accuracy, error_percent;
    evaluate_anfis(&val_data, &params, &accuracy, &error_percent);
    
    // Salvar parâmetros e resultados
    printf("Salvando parâmetros e resultados...\n");
    save_params(&params);
    save_results(mse_history, accuracy, error_percent);
    
    // Mostrar estatísticas finais do treinamento
    printf("\n=== ESTATÍSTICAS DO TREINAMENTO ===\n");
    printf("MSE inicial: %.6f\n", mse_history[0]);
    printf("MSE final: %.6f\n", mse_history[MAX_EPOCHS - 1]);
    printf("Redução do erro: %.2f%%\n", 
           (1.0 - mse_history[MAX_EPOCHS - 1] / mse_history[0]) * 100.0);
    
    // Limpeza de memória
    free(raw_data);
    free(normalized_inputs);
    free(outputs);
    free(mse_history);
    
    printf("\nPrograma finalizado com sucesso!\n");
    return 0;
}