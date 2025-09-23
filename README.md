# ANFIS em C

Este projeto implementa um sistema ANFIS (Adaptive Neuro-Fuzzy Inference System) em linguagem C, convertido do código MATLAB original.

## Estrutura do Projeto

- `anfis.h` - Header com definições de estruturas e protótipos de funções
- `anfis.c` - Implementação das funções principais do ANFIS
- `main.c` - Programa principal
- `Makefile` - Script de compilação
- `README.md` - Este arquivo

## Compilação

### Windows (MinGW/MSYS2)
```cmd
gcc main.c anfis.c -o anfis.exe -lm
```

### Linux/MacOS
```bash
gcc main.c anfis.c -o anfis -lm
```

### Usando Makefile
```bash
make          # Compilação otimizada
make debug    # Compilação com debug
make run      # Compilar e executar
make clean    # Limpar arquivos gerados
```

## Execução

```bash
./anfis       # Linux/MacOS
anfis.exe     # Windows
```

## Formato dos Dados de Entrada

O programa espera um arquivo CSV com a seguinte estrutura:
```
speed,acc_norm,engine_speed,throttle_position,delta_acc_lat,cluster_id
120.5,2.3,8500,75.2,1.8,2
...
```

Onde:
- `speed`: Velocidade do veículo (0-120)
- `acc_norm`: Aceleração normalizada (0-5)
- `engine_speed`: Rotação do motor (0-10000)
- `throttle_position`: Posição do acelerador (0-100)
- `delta_acc_lat`: Variação da aceleração lateral (0-3)
- `cluster_id`: ID do cluster/classe (1-3)

## Parâmetros Configuráveis

No arquivo `anfis.h`, você pode modificar:

```c
#define NUM_RULES 5          // Número de regras fuzzy
#define MAX_EPOCHS 100       // Número máximo de épocas
#define ALPHA 0.001          // Taxa de aprendizado
#define NUM_FEATURES 5       // Número de variáveis de entrada
#define MAX_SAMPLES 10000    // Número máximo de amostras
```

## Arquivos de Saída

O programa gera os seguintes arquivos:

- `c.csv` - Centros das funções de pertinência
- `s.csv` - Larguras das funções de pertinência
- `p.csv` - Coeficientes lineares das consequências
- `q.csv` - Termos constantes das consequências
- `training_results.csv` - Histórico do MSE durante o treinamento

## Principais Diferenças do MATLAB

1. **Gerenciamento de Memória**: Alocação e liberação manual de memória
2. **Leitura de Arquivos**: Implementação manual do parser CSV
3. **Estruturas de Dados**: Uso de structs para organizar dados
4. **Divisão de Dados**: Implementação simplificada (não estratificada)
5. **Visualização**: Não inclui gráficos (apenas dados CSV)

## Exemplo de Uso

```c
// Carregar dados
int num_samples = load_data("data.csv", raw_data);

// Normalizar
normalize_data(raw_data, num_samples, normalized_inputs);

// Dividir dados
split_data(normalized_inputs, outputs, num_samples, &train_data, &val_data, 0.7);

// Inicializar e treinar
initialize_params(&params, train_data.inputs, train_data.num_samples);
train_anfis(&train_data, &params, mse_history);

// Avaliar
evaluate_anfis(&val_data, &params, &accuracy, &error_percent);
```

## Performance

O código C é significativamente mais rápido que o MATLAB, especialmente para:
- Grandes volumes de dados
- Múltiplas execuções
- Sistemas embarcados

## Limitações

1. Não implementa divisão estratificada dos dados
2. Não gera gráficos (apenas arquivos CSV)
3. Parsing CSV básico (pode não funcionar com todos os formatos)
4. Sem validação robusta de entrada

## Possíveis Melhorias

1. Implementar divisão estratificada
2. Adicionar validação de entrada mais robusta
3. Implementar early stopping
4. Adicionar regularização
5. Paralelização com OpenMP
6. Interface gráfica com bibliotecas como GTK+ ou Qt