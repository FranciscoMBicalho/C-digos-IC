# Makefile para compilar o projeto ANFIS

# Compilador e flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -O2 -lm
DEBUG_FLAGS = -g -DDEBUG

# Arquivos
SOURCES = main.c anfis.c
HEADERS = anfis.h
EXECUTABLE = anfis
EXECUTABLE_DEBUG = anfis_debug

# Regra padrão
all: $(EXECUTABLE)

# Regra para compilação otimizada
$(EXECUTABLE): $(SOURCES) $(HEADERS)
	$(CC) $(SOURCES) -o $(EXECUTABLE) $(CFLAGS)

# Regra para compilação com debug
debug: $(SOURCES) $(HEADERS)
	$(CC) $(SOURCES) -o $(EXECUTABLE_DEBUG) $(CFLAGS) $(DEBUG_FLAGS)

# Regra para limpeza
clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE_DEBUG) c.csv	p.csv	q.csv	s.csv	training_results.csv *.o

# Regra para executar
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Regra para executar com debug
run-debug: debug
	./$(EXECUTABLE_DEBUG)

# Regras que não geram arquivos
.PHONY: all clean run run-debug debug