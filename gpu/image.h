// referencia https://ugurkoltuk.wordpress.com/2010/03/04/an-extreme-simple-pgm-io-api/
#include <stdlib.h>

// Estrutura de dados para as imagens PGM
typedef struct _PGMData {
   int row;
   int col;
   int max_intensity;
   int channels;
   int *matrix;
} PGMData;

// Rotina para ignorar comentarios em arquivos PGM
void SkipComments(FILE *fp);

// Ajusta informacoes e aloca dados para estrtura de armazenamento de imagens PGM
void createImage(PGMData *data, int row, int col, int max_intensity, int channels);

// Executa a leitura de um arquivo PGM
void readImage(const char *file_name, PGMData *data);

// Grava um arquivo PGM
void writeImage(const char *filename, const PGMData *data, unsigned char binary);
