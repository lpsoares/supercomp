// https://ugurkoltuk.wordpress.com/2010/03/04/an-extreme-simple-pgm-io-api/
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "image.h" 

// Rotina para ignorar comentarios em arquivos PGM
void SkipComments(FILE *fp)
{
   int ch;
   char line[256];
   while ((ch = fgetc(fp)) != EOF && isspace(ch)) { }
   if (ch == '#') {
      fgets(line, sizeof(line), fp);
      SkipComments(fp);
   } else fseek(fp, -1, SEEK_CUR);
}

// Ajusta informacoes e aloca dados para estrtura de armazenamento de imagens PGM
void createImage(PGMData *data, int row, int col, int max_gray)
{
   data->row = row;
   data->col = col;
   data->max_gray = max_gray;
   data->matrix = (int *)malloc(sizeof(int) * data->row * data->col);
}

// Executa a leitura de um arquivo PGM
void readImage(const char *file_name, PGMData *data)
{
   FILE *pgmFile;
   char version[3];
   int i, j;
   int tmp;
   unsigned char binary = 0;
   pgmFile = fopen(file_name, "rb");
   if (pgmFile == NULL) {
      perror("Falha ao abrir arquivo para leitura");
      exit(EXIT_FAILURE);
   }
   fgets(version, sizeof(version), pgmFile);
   if (!strcmp(version, "P5")) binary = 1;
   else if (!strcmp(version, "P2")) binary = 0;
   else {
      fprintf(stderr, "Erro ao identificar arquivo PGM!\n");
      exit(EXIT_FAILURE);
   }
   SkipComments(pgmFile);
   fscanf(pgmFile, "%d", &data->col);
   fscanf(pgmFile, "%d", &data->row);
   fscanf(pgmFile, "%d", &data->max_gray);
   fgetc(pgmFile);
 
   data->matrix = (int *)malloc(sizeof(int) * data->row * data->col);
   for (i = 0; i < data->row; ++i)
      for (j = 0; j < data->col; ++j) {
         if(binary) {
            data->matrix[i*data->col+j] = fgetc(pgmFile);
         } else {
            fscanf(pgmFile, "%d", &tmp);
            data->matrix[i*data->col+j] = tmp;
         }
      }
 
   fclose(pgmFile);
}

// Grava um arquivo PGM
void writeImage(const char *filename, const PGMData *data, unsigned char binary)
{
   FILE *pgmFile;
   int i, j;
 
   pgmFile = fopen(filename, "w");
   if (pgmFile == NULL) {
      perror("Falha ao abrir arquivo para escrita");
      exit(EXIT_FAILURE);
   }
 
   fprintf(pgmFile, (binary?"P5\n":"P2\n"));
   fprintf(pgmFile, "%d %d\n", data->col, data->row);
   fprintf(pgmFile, "%d\n", data->max_gray);
 
   for (i = 0; i < data->row; ++i) {
      for (j = 0; j < data->col; ++j) {
         if(binary) {
            fputc(data->matrix[i*data->col+j], pgmFile);   
         } else {
            fprintf(pgmFile, " %d ", data->matrix[i*data->col+j]);
         }
      }
      if(!binary) {
         fprintf(pgmFile, "\n");
      }
   }
         
   fclose(pgmFile);
   free(data->matrix);
}