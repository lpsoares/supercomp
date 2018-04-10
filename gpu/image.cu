// Leitura de arquivos PGM, PPM, etc
// Criado por: Luciano P. Soares (10 de Abril de 2018)
// originalmente baseado em: ugurkoltuk.wordpress.com/2010/03/04/an-extreme-simple-pgm-io-api/

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
void createImage(PGMData *data, int row, int col, int max_intensity, int channels)
{
   data->row = row;
   data->col = col;
   data->max_intensity = max_intensity;
   data->channels = channels;
   data->matrix = (int *)malloc(sizeof(int) * data->row * data->col * data->channels);
}

// Executa a leitura de um arquivo PGM
void readImage(const char *file_name, PGMData *data)
{
   FILE *pgmFile;
   char version[3];
   int i, j, k;
   int tmp;
   unsigned char binary = 0;

   pgmFile = fopen(file_name, "rb");
   if (pgmFile == NULL) {
      perror("Falha ao abrir arquivo para leitura");
      exit(EXIT_FAILURE);
   }
   fgets(version, sizeof(version), pgmFile);
   if (!strcmp(version, "P6")) { data->channels= 3; binary = 1; }
   else if (!strcmp(version, "P5")) { data->channels= 1; binary = 1; }
   else if (!strcmp(version, "P3")) { data->channels = 3; binary = 0; }
   else if (!strcmp(version, "P2")) { data->channels = 1; binary = 0; }
      else {
      fprintf(stderr, "Erro ao identificar arquivo PGM: %s\n",version);
      exit(EXIT_FAILURE);
   }
   SkipComments(pgmFile);
   fscanf(pgmFile, "%d", &data->col);
   fscanf(pgmFile, "%d", &data->row);
   fscanf(pgmFile, "%d", &data->max_intensity);
   fgetc(pgmFile);
 
   data->matrix = (int *)malloc(sizeof(int) * data->row * data->col * data->channels);
   for (i = 0; i < data->row; i++)
      for (j = 0; j < data->col; j++) {
         for (k = 0; k < data->channels; k++) {
            if(binary) {
               data->matrix[(i*data->col+j)*data->channels+k] = fgetc(pgmFile);
            } else {
               fscanf(pgmFile, "%d", &tmp);
               data->matrix[(i*data->col+j)*data->channels+k] = tmp;
            }
         }
      }
 
   fclose(pgmFile);
}

// Grava um arquivo PGM
void writeImage(const char *filename, const PGMData *data, unsigned char binary)
{
   FILE *pgmFile;
   int i, j, k;
 
   pgmFile = fopen(filename, "w");
   if (pgmFile == NULL) {
      perror("Falha ao abrir arquivo para escrita");
      exit(EXIT_FAILURE);
   }
 
   fprintf(pgmFile, ( data->channels==3 ? (binary?"P6\n":"P3\n") : (binary?"P5\n":"P2\n") ) );
   fprintf(pgmFile, "%d %d\n", data->col, data->row);
   fprintf(pgmFile, "%d\n", data->max_intensity);
 
   for (i = 0; i < data->row; ++i) {
      for (j = 0; j < data->col; ++j) {
         for (k = 0; k < data->channels; k++) {
            if(binary) {
               fputc(data->matrix[(i*data->col+j)*data->channels+k], pgmFile);   
            } else {
               fprintf(pgmFile, " %d", data->matrix[(i*data->col+j)*data->channels+k]);
            }
         }
         if(!binary) fprintf(pgmFile, " ");
      }
      if(!binary) fprintf(pgmFile, "\n");
   }
         
   fclose(pgmFile);
   free(data->matrix);
}