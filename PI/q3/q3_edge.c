#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>

#include "image.h" 

#define MAX(y,x) (y>x?y:x)    // Calcula valor maximo
#define MIN(y,x) (y<x?y:x)    // Calcula valor minimo

// Filtro de bordas
void edgeFilter(int *in, int *out, int rowStart, int rowEnd, int colStart, int colEnd)
{
   int i,j,di,dj;
   for(i = rowStart; i < rowEnd; ++i) {
      for(j = colStart; j < colEnd; ++j) {
         int min = 256;
         int max = 0;
         for(di = MAX(rowStart, i - 1); di <= MIN(i + 1, rowEnd - 1); di++) {
            for(dj = MAX(colStart, j - 1); dj <= MIN(j + 1, colEnd - 1); dj++) {
               if(min>in[di*(colEnd-colStart)+dj]) min = in[di*(colEnd-colStart)+dj];
               if(max<in[di*(colEnd-colStart)+dj]) max = in[di*(colEnd-colStart)+dj]; 
            }
         }
         out[i*(colEnd-colStart)+j] = max-min;
      }
   }
}


int main(int argc, char** argv)
{

   // Estruturas que organizam as imagens PGM
   PGMData *imagemIn = malloc(sizeof(PGMData));
   PGMData *imagemOut = malloc(sizeof(PGMData));

   readImage(argv[1],imagemIn);

   createImage(imagemOut, imagemIn->row, imagemIn->col, imagemIn->max_gray);

   // Processa os dados da imagem para a deteccao de borda
   edgeFilter(imagemIn->matrix, imagemOut->matrix, 0, imagemIn->row, 0, imagemIn->col);

   writeImage(argv[2],imagemOut,0);

   return 0;
}