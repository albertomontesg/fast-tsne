#ifndef IO_H
#define IO_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "data_type.h"
#include <string.h>

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool load_data(dt* data, int n, int* d, char* data_file) {

	// Open file, read first 2 integers, allocate memory, and read the data
    FILE *h;
	if((h = fopen(data_file, "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}

    int magic_number, origN, rows, cols;
    fread(&magic_number, sizeof(int), 1, h);
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) {
        printf("Invalid magic number, it should be 2051, instead %d found\n", magic_number);
        return false;
    }
    fread(&origN, sizeof(uint32_t), 1, h);
    origN = reverse_int(origN);
    #ifdef DEBUG
    printf("Number of samples available: %d\n", origN);
    #endif
    if (n > origN) {
        printf("N can not be larger than the number of samples in the data file (%d)\n", origN);
        return false;
    }

    fread(&rows, sizeof(int), 1, h);
    fread(&cols, sizeof(int), 1, h);
    rows = reverse_int(rows);
    cols = reverse_int(cols);
    *d = rows * cols;

    // Read the data
    unsigned char* raw_data = (unsigned char*) malloc(*d * n * sizeof(uint8_t));
    if(data == NULL) { printf("[data] Memory allocation failed!\n"); exit(1); }
    if(raw_data == NULL) { printf("[raw_data] Memory allocation failed!\n"); exit(1); }

    // Read raw data into unsigned char vector
    fread(raw_data, sizeof(uint8_t), n * *d, h);

    for (int i = 0; i < *d * n; i++) {
        data[i] = ((dt) raw_data[i]) / 255.;
    }

    // Showing image
    if (false) {
        int offset = 10;
        printf("Sample %d\n\n", offset+1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double v = data[offset * *d + i*cols+j];
                if (v > 0.)
                    printf("\x1B[31m %.2f \x1B[0m\t", v);
                else printf("%.2f\t", v);
            }
            printf("\n");
        }
    }
    free(raw_data); raw_data = NULL;
    return true;
}

void save_data(float* data, int n, int d, char* data_file) {
    /* Function to save a matrix into a file, writing the size at the beginning
    and then all the data values */

	FILE *h;
	if((h = fopen(data_file, "w+b")) == NULL) {
		printf("Error: could not open data file (%s).\n", data_file);
		return;
	}
	fwrite(&n, sizeof(int), 1, h);
	fwrite(&d, sizeof(int), 1, h);
    fwrite(data, sizeof(dt), n * d, h);
    fclose(h);
    #ifdef DEBUG
	printf("Wrote the %i x %i data matrix successfully!\n", n, d);
    #endif
}

bool validate_data(dt* data, int n, int d, char* ref_file){
    dt *ref_data = (dt*) malloc(d * n * sizeof(dt));

    int ref_d; ref_d = -1;
    if(!load_data(ref_data, n, &ref_d, ref_file)){
        printf("Error: could not open reference file (%s).\n", ref_file);
        return false;
    }

    if(ref_d != d){
        printf("Error: non matching dimensions between output and reference file (%s).\n", ref_file);
        printf("Output dimension data: %d  Output dimension reference file: %d. \n", d, ref_d);
        return false;
    }

    for(int i=0; i<n*d; i++){
        if(data[i] != ref_data[i]){
            printf("Calculated data does not match reference! \n");
            printf("data point %d, dimension %d: \n", i/n, i%d);
            printf("-reference:  %f\n", ref_data[i]);
            printf("-calculated: %f\n\n", data[i]);
            return false;
        }
    }
    free(ref_data);
    return true;
}

#endif
