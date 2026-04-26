#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_FEATURES 23
#define MAX_SAMPLES 1500

typedef struct {
    double features[MAX_FEATURES];
    int label;
} DataPoint;

typedef struct {
    DataPoint data[MAX_SAMPLES];
    int size;
    int num_features;
    double min[MAX_FEATURES];
    double max[MAX_FEATURES];
} Dataset;

// Indices of numeric features in the IBM dataset
int numeric_indices[] = {0, 3, 5, 6, 10, 12, 13, 14, 16, 18, 19, 20, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34};

Dataset load_dataset(const char* filename) {
    Dataset ds;
    ds.size = 0;
    ds.num_features = MAX_FEATURES;

    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return ds;
    }

    char line[2048];
    // Skip header
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file) && ds.size < MAX_SAMPLES) {
        char* tmp = strdup(line);
        char* token;
        int col = 0;
        int feat_idx = 0;

        token = strtok(tmp, ",");
        while (token != NULL) {
            // Attrition is at column 1
            if (col == 1) {
                if (strcmp(token, "Yes") == 0) ds.data[ds.size].label = 1;
                else ds.data[ds.size].label = 0;
            } else {
                // Check if this column is one of our numeric features
                for (int i = 0; i < MAX_FEATURES; i++) {
                    if (col == numeric_indices[i]) {
                        ds.data[ds.size].features[i] = atof(token);
                        break;
                    }
                }
            }
            token = strtok(NULL, ",");
            col++;
        }
        free(tmp);
        ds.size++;
    }
    fclose(file);
    return ds;
}

void normalize_dataset(Dataset* ds) {
    for (int j = 0; j < ds->num_features; j++) {
        ds->min[j] = 1e18;
        ds->max[j] = -1e18;
    }

    for (int i = 0; i < ds->size; i++) {
        for (int j = 0; j < ds->num_features; j++) {
            if (ds->data[i].features[j] < ds->min[j]) ds->min[j] = ds->data[i].features[j];
            if (ds->data[i].features[j] > ds->max[j]) ds->max[j] = ds->data[i].features[j];
        }
    }

    for (int i = 0; i < ds->size; i++) {
        for (int j = 0; j < ds->num_features; j++) {
            if (ds->max[j] - ds->min[j] > 1e-9) {
                ds->data[i].features[j] = (ds->data[i].features[j] - ds->min[j]) / (ds->max[j] - ds->min[j]);
            }
        }
    }
}

void normalize_raw_input(Dataset* ds, double* raw_input, double* normalized_output) {
    for (int j = 0; j < ds->num_features; j++) {
        if (ds->max[j] - ds->min[j] > 1e-9) {
            normalized_output[j] = (raw_input[j] - ds->min[j]) / (ds->max[j] - ds->min[j]);
        } else {
            normalized_output[j] = 0.0;
        }
    }
}

void shuffle_dataset(Dataset* ds) {
    for (int i = ds->size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        DataPoint temp = ds->data[i];
        ds->data[i] = ds->data[j];
        ds->data[j] = temp;
    }
}

#endif
