#include "dataset.h"
#include <stdio.h>
#include <math.h>

#define WIDTH 70
#define HEIGHT 15

void draw_nb_curve() {
    // Mathematical constants for a standard bell curve
    double mean = 0.5, stddev = 0.15;
    char grid[HEIGHT][WIDTH];

    // Initialize grid
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            grid[i][j] = ' ';
        }
    }

    // Calculate Gaussian PDF: (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x-mu)/sigma)^2)
    for (int j = 0; j < WIDTH; j++) {
        double x = (double)j / WIDTH;
        double exponent = -0.5 * pow((x - mean) / stddev, 2);
        double pdf = (1.0 / (stddev * sqrt(2 * M_PI))) * exp(exponent);
        
        // Scale PDF to height
        int h = (int)((pdf / 3.0) * (HEIGHT - 1));
        if (h >= 0 && h < HEIGHT) {
            grid[HEIGHT - 1 - h][j] = '#';
        }
    }

    printf("\n=== Naive Bayes ASCII Graph (Gaussian PDF) ===\n");
    printf("Visualizing Probability Density Function for Attrition Risk\n\n");
    for (int i = 0; i < HEIGHT; i++) {
        printf(" |");
        for (int j = 0; j < WIDTH; j++) {
            printf("%c", grid[i][j]);
        }
        printf("\n");
    }
    printf(" +----------------------------------------------------------------------\n");
    printf("  Low Risk                      Mean Probability                     High Risk\n\n");
}

int main() {
    draw_nb_curve();
    return 0;
}
