#include "dataset.h"
#include <stdio.h>

#define WIDTH 60
#define HEIGHT 20

void draw_knn_graph(Dataset* ds) {
    char grid[HEIGHT][WIDTH];
    
    // Initialize grid
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            grid[i][j] = ' ';
        }
    }

    // Plot points
    for (int i = 0; i < ds->size; i++) {
        int x = (int)(ds->data[i].features[0] * (WIDTH - 1)); // Age
        int y = (int)(ds->data[i].features[9] * (HEIGHT - 1)); // Monthly Income (Index 9 in raw, mapped in numeric_indices)
        
        // Ensure within bounds
        if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
            grid[HEIGHT - 1 - y][x] = (ds->data[i].label == 1) ? 'X' : '.';
        }
    }

    // Print Grid
    printf("\n=== KNN ASCII Graph (Age vs Income) ===\n");
    printf("Legend: '.' = Stay, 'X' = Attrition\n\n");
    printf("Income ^\n");
    for (int i = 0; i < HEIGHT; i++) {
        printf("%-7s |", (i == 0) ? "High" : (i == HEIGHT-1) ? "Low" : "");
        for (int j = 0; j < WIDTH; j++) {
            printf("%c", grid[i][j]);
        }
        printf("\n");
    }
    printf("        +------------------------------------------------------------\n");
    printf("        | Young                                                Old  -> Age\n\n");
}

int main() {
    Dataset ds = load_dataset("IBM_HR_Attrition.csv");
    if (ds.size == 0) return 1;
    normalize_dataset(&ds);
    draw_knn_graph(&ds);
    return 0;
}
