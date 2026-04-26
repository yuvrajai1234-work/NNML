#include <stdio.h>

#define WIDTH 60
#define HEIGHT 15

void draw_lr_line() {
    char grid[HEIGHT][WIDTH];
    
    // Initialize grid
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            grid[i][j] = ' ';
        }
    }

    // Linear Equation: y = 0.5x + 0.2 (Sample slope for visualization)
    for (int j = 0; j < WIDTH; j++) {
        double x = (double)j / WIDTH;
        double y = 0.8 * x + 0.1; // Linear trend
        
        int h = (int)(y * (HEIGHT - 1));
        if (h >= 0 && h < HEIGHT) {
            grid[HEIGHT - 1 - h][j] = '/';
        }
    }

    printf("\n=== Linear Regression ASCII Graph (Decision Slope) ===\n");
    printf("Visualizing Linear Correlation between Features and Attrition\n\n");
    printf("Prob ^\n");
    for (int i = 0; i < HEIGHT; i++) {
        printf(" 1.0 |");
        for (int j = 0; j < WIDTH; j++) {
            printf("%c", grid[i][j]);
        }
        printf("\n");
    }
    printf(" 0.0 +------------------------------------------------------------\n");
    printf("     0.0                      Feature Value                        1.0\n\n");
}

int main() {
    draw_lr_line();
    return 0;
}
