#include "dataset.h"
#include <stdio.h>
#include <time.h>

double euclidean_distance(double* a, double* b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
}

typedef struct {
    double dist;
    int label;
} Neighbor;

int compare_neighbors(const void* a, const void* b) {
    Neighbor* n1 = (Neighbor*)a;
    Neighbor* n2 = (Neighbor*)b;
    if (n1->dist < n2->dist) return -1;
    if (n1->dist > n2->dist) return 1;
    return 0;
}

int predict_knn(Dataset* train, double* test_feat, int k) {
    Neighbor* neighbors = malloc(train->size * sizeof(Neighbor));
    for (int i = 0; i < train->size; i++) {
        neighbors[i].dist = euclidean_distance(train->data[i].features, test_feat, train->num_features);
        neighbors[i].label = train->data[i].label;
    }

    qsort(neighbors, train->size, sizeof(Neighbor), compare_neighbors);

    int count1 = 0, count0 = 0;
    for (int i = 0; i < k; i++) {
        if (neighbors[i].label == 1) count1++;
        else count0++;
    }

    free(neighbors);
    return (count1 > count0) ? 1 : 0;
}

int main() {
    srand(42);
    Dataset ds = load_dataset("IBM_HR_Attrition.csv");
    if (ds.size == 0) return 1;

    normalize_dataset(&ds);
    shuffle_dataset(&ds);

    int train_size = ds.size * 0.8;
    Dataset train;
    train.size = train_size;
    train.num_features = ds.num_features;
    for(int i=0; i<train_size; i++) train.data[i] = ds.data[i];

    int test_size = ds.size - train_size;
    int tp = 0, tn = 0, fp = 0, fn = 0;

    printf("=== KNN Prediction on Image Data (Raw) ===\n");
    // Raw data from the image
    double raw_person[MAX_FEATURES] = {
        41, 1102, 1, 2, 2, 94, 3, 2, 4, 5993, 19479, 8, 11, 3, 1, 0, 8, 0, 1, 6, 4, 0, 5
    };
    
    double normalized_person[MAX_FEATURES];
    normalize_raw_input(&ds, raw_person, normalized_person);

    int pred = predict_knn(&train, normalized_person, 5);
    
    printf("Input Data (Raw): Age:%d, MonthlyIncome:%d, Distance:%d, OverTime:Yes...\n", (int)raw_person[0], (int)raw_person[9], (int)raw_person[2]);
    printf("Result: This person is predicted to %s\n", pred ? "LEAVE (Attrition)" : "STAY");
    printf("--------------------------------------------------------------------------\n");

    // Still calculate overall metrics for accuracy verification
    for (int i = train_size; i < ds.size; i++) {
        int pred = predict_knn(&train, ds.data[i].features, 5);
        int actual = ds.data[i].label;
        if (pred == 1 && actual == 1) tp++;
        else if (pred == 0 && actual == 0) tn++;
        else if (pred == 1 && actual == 0) fp++;
        else if (pred == 0 && actual == 1) fn++;
    }

    printf("--------------------------------------------------\n");
    printf("KNN Final Metrics:\n");
    double acc = (double)(tp + tn) / test_size;
    double prec = (tp + fp > 0) ? (double)tp / (tp + fp) : 0;
    double rec = (tp + fn > 0) ? (double)tp / (tp + fn) : 0;
    double f1 = (prec + rec > 0) ? 2 * prec * rec / (prec + rec) : 0;

    printf("Accuracy: %.4f\n", acc);
    printf("Precision: %.4f\n", prec);
    printf("Recall: %.4f\n", rec);
    printf("F1-score: %.4f\n", f1);

    return 0;
}
