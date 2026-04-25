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

    for (int i = train_size; i < ds.size; i++) {
        int pred = predict_knn(&train, ds.data[i].features, 5);
        int actual = ds.data[i].label;

        if (pred == 1 && actual == 1) tp++;
        else if (pred == 0 && actual == 0) tn++;
        else if (pred == 1 && actual == 0) fp++;
        else if (pred == 0 && actual == 1) fn++;
    }

    double acc = (double)(tp + tn) / test_size;
    double prec = (tp + fp > 0) ? (double)tp / (tp + fp) : 0;
    double rec = (tp + fn > 0) ? (double)tp / (tp + fn) : 0;
    double f1 = (prec + rec > 0) ? 2 * prec * rec / (prec + rec) : 0;

    printf("KNN Results:\n");
    printf("Accuracy: %.4f\n", acc);
    printf("Precision: %.4f\n", prec);
    printf("Recall: %.4f\n", rec);
    printf("F1-score: %.4f\n", f1);

    return 0;
}
