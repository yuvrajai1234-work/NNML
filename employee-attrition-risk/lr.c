#include "dataset.h"
#include <stdio.h>
#include <time.h>

typedef struct {
    double weights[MAX_FEATURES];
    double bias;
} LRModel;

LRModel train_lr(Dataset* train, double lr, int epochs) {
    LRModel model;
    for (int j = 0; j < train->num_features; j++) model.weights[j] = 0.0;
    model.bias = 0.0;

    for (int e = 0; e < epochs; e++) {
        double dw[MAX_FEATURES] = {0};
        double db = 0;
        for (int i = 0; i < train->size; i++) {
            double y_pred = model.bias;
            for (int j = 0; j < train->num_features; j++) {
                y_pred += model.weights[j] * train->data[i].features[j];
            }
            double error = y_pred - train->data[i].label;
            
            for (int j = 0; j < train->num_features; j++) {
                dw[j] += error * train->data[i].features[j];
            }
            db += error;
        }
        for (int j = 0; j < train->num_features; j++) {
            model.weights[j] -= lr * dw[j] / train->size;
        }
        model.bias -= lr * db / train->size;
    }
    return model;
}

int predict_lr(LRModel* model, double* features, int num_features) {
    double y = model->bias;
    for (int j = 0; j < num_features; j++) {
        y += model->weights[j] * features[j];
    }
    return (y >= 0.5) ? 1 : 0;
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

    LRModel model = train_lr(&train, 0.1, 5000);

    int test_size = ds.size - train_size;
    int tp = 0, tn = 0, fp = 0, fn = 0;

    for (int i = train_size; i < ds.size; i++) {
        int pred = predict_lr(&model, ds.data[i].features, ds.num_features);
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

    printf("Linear Regression Results:\n");
    printf("Accuracy: %.4f\n", acc);
    printf("Precision: %.4f\n", prec);
    printf("Recall: %.4f\n", rec);
    printf("F1-score: %.4f\n", f1);

    return 0;
}
