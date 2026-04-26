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

    printf("=== Linear Regression Prediction on Image Data (Raw) ===\n");
    // Raw data from the image
    double raw_person[MAX_FEATURES] = {
        41, 1102, 1, 2, 2, 94, 3, 2, 4, 5993, 19479, 8, 11, 3, 1, 0, 8, 0, 1, 6, 4, 0, 5
    };
    
    double normalized_person[MAX_FEATURES];
    normalize_raw_input(&ds, raw_person, normalized_person);

    int pred = predict_lr(&model, normalized_person, ds.num_features);
    
    printf("Input Data (Raw): Age:%d, MonthlyIncome:%d, Distance:%d, OverTime:Yes...\n", (int)raw_person[0], (int)raw_person[9], (int)raw_person[2]);
    printf("Result: This person is predicted to %s\n", pred ? "LEAVE (Attrition)" : "STAY");
    printf("--------------------------------------------------------------------------\n");

    // Still calculate overall metrics for accuracy verification
    for (int i = train_size; i < ds.size; i++) {
        int pred = predict_lr(&model, ds.data[i].features, ds.num_features);
        int actual = ds.data[i].label;
        if (pred == 1 && actual == 1) tp++;
        else if (pred == 0 && actual == 0) tn++;
        else if (pred == 1 && actual == 0) fp++;
        else if (pred == 0 && actual == 1) fn++;
    }

    printf("--------------------------------------------------\n");
    printf("Linear Regression Final Metrics:\n");
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
