#include "dataset.h"
#include <stdio.h>
#include <time.h>

typedef struct {
    double mean;
    double var;
} Stats;

typedef struct {
    Stats stats[2][MAX_FEATURES];
    double prior[2];
} NBModel;

NBModel train_nb(Dataset* train) {
    NBModel model;
    int count[2] = {0, 0};
    for (int i = 0; i < train->size; i++) count[train->data[i].label]++;

    model.prior[0] = (double)count[0] / train->size;
    model.prior[1] = (double)count[1] / train->size;

    for (int label = 0; label < 2; label++) {
        for (int j = 0; j < train->num_features; j++) {
            double sum = 0;
            int n = 0;
            for (int i = 0; i < train->size; i++) {
                if (train->data[i].label == label) {
                    sum += train->data[i].features[j];
                    n++;
                }
            }
            double mean = sum / n;
            double var_sum = 0;
            for (int i = 0; i < train->size; i++) {
                if (train->data[i].label == label) {
                    var_sum += pow(train->data[i].features[j] - mean, 2);
                }
            }
            model.stats[label][j].mean = mean;
            model.stats[label][j].var = var_sum / n + 1e-9;
        }
    }
    return model;
}

int predict_nb(NBModel* model, double* features, int num_features) {
    double max_prob = -1e18;
    int best_label = 0;

    for (int label = 0; label < 2; label++) {
        double prob = log(model->prior[label]);
        for (int j = 0; j < num_features; j++) {
            double mean = model->stats[label][j].mean;
            double var = model->stats[label][j].var;
            double x = features[j];
            double exponent = exp(-pow(x - mean, 2) / (2 * var));
            prob += log((1.0 / sqrt(2 * M_PI * var)) * exponent);
        }
        if (prob > max_prob) {
            max_prob = prob;
            best_label = label;
        }
    }
    return best_label;
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

    NBModel model = train_nb(&train);

    int test_size = ds.size - train_size;
    int tp = 0, tn = 0, fp = 0, fn = 0;

    printf("=== Naive Bayes Prediction on Image Data (Raw) ===\n");
    // Raw data from the image
    double raw_person[MAX_FEATURES] = {
        41, 1102, 1, 2, 2, 94, 3, 2, 4, 5993, 19479, 8, 11, 3, 1, 0, 8, 0, 1, 6, 4, 0, 5
    };
    
    double normalized_person[MAX_FEATURES];
    normalize_raw_input(&ds, raw_person, normalized_person);

    int pred = predict_nb(&model, normalized_person, ds.num_features);
    
    printf("Input Data (Raw): Age:%d, MonthlyIncome:%d, Distance:%d, OverTime:Yes...\n", (int)raw_person[0], (int)raw_person[9], (int)raw_person[2]);
    printf("Result: This person is predicted to %s\n", pred ? "LEAVE (Attrition)" : "STAY");
    printf("--------------------------------------------------------------------------\n");

    // Still calculate overall metrics for accuracy verification
    for (int i = train_size; i < ds.size; i++) {
        int pred = predict_nb(&model, ds.data[i].features, ds.num_features);
        int actual = ds.data[i].label;
        if (pred == 1 && actual == 1) tp++;
        else if (pred == 0 && actual == 0) tn++;
        else if (pred == 1 && actual == 0) fp++;
        else if (pred == 0 && actual == 1) fn++;
    }

    printf("--------------------------------------------------\n");
    printf("Naive Bayes Final Metrics:\n");
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
