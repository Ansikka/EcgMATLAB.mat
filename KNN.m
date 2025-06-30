%% firstly paste & run this: step 1

%% Load ECG signals
load('ecg_normal1.mat');    % should load variable: signal
signal_normal = signal;

load('ecg_disease1.mat');   % should load variable: signal
signal_disease = signal;

%% Segment parameters
segment_length = 250;
num_segments = 5000;
total_length = segment_length * num_segments;

%% Repeat signals if needed
signal_normal = repmat(signal_normal, 1, ceil(total_length / length(signal_normal)));
signal_disease = repmat(signal_disease, 1, ceil(total_length / length(signal_disease)));

signal_normal = signal_normal(1:total_length);
signal_disease = signal_disease(1:total_length);

%% Reshape into segments
segments_normal = reshape(signal_normal, segment_length, num_segments);
segments_disease = reshape(signal_disease, segment_length, num_segments);

%% Z-score normalize
segments_normal = (segments_normal - mean(segments_normal)) ./ std(segments_normal);
segments_disease = (segments_disease - mean(segments_disease)) ./ std(segments_disease);

%% LHOM feature extraction (1–100) for all 10,000 segments
lhom_normal = zeros(num_segments, 100);
lhom_disease = zeros(num_segments, 100);

for i = 1:num_segments
    for k = 1:100
        lhom_normal(i, k) = log10(abs(mean(segments_normal(:, i).^k)) + eps);
        lhom_disease(i, k) = log10(abs(mean(segments_disease(:, i).^k)) + eps);
    end
end

%% Combine and label
features = [lhom_normal; lhom_disease];                 % (10000 × 100)
labels = [zeros(num_segments,1); ones(num_segments,1)]; % (10000 × 1)
features_with_labels = [features, labels];              % (10000 × 101)

%% Save to CSV
writematrix(features_with_labels, 'lhom_features_10000.csv');
disp('✅ Saved to lhom_features_10000.csv');


%% ------ 2nd step paste and run this on MATLAB
clc; clear; close all;

%% Load features
data = readmatrix('lhom_features_10000.csv');
X = data(:, 1:100);
Y = data(:, 101);

%% Train-test split
cv = cvpartition(Y, 'HoldOut', 0.2);
X_train = X(training(cv), :);
Y_train = Y(training(cv));
X_test = X(test(cv), :);
Y_test = Y(test(cv));

%% SVM
svmModel = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf', 'Standardize', true);
Y_pred_svm = predict(svmModel, X_test);
acc_svm = mean(Y_pred_svm == Y_test) * 100;

%% KNN
knnModel = fitcknn(X_train, Y_train, 'NumNeighbors', 5);
Y_pred_knn = predict(knnModel, X_test);
acc_knn = mean(Y_pred_knn == Y_test) * 100;

%% Random Forest
rfModel = TreeBagger(100, X_train, Y_train, 'Method', 'classification');
Y_pred_rf = str2double(predict(rfModel, X_test));
acc_rf = mean(Y_pred_rf == Y_test) * 100;

%% Show results
fprintf('\nClassification Accuracies:\n');
fprintf('SVM Accuracy        : %.2f%%\n', acc_svm);
fprintf('KNN Accuracy        : %.2f%%\n', acc_knn);
fprintf('Random Forest Acc   : %.2f%%\n', acc_rf);

%% Confusion Matrix
figure;
subplot(1,3,1);
confusionchart(Y_test, Y_pred_svm); title('SVM');

subplot(1,3,2);
confusionchart(Y_test, Y_pred_knn); title('KNN');

subplot(1,3,3);
confusionchart(Y_test, Y_pred_rf); title('Random Forest');
