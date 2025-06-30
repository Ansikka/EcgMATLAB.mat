%%%%---------- 70% Training & 30% Testing -------------%%%%

%% Load Data
data = readmatrix('lhom_features_10000.csv');
X = data(:, 1:100);
Y = data(:, 101);

%% 70/30 Split
cv = cvpartition(Y, 'HoldOut', 0.3);
X_train = X(training(cv), :);
Y_train = Y(training(cv));
X_test  = X(test(cv), :);
Y_test  = Y(test(cv));

%% Train Models
svmModel = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf', 'Standardize', true);
knnModel = fitcknn(X_train, Y_train, 'NumNeighbors', 5);
rfModel  = TreeBagger(100, X_train, Y_train, 'Method', 'classification');

%% Predict
Y_pred_svm = predict(svmModel, X_test);
Y_pred_knn = predict(knnModel, X_test);
Y_pred_rf  = str2double(predict(rfModel, X_test));

%% Accuracy
acc_svm = mean(Y_pred_svm == Y_test) * 100;
acc_knn = mean(Y_pred_knn == Y_test) * 100;
acc_rf  = mean(Y_pred_rf == Y_test) * 100;

%% Results
fprintf('70/30 Split:\n');
fprintf('SVM Accuracy       : %.2f%%\n', acc_svm);
fprintf('KNN Accuracy       : %.2f%%\n', acc_knn);
fprintf('Random Forest Acc  : %.2f%%\n', acc_rf);

                                                                                ______________RESULT 70/30 Split: _______________
SVM Accuracy       : 100.00%
KNN Accuracy       : 99.70%
Random Forest Acc  : 100.00%



                                            %%%%%%%%%%%%-----------------This doesn’t use a fixed 70:30 split, but instead rotates test/train folds--------------------%%%%%%%%%%%%
%% Load Data
data = readmatrix('lhom_features_10000.csv');
X = data(:, 1:100);
Y = data(:, 101);

k = 5;   % number of folds
cv = cvpartition(Y, 'KFold', k);

acc_svm = zeros(k,1);
acc_knn = zeros(k,1);
acc_rf  = zeros(k,1);

for i = 1:k
    trainIdx = training(cv, i);
    testIdx = test(cv, i);

    X_train = X(trainIdx, :);
    Y_train = Y(trainIdx);
    X_test  = X(testIdx, :);
    Y_test  = Y(testIdx);

    % Train
    svmModel = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf', 'Standardize', true);
    knnModel = fitcknn(X_train, Y_train, 'NumNeighbors', 5);
    rfModel  = TreeBagger(100, X_train, Y_train, 'Method', 'classification');

    % Predict
    acc_svm(i) = mean(predict(svmModel, X_test) == Y_test) * 100;
    acc_knn(i) = mean(predict(knnModel, X_test) == Y_test) * 100;
    acc_rf(i)  = mean(str2double(predict(rfModel, X_test)) == Y_test) * 100;
end

% Average results
fprintf('5-Fold Cross Validation:\n');
fprintf('Average SVM Accuracy       : %.2f%%\n', mean(acc_svm));
fprintf('Average KNN Accuracy       : %.2f%%\n', mean(acc_knn));
fprintf('Average Random Forest Acc  : %.2f%%\n', mean(acc_rf));

                                                            _________________    RESULT 5-Fold Cross Validation:        _____________________
Average SVM Accuracy       : 100.00%
Average KNN Accuracy       : 99.89%
Average Random Forest Acc  : 99.95%
>> 
