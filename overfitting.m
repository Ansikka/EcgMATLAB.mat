%% Load Data
data = readmatrix('lhom_features_10000.csv');
X = data(:, 1:100);
Y = data(:, 101);

%% Define training percentages
train_ratios = [0.1 0.3 0.5 0.7 0.9];
n = length(Y);

%% Preallocate for plotting
train_acc_svm = zeros(size(train_ratios));
test_acc_svm  = zeros(size(train_ratios));

train_acc_knn = zeros(size(train_ratios));
test_acc_knn  = zeros(size(train_ratios));

train_acc_rf = zeros(size(train_ratios));
test_acc_rf  = zeros(size(train_ratios));

for i = 1:length(train_ratios)
    % Split dataset
    p = train_ratios(i);
    cv = cvpartition(Y, 'HoldOut', 1 - p);
    
    X_train = X(training(cv), :);
    Y_train = Y(training(cv));
    X_test  = X(test(cv), :);
    Y_test  = Y(test(cv));

    %% Train SVM
    svmModel = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf', 'Standardize', true);
    train_acc_svm(i) = mean(predict(svmModel, X_train) == Y_train) * 100;
    test_acc_svm(i)  = mean(predict(svmModel, X_test)  == Y_test)  * 100;

    %% Train KNN
    knnModel = fitcknn(X_train, Y_train, 'NumNeighbors', 5);
    train_acc_knn(i) = mean(predict(knnModel, X_train) == Y_train) * 100;
    test_acc_knn(i)  = mean(predict(knnModel, X_test)  == Y_test)  * 100;

    %% Train Random Forest
    rfModel = TreeBagger(100, X_train, Y_train, 'Method', 'classification');
    train_acc_rf(i) = mean(str2double(predict(rfModel, X_train)) == Y_train) * 100;
    test_acc_rf(i)  = mean(str2double(predict(rfModel, X_test))  == Y_test)  * 100;
end

%% Plot overfitting
figure;
subplot(1,3,1);
plot(train_ratios*100, train_acc_svm, '-o', 'LineWidth', 2); hold on;
plot(train_ratios*100, test_acc_svm, '-s', 'LineWidth', 2);
xlabel('Training Size (%)'); ylabel('Accuracy (%)');
title('SVM Overfitting'); legend('Train','Test'); grid on;

subplot(1,3,2);
plot(train_ratios*100, train_acc_knn, '-o', 'LineWidth', 2); hold on;
plot(train_ratios*100, test_acc_knn, '-s', 'LineWidth', 2);
xlabel('Training Size (%)'); ylabel('Accuracy (%)');
title('KNN Overfitting'); legend('Train','Test'); grid on;

subplot(1,3,3);
plot(train_ratios*100, train_acc_rf, '-o', 'LineWidth', 2); hold on;
plot(train_ratios*100, test_acc_rf, '-s', 'LineWidth', 2);
xlabel('Training Size (%)'); ylabel('Accuracy (%)');
title('Random Forest Overfitting'); legend('Train','Test'); grid on;
