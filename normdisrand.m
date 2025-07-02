%% Configuration
num_signals = 5;
orders = 1:100;
noise_level = 0.05;
rng(1);  % For reproducibility

% Color scheme for up to 5 signals
colors = lines(num_signals);

% Initialize arrays
logmom_norm_all = zeros(num_signals, length(orders));
logmom_dis_all  = zeros(num_signals, length(orders));
logmom_rand_all = zeros(num_signals, length(orders));

%% Loop over each signal
for i = 1:num_signals
    %% Load and preprocess Normal ECG
    load(['ecg_normal' num2str(i) '.mat']);  % variable: signal
    signal_normal = signal + noise_level * randn(size(signal));
    sig_norm = (signal_normal - mean(signal_normal)) / std(signal_normal);

    %% Load and preprocess Diseased ECG
    load(['ecg_disease' num2str(i) '.mat']);
    signal_disease = signal + noise_level * randn(size(signal));
    sig_dis = (signal_disease - mean(signal_disease)) / std(signal_disease);

    %% Generate and preprocess Random Noise
    signal_random = randn(size(signal));
    sig_rand = (signal_random - mean(signal_random)) / std(signal_random);

    %% Compute log moments (not central moments)
    for j = 1:length(orders)
        k = orders(j);
        logmom_norm_all(i,j) = log10(abs(mean(sig_norm.^k)) + eps);
        logmom_dis_all(i,j)  = log10(abs(mean(sig_dis.^k)) + eps);
        logmom_rand_all(i,j) = log10(abs(mean(sig_rand.^k)) + eps);
    end
end

%% Plot: Log Moments (1st to 100th order)
figure;
for i = 1:num_signals
    plot(orders, logmom_norm_all(i,:), '-', 'Color', colors(i,:), 'LineWidth', 1.5); hold on;
    plot(orders, logmom_dis_all(i,:), '--', 'Color', colors(i,:), 'LineWidth', 1.5);
    plot(orders, logmom_rand_all(i,:), ':', 'Color', colors(i,:), 'LineWidth', 1.5);
end

xlabel('Order of Moment (k)');
ylabel('log_{10}(Moment)');
title('Log Moments (1st to 100th Order): Normal vs Diseased vs Random Noise');
legend([compose("Normal %d", 1:5), compose("Diseased %d", 1:5), compose("Noise %d", 1:5)], 'Location', 'bestoutside');
grid on; axis tight;
