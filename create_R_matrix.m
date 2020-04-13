function [R]=create_R_matrix_release(features, N_wind)
    %
    % get_features_release.m
    %
    % Instructions: Write a function to calculate R matrix.             
    %
    % Input:    features:   (samples x (channels*features))
    %           N_wind:     Number of windows to use
    %
    % Output:   R:          (samples x (N_wind*channels*features))
    % 
%% Your code here (5 points)
num_samples = length(features(:,1));
num_ch_x_feats = length(features(1,:));
rep_features = zeros(num_samples + N_wind - 1, num_ch_x_feats);
rep_features(1:N_wind-1,:) = features(1:N_wind-1,:);
rep_features(N_wind:num_samples + N_wind - 1,:) = features;
R = zeros(num_samples, 1 + N_wind*num_ch_x_feats);
R(:,1) = ones(num_samples,1);
base = 0;
for i = 1:num_samples
    R_row = [];
   for j = 1:num_ch_x_feats
       R_row = [R_row rep_features(base + 1:base + N_wind,j).'];
   end
   R(i,2:num_ch_x_feats*N_wind+1) = R_row;
base = base + 1;
end
end