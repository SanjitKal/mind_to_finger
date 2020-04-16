function [all_feats]=getWindowedFeats_release(raw_data, fs, window_length, window_overlap)
    %
    % getWindowedFeats_release.m
    %
    % Instructions: Write a function which processes data through the steps
    %               of filtering, feature calculation, creation of R matrix
    %               and returns features.
    %
    %               Points will be awarded for completing each step
    %               appropriately (note that if one of the functions you call
    %               within this script returns a bad output you won't be double
    %               penalized)
    %
    %               Note that you will need to run the filter_data and
    %               get_features functions within this script. We also 
    %               recommend applying the create_R_matrix function here
    %               too.
    %
    % Inputs:   raw_data:       The raw data for all patients
    %           fs:             The raw sampling frequency
    %           window_length:  The length of window
    %           window_overlap: The overlap in window
    %
    % Output:   all_feats:      All calculated features
    %
%% Your code here (3 points)

% First, filter the raw data
clean_data = filter_data(raw_data);
% number of total samples per channel
num_samples = length(clean_data(:,1));
% number of samples in window
num_per_window = window_length*fs;
% number of samples in overlap
num_per_overlap = window_overlap*fs;
% number of windows
num_windows = (num_samples/num_per_window)*(num_per_window/num_per_overlap) - 1;
num_features = 6;
num_channels = length(clean_data(1,:));
all_feats = zeros(num_windows, num_features*num_channels);

% Then, loop through sliding windows
curr = 0;
for i = 1:num_windows
    % Within loop calculate feature for each segment (call get_features)
    new_feature_row = get_features(clean_data(curr + 1:curr+num_per_window,:), fs);
    all_feats(i,:) = new_feature_row;
    curr = curr + num_per_overlap;
end

% Finally, return feature matrix

end