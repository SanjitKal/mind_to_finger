function [features] = get_features_release(clean_data,fs)
    %
    % get_features_release.m
    %
    % Instructions: Write a function to calculate features.
    %               Please create 4 OR MORE different features for each channel.
    %               Some of these features can be of the same type (for example, 
    %               power in different frequency bands, etc) but you should
    %               have at least 2 different types of features as well
    %               (Such as frequency dependent, signal morphology, etc.)
    %               Feel free to use features you have seen before in this
    %               class, features that have been used in the literature
    %               for similar problems, or design your own!
    %
    % Input:    clean_data: (samples x channels)
    %           fs:         sampling frequency
    %
    % Output:   features:   (1 x (channels*features))
    % 
%% Your code here (8 points)
    num_channels = length(clean_data(1,:));
    num_features = 4;
    feature_1 = zeros(1,num_channels); %spike counts
    feature_2 = zeros(1,num_channels); %bandpower in 60-80hz range
    feature_3 = zeros(1,num_channels); %bandpower in 80-100hz range
    feature_4 = zeros(1,num_channels); %bandpower in 40-60hz range
    for i = 1:num_channels
        feature_1(1,i) = length(findpeaks(clean_data(:,i)));
        feature_2(1,i) = bandpower(clean_data(:,i),fs,[60,80]);
        feature_3(1,i) = bandpower(clean_data(:,i),fs,[80,100]);
        feature_4(1,i) = bandpower(clean_data(:,i),fs,[40,60]);
    end
    

    features = [feature_1 feature_2 feature_3 feature_4];
    
end

