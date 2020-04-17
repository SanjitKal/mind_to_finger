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
    feature_1 = zeros(1,num_channels); %average time-domain voltage (LMP)
    feature_2 = zeros(1,num_channels);
    feature_3= zeros(1,num_channels);
    feature_4 = zeros(1,num_channels); 
    feature_5 = zeros(1,num_channels);
    feature_6 = zeros(1,num_channels);
    for i = 1:num_channels
        feature_1(1,i) = mean(clean_data(:,i));
        feature_2(1,i) = bandpower(clean_data(:,i),fs,[8,12]);
        feature_3(1,i) = bandpower(clean_data(:,i),fs,[18,24]);
        feature_4(1,i) = bandpower(clean_data(:,i),fs,[75,115]);
        feature_5(1,i) = bandpower(clean_data(:,i),fs,[125,160]);
        feature_6(1,i) = bandpower(clean_data(:,i),fs,[160,175]);
    end
    

    features = [feature_1 feature_2 feature_3 feature_4 feature_5 feature_6];
    
end

