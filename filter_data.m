function clean_data = filter_data_release(raw_eeg)
    %
    % filter_data_release.m
    %
    % Instructions: Write a filter function to clean underlying data.
    %               The filter type and parameters are up to you.
    %               Points will be awarded for reasonable filter type,
    %               parameters, and correct application. Please note there 
    %               are many acceptable answers, but make sure you aren't 
    %               throwing out crucial data or adversely distorting the 
    %               underlying data!
    %
    % Input:    raw_eeg (samples x channels)
    %
    % Output:   clean_adata (samples x channels)
    % 
%% Your code here (2 points)
% clean_data = highpass(raw_eeg, 40, 1000);
% clean_data = zeros(length(raw_eeg(:,1)), length(raw_eeg(1,:)));
% avg = mean(mean(raw_eeg));
% for i = 1:length(clean_data(:,1))
%     clean_data(i,:) = arrayfun(@(x) (x - avg), raw_eeg(i,:));
% end
clean_data = raw_eeg;
%clean_data = lowpass(raw_eeg, 175, 1000);
