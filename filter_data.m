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
clean_data = zeros(length(raw_eeg(:,1)), length(raw_eeg(1,:)));
for i = 1:length(raw_eeg(:,1))
    ch_avg = mean(raw_eeg(i,:));
    clean_data(i,:) = arrayfun(@(x) x - ch_avg, raw_eeg(i,:));
end
clean_data = lowpass(clean_data, 30, 1000);
