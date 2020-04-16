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
clean_data = lowpass(raw_eeg, 30, 1000);
for i = 1:length(clean_data(:,1))
    ch_avg = mean(clean_data(i,:));
    ch_std = std(clean_data(i,:));
    clean_data(i,:) = arrayfun(@(x) (x - ch_avg)/ch_std, clean_data(i,:));
end
