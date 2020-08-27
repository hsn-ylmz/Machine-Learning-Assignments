function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1.


X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu(:,2:3) = mean(X);  %Calculate the mean value of X
sigma(:,2:3) = std(X); %Calculate standart deviation of X
X_norm = (X-mu(:,2:3))./sigma(:,2:3); %Normalizing of X


end
