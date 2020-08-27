function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h = 1 ./ (1+(exp(-(theta')*X')));  % Create hypothesis with data and initial theta

J = 1/m * sum(-y.*log(h') -(1-y).*log(1-h'));  %Calculate the cost of hypothesis

grad = 1/m .* ((h - y')*X);   %Aplly gradient descent method for hypothesis


end
