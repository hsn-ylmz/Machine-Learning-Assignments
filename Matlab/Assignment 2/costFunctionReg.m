function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h = sigmoid(theta' * X');   % Create hypothesis with data and initial theta

%Calculate the cost of hypothesis
J = 1/m * sum(-y.*log(h') -(1-y).*log(1-h')) + ((lambda / (2*m)) *( sum ((theta(2:end)).^2)));

%Aplly gradient descent method for hypothesis
grad = 1/m .* ((h - y')*X);
%Put the regularization factor to method except first weight.
grad(2:end) = grad(2:end) + lambda / m .* (theta(2:end))';  


end
