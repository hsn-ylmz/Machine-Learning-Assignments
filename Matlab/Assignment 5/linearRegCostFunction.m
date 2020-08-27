function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h = X*theta; %Create hypothesis

%Compute cost of theta
J =(1/(2*m)) * sum((h-y).^2) + ((lambda / (2*m)) *( sum ((theta(2:end)).^2)));

%Apply gradiation descent method to minimize cost
grad = (1/m) .* (X'*(h - y));
%Do not aplly regularization on first weight (Q0(theta zero))
grad(2:end) = grad(2:end) + (lambda / m) .* theta(2:end);

grad = grad(:);

end
