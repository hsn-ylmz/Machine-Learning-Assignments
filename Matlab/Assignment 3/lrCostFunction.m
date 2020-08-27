function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 


m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h = 1 / (1 + exp(X*theta)); %Create hypothesis

%Compute cost of theta
J = 1/m * sum(-y.*log(h) -(1-y).*log(1-h)) + ((lambda / (2*m)) *( sum ((theta(2:end)).^2)));

%Apply gradiation descent method to minimize cost
grad = 1/m .* (X'*(h - y));
%Do not aplly regularization on first weight (Q0(theta zero))
grad(2:end) = grad(2:end) + (lambda / m) .* theta(2:end);


grad = grad(:);

end
