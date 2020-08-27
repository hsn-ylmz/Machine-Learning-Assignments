function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);


p = sigmoid(X*theta); %create hypothesis with sigmoid function
p(p>=0.5)=1; %Return true if result of the hypothesis greater than 0.5
p(p<=0.5)=0; %Else return false



end
