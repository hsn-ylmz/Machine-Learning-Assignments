function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)


m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);


X = [ones(m, 1) X]; % Add X0 to X matrix
a2 = sigmoid(X*Theta1'); %Input layer of neural network
a2 = [ones(m, 1) a2]; %Add a0 to a2 matrix
t = sigmoid(a2*Theta2'); %output layer (hypothesis)

for r=1:m;
[M ,p(r)] = max(t(r,:));  %selecting best possibility for every individual
end



end
