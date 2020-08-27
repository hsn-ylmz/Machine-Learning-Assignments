function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters


    h = theta' * X';  % Creates a hypothesis for variables(X)and initial weights(theta) 

    theta = theta - (alpha*(1/m).*sum((h'-y).*X))'; %Applying gradient descent for theta


    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
