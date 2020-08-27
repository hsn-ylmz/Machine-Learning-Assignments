function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

param = [0.01 0.03 0.1 0.3 1 3 10 30];
err = 1;
for i = 1 : length(param);
    for j = 1: length(param);
        C = param(i);
        sigma = param(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        pred = svmPredict(model, Xval);
        error = mean(double(pred ~= yval));
        if error > 0 & error < err;
            err = error;
            c = C;
            s = sigma;
        end
    end
end
C = c;
sigma = s;
        

end
