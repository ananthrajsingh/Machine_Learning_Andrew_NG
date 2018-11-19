function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELLO FROM ANANTH!

C_vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
error = 100000;

% Below for loops make sure we go through every combintion of C and sigma
for i = 1:size(C_vals)
	for j = 1:size(sigma_vals)
		% We will train the model using current C and sigma
		model= svmTrain(X, y, C_vals(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vals(j))); 

		% Okay we have the model
		% Let us do the checking part

		% Here we will get the predictions on Xval
		predictions = svmPredict(model, Xval);

		% calculate the error
		current_error = mean(double(predictions ~= yval));

		if (current_error < error)
			error = current_error;
			C = C_vals(i);
			sigma = sigma_vals(j);
		endif
	endfor
endfor			





% =========================================================================

end
