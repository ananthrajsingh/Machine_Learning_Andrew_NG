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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELLO FROM ANANTH!

% Okay let us work out to find the cost of this bitch

% X has 1 feature for every example therefore its dimension 12 x 2, including one bias
% Hence size of y 12 x 1
% We have one parameter only therefore theta has size 2 x 1

% Let us compute h(theta)
h_theta = X * theta;

squared_diff_summation = sum((h_theta - y).^2);

%This is important as we start from second parameter
theta(1) = 0;

theta_squared_summation = sum(theta.^2);

J = (squared_diff_summation + lambda * theta_squared_summation)/(2 * m);

%%%%%%%%%%%%%%%%%%%%%%%
% Calculting gradient
%%%%%%%%%%%%%%%%%%%%%%%

diff = (h_theta - y); % 12x1
first_term = X' * diff; %'
grad = (first_term + lambda * theta)/m;















% =========================================================================

grad = grad(:);

end
