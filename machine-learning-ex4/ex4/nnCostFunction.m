function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELLO FROM ANANTH!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% y_matrix has the matrix size 5000 x 10 since we have 5000 examples 
% and 10 output units (num_labels)
y_matrix = eye(num_labels)(y,:);

% Adding a column os 1s to X, so new dimension = 5000 x 401
a1 = [ones(size(X,1), 1) X];

% Now we need to calculate z2, which will be calculated by the product of a1 and Theta1
% z2 will be a matrix of dimension 5000 x 25
z2 = a1 * Theta1';

% ' Now we will calculate a2, which will be z2 passed through sigmoid function
% a2 will also have a column of 1s added for bias units then it will have size 5000 x 26
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];

% Calculating z3 now, z3 will be the product of a2 and Theta2, it will be 5000 x 10
z3 = a2 * Theta2';

%'
a3 = sigmoid(z3);


A = -y_matrix .* log(a3) - (1 - y_matrix) .* log(1 - a3);
% below we are doing the work of both summation symbol in the formula
B = sum(A(:));
% C = B/m;

% Now we will practice regularization
%First we will remove the first column from Theta1 and Theta2

%Theta1(:, [1]) = [];
%Theta2(:, [1]) = [];

% Now let us square each term of the above two
Theta1sq = Theta1(:, 2:end).^2;
Theta2sq = Theta2(:, 2:end).^2;

% Add all the terms of both matrices
sum1 = sum(Theta1sq(:));
sum2 = sum(Theta2sq(:));

% Final calculation of cost function

C = (lambda*(sum1 + sum2))/2;
J = (B + C)/m;






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pan through all examples , here m = 5000 cuz 5000 x 400
% Adding a column os 1s to X, so new dimension = 5000 x 401
X = [ones(size(X,1), 1) X];
%Theta1 = [ones(size(Theta1, 1), 1) Theta1];
%Theta2 = [ones(size(Theta2, 1), 1) Theta2

DELTA1 = zeros(hidden_layer_size, input_layer_size + 1);
DELTA2 = zeros(num_labels, hidden_layer_size + 1);
for i = 1:m
	% we have to feed ith row to input layer this will be a vector with 401 x 1
	a_1 = X(i,:)';
	%' perform feedforward

	z_2 = Theta1 * a_1; % result is 25 x 1
	a_2 = sigmoid(z_2)'; %' since we have to add a column therfore transpose
	a_2 = [ones(size(a_2,1), 1) a_2]';  %' 26 x 1

	z_3 = Theta2 * a_2; % 10 x 1
	a_3 = sigmoid(z_3);

	% Time to find error in final layer and storing it delta_3
	% Below code will arrange the error from each output unit columnwise
	% Therefore, at the end of for loop we will have del2 to be 10 x 5000
	% we need to extract the first row from y_matrix and take its transpose
	temp = y_matrix(i, :)'; %' This is 10 x 1 matrix
	% delta_3(:, i) = a_3 - temp;
	delta_3 = a_3 - temp;

	delta_2 = (Theta2(:, 2:end))' * delta_3 .* sigmoidGradient(z_2); %' 25 x 1

	temp_1 = (delta_2 * a_1');
	temp_2 = (delta_3 * a_2');


	DELTA1 = DELTA1 + temp_1;
	DELTA2 = DELTA2 + temp_2;
endfor



Theta1_grad = DELTA1/m;
Theta2_grad = DELTA2/m;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
