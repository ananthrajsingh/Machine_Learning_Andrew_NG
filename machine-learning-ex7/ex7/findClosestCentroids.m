function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4N4NTH HERE!

% Currently we have size of X as 300 x 2 , that is, 2-D

% We will iterate over every present value of x
for i = 1:size(idx, 1)

	% Let us calculate distance of current example from all 
	% centroids

	current_x = X(i, :);

	% Distance between current point and every centroid
	distance = centroids - current_x;

	% We need squared distance
	% Also we need to perform columnwise sum, so I am taking transpose to
	% turn columns into rows because sum() works row-wise
	% i am just being a bit lazy on this
	distance = (distance.^2)';

	%' Taking sum of all the distance
	% This will give squared distance between the current point
	% and all the centroids.
	% We will choose the centroid which will have minimum distance
	distance_sum = sum(distance);

	[min_distance_val, min_val_column] = min(distance_sum);

	idx(i) = min_val_column;

endfor	










% =============================================================

end

