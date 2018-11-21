function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4N4NTH HERE

% Going over every centroid
for i = 1:K
	% Let us see which rows in idx have current centroid.
	% In other words, let us see what all values in X are assigned to 
	% cluster with current centroid
	indices = find(idx == i);

	% Now, get the values of all the assigned X points, that we have in
	% indices
	X_current_centroid = X(indices, :);

	% Since we have to take the average, we need to know how 
	% many points are assigned to current centroid
	num_of_points = size(X_current_centroid, 1);

	% Taking sum of all points and dividing by number of points
	% to obtain average
	new_centroid = sum(X_current_centroid)./num_of_points

	centroids(i, :) = new_centroid;
endfor	








% =============================================================


end

