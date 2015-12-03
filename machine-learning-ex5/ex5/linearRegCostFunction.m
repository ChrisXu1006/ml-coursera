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

z = X * theta;
z = z - y;
for i = 1:m,
	z(i,1) = z(i,1)*z(i,1);
end

theta_prime = theta.^2;
theta_prime(1,1) = 0;
J = sum(z)/(2*m) + sum(theta_prime)*lambda/(2*m);

z = X * theta - y;
theta_prime = [0; theta(2:end,:)];
grad = (X' * z / m) + (lambda * theta_prime /m);

% =========================================================================

grad = grad(:);

end
