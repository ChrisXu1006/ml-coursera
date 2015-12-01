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

%  Compute J without regularation

X = [ones(m, 1) X];
A1 = sigmoid(X * Theta1');
A1 = [ones(size(A1, 1), 1) A1];
A2 = sigmoid(A1 * Theta2');

for i = 1:m,
	yvec = zeros(num_labels,1);
	yvec(y(i)) = 1;
	xvec = A2(i,:);
	J = J + (log(xvec)*(-yvec)-log(1-xvec)*(1-yvec));
end
J = J/m;

% with Regularation
Theta1_prime = Theta1(:,2:end);
Theta2_prime = Theta2(:,2:end);
J = J + (sum(sum(Theta1_prime.^2)) + sum(sum(Theta2_prime.^2))) * (lambda/(2*m));

% Backpropagation

for i = 1:m,
	% step1
	a1 = X(i,:);
	a1 = a1';
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1 ; a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	% step2
	yvec = zeros(num_labels,1);
	yvec(y(i)) = 1;
	delta3 = a3 - yvec;
	%% step3
	delta2 = (Theta2'*delta3).*[1;sigmoidGradient(z2)];
	%% step4
	delta2 = delta2(2:end);
	Theta2_grad = Theta2_grad + delta3*a2';
	Theta1_grad = Theta1_grad + delta2*a1';
end
	% step5
	Theta2_grad = Theta2_grad/m;
	Theta1_grad = Theta1_grad/m;

	regular1 = [zeros(size(Theta1,1),1) Theta1(:,2:end)];
	regular2 = [zeros(size(Theta2,1),1) Theta2(:,2:end)];
	Theta2_grad = Theta2_grad + (lambda/m)*regular2;
	Theta1_grad = Theta1_grad + (lambda/m)*regular1;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
