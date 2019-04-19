#pragma once

#include <vector>

class NeuralNetwork {
public:
	NeuralNetwork();
	~NeuralNetwork();

	virtual Tensor forward(Tensor inputs) = 0;
	virtual Tensor backward(Tensor outputs) = 0;
};