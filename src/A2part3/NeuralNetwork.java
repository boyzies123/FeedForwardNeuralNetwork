package A2part3;
import java.util.Arrays;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;
    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;
    private final double [] hidden_layer_bias = {-0.02, -0.20};
    private final double [] output_layer_bias = {-0.33, 0.26, 0.06};
    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights, double[][] initial_output_layer_weights, double learning_rate) {
        //Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;
        
        this.learning_rate = learning_rate;
    }


    //Calculate neuron activation for an input
    public double sigmoid(double input) {
        double output = 1/(1+Math.exp(-input)); 
        return output;
    }

    //Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {
    	double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
        	double weighted_sum = 0;
            for (int a = 0; a < num_inputs; a++) {
            	weighted_sum = weighted_sum + inputs[a] * hidden_layer_weights[a][i];
            }
            //add bias
        	weighted_sum = weighted_sum + hidden_layer_bias[i];
            double output = sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            double weighted_sum = 0;
            for (int a = 0; a < num_hidden; a++) {
            	weighted_sum = weighted_sum + hidden_layer_outputs[a] * output_layer_weights[a][i];
          
            }
            //add bias
            weighted_sum = weighted_sum + output_layer_bias[i];
            double output = sigmoid(weighted_sum);
            
            output_layer_outputs[i] = output;
          
        }
       
        
        return new double[][]{hidden_layer_outputs, output_layer_outputs};
    }

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs,
                                                 double[] output_layer_outputs, int desired_outputs) {
        double[] output_layer_betas = new double[num_outputs];
        int [] encoded = new int [3];
        if (desired_outputs == 0) {
        	encoded[0] = 1;
        	encoded[1] = 0;
        	encoded[2] = 0;
        }
        else if (desired_outputs == 1) {
        	encoded[0] = 0;
        	encoded[1] = 1;
        	encoded[2] = 0;
        }
        else {
        	encoded[0] = 0;
        	encoded[1] = 0;
        	encoded[2] = 1;
        }
        for (int i = 0; i < num_outputs; i++) {
        	output_layer_betas[i] = (encoded[i]- output_layer_outputs[i]);
        	//update output layer bias (times learning rate by gradient https://www.researchgate.net/post/How_do_Neural_Networks_update_weights_and_Biases_during_Back_Propagation)
        	output_layer_bias[i] = output_layer_bias[i] + (learning_rate*output_layer_outputs[i] * (1 - output_layer_outputs[i]) * output_layer_betas[i]);
        }
        
        double[] hidden_layer_betas = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
        	double sum = 0;
        	for (int a = 0; a < num_outputs; a++) {
        		sum = sum + (output_layer_weights[i][a]*output_layer_outputs[a]*(1-output_layer_outputs[a])*output_layer_betas[a]);
        	}
 
        	hidden_layer_betas[i] = sum;

        }


        // This is a HxO array (H hidden nodes, O outputs)
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];
        for (int i = 0; i < num_hidden; i++) {
        	for (int a = 0; a < num_outputs; a++) {
        		delta_output_layer_weights[i][a] = (learning_rate) * output_layer_betas[a] * hidden_layer_outputs[i]*output_layer_outputs[a]*(1-output_layer_outputs[a]);
        	}
        }
        // This is a IxH array (I inputs, H hidden nodes)
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
        
        for (int i = 0; i < num_inputs; i++) {
        	for (int a = 0; a < num_hidden; a++) {
        		delta_hidden_layer_weights[i][a] = learning_rate * hidden_layer_betas[a] * inputs[i] *hidden_layer_outputs[a]*(1-hidden_layer_outputs[a]);
        	}
        }
        //update hidden layer bias
        for (int i = 0; i < num_hidden; i++) {
			hidden_layer_bias[i] = hidden_layer_bias[i] + (learning_rate* hidden_layer_betas[i] * hidden_layer_outputs[i] * (1 - hidden_layer_outputs[i]));
		}
        // Return the weights we calculated, so they can be used to update all the weights.
        return new double[][][]{delta_output_layer_weights, delta_hidden_layer_weights};
    }

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights) {
    	for (int i = 0; i < num_hidden; i++) {
        	for (int a = 0; a < num_outputs; a++) {
        		double originalWeight = output_layer_weights[i][a];
        		output_layer_weights[i][a] = delta_output_layer_weights[i][a] + originalWeight;
        	}
        }
    	for (int i = 0; i < num_inputs; i++) {
        	for (int a = 0; a < num_hidden; a++) {
        		double originalWeight = hidden_layer_weights[i][a];
        		hidden_layer_weights[i][a] = delta_hidden_layer_weights[i][a] + originalWeight;
        	}
        }
    	
        
    }

    public void train(double[][] instances, int[] desired_outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("epoch = " + epoch);
            int[] predictions = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i]);
                int predicted_class = predict(instances)[i]; // TODO!
                predictions[i] = predicted_class;
                
                //We use online learning, i.e. update the weights after every instance.
                update_weights(delta_weights[0], delta_weights[1]);
            }

            // Print new weights
            System.out.println("Hidden layer weights \n" + Arrays.deepToString(hidden_layer_weights));
            System.out.println("Output layer weights  \n" + Arrays.deepToString(output_layer_weights));

            int correctClasses = 0;
            for (int i = 0; i < instances.length; i++) {
       
                if (predictions[i] == desired_outputs[i]) {
                	correctClasses++;
                }
            }
            double acc = (double)correctClasses/instances.length;
            System.out.println("acc = " + acc);
        }
    }

    public int[] predict(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            
            int predicted_class = -1;  
            double max = Double.MIN_VALUE;
            for (int a = 0; a < num_outputs; a++) {
            	if (outputs[1][a] > max) {
            		max = outputs[1][a];
            		predicted_class = a;
            	}
            }
            predictions[i] = predicted_class;
        }
   
        return predictions;
    }

}
