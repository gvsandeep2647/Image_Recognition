package artificial.neural.networks;

/**
 * This class encapsulates the methods needed to run an MLP
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
public class Runner {
    
    MLP M;
    /**
     * Constructs a runner with the given MLP
     * @param m the MLP to be run
     */
    Runner(MLP m)
    {
        M=m;
    }
    /**
     * Runs a single instance of input through the MLP
     * @param inputs a row matrix having the input values
     * @param onehot decides whether the output is to be rounded to high and low or not
     * @return returns the output matrix
     */
    Matrix run(Matrix inputs, boolean onehot)//inputs vectors are arranged rowwise.
    {   //inputs is row matrix of input. Result is a row matrix of predicted output. If onehot is true, the output vector is rounded off to the low/high values.
        Matrix O1=inputs.matmult(M.weights1).add(M.bias1).sigmoid();//Output of the hidden layer
        Matrix OF=O1.matmult(M.weights2).add(M.bias2).sigmoid();//The final outputs
        if(!onehot)
        {
            return OF;
        }
        else
        {
            return OF.onehot(M.high, M.low);
        }  
        
    }
    /**
     * Tests the given data on the MLP and returns the accuracy produced
     * @param inputs the matrix of the input vectors
     * @param outputs the matrix of the expected output vectors
     * @return the accuracy percentage
     */
    double test(Matrix inputs, Matrix outputs)//returns the percent of  correct values predicted
    {
        
        int score=0;
        for(int i=0;i<inputs.rownum();i++)
        {
            Matrix result=run(inputs.getrow(i),true);
            if(result.equals(outputs.getrow(i)))score++;
            else System.out.println("Misclassified example: "+i);
        }
        return ((double)score/(double)inputs.rownum())*100.0;
    }
}
