package artificial.neural.networks;
public class Runner {
    
    MLP M;
    Runner(MLP m)
    {
        M=m;
    }
    
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
    double test(Matrix inputs, Matrix outputs)//returns the percent of  correct values predicted
    {
        
        int score=0;
        for(int i=0;i<inputs.rownum();i++)
        {
            Matrix result=run(inputs.getrow(i),true);
            if(result.equals(outputs.getrow(i)))score++;
        }
        return ((double)score/(double)inputs.rownum())*100.0;
    }
}
