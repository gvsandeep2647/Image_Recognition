package artificial.neural.networks;
import java.io.*;
import java.util.*;

/**
 * A class that encapsulates all the training operations
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
public class Trainer {
    MLP M;
    /**
     * Constructor that initializes the trainer with an MLP object
     * @param m the MLP object to be trained
     */
    Trainer(MLP m)
    {
        M=m;
    }
    /**
     * returns the pointwise gradient of the sigmoid function vector given the output of the sigmoid function
     * @param m the output of a sigmoid function
     * @return the gradient vector
     */
    Matrix getGrad(Matrix m)//given a matrix of sigmoid values, calculates the derivative of sigmoid at each point
    {
        Matrix mat=Matrix.zeros(m.rownum(), m.colnum());
        for(int i=0;i<m.rownum();i++)
        {
            for(int j=0;j<m.colnum();j++)
            {
                mat.set(i, j, m.get(i, j)*(1-m.get(i, j)));
            }
        }
        return mat;
    }
    /**
     * Trains the MLP of this Trainer object using stochastic gradient descent
     * @param inputs The training input matrix
     * @param outputs The training output matrix
     * @param rate The learning rate
     * @param momentum The momentum
     * @param epochs The number of epochs
     */
    void stochGradDesc(Matrix inputs, Matrix outputs, double rate, double momentum, int epochs)
    {
        if(inputs.colnum()!=M.inputs||outputs.colnum()!=M.outputs)
        {
            System.err.println("Data incompatible with MLP");
            System.exit(1);
        }
        Runner R=new Runner(M);
        List<Integer> order=new ArrayList<>();
        for(int i=0;i<inputs.rownum();i++)
        {
            order.add(i);
        }
        //To train in a random order
        Matrix delW1=Matrix.zeros(M.weights1.rownum(), M.weights1.colnum());
        Matrix delW2=Matrix.zeros(M.weights2.rownum(),M.weights2.colnum());
        Matrix delB1=Matrix.zeros(M.bias1.rownum(), M.bias1.colnum());
        Matrix delB2=Matrix.zeros(M.bias2.rownum(),M.bias2.colnum());
        
        for(int i=0;i<epochs;i++)
        {
            Collections.shuffle(order);
            double err=0.0d;
            for(int j=0;j<inputs.rownum();j++)
            {
                Matrix I=inputs.getrow(order.get(j));//getting one row for input
                Matrix T=outputs.getrow(order.get(j)).transpose();//getting one row for output
                
                ///////////////////////Forward Prop starts/////////////////////////////
                Matrix OH=I.matmult(M.weights1).add(M.bias1).sigmoid();//output of hidden layer
                Matrix OF=OH.matmult(M.weights2).add(M.bias2).sigmoid();//the final output
                
                //////////////////////Forward Prop ends///////////////////////////////
                
                //////////////////////Back Prop starts////////////////////////////////
                Matrix Er=T.sub(OF.transpose());//Target minus Output                
                Matrix sqEr=Er.pmult(Er);//The squared error
                Matrix DeltaO=Er.pmult(getGrad(OF.transpose()));//Deltas for the output layer
                Matrix DeltaH=M.weights2.matmult(DeltaO).pmult(getGrad(OH.transpose()));//Deltas for the hidden layer    
                DeltaO=DeltaO.scamult(rate);
                DeltaH=DeltaH.scamult(rate);
                delW2=DeltaO.matmult(OH).transpose().add(delW2.scamult(momentum));
                delB2=DeltaO.transpose().add(delB2.scamult(momentum));                
                delW1=DeltaH.matmult(I).transpose().add(delW1.scamult(momentum));
                delB1=DeltaH.transpose().add(delB1.scamult(momentum));                
                /////////////////////Back Prop ends////////////////////////////////////
                
                /////////////////////Updating weights///////////////////////////////////
                M.weights1=M.weights1.add(delW1);
                M.weights2=M.weights2.add(delW2);
                M.bias1=M.bias1.add(delB1);
                M.bias2=M.bias2.add(delB2);
                //////////////////////////////////////////////////////////////////////////
                
                err+=sqEr.sum();
            }
            err/=inputs.rownum();
            System.out.println(i+" iteration complete.");
            System.out.println("Percentage accuracy for training data is "+R.test(inputs, outputs));
            System.out.println("Average sum of squares error for all training examples is "+err);
        }
    }
    
    /**
     * Trains as well as evaluates the MLP of this Trainer object
     * @param inputs the training inputs
     * @param outputs the training outputs
     * @param rate the learning rate
     * @param momentum the momentum
     * @param epochs the number of epochs
     * @param test1i the input matrix for test set 1
     * @param test1o the output matrix for test set 1     * 
     * @param test2i the input matrix for test set 2
     * @param test2o the output matrix for test set 2
     * @param log logs the results if true
     */
    void stochGradDesc_eval(Matrix inputs, Matrix outputs, double rate, double momentum, int epochs, Matrix test1i, Matrix test1o, Matrix test2i, Matrix test2o, boolean log)
    {// This method simultaneously evaluates the accuracy for test1 and test2 as well. 
     // If log variable is true, then the values will be written to a file in the format <iteration>,<training acc.>,<test1 acc.>,<test2 acc>,<average training sse>
        try
        {
            PrintWriter pw = new PrintWriter (new BufferedWriter(new FileWriter("log.txt" ,false )), true);
            if(inputs.colnum()!=M.inputs||outputs.colnum()!=M.outputs)
            {
                System.err.println("Data incompatible with MLP");
                System.exit(1);
            }
            Runner R=new Runner(M);
            List<Integer> order=new ArrayList<>();
            for(int i=0;i<inputs.rownum();i++)
            {
                order.add(i);
            }
            //To train in a random order
            Matrix delW1=Matrix.zeros(M.weights1.rownum(), M.weights1.colnum());
            Matrix delW2=Matrix.zeros(M.weights2.rownum(),M.weights2.colnum());
            Matrix delB1=Matrix.zeros(M.bias1.rownum(), M.bias1.colnum());
            Matrix delB2=Matrix.zeros(M.bias2.rownum(),M.bias2.colnum());

            for(int i=0;i<epochs;i++)
            {
                Collections.shuffle(order);
                double err=0.0d;
                for(int j=0;j<inputs.rownum();j++)
                {
                    Matrix I=inputs.getrow(order.get(j));//getting one row for input
                    Matrix T=outputs.getrow(order.get(j)).transpose();//getting one row for output

                    ///////////////////////Forward Prop starts/////////////////////////////
                    Matrix OH=I.matmult(M.weights1).add(M.bias1).sigmoid();//output of hidden layer
                    Matrix OF=OH.matmult(M.weights2).add(M.bias2).sigmoid();//the final output

                    //////////////////////Forward Prop ends///////////////////////////////

                    //////////////////////Back Prop starts////////////////////////////////
                    Matrix Er=T.sub(OF.transpose());//Target minus Output                
                    Matrix sqEr=Er.pmult(Er);//The squared error
                    Matrix DeltaO=Er.pmult(getGrad(OF.transpose()));//Deltas for the output layer
                    Matrix DeltaH=M.weights2.matmult(DeltaO).pmult(getGrad(OH.transpose()));//Deltas for the hidden layer    
                    DeltaO=DeltaO.scamult(rate);
                    DeltaH=DeltaH.scamult(rate);
                    delW2=DeltaO.matmult(OH).transpose().add(delW2.scamult(momentum));
                    delB2=DeltaO.transpose().add(delB2.scamult(momentum));                
                    delW1=DeltaH.matmult(I).transpose().add(delW1.scamult(momentum));
                    delB1=DeltaH.transpose().add(delB1.scamult(momentum));                
                    /////////////////////Back Prop ends////////////////////////////////////

                    /////////////////////Updating weights///////////////////////////////////
                    M.weights1=M.weights1.add(delW1);
                    M.weights2=M.weights2.add(delW2);
                    M.bias1=M.bias1.add(delB1);
                    M.bias2=M.bias2.add(delB2);
                    //////////////////////////////////////////////////////////////////////////

                    err+=sqEr.sum();
                }
                err/=inputs.rownum();
                if(!log)
                {
                    System.out.println(i+" iteration complete.");
                    System.out.println("Percentage accuracy for training data is "+R.test(inputs, outputs));
                    System.out.println("Percentage accuracy for Test1 data is "+R.test(test1i, test1o));
                    System.out.println("Percentage accuracy for Test2 data is "+ R.test(test2i, test2o));
                    System.out.println("Average sum of squares error for all training examples is "+err);
                    System.out.println();
                }
                else
                {
                    pw.println(i+","+R.test(inputs, outputs)+","+R.test(test1i, test1o)+","+R.test(test2i, test2o)+","+err);
                }
            }
        }
        catch(IOException e)
        {
            System.err.println("IO Exception occurred.");
            System.exit(1);
        }
        catch(Exception e)
        {
            System.err.println("Some exception occurred.");
            System.exit(1);            
        }
    }
    
    
}
