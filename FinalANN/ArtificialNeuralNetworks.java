import java.io.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.util.*;
/**
 * The main class
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
public class ArtificialNeuralNetworks {

    /**
     * The main function trains and evaluates on all 3 data sets.
     * @param args the command line arguments (not needed)
     */
    public static void main(String[] args)throws InterruptedException {
        System.out.println("Training for sunglasses recognition:");
        Thread.sleep(2000);
        MLP M=Faces.sunglasses_train_eval(4,0.3, 0.3, 200, 0.9, 0.1, false);
        MLP.tofile(M, "MLPsunglasses.txt");
        MLP.toImage(M, 30,32, "sunglasses");
		
        System.out.println("Training for pose recognition:");
        Thread.sleep(10000);
        MLP M2=Faces.pose_train_eval(6,0.3, 0.3, 200, 0.9, 0.1, false);
        MLP.tofile(M2, "MLPpose.txt");
        MLP.toImage(M2, 30, 32, "pose");
		
        System.out.println("Training for face recognition:");
        Thread.sleep(10000);
        MLP M3=Faces.face_train_eval(20,0.3, 0.3, 200, 0.9, 0.1, false);
        MLP.tofile(M3, "MLPface.txt");
        MLP.toImage(M3, 30, 32, "face");
        
        
    }
    
}





/**
 *A class for methods that train and evaluate the neural network on the faces dataset
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
class Faces {
    /**
     * This method ravels and normalizes the image matrix
     * @param image image in the form of an integer matrix
     * @return raveled and normalized array of the matrix values
     */
    static double[] ravel(int[][] image)//ravels a 2D image into a 1D array and also normalizes the values by dividing by 255.0 
    {
        double arr[]=new double[image.length*image[0].length];
        int index=0;
        for(int i=0;i<image.length;i++)
        {
            for(int j=0;j<image[0].length;j++)
            {
                arr[index++]=image[i][j]/255.0;
            }
        }
        return arr;
    }
    /**
     * Reads the images and returns input and output matrices.
     * @param filename The name of the files of the list of the images to be read
     * @param n The number of these images
     * @param high The threshold for "ON" of the output Neurons
     * @param low The threshold for "OFF: of the output Neurons
     * @return The Matrices of input and output data.
     */
    static Matrix[] sunglasses_get(String filename, int n, double high, double low)
    {
        try
        {
            BufferedReader br=new BufferedReader(new FileReader(filename));
            Matrix Input=Matrix.zeros(n, 960);
            Matrix Output=Matrix.value(n, 1, low);
            String S;
            int index=0;
            while((S=br.readLine())!=null)
            {
                Input.setrow(ravel(PGMparser.parse(S)), index);
                if(S.split("/")[2].split("_")[3].equals("sunglasses"))
                {
                    Output.set(index, 0, high);
                }
                index++;
            }
            Matrix Arr[]=new Matrix[2];
            Arr[0]=Input;
            Arr[1]=Output;
            return Arr;
            
        }
        catch(FileNotFoundException e)
        {
            System.err.println("Training list not found.");
            System.exit(1);
        }
        catch(IOException e)
        {
            System.err.println("IO Exception occurred");
            System.exit(1);
        }
        return null;
    }
    /**
     * Reads the images and returns input and output matrices.
     * @param filename The name of the files of the list of the images to be read
     * @param n The number of these images
     * @param high The threshold for "ON" of the output Neurons
     * @param low The threshold for "OFF: of the output Neurons
     * @return The Matrices of input and output data.
     */
    static Matrix[] pose_get(String filename, int n, double high, double low)
    {
        try
        {
            BufferedReader br=new BufferedReader(new FileReader(filename));
            Matrix Input=Matrix.zeros(n, 960);
            Matrix Output=Matrix.value(n, 4,low);
            String S;
            int index=0;
            while((S=br.readLine())!=null)
            {
                Input.setrow(ravel(PGMparser.parse(S)), index);
                String pose=S.split("/")[2].split("_")[1];
                
                if(pose.equals("left"))
                {
                    Output.set(index, 0, high);
                }
                else if(pose.equals("up"))
                {
                    Output.set(index, 1, high);
                }
                else if(pose.equals("right"))
                {
                    Output.set(index, 2, high);
                }
                else if (pose.equals("straight"))
                {
                    Output.set(index, 3, high);
                }
                
                index++;
            }
            Matrix Arr[]=new Matrix[2];
            Arr[0]=Input;
            Arr[1]=Output;
            return Arr;
        }
        catch(FileNotFoundException e)
        {
            System.err.println("Training list not found.");
            System.exit(1);
        }
        catch(IOException e)
        {
            System.err.println("IO Exception occurred");
            System.exit(1);
        }
        return null;
    }
     /**
     * Reads the images and returns input and output matrices.
     * @param filename The name of the files of the list of the images to be read
     * @param n The number of these images
     * @param high The threshold for "ON" of the output Neurons
     * @param low The threshold for "OFF: of the output Neurons
     * @return The Matrices of input and output data.
     */
    
    static Matrix[] face_get(String filename, int n, double high, double low)
    {
        try
        {
            BufferedReader br=new BufferedReader(new FileReader(filename));
            Matrix Input=Matrix.zeros(n, 960);
            Matrix Output=Matrix.value(n,20,low);
            String S;
            int index=0;
            while((S=br.readLine())!=null)
            {
                Input.setrow(ravel(PGMparser.parse(S)), index);
                String pose=S.split("/")[2].split("_")[0];
                
                if(pose.equals("an2i"))
                {
                    Output.set(index, 0, high);
                }
                else if(pose.equals("at33"))
                {
                    Output.set(index, 1, high);
                }
                else if(pose.equals("boland"))
                {
                    Output.set(index, 2, high);
                }
                else if (pose.equals("bpm"))
                {
                    Output.set(index, 3, high);
                }
                else if (pose.equals("ch4f"))
                {
                    Output.set(index, 4, high);
                }
                else if (pose.equals("cheyer"))
                {
                    Output.set(index, 5, high);
                }
                else if (pose.equals("choon"))
                {
                    Output.set(index, 6, high);
                }
                else if (pose.equals("danieln"))
                {
                    Output.set(index, 7, high);
                }
                else if (pose.equals("glickman"))
                {
                    Output.set(index, 8, high);
                }
                else if (pose.equals("karyadi"))
                {
                    Output.set(index, 9, high);
                }
                else if (pose.equals("kawamura"))
                {
                    Output.set(index, 10, high);
                }
                else if (pose.equals("kk49"))
                {
                    Output.set(index, 11, high);
                }
                else if (pose.equals("megak"))
                {
                    Output.set(index, 12, high);
                }
                else if (pose.equals("mitchell"))
                {
                    Output.set(index, 13, high);
                }
                else if (pose.equals("night"))
                {
                    Output.set(index, 14, high);
                }
                else if (pose.equals("phoebe"))
                {
                    Output.set(index, 15, high);
                }
                else if (pose.equals("saavik"))
                {
                    Output.set(index, 16, high);
                }
                else if (pose.equals("steffi"))
                {
                    Output.set(index, 17, high);
                }
                else if (pose.equals("sz24"))
                {
                    Output.set(index, 18, high);
                }
                else if (pose.equals("tammo"))
                {
                    Output.set(index, 19, high);
                }
                
                index++;
            }
            
            Matrix arr[]=new Matrix[2];
            arr[0]=Input;
            arr[1]=Output;
            return arr;
        }
        catch(FileNotFoundException e)
        {
            System.err.println("Training list not found.");
            System.exit(1);
        }
        catch(IOException e)
        {
            System.err.println("IO Exception occurred");
            System.exit(1);
        }
        return null;
    }
    /**
     * Trains a Multi-Layered Perceptron on the sunglasses training set
     * @param hidden number of hidden units
     * @param rate the learning rate
     * @param momentum the momentum
     * @param epochs the number of epochs for training
     * @param high threshold value for "ON" of the output neurons
     * @param low threshold value for "OFF" of the output neurons
     * @return A trained MLP
     */
    static MLP sunglasses_train(int hidden, double rate, double momentum, int epochs, double high, double low)//high and low are the on and off values
    {
        Matrix arr[]=sunglasses_get("straightrnd_train.list", 70, high, low);
        MLP M=MLP.random(960, hidden, 1, high, low);
        Trainer T=new Trainer(M);
        T.stochGradDesc(arr[0], arr[1], rate, momentum, epochs);
        return M;    

    }
    /**
     * Trains and evaluates a Multi-Layered Perceptron on the sunglasses dataset
     * @param hidden number of hidden units
     * @param rate the learning rate
     * @param momentum the momentum
     * @param epochs the number of epochs for training
     * @param high threshold value for "ON" of the output neurons
     * @param low threshold values for "OFF" of the output neurons
     * @param log should the results be logged
     * @return the trained MLP
     */
    static MLP sunglasses_train_eval(int hidden, double rate, double momentum, int epochs, double high, double low, boolean log)
    {
        Matrix arr[]=sunglasses_get("straightrnd_train.list", 70, high, low);
        Matrix arr2[]=sunglasses_get("straightrnd_test1.list",34, high, low);
        Matrix arr3[]=sunglasses_get("straightrnd_test2.list", 52, high, low);
        MLP M=MLP.random(960, hidden, 1, high, low);
        Trainer T=new Trainer(M);
        T.stochGradDesc_eval(arr[0], arr[1], rate,momentum, epochs, arr2[0], arr2[1], arr3[0], arr3[1], log);
        return M;
    }
    /**
     * Trains a Multi-Layered Perceptron on the pose training set
     * @param hidden number of hidden units
     * @param rate the learning rate
     * @param momentum the momentum
     * @param epochs the number of epochs for training
     * @param high threshold value for "ON" of the output neurons
     * @param low threshold value for "OFF" of the output neurons
     * @return A trained MLP
     */
    static MLP pose_train(int hidden, double rate, double momentum, int epochs, double high, double low)
    {
            Matrix arr[]=pose_get("all_train.list", 277, high, low);            
            MLP M=MLP.random(960, hidden, 4, high, low);
            Trainer T=new Trainer(M);
            T.stochGradDesc(arr[0], arr[1], rate, momentum, epochs);
            return M;

    }
    /**
     * Trains and evaluates a Multi-Layered Perceptron on the pose dataset
     * @param hidden number of hidden units
     * @param rate the learning rate
     * @param momentum the momentum
     * @param epochs the number of epochs for training
     * @param high threshold value for "ON" of the output neurons
     * @param low threshold values for "OFF" of the output neurons
     * @param log should the results be logged
     * @return the trained MLP
     */
    
    static MLP pose_train_eval(int hidden, double rate, double momentum, int epochs, double high, double low, boolean log)
    {
        Matrix arr[]=pose_get("all_train.list", 277, high, low);
        Matrix arr2[]=pose_get("all_test1.list", 139, high, low);
        Matrix arr3[]=pose_get("all_test2.list", 208, high, low);
        MLP M=MLP.random(960, hidden, 4, high, low);
        Trainer T=new Trainer(M);
        T.stochGradDesc_eval(arr[0], arr[1], rate, momentum, epochs, arr2[0], arr2[1], arr3[0], arr3[1], log);
        return M;
    }
    /**
     * Trains a Multi-Layered Perceptron on the face detection training set
     * @param hidden number of hidden units
     * @param rate the learning rate
     * @param momentum the momentum
     * @param epochs the number of epochs for training
     * @param high threshold value for "ON" of the output neurons
     * @param low threshold value for "OFF" of the output neurons
     * @return A trained MLP
     */
    static MLP face_train(int hidden, double rate, double momentum, int epochs, double high, double low)
    {
        Matrix arr[]=face_get("straighteven_train.list", 80, high, low);
        MLP M=MLP.random(960, hidden, 20, high, low);
        Trainer T=new Trainer(M);
        T.stochGradDesc(arr[0], arr[1], rate, momentum, epochs);
        return M;

    }
    /**
     * Trains and evaluates a Multi-Layered Perceptron on the face detection dataset
     * @param hidden number of hidden units
     * @param rate the learning rate
     * @param momentum the momentum
     * @param epochs the number of epochs for training
     * @param high threshold value for "ON" of the output neurons
     * @param low threshold values for "OFF" of the output neurons
     * @param log should the results be logged
     * @return the trained MLP
     */
    static MLP face_train_eval(int hidden, double rate, double momentum, int epochs, double high, double low, boolean log)
    {
        Matrix arr[]=face_get("straighteven_train.list", 80, high, low);
        Matrix arr2[]=face_get("straighteven_test1.list", 36, high, low);
        Matrix arr3[]=face_get("straighteven_test2.list", 40, high, low);
        MLP M=MLP.random(960, hidden, 20, high, low);
        Trainer T=new Trainer(M);
        T.stochGradDesc_eval(arr[0], arr[1], rate, momentum, epochs, arr2[0], arr2[1], arr3[0], arr3[1], log);
        return M;
    }
    
}





/**
 * Implements a matrix data structure and various methods associated with it
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
class Matrix {
    private double data[][];//the actual matrix data
    private int rows;//number of rows in the matrix
    private int columns;//number of columns in the matrix
    /**
     * Constructs a new matrix of zeros of the given dimensions
     * @param rows Number of rows of the matrix
     * @param columns Number of columns of the matrix
     */
    private Matrix(int rows, int columns)//Private constructor. Use the static methods to create.
    {
        this.rows=rows;
        this.columns=columns;
        data=new double[rows][columns];
    }
    /**
     * Returns the sigmoid of any double value
     * @param d the value
     * @return sigmoid(d)
     */
    private static double sigmoid(double d)//convenience function used to calculate  sigmoid of a number
    {
        return 1.0/(1+Math.exp(-d));
    }
    /**
     * Gets the number of rows of this matrix
     * @return number of rows
     */
    int rownum()//returns number of rows
    {
        return rows;
    }
    /**
     * Gets the number of columns of this matrix
     * @return number of  columns
     */
    int colnum()//returns number of columns
    {
        return columns;
    }
    /**
     * Sets the (i,j) value in this matrix to value
     * @param i row number
     * @param j column number
     * @param value value to be set
     */
    void set(int i, int j, double value)//sets the value at a particular location
    {
        data[i][j]=value;
    }
    /**
     * Gets the (i,j) value of this matrix
     * @param i row number
     * @param j column number
     * @return the value at (i,j)
     */
    double get(int i, int j)//gets a particular value from the matrix
    {
        return data[i][j];
    }
    /**
     * Returns a matrix filled with zeros
     * @param rows number of rows
     * @param columns number of columns
     * @return the matrix constructed
     */
    static Matrix zeros(int rows, int columns)//returns a matrix of zeros of the given dimensions
    {
        return new Matrix(rows, columns);
    }
    /**
     * Returns a matrix filled with ones
     * @param rows number of rows
     * @param columns number of columns
     * @return the matrix constructed
     */
    static Matrix ones(int rows, int columns)//returns a matrix of ones of the given dimensions
    {
        Matrix Mat=new Matrix(rows, columns);
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<columns;j++)
            {
                Mat.set(i,j,1.0d);
            }
        }
        return Mat;
    }
    /**
     * Returns a matrix filled with the given value
     * @param rows number of rows
     * @param columns number of columns
     * @param value the value to be filled
     * @return the constructed matrix
     */
    static Matrix value(int rows, int columns, double value)//returns a matrix filled with the given values
    {
        Matrix Mat=new Matrix(rows, columns);
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<columns;j++)
            {
                Mat.set(i,j,value);
            }
        }
        return Mat;       
    }
    /**
     * Retruns a matrix filled with random values in the given range
     * @param rows number of rows
     * @param columns number of columns
     * @param start the start of the range
     * @param end the end of the range
     * @return the constructed matrix
     */
    static Matrix random(int rows, int columns, double start, double end)//returns a matrix of random values in the given range
    {
        Matrix Mat=new Matrix(rows, columns);
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<columns;j++)
            {
                double val=start+Math.random()*(end-start);
                Mat.set(i,j,val);
            }
        }
        return Mat;
    }
    /**
     * Multiplies two matrices and returns the result
     * @param A the left matrix
     * @param B the right matrix
     * @return the resultant matrix
     */
    static Matrix matmult(Matrix A, Matrix B)//matrix multiplication of A and B
    {
        if(A.colnum()!=B.rownum())
        {
            System.err.println("Matrices are incompatible for matrix multiplication.");
            System.exit(1);
        }
        Matrix mat=new Matrix(A.rownum(),B.colnum());
        for(int i=0;i<A.rownum();i++)
        {
            for(int j=0;j<B.colnum();j++)
            {
                double sum=0.0d;
                for(int k=0;k<A.colnum();k++)
                {
                    sum+=A.get(i, k)*B.get(k,j);
                }
                mat.set(i,j,sum);
            }
        }
        return mat;
    }
    /**
     * Point-wise multiplies two matrices
     * @param A the first matrix
     * @param B the second matrix
     * @return the resultant matrix
     */
    static Matrix pmult(Matrix A, Matrix B)//pointwise multiplicaiton of A and B
    {
        if(A.colnum()!=B.colnum()||A.rownum()!=B.rownum())
        {
            System.err.println("Matrices are incompatible for pointwise multiplication.");
            System.exit(1);
        }
        Matrix mat=new Matrix(A.rownum(), A.colnum());
        for(int i=0;i<A.rownum();i++)
        {
            for(int j=0;j<A.colnum();j++)
            {
                mat.set(i,j,A.get(i, j)*B.get(i,j));
            }
        }
        return mat;
    }
    /**
     * Adds two matrices
     * @param A the first matrix
     * @param B the second matrix
     * @return the resultant matrix
     */
    static Matrix add(Matrix A, Matrix B)//addition of A and B
    {
        if(A.colnum()!=B.colnum()||A.rownum()!=B.rownum())
        {
            System.err.println("Matrices are incompatible for addition.");
            System.exit(1);
        }
        Matrix mat=new Matrix(A.rownum(), A.colnum());
        for(int i=0;i<A.rownum();i++)
        {
            for(int j=0;j<A.colnum();j++)
            {
                mat.set(i,j,A.get(i, j)+B.get(i,j));
            }
        }
        return mat;
    }
    /**
     * Subtracts two matrices
     * @param A the first matrix
     * @param B the second matrix
     * @return the resultant matrix
     */
    static Matrix sub(Matrix A, Matrix B)//subtraction of A and B
    {
        if(A.colnum()!=B.colnum()||A.rownum()!=B.rownum())
        {
            System.err.println("Matrices are incompatible for subtraction.");
            System.exit(1);
        }
        Matrix mat=new Matrix(A.rownum(), A.colnum());
        for(int i=0;i<A.rownum();i++)
        {
            for(int j=0;j<A.colnum();j++)
            {
                mat.set(i,j,A.get(i, j)-B.get(i,j));
            }
        }
        return mat;        
    }
    /**
     * Multiplies the matrix with a scalar
     * @param val the scalar value
     * @param A the matrix
     * @return the resultant matrix
     */
    static Matrix scamult(double val, Matrix A)//multiplication of the matrix with a scalar
    {
        Matrix mat=new Matrix(A.rownum(), A.colnum());
        for(int i=0;i<A.rownum();i++)
        {
            for(int j=0;j<A.colnum();j++)
            {
                mat.set(i,j,A.get(i, j)*val);
            }
        }
        return mat;
    }
    /**
     * Exponentiates every value of the matrix
     * @param A the matrix
     * @return the resultant matrix
     */
    static Matrix exp(Matrix A)//exponentiating each value in the matrix
    {
        Matrix mat=new Matrix(A.rownum(), A.colnum());
        for(int i=0;i<A.rownum();i++)
        {
            for(int j=0;j<A.colnum();j++)
            {
                mat.set(i,j,Math.exp(A.get(i, j)));
            }
        }
        return mat;
    }
    /**
     * Calculates sigmoid of every value of the matrix
     * @param A the matrix
     * @return the resultant matrix
     */
    static Matrix sigmoid(Matrix A)//sigmoid function of each value in the matrix
    {
        Matrix mat=new Matrix(A.rownum(), A.colnum());
        for(int i=0;i<A.rownum();i++)
        {
            for(int j=0;j<A.colnum();j++)
            {
                mat.set(i,j,Matrix.sigmoid(A.get(i, j)));
            }
        }
        return mat;        
    }
    /**
     * Point-wise divides the two matrices
     * @param A the dividend matrix
     * @param B the divisor matrix
     * @return the resultant matrix
     */
    static Matrix pdivide(Matrix A, Matrix B)//pointwise division of A and B
    {
        if(A.colnum()!=B.colnum()||A.rownum()!=B.rownum())
        {
            System.err.println("Matrices are incompatible for pointwise division.");
            System.exit(1);
        }
        Matrix mat=new Matrix(A.rownum(), A.colnum());
        for(int i=0;i<A.rownum();i++)
        {
            for(int j=0;j<A.colnum();j++)
            {
                mat.set(i,j,A.get(i, j)/B.get(i,j));
            }
        }
        return mat;        
    }
    /**
     * Calculates the transpose of a matrix
     * @param A the matrix
     * @return the resultant matrix
     */
    static Matrix transpose(Matrix A)//returns the transpose of A
    {
        Matrix mat=new Matrix(A.colnum(), A.rownum());
        for(int i=0;i<A.colnum();i++)
        {
            for(int j=0;j<A.rownum();j++)
            {
                mat.set(i,j,A.get(j,i));
            }
        }
        return mat;
    }
    /**
     * Calculates the square root of each element of a matrix
     * @param A the matrix
     * @return the resultant matrix
     */
    static Matrix sqrt(Matrix A)
    {
        Matrix mat=new Matrix(A.rows, A.columns);
        for(int i=0;i<A.rownum();i++)
        {
            for(int j=0;j<A.colnum();j++)
            {
                mat.set(i, j, Math.sqrt(A.get(i,j)));
            }
        }
        return mat;
    }
    /**
     * Calculates the transpose of this matrix
     * @return the resultant matrix
     */
    Matrix transpose()//returns the transpose of this
    {
        return Matrix.transpose(this);
    }
    /**
     * Point-wise divides this matrix with B
     * @param B the divisor matrix
     * @return the resultant matrix
     */
    Matrix pdivide(Matrix B)//pointwise division of this by B
    {
        return Matrix.pdivide(this,B);
    }
    /**
     * Exponentiates this matrix
     * @return the resultant matrix
     */
    Matrix exp()//exponentiation of this
    {
        return Matrix.exp(this);
    }
    /**
     * Calculates the sigmoid of this matrix
     * @return the resultant matrix
     */
    Matrix sigmoid()//sigmoid of this
    {
        return Matrix.sigmoid(this);
    }
    /**
     * Multiplies this matrix with a scalar
     * @param val the scalar value
     * @return the resultant matrix
     */
    Matrix scamult(double val)//scalar multiplication of this with value
    {
        return Matrix.scamult(val, this);
    }
    /**
     * Point-wise multiplies this matrix with another
     * @param B the other matrix
     * @return the resultant matrix
     */
    Matrix pmult(Matrix B)//pointwise multiplication of this
    {
        return Matrix.pmult(this, B);
    }
    /**
     * Multiplies this matrix with another
     * @param B the other matrix
     * @return the resultant matrix
     */
    Matrix matmult(Matrix B)//matrix multiplication of this with B
    {
        return Matrix.matmult(this, B);
    }
    /**
     * Subtracts another matrix from this matrix
     * @param B the other matrix
     * @return the resultant matrix
     */
    Matrix sub(Matrix B)//matrix subtraction of this and B
    {
        return Matrix.sub(this,B);
    }
    /**
     * Adds this matrix to another
     * @param B the other matrix
     * @return the resultant matrix
     */
    Matrix add(Matrix B)//matrix addition of this and B
    {
        return Matrix.add(this, B);
    }
    /**
     * Returns a view of the ith row of this matrix as a matrix
     * @param i the required row number
     * @return the row as a matrix
     */
    Matrix getrow(int i)//returns a view of the ith row of this matrix as a matrix
    {
        Matrix m=new Matrix(1,this.colnum());
        m.data[0]=this.data[i];
        return m;
    }
    /**
     * Returns a copy of the ith column of this matrix as a matrix
     * @param i the required column number
     * @return the column as a matrix
     */
    Matrix getcolumn(int i)//returns a copy of the ith column of this matrix as a matrix
    {
        Matrix m=new Matrix(this.rownum(), 1);
        for(int j=0;j<this.rownum();j++)
        {
            m.set(j,1,this.get(j,i));
        }
        return m;
    }
    /**
     * Rounds every value in the matrix to low/high
     * @param high the high value
     * @param low the low value
     * @return the resultant matrix
     */
    Matrix onehot(double high, double low)
    {
        Matrix m=Matrix.zeros(rows,columns);
        for(int i=0;i<this.rows;i++)
        {
            for(int j=0;j<this.columns;j++)
            {
                if(Math.abs(this.data[i][j]-high)>Math.abs(this.data[i][j]-low))m.data[i][j]=low;
                else
                m.data[i][j]=high;
            }

        }
        return m;
    }
    /**
     * Calculates the sum of all values of the matrix
     * @return the calculated sum
     */
    double sum()//returns the sum of all elements of this matrix
    {
        double sum=0.0d;
        for(int i=0;i<this.rows;i++)
        {
            for(int j=0;j<this.columns;j++)
            {
                sum+=this.data[i][j];
            }
        }
        return sum;
    }
    /**
     * Checks if this matrix is equal to another
     * @param B the other matrix
     * @return the result
     */
    boolean equals(Matrix B)//checks for equality between two matrices
    {
        if(B.rownum()!=this.rownum()||B.colnum()!=this.colnum())return false;
        for(int i=0;i<B.rownum();i++)
        {
            for(int j=0;j<B.colnum();j++)
            {
                if(B.get(i, j)!=this.get(i,j)){return false;}
                
            }
        }
        return true;
    }
    /**
     * replaces the ith row of the matrix with the given double array
     * @param row the double array
     * @param i the row number
     */
    void setrow(double[] row, int i)//sets the ith row of the matrix.
    {
        if (row.length!=this.colnum())
        {
            System.err.println("The row is of incompatible shape.");
            System.exit(1);
        }
        this.data[i]=row;
    }
    /**
     * Calculates the square root of values of this matrix
     * @return the resultant matrix
     */
    Matrix sqrt()
    {
        return Matrix.sqrt(this);
    }

    /**
     * Returns a string representation of the matrix
     * @return a string representation of this matrix
     */
    @Override
    public String toString()
    {
        String S="";
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<columns;j++)
            {
                S=S+data[i][j]+"\t";
            }
            S=S+"\n";
        }
        return S;
    }
}





/**
 * Class describing the Multi-Layered Perceptron structure
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
class MLP {//representation of a one-hidden-layer MLP. Input vectors are assumed to be column vectors.
    int inputs;//number of input nodes
    int outputs;//number of output nodes
    int hidden;//number of nodes in the hidden layer
    Matrix weights1;//weights between input and hidden layer. [i][j] represents the weight between ith input node and jth hidden node
    Matrix bias1;//bias for the hidden units
    Matrix weights2;//weights between the hidden and the output units. [i][j] retpresents the weight between ith hidden node and ith output node
    Matrix bias2; //bias for the output units;
    double high;
    double low;
    /**
     * Constructor for the MLP class. Initializes with zero weight values.
     * @param inputs number of input units
     * @param hidden number of hidden units
     * @param outputs number of output units
     * @param high threshold for "ON" of the output neurons
     * @param low threshold for "OFF" of the output neurons
     */
    MLP(int inputs, int hidden, int outputs, double high, double low)//zero weight constructor
    {
        this.inputs=inputs;
        this.hidden=hidden;
        this.outputs=outputs;
        weights1=Matrix.zeros(inputs, hidden);
        bias1=Matrix.zeros(1,hidden);
        weights2=Matrix.zeros(hidden, outputs);
        bias2=Matrix.zeros(1, outputs);
        this.high=high;
        this.low=low;
    }
    /**
     * Initializes an MLP with random weights using Xavier initialization
     * @param inputs number of input units
     * @param hidden number of hidden units
     * @param outputs number of output units
     * @param high threshold for "ON" of the output neurons
     * @param low threshold for "OFF" of the output neurons
     * @return an MLP object of the given structure and random weights
     */
    static MLP random(int inputs, int hidden, int outputs, double high, double low)//initializes an MLP with random weights
    {
        MLP m=new MLP(inputs, hidden, outputs, high, low);
        double z=4*Math.sqrt(6.0/(inputs+1+outputs));
        double a=4*Math.sqrt(6.0/(2+hidden));
        m.weights1=Matrix.random(inputs, hidden, -z, z);
        m.bias1=Matrix.random(1, hidden, -z, z);
        m.weights2=Matrix.random(hidden, outputs, -a,a);
        m.bias2=Matrix.random(1, outputs, -a, a);
        return m;
    }
    /**
     * Reads an MLP from the given file
     * @param filename the name of the file
     * @return the MLP object corresponding to the given file
     */
    static MLP fromfile(String filename)//reads an MLP from the given file
    {
        try{
            BufferedReader br=new BufferedReader(new FileReader(filename));
            int inputs=Integer.parseInt(br.readLine());
            int hidden=Integer.parseInt(br.readLine());
            int outputs=Integer.parseInt(br.readLine());
            double high=Double.parseDouble(br.readLine());
            double low=Double.parseDouble(br.readLine());
            MLP m=new MLP(inputs, hidden, outputs, high, low);
            for(int i=0;i<inputs;i++)
            {
                for(int j=0;j<hidden;j++)
                {
                    m.weights1.set(i, j, Double.parseDouble(br.readLine()));
                }
            }
            for(int i=0;i<hidden;i++)
            {
                m.bias1.set(1,i,Double.parseDouble(br.readLine()));
            }
            for(int i=0;i<hidden;i++)
            {
                for(int j=0;j<outputs;j++)
                {
                    m.weights2.set(i, j, Double.parseDouble(br.readLine()));
                }
            }
            for(int i=0;i<outputs;i++)
            {
                m.bias2.set(1, i, Double.parseDouble(br.readLine()));
            }
            br.close();
            return m;
        }
        catch(FileNotFoundException a)
        {
            System.err.println("File not found.");
            System.exit(1);
            return null;
        }
        catch(IOException a)
        {
            System.err.println("IO Exception occurred.");
            System.exit(1);
            return null;
        }
        catch(Exception e)
        {
            System.err.println("Some exception occurred.");
            System.exit(1);
            return null;
        }
        
            
    }
    /**
     * Serializes the MLP into a file
     * @param m the MLP object to be serialized
     * @param filename the name of the file
     */
    static void tofile(MLP m, String filename)//serializes an MLP into the file
    {
        try
        {
            PrintWriter pw = new PrintWriter (new BufferedWriter(new FileWriter(filename ,false )), true);
            pw.println(m.inputs);
            pw.println(m.hidden);
            pw.println(m.outputs);
            pw.println(m.high);
            pw.println(m.low);
            for(int i=0;i<m.inputs;i++)
            {
                for(int j=0;j<m.hidden;j++)
                {
                    pw.println(m.weights1.get(i, j));
                }
            }
            for(int i=0;i<m.hidden;i++)
            {
                pw.println(m.bias1.get(0,i));
            }
            for(int i=0;i<m.hidden;i++)
            {
                for(int j=0;j<m.outputs;j++)
                {
                    pw.println(m.weights2.get(i, j));
                }
            }
            for(int i=0;i<m.outputs;i++)
            {
                pw.println(m.bias2.get(0, i));
            }            
            pw.close();
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
    /**
     * Converts the weights of the hidden layer into images by reshaping and normalizing
     * @param m the MLP
     * @param rows number of rows of the images
     * @param columns number of columns of the images
     * @param name Base name of the output files. The images are named <name>_0.jpg <name>_1.jpg etc. for each hidden neuron
     */
    static void toImage(MLP m, int rows, int columns, String name)
    {
        assert m.weights1.rownum()==rows*columns;
        for(int i=0;i<m.weights1.colnum();i++)
        {
            double arr[]=new double[rows*columns];
            for(int j=0;j<m.weights1.rownum();j++)
            {
                arr[j]=m.weights1.get(j,i);
            }
            printimage(arr, name+"_"+i+".jpg", rows, columns);
        }
    }
    /**
     * Takes a single array of doubles and prints it into a grayscale image by reshaping and normalizing
     * @param arr the array to be converted to image
     * @param name the name of the image file
     * @param rows the number of rows in the image
     * @param columns the number of columns in the image
     */
    static void printimage(double[] arr, String name, int rows, int columns)
    {
        double max=-100000;
        double min=100000;
        for(int i=0;i<arr.length;i++)
        {
            if(arr[i]<min)min=arr[i];
            if(arr[i]>max)max=arr[i];
        }
        
        try{
            BufferedImage img = new BufferedImage(columns, rows, BufferedImage.TYPE_BYTE_GRAY);
            for(int i=0; i<columns;i++)
            {
                for(int j=0;j<rows;j++)
                {
                    double value=arr[j*columns + i];
                    value=(value-min)*255.0/(max-min);
                    int val=(int)value;
                    
                    img.setRGB(i, j, (val << 16) | (val << 8) | val);
                }
            }
            ImageIO.write(img, "JPG", new File(name));
            
        }
        catch(Exception e)
        {
            System.err.println(e);
        }
    }
}






/**
 * A class with methods to parse .pgm files
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
class PGMparser {
    /**
     * Reads a .pgm file and returns as an integer array
     * @param S name of the file
     * @return 2D integer array containing the image
     */
    static int[][] parse(String S)//reads a .pgm file with name S and returns as an integer array
    {
        try
        {
            BufferedReader br=new BufferedReader(new FileReader(S));
            String Type=br.readLine();
            if(Type.equals("P5"))
            {
				br.close();
                return parseP5(S);
            }
            else
            {
				br.close();
                return parseP2(S);
            }
            
        }
        catch(FileNotFoundException e)
        {
            System.err.println("Image file not found.");
            System.exit(1);
        }
        catch(IOException e)
        {
            System.err.println("IO exception occurred.");
            System.exit(1);
        }
        catch(Exception e)
        {
            System.err.println("Some exception occurred.");
            System.exit(1);
        }
        return null;
    }
    /**
     * Parses the P2 flavor of .pgm files
     * @param S name of the file
     * @return 2D array containing the image
     */
    static int[][] parseP2(String S)//parses a P2 .pgm file
    {
        try
        {
            Scanner sc=new Scanner(new File(S));
            sc.nextLine();//skip P*
            int columns=sc.nextInt();
            int rows=sc.nextInt();
            int max=sc.nextInt();
            int image[][]=new int[rows][columns];
            for(int i=0;i<rows;i++)
            {
                for(int j=0;j<columns;j++)
                {
                    image[i][j]=sc.nextInt();
                }
            }
            sc.close();
            return image;
        }
        catch(FileNotFoundException e)
        {
            System.err.println("Image file not found.");
            System.exit(1);
        }
        catch(Exception e)
        {
            System.err.println("Some error occurred.");
            System.exit(1);
        }
        return null;
    }
    /**
     * Parses the P5 flavor of .pgm files
     * @param S name of the file
     * @return 2D array containing the image
     */
    static int[][] parseP5(String S)//parses a P5 .pgm file
    {
        try
        {
			BufferedReader br = new BufferedReader(new FileReader(S));
			br.readLine();
			String[] arr=br.readLine().split(" ");
            int columns=Integer.parseInt(arr[0]);
            int rows=Integer.parseInt(arr[1]);
            int image[][]=new int[rows][columns];
            File F=new File(S);
            long len=F.length();
            long skip=len-rows*columns;
            FileInputStream FIS=new FileInputStream(F);
            FIS.skip(skip);
            for(int i=0;i<rows;i++)
            {
                for(int j=0;j<columns;j++)
                {
                    image[i][j]=(int)FIS.read();
                }
            }
            FIS.close();
			br.close();
            return image;
        }
        catch(FileNotFoundException e)
        {
            System.err.println("Image file not found.");
            System.exit(1);
        }
        catch(IOException e)
        {
            System.err.println("IO Exception occurred.");
            System.exit(1);
        }
        catch(Exception e)
        {
            System.err.println("Some error");
            System.exit(1);
        }
        return null;
    }
    
}



/**
 * This class encapsulates the methods needed to run an MLP
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
class Runner {
    
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





/**
 * A class that encapsulates all the training operations
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
class Trainer {
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

