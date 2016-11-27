package artificial.neural.networks;
import java.util.Random;

/**
 * Implements a matrix data structure and various methods associated with it
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
public class Matrix {
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
