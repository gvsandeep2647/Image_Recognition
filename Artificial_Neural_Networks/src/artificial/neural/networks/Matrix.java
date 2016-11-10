package artificial.neural.networks;
//import java.util.Random;
public class Matrix {
    private double data[][];//the actual matrix data
    private int rows;//number of rows in the matrix
    private int columns;//number of columns in the matrix
    private Matrix(int rows, int columns)//Private constructor. Use the static methods to create.
    {
        this.rows=rows;
        this.columns=columns;
        data=new double[rows][columns];
    }
    private static double sigmoid(double d)//convenience function used to calculate  sigmoid of a number
    {
        return 1.0/(1+Math.exp(-d));
    }
    int rownum()//returns number of rows
    {
        return rows;
    }
    int colnum()//returns number of columns
    {
        return columns;
    }
    void set(int i, int j, double value)//sets the value at a particular location
    {
        data[i][j]=value;
    }
    double get(int i, int j)//gets a particular value from the matrix
    {
        return data[i][j];
    }
    static Matrix zeros(int rows, int columns)//returns a matrix of zeros of the given dimensions
    {
        return new Matrix(rows, columns);
    }
    static Matrix ones(int rows, int columns)//returns a matrix of ones of the given dimensions
    {
        Matrix Mat=new Matrix(rows, columns);
        for(int i=0;i<rows;i++)
        {
            for(int j=0;j<rows;j++)
            {
                Mat.set(i,j,1.0d);
            }
        }
        return Mat;
    }
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
    Matrix transpose()//returns the transpose of this
    {
        return Matrix.transpose(this);
    }
    Matrix pdivide(Matrix B)//pointwise division of this by B
    {
        return Matrix.pdivide(this,B);
    }
    Matrix exp()//exponentiation of this
    {
        return Matrix.exp(this);
    }
    Matrix sigmoid()//sigmoid of this
    {
        return Matrix.sigmoid(this);
    }
    Matrix scamult(double val)//scalar multiplication of this with value
    {
        return Matrix.scamult(val, this);
    }
    
    Matrix pmult(Matrix B)//pointwise multiplication of this
    {
        return Matrix.pmult(this, B);
    }
    Matrix matmult(Matrix B)//matrix multiplication of this with B
    {
        return Matrix.matmult(this, B);
    }
    Matrix sub(Matrix B)//matrix subtraction of this and B
    {
        return Matrix.sub(this,B);
    }
    Matrix add(Matrix B)//matrix addition of this and B
    {
        return Matrix.add(this, B);
    }
    Matrix getrow(int i)//returns a view of the ith row of this matrix as a matrix
    {
        Matrix m=new Matrix(1,this.colnum());
        m.data[0]=this.data[i];
        return m;
    }
    Matrix getcolumn(int i)//returns a copy of the ith column of this matrix as a matrix
    {
        Matrix m=new Matrix(this.rownum(), 1);
        for(int j=0;j<this.rownum();j++)
        {
            m.set(j,1,this.get(j,i));
        }
        return m;
    }
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
    void setrow(double[] row, int i)//sets the ith row of the matrix.
    {
        if (row.length!=this.colnum())
        {
            System.err.println("The row is of incompatible shape.");
            System.exit(1);
        }
        this.data[i]=row;
    }
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
