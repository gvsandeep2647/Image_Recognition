package artificial.neural.networks;
import java.io.*;
public class MLP {//representation of a one-hidden-layer MLP. Input vectors are assumed to be column vectors.
    int inputs;//number of input nodes
    int outputs;//number of output nodes
    int hidden;//number of nodes in the hidden layer
    Matrix weights1;//weights between input and hidden layer. [i][j] represents the weight between ith input node and jth hidden node
    Matrix bias1;//bias for the hidden units
    Matrix weights2;//weights between the hidden and the output units. [i][j] retpresents the weight between ith hidden node and ith output node
    Matrix bias2; //bias for the output units;
    double high;
    double low;
    
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
}
