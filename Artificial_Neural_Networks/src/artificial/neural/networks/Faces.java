package artificial.neural.networks;
import java.io.*;
public class Faces {
    
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
    static Matrix[] sunglasses_get(String filename, int n, double high, double low)//reads n images from a list in the file "filename". The returned array has the input matrix at 0 index and output at 1
    {//high and low are the on and off values respectively
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
    static MLP sunglasses_train(int hidden, double rate, double momentum, int epochs, double high, double low)//high and low are the on and off values
    {
        Matrix arr[]=sunglasses_get("straightrnd_train.list", 70, high, low);
        MLP M=MLP.random(960, hidden, 1, high, low);
        Trainer T=new Trainer(M);
        T.stochGradDesc(arr[0], arr[1], rate, momentum, epochs);
        return M;    

    }
    
    static MLP sunglasses_train_eval(int hidden, double rate, double momentum, int epochs, double high, double low, boolean log)
    {
        Matrix arr[]=sunglasses_get("straightrnd_train.list", 70, high, low);
        Matrix arr2[]=sunglasses_get("straightrnd_test1.list",34, high, low);
        Matrix arr3[]=sunglasses_get("straightrnd_test2.list", 52, high, low);
        MLP M=MLP.random(960, hidden, 1, high, low);
        Trainer T=new Trainer(M);
        T.stochGradDesc_eval(arr[0], arr[1], rate, momentum, epochs, arr2[0], arr2[1], arr3[0], arr3[1], log);
        return M;
    }
    static MLP pose_train(int hidden, double rate, double momentum, int epochs, double high, double low)
    {
            Matrix arr[]=pose_get("all_train.list", 277, high, low);            
            MLP M=MLP.random(960, hidden, 4, high, low);
            Trainer T=new Trainer(M);
            T.stochGradDesc(arr[0], arr[1], rate, momentum, epochs);
            return M;

    }
    
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
    static MLP face_train(int hidden, double rate, double momentum, int epochs, double high, double low)
    {
        Matrix arr[]=face_get("straighteven_train.list", 80, high, low);
        MLP M=MLP.random(960, hidden, 20, high, low);
        Trainer T=new Trainer(M);
        T.stochGradDesc(arr[0], arr[1], rate, momentum, epochs);
        return M;

    }
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
