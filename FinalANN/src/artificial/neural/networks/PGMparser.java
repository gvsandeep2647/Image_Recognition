package artificial.neural.networks;
import java.io.*;
import java.util.Scanner;

/**
 * A class with methods to parse .pgm files
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
public class PGMparser {
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
                return parseP5(S);
            }
            else
            {
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
            Scanner sc=new Scanner(new File(S));
            sc.nextLine();//skip P*
            int columns=sc.nextInt();
            int rows=sc.nextInt();
            sc.close();
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
            System.err.println(e);
            System.exit(1);
        }
        return null;
    }
    
}
