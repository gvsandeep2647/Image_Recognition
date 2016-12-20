import java.io.*;
import java.util.*;


/**
 * @author Sandeep,Snehal
 * AIM : To apply neural network learning to the problem of face recognition. To build a sunglasses recognizer, a face recognizer, and a pose recognizer.
 */

public class ANN {
	/** Stores the normalized pixel values for the current image */
	static double image[][];

	/** Class instantiation for handling PGM Images*/
	static PGMReader pgm = new PGMReader();

	/** A HashMap which gives a unique ID*/
	static HashMap<Integer,String> faceID = new HashMap<>();

	/** A HashMap which gives every direction an unique number*/
	static HashMap<Integer, String> directionID = new HashMap<>();


	public static void main(String[] args) {
		initializeHashMaps();
		
		
		/* Everything related to Sunglasses Predictor */

		long startTime = System.currentTimeMillis();
		try{
			Network sunGlasses = new Network(960,4,1,0.3,0.3);
			imageHandler("straightrnd_train.list",sunGlasses,1);
			Validation v1 = new Validation();
			handlingTestData("straightrnd_test1.list",sunGlasses,1, v1);
			Validation v2 = new Validation();
			handlingTestData("straightrnd_test2.list",sunGlasses,1, v2);
		}
		catch(Exception e){
			System.err.println(e);
		}
		long endTime = System.currentTimeMillis();
		long elapsedTime = endTime - startTime;
		System.out.println();
		System.out.println("Time taken for learning and predicting whether the user has worn sunglasses : " + (double)elapsedTime/1000 + " seconds");
		System.out.println("*******************"); 
		System.out.println();
		
		
		
		/* Everything related to Face Recognizer */
		startTime = System.currentTimeMillis();
		try{
			Network faceRecognizer = new Network(960,20,20,0.3,0.3);
			imageHandler("straighteven_train.list",faceRecognizer,2);
			Validation v3 = new Validation();
			handlingTestData("straighteven_test1.list",faceRecognizer,2,v3);
			Validation v4 = new Validation();
			handlingTestData("straighteven_test2.list",faceRecognizer,2,v4);
		}
		catch(Exception e){
			System.err.println(e);
		}
		endTime = System.currentTimeMillis();
		elapsedTime = endTime - startTime;
		System.out.println();
		System.out.println("Time taken for learning and predicting the person's name : " + (double)elapsedTime/1000 + " seconds");
		System.out.println("*******************"); 
		System.out.println();
		
		

		/* Everything related to Pose Recognizer */
		startTime = System.currentTimeMillis();
		try{
			Network poseRecognizer = new Network(960,6,4,0.3,0.3);
			imageHandler("all_train.list",poseRecognizer,3);
			Validation v5 = new Validation();
			handlingTestData("all_test1.list",poseRecognizer,3, v5);
			Validation v6 = new Validation();
			handlingTestData("all_test2.list",poseRecognizer,3, v6);
		}
		catch(Exception e){
			System.err.println(e);
		}
		endTime = System.currentTimeMillis();
		elapsedTime = endTime - startTime;
		System.out.println();
		System.out.println("Time taken for learning and predicting the person's pose : " + (double)elapsedTime/1000 + " seconds");
		System.out.println("*******************");
		System.out.println();
		
	}


	/**
	 * @param filename : The filename which contains a list of image locations from which we will read the images
	 * @param mode : 1 = Sun glasses predictor 2 = Face Recognizer 3 = Pose Recognizer
	 * @param net : The network on which we are currently working
	 * @throws IOException
	 */
	
	public static void imageHandler(String filename,Network net,int mode) throws IOException{		

		for(int i=0;i<300;i++)
		{
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line = null;
			int k = 0;
			while((line=br.readLine()) != null) {
				image = pgm.readImage(line);
				double input[] = linearize(image);
				double targetVector[] = genTargetVector(line,mode);
				net.updateWeights(input,targetVector,k);
				k++;
			}
			br.close();
		}
	}

	
	/**
	 * @param filename : the filename in which we have the testing data
	 * @param net : the neural network on which we will be testing
	 * @param mode : 1 = Sun glasses predictor 2 = Face Recognizer 3 = Pose Recognizer
	 * @param validation : A utility class to see how many correct and wrong predictions the model has made.
	 * @throws IOException
	 */
	
	public static void handlingTestData (String filename,Network net,int mode,Validation validation) throws IOException{
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line = null;
		
		while((line = br.readLine())!=null){
			image = pgm.readImage(line);
			double input[] = linearize(image);
			validation.updateResults(net.getPrediction(input,mode),line, mode);
		 }
		br.close();
		calcAccuracy(validation.correct, validation.wrong, mode,filename);
	}

	/**
	 * @param line : file path of the image whose target vector we want to create
	 * @param mode : 1. Sunglasses 2. Face Recognizer 3. Pose Recognizer
	 * @return the target vector
	 */
	
	public static double[] genTargetVector(String line,int mode){
		double target[] = new double[1];
		switch(mode){
		case 1:
			target = new double[1];
			if(line.contains("sunglasses"))
				target[0] = 0.9;
			else
				target[0] = 0.1;
			break;
		case 2:
			target = new double[20];
			for(int i=0;i<faceID.size();i++)
			{
				if(line.contains(faceID.get(i)))
					target[i] = 0.9;
				else
					target[i] = 0.1;
			}
			break;
		case 3:
			target = new double[4];
			for(int i=0;i<directionID.size();i++)
			{
				if(line.contains(directionID.get(i)))
					target[i] = 0.9;
				else
					target[i] = 0.1;
			}
			break;
		}
		return target;
	}


	/**
	 * @param image : the 2D array which we want to convert to 1D array
	 * @return 1D array of the image
	 */

	public static double[] linearize(double[][] image){
		double temp[] = new double[image.length * (image[0].length)];
		int k = 0;
		for(int i = 0;i<image.length;i++)
		{
			for(int j = 0;j<image[i].length;j++)
			{
				temp[k] = image[i][j];
				k++;
			}
		}
		return temp;
	}




	/**
	 * @param correct : No of correct predictions
	 * @param wrong : No of wrong predictions
	 * @param mode : 1. Sunglasses 2. Face Recognizer 3. Pose Recognizer
	 * @param filename : The test set on which the model is predicting
	 */
	
	public static void calcAccuracy(int correct,int wrong,int mode,String filename){
		double accuracy = ((double)correct/(wrong+correct))*100;
		accuracy = Math.round(accuracy*100) / 100.0;
		String strMode="";
		switch (mode) {
		case 1:
			strMode = "Sunglasses Predictor";
			break;
		case 2:
			strMode = "Face Recognizer";
			break;
		case 3:
			strMode = "Pose Recognizer";
			break;
		}

		System.out.println("Accuracy of the " + strMode +" on the file " + filename + " is : " + accuracy +"%");
		System.out.println("It has correctly classified "+ correct + " instances out of "+ (wrong+correct) +" instances" );
		System.out.println();
	}
	

	/**
	 * A function to populate the HashMaps 
	 */
	
	public static void initializeHashMaps(){
		directionID.put(0,"straight");
		directionID.put(1,"right");
		directionID.put(2,"left");
		directionID.put(3,"up");

		faceID.put(0,"an2i");
		faceID.put(1,"at33");
		faceID.put(2,"boland");
		faceID.put(3,"bpm");
		faceID.put(4,"ch4f");
		faceID.put(5,"cheyer");
		faceID.put(6,"choon");
		faceID.put(7,"danieln");
		faceID.put(8,"glcikman");
		faceID.put(9,"karyadi");
		faceID.put(10,"kawamura");
		faceID.put(11,"kk49");
		faceID.put(12,"megak");
		faceID.put(13,"mitchell");
		faceID.put(14,"night");
		faceID.put(15,"phoebe");
		faceID.put(16,"saavik");
		faceID.put(17,"steffi");
		faceID.put(18,"sz24");
		faceID.put(19,"tammo");
	}
}



/**
 * An exception which would be raised if the two vectors whose do predict we are supposed to calculate are not of the same length 
 */

class IncompatibleLengthException extends Exception{
	private static final long serialVersionUID = 1L;

	/**
	 * @param s : the error message to be displayed
	 */
	
	IncompatibleLengthException(String s) {
		super(s);
	}
}

/**
 * The Artificial Neural Network which will be dynamically created as per user's requirements
 */

class Network {
	int inputs;
	int hidden;
	int outputs;
	double eta;
	double momentum;
	double hiddenWeights[][];
	double outputWeights[][];
	HashMap<Integer, double[][]> deltaHiddenW = new HashMap<>();
	HashMap<Integer,double[][]> deltaOutputW = new HashMap<>();
	double deltaK[];
	double deltaH[];
	HashMap<int[], Double> forMomentum = new HashMap<>();

	/**
	 * @param inputs : No. of inputs to the network 
	 * @param hidden : No. of hidden layers in the network
	 * @param outputs : No. of output layers in the network
	 * @param eta : Learning rate of the network
	 * @param momentum : Momentum of the network
	 */

	Network(int inputs, int hidden, int outputs, double eta, double momentum){
		this.inputs = inputs;
		this.hidden = hidden;
		this.outputs = outputs;
		this.eta = eta;
		this.momentum = momentum;

		hiddenWeights = new double[hidden][inputs];
		outputWeights = new double[outputs][hidden];
		deltaK = new double[outputs];
		deltaH = new double[hidden];

		/* Initializing weight vectors with small random weights */
		for(int i = 0;i<hidden;i++){
			for(int j=0;j<inputs;j++){
				hiddenWeights[i][j] = Math.random()*0.1 - 0.05;
			}
		}

		for(int i = 0;i<outputs;i++){
			for(int j = 0;j<hidden;j++){
				outputWeights[i][j] = Math.random()*0.1 - 0.05;
			}
		}
	}



	/**
	 *@param input : The input vector
	 * @param target : The target output vector
	 * @param exampleNo : The example identity which we are dealing with
	 */
	
	void updateWeights(double input[], double target[],int exampleNo){
		double op[] = networkHiddenOutput(input, 0);
		double hop[] = networkHiddenOutput(input, 1);
		double deltaW = 0.0;
		for(int k=0;k<outputs;k++){
			deltaK[k] = op[k]*(1-op[k])*(target[k]-op[k]);
		}
		for(int h = 0;h<hidden;h++){
			for(int k =0;k<outputs;k++){
				deltaH[h] += outputWeights[k][h]*deltaK[k];
			}
			deltaH[h] = deltaH[h]*hop[h]*(1-hop[h]);
		}
		if(deltaHiddenW.get(exampleNo)==null){
			deltaHiddenW.put(exampleNo, new double[hidden][inputs]);
		}
		if(deltaOutputW.get(exampleNo)==null){
			deltaOutputW.put(exampleNo, new double[outputs][hidden]);
		}
		
		for(int i=0;i<outputs;i++){
			for(int j=0;j<hidden;j++){
					deltaW = eta*hop[j]*deltaK[i] + momentum * deltaOutputW.get(exampleNo)[i][j];
					outputWeights[i][j] += deltaW;
					deltaOutputW.get(exampleNo)[i][j] = deltaW;
			}
		}

		for(int i=0;i<hidden;i++){
			for(int j=0;j<inputs;j++){
					deltaW =  eta*input[j]*deltaH[i] + momentum*deltaHiddenW.get(exampleNo)[i][j];
					hiddenWeights[i][j] += deltaW;
					deltaHiddenW.get(exampleNo)[i][j] = deltaW;
			}
		}
	}

	/**
	 * @param input : The input vector
	 * @param flag : Indicates what to return. flag = 0 returns the output of the network, flag = 1 returns the
	 * 					output of the hidden layer
	 * @return : Network output vector or output of the hidden layer 
	 */

	double[] networkHiddenOutput(double input[], int flag){
		double[] output = new double[outputs];
		double[] hiddenOp = new double[hidden];

		for(int i = 0; i<hidden;i++){
			try{
				hiddenOp[i] = sigmoidOutput(dot(hiddenWeights[i], input));
			}catch(Exception e){
				System.err.println("Error in hiddenop["+i+"]");
			}
		}

		for(int i = 0; i<outputs; i++){
			try{
				output[i] = sigmoidOutput(dot(outputWeights[i], hiddenOp));
			}catch(Exception e){
				System.err.println("Error in output["+i+"]");
			}
		}
		if(flag == 0)
			return output;
		else
			return hiddenOp;
	}

	/**
	 * @param z : Input on which sigmoid function is to be applied
	 * @return : Output after applying sigmoid function on the input
	 */

	double sigmoidOutput(double z){
		double sig;
		sig = 1/(1+Math.exp(-1*z));
		return sig;
	}

	/**
	 * @param weight : Weight vector
	 * @param input : input vector
	 * @return : dot product of the vector
	 * @throws IncompatibleLengthException
	 */

	double dot(double weight[], double input[])throws IncompatibleLengthException{
		double result=0.0;

		if(weight.length != input.length){
			throw new IncompatibleLengthException("Lengths are not same");
		}
		else{
			for(int i=0;i<weight.length;i++){
				result += weight[i]*input[i];
			}
		}
		return result;
	}

	/**
	 * @param input : The image of which we are trying to predict an attribute
	 * @param mode 
	 * @return : our prediction
	 */
	
	int getPrediction(double input[],int mode){
		double[] output_temp = new double[outputs];
			
			try{
				output_temp = networkHiddenOutput(input,0);
			}catch(Exception e){
				System.err.println("Error in calculating output_temp");
			}
		

		if(mode == 1)
		{	
			if(output_temp[0]>0.5)
				return 1;
			else
				return 0;
		}else{
			return getMaxIndex(output_temp);
		}
	}

	/**
	 * @param array
	 * @return An utility function which returns the index of the maximal element in the array
	 */
	
	int getMaxIndex(double array[]){
		double max = -Double.MAX_VALUE;
		int index = 0;
		for(int i=0;i<array.length;i++)
		{
			if(array[i]>max)
			{
				max = array[i];
				index = i;
			}
		}
		return index;
	}
}



/**
 * @author Snehal, Sandeep
 * Provides a way to read PGM files. 
 */
class PGMReader {

	/**
	 * @param filename : Name of the PGM file to be parsed.
	 * @return a 2D double array where each double has the gray-scale value (normalised) of the pixel and size of array being 
	 * 			equal to the size of the image in the PGM file. 
	 * @throws IOException
	 */
	
	double[][] readImage(String filename) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String magic = br.readLine();
		String size = br.readLine();
		
		while(size.startsWith("#")){
			size = br.readLine();
		}
		
		Scanner s = new Scanner(size);
		int columns = s.nextInt();
		int rows = s.nextInt();
		s.close();
		
		String line = br.readLine();
		s = new Scanner(line);
		s.nextInt(); //Skipping the max value mentioned in the PGM Encoding
		double image[][] = new double[rows][columns];
		
		if("P5".equals(magic)){
			FileInputStream img = new FileInputStream(filename);
			File forLength = new File(filename);
			long length = forLength.length();
			long skip = length - rows*columns;
			img.skip(skip);
			for(int i=0;i<rows;i++){
				for(int j=0;j<columns;j++){
					image[i][j] = (double)img.read()/255.0;
				}
			}
			img.close();
		}
		else{
			System.out.println(filename);
		}
		
		s.close();
		br.close();

		return image;
	}
}


class Validation {
	/** Number of predictions we got right*/
	int correct = 0;
	
	/** Number of predictions we got wrong*/
	int wrong = 0;
	
	/** Our prediction */
	int prediction = 0;
	
	/**
	 * @param prediction : Our prediction
	 * @param filename : The image on which we are predicting
	 * @param mode : 1. Sunglasses 2. Face Recognizer 3. Pose Recognizer
	 */
	
	public void updateResults(int prediction,String filename,int mode){
		ANN.initializeHashMaps(); //Just to be safe
		switch (mode){
		case 1:
			if(prediction == 1)
			{
				if(filename.contains("sunglasses"))
					correct++;
				else
					wrong++;
			}
			else{
				if(filename.contains("open"))
					correct++;
				else
					wrong++;
			}
			break;
		case 2:
			if(filename.contains(ANN.faceID.get(prediction)))
				correct++;
			else
				wrong++;
			break;
		case 3:
			if(filename.contains(ANN.directionID.get(prediction)))
				correct++;
			else
				wrong++;
			break;
		}
	}
}