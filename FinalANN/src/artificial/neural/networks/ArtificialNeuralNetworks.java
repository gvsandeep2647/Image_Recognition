package artificial.neural.networks;

/**
 * The main class
 * @author Kushagra, Sandeep, Snehal, Tanmaya
 */
public class ArtificialNeuralNetworks {

    /**
     * The main function trains and evaluates on all 3 data sets.
     * @param args the command line arguments (not needed)
     */
    public static void main(String[] args) {
        System.out.println("Training for sunglasses recognition:");
        MLP M=Faces.sunglasses_train_eval(4,0.3, 0.3, 200, 0.9, 0.1, false);
        MLP.tofile(M, "MLPsunglasses.txt");
        MLP.toImage(M, 30,32, "sunglasses");
        System.out.println("Training for pose recognition:");
        MLP M2=Faces.pose_train_eval(6,0.3, 0.3, 200, 0.9, 0.1, false);
        MLP.tofile(M2, "MLPpose.txt");
        MLP.toImage(M2, 30, 32, "pose");
        System.out.println("Training for face recognition:");
        MLP M3=Faces.face_train_eval(20,0.3, 0.3, 200, 0.9, 0.1, false);
        MLP.tofile(M3, "MLPface.txt");
        MLP.toImage(M3, 30, 32, "face");
        
        
    }
    
}
