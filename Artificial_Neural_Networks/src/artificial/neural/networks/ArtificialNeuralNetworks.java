package artificial.neural.networks;
public class ArtificialNeuralNetworks {

    public static void main(String[] args) {
        
        /*MLP M=Faces.face_train_eval(20, 0.3, 0.3, 150, 0.9, 0.1, false);
        MLP.tofile(M, "MLPface.txt");
        M=Faces.pose_train_eval(10, 0.3, 0.3, 150, 0.9, 0.1, false);
        MLP.tofile(M, "MLPpose.txt");*/
        MLP M;
    	M=Faces.sunglasses_train_eval(4, 0.3, 0.3, 150, 0.9, 0.1, false);
        MLP.tofile(M, "MLPsunglasses.txt");
        
        
    }
    
}
