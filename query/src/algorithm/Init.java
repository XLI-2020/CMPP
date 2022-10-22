package algorithm;

import datagenerate.HSMDataGenRead;
import iDModel.GenTopology;
import indoor_entitity.IndoorSpace;
import indoor_entitity.Partition;
import populationInfo.PopulationGen;
import utilities.DataGenConstant;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;

public class Init {
    public static void init(String populationType, int sampleInterval) throws IOException {
        PrintOut printOut = new PrintOut();

        HSMDataGenRead dateGenReadMen = new HSMDataGenRead();
        dateGenReadMen.dataGen("newHsm");

        GenTopology genTopology = new GenTopology();
        genTopology.genTopology();

        PopulationGen.readPopulation(populationType, sampleInterval);
//        PopulationGen.readTimestamp();
        CalProbability.jsonConvert();
    }

    public static void main(String arg[]) throws IOException {
        Init.init("pre", 10);

        for (int i = 0; i < IndoorSpace.iPartitions.size(); i++) {

            Partition par = IndoorSpace.iPartitions.get(i);
            for (int t = 0; t < DataGenConstant.endTime; t+=10) {
                ArrayList<Double> info = par.getPopulation_le(t);
                System.out.print(t + ", " + info + "\t");
            }

//            System.out.print(par.getmID() + "\t");
//            for (Map.Entry<Double, ArrayList<Double>> entry : par.getPopulation().entrySet()) {
//                double t = entry.getKey();
//                double a = entry.getValue().get(0);
//                double u = entry.getValue().get(1);
//
//                System.out.print(t + ", " + a + "," + u + "\t");
//            }

            System.out.println();
        }
    }
}

