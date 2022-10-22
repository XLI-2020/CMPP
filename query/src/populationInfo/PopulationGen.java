package populationInfo;

import indoor_entitity.IndoorSpace;
import indoor_entitity.Partition;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;



public class PopulationGen {
    public static String population_le = System.getProperty("user.dir") + "/population/le_population_";
    public static String population_ge = System.getProperty("user.dir") + "/population/ge_population_";
    public static String population_stgcn = System.getProperty("user.dir") + "/population/stgcn_population_";
    public static String population_tgcn = System.getProperty("user.dir") + "/population/tgcn_population_";
    public static String population_astgnn = System.getProperty("user.dir") + "/population/astgnn_population_";
//    public static String timestamp = System.getProperty("user.dir") + "/population/timestamp.txt";
//    public static ArrayList<Double> timestamps = new ArrayList<>();


    public static void readPopulation(String populationType, int sampleInterval) throws IOException {
        Path path = Paths.get(population_le + "final_" + populationType + "_" + sampleInterval + ".txt");
        Scanner scanner = new Scanner(path);

        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            String [] tempArr = line.split("\t");
            int parId = Integer.parseInt(tempArr[0]);
            Partition par = IndoorSpace.iPartitions.get(parId);

//            System.out.print(parId + "\t");

            for (int i = 1; i < tempArr.length; i++) {
                String[] info = tempArr[i].split(",");
                int t = (int)Double.parseDouble(info[0]);
                double a = Double.parseDouble(info[1]);
                double u = Double.parseDouble(info[2]);
                par.addPopulation_le(t, a, u);
//                System.out.print(t + "," + a + "," + u + "\t");
            }
//            System.out.println();
        }

        Path path1 = Paths.get(population_ge + "final_" + populationType + "_" + sampleInterval + ".txt");
        Scanner scanner1 = new Scanner(path1);

        while (scanner1.hasNextLine()) {
            String line = scanner1.nextLine();
            String [] tempArr = line.split("\t");
            int parId = Integer.parseInt(tempArr[0]);
            Partition par = IndoorSpace.iPartitions.get(parId);

//            System.out.print(parId + "\t");

            for (int i = 1; i < tempArr.length; i++) {
                String[] info = tempArr[i].split(",");
                int t = (int)Double.parseDouble(info[0]);
                double a = Double.parseDouble(info[1]);
                double u = Double.parseDouble(info[2]);
                par.addPopulation_ge(t, a, u);
//                System.out.print(t + "," + a + "," + u + "\t");
            }
//            System.out.println();
        }

        Path path2 = Paths.get(population_stgcn + "final_" + populationType + "_" + sampleInterval + ".txt");
        Scanner scanner2 = new Scanner(path2);

        while (scanner2.hasNextLine()) {
            String line = scanner2.nextLine();
            String [] tempArr = line.split("\t");
            int parId = Integer.parseInt(tempArr[0]);
            Partition par = IndoorSpace.iPartitions.get(parId);

//            System.out.print(parId + "\t");

            for (int i = 1; i < tempArr.length; i++) {
                String[] info = tempArr[i].split(",");
                int t = (int)Double.parseDouble(info[0]);
                double a = Double.parseDouble(info[1]);
                double u = Double.parseDouble(info[2]);
                par.addPopulation_stgcn(t, a, u);
//                System.out.print(t + "," + a + "," + u + "\t");
            }
//            System.out.println();
        }

        Path path3 = Paths.get(population_tgcn + "final_" + populationType + "_" + sampleInterval + ".txt");
        Scanner scanner3 = new Scanner(path3);

        while (scanner3.hasNextLine()) {
            String line = scanner3.nextLine();
            String [] tempArr = line.split("\t");
            int parId = Integer.parseInt(tempArr[0]);
            Partition par = IndoorSpace.iPartitions.get(parId);

//            System.out.print(parId + "\t");

            for (int i = 1; i < tempArr.length; i++) {
                String[] info = tempArr[i].split(",");
                int t = (int)Double.parseDouble(info[0]);
                double a = Double.parseDouble(info[1]);
                double u = Double.parseDouble(info[2]);
                par.addPopulation_tgcn(t, a, u);
//                System.out.print(t + "," + a + "," + u + "\t");
            }
//            System.out.println();
        }

        Path path4 = Paths.get(population_astgnn + "final_" + populationType + "_" + sampleInterval + ".txt");
        Scanner scanner4 = new Scanner(path4);

        while (scanner4.hasNextLine()) {
            String line = scanner4.nextLine();
            String [] tempArr = line.split("\t");
            int parId = Integer.parseInt(tempArr[0]);
            Partition par = IndoorSpace.iPartitions.get(parId);

//            System.out.print(parId + "\t");

            for (int i = 1; i < tempArr.length; i++) {
                String[] info = tempArr[i].split(",");
                int t = (int)Double.parseDouble(info[0]);
                double a = Double.parseDouble(info[1]);
                double u = Double.parseDouble(info[2]);
                par.addPopulation_astgnn(t, a, u);
//                System.out.print(t + "," + a + "," + u + "\t");
            }
//            System.out.println();
        }
    }

//    public static void readTimestamp() throws IOException {
//        Path path = Paths.get(timestamp);
//        Scanner scanner = new Scanner(path);
//
//        while (scanner.hasNextLine()) {
//            String line = scanner.nextLine();
//            double t = Double.parseDouble(line);
//            timestamps.add(t);
//        }
//
//    }
}
