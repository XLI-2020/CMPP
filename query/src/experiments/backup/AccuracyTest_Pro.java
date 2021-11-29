package experiments;

import algorithm.Init;
import algorithm.PTQ_Range;
import indoor_entitity.Point;
import utilities.DataGenConstant;

import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;

public class AccuracyTest_Pro {
    static int range = 1500;
    static int sampleInterval = 60;
    static int populationThreshold = 2;
//    static double probabilityThreshold = 0.5;
    public static String result_true_probability = System.getProperty("user.dir") + "/result/result_true_probability.txt";
    public static String result_pre_probability_exact = System.getProperty("user.dir") + "/result/result_pre_probability_exact.txt";
    public static String result_pre_probability_appro = System.getProperty("user.dir") + "/result/result_pre_probability_appro.txt";

    public static String outFileAcc = System.getProperty("user.dir") + "/result/" + "newHSM_probability_acc.csv";
    public static String outFileRecall = System.getProperty("user.dir") + "/result/" + "newHSM_probability_recall.csv";

    public static void getGroundTruth() throws IOException {
        Init.init("true", sampleInterval);
        PointPrepare.trajectoryGen_read(PointPrepare.trajectorys, "newHSM", sampleInterval);
        System.out.println("start querying...");
        ArrayList<Double> probabilitys = new ArrayList<>(Arrays.asList(0.5, 0.6, 0.7, 0.8, 0.9));
        String s = "";
        for (int i = 0; i < probabilitys.size(); i++) {
            double probabilityThreshold = probabilitys.get(i);
            s += probabilityThreshold;
            for (int j = 0; j < 10; j++) {
                s += "\t" + j;
                System.out.println("traId: " + j);
                HashMap<Integer, Point> tra = PointPrepare.trajectory.get(j);

                for (int t = 0; t < DataGenConstant.endTime; t += sampleInterval) {
                    s += ";" + t;
                    Point p = tra.get(t);
                    ArrayList<Integer> result = PTQ_Range.range(p, range, t, populationThreshold, probabilityThreshold, false);
                    for (int parId : result) {
                        s += "," + parId;
                    }
                }

            }
            s += "\n";
        }

        try {
            FileWriter output = new FileWriter(result_true_probability);
            output.write(s);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    public static void calAccuracy() throws IOException {
        String resultAcc = "probability" + "\t" + "exact" + "\t" + "appro" + "\n";
        String resultRecall = "probability" + "\t" + "exact" + "\t" + "appro" + "\n";

        Path path = Paths.get(result_true_probability);
        Scanner scanner = new Scanner(path);

        HashMap<Double, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> probability_true_map = new HashMap<>();
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            String [] tempArr1 = line.split("\t");
            Double probability = Double.parseDouble(tempArr1[0]);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_map = new HashMap<>();
            for (int m = 1; m < tempArr1.length; m++) {
                String [] tempArr2 = tempArr1[m].split(";");
                int traId = Integer.parseInt(tempArr2[0]);
                HashMap<Integer, ArrayList<Integer>> time_map = new HashMap<>();
                for (int n = 1; n < tempArr2.length; n++) {
                    String [] tempArr3 = tempArr2[n].split(",");
                    int t = Integer.parseInt(tempArr3[0]);
                    ArrayList<Integer> true_list = new ArrayList<>();
                    for (int i = 1; i < tempArr3.length - 1; i++) {
                        int parId = Integer.parseInt(tempArr3[i]);
                        true_list.add(parId);
                    }
                    time_map.put(t, true_list);
                }
                tra_map.put(traId, time_map);
            }

            probability_true_map.put(probability, tra_map);
        }

        Path path1 = Paths.get(result_pre_probability_exact);
        Scanner scanner1 = new Scanner(path1);

        HashMap<Double, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> probability_exact_map = new HashMap<>();
        while (scanner1.hasNextLine()) {
            String line = scanner1.nextLine();
            String [] tempArr1 = line.split("\t");
            double probability = Double.parseDouble(tempArr1[0]);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_map = new HashMap<>();
            for (int m = 1; m < tempArr1.length; m++) {
                String [] tempArr2 = tempArr1[m].split(";");
                int traId = Integer.parseInt(tempArr2[0]);
                HashMap<Integer, ArrayList<Integer>> time_map = new HashMap<>();
                for (int n = 1; n < tempArr2.length; n++) {
                    String [] tempArr3 = tempArr2[n].split(",");
                    int t = Integer.parseInt(tempArr3[0]);
                    ArrayList<Integer> pre_list = new ArrayList<>();
                    for (int i = 1; i < tempArr3.length - 1; i++) {
                        int parId = Integer.parseInt(tempArr3[i]);
                        pre_list.add(parId);
                    }
                    time_map.put(t, pre_list);
                }
                tra_map.put(traId, time_map);
            }

            probability_exact_map.put(probability, tra_map);
        }

        Path path2 = Paths.get(result_pre_probability_appro);
        Scanner scanner2 = new Scanner(path2);

        HashMap<Double, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> probability_appro_map = new HashMap<>();
        while (scanner2.hasNextLine()) {
            String line = scanner2.nextLine();
            String [] tempArr1 = line.split("\t");
            double probability = Double.parseDouble(tempArr1[0]);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_map = new HashMap<>();
            for (int m = 1; m < tempArr1.length; m++) {
                String [] tempArr2 = tempArr1[m].split(";");
                int traId = Integer.parseInt(tempArr2[0]);
                HashMap<Integer, ArrayList<Integer>> time_map = new HashMap<>();
                for (int n = 1; n < tempArr2.length; n++) {
                    String [] tempArr3 = tempArr2[n].split(",");
                    int t = Integer.parseInt(tempArr3[0]);
                    ArrayList<Integer> pre_list = new ArrayList<>();
                    for (int i = 1; i < tempArr3.length - 1; i++) {
                        int parId = Integer.parseInt(tempArr3[i]);
                        pre_list.add(parId);
                    }
                    time_map.put(t, pre_list);
                }
                tra_map.put(traId, time_map);
            }

            probability_appro_map.put(probability, tra_map);
        }

        ArrayList<Double> probabilitys = new ArrayList<>(Arrays.asList(0.5, 0.6, 0.7, 0.8, 0.9));

        for (double probabilityThreshold: probabilitys) {
            resultAcc += probabilityThreshold;
            resultRecall += probabilityThreshold;

            System.out.println("probability: " + probabilityThreshold);
            ArrayList<Double> accList1 = new ArrayList<>();
            ArrayList<Double> recallList1 = new ArrayList<>();

            ArrayList<Double> accList2 = new ArrayList<>();
            ArrayList<Double> recallList2 = new ArrayList<>();

            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_exact_map = probability_exact_map.get(probabilityThreshold);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_true_map = probability_true_map.get(probabilityThreshold);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_appro_map = probability_appro_map.get(probabilityThreshold);
            System.out.println("tra_true_map size: " + tra_true_map.size());
            System.out.println("tra_exact_map size: " + tra_exact_map.size());
            System.out.println("tra_appro_map size: " + tra_appro_map.size());

            for (int i = 0; i < 10; i++) {
                System.out.println("traId: " + i);
                HashMap<Integer, ArrayList<Integer>> time_exact_map = tra_exact_map.get(i);
                HashMap<Integer, ArrayList<Integer>> time_true_map = tra_true_map.get(i);
                HashMap<Integer, ArrayList<Integer>> time_appro_map = tra_appro_map.get(i);
                System.out.println("time_true_map size: " + time_true_map.size());
                System.out.println("time_exact_map size: " + time_exact_map.size());
                System.out.println("time_appro_map size: " + time_appro_map.size());

                for (int t = 0; t < DataGenConstant.endTime; t += sampleInterval) {
                    System.out.println("t: " + t);
                    ArrayList<Integer> exact_list = time_exact_map.get(t);
                    ArrayList<Integer> true_list = time_true_map.get(t);
                    ArrayList<Integer> appro_list = time_appro_map.get(t);
                    System.out.println("true_list size: " + true_list.size());
                    System.out.println("exact_list size: " + exact_list.size());
                    System.out.println("appro_list size: " + appro_list.size());
                    int num1 = 0;
                    int num2 = 0;
                    for (int true_parId: true_list) {
                        for (int exact_parId: exact_list) {
                            if (exact_parId == true_parId) {
                                num1++;
                            }
                        }
                        for (int appro_parId: appro_list) {
                            if (appro_parId == true_parId) {
                                num2++;
                            }
                        }
                    }

                    double acc1 = (double)num1 / (double)exact_list.size();
                    double recall1 = (double)num1 / (double)true_list.size();

                    double acc2 = (double)num2 / (double)appro_list.size();
                    double recall2 = (double)num2 / (double)true_list.size();
                    System.out.println("acc1: " + acc1 + " recall1: " + recall1);
                    System.out.println("acc2: " + acc2 + " recall2: " + recall2);
                    accList1.add(acc1);
                    recallList1.add(recall1);
                    accList2.add(acc2);
                    recallList2.add(recall2);
                }

            }

            System.out.println("accList1: " + accList1);
            System.out.println("accList2: " + accList2);
            System.out.println("recallList1: " + recallList1);
            System.out.println("recallList2: " + recallList2);

            ArrayList<ArrayList<Double>> accLists = new ArrayList<>();
            accLists.add(accList1);
            accLists.add(accList2);

            for (int i = 0; i < accLists.size(); i++) {
                double sum = 0;
                double ave = 0;
                ArrayList<Double> accList = accLists.get(i);
                for (double acc: accList) {
                    sum += acc;
                    System.out.println("sum + acc: " + sum);
                }
                ave = sum / accList.size();
                resultAcc += "\t" + ave;
            }
            resultAcc += "\n";

            ArrayList<ArrayList<Double>> recallLists = new ArrayList<>();
            recallLists.add(recallList1);
            recallLists.add(recallList2);
            for (int i = 0; i < recallLists.size(); i++) {
                double sum = 0;
                double ave = 0;
                ArrayList<Double> recallList = recallLists.get(i);
                for (double recall: recallList) {
                    sum += recall;
                }
                ave = sum / recallList.size();
                resultRecall += "\t" + ave;
            }
            resultRecall += "\n";

        }

        FileOutputStream outputTime = new FileOutputStream(outFileAcc);
        outputTime.write(resultAcc.getBytes());
        outputTime.flush();
        outputTime.close();

        FileOutputStream outputMem = new FileOutputStream(outFileRecall);
        outputMem.write(resultRecall.getBytes());
        outputMem.flush();
        outputMem.close();


    }

    public static void main(String[] arg) throws IOException{
//        getGroundTruth();
        calAccuracy();
    }
}

