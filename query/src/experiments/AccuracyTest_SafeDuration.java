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

public class AccuracyTest_SafeDuration {

    static int sampleInterval = 60;
    static int range = 1500;
    static int populationThreshold = 2;
    static double probabilityThreshold = 0.5;
    public static String result_true_safeDuration = System.getProperty("user.dir") + "/result/result_true_safeDuration.txt";
    public static String result_pre_safeDuration_ge = System.getProperty("user.dir") + "/result/result_pre_safeDuration_ge.txt";
    public static String result_pre_safeDuration_le = System.getProperty("user.dir") + "/result/result_pre_safeDuration_le.txt";
    public static String result_pre_safeDuration_stgcn = System.getProperty("user.dir") + "/result/result_pre_safeDuration_stgcn.txt";
    public static String result_pre_safeDuration_tgcn = System.getProperty("user.dir") + "/result/result_pre_safeDuration_tgcn.txt";
    public static String result_pre_safeDuration_astgnn = System.getProperty("user.dir") + "/result/result_pre_safeDuration_astgnn.txt";

    public static String outFileAcc = System.getProperty("user.dir") + "/result/" + "newHSM_safeDuration_precision.csv";
    public static String outFileRecall = System.getProperty("user.dir") + "/result/" + "newHSM_safeDuration_recall.csv";
    public static String outFileF = System.getProperty("user.dir") + "/result/" + "newHSM_safeDuration_F.csv";

    public static void getGroundTruth() throws IOException {
        Init.init("true", sampleInterval);
        PointPrepare.trajectoryGen_read(PointPrepare.trajectorys, "newHSM", sampleInterval);
        System.out.println("start querying...");
        ArrayList<Integer> safeDurations = new ArrayList<>(Arrays.asList(60, 120, 180, 240));
        String s = "";
        for (int i = 0; i < safeDurations.size(); i++) {
            int safe_duration = safeDurations.get(i);
            s += safe_duration;
            for (int j = 0; j < 10; j++) {
                s += "\t" + j;
                System.out.println("traId: " + j);
                HashMap<Integer, Point> tra = PointPrepare.trajectory.get(j);

                for (int t = 0; t < 3600; t += sampleInterval) {
                    Point p = tra.get(t);
                    if (t > 0 && t < 3600 - sampleInterval) {
                        Point p1 = tra.get(t - sampleInterval);
                        Point p2 = tra.get(t + sampleInterval);
                        if (p.isEqual(p1) && p.isEqual(p2)) continue;
                    }
                    s += ";" + t;
                    ArrayList<Integer> result = PTQ_Range.range(p, range, t, populationThreshold, probabilityThreshold, false);
                    for (int parId : result) {
                        s += "," + parId;
                    }
                }

            }
            s += "\n";
        }

        try {
            FileWriter output = new FileWriter(result_true_safeDuration);
            output.write(s);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    public static void calAccuracy() throws IOException {
        String resultAcc = "range" + "\t" + "ge" + "\t" + "le" + "\t" + "stgcn" + "\t" + "tgcn" + "\t" + "astgnn" + "\n";
        String resultRecall = "range" + "\t" + "ge" + "\t" + "le" + "\t" + "stgcn" + "\t" + "tgcn" + "\t" + "astgnn" + "\n";
        String resultF = "range" + "\t" + "ge" + "\t" + "le" + "\t" + "stgcn" + "\t" + "tgcn" + "\t" + "astgnn" + "\n";

        Path path = Paths.get(result_true_safeDuration);
        Scanner scanner = new Scanner(path);

        HashMap<Integer, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> safeDuration_true_map = new HashMap<>();
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            String [] tempArr1 = line.split("\t");
            int safeDuration = Integer.parseInt(tempArr1[0]);
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

            safeDuration_true_map.put(safeDuration, tra_map);
        }

        Path path1 = Paths.get(result_pre_safeDuration_ge);
        Scanner scanner1 = new Scanner(path1);

        HashMap<Integer, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> safeDuration_ge_map = new HashMap<>();
        while (scanner1.hasNextLine()) {
            String line = scanner1.nextLine();
            String [] tempArr1 = line.split("\t");
            int safeDuration = Integer.parseInt(tempArr1[0]);
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

            safeDuration_ge_map.put(safeDuration, tra_map);
        }

        Path path2 = Paths.get(result_pre_safeDuration_le);
        Scanner scanner2 = new Scanner(path2);

        HashMap<Integer, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> safeDuration_le_map = new HashMap<>();
        while (scanner2.hasNextLine()) {
            String line = scanner2.nextLine();
            String [] tempArr1 = line.split("\t");
            int safeDuration = Integer.parseInt(tempArr1[0]);
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

            safeDuration_le_map.put(safeDuration, tra_map);
        }

        Path path3 = Paths.get(result_pre_safeDuration_stgcn);
        Scanner scanner3 = new Scanner(path3);

        HashMap<Integer, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> safeDuration_stgcn_map = new HashMap<>();
        while (scanner3.hasNextLine()) {
            String line = scanner3.nextLine();
            String [] tempArr1 = line.split("\t");
            int safeDuration = Integer.parseInt(tempArr1[0]);
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

            safeDuration_stgcn_map.put(safeDuration, tra_map);
        }

        Path path4 = Paths.get(result_pre_safeDuration_tgcn);
        Scanner scanner4 = new Scanner(path4);

        HashMap<Integer, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> safeDuration_tgcn_map = new HashMap<>();
        while (scanner4.hasNextLine()) {
            String line = scanner4.nextLine();
            String [] tempArr1 = line.split("\t");
            int safeDuration = Integer.parseInt(tempArr1[0]);
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

            safeDuration_tgcn_map.put(safeDuration, tra_map);
        }

        Path path5 = Paths.get(result_pre_safeDuration_astgnn);
        Scanner scanner5 = new Scanner(path5);

        HashMap<Integer, HashMap<Integer, HashMap<Integer, ArrayList<Integer>>>> safeDuration_astgnn_map = new HashMap<>();
        while (scanner5.hasNextLine()) {
            String line = scanner5.nextLine();
            String [] tempArr1 = line.split("\t");
            int safeDuration = Integer.parseInt(tempArr1[0]);
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

            safeDuration_astgnn_map.put(safeDuration, tra_map);
        }

        ArrayList<Integer> safeDurations = new ArrayList<>(Arrays.asList(60, 120, 180, 240));

        for (int safeDuration: safeDurations) {
            resultAcc += safeDuration;
            resultRecall += safeDuration;
            resultF += safeDuration;

            System.out.println("safeDuration: " + safeDuration);
            ArrayList<Double> accList1 = new ArrayList<>();
            ArrayList<Double> recallList1 = new ArrayList<>();
            ArrayList<Double> FList1 = new ArrayList<>();

            ArrayList<Double> accList2 = new ArrayList<>();
            ArrayList<Double> recallList2 = new ArrayList<>();
            ArrayList<Double> FList2 = new ArrayList<>();

            ArrayList<Double> accList3 = new ArrayList<>();
            ArrayList<Double> recallList3 = new ArrayList<>();
            ArrayList<Double> FList3 = new ArrayList<>();

            ArrayList<Double> accList4 = new ArrayList<>();
            ArrayList<Double> recallList4 = new ArrayList<>();
            ArrayList<Double> FList4 = new ArrayList<>();

            ArrayList<Double> accList5 = new ArrayList<>();
            ArrayList<Double> recallList5 = new ArrayList<>();
            ArrayList<Double> FList5 = new ArrayList<>();

            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_ge_map = safeDuration_ge_map.get(safeDuration);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_true_map = safeDuration_true_map.get(safeDuration);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_le_map = safeDuration_le_map.get(safeDuration);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_stgcn_map = safeDuration_stgcn_map.get(safeDuration);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_tgcn_map = safeDuration_tgcn_map.get(safeDuration);
            HashMap<Integer, HashMap<Integer, ArrayList<Integer>>> tra_astgnn_map = safeDuration_astgnn_map.get(safeDuration);
            System.out.println("tra_true_map size: " + tra_true_map.size());
            System.out.println("tra_ge_map size: " + tra_ge_map.size());
            System.out.println("tra_le_map size: " + tra_le_map.size());
            System.out.println("tra_stgcn_map size: " + tra_stgcn_map.size());
            System.out.println("tra_tgcn_map size: " + tra_tgcn_map.size());
            System.out.println("tra_astgnn_map size: " + tra_astgnn_map.size());

            for (int i = 0; i < 10; i++) {
                System.out.println("traId: " + i);
                HashMap<Integer, ArrayList<Integer>> time_ge_map = tra_ge_map.get(i);
                HashMap<Integer, ArrayList<Integer>> time_true_map = tra_true_map.get(i);
                HashMap<Integer, ArrayList<Integer>> time_le_map = tra_le_map.get(i);
                HashMap<Integer, ArrayList<Integer>> time_stgcn_map = tra_stgcn_map.get(i);
                HashMap<Integer, ArrayList<Integer>> time_tgcn_map = tra_tgcn_map.get(i);
                HashMap<Integer, ArrayList<Integer>> time_astgnn_map = tra_astgnn_map.get(i);
                System.out.println("time_true_map size: " + time_true_map.size());
                System.out.println("time_ge_map size: " + time_ge_map.size());
                System.out.println("time_le_map size: " + time_le_map.size());
                System.out.println("time_stgcn_map size: " + time_stgcn_map.size());
                System.out.println("time_tgcn_map size: " + time_tgcn_map.size());
                System.out.println("time_astgnn_map size: " + time_astgnn_map.size());

                for (int t = 0; t < DataGenConstant.endTime; t += sampleInterval) {
                    System.out.println("t: " + t);
                    ArrayList<Integer> ge_list = time_ge_map.get(t);
                    ArrayList<Integer> true_list = time_true_map.get(t);
                    ArrayList<Integer> le_list = time_le_map.get(t);
                    ArrayList<Integer> stgcn_list = time_stgcn_map.get(t);
                    ArrayList<Integer> tgcn_list = time_tgcn_map.get(t);
                    ArrayList<Integer> astgnn_list = time_astgnn_map.get(t);
                    if (ge_list == null || true_list == null || le_list == null) continue;
                    System.out.println("true_list size: " + true_list.size());
                    System.out.println("ge_list size: " + ge_list.size());
                    System.out.println("le_list size: " + le_list.size());
                    System.out.println("stgcn_list size: " + stgcn_list.size());
                    System.out.println("tgcn_list size: " + tgcn_list.size());
                    System.out.println("astgnn_list size: " + astgnn_list.size());
                    int num1 = 0;
                    int num2 = 0;
                    int num3 = 0;
                    int num4 = 0;
                    int num5 = 0;
                    for (int true_parId: true_list) {
                        for (int ge_parId: ge_list) {
                            if (ge_parId == true_parId) {
                                num1++;
                            }
                        }
                        for (int le_parId: le_list) {
                            if (le_parId == true_parId) {
                                num2++;
                            }
                        }
                        for (int stgcn_parId: stgcn_list) {
                            if (stgcn_parId == true_parId) {
                                num3++;
                            }
                        }
                        for (int tgcn_parId: tgcn_list) {
                            if (tgcn_parId == true_parId) {
                                num4++;
                            }
                        }
                        for (int astgnn_parId: astgnn_list) {
                            if (astgnn_parId == true_parId) {
                                num5++;
                            }
                        }
                    }

                    double acc1;
                    double recall1;
                    double F1;

                    double acc2;
                    double recall2;
                    double F2;

                    double acc3;
                    double recall3;
                    double F3;

                    double acc4;
                    double recall4;
                    double F4;

                    double acc5;
                    double recall5;
                    double F5;

                    if (true_list.size() == 0) {
                        recall1 = 1;
                        recall2 = 1;
                        recall3 = 1;
                        recall4 = 1;
                        recall5 = 1;
                    }
                    else {
                        recall1 = (double)num1 / (double)true_list.size();
                        recall2 = (double)num2 / (double)true_list.size();
                        recall3 = (double)num3 / (double)true_list.size();
                        recall4 = (double)num4 / (double)true_list.size();
                        recall5 = (double)num5 / (double)true_list.size();

                    }

                    if (ge_list.size() == 0) {
                        acc1 = 1;
                    }
                    else {
                        acc1 = (double)num1 / (double)ge_list.size();
                    }


                    if (le_list.size() == 0) {
                        acc2 = 1;
                    }
                    else {
                        acc2 = (double)num2 / (double)le_list.size();
                    }

                    if (stgcn_list.size() == 0) {
                        acc3 = 1;
                    }
                    else {
                        acc3 = (double)num3 / (double)stgcn_list.size();
                    }

                    if (tgcn_list.size() == 0) {
                        acc4 = 1;
                    }
                    else {
                        acc4 = (double)num4 / (double)tgcn_list.size();
                    }

                    if (astgnn_list.size() == 0) {
                        acc5 = 1;
                    }
                    else {
                        acc5 = (double)num5 / (double)astgnn_list.size();
                    }

                    if (acc1 == 0 && recall1 == 0) {
                        F1 = 0;
                    }
                    else {
                        F1 = 2 * (acc1 * recall1) / (acc1 + recall1);
                    }

                    if (acc2 == 0 && recall2 == 0) {
                        F2 = 0;
                    }
                    else {
                        F2 = 2 * (acc2 * recall2) / (acc2 + recall2);
                    }

                    if (acc3 == 0 && recall3 == 0) {
                        F3 = 0;
                    }
                    else {
                        F3 = 2 * (acc3 * recall3) / (acc3 + recall3);
                    }

                    if (acc4 == 0 && recall4 == 0) {
                        F4 = 0;
                    }
                    else {
                        F4 = 2 * (acc4 * recall4) / (acc4 + recall4);
                    }

                    if (acc5 == 0 && recall5 == 0) {
                        F5 = 0;
                    }
                    else {
                        F5 = 2 * (acc5 * recall5) / (acc5 + recall5);
                    }



                    System.out.println("acc1: " + acc1 + " recall1: " + recall1 + " F: " + F1);
                    System.out.println("acc2: " + acc2 + " recall2: " + recall2 + " F: " + F2);
                    System.out.println("acc3: " + acc3 + " recall3: " + recall3 + " F: " + F3);
                    System.out.println("acc4: " + acc4 + " recall4: " + recall4 + " F: " + F4);
                    System.out.println("acc5: " + acc5 + " recall5: " + recall5 + " F: " + F5);

                    accList1.add(acc1);
                    recallList1.add(recall1);
                    FList1.add(F1);

                    accList2.add(acc2);
                    recallList2.add(recall2);
                    FList2.add(F2);

                    accList3.add(acc3);
                    recallList3.add(recall3);
                    FList3.add(F3);

                    accList4.add(acc4);
                    recallList4.add(recall4);
                    FList4.add(F4);

                    accList5.add(acc5);
                    recallList5.add(recall5);
                    FList5.add(F5);

                }

            }

            System.out.println("accList1: " + accList1);
            System.out.println("accList2: " + accList2);
            System.out.println("accList3: " + accList3);
            System.out.println("accList4: " + accList4);
            System.out.println("accList5: " + accList5);
            System.out.println("recallList1: " + recallList1);
            System.out.println("recallList2: " + recallList2);
            System.out.println("recallList3: " + recallList3);
            System.out.println("recallList4: " + recallList4);
            System.out.println("recallList5: " + recallList5);

            ArrayList<ArrayList<Double>> accLists = new ArrayList<>();
            accLists.add(accList1);
            accLists.add(accList2);
            accLists.add(accList3);
            accLists.add(accList4);
            accLists.add(accList5);

            for (int i = 0; i < accLists.size(); i++) {
                double sum = 0;
                double ave = 0;
                ArrayList<Double> accList = accLists.get(i);
                for (double acc: accList) {
                    sum += acc;
//                    System.out.println("sum + acc: " + sum);
                }
                ave = sum / accList.size();
                resultAcc += "\t" + ave;
            }
            resultAcc += "\n";

            ArrayList<ArrayList<Double>> recallLists = new ArrayList<>();
            recallLists.add(recallList1);
            recallLists.add(recallList2);
            recallLists.add(recallList3);
            recallLists.add(recallList4);
            recallLists.add(recallList5);
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

            ArrayList<ArrayList<Double>> FLists = new ArrayList<>();
            FLists.add(FList1);
            FLists.add(FList2);
            FLists.add(FList3);
            FLists.add(FList4);
            FLists.add(FList5);
            for (int i = 0; i < FLists.size(); i++) {
                System.out.println("method: " + i);
                double sum = 0;
                double ave = 0;
                ArrayList<Double> FList = FLists.get(i);
                for (double F: FList) {
                    sum += F;
                    System.out.println("F: " + F);
                }
                ave = sum / FList.size();
                System.out.println("ave: " + ave);
                resultF += "\t" + ave;
            }
            resultF += "\n";

        }

        FileOutputStream outputAcc = new FileOutputStream(outFileAcc);
        outputAcc.write(resultAcc.getBytes());
        outputAcc.flush();
        outputAcc.close();

        FileOutputStream outputRecall = new FileOutputStream(outFileRecall);
        outputRecall.write(resultRecall.getBytes());
        outputRecall.flush();
        outputRecall.close();

        FileOutputStream outputF = new FileOutputStream(outFileF);
        outputF.write(resultF.getBytes());
        outputF.flush();
        outputF.close();


    }

    public static void main(String[] arg) throws IOException{
        getGroundTruth();
        calAccuracy();
    }
}

