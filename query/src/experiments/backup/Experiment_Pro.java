package experiments.backup;

import algorithm.*;
import experiments.PointPrepare;
import indoor_entitity.Point;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class Experiment_Pro {
    static int range = 1500;
    static int safe_duration;
    static int sampleInterval = 30;
    static int populationThreshold = 1;
//    static double probabilityThreshold = 0.5;
    public static String result_pre_probability_ge = System.getProperty("user.dir") + "/result/result_pre_probability_ge.txt";
    public static String result_pre_probability_le = System.getProperty("user.dir") + "/result/result_pre_probability_le.txt";

    public static void run(String outFileTime, String outFileMemory, String outFileNum) throws IOException {
        String resultTime = "probability" + "\t" + "ge" + "\t" + "le" + "\n";
        String resultMemo = "probability" + "\t" + "ge" + "\t" + "le" + "\n";
        String resultNum = "probability" + "\t" + "le" + "\n";

        ArrayList<Double> probabilitys = new ArrayList<>(Arrays.asList(0.5, 0.6, 0.7, 0.8, 0.9));

        String s1 = "";
        String s2 = "";
        for (int i = 0; i < probabilitys.size(); i++) {
            double probabilityThreshold = probabilitys.get(i);
            System.out.println("probability: " + probabilityThreshold);

            resultTime += probabilityThreshold + "\t";
            resultMemo += probabilityThreshold + "\t";
            resultNum += probabilityThreshold + "\t";


            ArrayList<Long> arrTime1 = new ArrayList<>();
            ArrayList<Long> arrTime2 = new ArrayList<>();


            ArrayList<Long> arrMem1 = new ArrayList<>();
            ArrayList<Long> arrMem2 = new ArrayList<>();

            ArrayList<Integer> arrNum2 = new ArrayList<>();

            s1 += probabilityThreshold;
            s2 += probabilityThreshold;





            for (int j = 0; j < 10; j++) {
                s1 += "\t" + j;
                s2 += "\t" + j;
                System.out.println("traId: " + j);
                HashMap<Integer, Point> tra = PointPrepare.trajectory.get(j);
                PTQ_Range_le.pros.clear();
                for (int t = 0; t < 3600; t += sampleInterval) {
                    s1 += ";" + t;
                    Point p = tra.get(t);
                    System.out.println("point: " + p.getX() + "," + p.getY() + "," + p.getmFloor());

                    // ge
                    Runtime runtime1 = Runtime.getRuntime();
                    runtime1.gc();
                    ArrayList<Integer> result1 = new ArrayList<>();
                    long startMem1 = runtime1.totalMemory() - runtime1.freeMemory();
                    long startTime1 = System.currentTimeMillis();
                    result1 = PTQ_Range_ge.range(p, range, t, populationThreshold, probabilityThreshold, safe_duration, "ge");
                    System.out.println("exact result: " + result1);
                    long endTime1 = System.currentTimeMillis();
                    long endMem1 = runtime1.totalMemory() - runtime1.freeMemory();
                    long mem1 = (endMem1 - startMem1) / 1024; // kb
                    long time1 = endTime1 - startTime1;
                    arrTime1.add(time1);
                    arrMem1.add(mem1);


                    for (int parId: result1) {
                        s1 += "," + parId;
                    }

//                    System.out.println("exact memory " + mem1);
                }

                for (int t = 0; t < 3600; t += sampleInterval) {
                    s2 += ";" + t;
                    Point p = tra.get(t);
                    // inexactS
                    Runtime runtime2 = Runtime.getRuntime();
                    runtime2.gc();
                    ArrayList<Integer> result2 = new ArrayList<>();
                    long startMem2 = runtime2.totalMemory() - runtime2.freeMemory();
                    long startTime2 = System.currentTimeMillis();
                    result2 = PTQ_Range_le.range(p, range, t, populationThreshold, probabilityThreshold, safe_duration);
                    System.out.println("appro result: " + result2);
                    long endTime2 = System.currentTimeMillis();
                    long endMem2 = runtime2.totalMemory() - runtime2.freeMemory();
                    long mem2 = (endMem2 - startMem2) / 1024; // kb
                    long time2 = endTime2 - startTime2;
                    arrTime2.add(time2);
                    arrMem2.add(mem2);
                    arrNum2.add(PTQ_Range_le.number);

                    for (int parId: result2) {
                        s2 += "," + parId;
                    }

//                    System.out.println("inexactS memory " + mem2);
                }
            }

            s1 += "\n";
            s2 += "\n";

            ArrayList<ArrayList<Long>> arrTimeAll = new ArrayList<>();
            arrTimeAll.add(arrTime1);
            arrTimeAll.add(arrTime2);

            for (int j = 0; j < arrTimeAll.size(); j++) {
                long sum = 0;
                long ave = 0;
                for (int h = 0; h < arrTimeAll.get(j).size(); h++) {
                    sum += arrTimeAll.get(j).get(h);
                }
                ave = sum / arrTimeAll.get(j).size();
                resultTime += ave + "\t";
            }
            resultTime += "\n";

            ArrayList<ArrayList<Long>> arrMemAll = new ArrayList<>();
            arrMemAll.add(arrMem1);
            arrMemAll.add(arrMem2);

            for (int j = 0; j < arrMemAll.size(); j++) {
                long sum = 0;
                long ave = 0;
                for (int h = 0; h < arrMemAll.get(j).size(); h++) {
                    sum += arrMemAll.get(j).get(h);
                }
                ave = sum / arrMemAll.get(j).size();
                resultMemo += ave + "\t";
            }
            resultMemo += "\n";

            double sum = 0;
            double ave = 0;
            for (int h = 0; h < arrNum2.size(); h++) {
                sum += arrNum2.get(h);
            }
            ave = sum / arrNum2.size();
            resultNum += ave + "\n";


        }

        FileOutputStream outputTime = new FileOutputStream(outFileTime);
        outputTime.write(resultTime.getBytes());
        outputTime.flush();
        outputTime.close();

        FileOutputStream outputMem = new FileOutputStream(outFileMemory);
        outputMem.write(resultMemo.getBytes());
        outputMem.flush();
        outputMem.close();

        FileOutputStream outputNum = new FileOutputStream(outFileNum);
        outputNum.write(resultNum.getBytes());
        outputNum.flush();
        outputNum.close();

        try {
            FileWriter output = new FileWriter(result_pre_probability_ge);
            output.write(s1);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        try {
            FileWriter output = new FileWriter(result_pre_probability_le);
            output.write(s2);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }


    public static void main(String arg[]) throws IOException {
        PrintOut printOut = new PrintOut();
        Init.init("pre", sampleInterval);
        PointPrepare.trajectoryGen_read(PointPrepare.trajectorys, "newHSM", sampleInterval);
//


        // population
        String outFileTime = System.getProperty("user.dir") + "/result/" + "newHSM_probability_time.csv";
        String outFileMemory = System.getProperty("user.dir") + "/result/" + "newHSM_probability_memory.csv";
        String outFileNum = System.getProperty("user.dir") + "/result/" + "newHSM_probability_num.csv";

        Experiment_Pro.run(outFileTime, outFileMemory, outFileNum);

    }

}

