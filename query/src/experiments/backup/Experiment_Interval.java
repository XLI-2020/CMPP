package experiments;

import algorithm.*;
import datagenerate.HSMDataGenRead;
import iDModel.GenTopology;
import indoor_entitity.IndoorSpace;
import indoor_entitity.Partition;
import indoor_entitity.Point;
import utilities.DataGenConstant;

import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class Experiment_Interval {
    static int range = 999999999;
    static int sampleInterval = 60;
    static int populationThreshold = 2;
    static double probabilityThreshold = 0.5;
    public static String result_pre_interval_exact = System.getProperty("user.dir") + "/result/result_pre_interval_exact_backup.txt";
    public static String result_pre_interval_appro = System.getProperty("user.dir") + "/result/result_pre_interval_appro_backup.txt";

    public static void run(String outFileTime, String outFileMemory) throws IOException {
        String resultTime = "interval" + "\t" + "exact" + "\t" + "appro" + "\n";
        String resultMemo = "interval" + "\t" + "exact" + "\t" + "appro" + "\n";

        String s1 = "";
        String s2 = "";


        System.out.println("interval: " + sampleInterval);

        resultTime += sampleInterval + "\t";
        resultMemo += sampleInterval + "\t";


        ArrayList<Long> arrTime1 = new ArrayList<>();
        ArrayList<Long> arrTime2 = new ArrayList<>();


        ArrayList<Long> arrMem1 = new ArrayList<>();
        ArrayList<Long> arrMem2 = new ArrayList<>();

        s1 += sampleInterval;
        s2 += sampleInterval;

        for (int j = 0; j < 10; j++) {
            s1 += "\t" + j;
            s2 += "\t" + j;
            System.out.println("traId: " + j);
            HashMap<Integer, Point> tra = PointPrepare.trajectory.get(j);
            for (int t = 0; t < 3600; t += sampleInterval) {
                s1 += ";" + t;
                Point p = tra.get(t);
                System.out.println("point: " + p.getX() + "," + p.getY() + "," + p.getmFloor());

                // exact
                Runtime runtime1 = Runtime.getRuntime();
                runtime1.gc();
                ArrayList<Integer> result1 = new ArrayList<>();
                long startMem1 = runtime1.totalMemory() - runtime1.freeMemory();
                long startTime1 = System.currentTimeMillis();
                result1 = PTQ_Range.range(p, range, t, populationThreshold, probabilityThreshold, false);
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
                result2 = PTQ_Range.range(p, range, t, populationThreshold, probabilityThreshold, true);
                System.out.println("appro result: " + result2);
                long endTime2 = System.currentTimeMillis();
                long endMem2 = runtime2.totalMemory() - runtime2.freeMemory();
                long mem2 = (endMem2 - startMem2) / 1024; // kb
                long time2 = endTime2 - startTime2;
                arrTime2.add(time2);
                arrMem2.add(mem2);

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




        FileOutputStream outputTime = new FileOutputStream(outFileTime, true);
        outputTime.write(resultTime.getBytes());
        outputTime.flush();
        outputTime.close();

        FileOutputStream outputMem = new FileOutputStream(outFileMemory, true);
        outputMem.write(resultMemo.getBytes());
        outputMem.flush();
        outputMem.close();

        try {
            FileWriter output = new FileWriter(result_pre_interval_exact, true);
            output.write(s1);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        try {
            FileWriter output = new FileWriter(result_pre_interval_appro, true);
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
        sampleInterval = 10;
        Init.init("pre", sampleInterval);
        PointPrepare.trajectoryGen_read(PointPrepare.trajectorys, "newHSM", sampleInterval);

        // interval
        String outFileTime = System.getProperty("user.dir") + "/result/" + "newHSM_interval_time.csv";
        String outFileMemory = System.getProperty("user.dir") + "/result/" + "newHSM_interval_memory.csv";

        Experiment_Interval.run(outFileTime, outFileMemory);

    }

}
