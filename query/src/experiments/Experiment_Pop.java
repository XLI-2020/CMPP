package experiments;

import algorithm.*;
import indoor_entitity.Point;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;


public class Experiment_Pop {
    static int safe_duration = 120;
    static int range = 1500;
    static int sampleInterval = 60;
//    static int populationThreshold = 2;
    static double probabilityThreshold = 0.5;
    public static String result_pre_population_ge = System.getProperty("user.dir") + "/result/result_pre_population_ge.txt";
    public static String result_pre_population_le = System.getProperty("user.dir") + "/result/result_pre_population_le.txt";
    public static String result_pre_population_stgcn = System.getProperty("user.dir") + "/result/result_pre_population_stgcn.txt";
    public static String result_pre_population_tgcn = System.getProperty("user.dir") + "/result/result_pre_population_tgcn.txt";
    public static String result_pre_population_astgnn = System.getProperty("user.dir") + "/result/result_pre_population_astgnn.txt";

    public static void run(String outFileTime, String outFileMemory, String outFileNum) throws IOException {
        String resultTime = "population" + "\t" + "ge" + "\t" + "le" + "\t" + "stgcn" + "\t" + "sgcn" + "\t" + "astgnn" + "\n";
        String resultMemo = "population" + "\t" + "ge" + "\t" + "le" + "\t" + "stgcn" + "\t" + "sgcn" + "\t" + "astgnn" + "\n";
        String resultNum = "population" + "\t" + "le" + "\n";

        ArrayList<Integer> populations = new ArrayList<>(Arrays.asList(2, 4, 6, 8, 10));

        String s1 = "";
        String s2 = "";
        String s3 = "";
        String s4 = "";
        String s5 = "";
        for (int i = 0; i < populations.size(); i++) {
            int populationThreshold = populations.get(i);
            System.out.println("population: " + populationThreshold + "-------------------------");

            resultTime += populationThreshold + "\t";
            resultMemo += populationThreshold + "\t";


            ArrayList<Long> arrTime1 = new ArrayList<>();
            ArrayList<Long> arrTime2 = new ArrayList<>();
            ArrayList<Long> arrTime3 = new ArrayList<>();
            ArrayList<Long> arrTime4 = new ArrayList<>();
            ArrayList<Long> arrTime5 = new ArrayList<>();


            ArrayList<Long> arrMem1 = new ArrayList<>();
            ArrayList<Long> arrMem2 = new ArrayList<>();
            ArrayList<Long> arrMem3 = new ArrayList<>();
            ArrayList<Long> arrMem4 = new ArrayList<>();
            ArrayList<Long> arrMem5 = new ArrayList<>();

            ArrayList<Integer> arrNum2 = new ArrayList<>();

            s1 += populationThreshold;
            s2 += populationThreshold;
            s3 += populationThreshold;
            s4 += populationThreshold;
            s5 += populationThreshold;


            for (int j = 0; j < 10; j++) {
                s1 += "\t" + j;
                s2 += "\t" + j;
                s3 += "\t" + j;
                s4 += "\t" + j;
                s5 += "\t" + j;
                System.out.println("traId: " + j + "---------------------------------------------------");
                HashMap<Integer, Point> tra = PointPrepare.trajectory.get(j);
                PTQ_Range_le.pros.clear();
                for (int t = 0; t < 3600; t += sampleInterval) {
                    Point p = tra.get(t);
                    if (t > 0 && t < 3600 - sampleInterval) {
                        Point p1 = tra.get(t - sampleInterval);
                        Point p2 = tra.get(t + sampleInterval);
                        if (p.isEqual(p1) && p.isEqual(p2)) continue;
                    }
                    s1 += ";" + t;
                    System.out.println("point: " + p.getX() + "," + p.getY() + "," + p.getmFloor());

                    // ge
                    Runtime runtime1 = Runtime.getRuntime();
                    runtime1.gc();
                    ArrayList<Integer> result1 = new ArrayList<>();
                    long startMem1 = runtime1.freeMemory();
                    long startTime1 = System.currentTimeMillis();
                    result1 = PTQ_Range_ge.range(p, range, t, populationThreshold, probabilityThreshold, safe_duration, "ge");
                    System.out.println("ge result: " + result1);
                    long endTime1 = System.currentTimeMillis();
                    long endMem1 = runtime1.freeMemory();
                    long mem1 = (startMem1 - endMem1) / 1024; // kb
                    long time1 = endTime1 - startTime1;
                    System.out.println("memory: " + mem1 + ", endMem: " + endMem1 + ", startMem: " + startMem1);
                    arrTime1.add(time1);
                    arrMem1.add(mem1);


                    for (int parId: result1) {
                        s1 += "," + parId;
                    }

//                    System.out.println("exact memory " + mem1);
                }

                for (int t = 0; t < 3600; t += sampleInterval) {
                    Point p = tra.get(t);
                    if (t > 0 && t < 3600 - sampleInterval) {
                        Point p1 = tra.get(t - sampleInterval);
                        Point p2 = tra.get(t + sampleInterval);
                        if (p.isEqual(p1) && p.isEqual(p2)) continue;
                    }
                    s2 += ";" + t;
                    // le
                    Runtime runtime2 = Runtime.getRuntime();
                    runtime2.gc();
                    ArrayList<Integer> result2 = new ArrayList<>();
                    long startMem2 = runtime2.freeMemory();
                    long startTime2 = System.currentTimeMillis();
                    result2 = PTQ_Range_le.range(p, range, t, populationThreshold, probabilityThreshold, safe_duration);
                    System.out.println("le result: " + result2);
                    long endTime2 = System.currentTimeMillis();
                    long endMem2 = runtime2.freeMemory();
                    long mem2 = (startMem2 - endMem2) / 1024; // kb
                    long time2 = endTime2 - startTime2;
                    System.out.println("memory: " + mem2 + ", endMem: " + endMem2 + ", startMem: " + startMem2);
                    arrTime2.add(time2);
                    arrMem2.add(mem2);
                    arrNum2.add(PTQ_Range_le.number);

                    for (int parId: result2) {
                        s2 += "," + parId;
                    }

//                    System.out.println("inexactS memory " + mem2);
                }

                for (int t = 0; t < 3600; t += sampleInterval) {
                    Point p = tra.get(t);
                    if (t > 0 && t < 3600 - sampleInterval) {
                        Point p1 = tra.get(t - sampleInterval);
                        Point p2 = tra.get(t + sampleInterval);
                        if (p.isEqual(p1) && p.isEqual(p2)) continue;
                    }
                    s3 += ";" + t;
                    System.out.println("point: " + p.getX() + "," + p.getY() + "," + p.getmFloor());

                    // stgcn
                    Runtime runtime3 = Runtime.getRuntime();
                    runtime3.gc();
                    ArrayList<Integer> result3 = new ArrayList<>();
                    long startMem3 = runtime3.freeMemory();
                    long startTime3 = System.currentTimeMillis();
                    result3 = PTQ_Range_ge.range(p, range, t, populationThreshold, probabilityThreshold, safe_duration, "stgcn");
                    System.out.println("stgcn result: " + result3);
                    long endTime3 = System.currentTimeMillis();
                    long endMem3 = runtime3.freeMemory();
                    long mem3 = (startMem3 - endMem3) / 1024; // kb
                    long time3 = endTime3 - startTime3;
                    System.out.println("memory: " + mem3 + ", endMem: " + endMem3 + ", startMem: " + startMem3);
                    arrTime3.add(time3);
                    arrMem3.add(mem3);


                    for (int parId: result3) {
                        s3 += "," + parId;
                    }

//                    System.out.println("exact memory " + mem1);
                }

                for (int t = 0; t < 3600; t += sampleInterval) {
                    Point p = tra.get(t);
                    if (t > 0 && t < 3600 - sampleInterval) {
                        Point p1 = tra.get(t - sampleInterval);
                        Point p2 = tra.get(t + sampleInterval);
                        if (p.isEqual(p1) && p.isEqual(p2)) continue;
                    }
                    s4 += ";" + t;
                    System.out.println("point: " + p.getX() + "," + p.getY() + "," + p.getmFloor());

                    // tgcn
                    Runtime runtime4 = Runtime.getRuntime();
                    runtime4.gc();
                    ArrayList<Integer> result4 = new ArrayList<>();
                    long startMem4 = runtime4.freeMemory();
                    long startTime4 = System.currentTimeMillis();
                    result4 = PTQ_Range_ge.range(p, range, t, populationThreshold, probabilityThreshold, safe_duration, "tgcn");
                    System.out.println("tgcn result: " + result4);
                    long endTime4 = System.currentTimeMillis();
                    long endMem4 = runtime4.freeMemory();
                    long mem4 = (startMem4 - endMem4) / 1024; // kb
                    long time4 = endTime4 - startTime4;
                    System.out.println("memory: " + mem4 + ", endMem: " + endMem4 + ", startMem: " + startMem4);
                    arrTime4.add(time4);
                    arrMem4.add(mem4);


                    for (int parId: result4) {
                        s4 += "," + parId;
                    }

//                    System.out.println("exact memory " + mem1);
                }

                for (int t = 0; t < 3600; t += sampleInterval) {
                    Point p = tra.get(t);
                    if (t > 0 && t < 3600 - sampleInterval) {
                        Point p1 = tra.get(t - sampleInterval);
                        Point p2 = tra.get(t + sampleInterval);
                        if (p.isEqual(p1) && p.isEqual(p2)) continue;
                    }
                    s5 += ";" + t;
                    System.out.println("point: " + p.getX() + "," + p.getY() + "," + p.getmFloor());

                    // astgnn
                    Runtime runtime5 = Runtime.getRuntime();
                    runtime5.gc();
                    ArrayList<Integer> result5 = new ArrayList<>();
                    long startMem5 = runtime5.freeMemory();
                    long startTime5 = System.currentTimeMillis();
                    result5 = PTQ_Range_ge.range(p, range, t, populationThreshold, probabilityThreshold, safe_duration, "astgnn");
                    System.out.println("astgnn result: " + result5);
                    long endTime5 = System.currentTimeMillis();
                    long endMem5 = runtime5.freeMemory();
                    long mem5 = (startMem5 - endMem5) / 1024; // kb
                    long time5 = endTime5 - startTime5;
                    System.out.println("memory: " + mem5 + ", endMem: " + endMem5 + ", startMem: " + startMem5);
                    arrTime5.add(time5);
                    arrMem5.add(mem5);


                    for (int parId: result5) {
                        s5 += "," + parId;
                    }

//                    System.out.println("exact memory " + mem1);
                }
            }

            s1 += "\n";
            s2 += "\n";
            s3 += "\n";
            s4 += "\n";
            s5 += "\n";

            ArrayList<ArrayList<Long>> arrTimeAll = new ArrayList<>();
            arrTimeAll.add(arrTime1);
            arrTimeAll.add(arrTime2);
            arrTimeAll.add(arrTime3);
            arrTimeAll.add(arrTime4);
            arrTimeAll.add(arrTime5);

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
            arrMemAll.add(arrMem3);
            arrMemAll.add(arrMem4);
            arrMemAll.add(arrMem5);

            System.out.println("population: " + i);
            for (int j = 0; j < arrMemAll.size(); j++) {
                System.out.println("method: " + j);
                long sum = 0;
                long ave = 0;
                for (int h = 0; h < arrMemAll.get(j).size(); h++) {
                    System.out.println("memory: " + arrMemAll.get(j).get(h) + ", arrMemSize: " + arrMemAll.get(j).size());
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
            FileWriter output = new FileWriter(result_pre_population_ge);
            output.write(s1);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        try {
            FileWriter output = new FileWriter(result_pre_population_le);
            output.write(s2);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        try {
            FileWriter output = new FileWriter(result_pre_population_stgcn);
            output.write(s3);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        try {
            FileWriter output = new FileWriter(result_pre_population_tgcn);
            output.write(s4);
            output.flush();
            output.close();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        try {
            FileWriter output = new FileWriter(result_pre_population_astgnn);
            output.write(s5);
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



        // population
        String outFileTime = System.getProperty("user.dir") + "/result/" + "newHSM_population_time.csv";
        String outFileMemory = System.getProperty("user.dir") + "/result/" + "newHSM_population_memory.csv";
        String outFileNum = System.getProperty("user.dir") + "/result/" + "newHSM_population_num.csv";

        Experiment_Pop.run(outFileTime, outFileMemory, outFileNum);

    }

}
