package algorithm;

import experiments.PointPrepare;
import indoor_entitity.Door;
import indoor_entitity.IndoorSpace;
import indoor_entitity.Partition;
import indoor_entitity.Point;
import utilities.Constant;
import utilities.DataGenConstant;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;


/**
 * PTQ_Range query using global model
 *
 */

public class PTQ_Range_ge {
    public static String result_pre = System.getProperty("user.dir") + "/result/result_pre.txt";
    public static boolean isCall;

    public static ArrayList<Integer> range(Point q, double r, int t, int populationThreshold, double probabilityThreshold, int safe_duration, String modelType) {
        isCall = false;
        ArrayList<Integer> R = new ArrayList<>();
        int [] isParVisited = new int [IndoorSpace.iPartitions.size()];
        int [] isDoorVisited = new int [IndoorSpace.iDoors.size()];
        double [] dist = new double[IndoorSpace.iDoors.size()];
        ArrayList<ArrayList<Integer>> prev = new ArrayList<>();

        int qPartitionId = getHostPartition(q);

        Partition qPar = IndoorSpace.iPartitions.get(qPartitionId);
        double pro;
        HashMap<Integer, ArrayList<Double>> population_q = new HashMap<>();
        if (modelType.equals("ge")) {
            population_q = qPar.getPopulation_ge();
        }
        else if (modelType.equals("stgcn")) {
            population_q = qPar.getPopulation_stgcn();
        }
        else if (modelType.equals("tgcn")) {
            population_q = qPar.getPopulation_tgcn();
        }
        else if (modelType.equals("astgnn")) {
            population_q = qPar.getPopulation_astgnn();
        }
        if (t % safe_duration == 0) {
            pro = CalProbability.calProbability(populationThreshold, population_q.get(t).get(0), population_q.get(t).get(1));
            isCall = true;
        }
        else {
            pro = CalProbability.calProbability(populationThreshold, population_q.get(t - (t % safe_duration)).get(0), population_q.get(t - (t % safe_duration)).get(1));
        }

        if (pro > probabilityThreshold) {
            R.add(qPartitionId);
        }

        isParVisited[qPartitionId] = 1;

        ArrayList<Integer> sDoors = new ArrayList<>();

        sDoors = qPar.getTopology().getP2DLeave();

        for(int i = 0; i < sDoors.size(); i++) {
            int sDoorId = sDoors.get(i);
            Door sDoor = IndoorSpace.iDoors.get(sDoorId);
            double dist1 = distPointDoor(q, sDoor);
            dist[sDoorId] = dist1;
        }

        // minHeap
        BinaryHeap<Double> H = new BinaryHeap<>(IndoorSpace.iDoors.size());
        for (int j = 0; j < IndoorSpace.iDoors.size(); j++) {
            if (!sDoors.contains(j)) {
                dist[j] = Constant.large;
            }

            H.insert(dist[j], j);
            prev.add(null);
        }

        while (H.heapSize > 0) {
            String minElement = H.delete_min();
            String [] minElementArr = minElement.split(",");
            int curDoorId = Integer.parseInt(minElementArr[1]);
            if (Double.parseDouble(minElementArr[0]) != dist[curDoorId]) {
                System.out.println("Something wrong with min heap: algorithm.IDModel_SPQ.pt2ptDistance3");
            }
            Door curDoor = IndoorSpace.iDoors.get(curDoorId);

            if (dist[curDoorId] >= r) break;
            if (sDoors.contains(curDoorId)) {
                prev.set(curDoorId, new ArrayList<>(Arrays.asList(qPartitionId, -1)));
            }

            int prevParId = prev.get(curDoorId).get(0);

            isDoorVisited[curDoorId] = 1;
            ArrayList<Integer> nextPars = curDoor.getD2PEnter();
            for (int i = 0; i < nextPars.size(); i++) {
                int nextParId = nextPars.get(i);
                if (nextParId == prevParId) continue;
                if (isParVisited[nextParId] == 1) continue;
                isParVisited[nextParId] = 1;
                Partition nextPar = IndoorSpace.iPartitions.get(nextParId);
//                System.out.println("nextParId: " + nextParId);
                HashMap<Integer, ArrayList<Double>> population = new HashMap<>();
                if (modelType.equals("ge")) {
                    population = nextPar.getPopulation_ge();
                }
                else if (modelType.equals("stgcn")) {
                    population = nextPar.getPopulation_stgcn();
                }
                else if (modelType.equals("tgcn")) {
                    population = nextPar.getPopulation_tgcn();
                }
                else if (modelType.equals("astgnn")) {
                    population = nextPar.getPopulation_astgnn();
                }
//                HashMap<Integer, ArrayList<Double>> population = nextPar.getPopulation_ge();
                double probability;
                if (t % safe_duration == 0) {
                    probability = CalProbability.calProbability(populationThreshold, population.get(t).get(0), population.get(t).get(1));
                    isCall = true;
//                    System.out.println("a: " + population.get(t).get(0));
//                    System.out.println("u: " + population.get(t).get(1));
                }
                else {
                    probability = CalProbability.calProbability(populationThreshold, population.get(t - (t % safe_duration)).get(0), population.get(t - (t % safe_duration)).get(1));
//                    System.out.println("a: " + population.get(t - (t % safe_duration)).get(0));
//                    System.out.println("u: " + population.get(t - (t % safe_duration)).get(1));
                }

//                System.out.println("probability: " + probability);

                if (probability > probabilityThreshold) {
                    R.add(nextParId);
                }


                ArrayList<Integer> leaveDoors = nextPar.getTopology().getP2DLeave();
                for (int k = 0; k < leaveDoors.size(); k++) {
                    int leaveDoorId = leaveDoors.get(k);
                    if (isDoorVisited[leaveDoorId] != 1) {

                        if (dist[curDoorId] + nextPar.getdistMatrix().getDistance(curDoorId, leaveDoorId) < dist[leaveDoorId]) {
                            double oldDist = dist[leaveDoorId];
                            dist[leaveDoorId] = dist[curDoorId] + nextPar.getdistMatrix().getDistance(curDoorId, leaveDoorId);
                            prev.set(leaveDoorId, new ArrayList<>(Arrays.asList(nextParId, leaveDoorId)));
                            H.updateNode(oldDist, leaveDoorId, dist[leaveDoorId], leaveDoorId);
                        }

                    }
                }
            }

        }
        System.out.println(R);
        return R;
    }


    /**
     * get host partition of a point
     */
    public static int getHostPartition(Point point) {
        int partitionId = -1;
        int floor = point.getmFloor();
        ArrayList<Integer> pars = IndoorSpace.iFloors.get(floor).getmPartitions();
        for (int i = 0; i < pars.size(); i++) {
            Partition par = IndoorSpace.iPartitions.get(pars.get(i));
            if (point.getX() >= par.getX1() && point.getX() <= par.getX2() && point.getY() >= par.getY1() && point.getY() <= par.getY2()) {
                partitionId = par.getmID();
                return partitionId;
            }
        }
        return partitionId;
    }


    /**
     * calculate distance between a point and a door
     */
    public static double distPointDoor(Point point, Door door) {
        double dist = 0;
        dist = Math.sqrt(Math.pow(point.getX() - door.getX(), 2) + Math.pow(point.getY() - door.getY(), 2));
        return dist;
    }

    public static void main(String[] arg) throws IOException {
        Init.init("pre", 10);
        PointPrepare.trajectoryGen_read(PointPrepare.trajectorys, "newHSM", 10);
        System.out.println("start querying...");
//        Runtime runtime = Runtime.getRuntime();
//        runtime.gc();
//        long startMem = runtime.totalMemory() - runtime.freeMemory();
//        long start = System.nanoTime();
        ArrayList<Integer> result = range(new Point(800, 800, 0), 1500, 0, 5, 0.5, 60, "ge");
//        long end = System.nanoTime();
//        long endMem = runtime.totalMemory() - runtime.freeMemory();
//
//        long time = (end - start) / 1000; // us
//        long memory = (endMem - startMem) / 1024; // b
//        System.out.println("time: " + time);
//        System.out.println("memory: " + memory);
        System.out.println(result);

//        HashMap<Integer, Point> tra = PointPrepare.trajectory.get(0);
//        for (int t = 0; t < 3600; t += 10){
//            Point p = tra.get(t);
//            ArrayList<Integer> result = range(p, 1500, t, 2, 0.5, 60);
//            String s = "" + 0 + "\t" + t + "\t";
//            for (int parId: result) {
//                s += parId + ",";
//            }
//            s += "\n";
//
//            try {
//                FileWriter output = new FileWriter(result_pre, true);
//                output.write(s);
//                output.flush();
//                output.close();
//            } catch (IOException e) {
//                // TODO Auto-generated catch block
//                e.printStackTrace();
//            }
//
//        }

//        String s = "";
//        for (int parId: result) {
//            s += parId + ",";
//        }
//        s += "\n";
//
//        try {
//            FileWriter output = new FileWriter(result_pre);
//            output.write(s);
//            output.flush();
//            output.close();
//        } catch (IOException e) {
//            // TODO Auto-generated catch block
//            e.printStackTrace();
//        }


    }

}

